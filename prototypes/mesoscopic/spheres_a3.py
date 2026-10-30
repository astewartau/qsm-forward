"""DECOMPOSE-targeted sphere phantom (A3): water-containing sub-voxel spheres.

Follows the numerical simulation of Chen et al. 2021 (NeuroImage 242:118477, Sec. 3.2):
each voxel is a periodic box of N^3 sub-voxels holding random spheres (radius >= 6
sub-voxels) that are either paramagnetic (chi_s+ > 0) or diamagnetic (chi_s- < 0), with
source susceptibilities of order 0.01-0.15 ppm, static dephasing, B0 = 3 T along z, and
every sub-voxel (inside and outside the spheres) holding water at equal proton density.
The voxel signal is the mean of exp(i * gamma * B0 * f(r) * t) over all sub-voxels; no
relaxivity, pool frequency or multi-pool signal equation enters. A common sub-voxel R2
(the paper's 20 1/s) factors out and is applied at read.

Two intra-source field conventions (both with D(0) = 0, i.e. the voxel-mean frequency is
zero and the macroscopic field stays that of qsm-forward's standard forward model):

* ``decompose``: the paper's (App. A3): water inside a source sees the macroscopic
  interior field, (2/3) chi_s B0 for an isolated sphere -> kernel 1 - kz^2/k^2 (no
  Lorentz-sphere correction).
* ``lorentz``: the standard Lorentz-corrected kernel 1/3 - kz^2/k^2 (qsm-forward's and
  meso.py's). Water mixed into a spherical source then has no frequency offset; spheres
  only dephase.

They differ pointwise by (2/3)(chi(r) - mean chi), so only intra-source water moves.

Per-voxel parameters come from bulk chi+/chi- maps through ``source_params`` (volume
fraction zeta and source susceptibility chi_s with zeta * chi_s = bulk chi exactly).

Outputs per voxel: the complex intra-voxel factor F(TE) (multiplies the voxel's
macroscopic-phase GRE signal) and the per-echo apparent susceptibility

    chi(TE) = chi_bulk + phi_res(TE) / ((2/3) gamma B0 TE),   phi_res = unwrap(angle F),

i.e. what DECOMPOSE's per-echo QSM input would hold for this voxel if the local signal
were known exactly (the "chimap-echoes" input).

Usage:
  python spheres_a3.py build --seed S --out DIR        # one realisation (LUT part)
  python spheres_a3.py merge --parts DIR --out FILE.npz
  python spheres_a3.py validate --out DIR
  python spheres_a3.py head --lut FILE.npz --out DIR   # Ridani anatomy -> chimap-echoes
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso  # noqa: E402

GAMMA = meso.GAMMA            # rad/ms/T
B0 = 3.0
N = 128                       # sub-voxels per side (Chen 2021)
R_MIN, R_MAX = 6.0, 16.0      # sphere radius, sub-voxels (uniform; Chen: > 6)
ZETA = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
CHI_S = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50, 0.75])
TE = np.arange(0.0, 50.0001, 0.5)                 # ms, LUT grid
TE_CHEN = 2.7 + 2.9 * np.arange(16)               # ms, Chen 2021 in vivo (2.7-46.2)
CONVENTIONS = ("decompose", "lorentz")
NBINS = 1024
# source_params mapping: zeta = ZETA_REF while chi_s rises 0.01 -> CHI_S_PAPER, then
# zeta rises to ZETA_MAX, then chi_s rises above the paper's range.
ZETA_REF, ZETA_MAX, CHI_S_MIN, CHI_S_PAPER = 0.25, 0.35, 0.01, 0.15


def wrap_limit_ppm(te_max_ms, b0=B0):
    """|chi_tot| above which the last echo's DECOMPOSE phase (2/3) gamma B0 chi TE
    leaves (-pi, pi]."""
    return np.pi / ((2 / 3) * GAMMA * b0 * 1e-6 * te_max_ms)


def source_params(chi_bulk):
    """Bulk susceptibility (either sign) -> (zeta, chi_s >= 0) with zeta*chi_s = |bulk|.

    |bulk| <= ZETA_REF*CHI_S_MIN: chi_s = 0.01, zeta < ZETA_REF;
    up to ZETA_REF*0.15 (0.0375): zeta = 0.25, chi_s 0.01..0.15 (Chen's range);
    up to ZETA_MAX*0.15 (0.0525): chi_s = 0.15, zeta 0.25..0.35;
    above: zeta = 0.35, chi_s = |bulk|/0.35 (outside the paper's chi_s range)."""
    b = np.abs(np.asarray(chi_bulk, float))
    chi_s = np.clip(b / ZETA_REF, CHI_S_MIN, CHI_S_PAPER)
    chi_s = np.where(b > ZETA_MAX * CHI_S_PAPER, b / ZETA_MAX, chi_s)
    zeta = np.where(chi_s > 0, b / chi_s, 0.0)
    return zeta, chi_s


# ----------------------------------------------------------------------------------
# geometry and fields
# ----------------------------------------------------------------------------------
def kernels(n=N):
    k = np.fft.fftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij", sparse=True)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = np.inf
    lor = (1 / 3 - kz ** 2 / k2).astype(np.float32)
    lor[0, 0, 0] = 0.0
    mac = (lor + np.float32(2 / 3))
    mac[0, 0, 0] = 0.0
    return dict(decompose=mac, lorentz=lor)


class SphereSet:
    """Union of random periodic spheres grown one by one; ``grow(target)`` adds spheres
    until the occupied fraction reaches target exactly (the last sphere's radius is
    chosen to land on it)."""

    def __init__(self, n, rng, exclude=None, rmin=R_MIN, rmax=R_MAX):
        self.n = n
        self.rmin, self.rmax = rmin, rmax
        self.rng = rng
        self.mask = np.zeros((n, n, n), bool)
        self.exclude = exclude
        self.count = 0

    def _ball(self, c, r):
        h = int(np.ceil(r)) + 1
        off = np.arange(-h, h + 1)
        idx = [(int(np.floor(ci)) + off) for ci in c]
        d2 = sum(((ix + 0.5 - ci) ** 2).reshape(sh) for ix, ci, sh in
                 zip(idx, c, ((-1, 1, 1), (1, -1, 1), (1, 1, -1))))
        sl = np.ix_(*(ix % self.n for ix in idx))
        return sl, d2

    def _new(self, sl, d2, r):
        b = d2 <= r * r
        b &= ~self.mask[sl]
        if self.exclude is not None:
            b &= ~self.exclude[sl]
        return b

    def grow(self, target):
        n3 = self.n ** 3
        while self.count < round(target * n3):
            need = round(target * n3) - self.count
            c = self.rng.uniform(0, self.n, 3)
            r = self.rng.uniform(self.rmin, self.rmax)
            sl, d2 = self._ball(c, r)
            b = self._new(sl, d2, r)
            nb = int(b.sum())
            if nb > need:   # shrink this last sphere to land on the target
                lo, hi = 0.5, r
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if int(self._new(sl, d2, mid).sum()) > need:
                        hi = mid
                    else:
                        lo = mid
                b = self._new(sl, d2, lo)
                nb = int(b.sum())
                if nb == 0:
                    continue
            m = self.mask[sl]
            m |= b
            self.mask[sl] = m
            self.count += nb
        return self.mask


def unit_field(mask, kern):
    """Field (ppm per ppm of source chi) of a unit-susceptibility mask."""
    return np.real(np.fft.ifftn(kern * np.fft.fftn(mask.astype(np.float32)))).astype(np.float32)


def _hist(fp, fm):
    """2D histogram of (unit para field, unit dia field) over all sub-voxels."""
    def edges(v):
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 1e-6, hi + 1e-6
        return np.linspace(lo, hi, NBINS + 1)
    ep, em = edges(fp), edges(fm)
    H, _, _ = np.histogram2d(fp.ravel(), fm.ravel(), bins=(ep, em))
    return H / H.sum(), 0.5 * (ep[1:] + ep[:-1]), 0.5 * (em[1:] + em[:-1])


def signals_from_hist(H, cp, cm, chi_s=CHI_S, te=TE, b0=B0):
    """S[t, a, b] = sum_ij H_ij exp(i gamma B0 t (chi_s[a] cp_i - chi_s[b] cm_j))."""
    w = GAMMA * b0 * 1e-6 * te                       # rad per ppm
    out = np.empty((len(te), len(chi_s), len(chi_s)), np.complex128)
    for k, wk in enumerate(w):
        U = np.exp(1j * wk * np.outer(chi_s, cp))
        V = np.exp(-1j * wk * np.outer(chi_s, cm))
        out[k] = U @ H @ V.T
    return out


def build(seed, outdir, n=N):
    """One realisation: S[conv][ip, im, a, b, t] over ZETA+ x ZETA- x CHI_S+ x CHI_S-."""
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    K = kernels(n)
    nz, nc = len(ZETA), len(CHI_S)
    S = {c: np.zeros((nz, nz, nc, nc, len(TE)), np.complex64) for c in CONVENTIONS}
    zeff = np.zeros((nz, nz, 2))
    intra = {c: np.zeros((nz, nz, 2)) for c in CONVENTIONS}   # mean unit field in +/- sources
    para = SphereSet(n, np.random.default_rng([seed, 1]))
    for ip, zp in enumerate(ZETA):
        mp = para.grow(zp).copy()
        Fp = {c: unit_field(mp, K[c]) for c in CONVENTIONS}
        dia = SphereSet(n, np.random.default_rng([seed, 2]), exclude=mp)
        for im, zm in enumerate(ZETA):
            mm = dia.grow(zm)
            Fm = {c: unit_field(mm, K[c]) for c in CONVENTIONS}
            zeff[ip, im] = mp.mean(), mm.mean()
            for c in CONVENTIONS:
                H, cp, cm = _hist(Fp[c], Fm[c])
                S[c][ip, im] = np.moveaxis(signals_from_hist(H, cp, cm), 0, -1)
                if zp > 0:
                    intra[c][ip, im, 0] = Fp[c][mp].mean()
                if zm > 0:
                    intra[c][ip, im, 1] = Fm[c][mm].mean()
        print(f"zeta+ {zp:.2f} done {time.time() - t0:.0f}s", flush=True)
    fn = os.path.join(outdir, f"a3_s{seed}.npz")
    np.savez_compressed(fn, zeta=ZETA, chi_s=CHI_S, te=TE, zeta_eff=zeff, n=n, seed=seed,
                        r_range=np.array([R_MIN, R_MAX]), b0=B0,
                        intra_decompose=intra["decompose"], intra_lorentz=intra["lorentz"],
                        **{f"S_{c}": S[c] for c in CONVENTIONS})
    print(f"wrote {fn} {time.time() - t0:.0f}s", flush=True)


def merge(parts, out):
    fs = sorted(glob.glob(os.path.join(parts, "a3_s*.npz")))
    d0 = np.load(fs[0])
    res = {k: d0[k] for k in ("zeta", "chi_s", "te", "n", "r_range", "b0")}
    for c in CONVENTIONS:
        A = np.stack([np.load(f)[f"S_{c}"] for f in fs])
        res[f"S_{c}"] = A.mean(0).astype(np.complex64)
        res[f"S_{c}_sd"] = np.abs(A - A.mean(0)).std(0).astype(np.float32)
        res[f"intra_{c}"] = np.mean([np.load(f)[f"intra_{c}"] for f in fs], 0)
    res["zeta_eff"] = np.mean([np.load(f)["zeta_eff"] for f in fs], 0)
    res["n_realisations"] = len(fs)
    np.savez_compressed(out, **res)
    print(f"wrote {out} from {len(fs)} realisations")


# ----------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------
def _logsig(S):
    S = np.where(np.isfinite(S) & (np.abs(S) > 0), S, 1.0)
    return np.log(np.abs(S)) + 1j * np.unwrap(np.angle(S), axis=-1)


class SphereLUT:
    def __init__(self, path):
        d = np.load(path)
        self.zeta, self.chi_s, self.te = d["zeta"], d["chi_s"], d["te"]
        self.b0 = float(d["b0"])
        self._f = {}
        for c in CONVENTIONS:
            L = _logsig(d[f"S_{c}"])
            pts = (self.zeta, self.zeta, self.chi_s, self.chi_s, self.te)
            self._f[c] = (RegularGridInterpolator(pts, L.real),
                          RegularGridInterpolator(pts, L.imag))

    def factor(self, chi_pos, chi_neg, te_ms, conv):
        """Complex intra-voxel factor (n_vox, n_te) for bulk chi+ >= 0, chi- <= 0."""
        zp, sp = source_params(np.maximum(chi_pos, 0))
        zm, sm = source_params(np.minimum(chi_neg, 0))
        sp = np.clip(sp, self.chi_s[0], self.chi_s[-1])
        sm = np.clip(sm, self.chi_s[0], self.chi_s[-1])
        te_ms = np.atleast_1d(te_ms)
        nv = np.size(zp)
        q = np.empty((nv, len(te_ms), 5))
        for k, v in enumerate((zp, zm, sp, sm)):
            q[..., k] = np.ravel(v)[:, None]
        q[..., 4] = te_ms[None]
        fr, fi = self._f[conv]
        return np.exp(fr(q) + 1j * fi(q))

    def chi_echoes(self, chi_pos, chi_neg, te_ms, conv):
        """Per-echo apparent chi(TE) (n_vox, n_te), phase unwrapped along TE from 0."""
        te_ms = np.atleast_1d(te_ms)
        grid = np.union1d(self.te[self.te < te_ms.max()], te_ms)
        F = self.factor(chi_pos, chi_neg, grid, conv)
        ph = np.unwrap(np.angle(F), axis=-1)[:, np.searchsorted(grid, te_ms)]
        bulk = (np.maximum(chi_pos, 0) + np.minimum(chi_neg, 0)).ravel()[:, None]
        return bulk + ph / ((2 / 3) * GAMMA * self.b0 * 1e-6 * te_ms[None])


def nonlinear_phase(ph, te):
    """Peak-to-peak residual (rad) of a straight-line fit (with intercept) of the
    unwrapped phase along the last axis (as in the decompose research probe)."""
    A = np.vstack([te, np.ones_like(te)]).T
    coef = np.linalg.lstsq(A, ph.reshape(-1, len(te)).T, rcond=None)[0]
    r = ph.reshape(-1, len(te)) - (A @ coef).T
    return np.ptp(r, axis=-1).reshape(ph.shape[:-1])


# ----------------------------------------------------------------------------------
# validation (generator vs analytic limits; never vs DECOMPOSE)
# ----------------------------------------------------------------------------------
def _yh_check(n, zeta, rmin, rmax, seeds, K):
    """Lorentz convention, one sign of spheres: |S| vs exp(-zeta Re f_YH(dw t)) and the
    long-time R2' vs Y&H a*zeta (per ppm of chi_s). Also the intra-source mean unit field
    per convention vs (2/3)(1 - zeta) (decompose) and 0 (lorentz)."""
    dw = GAMMA * B0 * 1e-6 / 3      # rad/ms per ppm chi_s
    x = np.concatenate([np.linspace(0.05, 1.5, 12), np.linspace(2, 30, 12)])
    t = x / dw
    per, intra = [], {c: [] for c in CONVENTIONS}
    for sd in seeds:
        m = SphereSet(n, np.random.default_rng([sd, 7]), rmin=rmin, rmax=rmax).grow(zeta)
        f = unit_field(m, K["lorentz"]).ravel().astype(np.float64)
        h, e = np.histogram(f, bins=20000)
        c = 0.5 * (e[1:] + e[:-1])
        S = np.exp(1j * np.outer(GAMMA * B0 * 1e-6 * t, c)) @ (h / h.sum())
        z = float(m.mean())
        mlong = x >= 10
        sl = -np.polyfit(t[mlong], np.log(np.abs(S[mlong])), 1)[0]
        Sth = np.exp(-z * meso.yh_f_sphere(x))
        per.append(dict(rel_err_R2p=float(sl / meso.yh_r2p_sphere(z, 1.0, B0) - 1),
                        max_dev_mag_over_loss=float(np.max(np.abs(np.abs(S) - np.abs(Sth)))
                                                    / np.max(1 - np.abs(Sth)))))
        for cv in CONVENTIONS:
            fin = float(unit_field(m, K[cv])[m].mean())
            th = (2 / 3) * (1 - z) if cv == "decompose" else 0.0
            intra[cv].append(fin - th)
    r = np.array([p["rel_err_R2p"] for p in per])
    dv = np.array([p["max_dev_mag_over_loss"] for p in per])
    return dict(n=n, zeta=zeta, radius_px=[rmin, rmax], seeds=len(seeds),
                R2p_YH_per_s_per_ppm=float(meso.yh_r2p_sphere(zeta, 1.0, B0) * 1e3),
                rel_err_R2p_mean=float(r.mean()), rel_err_R2p_sd=float(r.std()),
                max_dev_mag_over_loss_mean=float(dv.mean()),
                intra_minus_analytic_decompose_mean=float(np.mean(intra["decompose"])),
                intra_minus_analytic_decompose_sd=float(np.std(intra["decompose"])),
                intra_minus_analytic_lorentz_mean=float(np.mean(intra["lorentz"])),
                intra_minus_analytic_lorentz_sd=float(np.std(intra["lorentz"])))


def validate(outdir):
    """(1) Y&H regime: many small spheres (r = 4 px, 256^3) -> the field code reproduces
    the static-dephasing theory. (2) Chen's geometry (r 6-16 px, 128^3): few large
    spheres per box; reports the realisation spread around Y&H (expected, not an error:
    the exterior fields do not average to the infinite-medium limit)."""
    os.makedirs(outdir, exist_ok=True)
    res = {"yh_regime": [], "chen_geometry": []}
    K = kernels(256)
    for z in (0.02, 0.05):
        res["yh_regime"].append(_yh_check(256, z, 4.0, 4.0, (0, 1), K))
        print(json.dumps(res["yh_regime"][-1]), flush=True)
    K = kernels(N)
    for z in (0.05, 0.20):
        res["chen_geometry"].append(_yh_check(N, z, R_MIN, R_MAX, range(6), K))
        print(json.dumps(res["chen_geometry"][-1]), flush=True)
    with open(os.path.join(outdir, "validation_a3.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def lut_stats(lut_path, outdir):
    """Non-linear phase over the paper's envelope (zeta 0.05-0.35, chi_s 0.01-0.15, both
    signs present) on the Chen in vivo protocol, per convention."""
    os.makedirs(outdir, exist_ok=True)
    d = np.load(lut_path)
    te = d["te"]
    k = np.searchsorted(te, np.round(TE_CHEN * 2) / 2)   # nearest 0.5 ms node
    zi = np.where(ZETA >= 0.05)[0]
    ci = np.where(CHI_S <= 0.15)[0]
    out = {}
    for c in CONVENTIONS:
        S = d[f"S_{c}"]
        ph = np.unwrap(np.angle(S), axis=-1)[..., k]
        nl = nonlinear_phase(ph, te[k])
        sub = nl[np.ix_(zi, zi, ci, ci)]
        valid = np.isfinite(sub) & (np.abs(S[np.ix_(zi, zi, ci, ci)][..., 1]) > 0)
        mag = np.abs(S[..., k])
        r2s = meso.fit_rate(te[k], np.moveaxis(mag, -1, 0))[np.ix_(zi, zi, ci, ci)]
        out[c] = dict(intravoxel_R2star_note="R2* of |F| only (no intrinsic R2)",
                      nonlinear_rad_min=float(sub[valid].min()),
                      nonlinear_rad_median=float(np.median(sub[valid])),
                      nonlinear_rad_max=float(sub[valid].max()),
                      intravoxel_R2star_per_s_median=float(np.median(r2s[valid])),
                      intravoxel_R2star_per_s_max=float(r2s[valid].max()))
    with open(os.path.join(outdir, "a3_lut_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


# ----------------------------------------------------------------------------------
# head phantom (Ridani anatomy)
# ----------------------------------------------------------------------------------
DATA = "/scratch/user/uqaste15/qsm-forward-dev/data"
CHI = "/scratch/user/uqaste15/ridani-ref/data/chimodel"
NAMES = {1: "caudate", 2: "pallidus", 3: "putamen", 4: "red_nucleus", 5: "dentate",
         6: "s_nigra", 7: "thalamus", 8: "wm", 9: "gm", 10: "csf", 11: "blood"}


def head(lut_path, outdir, slice_only=False, chunk=200000):
    import nibabel as nib
    os.makedirs(outdir, exist_ok=True)
    L = SphereLUT(lut_path)
    segi = nib.load(f"{DATA}/masks/SegmentedModel.nii.gz")
    seg = segi.get_fdata().astype(int)
    cpi = nib.load(f"{CHI}/Chi_positive.nii.gz")
    cp = cpi.get_fdata()
    cn = nib.load(f"{CHI}/Chi_negative.nii.gz").get_fdata()
    m = (seg > 0) & (seg != 10) & (seg != 11)      # brain tissue (CSF, blood: no sources here)
    if slice_only:
        keep = np.zeros_like(m)
        keep[:, :, seg.shape[2] // 2] = True
        m &= keep
    idx = np.where(m.ravel())[0]
    lim = wrap_limit_ppm(TE_CHEN[-1])
    tot = (cp + cn).ravel()[idx]
    stats = dict(wrap_limit_ppm=float(lim), te_ms=TE_CHEN.tolist(), n_vox=int(len(idx)),
                 frac_over_wrap=float(np.mean(np.abs(tot) > lim)), tissues={})
    zpp, spp = source_params(cp.ravel()[idx])
    stats["frac_chi_s_pos_above_0.15"] = float(np.mean(spp > CHI_S_PAPER + 1e-9))
    zmm, smm = source_params(cn.ravel()[idx])
    stats["frac_chi_s_neg_above_0.15"] = float(np.mean(smm > CHI_S_PAPER + 1e-9))
    for c in CONVENTIONS:
        chi_e = np.zeros((len(idx), len(TE_CHEN)), np.float32)
        nl = np.zeros(len(idx), np.float32)
        r2s = np.zeros(len(idx), np.float32)
        for s0 in range(0, len(idx), chunk):
            sl = slice(s0, s0 + chunk)
            ii = idx[sl]
            ce = L.chi_echoes(cp.ravel()[ii], cn.ravel()[ii], TE_CHEN, c)
            chi_e[sl] = ce
            bulk = (cp.ravel()[ii] + cn.ravel()[ii])[:, None]
            ph = (ce - bulk) * (2 / 3) * GAMMA * B0 * 1e-6 * TE_CHEN[None]
            nl[sl] = nonlinear_phase(ph, TE_CHEN)
            F = L.factor(cp.ravel()[ii], cn.ravel()[ii], TE_CHEN, c)
            r2s[sl] = meso.fit_rate(TE_CHEN, np.abs(F).T)
        # DECOMPOSE's synthesised phase (2/3) gamma B0 chi(TE) TE must stay in (-pi, pi]
        dph = (2 / 3) * GAMMA * B0 * 1e-6 * chi_e * TE_CHEN[None]
        stats[f"{c}_frac_echo_phase_wrapped"] = float(np.mean(np.any(np.abs(dph) > np.pi, 1)))
        segv = seg.ravel()[idx]
        for lab, name in NAMES.items():
            mm = segv == lab
            if not mm.any():
                continue
            t = stats["tissues"].setdefault(name, dict(
                n=int(mm.sum()), chi_pos=float(np.median(cp.ravel()[idx][mm])),
                chi_neg=float(np.median(cn.ravel()[idx][mm])),
                frac_over_wrap=float(np.mean(np.abs(tot[mm]) > lim))))
            t[f"{c}_nonlinear_rad_median"] = float(np.median(nl[mm]))
            t[f"{c}_nonlinear_rad_p95"] = float(np.percentile(nl[mm], 95))
            t[f"{c}_chi_TE_first_last_median"] = [float(np.median(chi_e[mm, 0])),
                                                  float(np.median(chi_e[mm, -1]))]
            t[f"{c}_intravoxel_R2star_median"] = float(np.median(r2s[mm]))
        stats[f"{c}_nonlinear_rad_brain"] = dict(
            median=float(np.median(nl)), p95=float(np.percentile(nl, 95)), max=float(nl.max()))
        if not slice_only:
            vol = np.zeros(seg.shape + (len(TE_CHEN),), np.float32)
            vol.reshape(-1, len(TE_CHEN))[idx] = chi_e
            img = nib.Nifti1Image(vol, cpi.affine)
            img.header.set_xyzt_units("mm")
            nib.save(img, os.path.join(outdir, f"a3_{c}_Chimap-echoes.nii.gz"))
            nlv = np.zeros(seg.shape, np.float32)
            nlv.ravel()[idx] = nl
            nib.save(nib.Nifti1Image(nlv, cpi.affine), os.path.join(outdir, f"a3_{c}_nonlinear-phase.nii.gz"))
    with open(os.path.join(outdir, "a3_head_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=N)
    m = sub.add_parser("merge")
    m.add_argument("--parts", required=True)
    m.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--out", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--lut", required=True)
    s.add_argument("--out", required=True)
    h = sub.add_parser("head")
    h.add_argument("--lut", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--slice-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.seed, a.out, a.n)
    elif a.cmd == "merge":
        merge(a.parts, a.out)
    elif a.cmd == "validate":
        validate(a.out)
    elif a.cmd == "stats":
        lut_stats(a.lut, a.out)
    else:
        head(a.lut, a.out, a.slice_only)
