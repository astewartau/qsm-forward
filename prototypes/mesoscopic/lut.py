"""Merge LUT parts and evaluate the mesoscopic signal factor per voxel.

Merge:
    python lut.py merge --parts DIR --mode {static,diffusion} --out FILE.npz

Evaluate (library use, e.g. from qsm-forward's signal model):
    L = MesoLUT("lut_diffusion.npz")
    F = L.decay(chi_pos, chi_neg, theta, B0, TE_ms, T2_ms=..., rho=...)   # complex

``decay`` returns the complex mesoscopic factor that replaces the real exp(-TE*R2*)
envelope of the voxel GRE signal (the macroscopic field phase is applied on top, as
now). It is sum_c w_c exp(-TE/T2_c) exp(i 2pi gbar B0 E_c TE) S_c(TE), normalised so
that sum_c w_c = 1, with S_c the LUT compartment signal (extra, myelin, axon),
w_c = rho_c * f_c (water density x volume fraction).
"""
import argparse
import glob
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

COMPARTMENTS = ("extra", "myelin", "axon")
GBAR = 42.577478e6  # Hz/T


def merge(parts_dir, mode, out):
    files = sorted(glob.glob(os.path.join(parts_dir, f"lut_{mode}_g*_s*.npz")))
    if not files:
        raise FileNotFoundError(parts_dir)
    d0 = np.load(files[0])
    chi_pos, theta, B0, TE, TE_SE = (d0[k] for k in ("chi_pos", "theta", "B0", "TE", "TE_SE"))
    # all chi- nodes across geometries
    import build_lut
    chi_neg = build_lut.CHI_NEG
    nn = len(chi_neg)
    acc = {k: [[] for _ in range(nn)] for k in ("gre", "se", "frac")}
    meta = {k: [None] * nn for k in ("fvf", "mvf", "sheath_scale")}
    for f in files:
        d = np.load(f)
        for a, ineg in enumerate(d["chi_neg_idx"]):
            acc["gre"][ineg].append(d["gre"][a])
            acc["se"][ineg].append(d["se"][a])
            acc["frac"][ineg].append(d["frac"])
            meta["fvf"][ineg] = float(d["fvf"])
            meta["mvf"][ineg] = float(d["mvf"])
            meta["sheath_scale"][ineg] = float(d["sheath_scale"][a])
    missing = [i for i in range(nn) if not acc["gre"][i]]
    if missing:
        raise RuntimeError(f"missing chi- nodes {missing}")
    nseed = [len(x) for x in acc["gre"]]
    gre = np.stack([np.mean(x, 0) for x in acc["gre"]])
    se = np.stack([np.mean(x, 0) for x in acc["se"]])
    gre_sd = np.stack([np.std(x, 0) for x in acc["gre"]]).astype(np.float32)
    se_sd = np.stack([np.std(x, 0) for x in acc["se"]]).astype(np.float32)
    frac = np.stack([np.mean(x, 0) for x in acc["frac"]])
    np.savez_compressed(out, gre=gre.astype(np.complex64), se=se.astype(np.complex64),
                        gre_sd=gre_sd, se_sd=se_sd, frac=frac, chi_pos=chi_pos,
                        chi_neg=chi_neg, theta=theta, B0=B0, TE=TE, TE_SE=TE_SE,
                        n_realisations=np.array(nseed), fvf=np.array(meta["fvf"]),
                        mvf=np.array(meta["mvf"]), sheath_scale=np.array(meta["sheath_scale"]),
                        params=d0["params"], mode=mode)
    print(f"wrote {out}: gre {gre.shape}, realisations per chi- node {nseed}")


def _logsig(S, axis=-1):
    """log of complex signal with phase unwrapped along TE; NaN -> 0."""
    S = np.where(np.isfinite(S), S, 1.0)
    mag = np.log(np.maximum(np.abs(S), 1e-30))
    ph = np.unwrap(np.angle(S), axis=axis)
    return mag + 1j * ph


class MesoLUT:
    def __init__(self, path):
        d = np.load(path)
        self.chi_pos = d["chi_pos"]
        self.chi_neg = d["chi_neg"]
        self.theta = d["theta"]
        self.B0 = d["B0"]
        self.TE = d["TE"]
        self.TE_SE = d["TE_SE"]
        self.frac = d["frac"]                      # (n_neg, 3)
        self.mode = str(d["mode"])
        gre = d["gre"]                             # (neg, pos, th, B0, comp, TE)
        se = d["se"]
        # interpolate in increasing chi- order (axis is |chi-|)
        self._neg_axis = np.abs(self.chi_neg)
        self._lg = _logsig(gre)
        self._ls = _logsig(se)
        self._fr = RegularGridInterpolator((self._neg_axis,), self.frac)

    def _interp(self, arr, tgrid, ib, ci, cp, cn, th, te):
        a = np.moveaxis(arr[:, :, :, ib, ci], 0, 1)   # (pos, neg, th, TE)
        pts = (self.chi_pos, self._neg_axis, self.theta, tgrid)
        f_re = RegularGridInterpolator(pts, a.real)
        f_im = RegularGridInterpolator(pts, a.imag)
        q = np.stack([cp, cn, th, np.full_like(cp, te)], -1)
        return np.exp(f_re(q) + 1j * f_im(q))

    def _prep(self, chi_pos, chi_neg, theta_rad):
        cp = np.clip(np.asarray(chi_pos, float).ravel(), self.chi_pos[0], self.chi_pos[-1])
        cn = np.clip(np.abs(np.minimum(np.asarray(chi_neg, float), 0)).ravel(),
                     self._neg_axis[0], self._neg_axis[-1])
        th = np.rad2deg(np.asarray(theta_rad, float)).ravel() % 180
        th = np.where(th > 90, 180 - th, th)
        return cp, cn, th

    def _ib(self, B0):
        ib = np.where(np.isclose(self.B0, B0))[0]
        if not len(ib):
            raise ValueError(f"B0={B0} not in LUT {self.B0}")
        return int(ib[0])

    def compartment_signals(self, chi_pos, chi_neg, theta_rad, B0, TE_ms, se=False,
                            powder=False):
        """Per-compartment complex signals (3, n_vox) at one TE (ms).

        ``powder``: fibres uniformly distributed over orientation within the voxel (grey
        matter, no V1). The voxel signal is then the solid-angle-weighted *linear*
        average of the complex signal over theta (each fibre population contributes its
        own water; no exchange between populations)."""
        cp, cn, th = self._prep(chi_pos, chi_neg, theta_rad if not powder else np.zeros(np.size(chi_pos)))
        ib = self._ib(B0)
        arr, tg = (self._ls, self.TE_SE) if se else (self._lg, self.TE)
        out = []
        for ci in range(3):
            if not powder:
                out.append(self._interp(arr, tg, ib, ci, cp, cn, th, TE_ms))
            else:
                acc = 0
                for tk, wk in zip(self.theta, self.powder_weights()):
                    acc = acc + wk * self._interp(arr, tg, ib, ci, cp, cn,
                                                  np.full_like(cp, tk), TE_ms)
                out.append(acc)
        shape = np.shape(chi_pos)
        return np.stack(out).reshape((3,) + shape)

    def powder_weights(self):
        """Solid-angle weights of the theta nodes (cos of the bin edges, half bins at
        0 and 90 deg) for a uniform orientation distribution."""
        t = np.deg2rad(self.theta)
        edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])
        w = np.cos(edges[:-1]) - np.cos(edges[1:])
        return w / w.sum()

    def weights(self, chi_neg, rho=(1.0, 0.5, 1.0)):
        cn = np.clip(np.abs(np.minimum(np.asarray(chi_neg, float), 0)), self._neg_axis[0],
                     self._neg_axis[-1])
        f = self._fr(cn.ravel()[:, None]).T * np.asarray(rho)[:, None]
        return (f / f.sum(0)).reshape((3,) + np.shape(chi_neg))

    def decay(self, chi_pos, chi_neg, theta_rad, B0, TE_ms, T2_ms=(48.0, 10.0, 64.0),
              rho=(1.0, 0.5, 1.0), E_ppm=(0.0, 0.02, 0.0), se=False, powder=False):
        """Complex voxel factor at one TE (ms). T2_ms / E_ppm: per compartment
        (extra, myelin, axon), scalars or voxel arrays. Returns array like chi_pos."""
        S = self.compartment_signals(chi_pos, chi_neg, theta_rad, B0, TE_ms, se, powder)
        w = self.weights(chi_neg, rho)
        out = 0
        for ci in range(3):
            fac = np.exp(-TE_ms / np.asarray(T2_ms[ci], float))
            if not se:
                fac = fac * np.exp(2j * np.pi * GBAR * B0 * np.asarray(E_ppm[ci]) * 1e-6
                                   * TE_ms * 1e-3)
            out = out + w[ci] * fac * S[ci]
        return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--parts", required=True)
    m.add_argument("--mode", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.parts, a.mode, a.out)


# ----------------------------------------------------------------------------------
# chi- ground truth
# ----------------------------------------------------------------------------------
def chi_neg_ground_truth(chi_neg):
    """Scored chi- (orientation independent): mean sheath susceptibility (tr/3 of the
    sheath tensor, i.e. chi_I x sheath scale) x myelin volume fraction. This is the LUT's
    chi- axis value itself, so the scored map is the input chi- map."""
    return np.minimum(np.asarray(chi_neg, float), 0.0)


def chi_neg_apparent(chi_neg, theta_rad, chi_i=-0.10, chi_a=-0.10, powder=False):
    """Secondary, orientation-dependent apparent chi- (what a scalar dipole inversion of
    the voxel-mean susceptibility tensor reports): chi_bar + dchi_bulk (cos^2 th - 1/3).

    For the radially anisotropic sheath chi_I I + chi_A diag(1, -1/2, -1/2) in
    (r, phi, z), the azimuthal average is chi_I + chi_A diag(1/4, 1/4, -1/2) in (x, y, z),
    so the bulk anisotropy is chi_par - chi_perp = -(3/4) chi_A x (sheath scale x MVF)
    = -(3/4) (chi_A/chi_I) chi-. Powder (grey matter): the cos^2 term averages to 1/3,
    so the apparent value equals the ground truth. Microstructural (hollow-cylinder)
    frequency offsets are not included: they are not susceptibility."""
    cn = chi_neg_ground_truth(chi_neg)
    if powder:
        return cn
    c2 = np.cos(np.asarray(theta_rad, float)) ** 2
    return cn * (1 - 0.75 * (chi_a / chi_i) * (c2 - 1 / 3))
