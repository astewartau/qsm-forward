"""Mesoscopic χ-separation signal models: per-voxel signal from explicit sub-voxel sources.

Two look-up-table (LUT) driven signal models for :mod:`qsm_forward.chisep`. In both, no
relaxivity (Dr), pool frequency or R2' formula goes into the signal: the decay, its
short-TE behaviour, its orientation dependence and any non-linear phase come out of the
field that explicitly placed sub-voxel sources produce. The LUTs are built offline by the
Monte Carlo / FFT generators on the ``feat/chisep-mesoscopic`` branch
(``prototypes/mesoscopic/``: ``build_lut.py``, ``spheres_a3.py``); this module only reads
them and evaluates them per voxel.

``mesoscopic`` (:class:`MesoLUT`, :class:`MesoscopicSignal`)
    χ- = myelinated axons: straight parallel hollow cylinders (polydisperse, g-ratio 0.7,
    fibre volume fraction <= 0.6) with the Wharton & Bowtell (2012) sheath tensor
    χ_I·I + χ_A·diag(1, -1/2, -1/2) in (r, φ, z), χ_I = χ_A = -0.10 ppm; beyond what FVF 0.6
    gives, the whole tensor is scaled so the voxel χ- equals the input map. Three water
    compartments: extra-axonal, myelin (proton density 0.5) and axonal.
    χ+ = iron in two sphere populations: clusters (R = 3 µm, Δχ = 20 ppm; static-to-
    intermediate regime, most of R2') and ferritin-scale spheres (R = 0.5 µm, Δχ = 3 ppm;
    motional narrowing, they raise R2 and hardly R2'). The cluster share is calibrated so that
    Dr+ = R2'/χ+ from mono-exponential fits over :data:`PROTOCOL_3T` (GRE minus Hahn SE) is
    ≈ 140 Hz/ppm at 3 T with diffusion (Shin et al. 2021: 137·B0/2.9).
    Tables: ``diffusion`` (Monte Carlo random walk, D = 1 µm²/ms; the default) and ``static``
    (static dephasing), same geometries. Axes χ+ (0-0.3 ppm), χ- (0 to -0.08 ppm), fibre-to-B0
    angle θ (0-90°), B0 (3, 7 T), TE (GRE 0-80 ms, SE 0-80 ms).
    White matter with a fibre direction uses its fibre angle; elsewhere (grey matter, deep
    grey matter, WM without a fibre direction) the orientation average: the solid-angle
    weighted linear average of the complex signal over θ.

``spheres`` (:class:`SphereLUT`, :class:`SphereSignal`)
    Chen et al. (2021, §3.2) recipe: per voxel a periodic box of 128³ sub-voxels holding random
    para- and diamagnetic spheres (radius 6-16 sub-voxels, no overlap between signs) that
    contain water at the same proton density as the medium, static dephasing, 3 T; voxel
    signal = mean of exp(iγB0 f t) over all sub-voxels. DECOMPOSE's multi-pool signal equation
    is not used. Two intra-source conventions, both with zero voxel-mean frequency:
    ``decompose`` (kernel 1 - kz²/k²: water inside a source precesses at (2/3)χ_s γB0, the
    paper's App. A3) and ``lorentz`` (1/3 - kz²/k², the Lorentz-corrected standard).
    Bulk χ maps to (volume fraction ζ, source χ_s) through :func:`source_params`.

Ground truth and conventions
    * The macroscopic field is qsm-forward's standard forward model of the voxel-mean χ; LUT
      signals are offsets from it (the LUT boxes have D(0) = 0).
    * χ+ = Σ Δχ·(sphere volume fraction); χ- = (tr/3 of the sheath tensor)·(myelin volume
      fraction), orientation independent: :func:`chi_neg_ground_truth`, the scored map.
      The voxel-mean sheath tensor projected on B0 gives the orientation-dependent apparent
      χ-, :func:`chi_neg_apparent` (equal to the ground truth under the orientation
      average); it is what drives the macroscopic field of white matter.
    * Compartment relaxation (``mesoscopic``): myelin water decays at ``T2_MYELIN_MS`` and
      carries the exchange shift ``E_MYELIN_PPM``; extra-axonal and axonal water decay at an
      intrinsic R2 chosen per voxel so that the simulated spin echo reproduces the tissue R2
      map (:meth:`MesoscopicSignal.intrinsic_r2`). Iron's diffusion-driven irreversible loss
      is in the LUT's spin-echo table, so the spin echo is evaluated from the LUT too, and the
      tissue R2 is not counted twice.
    * Interpolation (``mesoscopic``): multilinear in the log-signal (log-magnitude plus phase
      unwrapped along TE); ``spheres``: in the complex signal (see :class:`SphereSignal`).
      χ+ beyond the table's last node is extrapolated linearly in the log-signal
      (log S is linear in volume fraction in the dilute static limit); χ-, θ and TE are clipped
      to the table.

LUT files (not in the repository; several MB each) are found through ``lut_path``: an
explicit path, else ``$QSM_FORWARD_MESO_LUT_DIR/<name>``.
TODO: publish the LUTs as a versioned download (e.g. a GitHub release asset of
astewartau/qsm-forward or a Hugging Face dataset) and fetch them on demand into a cache;
until then they are read from a local path.

References
    Wharton S., Bowtell R. PNAS 2012;109:18559. doi:10.1073/pnas.1211075109
    Chen J. et al. NeuroImage 2021;242:118477. doi:10.1016/j.neuroimage.2021.118477
    Shin H.-G. et al. NeuroImage 2021;240:118371. doi:10.1016/j.neuroimage.2021.118371
    Yablonskiy D.A., Haacke E.M. MRM 1994;32:749. doi:10.1002/mrm.1910320610
    Wei H. et al. NMR Biomed 2015;28:1294 (STAR-QSM). doi:10.1002/nbm.3383
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

GAMMA = 2.6752218744e5          # rad / ms / T (1H)
GAMMA_BAR = 42.577478e6         # Hz / T, as the LUT generators use
COMPARTMENTS = ("extra", "myelin", "axon")
RHO = (1.0, 0.5, 1.0)           # proton density per compartment (myelin water 0.5)
T2_MYELIN_MS = 10.0             # myelin-water T2 (Wharton & Bowtell 2012 model, ~10 ms)
E_MYELIN_PPM = 0.02             # myelin-water exchange shift (ppm)
CHI_I, CHI_A = -0.10, -0.10     # sheath tensor (ppm), Wharton & Bowtell 2012
# Mono-exponential fit protocol of the iron calibration (ms): GRE 5-40 ms, Hahn SE 10-40 ms
PROTOCOL_3T = dict(gre_ms=np.arange(5.0, 40.01, 5.0), se_ms=np.arange(10.0, 40.01, 10.0))
BRAIN_LABELS = tuple(range(1, 10))   # head-model tissue labels that carry sub-voxel sources
WM_LABEL = 8

LUT_ENV = "QSM_FORWARD_MESO_LUT_DIR"
LUT_FILES = {
    "diffusion": "meso_diffusion_v3.npz",
    "static": "meso_static_v3.npz",
    "spheres": "spheres_a3_v1.npz",
}


def lut_path(kind: str, path: Optional[str] = None) -> str:
    """Path of LUT ``kind`` ("diffusion", "static" or "spheres"): ``path`` if given, else
    ``$QSM_FORWARD_MESO_LUT_DIR/<LUT_FILES[kind]>``."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    if kind not in LUT_FILES:
        raise KeyError(f"unknown LUT {kind!r}; choose from {sorted(LUT_FILES)}")
    root = os.environ.get(LUT_ENV)
    if not root:
        raise FileNotFoundError(
            f"the {kind} LUT ({LUT_FILES[kind]}) is not bundled; set ${LUT_ENV} to the directory "
            f"holding it, or pass its path (ChisepConfig.meso_lut)")
    p = os.path.join(root, LUT_FILES[kind])
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def _logsig(S, axis=-1):
    """log of a complex signal with the phase unwrapped along ``axis`` (TE); non-finite or
    zero entries -> log 1."""
    S = np.where(np.isfinite(S) & (np.abs(S) > 0), S, 1.0)
    return np.log(np.abs(S)) + 1j * np.unwrap(np.angle(S), axis=axis)


def fit_rate(t_ms, S):
    """Mono-exponential rate (1/s) of |S| along axis 0 from a log-linear least-squares fit
    (t in ms)."""
    t = np.asarray(t_ms, float)
    y = np.log(np.maximum(np.abs(S), 1e-30))
    tc = (t - t.mean()).reshape((-1,) + (1,) * (y.ndim - 1))
    return -(y * tc).sum(0) / (tc ** 2).sum() * 1e3


# ----------------------------------------------------------------------------------------
# χ- ground truth
# ----------------------------------------------------------------------------------------
def chi_neg_ground_truth(chi_neg):
    """Scored χ- (orientation independent): sheath tr/3 × myelin volume fraction, i.e. the LUT's
    χ- axis value, so the scored map is the input χ- map (clipped to <= 0)."""
    return np.minimum(np.asarray(chi_neg, float), 0.0)


def chi_neg_apparent(chi_neg, theta_rad, chi_i=CHI_I, chi_a=CHI_A):
    """Orientation-dependent apparent χ-: the voxel-mean sheath tensor projected on B0,
    χ-·[1 - (3/4)(χ_A/χ_I)(cos²θ - 1/3)].

    The azimuthal average of χ_I·I + χ_A·diag(1, -1/2, -1/2) in (r, φ, z) is
    χ_I·I + χ_A·diag(1/4, 1/4, -1/2) in (x, y, z), so χ_par - χ_perp = -(3/4)χ_A per unit
    sheath. NaN θ (no fibre direction: orientation average) gives the ground truth. The
    hollow-cylinder frequency offsets of the compartments are not susceptibility and are not
    included."""
    cn = chi_neg_ground_truth(chi_neg)
    th = np.asarray(theta_rad, float)
    c2 = np.where(np.isfinite(th), np.cos(np.nan_to_num(th)) ** 2, 1.0 / 3.0)
    return cn * (1 - 0.75 * (chi_a / chi_i) * (c2 - 1.0 / 3.0))


# ----------------------------------------------------------------------------------------
# mesoscopic (myelin + iron) LUT
# ----------------------------------------------------------------------------------------
class MesoLUT:
    """Reader of a merged mesoscopic LUT: ``gre[χ-, χ+, θ, B0, compartment, TE]`` (complex),
    ``se[..., TE_SE]``, ``frac[χ-, compartment]`` and the axes."""

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.chi_pos = np.asarray(d["chi_pos"], float)
        self.chi_neg = np.asarray(d["chi_neg"], float)
        self.theta = np.asarray(d["theta"], float)
        self.B0 = np.asarray(d["B0"], float)
        self.TE = np.asarray(d["TE"], float)
        self.TE_SE = np.asarray(d["TE_SE"], float)
        self.frac = np.asarray(d["frac"], float)
        self.mode = str(d["mode"]) if "mode" in d.files else "unknown"
        self.params = str(d["params"]) if "params" in d.files else ""
        self._neg = np.abs(self.chi_neg)
        if np.any(np.diff(self._neg) <= 0):
            raise ValueError("LUT χ- axis must run 0, -a, -b, ... (|χ-| increasing)")
        self._gre = np.asarray(d["gre"])
        self._se = np.asarray(d["se"])
        self._fr = RegularGridInterpolator((self._neg,), self.frac)
        self._cache = {}

    # ---- tables ----
    def powder_weights(self):
        """Solid-angle weights of the θ nodes for a uniform orientation distribution (cosines
        of the bin edges, half bins at 0° and 90°)."""
        t = np.deg2rad(self.theta)
        edges = np.concatenate([[t[0]], 0.5 * (t[1:] + t[:-1]), [t[-1]]])
        w = np.cos(edges[:-1]) - np.cos(edges[1:])
        return w / w.sum()

    def _ib(self, B0):
        ib = np.where(np.isclose(self.B0, B0))[0]
        if not len(ib):
            raise ValueError(f"B0 = {B0} T is not in the LUT ({self.B0.tolist()} T)")
        return int(ib[0])

    def _interpolator(self, se, ib, ci, powder):
        key = (se, ib, ci, powder)
        if key not in self._cache:
            arr = self._se if se else self._gre
            tg = self.TE_SE if se else self.TE
            a = arr[:, :, :, ib, ci]                                  # (neg, pos, th, TE)
            if powder:
                # linear average of the complex signal over orientation, then the log table
                a = np.tensordot(a, self.powder_weights(), axes=([2], [0]))    # (neg, pos, TE)
                L = np.moveaxis(_logsig(a), 0, 1)                     # (pos, neg, TE)
                pts = (self.chi_pos, self._neg, tg)
            else:
                L = np.moveaxis(_logsig(a), 0, 1)                     # (pos, neg, th, TE)
                pts = (self.chi_pos, self._neg, self.theta, tg)
            self._cache[key] = RegularGridInterpolator(pts, L, bounds_error=False, fill_value=None)
        return self._cache[key]

    def _query(self, chi_pos, chi_neg, theta_rad, te, powder, tgrid):
        cp = np.maximum(np.asarray(chi_pos, float).ravel(), 0.0)          # extrapolated above
        cn = np.clip(-np.minimum(np.asarray(chi_neg, float).ravel(), 0.0), self._neg[0], self._neg[-1])
        t = np.full_like(cp, np.clip(te, tgrid[0], tgrid[-1]))
        if powder:
            return np.stack([cp, cn, t], -1)
        th = np.rad2deg(np.asarray(theta_rad, float)).ravel() % 180.0
        th = np.clip(np.where(th > 90.0, 180.0 - th, th), self.theta[0], self.theta[-1])
        return np.stack([cp, cn, th, t], -1)

    def compartment_signals(self, chi_pos, chi_neg, theta_rad, B0, TE_ms, se=False, powder=False):
        """Per-compartment complex signals, shape (3,) + shape(chi_pos), at one TE (ms)."""
        ib = self._ib(B0)
        tg = self.TE_SE if se else self.TE
        q = self._query(chi_pos, chi_neg, theta_rad, TE_ms, powder, tg)
        out = [np.exp(self._interpolator(se, ib, ci, powder)(q)) for ci in range(3)]
        return np.stack(out).reshape((3,) + np.shape(chi_pos))

    def weights(self, chi_neg, rho=RHO):
        """Signal weights ρ_c·f_c of the compartments (normalised), shape (3,) + shape(chi_neg)."""
        cn = np.clip(-np.minimum(np.asarray(chi_neg, float), 0.0), self._neg[0], self._neg[-1])
        f = self._fr(cn.ravel()[:, None]).T * np.asarray(rho, float)[:, None]
        return (f / f.sum(0)).reshape((3,) + np.shape(chi_neg))

    def decay(self, chi_pos, chi_neg, theta_rad, B0, TE_ms, T2_ms, rho=RHO,
              E_ppm=(0.0, E_MYELIN_PPM, 0.0), se=False, powder=False):
        """Complex voxel factor at one TE (ms): Σ_c w_c exp(-TE/T2_c) exp(i2πγ̄B0 E_c TE) S_c(TE).
        ``T2_ms``/``E_ppm``: per compartment (extra, myelin, axon), scalars or voxel arrays.
        The spin echo (``se``) refocuses the exchange shift."""
        S = self.compartment_signals(chi_pos, chi_neg, theta_rad, B0, TE_ms, se, powder)
        w = self.weights(chi_neg, rho)
        out = 0
        for ci in range(3):
            fac = np.exp(-TE_ms / np.asarray(T2_ms[ci], float))
            if not se:
                fac = fac * np.exp(2j * np.pi * GAMMA_BAR * B0 * np.asarray(E_ppm[ci]) * 1e-9 * TE_ms)
            out = out + w[ci] * fac * S[ci]
        return out

    def iron_relaxivity(self, B0, chis=(0.05, 0.10, 0.15, 0.20), protocol=PROTOCOL_3T):
        """Dr+ = R2'/χ+ and R2/χ+ of iron alone (χ- = 0, orientation average, extra-axonal water,
        no intrinsic T2): mono-exponential fits over ``protocol`` (GRE minus Hahn SE)."""
        cp = np.asarray(chis, float)
        z = np.zeros_like(cp)
        kw = dict(T2_ms=(1e12,) * 3, rho=(1.0, 0.0, 0.0), powder=True)
        g = np.array([self.decay(cp, z, z, B0, t, **kw) for t in protocol["gre_ms"]])
        e = np.array([self.decay(cp, z, z, B0, t, se=True, **kw) for t in protocol["se_ms"]])
        r2s, r2 = fit_rate(protocol["gre_ms"], g), fit_rate(protocol["se_ms"], e)
        return dict(chi_pos=cp.tolist(), Dr_plus=((r2s - r2) / cp).tolist(),
                    R2_per_ppm=(r2 / cp).tolist(), R2star_per_ppm=(r2s / cp).tolist())


class MesoscopicSignal:
    """Per-voxel complex GRE and SE factors of the ``mesoscopic`` model on a simulation grid.

    ``theta_deg`` is the fibre-to-B0 angle (degrees, NaN where there is no fibre direction).
    Voxels with a label in ``BRAIN_LABELS`` get the LUT; white matter (``WM_LABEL``) with a
    finite fibre angle uses it, every other brain voxel the orientation average. ``gre`` and
    ``se`` return ``(mask, values)``: the factor replaces the real decay envelope in those
    voxels (the macroscopic phase and steady-state weighting go on top), and the caller keeps
    its own model elsewhere."""

    def __init__(self, lut: MesoLUT, chi_pos, chi_neg, theta_deg, seg, B0,
                 se_tes_ms=PROTOCOL_3T["se_ms"], r2_floor=1.0):
        self.lut = lut
        self.B0 = float(B0)
        self.se_tes_ms = np.asarray(se_tes_ms, float)
        self.r2_floor = float(r2_floor)
        self._r2_key = None
        lut._ib(self.B0)
        seg = np.rint(np.asarray(seg)).astype(int)
        self.mask = np.isin(seg, BRAIN_LABELS)
        th = np.asarray(theta_deg, float) if theta_deg is not None else np.full(seg.shape, np.nan)
        fibre = self.mask & (seg == WM_LABEL) & np.isfinite(th)
        m = self.mask
        self._fibre = fibre[m]
        self._cp = np.asarray(chi_pos, float)[m]
        self._cn = np.asarray(chi_neg, float)[m]
        self._th = np.deg2rad(np.nan_to_num(th[m]))

    def intrinsic_r2(self, R2):
        """Intrinsic R2 (1/s) of the extra-axonal and axonal water, per brain voxel, such that the
        simulated spin echo reproduces the tissue R2 map: the tissue map is what a spin-echo
        measurement gives, and part of it is the LUT's own irreversible loss (iron in the
        motional-narrowing regime, the fast myelin-water pool). One correction step: evaluate the
        SE over ``se_tes_ms`` at R2_0 = R2_tissue, fit its rate, subtract the excess; floored at
        ``r2_floor``. ``floored_fraction`` records how often the floor applied."""
        key = id(R2)
        if self._r2_key == key:
            return self._r2_0
        r2 = np.broadcast_to(np.asarray(R2, float), self.mask.shape)[self.mask].astype(float)
        se = np.array([np.abs(self._eval(t, r2, se=True, r2_0=r2)) for t in self.se_tes_ms])
        excess = fit_rate(self.se_tes_ms, se) - r2
        r2_0 = r2 - excess
        self.floored_fraction = float(np.mean(r2_0 < self.r2_floor)) if r2_0.size else 0.0
        self._r2_0 = np.maximum(r2_0, self.r2_floor)
        self._r2_key = key
        return self._r2_0

    def _eval(self, TE_ms, R2, se, r2_0=None):
        r2 = self.intrinsic_r2(R2) if r2_0 is None else r2_0
        t2 = 1e3 / np.maximum(r2, 1e-9)                     # ms
        out = np.empty(self._cp.shape, np.complex128)
        for powder, sel in ((False, self._fibre), (True, ~self._fibre)):
            if not sel.any():
                continue
            out[sel] = self.lut.decay(self._cp[sel], self._cn[sel], self._th[sel], self.B0, TE_ms,
                                      T2_ms=(t2[sel], T2_MYELIN_MS, t2[sel]), se=se, powder=powder)
        return out

    def gre(self, TE_s, R2):
        return self.mask, self._eval(TE_s * 1e3, R2, se=False)

    def se(self, TE_s, R2):
        return self.mask, np.abs(self._eval(TE_s * 1e3, R2, se=True))


# ----------------------------------------------------------------------------------------
# water-containing spheres (DECOMPOSE-targeted, A3)
# ----------------------------------------------------------------------------------------
SPHERE_CONVENTIONS = ("decompose", "lorentz")
# source_params: ζ = ZETA_REF while χ_s rises 0.01 -> 0.15 ppm (Chen 2021's range), then ζ
# rises to ZETA_MAX, then χ_s rises above the paper's range.
ZETA_REF, ZETA_MAX, CHI_S_MIN, CHI_S_PAPER = 0.25, 0.35, 0.01, 0.15


def source_params(chi_bulk):
    """Bulk susceptibility (either sign) -> (ζ, χ_s >= 0) with ζ·χ_s = |bulk| exactly.

    |bulk| <= 0.0375 ppm: ζ = 0.25 (below 0.0025 ppm: χ_s = 0.01, ζ < 0.25) and χ_s 0.01-0.15;
    up to 0.0525 ppm: χ_s = 0.15, ζ 0.25-0.35; above: ζ = 0.35, χ_s = |bulk|/0.35 (outside
    the paper's χ_s range)."""
    b = np.abs(np.asarray(chi_bulk, float))
    chi_s = np.clip(b / ZETA_REF, CHI_S_MIN, CHI_S_PAPER)
    chi_s = np.where(b > ZETA_MAX * CHI_S_PAPER, b / ZETA_MAX, chi_s)
    zeta = np.minimum(b / chi_s, ZETA_MAX)
    return zeta, chi_s


def wrap_limit_ppm(te_max_ms, b0=3.0):
    """|χ| above which DECOMPOSE's synthesised phase (2/3)γB0·χ·TE leaves (-π, π] at TE_max."""
    return np.pi / ((2.0 / 3.0) * GAMMA * b0 * 1e-6 * te_max_ms)


class SphereLUT:
    """Reader of the sphere LUT: ``S_<convention>[ζ+, ζ-, χ_s+, χ_s-, TE]`` (complex), axes
    ``zeta``, ``chi_s``, ``te`` (ms) and ``b0``."""

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.zeta = np.asarray(d["zeta"], float)
        self.chi_s = np.asarray(d["chi_s"], float)
        self.te = np.asarray(d["te"], float)
        self.b0 = float(d["b0"])
        self._S = {c: np.asarray(d[f"S_{c}"]) for c in SPHERE_CONVENTIONS if f"S_{c}" in d.files}
        self._f = {}

    def _interp(self, conv, log):
        key = (conv, log)
        if key not in self._f:
            if conv not in self._S:
                raise KeyError(f"convention {conv!r} not in {self.path}")
            S = self._S[conv]
            vals = _logsig(S) if log else S.astype(np.complex128)
            self._f[key] = RegularGridInterpolator((self.zeta, self.zeta, self.chi_s, self.chi_s, self.te), vals)
        return self._f[key]

    def _q(self, chi_pos, chi_neg, te_ms):
        zp, sp = source_params(np.maximum(chi_pos, 0))
        zm, sm = source_params(np.minimum(chi_neg, 0))
        sp = np.clip(sp, self.chi_s[0], self.chi_s[-1])
        sm = np.clip(sm, self.chi_s[0], self.chi_s[-1])
        te_ms = np.clip(np.atleast_1d(np.asarray(te_ms, float)), self.te[0], self.te[-1])
        q = np.empty((np.size(zp), len(te_ms), 5))
        for k, v in enumerate((zp, zm, sp, sm)):
            q[..., k] = np.ravel(v)[:, None]
        q[..., 4] = te_ms[None]
        return q

    def log_factor(self, chi_pos, chi_neg, te_ms, conv):
        """log of the intra-voxel factor, (n_vox, n_te): log|F| + i·(phase unwrapped along TE),
        interpolated multilinearly in the log-signal."""
        return self._interp(conv, True)(self._q(chi_pos, chi_neg, te_ms))

    def factor(self, chi_pos, chi_neg, te_ms, conv, interp="log"):
        """Complex intra-voxel factor (n_vox, n_te) for bulk χ+ >= 0, χ- <= 0. ``interp``:
        "log" (log-signal, the default) or "complex" (the complex signal)."""
        q = self._q(chi_pos, chi_neg, te_ms)
        if interp == "log":
            return np.exp(self._interp(conv, True)(q))
        if interp == "complex":
            return self._interp(conv, False)(q)
        raise ValueError(f"interp must be 'log' or 'complex', got {interp!r}")

    def unwrapped_phase(self, chi_pos, chi_neg, te_ms, conv, interp="log"):
        """Phase of the intra-voxel factor (n_vox, n_te), unwrapped along TE from 0. With
        ``interp="complex"`` the factor is evaluated on the LUT's TE grid up to the last echo
        and unwrapped there."""
        te_ms = np.atleast_1d(np.asarray(te_ms, float))
        if interp == "log":
            return self.log_factor(chi_pos, chi_neg, te_ms, conv).imag
        grid = np.union1d(self.te[self.te < te_ms.max()], te_ms)
        F = self.factor(chi_pos, chi_neg, grid, conv, interp="complex")
        return np.unwrap(np.angle(F), axis=-1)[:, np.searchsorted(grid, te_ms)]


def voxelwise_chi_echoes(lut: SphereLUT, chi_pos, chi_neg, te_ms, conv, interp="log"):
    """The voxel-by-voxel per-echo χ(TE) = χ_bulk + φ_res(TE)/((2/3)γB0·TE), φ_res the intra-
    voxel phase unwrapped along TE. Kept for comparison only: it applies DECOMPOSE's own (2/3)
    conversion to its input, and no reconstruction produces it. :func:`star_chimap_echoes` is
    the shipped definition."""
    te_ms = np.atleast_1d(np.asarray(te_ms, float))
    ph = lut.unwrapped_phase(chi_pos, chi_neg, te_ms, conv, interp)
    bulk = (np.maximum(chi_pos, 0) + np.minimum(chi_neg, 0)).ravel()[:, None]
    return bulk + ph / ((2.0 / 3.0) * GAMMA * lut.b0 * 1e-6 * te_ms[None])


class SphereSignal:
    """Per-voxel GRE factor of the ``spheres`` model: F(TE)·exp(-TE·R2) in brain voxels, F from
    the sphere LUT (static dephasing; a common sub-voxel R2 factors out). F is interpolated in the
    complex signal by default (``interp="complex"``): where the para- and diamagnetic pools beat
    deeply (|F| near 0, e.g. the thalamus with the ``decompose`` convention) neighbouring nodes'
    TE-unwrapped phases can sit on different 2π branches, and the log-signal interpolation then
    mixes branches; a linear blend of complex signals is the signal of a mixture of the node
    configurations and stays continuous. Static spheres are
    fully refocused by a spin echo, so the SE keeps the caller's exp(-TE·R2) (``se`` returns
    None)."""

    def __init__(self, lut: SphereLUT, chi_pos, chi_neg, seg, B0, convention, interp="complex"):
        if not np.isclose(B0, lut.b0):
            raise ValueError(f"the sphere LUT is built at {lut.b0} T, not {B0} T")
        if convention not in SPHERE_CONVENTIONS:
            raise ValueError(f"convention must be one of {SPHERE_CONVENTIONS}, got {convention!r}")
        self.lut, self.convention, self.interp = lut, convention, interp
        self.mask = np.isin(np.rint(np.asarray(seg)).astype(int), BRAIN_LABELS)
        self._cp = np.asarray(chi_pos, float)[self.mask]
        self._cn = np.asarray(chi_neg, float)[self.mask]

    def intra_factor(self, TE_s):
        return self.lut.factor(self._cp, self._cn, [TE_s * 1e3], self.convention, self.interp)[:, 0]

    def gre(self, TE_s, R2):
        r2 = np.broadcast_to(np.asarray(R2, float), self.mask.shape)[self.mask]
        return self.mask, self.intra_factor(TE_s) * np.exp(-TE_s * r2)

    def se(self, TE_s, R2):
        return None


# ----------------------------------------------------------------------------------------
# per-echo χ(TE) by STAR-QSM (chimap-echoes)
# ----------------------------------------------------------------------------------------
QSMRS_ENV = "QSM_FORWARD_QSMRS_BIN"


class QsmrsTools:
    """QSM.rs command-line examples used for the per-echo reconstruction: ``laplacian_unwrap_file``
    (``--solver fft --pad 64`` = STI Suite 3.0's MRPhaseUnwrap) and ``star_file`` (STI Suite
    3.0's QSM_star, exact port; default padding [8, 8, 20] as STI's GUI passes it). Built with
    ``cargo build --release --features parallel --example star_file --example
    laplacian_unwrap_file``; ``bin_dir`` (or ``$QSM_FORWARD_QSMRS_BIN``) holds the binaries."""

    def __init__(self, bin_dir: Optional[str] = None):
        bin_dir = bin_dir or os.environ.get(QSMRS_ENV)
        if not bin_dir:
            raise FileNotFoundError(f"set ${QSMRS_ENV} to the directory of QSM.rs's star_file and "
                                    "laplacian_unwrap_file example binaries")
        self.star_bin = os.path.join(bin_dir, "star_file")
        self.unwrap_bin = os.path.join(bin_dir, "laplacian_unwrap_file")
        for b in (self.star_bin, self.unwrap_bin):
            if not os.access(b, os.X_OK):
                raise FileNotFoundError(b)

    @staticmethod
    def _save(path, arr, voxel):
        import nibabel as nib
        aff = np.diag(list(np.asarray(voxel, float)) + [1.0])
        nib.save(nib.Nifti1Image(np.asarray(arr, np.float64), aff), path)

    def unwrap(self, phases, mask, voxel):
        """Laplacian-unwrapped phases (list of 3D arrays)."""
        import nibabel as nib
        with tempfile.TemporaryDirectory() as d:
            files = []
            for i, p in enumerate(phases):
                files.append(os.path.join(d, f"ph{i}.nii"))
                self._save(files[-1], p, voxel)
            self._save(os.path.join(d, "mask.nii"), mask.astype(np.float64), voxel)
            subprocess.run([self.unwrap_bin, "--out", os.path.join(d, "u"), "--solver", "fft", "--pad", "64",
                            "--mask", os.path.join(d, "mask.nii"), *files],
                           check=True, capture_output=True)
            return [nib.load(os.path.join(d, f"u_echo-{i + 1}_unwrapped.nii.gz")).get_fdata()
                    for i in range(len(phases))]

    def star(self, phase, mask, te_ms, b0, voxel):
        """STAR-QSM χ (ppm) of a tissue phase (rad at echo time ``te_ms``), B0 along the third
        voxel axis."""
        import nibabel as nib
        with tempfile.TemporaryDirectory() as d:
            self._save(os.path.join(d, "ph.nii"), phase, voxel)
            self._save(os.path.join(d, "mask.nii"), mask.astype(np.float64), voxel)
            subprocess.run([self.star_bin, os.path.join(d, "ph.nii"), os.path.join(d, "mask.nii"),
                            os.path.join(d, "chi.nii.gz"), repr(float(te_ms)), repr(float(b0)), "0", "0", "1"],
                           check=True, capture_output=True)
            return nib.load(os.path.join(d, "chi.nii.gz")).get_fdata()


def echo_local_phases(local_field_ppm, te_ms, b0, intra_phase=None):
    """Per-echo local phase (rad), unwrapped along TE: 2πγ̄B0·TE·field + φ_intra(TE), with
    ``intra_phase`` of shape (n_te,) + field shape (the intra-voxel phase, unwrapped along TE)
    or None. γ̄ = 42.58 MHz/T, as qsm-forward's GRE signal uses."""
    out = []
    for i, te in enumerate(np.atleast_1d(te_ms)):
        ph = 2 * np.pi * 42.58 * b0 * te * 1e-3 * np.asarray(local_field_ppm, float)
        if intra_phase is not None:
            ph = ph + intra_phase[i]
        out.append(ph)
    return out


def star_chimap_echoes(chi_gt, local_phases, mask, te_ms, b0, voxel, tools,
                       unwrap="laplacian", return_parts=False):
    """chimap-echoes = χ_GT + [STAR(echo i) - STAR(echo-combined)], in ``mask``.

    Each echo's local phase (as acquired: wrapped, then Laplacian-unwrapped with STI's
    MRPhaseUnwrap settings, after Chen et al. 2021 §3.6; or, with ``unwrap="exact"``, the true
    phase unwrapped along TE) has its mask mean removed (the constant a background-field removal
    takes out) and goes through STAR-QSM at its own TE. The combined reconstruction is STAR on the
    sum of the echoes' unwrapped phases with TE = ΣTE, as STI Suite's GUI combines echoes. The
    bulk part stays the ground truth; only the echo dependence comes from the reconstruction.
    ``tools`` provides ``unwrap(phases, mask, voxel)`` and ``star(phase, mask, te_ms, b0, voxel)``
    (:class:`QsmrsTools`)."""
    te_ms = np.atleast_1d(np.asarray(te_ms, float))
    mask = np.asarray(mask, bool)
    if unwrap == "laplacian":
        wrapped = [np.angle(np.exp(1j * p)) * mask for p in local_phases]
        phases = tools.unwrap(wrapped, mask, voxel)
    elif unwrap == "exact":
        phases = [np.asarray(p, float) for p in local_phases]
    else:
        raise ValueError(f"unwrap must be 'laplacian' or 'exact', got {unwrap!r}")
    phases = [np.where(mask, p - p[mask].mean(), 0.0) for p in phases]
    star_e = np.stack([tools.star(p, mask, te, b0, voxel) for p, te in zip(phases, te_ms)])
    star_c = tools.star(np.sum(phases, axis=0), mask, te_ms.sum(), b0, voxel)
    out = np.where(mask[None], np.asarray(chi_gt, float)[None] + star_e - star_c[None], 0.0)
    if return_parts:
        return out, star_e, star_c
    return out
