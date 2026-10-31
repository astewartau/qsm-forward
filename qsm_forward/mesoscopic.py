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

LUT files (not in the repository; 6-28 MB each) are found through ``lut_path``: an explicit
path, else ``$QSM_FORWARD_MESO_LUT_DIR/<name>``, else a download from the Hugging Face dataset
``qsmxt/qsm-forward-mesoscopic-luts`` at a pinned revision into a per-user cache, checked
against its SHA-256 (``LUT_SHA256``). The dataset's README documents the tables.

References
    Wharton S., Bowtell R. PNAS 2012;109:18559. doi:10.1073/pnas.1211075109
    Chen J. et al. NeuroImage 2021;242:118477. doi:10.1016/j.neuroimage.2021.118477
    Shin H.-G. et al. NeuroImage 2021;240:118371. doi:10.1016/j.neuroimage.2021.118371
    Yablonskiy D.A., Haacke E.M. MRM 1994;32:749. doi:10.1002/mrm.1910320610
    Wei H. et al. NMR Biomed 2015;28:1294 (STAR-QSM). doi:10.1002/nbm.3383
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
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
    "spheres": "spheres_a3_v2.npz",
}


LUT_SHA256 = {
    "meso_diffusion_v3.npz": "2d7508c547810ddf13e527f9d199d31b37f07e0b751a5a6bd6d8e5e9ccdc60b8",
    "meso_static_v3.npz": "a66a336e7d298ea2ea27d7c7677e9e70ff9a1bc9e2379834d5dfd04a946ea5a1",
    "spheres_a3_v2.npz": "38ec5f8fdd734c80bea7c1b662e45e7a31dab6eba69ec0bd2c16ed1d49546b6b",
}
# Pinned so a phantom generated today is regenerated from the same tables; bump it together
# with LUT_FILES / LUT_SHA256 when a table changes.
LUT_REPO = "qsmxt/qsm-forward-mesoscopic-luts"
LUT_REVISION = "04541e4559943343065fc697e3b8315cc8733d2a"


def _lut_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "qsm-forward", "meso-luts", LUT_REVISION[:12])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: str) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _download_lut(name: str) -> str:
    """The cached copy of LUT ``name``, downloaded and SHA-256-checked on first use."""
    cache = _lut_cache_dir()
    p = os.path.join(cache, name)
    if os.path.exists(p) and _sha256(p) == LUT_SHA256[name]:
        return p
    os.makedirs(cache, exist_ok=True)
    url = f"https://huggingface.co/datasets/{LUT_REPO}/resolve/{LUT_REVISION}/{name}"
    fd, tmp = tempfile.mkstemp(dir=cache, suffix=".part")
    os.close(fd)
    try:
        _fetch(url, tmp)
        got = _sha256(tmp)
        if got != LUT_SHA256[name]:
            raise OSError(f"{name} from {url} has SHA-256 {got}, expected {LUT_SHA256[name]}")
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return p


def lut_path(kind: str, path: Optional[str] = None) -> str:
    """Path of LUT ``kind`` ("diffusion", "static" or "spheres"): ``path`` if given, else
    ``$QSM_FORWARD_MESO_LUT_DIR/<LUT_FILES[kind]>``, else the pinned Hugging Face copy, downloaded
    into the cache on first use."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    if kind not in LUT_FILES:
        raise KeyError(f"unknown LUT {kind!r}; choose from {sorted(LUT_FILES)}")
    root = os.environ.get(LUT_ENV)
    if root:
        p = os.path.join(root, LUT_FILES[kind])
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    return _download_lut(LUT_FILES[kind])


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


def free_water_r2(B0):
    """The head model's CSF R2 (1/s) at ``B0``: 1/T2 of label 10 in
    ``qsm_forward.T2_TISSUE_PARAMS_7T`` (1029.6 ms at 7 T), scaled by 0.65 at 3 T as
    :func:`qsm_forward.qsm_forward.generate_t2_map` does: 0.97 at 7 T, 0.63 at 3 T. The floor of
    the intrinsic water R2 of the mesoscopic model."""
    from .qsm_forward import T2_TISSUE_PARAMS_7T
    r2 = 1000.0 / T2_TISSUE_PARAMS_7T[10]
    return r2 * 0.65 if float(B0) == 3.0 else r2


class MesoscopicSignal:
    """Per-voxel complex GRE and SE factors of the ``mesoscopic`` model on a simulation grid.

    ``theta_deg`` is the fibre-to-B0 angle (degrees, NaN where there is no fibre direction).
    Voxels with a label in ``BRAIN_LABELS`` get the LUT; white matter (``WM_LABEL``) with a
    finite fibre angle uses it, every other brain voxel the orientation average. ``gre`` and
    ``se`` return ``(mask, values)``: the factor replaces the real decay envelope in those
    voxels (the macroscopic phase and steady-state weighting go on top), and the caller keeps
    its own model elsewhere."""

    def __init__(self, lut: MesoLUT, chi_pos, chi_neg, theta_deg, seg, B0,
                 se_tes_ms=PROTOCOL_3T["se_ms"], r2_floor=None, r2_iterations=4, r2_tol=0.01):
        self.lut = lut
        self.B0 = float(B0)
        self.se_tes_ms = np.asarray(se_tes_ms, float)
        self.r2_floor = free_water_r2(self.B0) if r2_floor is None else float(r2_floor)
        self.r2_iterations, self.r2_tol = int(r2_iterations), float(r2_tol)
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
        simulated spin echo reproduces the tissue R2 map.

        The tissue map is the head model's spin-echo R2: it already contains the irreversible loss
        the LUT adds on its own (iron in the motional-narrowing regime, the fast myelin-water pool).
        Giving the water the full tissue R2 would count that loss twice, so the water's (non-iron)
        R2 is the tissue R2 minus the LUT's excess: R2_0 <- R2_0 - [R2_SE(R2_0) - R2_tissue], with
        R2_SE the mono-exponential rate of the simulated SE over ``se_tes_ms``, iterated from
        R2_0 = R2_tissue (``r2_iterations`` steps at most, until the largest residual among the
        unfloored voxels is below ``r2_tol``).

        R2_0 is floored at ``r2_floor``: by default the head model's CSF R2 at this field
        (:func:`free_water_r2`), as no tissue water relaxes more slowly than free water. Where the
        floor binds, the LUT's own SE loss alone exceeds the tissue map (the iron nuclei: the
        tabulated iron R2 per ppm is larger than the head model leaves room for), and the simulated
        SE R2 stays above the map there. ``floored`` (brain voxels) and ``se_r2_residual`` (simulated
        SE R2 - tissue R2, 1/s) record where and by how much; ``floored_fraction`` is their share."""
        key = id(R2)
        if self._r2_key == key:
            return self._r2_0
        r2 = np.broadcast_to(np.asarray(R2, float), self.mask.shape)[self.mask].astype(float)
        r2_0 = np.maximum(r2, self.r2_floor)
        for _ in range(max(self.r2_iterations, 1)):
            resid = self._se_rate(r2_0) - r2
            free = r2_0 - resid >= self.r2_floor
            r2_0 = np.maximum(r2_0 - resid, self.r2_floor)
            if not free.any() or np.abs(resid[free]).max() < self.r2_tol:
                break
        self.se_r2_residual = self._se_rate(r2_0) - r2
        self.floored = r2_0 <= self.r2_floor
        self.floored_fraction = float(np.mean(self.floored)) if r2_0.size else 0.0
        self._r2_0 = r2_0
        self._r2_key = key
        return self._r2_0

    def _se_rate(self, r2_0):
        se = np.array([np.abs(self._eval(t, None, se=True, r2_0=r2_0)) for t in self.se_tes_ms])
        return fit_rate(self.se_tes_ms, se)

    def floor_report(self, seg):
        """Where the intrinsic-R2 floor binds, per brain label: voxel count, fraction, and the median
        simulated SE R2 minus the tissue R2 map there (1/s). Call after the signal was simulated."""
        lab = np.rint(np.asarray(seg)).astype(int)[self.mask]
        free = np.abs(self.se_r2_residual[~self.floored])
        out = {"FloorPerSecond": self.r2_floor, "FlooredFraction": self.floored_fraction,
               # unfloored voxels whose SE R2 still misses the map by > 0.1 1/s: R2 maps so high
               # (> ~100 1/s, head-model outliers) that the SE over se_tes_ms has decayed away
               "UnflooredResidualAbove0p1": int(np.sum(free > 0.1)),
               "UnflooredResidualP99": float(np.percentile(free, 99)) if free.size else 0.0,
               "ByLabel": {}}
        for k in BRAIN_LABELS:
            m = lab == k
            if not m.any():
                continue
            f = m & self.floored
            out["ByLabel"][str(k)] = {
                "n": int(m.sum()), "floored": int(f.sum()), "fraction": float(f.sum() / m.sum()),
                "median_se_r2_excess_floored": float(np.median(self.se_r2_residual[f])) if f.any() else 0.0}
        return out

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
SOURCE_RANGES = ("in-range", "highchi")
# source_params ("highchi"): ζ = ZETA_REF while χ_s rises 0.01 -> 0.15 ppm (Chen 2021's range), then
# ζ rises to ZETA_MAX, then χ_s rises above the paper's range.
ZETA_REF, ZETA_MAX, CHI_S_MIN, CHI_S_PAPER = 0.25, 0.35, 0.01, 0.15
# "in-range": χ_s stays at or below a cap and the bulk χ is reached by the volume fraction, up to a
# total source fraction ζ+ + ζ- of ZETA_TOTAL_MAX (the water medium stays the larger part of the voxel;
# beyond about 0.6 the "spheres in water" picture no longer holds: same-sign spheres merge).
ZETA_TOTAL_MAX = 0.60
# source-range flags (``*_desc-sourcerange_mask``)
SRC_IN_RANGE, SRC_CHI_S_ABOVE_CAP, SRC_CHI_S_CLIPPED = 0, 1, 2


def source_params(chi_bulk):
    """The ``highchi`` mapping, one sign: bulk susceptibility -> (ζ, χ_s >= 0) with ζ·χ_s = |bulk|.

    |bulk| <= 0.0375 ppm: ζ = 0.25 (below 0.0025 ppm: χ_s = 0.01, ζ < 0.25) and χ_s 0.01-0.15;
    up to 0.0525 ppm: χ_s = 0.15, ζ 0.25-0.35; above: ζ = 0.35, χ_s = |bulk|/0.35 (outside
    the paper's χ_s range: up to 0.40 ppm in the pallidus)."""
    b = np.abs(np.asarray(chi_bulk, float))
    chi_s = np.clip(b / ZETA_REF, CHI_S_MIN, CHI_S_PAPER)
    chi_s = np.where(b > ZETA_MAX * CHI_S_PAPER, b / ZETA_MAX, chi_s)
    zeta = np.minimum(b / chi_s, ZETA_MAX)
    return zeta, chi_s


def source_params_inrange(chi_pos, chi_neg, chi_s_max, zeta_total_max=ZETA_TOTAL_MAX):
    """The ``in-range`` mapping, both signs jointly: bulk χ+ >= 0, χ- <= 0 -> (ζ+, ζ-, χ_s+, χ_s-,
    capped) with ζ±·χ_s± = |χ±| exactly.

    * Each sign prefers ζ = ZETA_REF (0.25) with χ_s = |χ|/0.25 in [0.01, ``chi_s_max``], as the
      ``highchi`` mapping does at low bulk.
    * A sign whose bulk needs more than that keeps χ_s = ``chi_s_max`` and raises ζ = |χ|/χ_s_max.
    * If ζ+ + ζ- exceeds ``zeta_total_max``, the preferred-fraction surplus (above the minimum
      |χ|/χ_s_max) is given up first, which raises that sign's χ_s toward the cap.
    * Only when even the minimum fractions sum above ``zeta_total_max`` are both scaled down to
      sum to it; χ_s then exceeds ``chi_s_max`` there and ``capped`` is True. The anatomy (bulk
      χ±) is never changed."""
    bp = np.abs(np.asarray(chi_pos, float))
    bm = np.abs(np.asarray(chi_neg, float))
    need_p, need_m = bp / chi_s_max, bm / chi_s_max
    pref_p = np.maximum(need_p, np.minimum(ZETA_REF, bp / CHI_S_MIN))
    pref_m = np.maximum(need_m, np.minimum(ZETA_REF, bm / CHI_S_MIN))
    need, pref = need_p + need_m, pref_p + pref_m
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.clip(np.where(pref > need, (zeta_total_max - need) / (pref - need), 1.0), 0.0, 1.0)
        zp = need_p + (pref_p - need_p) * f
        zm = need_m + (pref_m - need_m) * f
        capped = need > zeta_total_max
        g = np.where(capped, zeta_total_max / need, 1.0)
        zp, zm = np.where(capped, need_p * g, zp), np.where(capped, need_m * g, zm)
        zp, zm = np.minimum(zp, zeta_total_max), np.minimum(zm, zeta_total_max)   # rounding only
        sp = np.where(zp > 0, bp / zp, CHI_S_MIN)
        sm = np.where(zm > 0, bm / zm, CHI_S_MIN)
    return zp, zm, sp, sm, capped


class SourceMapping:
    """Bulk (χ+, χ-) -> sphere parameters (ζ+, ζ-, χ_s+, χ_s-) and a per-voxel source-range flag.

    ``kind="in-range"`` (:func:`source_params_inrange`): χ_s <= ``chi_s_max`` (by default the source
    phase limit π/((2/3)γB0·TE_max) of the protocol, inside Chen et al. 2021's simulated 0.01-0.15
    ppm) wherever ζ+ + ζ- <= ``zeta_total_max`` suffices. ``kind="highchi"``
    (:func:`source_params`): ζ <= 0.35 per sign, χ_s up to 0.40 ppm in deep grey matter (outside the
    authors' validated range; robustness only).

    Flags: 0 in range, 1 χ_s above the cap (in-range: the fraction cap binds; highchi: χ_s > 0.15),
    2 χ_s clipped at the LUT's last node (the signal then under-represents the bulk χ)."""

    def __init__(self, kind="in-range", chi_s_max=None, zeta_total_max=ZETA_TOTAL_MAX):
        if kind not in SOURCE_RANGES:
            raise ValueError(f"source range must be one of {SOURCE_RANGES}, got {kind!r}")
        if kind == "in-range" and not (chi_s_max and chi_s_max > 0):
            raise ValueError("the in-range mapping needs a positive chi_s_max (ppm)")
        self.kind, self.chi_s_max, self.zeta_total_max = kind, chi_s_max, zeta_total_max

    def __call__(self, chi_pos, chi_neg, chi_s_lut_max=np.inf):
        cp, cn = np.maximum(chi_pos, 0), np.minimum(chi_neg, 0)
        if self.kind == "in-range":
            zp, zm, sp, sm, capped = source_params_inrange(cp, cn, self.chi_s_max, self.zeta_total_max)
            above = capped
        else:
            (zp, sp), (zm, sm) = source_params(cp), source_params(cn)
            above = (sp > CHI_S_PAPER + 1e-9) | (sm > CHI_S_PAPER + 1e-9)
        flag = np.where(above, SRC_CHI_S_ABOVE_CAP, SRC_IN_RANGE)
        flag = np.where((sp > chi_s_lut_max + 1e-9) | (sm > chi_s_lut_max + 1e-9), SRC_CHI_S_CLIPPED, flag)
        return zp, zm, sp, sm, flag.astype(np.uint8)

    def describe(self):
        if self.kind == "in-range":
            return {"SourceRange": "in-range", "ChiSMaxPpm": float(self.chi_s_max),
                    "ZetaTotalMax": float(self.zeta_total_max), "ZetaRef": ZETA_REF, "ChiSMinPpm": CHI_S_MIN}
        return {"SourceRange": "highchi", "ZetaRef": ZETA_REF, "ZetaMax": ZETA_MAX, "ChiSPaperPpm": CHI_S_PAPER,
                "Note": "χ_s exceeds Chen et al. 2021's simulated 0.01-0.15 ppm in deep grey matter "
                        "(outside the authors' validated range; a robustness phantom)"}


HIGHCHI = SourceMapping("highchi")


def wrap_limit_ppm(te_max_ms, b0=3.0):
    """|χ| above which DECOMPOSE's synthesised phase (2/3)γB0·χ·TE leaves (-π, π] at TE_max. Also the
    source-phase limit: intra-source water ((2/3)χ_s γB0 in the ``decompose`` convention) beats against
    the medium within the echo train once χ_s exceeds it."""
    return np.pi / ((2.0 / 3.0) * GAMMA * b0 * 1e-6 * te_max_ms)


class SphereLUT:
    """Reader of the sphere LUT: ``S_<convention>[ζ+, ζ-, χ_s+, χ_s-, TE]`` (complex), axes
    ``zeta``, ``chi_s``, ``te`` (ms) and ``b0``. v2 tables (``spheres_a3_v2.npz``) run ζ to 0.60 and
    hold node pairs with ζ+ + ζ- <= ``zeta_pair_max`` (0.70; ``built``); their v1 nodes (ζ <= 0.35)
    are identical to v1's. Every method takes a ``mapping`` (:class:`SourceMapping`; default the
    ``highchi`` one, the v1 behaviour)."""

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.zeta = np.asarray(d["zeta"], float)
        self.chi_s = np.asarray(d["chi_s"], float)
        self.te = np.asarray(d["te"], float)
        self.b0 = float(d["b0"])
        self._S = {c: np.asarray(d[f"S_{c}"]) for c in SPHERE_CONVENTIONS if f"S_{c}" in d.files}
        self.zeta_pair_max = float(d["zeta_pair_max"]) if "zeta_pair_max" in d.files else 2 * self.zeta[-1]
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

    def check_mapping(self, mapping):
        """The table must cover the mapping's fractions: every interpolation cell's corners built."""
        zmax = mapping.zeta_total_max if mapping.kind == "in-range" else ZETA_MAX
        step = float(np.max(np.diff(self.zeta)))
        if zmax > self.zeta[-1] + 1e-9 or zmax + 2 * step > self.zeta_pair_max + 1e-9:
            raise ValueError(f"{os.path.basename(self.path)} covers ζ <= {self.zeta[-1]:.2f} (pairs <= "
                             f"{self.zeta_pair_max:.2f}); the {mapping.kind} mapping needs ζ+ + ζ- up to "
                             f"{zmax:.2f} (use spheres_a3_v2.npz)")

    def params(self, chi_pos, chi_neg, mapping=None):
        """(ζ+, ζ-, χ_s+, χ_s-, flag) of ``mapping`` for bulk χ+ >= 0, χ- <= 0."""
        return (mapping or HIGHCHI)(chi_pos, chi_neg, self.chi_s[-1])

    def _q(self, chi_pos, chi_neg, te_ms, mapping=None):
        mapping = mapping or HIGHCHI
        self.check_mapping(mapping)
        zp, zm, sp, sm, _ = self.params(chi_pos, chi_neg, mapping)
        zp = np.clip(zp, self.zeta[0], self.zeta[-1])      # rounding only (the mapping caps ζ)
        zm = np.clip(zm, self.zeta[0], self.zeta[-1])
        sp = np.clip(sp, self.chi_s[0], self.chi_s[-1])
        sm = np.clip(sm, self.chi_s[0], self.chi_s[-1])
        te_ms = np.clip(np.atleast_1d(np.asarray(te_ms, float)), self.te[0], self.te[-1])
        q = np.empty((np.size(zp), len(te_ms), 5))
        for k, v in enumerate((zp, zm, sp, sm)):
            q[..., k] = np.ravel(v)[:, None]
        q[..., 4] = te_ms[None]
        return q

    def log_factor(self, chi_pos, chi_neg, te_ms, conv, mapping=None):
        """log of the intra-voxel factor, (n_vox, n_te): log|F| + i·(phase unwrapped along TE),
        interpolated multilinearly in the log-signal."""
        return self._interp(conv, True)(self._q(chi_pos, chi_neg, te_ms, mapping))

    def factor(self, chi_pos, chi_neg, te_ms, conv, interp="log", mapping=None):
        """Complex intra-voxel factor (n_vox, n_te) for bulk χ+ >= 0, χ- <= 0. ``interp``:
        "log" (log-signal, the default) or "complex" (the complex signal)."""
        q = self._q(chi_pos, chi_neg, te_ms, mapping)
        if interp == "log":
            return np.exp(self._interp(conv, True)(q))
        if interp == "complex":
            return self._interp(conv, False)(q)
        raise ValueError(f"interp must be 'log' or 'complex', got {interp!r}")

    def unwrapped_phase(self, chi_pos, chi_neg, te_ms, conv, interp="log", mapping=None):
        """Phase of the intra-voxel factor (n_vox, n_te), unwrapped along TE from 0. With
        ``interp="complex"`` the factor is evaluated on the LUT's TE grid up to the last echo
        and unwrapped there."""
        te_ms = np.atleast_1d(np.asarray(te_ms, float))
        if interp == "log":
            return self.log_factor(chi_pos, chi_neg, te_ms, conv, mapping).imag
        grid = np.union1d(self.te[self.te < te_ms.max()], te_ms)
        F = self.factor(chi_pos, chi_neg, grid, conv, interp="complex", mapping=mapping)
        return np.unwrap(np.angle(F), axis=-1)[:, np.searchsorted(grid, te_ms)]


def voxelwise_chi_echoes(lut: SphereLUT, chi_pos, chi_neg, te_ms, conv, interp="log", mapping=None):
    """The voxel-by-voxel per-echo χ(TE) = χ_bulk + φ_res(TE)/((2/3)γB0·TE), φ_res the intra-
    voxel phase unwrapped along TE. Kept for comparison only: it applies DECOMPOSE's own (2/3)
    conversion to its input, and no reconstruction produces it. :func:`star_chimap_echoes` is
    the shipped definition."""
    te_ms = np.atleast_1d(np.asarray(te_ms, float))
    ph = lut.unwrapped_phase(chi_pos, chi_neg, te_ms, conv, interp, mapping)
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
    None). ``mapping`` (:class:`SourceMapping`, default ``highchi``) turns bulk χ± into sphere
    parameters; ``source_flag`` holds its per-voxel flag on the brain voxels."""

    def __init__(self, lut: SphereLUT, chi_pos, chi_neg, seg, B0, convention, interp="complex", mapping=None):
        if not np.isclose(B0, lut.b0):
            raise ValueError(f"the sphere LUT is built at {lut.b0} T, not {B0} T")
        if convention not in SPHERE_CONVENTIONS:
            raise ValueError(f"convention must be one of {SPHERE_CONVENTIONS}, got {convention!r}")
        self.lut, self.convention, self.interp = lut, convention, interp
        self.mapping = mapping or HIGHCHI
        lut.check_mapping(self.mapping)
        self.mask = np.isin(np.rint(np.asarray(seg)).astype(int), BRAIN_LABELS)
        self.seg = np.rint(np.asarray(seg)).astype(int)[self.mask]
        self._cp = np.asarray(chi_pos, float)[self.mask]
        self._cn = np.asarray(chi_neg, float)[self.mask]
        self.zeta_pos, self.zeta_neg, self.chi_s_pos, self.chi_s_neg, self.source_flag = \
            lut.params(self._cp, self._cn, self.mapping)

    def intra_factor(self, TE_s):
        return self.lut.factor(self._cp, self._cn, [TE_s * 1e3], self.convention, self.interp,
                               mapping=self.mapping)[:, 0]

    def source_report(self, te_max_ms):
        """What the mapping did, for the sidecar: the flagged share per brain label, the fractions
        used, and the in-range check, the share of voxels whose source phase (2/3)γB0·χ_s·TE_max
        exceeds π (χ_s above :func:`wrap_limit_ppm`), per sign."""
        lim = float(wrap_limit_ppm(te_max_ms, self.lut.b0))
        cap = self.mapping.chi_s_max or CHI_S_PAPER
        need = (np.abs(self._cp) + np.abs(np.minimum(self._cn, 0))) / cap
        zt = self.zeta_pos + self.zeta_neg

        def stats(sel):
            n = int(sel.sum())
            if not n:
                return {"n": 0}
            sp, sm = self.chi_s_pos[sel] * (self.zeta_pos[sel] > 0), self.chi_s_neg[sel] * (self.zeta_neg[sel] > 0)
            return {"n": n,
                    "flag_above_cap": float(np.mean(self.source_flag[sel] == SRC_CHI_S_ABOVE_CAP)),
                    "flag_clipped": float(np.mean(self.source_flag[sel] == SRC_CHI_S_CLIPPED)),
                    "chi_s_pos_over_phase_limit": float(np.mean(sp > lim + 1e-12)),
                    "chi_s_neg_over_phase_limit": float(np.mean(sm > lim + 1e-12)),
                    "either_over_phase_limit": float(np.mean((sp > lim + 1e-12) | (sm > lim + 1e-12))),
                    "zeta_needed_at_cap_median": float(np.median(need[sel])),
                    "zeta_needed_at_cap_max": float(need[sel].max()),
                    "zeta_total_median": float(np.median(zt[sel])), "zeta_total_max": float(zt[sel].max()),
                    "chi_s_pos_max": float(sp.max()), "chi_s_neg_max": float(sm.max())}
        out = {**self.mapping.describe(), "PhaseLimitPpm": lim, "Flags": "0 in range, 1 χ_s above the cap "
               "(fraction cap binds / highchi χ_s > 0.15), 2 χ_s clipped at the LUT's last node",
               "Brain": stats(np.ones(self.seg.shape, bool)), "ByLabel": {}}
        for lab in BRAIN_LABELS:
            sel = self.seg == lab
            if sel.any():
                out["ByLabel"][str(lab)] = stats(sel)
        return out

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


def star_chimap_echoes(chi_gt, local_phases, ref_phases, mask, te_ms, b0, voxel, tools,
                       unwrap="laplacian", return_parts=False):
    """chimap-echoes = χ_GT + [STAR(echo i) - STAR_ref(echo i)], in ``mask``.

    ``local_phases`` are the echoes' local phases with the intra-voxel phase, ``ref_phases`` the
    reference: the same echoes with a perfectly linear phase (the ground-truth macroscopic local
    field × TE, :func:`echo_local_phases` without ``intra_phase``). STAR-QSM reads only the phase,
    so the reference shares the magnitude by construction. Both go through the identical
    processing, echo by echo: wrapped, then Laplacian-unwrapped with STI's MRPhaseUnwrap settings
    (after Chen et al. 2021 §3.6; or, with ``unwrap="exact"``, used as given, unwrapped along TE),
    mask mean removed (the constant a background-field removal takes out), and STAR-QSM at the
    echo's TE. The difference keeps only the echo dependence the tissue's intra-voxel phase causes
    as seen through that processing: STAR's own echo dependence (it returns χ 20-40 % high at
    2.7 ms on a linear phase, falling to the ground truth by about 15-20 ms) is in both terms and
    cancels, and with no intra-voxel phase the output is the ground truth at every echo.
    ``tools`` provides ``unwrap(phases, mask, voxel)`` and ``star(phase, mask, te_ms, b0, voxel)``
    (:class:`QsmrsTools`). ``return_parts`` also returns STAR(echo i) and STAR_ref(echo i)."""
    te_ms = np.atleast_1d(np.asarray(te_ms, float))
    mask = np.asarray(mask, bool)

    def process(local):
        if unwrap == "laplacian":
            phases = tools.unwrap([np.angle(np.exp(1j * p)) * mask for p in local], mask, voxel)
        elif unwrap == "exact":
            phases = [np.asarray(p, float) for p in local]
        else:
            raise ValueError(f"unwrap must be 'laplacian' or 'exact', got {unwrap!r}")
        phases = [np.where(mask, p - p[mask].mean(), 0.0) for p in phases]
        return np.stack([tools.star(p, mask, te, b0, voxel) for p, te in zip(phases, te_ms)])

    star_e = process(local_phases)
    star_r = process(ref_phases)
    out = np.where(mask[None], np.asarray(chi_gt, float)[None] + star_e - star_r, 0.0)
    if return_parts:
        return out, star_e, star_r
    return out
