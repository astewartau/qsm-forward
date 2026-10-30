"""Susceptibility source-separation (chi-separation) phantoms on the QSM Challenge 2.0 head model.

Builds χ+ / χ- ground truth with the Ridani et al. (2026) construction [1], then simulates the
multi-echo GRE (and optional spin-echo) acquisition. Every phantom shares that χ+/χ- anatomy;
they differ only in how the sources relax, the signal model and the protocol. Named presets
(:data:`PRESETS`) hold the published Ridani configurations, the QSM-CI benchmark phantoms and
phantoms that each satisfy one family of methods' assumptions; :func:`generate_chisep_bids` runs
any preset or a custom :class:`ChisepConfig`.

Relaxation (``relaxation``), i.e. how χ+ and χ- set the reversible rate R2':
  ridani   region-disjoint, as in [1]: Dr+ acts outside white matter only and Dr-(θ) inside it
           only, R2' = Dr+|χ+| + Dr-|χ-|. The magnitude follows ``dr_model``:
             scaled  static-dephasing kernels, Dr+ = 2πγB0/(9√3) for spheres, Dr- = ½γB0·sin²θ for
                     cylinders (Dr+ = 107.84·B0, Dr- = 133.77·sin²θ·B0 Hz/ppm) [1, 4].
             fixed   Dr+ anchored at ``dr_fixed`` Hz/ppm; Dr- scaled by the same factor.
  linear   co-located and isotropic, the model of χ-separation [5] and the methods built on it:
           both sources relax in every voxel with one relaxivity ``dr``,
           R2' = dr·(|χ+| + |χ-|).

Irreversible R2 (``r2_model``):
  tissue   per-tissue literature T2 with R2*/M0 texture (T2_simulation.m of [1]).
  source   R2 = (r - dr)·(|χ+| + |χ-|) with r = ``r2star_relaxivity``, so that
           R2* = R2 + R2' = r·(|χ+| + |χ-|) holds exactly: the R2*-only separation model of
           Dimov et al. 2022 [6], which absorbs R2 into r. Requires ``relaxation="linear"``.
           R2 is then zero wherever the voxel has no susceptibility source (e.g. CSF), which is
           not physical; the phantom exists to satisfy that model, not to look like tissue.

Signal models (``signal_model``):
  mono             mono-exponential decay at R2* = R2 + R2', phase linear in TE [1].
  hollow-cylinder  white matter holds myelin, axonal and extra-axonal water pools (Wharton &
                   Bowtell 2012 [2]); elsewhere mono. Requires ``relaxation="ridani"``.
New signal models plug in through :data:`SIGNAL_MODELS` and :func:`_signal_kwargs`.

The shipped R2' (``r2prime_source``):
  formula  the R2' the generator imposes (the relaxation formula above, or the hollow-cylinder
           white-matter equivalent).
  fitted   what an R2* - R2 pipeline measures from the simulated data: mono-exponential
           (log-linear) fits of the noiseless GRE over ``tes_ms`` and of the noiseless spin echo
           over ``se_tes_ms``. The formula map is then written as ``*_desc-formula_R2prime.nii``.

Resolution: the maps are built at the native 0.64 mm (the χ+/χ- texture construction needs the
real R1/R2* maps), downsampled once in image space to ``voxel`` mm, and the field and signals are
then simulated at that resolution, so the inputs and the ground truth share one PSF. ``voxel=0``
keeps the native grid.

  [1] Ridani D., De Leener B., Alonso-Ortiz E. Magn Reson Med 2026. doi:10.1002/mrm.70468
  [2] Wharton S., Bowtell R. PNAS 2012;109(45):18559-18564. doi:10.1073/pnas.1211075109
  [3] Kim M. et al. Hum Brain Mapp 2025;46(4):e70136. doi:10.1002/hbm.70136 (χ-sepnet)
  [4] Yablonskiy D.A., Haacke E.M. Magn Reson Med 1994;32(6):749-763. doi:10.1002/mrm.1910320610
  [5] Shin H.G. et al. NeuroImage 2021;240:118371. doi:10.1016/j.neuroimage.2021.118371
  [6] Dimov A.V. et al. Tomography 2022;8(3):1544-1551. doi:10.3390/tomography8030127
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import nibabel as nib
import numpy as np

from .qsm_forward import (
    DR_KERNEL, ReconParams, TissueParams, generate_bids, generate_dr_maps_ridani,
    generate_r2prime, generate_theta_from_v1, resize,
)

SIGNAL_MODELS = ("mono", "hollow-cylinder")
RELAXATION_MODELS = ("ridani", "linear")
R2_MODELS = ("tissue", "source")
R2PRIME_SOURCES = ("formula", "fitted")

GAMMA_BAR = 42.58  # MHz/T, as in the Ridani reference code


def dr_toolbox(b0):
    """Relaxivity (Hz/ppm) the χ-separation toolbox assumes by default: 137 Hz/ppm measured at
    2.9 T [5], scaled linearly with field (141.7 at 3 T)."""
    return 137.0 * b0 / 2.9


def dr_static_dephasing(b0):
    """Static-dephasing relaxivity (Hz/ppm) of randomly placed spheres [4], 2πγB0/(9√3) with γ in
    rad/s/T (323.5 at 3 T); the kernel of DECOMPOSE-QSM, APART-QSM and SUSEP-Net."""
    return (2 * np.pi) ** 2 * GAMMA_BAR * b0 / (9 * np.sqrt(3))


DR_CHISEPNET = 114.0           # Hz/ppm at 3 T, fixed in χ-sepnet's input normalisation [3]
R2STAR_RELAXIVITY_3T = 274.0   # Hz/ppm at 3 T, the R2*-only model's constant [6]


@dataclass
class ChisepConfig:
    """One chi-sep phantom configuration. Times are in milliseconds, fields in tesla,
    relaxivities in Hz/ppm."""
    b0: float = 7.0
    tes_ms: tuple = (4, 12, 20, 28)
    se_tes_ms: tuple = (10, 30, 50, 70)
    tr: float = 0.05
    flip_angle: float = 15.0
    se_tr: float = 1.5
    voxel: float = 1.0             # mm; 0 keeps the native 0.64 mm grid
    anisotropic: bool = True       # WM χ- = Δχ·cos²θ + χ0 (and θ-dependent Dr- for "ridani")
    relaxation: str = "ridani"     # see RELAXATION_MODELS
    dr_model: str = "scaled"       # "ridani" only: "scaled" | "fixed"
    dr_fixed: float = DR_KERNEL    # "ridani" + "fixed": the Dr+ anchor
    dr: Optional[float] = None     # "linear": the one relaxivity of both sources
    r2_model: str = "tissue"       # see R2_MODELS
    r2star_relaxivity: Optional[float] = None  # "source": r in R2* = r·(|χ+| + |χ-|)
    signal_model: str = "mono"     # see SIGNAL_MODELS
    hc_b0: Optional[float] = None  # hollow-cylinder pools evaluated at this field (default b0)
    r2prime_source: str = "formula"  # see R2PRIME_SOURCES
    peak_snr: float = 100.0
    noiseless: bool = False
    seed: int = 42
    se: bool = True                # save the matched multi-echo spin echo
    r2: bool = False               # save the ground-truth R2 map
    v1_flip: bool = True           # register V1 (stored L-A-S) to the R-A-S maps
    description: str = ""

    def __post_init__(self):
        def check(name, allowed):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        check("signal_model", SIGNAL_MODELS)
        check("relaxation", RELAXATION_MODELS)
        check("r2_model", R2_MODELS)
        check("r2prime_source", R2PRIME_SOURCES)
        check("dr_model", ("scaled", "fixed"))
        if self.relaxation == "linear" and not (self.dr and self.dr > 0):
            raise ValueError("relaxation='linear' needs a positive dr (Hz/ppm)")
        if self.r2_model == "source":
            if self.relaxation != "linear":
                raise ValueError("r2_model='source' is defined for relaxation='linear'")
            if self.r2star_relaxivity is None or self.r2star_relaxivity < self.dr:
                raise ValueError("r2_model='source' needs r2star_relaxivity >= dr (R2 = (r - dr)·Σ|χ| >= 0)")
        if self.signal_model == "hollow-cylinder" and self.relaxation != "ridani":
            raise ValueError("signal_model='hollow-cylinder' is defined for relaxation='ridani'")


_RIDANI = dict(dr_model="scaled", noiseless=True, se=False, r2=True, voxel=0.0, tr=0.05, flip_angle=15.0)
_RIDANI_3T = dict(b0=3.0, tes_ms=(3, 7, 11, 15, 19, 23), **_RIDANI)
# Co-located linear phantoms: 3 T, 1 mm, six echoes 5-30 ms, matched spin echo, peak SNR 100, the
# shipped R2' fitted from the signals. Only the relaxivity (so R2' and the magnitude) differs.
_LINEAR_3T = dict(b0=3.0, tes_ms=(5, 10, 15, 20, 25, 30), se_tes_ms=(10, 30, 50, 70), voxel=1.0,
                  anisotropic=True, relaxation="linear", r2prime_source="fitted", peak_snr=100.0,
                  se=True, r2=True)

PRESETS = {
    # As published by Ridani et al. (2026): noiseless, GRE-only, native resolution.
    "ridani-3t-iso": ChisepConfig(anisotropic=False, **_RIDANI_3T,
                                  description="Ridani et al. 2026, 3 T, isotropic WM"),
    "ridani-3t-aniso": ChisepConfig(anisotropic=True, **_RIDANI_3T,
                                    description="Ridani et al. 2026, 3 T, anisotropic WM"),
    "ridani-7t-aniso": ChisepConfig(b0=7.0, tes_ms=(4, 12, 20, 28), anisotropic=True, **_RIDANI,
                                    description="Ridani et al. 2026, 7 T, anisotropic WM"),
    # 7 T, 8 echoes, hollow-cylinder white matter, matched spin echo, SNR 100 (QSM-CI chisep-mc).
    "chisep-mc": ChisepConfig(b0=7.0, tes_ms=(3, 9, 15, 21, 27, 33, 39, 45), dr_model="fixed",
                              dr_fixed=320.0, signal_model="hollow-cylinder",
                              description="7 T multicompartment (hollow-cylinder white matter)"),
    # Co-located linear model, one preset per relaxivity in use.
    "linear-3t-dr142": ChisepConfig(dr=dr_toolbox(3.0), **_LINEAR_3T,
                                    description="3 T co-located linear R2', Dr = 137·B0/2.9 (χ-separation toolbox)"),
    "linear-3t-dr114": ChisepConfig(dr=DR_CHISEPNET, **_LINEAR_3T,
                                    description="3 T co-located linear R2', Dr = 114 (χ-sepnet)"),
    "linear-3t-dr324": ChisepConfig(dr=dr_static_dephasing(3.0), **_LINEAR_3T,
                                    description="3 T co-located linear R2', Dr = 2πγB0/(9√3) (static dephasing)"),
    # The R2*-only model: R2 tracks the sources so that R2* = 274·(|χ+| + |χ-|) exactly.
    "r2star-3t": ChisepConfig(dr=dr_toolbox(3.0), r2_model="source", r2star_relaxivity=R2STAR_RELAXIVITY_3T,
                              **_LINEAR_3T,
                              description="3 T co-located linear R2' (Dr = 141.7) with R2* = 274·(|χ+| + |χ-|)"),
}


def preset(name: str, **overrides) -> ChisepConfig:
    """A named configuration from :data:`PRESETS`, with any fields overridden."""
    if name not in PRESETS:
        raise KeyError(f"unknown chi-sep preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)


@dataclass
class ChisepMaps:
    """Native-grid χ+/χ- ground truth and the derived maps of one configuration."""
    tissue: TissueParams
    chi_pos: np.ndarray
    chi_neg: np.ndarray
    theta: Optional[np.ndarray]    # fibre-to-B0 angle in degrees, NaN where V1 has no fibre
    dr_pos: np.ndarray
    dr_neg: np.ndarray
    r2prime: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def chi_total(self):
        return self.chi_pos + self.chi_neg


def relaxivity_maps(seg, theta, config: ChisepConfig):
    """Dr+ and Dr- maps (Hz/ppm) on the grid of ``seg`` for ``config.relaxation``."""
    if config.relaxation == "linear":
        dr = np.full(seg.shape, config.dr, np.float32)
        return dr, dr.copy()
    dr_fixed = config.dr_fixed if config.dr_model == "fixed" else None
    return generate_dr_maps_ridani(seg, theta, B0=config.b0, anisotropic=config.anisotropic, dr_fixed=dr_fixed)


def source_r2(chi_pos, chi_neg, config: ChisepConfig):
    """R2 = (r - dr)·(|χ+| + |χ-|) for ``r2_model="source"``, so R2* = r·(|χ+| + |χ-|)."""
    return ((config.r2star_relaxivity - config.dr) * (np.abs(chi_pos) + np.abs(chi_neg))).astype(np.float32)


def build_chisep_maps(data_dir: str, config: ChisepConfig, b0_dir=(0.0, 0.0, 1.0)) -> ChisepMaps:
    """Build χ+ / χ- over the whole head (so the simulated field keeps its background part),
    the fibre angle, the relaxivity maps and the source R2'."""
    v1_kw = {}
    v1_path = os.path.join(data_dir, "maps", "V1.nii.gz")
    if config.v1_flip and os.path.exists(v1_path):
        # The challenge V1.nii.gz is stored L-A-S, mirrored along x relative to every other map;
        # flipping axis 0 registers it (Ridani's published phantom does the same). θ only uses
        # squared components, so no vector-component negation is needed.
        v1_kw["v1"] = np.flip(nib.load(v1_path).get_fdata().astype(np.float32), axis=0).copy()
    tp = TissueParams(root_dir=data_dir, chisep_anisotropy=config.anisotropic,
                      chisep_apply_brain_mask=False, **v1_kw)
    chi_pos = tp.chi_pos.get_fdata().astype(np.float32)
    chi_neg = tp.chi_neg.get_fdata().astype(np.float32)
    seg = tp.seg.get_fdata()
    theta = generate_theta_from_v1(tp.v1.get_fdata(), np.asarray(b0_dir)) if tp.v1 is not None else None
    dr_pos, dr_neg = relaxivity_maps(seg, theta, config)
    r2prime = generate_r2prime(chi_pos, chi_neg, dr=dr_pos, dr_neg=dr_neg)
    return ChisepMaps(tp, chi_pos, chi_neg, theta, dr_pos, dr_neg, r2prime)


def _downsample_theta(theta, ds):
    """Downsample a fibre-angle map whose no-fibre voxels are NaN by normalised interpolation:
    interpolate the NaN-zeroed values and a validity indicator separately, divide, and mark
    voxels with less than half fibre support as NaN."""
    valid = np.isfinite(theta).astype(np.float32)
    num = ds(np.nan_to_num(theta).astype(np.float32))
    den = ds(valid)
    out = np.full(num.shape, np.nan, np.float32)
    ok = den > 0.5
    out[ok] = num[ok] / den[ok]
    return out


def _signal_kwargs(config: ChisepConfig) -> dict:
    """generate_bids keyword arguments that select ``config.signal_model``."""
    if config.signal_model == "hollow-cylinder":
        return dict(chisep_multicompartment=True, chisep_hc_b0=config.hc_b0)
    return {}


def phantom_metadata(config: ChisepConfig, maps: ChisepMaps) -> dict:
    """What the phantom is, for its sidecar: the configuration and the relaxivity it imposes.
    The relaxivity is recorded for documentation and for scoring assumption-matched variants;
    methods keep their own defaults."""
    if config.relaxation == "linear":
        relax = {"Model": "linear", "Dr": float(config.dr)}
    else:
        brain = maps.dr_pos > 0
        relax = {"Model": "ridani", "DrModel": config.dr_model,
                 "DrPos": float(maps.dr_pos[brain].max()) if brain.any() else 0.0,
                 "DrNegMax": float(np.nanmax(maps.dr_neg))}
    if config.r2_model == "source":
        relax["R2starRelaxivity"] = float(config.r2star_relaxivity)
    return {
        "ChiSeparationPhantom": asdict(config),
        "Relaxivity": relax,
        "RelaxivityUnits": "Hz/ppm",
        "R2primeSource": config.r2prime_source,
    }


def generate_chisep_bids(data_dir: str, bids_dir: str, config: ChisepConfig, maps: ChisepMaps = None,
                         subject: str = "1", wm_rois: Optional[str] = None) -> dict:
    """Simulate one chi-sep phantom into a BIDS directory.

    Writes the GRE (magnitude + phase), the optional spin echo, the field and local field, χ,
    χ+ / |χ-|, R2' (fitted or formula, per ``config.r2prime_source``; with "fitted" the formula map
    as ``*_desc-formula_R2prime.nii``), the mask and segmentation, and — in
    ``derivatives/qsm-forward/sub-<subject>/anat`` — the fibre angle (``*_fibreangle.nii``,
    degrees, 0 where there is no fibre), the ground-truth R2 when ``config.r2``, and
    ``*_chisep.json`` (the configuration and its relaxivity). ``wm_rois`` is a label atlas on the
    native grid (e.g. Ridani's WM_fibers_seg.nii.gz) to resample next to them as
    ``*_desc-wmrois_dseg.nii``. Returns a dict of the written derivative paths."""
    maps = maps or build_chisep_maps(data_dir, config)
    tp = maps.tissue
    tes = np.asarray(config.tes_ms, float) * 1e-3
    se_tes = np.asarray(config.se_tes_ms, float) * 1e-3
    native = config.voxel <= 0
    voxel = np.asarray(tp.voxel_size, float) if native else np.array([config.voxel] * 3, float)

    def ds(arr, interp="continuous"):
        if native:
            return np.asarray(arr, np.float32)
        nii = nib.Nifti1Image(np.asarray(arr, np.float32), tp.nii_affine, tp.nii_header)
        return resize(nii, voxel, interp).get_fdata().astype(np.float32)

    chi_pos = ds(maps.chi_pos)
    chi_neg = ds(maps.chi_neg)
    chi_total = chi_pos + chi_neg
    if config.relaxation == "linear":
        # a constant map: build it on the simulation grid rather than interpolate it there
        dr_pos = dr_neg = np.full(chi_pos.shape, config.dr, np.float32)
    else:
        dr_pos = ds(maps.dr_pos)
        dr_neg = ds(maps.dr_neg)
    theta = None
    if maps.theta is not None:
        theta = maps.theta.astype(np.float32) if native else _downsample_theta(maps.theta, ds)
    affine = tp.nii_affine if native else resize(
        nib.Nifti1Image(chi_total, tp.nii_affine, tp.nii_header), voxel).affine

    with tempfile.TemporaryDirectory() as tmpd:
        def w(name, arr):
            p = os.path.join(tmpd, name)
            nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), affine), p)
            return p
        # R2 that tracks the sources is built from the simulated grid's χ, so R2* = r·Σ|χ| holds
        # exactly on the grid the signal is simulated on
        r2_kw = {"R2": w("R2.nii.gz", source_r2(chi_pos, chi_neg, config))} if config.r2_model == "source" else {}
        tp_sim = TissueParams(
            root_dir=data_dir,
            chi=w("chi.nii.gz", chi_total),
            chi_pos=w("chi_pos.nii.gz", chi_pos),
            chi_neg=w("chi_neg.nii.gz", chi_neg),
            mask=w("mask.nii.gz", ds(tp.mask.get_fdata(), "nearest")),
            seg=w("seg.nii.gz", ds(tp.seg.get_fdata(), "nearest")),
            M0=w("M0.nii.gz", ds(tp.M0.get_fdata())),
            R1=w("R1.nii.gz", ds(tp.R1.get_fdata())),
            R2star=w("R2star.nii.gz", ds(tp.R2star.get_fdata())),
            # the hollow-cylinder model reads the fibre angle (degrees) as angle_map; NaN voxels
            # (no fibre direction) get no hollow-cylinder enrichment
            angle_map=w("theta.nii.gz", theta) if theta is not None else None,
            **r2_kw,
        )
        rp = ReconParams(subject=subject, B0=config.b0, TEs=tes, TR=config.tr, flip_angle=config.flip_angle,
                         voxel_size=voxel, peak_snr=config.peak_snr,
                         random_seed=None if config.noiseless else config.seed,
                         se_TR=config.se_tr, se_TEs=se_tes)
        generate_bids(
            tp_sim, rp, bids_dir,
            save_field=True, save_chi_pos=True, save_chi_neg=True, save_r2prime=True,
            dr_pos_map=dr_pos, dr_neg_map=dr_neg, dr=DR_KERNEL,
            chisep_signal=True, save_se=config.se, save_r2=config.r2,
            r2prime_source=config.r2prime_source,
            **_signal_kwargs(config),
        )

    anat = os.path.join(bids_dir, "derivatives", "qsm-forward", f"sub-{subject}", "anat")
    os.makedirs(anat, exist_ok=True)
    written = {}
    ref = sorted(glob.glob(os.path.join(anat, "*_Chimap-neg.nii*")))[0]
    ref_nii = nib.load(ref)
    stem = os.path.basename(ref).split("_Chimap-neg")[0]
    if theta is not None and config.anisotropic:
        # [0, 90]: interpolation can overshoot past 90° (harmless for cos²θ, but keep it physical)
        out = os.path.join(anat, f"{stem}_fibreangle.nii")
        nib.save(nib.Nifti1Image(np.clip(np.nan_to_num(theta), 0.0, 90.0).astype(np.float32),
                                 ref_nii.affine, ref_nii.header), out)
        written["fibreangle"] = out
    if wm_rois:
        src = nib.load(wm_rois)
        lab = src.get_fdata()
        if src.shape != ref_nii.shape:
            lab = resize(nib.Nifti1Image(np.rint(lab).astype(np.float32), src.affine, src.header),
                         np.asarray(ref_nii.header.get_zooms()[:3], float), "nearest").get_fdata()
        out = os.path.join(anat, f"{stem}_desc-wmrois_dseg.nii")
        nib.save(nib.Nifti1Image(np.rint(lab).astype(np.int16), ref_nii.affine, ref_nii.header), out)
        written["wmrois"] = out
    out = os.path.join(anat, f"{stem}_chisep.json")
    with open(out, "w") as f:
        json.dump(phantom_metadata(config, maps), f, indent=2)
    written["metadata"] = out
    for key, pattern in (("r2prime", "_R2prime.nii*"), ("r2prime_formula", "_desc-formula_R2prime.nii*"),
                         ("r2", "_R2map.nii*")):
        hits = sorted(glob.glob(os.path.join(anat, f"{stem}{pattern}")))
        if hits:
            written[key] = hits[0]
    return written
