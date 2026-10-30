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
  mesoscopic       brain voxels get the signal of explicit sub-voxel sources from a look-up
                   table (:mod:`qsm_forward.mesoscopic`): myelinated axons as hollow cylinders
                   with the Wharton & Bowtell sheath tensor and iron as sphere populations,
                   diffusion Monte Carlo (``meso_table="diffusion"``, default) or static
                   dephasing (``"static"``). White matter uses its fibre angle, other tissue the
                   orientation average. χ- is the input map as the orientation-independent
                   sheath χ·MVF (scored); the field follows the apparent χ- (the voxel-mean
                   sheath tensor projected on B0), written as ``*_desc-apparent_Chimap-neg``.
  spheres          brain voxels get the signal of water-containing para- and diamagnetic
                   spheres (Chen et al. 2021's simulation [7]), static dephasing, 3 T, with the
                   intra-source convention ``sphere_convention`` ("decompose": water inside a
                   source precesses at (2/3)χγB0; "lorentz": Lorentz-corrected). With
                   ``chimap_echoes`` the per-echo χ(TE) input of DECOMPOSE-QSM is written as
                   ``*_Chimap-echoes.nii``: the ground-truth χ plus the echo dependence of a
                   per-echo STAR-QSM reconstruction of the noiseless signal
                   (:func:`qsm_forward.mesoscopic.star_chimap_echoes`).
  Both need ``relaxation="linear"``: its ``dr`` is the nominal relaxivity recorded in the
  sidecar, used outside the brain labels and for the ``*_desc-formula_R2prime`` reference; the
  signal itself takes no relaxivity.
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
  [7] Chen J. et al. NeuroImage 2021;242:118477. doi:10.1016/j.neuroimage.2021.118477
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

from . import mesoscopic
from .qsm_forward import (
    DR_KERNEL, ReconParams, TissueParams, generate_bids, generate_dr_maps_ridani,
    generate_r2prime, generate_theta_from_v1, resize,
)

SIGNAL_MODELS = ("mono", "hollow-cylinder", "mesoscopic", "spheres")
MESO_TABLES = ("diffusion", "static")
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
    meso_table: str = "diffusion"  # "mesoscopic": "diffusion" | "static" LUT
    meso_lut: Optional[str] = None   # "mesoscopic"/"spheres": LUT path (default: mesoscopic.lut_path)
    sphere_convention: str = "decompose"  # "spheres": "decompose" | "lorentz"
    chimap_echoes: bool = False    # "spheres": write the per-echo χ(TE) (needs QSM.rs's STAR)
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
        check("meso_table", MESO_TABLES)
        check("sphere_convention", mesoscopic.SPHERE_CONVENTIONS)
        if self.signal_model in ("mesoscopic", "spheres") and self.relaxation != "linear":
            raise ValueError(f"signal_model={self.signal_model!r} needs relaxation='linear' (its dr is the "
                             "nominal relaxivity outside the brain labels and of the formula R2')")
        if self.signal_model == "spheres" and self.b0 != 3.0:
            raise ValueError("signal_model='spheres' is tabulated at 3 T only")
        if self.chimap_echoes and self.signal_model != "spheres":
            raise ValueError("chimap_echoes is defined for signal_model='spheres'")


_RIDANI = dict(dr_model="scaled", noiseless=True, se=False, r2=True, voxel=0.0, tr=0.05, flip_angle=15.0)
_RIDANI_3T = dict(b0=3.0, tes_ms=(3, 7, 11, 15, 19, 23), **_RIDANI)
# Co-located linear phantoms: 3 T, 1 mm, six echoes 5-30 ms, matched spin echo, peak SNR 100, the
# shipped R2' fitted from the signals. Only the relaxivity (so R2' and the magnitude) differs.
_LINEAR_3T = dict(b0=3.0, tes_ms=(5, 10, 15, 20, 25, 30), se_tes_ms=(10, 30, 50, 70), voxel=1.0,
                  anisotropic=True, relaxation="linear", r2prime_source="fitted", peak_snr=100.0,
                  se=True, r2=True)

# Mesoscopic and sphere phantoms: 1 mm, peak SNR 100, matched spin echo, R2' fitted from the
# signals, isotropic χ- input (the orientation dependence comes out of the signal model).
_MESO = dict(voxel=1.0, anisotropic=False, relaxation="linear", r2prime_source="fitted",
             peak_snr=100.0, se=True, r2=True, se_tes_ms=(10, 30, 50, 70))
# Chen et al. 2021's in vivo 3 T protocol: 16 echoes 2.7:2.9:46.2 ms, TR 59.3 ms
_CHEN_3T = dict(b0=3.0, tes_ms=tuple(round(2.7 + 2.9 * i, 1) for i in range(16)), tr=0.0593)

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
    # Mesoscopic myelin + iron (diffusion Monte Carlo LUT), Chen-like 3 T protocol.
    "meso-3t": ChisepConfig(signal_model="mesoscopic", meso_table="diffusion", dr=dr_toolbox(3.0),
                            **_CHEN_3T, **_MESO,
                            description="3 T mesoscopic sub-voxel sources (hollow-cylinder myelin, iron spheres; diffusion)"),
    # 7 T, 8 echoes 3-45 ms, white matter from mesoscopic hollow cylinders (not a closed form).
    "wm-micro-7t": ChisepConfig(signal_model="mesoscopic", meso_table="diffusion", dr=dr_toolbox(7.0),
                                b0=7.0, tes_ms=(3, 9, 15, 21, 27, 33, 39, 45), **_MESO,
                                description="7 T mesoscopic white-matter microstructure (hollow cylinders; diffusion)"),
    # Water-containing spheres after Chen et al. 2021, per-echo χ(TE); one per intra-source convention.
    "spheres-3t-paper": ChisepConfig(signal_model="spheres", sphere_convention="decompose", chimap_echoes=True,
                                     dr=dr_static_dephasing(3.0), **_CHEN_3T, **_MESO,
                                     description="3 T water-containing sub-voxel spheres, intra-source water at (2/3)χγB0"),
    "spheres-3t-lorentz": ChisepConfig(signal_model="spheres", sphere_convention="lorentz", chimap_echoes=True,
                                       dr=dr_static_dephasing(3.0), **_CHEN_3T, **_MESO,
                                       description="3 T water-containing sub-voxel spheres, Lorentz-corrected intra-source field"),
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


def _signal_kwargs(config: ChisepConfig, factor=None) -> dict:
    """generate_bids keyword arguments that select ``config.signal_model``. ``factor`` is the
    per-voxel model of the LUT-driven signal models (:func:`signal_factor`)."""
    if config.signal_model == "hollow-cylinder":
        return dict(chisep_multicompartment=True, chisep_hc_b0=config.hc_b0)
    if config.signal_model in ("mesoscopic", "spheres"):
        return dict(signal_factor=factor)
    return {}


def load_lut(config: ChisepConfig):
    """The LUT of a LUT-driven signal model, or None."""
    if config.signal_model == "mesoscopic":
        return mesoscopic.MesoLUT(mesoscopic.lut_path(config.meso_table, config.meso_lut))
    if config.signal_model == "spheres":
        return mesoscopic.SphereLUT(mesoscopic.lut_path("spheres", config.meso_lut))
    return None


def signal_factor(config: ChisepConfig, lut, chi_pos, chi_neg, theta_deg, seg):
    """Per-voxel signal model on the simulation grid for ``config.signal_model`` (or None)."""
    if config.signal_model == "mesoscopic":
        return mesoscopic.MesoscopicSignal(lut, chi_pos, chi_neg, theta_deg, seg, config.b0,
                                           se_tes_ms=config.se_tes_ms)
    if config.signal_model == "spheres":
        return mesoscopic.SphereSignal(lut, chi_pos, chi_neg, seg, config.b0, config.sphere_convention)
    return None


def fibre_theta(theta_deg, seg):
    """Fibre angle (degrees) where the mesoscopic model uses one: white matter with a fibre
    direction; NaN elsewhere (orientation average)."""
    th = np.asarray(theta_deg, float) if theta_deg is not None else np.full(np.shape(seg), np.nan)
    wm = np.rint(np.asarray(seg)).astype(int) == mesoscopic.WM_LABEL
    return np.where(wm & np.isfinite(th), th, np.nan)


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
    if config.signal_model in ("mesoscopic", "spheres"):
        relax["Note"] = ("nominal: the signal takes no relaxivity inside the brain labels; Dr is used outside "
                         "them and for the desc-formula R2' reference")
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

    # LUT-driven signal models, on the simulation grid
    lut = load_lut(config)
    factor = None
    chi_neg_app = None
    if lut is not None:
        seg_sim = ds(tp.seg.get_fdata(), "nearest")
        theta_fibre = fibre_theta(theta, seg_sim)
        if config.signal_model == "mesoscopic":
            # the field follows the voxel-mean sheath tensor: apparent χ- in white matter
            chi_neg_app = mesoscopic.chi_neg_apparent(chi_neg, np.deg2rad(theta_fibre)).astype(np.float32)
            chi_total = chi_pos + chi_neg_app
        factor = signal_factor(config, lut, chi_pos, chi_neg, theta_fibre, seg_sim)

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
            **_signal_kwargs(config, factor),
        )

    anat = os.path.join(bids_dir, "derivatives", "qsm-forward", f"sub-{subject}", "anat")
    os.makedirs(anat, exist_ok=True)
    written = {}
    ref = sorted(glob.glob(os.path.join(anat, "*_Chimap-neg.nii*")))[0]
    ref_nii = nib.load(ref)
    stem = os.path.basename(ref).split("_Chimap-neg")[0]
    if theta is not None and (config.anisotropic or config.signal_model == "mesoscopic"):
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
    def save_like_ref(name, arr):
        p = os.path.join(anat, f"{stem}_{name}.nii")
        nib.save(nib.Nifti1Image(np.asarray(arr, np.float32), ref_nii.affine, ref_nii.header), p)
        return p

    meta = phantom_metadata(config, maps)
    if config.signal_model == "mesoscopic":
        written["chi_neg_apparent"] = save_like_ref("desc-apparent_Chimap-neg", np.abs(chi_neg_app))
        meta["SignalModel"] = {
            "Model": "mesoscopic", "Table": lut.mode, "LUT": os.path.basename(lut.path),
            "LUTParameters": lut.params, "IronRelaxivity": lut.iron_relaxivity(config.b0),
            "IntrinsicR2FlooredFraction": getattr(factor, "floored_fraction", None),
            "ChiNeg": "scored: orientation-independent sheath χ·MVF (= input map); "
                      "desc-apparent: voxel-mean sheath tensor projected on B0 (drives the field)",
            "Chimap": "χ+ + apparent χ-",
        }
    elif config.signal_model == "spheres":
        meta["SignalModel"] = {"Model": "spheres", "Convention": config.sphere_convention,
                               "LUT": os.path.basename(lut.path),
                               "WrapLimitPpm": float(mesoscopic.wrap_limit_ppm(max(config.tes_ms), config.b0))}
        if config.chimap_echoes:
            written["chimap_echoes"] = write_chimap_echoes(anat, stem, config, factor)
            meta["SignalModel"]["ChimapEchoes"] = (
                "χ_GT + [STAR(echo i) - STAR(echo-combined)]: per-echo STAR-QSM (QSM.rs, exact port of "
                "STI Suite 3.0 QSM_star) of the noiseless local phase, Laplacian-unwrapped (STI "
                "MRPhaseUnwrap settings), mask mean removed; combined = STAR of the summed phases at ΣTE")
    out = os.path.join(anat, f"{stem}_chisep.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    written["metadata"] = out
    for key, pattern in (("r2prime", "_R2prime.nii*"), ("r2prime_formula", "_desc-formula_R2prime.nii*"),
                         ("r2", "_R2map.nii*")):
        hits = sorted(glob.glob(os.path.join(anat, f"{stem}{pattern}")))
        if hits:
            written[key] = hits[0]
    return written


def write_chimap_echoes(anat, stem, config: ChisepConfig, factor, tools=None, unwrap="laplacian"):
    """Write ``*_Chimap-echoes.nii`` (4D, ppm) for a ``spheres`` phantom from the written ground
    truth (``*_Chimap``, ``*_fieldmap-local``, ``*_mask``) and the per-voxel intra-voxel phase
    (:func:`qsm_forward.mesoscopic.star_chimap_echoes`). ``tools`` defaults to
    :class:`qsm_forward.mesoscopic.QsmrsTools`."""
    tools = tools or mesoscopic.QsmrsTools()
    load = lambda name: nib.load(os.path.join(anat, f"{stem}_{name}.nii"))
    ref = load("Chimap")
    chi_gt = ref.get_fdata()
    local = load("fieldmap-local").get_fdata()
    mask = load("mask").get_fdata() > 0.5
    if local.shape != factor.mask.shape:
        raise ValueError("chimap_echoes needs the signal simulated on the output grid")
    tes = np.asarray(config.tes_ms, float)
    intra = np.zeros((len(tes),) + local.shape)
    ph = factor.lut.unwrapped_phase(factor._cp, factor._cn, tes, factor.convention, factor.interp)
    intra[:, factor.mask] = ph.T
    phases = mesoscopic.echo_local_phases(local, tes, config.b0, intra)
    voxel = np.asarray(ref.header.get_zooms()[:3], float)
    chi_e = mesoscopic.star_chimap_echoes(chi_gt, phases, mask, tes, config.b0, voxel, tools, unwrap=unwrap)
    out = os.path.join(anat, f"{stem}_Chimap-echoes.nii")
    nib.save(nib.Nifti1Image(np.moveaxis(chi_e, 0, -1).astype(np.float32), ref.affine), out)
    return out
