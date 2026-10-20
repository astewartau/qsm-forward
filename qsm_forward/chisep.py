"""Susceptibility source-separation (chi-separation) phantoms on the QSM Challenge 2.0 head model.

Builds χ+ / χ- ground truth with the Ridani et al. (2026) construction [1], then simulates the
multi-echo GRE (and optional spin-echo) acquisition with one of qsm-forward's chi-sep signal
models. Named presets reproduce the published Ridani configurations and the phantoms the QSM-CI
benchmark scores on; :func:`generate_chisep_bids` runs any preset or a custom configuration.

Signal models (``signal_model``):
  mono          mono-exponential decay, R2* = R2 + Dr+|χ+| + Dr-|χ-|, phase linear in TE [1].
  hollow-cylinder  white matter holds myelin, axonal and extra-axonal water pools (Wharton &
                Bowtell 2012 [2]); elsewhere mono.
  decompose     every voxel holds paramagnetic, diamagnetic and neutral pools with the decay and
                frequency of the DECOMPOSE-QSM signal model (Chen et al. 2021 [3]).

Relaxivity magnitude (``dr_model``), the same sin²θ shape for Dr- in both:
  scaled        static-dephasing kernels, Dr+ = 2πγB0/(9√3) for spheres, Dr- = ½γB0·sin²θ for
                cylinders (Dr+ = 107.84·B0, Dr- = 133.77·sin²θ·B0 Hz/ppm) [1, 4].
  fixed         Dr+ anchored at ``dr_fixed`` Hz/ppm (Shin et al. 2021 [5]); Dr- scaled by the same
                factor.

Resolution: the maps are built at the native 0.64 mm (the χ+/χ- texture construction needs the
real R1/R2* maps), downsampled once in image space to ``voxel`` mm, and the field and signals are
then simulated at that resolution, so the inputs and the ground truth share one PSF. ``voxel=0``
keeps the native grid.

  [1] Ridani D., De Leener B., Alonso-Ortiz E. Magn Reson Med 2026. doi:10.1002/mrm.70468
  [2] Wharton S., Bowtell R. PNAS 2012;109(45):18559-18564. doi:10.1073/pnas.1211075109
  [3] Chen J. et al. NeuroImage 2021;242:118477. doi:10.1016/j.neuroimage.2021.118477
  [4] Yablonskiy D.A., Haacke E.M. Magn Reson Med 1994;32(6):749-763. doi:10.1002/mrm.1910320610
  [5] Shin H.G. et al. NeuroImage 2021;240:118371. doi:10.1016/j.neuroimage.2021.118371
"""

from __future__ import annotations

import glob
import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import Optional

import nibabel as nib
import numpy as np

from .qsm_forward import (
    DR_KERNEL, ReconParams, TissueParams, generate_bids, generate_dr_maps_ridani,
    generate_r2prime, generate_theta_from_v1, resize,
)

SIGNAL_MODELS = ("mono", "hollow-cylinder", "decompose")


@dataclass
class ChisepConfig:
    """One chi-sep phantom configuration. Times are in milliseconds, fields in tesla."""
    b0: float = 7.0
    tes_ms: tuple = (4, 12, 20, 28)
    se_tes_ms: tuple = (10, 30, 50, 70)
    tr: float = 0.05
    flip_angle: float = 15.0
    se_tr: float = 1.5
    voxel: float = 1.0             # mm; 0 keeps the native 0.64 mm grid
    anisotropic: bool = True       # χ- = Δχ·cos²θ + χ0 and θ-dependent Dr-
    dr_model: str = "scaled"       # "scaled" | "fixed"
    dr_fixed: float = DR_KERNEL    # Dr+ anchor (Hz/ppm) for dr_model="fixed"
    signal_model: str = "mono"     # see SIGNAL_MODELS
    hc_b0: Optional[float] = None  # hollow-cylinder pools evaluated at this field (default b0)
    decompose_chi_c: Optional[float] = None  # DECOMPOSE pool susceptibility (default 0.2 ppm)
    peak_snr: float = 100.0
    noiseless: bool = False
    seed: int = 42
    se: bool = True                # simulate the matched multi-echo spin echo
    r2: bool = False               # save the ground-truth R2 map
    v1_flip: bool = True           # register V1 (stored L-A-S) to the R-A-S maps
    description: str = ""

    def __post_init__(self):
        if self.signal_model not in SIGNAL_MODELS:
            raise ValueError(f"signal_model must be one of {SIGNAL_MODELS}, got {self.signal_model!r}")
        if self.dr_model not in ("scaled", "fixed"):
            raise ValueError(f"dr_model must be 'scaled' or 'fixed', got {self.dr_model!r}")


_RIDANI = dict(dr_model="scaled", noiseless=True, se=False, r2=True, voxel=0.0, tr=0.05, flip_angle=15.0)
_RIDANI_3T = dict(b0=3.0, tes_ms=(3, 7, 11, 15, 19, 23), **_RIDANI)
_DECOMPOSE_3T = dict(b0=3.0, tes_ms=(3, 7, 11, 15, 19, 23), dr_model="scaled", se=False, r2=True,
                     voxel=1.0, signal_model="decompose")

_DECOMPOSE_3T_CHEN = dict(_DECOMPOSE_3T, tes_ms=tuple(round(2.7 + 2.9 * k, 1) for k in range(16)), tr=0.0593)

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
    # The ridani-3t-aniso sources and protocol at 1 mm with DECOMPOSE's three-pool signal.
    "decompose-3t": ChisepConfig(noiseless=True, **_DECOMPOSE_3T,
                                 description="3 T, DECOMPOSE three-pool signal, noiseless"),
    "decompose-3t-noise": ChisepConfig(noiseless=False, **_DECOMPOSE_3T,
                                       description="3 T, DECOMPOSE three-pool signal, peak SNR 100"),
    # The same sources and signal with the in vivo protocol of Chen et al. (2021): 16 echoes,
    # TE1/ΔTE/TE16 = 2.7/2.9/46.2 ms, TR 59.3 ms.
    "decompose-3t-chen": ChisepConfig(noiseless=True, **_DECOMPOSE_3T_CHEN,
                                      description="3 T, DECOMPOSE three-pool signal, Chen 2021 protocol, noiseless"),
    "decompose-3t-chen-noise": ChisepConfig(noiseless=False, **_DECOMPOSE_3T_CHEN,
                                            description="3 T, DECOMPOSE three-pool signal, Chen 2021 protocol, peak SNR 100"),
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


def build_chisep_maps(data_dir: str, config: ChisepConfig, b0_dir=(0.0, 0.0, 1.0)) -> ChisepMaps:
    """Build χ+ / χ- over the whole head (so the simulated field keeps its background part),
    the fibre angle, the relaxivity maps and the source R2' = Dr+|χ+| + Dr-|χ-|."""
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
    dr_fixed = config.dr_fixed if config.dr_model == "fixed" else None
    dr_pos, dr_neg = generate_dr_maps_ridani(seg, theta, B0=config.b0, anisotropic=config.anisotropic,
                                             dr_fixed=dr_fixed)
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


def generate_chisep_bids(data_dir: str, bids_dir: str, config: ChisepConfig, maps: ChisepMaps = None,
                         subject: str = "1", wm_rois: Optional[str] = None) -> dict:
    """Simulate one chi-sep phantom into a BIDS directory.

    Writes the GRE (magnitude + phase), the optional spin echo, the field and local field, χ, the
    echo-dependent χ(TE) (``*_Chimap-echoes.nii``, 4D; DECOMPOSE-QSM's input), χ+ / |χ-|, R2' (the signal-derived value for the decompose model), the mask and segmentation,
    and — in ``derivatives/qsm-forward/sub-<subject>/anat`` — the fibre angle (``*_fibreangle.nii``,
    degrees, 0 where there is no fibre) and the ground-truth R2 when ``config.r2``. ``wm_rois`` is a
    label atlas on the native grid (e.g. Ridani's WM_fibers_seg.nii.gz) to resample next to them as
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
        )
        rp = ReconParams(subject=subject, B0=config.b0, TEs=tes, TR=config.tr, flip_angle=config.flip_angle,
                         voxel_size=voxel, peak_snr=config.peak_snr,
                         random_seed=None if config.noiseless else config.seed,
                         se_TR=config.se_tr, se_TEs=se_tes)
        generate_bids(
            tp_sim, rp, bids_dir,
            save_field=True, save_chi_pos=True, save_chi_neg=True, save_r2prime=True,
            dr_pos_map=dr_pos, dr_neg_map=dr_neg, dr=DR_KERNEL,
            chisep_signal=True,
            chisep_multicompartment=config.signal_model == "hollow-cylinder",
            chisep_hc_b0=config.hc_b0,
            chisep_decompose=config.signal_model == "decompose",
            chisep_decompose_chi_c=config.decompose_chi_c,
            save_se=config.se, save_r2=config.r2, save_chi_echoes=True,
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
    echoes = sorted(glob.glob(os.path.join(anat, "*_Chimap-echoes.nii*")))
    if echoes:
        written["chimap_echoes"] = echoes[0]
    r2s = sorted(glob.glob(os.path.join(anat, "*_R2map.nii*")))
    if r2s:
        written["r2"] = r2s[0]
    return written
