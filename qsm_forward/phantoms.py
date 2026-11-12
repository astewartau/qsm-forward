"""Head phantoms built from in vivo data, and strong-source / pathology insertion.

The 2019 challenge head phantom (``data/chimodel``) is one anatomy whose χ is piecewise smooth
inside segmented regions. Models trained on it fingerprint that phantom: its cortical rim after
background removal and its steeper-than-in-vivo spectrum. This module builds phantoms whose χ
truth comes from in vivo reconstructions, so every subject and scanner gives a new anatomy with
in vivo texture, and adds strong sources (calcifications, haemorrhages, microbleeds, MS-like rim
lesions, enlarged veins) whose position, size and strength are known exactly.

In vivo-derived phantom (:func:`build_invivo_phantom`)
  Inputs: an in vivo χ map (ppm), the multi-echo magnitude it came from, the echo times, and an
  anatomical brain mask. The χ map becomes the truth inside the brain after cleaning
  (:func:`clean_chi`):

  1. referencing (median over the brain = 0);
  2. de-spiking: voxels more than ``despike_sd`` robust SDs from their 3x3x3 median are replaced
     by it (rim errors of background-field removal, isolated unwrapping blips);
  3. filling: voxels of the brain mask that the source map does not cover (V-SHARP erodes 2-3 mm
     of cortex) get a smooth extrapolation of the low-pass map plus the high-pass texture of the
     mirror-image voxel inside, so the rim keeps in vivo texture rather than a smooth band;
  4. magic-cone clamp: per |k| shell, power within ``cone_halfwidth_deg`` of the magic angle is
     scaled down to the power 15-35 degrees away (only down, never up), which removes the
     streaking the source inversion left without suppressing the cone below isotropy;
  5. denoising: ``"wiener"`` (default) fits P(k) = A|k|^-beta + N to the radial spectrum and
     applies S/(S + N), which leaves the in vivo power law intact where signal dominates and
     only rolls off where the source map's noise floor dominates (nothing happens if the source
     inversion already suppressed its high |k|); ``"tv"`` is an edge-preserving Chambolle TV
     denoise; ``None`` keeps the source texture.

  The trade-off: whatever noise and streaking survive become "anatomy" in the truth (a model
  trained on the phantom then learns to accept them as normal), while every extra step of
  cleaning pushes the truth towards the too-smooth piecewise phantom the module is meant to
  replace. The defaults are the least cleaning that removes the parts that are clearly not
  anatomy (spikes, cone excess, the white noise floor). :func:`clean_chi` reports what it did.

  Outside the brain the phantom has a head model derived from the magnitude
  (:func:`head_model`): a head hull (threshold, closing, hole filling), and inside it soft tissue
  (bright), and dark voxels split into air cavities (sinuses, mastoids: at the noise level of
  outside air and bulky) and bone (the rest). χ, M0 and R2* values follow the 2019 head phantom
  (air +9.2 ppm, bone -2.1 ppm, soft tissue 0 ppm, against water/CSF at about 0), so background
  fields have the same sources as there but real head geometry. M0 and R2* everywhere else come
  from a log-linear, signal-weighted mono-exponential fit of the magnitude echoes; M0 divides out
  the steady-state factor of the source TR and flip angle with literature R1 values, so
  re-simulating with the source protocol reproduces the source magnitude.

  The field of view of in vivo data usually cuts through the head inferiorly. The FFT forward
  model pads the volume with its corner value (air), which would put an air interface across the
  bottom slice. ``neck_mm`` extends the phantom inferiorly by replicating its bottom slice (brain
  replaced by soft tissue), and ``cut_extend_mm`` does the same for any other face the head
  crosses; :func:`simulate` simulates on the extended grid and crops every output back to the
  source field of view.

Strong sources and pathology (:func:`insert_sources`)
  :class:`SourceSpec` describes a kind of source with ranges for count, size and strength and a
  location constraint; :data:`PRESETS` holds named sets, including ``"streak-heavy"``. Insertion
  is seeded, uses partial-volume occupancy at the edges, changes M0 and R2* to match (signal voids
  from R2* = R2*_tissue + DR_KERNEL·|Δχ|, low proton density in calcifications), records each
  source (centre, radius, χ, volume, moment) and returns an instance label map. Calcifications
  carry seg label 16, the label QSM-CI's calcification metrics read.

Streak labels (:func:`cone_restricted_error`)
  The reconstruction error restricted to a narrow band around the magic-angle cone in k-space:
  with the exact truth of a phantom, this is the exact streaking label.

Simulation (:func:`simulate`)
  Saves the phantom in the qsm-forward layout (``chimodel/``, ``maps/``, ``masks/``), runs
  :func:`qsm_forward.generate_bids` with the true total field (``*_fieldmap.nii``), true local field
  (``*_fieldmap-local.nii``) and shimmed total field saved, writes the source labels, and crops
  back to the source field of view.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .qsm_forward import DR_KERNEL, ReconGrid, ReconParams, TissueParams, generate_bids

# ---------------------------------------------------------------------------------------------
# labels and tissue values

# seg labels; 11 and 13-16 match label.json / the 2019 head phantom ("Muscle", 15, is the head
# phantom's label for all extracranial soft tissue). 17-21 are this module's own.
LABEL_VEIN = 11
LABEL_BONE = 13
LABEL_AIR = 14
LABEL_SOFT_TISSUE = 15
LABEL_CALCIFICATION = 16
LABEL_BRAIN = 17           # brain without a tissue segmentation
LABEL_HAEMORRHAGE = 18
LABEL_MICROBLEED = 19
LABEL_LESION = 20          # MS-like lesion core
LABEL_LESION_RIM = 21      # paramagnetic rim
LABEL_NAMES = {
    LABEL_VEIN: "Blood", LABEL_BONE: "Bone", LABEL_AIR: "Air", LABEL_SOFT_TISSUE: "Muscle",
    LABEL_CALCIFICATION: "Calcification", LABEL_BRAIN: "Brain (unsegmented)",
    LABEL_HAEMORRHAGE: "Haemorrhage", LABEL_MICROBLEED: "Microbleed", LABEL_LESION: "Lesion",
    LABEL_LESION_RIM: "Lesion rim",
}

# χ (ppm, relative to water/CSF ~ 0) outside the brain, as in the 2019 head phantom.
# Air: +9.2 ppm against water (Schenck 1996, Med Phys 23:815: water -9.05, air +0.36 ppm SI).
# Cortical bone: -2.1 ppm (head phantom; in vivo estimates -1.5 to -2.5 ppm).
CHI_AIR_PPM = 9.2
CHI_BONE_PPM = -2.1
CHI_SOFT_TISSUE_PPM = 0.0

# R1 (1/s) at 3 T used to divide the steady-state factor out of the fitted S0 (literature means:
# brain GM/WM mix ~0.9, muscle/scalp ~0.8-1.0, fat ~2.7; bone has almost no signal anyway).
R1_BRAIN = 0.9
R1_SOFT_TISSUE = 1.0
R1_BONE = 1.0

# T1 (s) at 3 T per seg label inside the brain, approximate literature means: WM 0.85 and cortical
# GM 1.35 (Wansapura et al., JMRI 1999; Marques et al., NeuroImage 2010, MP2RAGE), CSF 4.2 (Lu et
# al., MRM 2005), venous blood 1.6 (Lu et al., MRM 2004), deep grey matter between GM and WM and
# lower where iron is high (caudate 1.35, putamen 1.25, globus pallidus 0.95, thalamus 1.10, red
# nucleus, substantia nigra 1.0, dentate 1.05; Marques 2010 and Wright et al., MAGMA 2008 give
# values in these ranges). Used for the steady-state factor, so they matter only when the
# phantom is simulated with a TR or flip angle other than the source's (M0 is S0 divided by the
# factor, and re-simulating with the source protocol multiplies it back).
T1_3T_BY_LABEL = {1: 1.35, 2: 0.95, 3: 1.25, 4: 1.0, 5: 1.05, 6: 1.0, 7: 1.10, 8: 0.85, 9: 1.35,
                  10: 4.2, 11: 1.6}
R1_BY_LABEL = {k: 1.0 / v for k, v in T1_3T_BY_LABEL.items()}

# FreeSurfer aseg ids (SynthSeg output) -> the head phantom's labels (1 caudate, 2 globus pallidus,
# 3 putamen, 7 thalamus, 8 white matter, 9 grey matter, 10 CSF), the scheme QSM-CI's region
# metrics read. Brainstem (16) and ventral diencephalon (28/60) are mostly white matter and go to 8;
# the red nucleus (4), dentate (5) and substantia nigra (6) are not in aseg and must come from
# another atlas (``seg_from_freesurfer(..., overrides=...)``). Hippocampus, amygdala and
# accumbens go to grey matter, choroid plexus to CSF.
FREESURFER_TO_PHANTOM = {
    2: 8, 41: 8, 7: 8, 46: 8, 16: 8, 28: 8, 60: 8,
    3: 9, 42: 9, 8: 9, 47: 9, 17: 9, 53: 9, 18: 9, 54: 9, 26: 9, 58: 9,
    4: 10, 43: 10, 5: 10, 44: 10, 14: 10, 15: 10, 24: 10, 31: 10, 63: 10,
    10: 7, 49: 7, 11: 1, 50: 1, 13: 2, 52: 2, 12: 3, 51: 3,
}


def seg_from_freesurfer(aseg: np.ndarray, overrides: Sequence = ()) -> np.ndarray:
    """Map a FreeSurfer/SynthSeg label map to the head phantom's brain labels.

    Unknown non-zero ids become :data:`LABEL_BRAIN`. ``overrides`` is a sequence of
    ``(bool mask, label)`` applied in order on top (e.g. red nucleus, dentate and substantia nigra
    from a χ atlas, which aseg does not have)."""
    a = np.asarray(aseg).astype(np.int32)
    lut = np.zeros(max(int(a.max()), max(FREESURFER_TO_PHANTOM)) + 1, np.int16)
    lut[1:] = LABEL_BRAIN
    for k, v in FREESURFER_TO_PHANTOM.items():
        lut[k] = v
    out = lut[np.clip(a, 0, None)]
    for m, lab in overrides:
        out[np.asarray(m, bool)] = lab
    return out

# ---------------------------------------------------------------------------------------------
# small helpers


def _vox(voxel_size) -> np.ndarray:
    return np.asarray(voxel_size, float)[:3]


def _robust_std(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _ball(radius_mm: float, vox) -> np.ndarray:
    r = np.maximum(np.ceil(radius_mm / _vox(vox)).astype(int), 1)
    g = np.meshgrid(*[np.arange(-k, k + 1) * v for k, v in zip(r, _vox(vox))], indexing="ij")
    return sum(gi ** 2 for gi in g) <= radius_mm ** 2 + 1e-9


def _largest_cc(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n <= 1:
        return m.astype(bool)
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    return lab == 1 + int(np.argmax(sizes))


def _kgrid(shape, vox):
    """k-space coordinates (cycles/mm), FFT order, one array per axis."""
    return np.meshgrid(*[np.fft.fftfreq(n, d=float(d)) for n, d in zip(shape, _vox(vox))], indexing="ij")


def _cone_deviation(shape, vox, B0_dir):
    """|k| (cycles/mm) and the angle (degrees) of k from the magic-angle cone around B0."""
    k = _kgrid(shape, vox)
    b = np.asarray(B0_dir, float)
    b = b / np.linalg.norm(b)
    kr = np.sqrt(k[0] ** 2 + k[1] ** 2 + k[2] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.abs(k[0] * b[0] + k[1] * b[1] + k[2] * b[2]) / np.where(kr > 0, kr, 1.0)
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    dev = np.abs(ang - math.degrees(math.acos(1.0 / math.sqrt(3.0))))
    return kr, dev


def _gauss_norm(x, w, sigma_vox):
    num = ndi.gaussian_filter(x * w, sigma_vox)
    den = ndi.gaussian_filter(w.astype(np.float64), sigma_vox)
    return num, den


# ---------------------------------------------------------------------------------------------
# magnitude: R2*, S0 and the head model


def fit_r2star(magnitude: np.ndarray, TEs: Sequence[float], mask: Optional[np.ndarray] = None,
               r2star_max: float = 1000.0):
    """Mono-exponential fit S(TE) = S0 exp(-R2* TE) of a 4-D magnitude (echoes last).

    Log-linear least squares weighted by S^2 (the inverse variance of log S at constant noise),
    so late, noise-dominated echoes count little. Returns ``(S0, R2star)``; R2* is clipped to
    ``[0, r2star_max]`` 1/s, and both are 0 outside ``mask`` when one is given.
    """
    te = np.asarray(TEs, float)
    if magnitude.ndim != 4 or magnitude.shape[-1] != te.size:
        raise ValueError(f"magnitude must be 4-D with {te.size} echoes last, got {magnitude.shape}")
    if te.size < 2:
        raise ValueError("need at least two echoes to fit R2*")
    e1 = np.asarray(magnitude[..., 0])
    pos = e1[e1 > 0]
    floor = max(float(np.percentile(pos, 1)) if pos.size else 1.0, 1e-12)
    s0 = np.zeros(magnitude.shape[:3], np.float64)
    r2 = np.zeros(magnitude.shape[:3], np.float64)
    for i in range(magnitude.shape[0]):  # slab by slab: the 4-D float64 temporaries are large
        s = np.maximum(np.asarray(magnitude[i], np.float64), 0.0)
        y = np.log(np.maximum(s, floor * 1e-3))
        w = s ** 2
        sw = w.sum(-1)
        st = (w * te).sum(-1)
        stt = (w * te ** 2).sum(-1)
        sy = (w * y).sum(-1)
        sty = (w * te * y).sum(-1)
        det = sw * stt - st ** 2
        ok = det > 1e-12 * np.maximum(sw, 1e-30) ** 2
        slope = np.where(ok, (sw * sty - st * sy) / np.where(ok, det, 1.0), 0.0)
        rate = np.clip(-slope, 0.0, r2star_max)
        # S0 from the weighted mean of log S + R2* TE with the clipped rate, so S0 matches R2*
        icpt = np.where(sw > 0, (sy + rate * st) / np.where(sw > 0, sw, 1.0), np.log(np.maximum(s[..., 0], floor * 1e-3)))
        s0[i] = np.where(ok, np.exp(icpt), s[..., 0])
        r2[i] = rate
    s0[~np.isfinite(s0)] = 0.0
    if mask is not None:
        s0 = np.where(mask, s0, 0.0)
        r2 = np.where(mask, r2, 0.0)
    return s0.astype(np.float32), r2.astype(np.float32)


def spgr_factor(R1, TR, flip_angle_deg):
    """Steady-state spoiled-GRE factor (1 - E1) sin(a) / (1 - cos(a) E1), as in generate_signal."""
    e1 = np.exp(-TR * np.asarray(R1, float))
    a = np.deg2rad(flip_angle_deg)
    return (1 - e1) * np.sin(a) / (1 - np.cos(a) * e1)


def head_model(magnitude_e1: np.ndarray, brain_mask: np.ndarray, voxel_size,
               t_head: float = 0.10, t_dark: float = 0.35, t_air: float = 0.06,
               closing_mm: float = 4.0, air_opening_mm: float = 2.5, min_air_ml: float = 0.3, bias_sigma_mm: float = 15.0,
               axial_axis: int = 2) -> np.ndarray:
    """Label the head from the first-echo magnitude: brain (the given mask), soft tissue, bone,
    air cavities inside the head, and air outside it.

    Thresholds are fractions of the median first-echo magnitude in the brain:
      - head hull: magnitude > ``t_head``, largest component, closed by ``closing_mm``, holes filled
        in 3-D and in every axial slice (so the skull and enclosed sinuses are inside the hull);
      - inside the hull, outside the brain: dark voxels (bias-normalised magnitude < ``t_dark``) are
        bone or air, the rest soft tissue (scalp, muscle, CSF outside the brain mask, fat);
      - dark voxels at the level of the air outside the head (raw magnitude < ``t_air``) that survive
        an ``air_opening_mm`` opening (bulky, unlike the skull shell), in components of at least
        ``min_air_ml``, are air cavities; the other dark voxels are bone.

    Returns an int16 label map with :data:`LABEL_BRAIN`, :data:`LABEL_SOFT_TISSUE`,
    :data:`LABEL_BONE` and :data:`LABEL_AIR` (outside the head as well).

    Known limits: bone and air both have almost no signal in GRE, so the split is geometric
    (bulky and at the noise floor = air) and thin air cells (mastoid) end up as bone, while thick
    cancellous bone with marrow signal ends up as soft tissue. Fat is not separated from soft tissue.
    """
    vox = _vox(voxel_size)
    e1 = np.asarray(magnitude_e1, np.float64)
    brain = np.asarray(brain_mask, bool)
    ref = float(np.median(e1[brain])) if brain.any() else float(np.percentile(e1, 90))
    if ref <= 0:
        raise ValueError("first-echo magnitude is zero inside the brain mask")
    head = _largest_cc((e1 > t_head * ref) | brain)
    head = ndi.binary_closing(head, structure=_ball(closing_mm, vox), iterations=1, border_value=0) | head
    head = ndi.binary_fill_holes(head)
    for z in range(head.shape[axial_axis]):
        sl = [slice(None)] * 3
        sl[axial_axis] = z
        head[tuple(sl)] = ndi.binary_fill_holes(head[tuple(sl)])
    head |= brain

    # bias field from bright tissue only (bone and air would pull it down)
    bright = head & (e1 > 0.3 * ref)
    num, den = _gauss_norm(e1, bright, bias_sigma_mm / vox)
    bias = np.where(den > 1e-3, num / np.maximum(den, 1e-12), ref)
    norm = e1 / np.maximum(bias, 1e-12 * ref)

    inside = head & ~brain
    dark = inside & (norm < t_dark)
    air = dark & (e1 < t_air * ref)
    air = ndi.binary_opening(air, structure=_ball(air_opening_mm, vox))
    lab, n = ndi.label(air)
    if n:
        sizes = ndi.sum(air, lab, index=np.arange(1, n + 1)) * float(np.prod(vox)) / 1000.0
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_air_ml
        air = keep[lab]
        # give the cavities back the edge the opening took (geodesically, within the dark voxels)
        air = ndi.binary_dilation(air, iterations=max(int(round(air_opening_mm / vox.min())), 1), mask=dark)
    bone = dark & ~air

    seg = np.full(e1.shape, LABEL_AIR, np.int16)
    seg[inside & ~dark] = LABEL_SOFT_TISSUE
    seg[bone] = LABEL_BONE
    seg[air] = LABEL_AIR
    seg[brain] = LABEL_BRAIN
    return seg


# ---------------------------------------------------------------------------------------------
# cleaning the source χ


def _fill_outside(chi, support, target, vox, sigmas_mm=(2.0, 4.0, 8.0, 16.0), texture_mm=6.0):
    """Values for every voxel: χ on ``support``; elsewhere a multi-scale normalised-convolution
    extrapolation of the low-pass map, plus (within ``target`` and up to ``texture_mm`` from the
    support) the high-pass texture of the mirror-image voxel inside the support. Returns
    (filled, lowpass)."""
    out = np.where(support, chi, 0.0).astype(np.float64)
    known = support.copy()
    s = np.asarray(sigmas_mm, float)
    num, den = _gauss_norm(np.where(support, chi, 0.0), support, s[0] / vox)
    low_support = np.where(den > 1e-6, num / np.maximum(den, 1e-12), 0.0)
    for sig in s:
        num, den = _gauss_norm(np.where(known, out, 0.0), known, sig / vox)
        fill = (~known) & (den > 0.05)
        out[fill] = num[fill] / den[fill]
        known |= fill
    if not known.all():
        out[~known] = float(np.median(chi[support]))
    # texture for the gap: high-pass of the mirror-image voxel
    hp = np.where(support, chi - low_support, 0.0)
    gap = target & ~support
    if gap.any():
        dist, idx = ndi.distance_transform_edt(~support, sampling=vox, return_indices=True)
        gi = np.nonzero(gap & (dist <= texture_mm))
        p = np.stack(gi, 0)
        nn = idx[(slice(None),) + gi]
        mir = np.clip(2 * nn - p, 0, np.array(chi.shape)[:, None] - 1)
        use_mir = support[tuple(mir)]
        src = np.where(use_mir[None, :], mir, nn)
        out[gi] += hp[tuple(src)]
    return out


def _tv_denoise(f: np.ndarray, weight: float, n_iter: int = 60) -> np.ndarray:
    """Chambolle (2004) projection TV denoising, isotropic, 3-D, forward differences."""
    f = f.astype(np.float32)
    p = np.zeros((3,) + f.shape, np.float32)
    tau = 1.0 / 8.0

    def div(q):
        # backward differences; q[a] is zero on the last plane (grad sets it), so the roll's
        # wrap-around term vanishes and this is the adjoint of grad with Neumann boundaries
        d = np.zeros_like(f)
        for a in range(3):
            d += q[a] - np.roll(q[a], 1, axis=a)
        return d

    def grad(u):
        g = np.zeros((3,) + u.shape, np.float32)
        for a in range(3):
            g[a] = np.roll(u, -1, axis=a) - u
            sl = [slice(None)] * 3
            sl[a] = -1
            g[a][tuple(sl)] = 0.0
        return g

    for _ in range(n_iter):
        g = grad(div(p) - f / weight)
        nrm = np.sqrt((g ** 2).sum(0))
        p = (p + tau * g) / (1.0 + tau * nrm)
    return f - weight * div(p)


def _radial_profile(P, kr, edges):
    idx = np.digitize(kr.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
    s = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=len(edges) - 1)
    kc = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        return kc, s / np.maximum(cnt, 1), cnt


def clean_chi(chi: np.ndarray, support: np.ndarray, target: np.ndarray, voxel_size,
              B0_dir=(0.0, 0.0, 1.0), despike_sd: Optional[float] = 8.0, despike_rim_mm: float = 3.0,
              despike_max_size: int = 2, cone: Optional[str] = "clamp", cone_max_gain: float = 2.0,
              cone_halfwidth_deg: float = 8.0, cone_iterations: int = 3, denoise: Optional[str] = "wiener",
              denoise_strength: float = 1.0, tv_weight: float = 0.004, reference: str = "median"):
    """Turn an in vivo χ map into a χ truth (see the module docstring for the steps).

    Parameters
    ----------
    chi : ndarray
        Source χ (ppm).
    support : ndarray of bool
        Where the source χ is valid (usually its non-zero support).
    target : ndarray of bool
        The brain mask the truth is defined on; voxels of it outside ``support`` are filled.
    voxel_size : sequence of float
    B0_dir : sequence of float
        B0 direction of the source reconstruction in voxel coordinates (for the cone clamp).
    despike_sd : float or None
        Outliers are voxels further than this many robust SDs from their 3x3x3 median; None
        skips de-spiking. Outliers within ``despike_rim_mm`` of the support edge (where
        background-field removal leaves its errors) are all replaced by the median; deeper in,
        only isolated ones (26-connected same-sign clusters of at most ``despike_max_size``
        voxels), so veins and other thin real structures are kept.
    despike_rim_mm, despike_max_size : float, int
    cone : {"clamp", "equalise", None}
        "clamp": scale magic-cone power DOWN to the off-cone power at the same |k| (removes
        streaking, leaves a cone deficit alone). "equalise": also scale it UP, by at most
        ``cone_max_gain`` in amplitude, which restores isotropic spectral statistics to sources
        whose inversion suppressed the cone (iLSQR, TV, TKD/TSVD) by amplifying the cone content
        they kept. None: leave the cone alone.
    cone_max_gain : float
    cone_halfwidth_deg : float
        Angular half-width of the clamped band (raised-cosine taper to zero at this angle).
    cone_iterations : int
        Clamp, re-mask and refill this many times (re-masking spreads some cone power back).
    denoise : {"wiener", "tv", None}
    denoise_strength : float
        Multiplies the noise power in the Wiener filter.
    tv_weight : float
        TV weight (ppm) for ``denoise="tv"``.
    reference : {"median", "mean", None}
        Shift so that this statistic of χ over ``target`` is zero.

    Returns
    -------
    chi_clean : ndarray (float32), zero outside ``target``
    info : dict
        What was done: spike fraction, filled fraction, cone excess removed (dB per band),
        fitted spectral exponent, noise-floor fraction and the |k| where the Wiener gain is 0.5.
    """
    vox = _vox(voxel_size)
    support = np.asarray(support, bool) & np.isfinite(chi)
    target = np.asarray(target, bool)
    if support.sum() < 100:
        raise ValueError("source χ support is (almost) empty")
    chi = np.where(support, np.asarray(chi, np.float64), 0.0)
    info = {"support_voxels": int(support.sum()), "target_voxels": int(target.sum()),
            "filled_fraction": float((target & ~support).sum() / max(target.sum(), 1))}

    # 2. de-spike: everything in the rim, only isolated outliers inside
    if despike_sd is not None:
        filled = _fill_outside(chi, support, support, vox, texture_mm=0.0)
        med = ndi.median_filter(filled, size=3)
        r = chi - med
        sd = _robust_std(r[support])
        out = support & (np.abs(r) > despike_sd * sd)
        rim = support & (ndi.distance_transform_edt(support, sampling=vox) <= despike_rim_mm)
        isolated = np.zeros_like(out)
        for sign in (1, -1):
            lab, n = ndi.label(out & (sign * r > 0), structure=np.ones((3, 3, 3)))
            if n:
                sizes = np.bincount(lab.ravel(), minlength=n + 1)
                small = sizes <= despike_max_size
                small[0] = False
                isolated |= small[lab]
        spikes = (out & rim) | isolated
        chi = np.where(spikes, med, chi)
        info["spike_fraction"] = float(spikes.sum() / support.sum())
        info["spike_fraction_rim"] = float((spikes & rim).sum() / max(rim.sum(), 1))
        info["spike_threshold_ppm"] = float(despike_sd * sd)

    # 3. fill the gap (and, smoothly, everything else, so the filters see no step at the edge)
    work = _fill_outside(chi, support, target, vox)
    region = ndi.binary_dilation(target | support, structure=_ball(3.0, vox))
    taper = ndi.gaussian_filter(region.astype(np.float64), 3.0 / vox)
    taper = np.clip(taper * 2.0, 0.0, 1.0)
    taper[region] = 1.0
    mu = float(np.mean(work[target])) if target.any() else 0.0
    work = (work - mu) * taper

    kr, dev = _cone_deviation(work.shape, vox, B0_dir)
    knyq = 0.5 / float(vox.max())
    edges = np.linspace(0.0, knyq, 41)

    def cone_ratio(F):
        P = np.abs(F) ** 2
        on = dev < 0.5 * cone_halfwidth_deg
        off = (dev > 15) & (dev < 35)
        kc, p_on, n_on = _radial_profile(np.where(on, P, 0.0), np.where(on, kr, -1.0), edges)
        _, p_off, n_off = _radial_profile(np.where(off, P, 0.0), np.where(off, kr, -1.0), edges)
        ratio = np.where((n_on > 10) & (n_off > 10) & (p_off > 0), p_on / np.maximum(p_off, 1e-300), 1.0)
        ratio[kc < 0.01] = 1.0
        band = (kc >= 0.02) & (kc <= 0.4) & (n_on > 10)
        med = float(10 * np.log10(np.median(np.maximum(ratio[band], 1e-12)))) if band.any() else 0.0
        mx = float(10 * np.log10(np.max(np.maximum(ratio[band], 1e-12)))) if band.any() else 0.0
        return kc, ratio, med, mx

    if cone not in ("clamp", "equalise", None):
        raise ValueError(f"cone must be 'clamp', 'equalise' or None, got {cone!r}")
    if cone is not None:
        w_ang = np.where(dev < cone_halfwidth_deg, 0.5 * (1 + np.cos(np.pi * dev / cone_halfwidth_deg)), 0.0)
        w_ang = np.where(dev < 0.5 * cone_halfwidth_deg, 1.0, w_ang)
        # measured on the brain-masked map (what a metric sees); the clamp is applied to the
        # filled map and repeated, because re-masking spreads some cone power back
        _, _, med0, mx0 = cone_ratio(np.fft.fftn(np.where(target, work, 0.0)))
        info["cone_excess_db_median"], info["cone_excess_db_max"] = med0, mx0
        for _it in range(int(cone_iterations)):
            F = np.fft.fftn(work)
            kc, ratio, _m, _x = cone_ratio(F)
            if cone == "clamp":
                gain_shell = np.where(ratio > 1.0, 1.0 / np.sqrt(ratio), 1.0)
            else:
                gain_shell = np.clip(1.0 / np.sqrt(np.maximum(ratio, 1e-12)), 0.0, cone_max_gain)
            g_k = np.interp(kr, kc, gain_shell, left=1.0, right=gain_shell[-1])
            F *= 1.0 - (1.0 - g_k) * w_ang
            work = np.real(np.fft.ifftn(F))
            if _it < int(cone_iterations) - 1:
                work = _fill_outside(work, target, target, vox, texture_mm=0.0) * taper
        _, _, med1, mx1 = cone_ratio(np.fft.fftn(np.where(target, work, 0.0)))
        info["cone_excess_db_median_after"], info["cone_excess_db_max_after"] = med1, mx1

    if denoise == "wiener":
        F = np.fft.fftn(work)
        P = np.abs(F) ** 2
        kc, pr, cnt = _radial_profile(P, kr, np.linspace(0.0, float(kr.max()) + 1e-9, 61))
        fit = (kc >= 0.05) & (kc <= 0.25) & (cnt > 10) & (pr > 0)
        hi = (kc >= 0.85 * knyq) & (kc <= knyq) & (cnt > 10) & (pr > 0)
        if fit.sum() >= 3 and hi.any():
            b, a = np.polyfit(np.log10(kc[fit]), np.log10(pr[fit]), 1)
            s_model = lambda k: 10 ** (a + b * np.log10(np.maximum(k, 1e-3)))  # noqa: E731
            floor = float(np.median(pr[hi] - np.minimum(s_model(kc[hi]), pr[hi])))
            N = max(floor, 0.0) * denoise_strength
            S = s_model(kr)
            H = np.where(kr > 0, S / (S + N), 1.0)
            F *= H
            info["spectral_exponent"] = float(b)
            info["noise_floor_fraction"] = float(N / max(float(np.median(pr[hi])), 1e-300))
            k_half = (10 ** ((math.log10(N) - a) / b)) if N > 0 else float("inf")
            info["wiener_k_half_cyc_per_mm"] = float(k_half)
        else:
            info["spectral_exponent"] = float("nan")
        work = np.real(np.fft.ifftn(F))

    if denoise == "tv":
        work = _tv_denoise(work, tv_weight)
    elif denoise not in (None, "wiener"):
        raise ValueError(f"denoise must be 'wiener', 'tv' or None, got {denoise!r}")

    out = np.where(target, work + mu, 0.0)
    if reference in ("median", "mean"):
        ref = float(np.median(out[target]) if reference == "median" else np.mean(out[target]))
        out = np.where(target, out - ref, 0.0)
        info["reference_shift_ppm"] = ref
    elif reference is not None:
        raise ValueError(f"reference must be 'median', 'mean' or None, got {reference!r}")
    info["denoise"] = denoise
    info["cone"] = cone
    return out.astype(np.float32), info


# ---------------------------------------------------------------------------------------------
# the phantom


@dataclass
class Phantom:
    """A head phantom on one grid. Arrays are 3-D, ``voxel_size`` in mm, ``affine`` voxel to
    world. ``fov`` is the index box (per axis ``[start, stop)``) of the source field of view inside
    the (possibly extended) grid; :func:`simulate` crops its outputs to it."""
    chi: np.ndarray
    M0: np.ndarray
    R1: np.ndarray
    R2star: np.ndarray
    mask: np.ndarray
    seg: np.ndarray
    voxel_size: np.ndarray
    affine: np.ndarray
    fov: list = None
    filled: Optional[np.ndarray] = None
    source_labels: Optional[np.ndarray] = None
    sources: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    FILES = {
        "chi": "chimodel/ChiModel.nii.gz", "M0": "maps/M0.nii.gz", "R1": "maps/R1.nii.gz",
        "R2star": "maps/R2star.nii.gz", "mask": "masks/BrainMask.nii.gz",
        "seg": "masks/SegmentedModel.nii.gz", "filled": "masks/FilledMask.nii.gz",
        "source_labels": "masks/SourceLabels.nii.gz",
    }

    def __post_init__(self):
        self.voxel_size = _vox(self.voxel_size)
        self.affine = np.asarray(self.affine, float)
        if self.fov is None:
            self.fov = [[0, int(n)] for n in self.chi.shape]

    def copy(self) -> "Phantom":
        return copy.deepcopy(self)

    def save(self, out_dir: str) -> str:
        """Write the phantom in the qsm-forward data layout plus ``phantom.json``."""
        dtypes = {"chi": np.float32, "M0": np.float32, "R1": np.float32, "R2star": np.float32,
                  "mask": np.uint8, "seg": np.int16, "filled": np.uint8, "source_labels": np.int16}
        for key, rel in self.FILES.items():
            arr = getattr(self, key)
            if arr is None:
                continue
            path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img = nib.Nifti1Image(np.asarray(arr).astype(dtypes[key]), self.affine)
            img.header.set_zooms(tuple(float(v) for v in self.voxel_size))
            nib.save(img, path)
        meta = {"fov": self.fov, "voxel_size": self.voxel_size.tolist(), "sources": self.sources,
                "labels": {str(k): v for k, v in LABEL_NAMES.items()}, **self.meta}
        with open(os.path.join(out_dir, "phantom.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=_json_default)
        self.meta["saved_to"] = os.path.abspath(out_dir)
        return out_dir

    @classmethod
    def load(cls, phantom_dir: str) -> "Phantom":
        with open(os.path.join(phantom_dir, "phantom.json")) as fh:
            meta = json.load(fh)
        arrays = {}
        affine = None
        for key, rel in cls.FILES.items():
            path = os.path.join(phantom_dir, rel)
            if os.path.exists(path):
                img = nib.load(path)
                affine = img.affine if affine is None else affine
                arrays[key] = np.asanyarray(img.dataobj)
                if key in ("mask", "filled"):
                    arrays[key] = arrays[key].astype(bool)
                elif key in ("seg", "source_labels"):
                    arrays[key] = arrays[key].astype(np.int16)
                else:
                    arrays[key] = arrays[key].astype(np.float32)
        fov = meta.pop("fov")
        vox = meta.pop("voxel_size")
        sources = meta.pop("sources", [])
        meta.pop("labels", None)
        meta["saved_to"] = os.path.abspath(phantom_dir)
        return cls(voxel_size=vox, affine=affine, fov=fov, sources=sources, meta=meta, **arrays)

    def tissue_params(self, out_dir: Optional[str] = None) -> TissueParams:
        """TissueParams pointing at the saved phantom (saving it to ``out_dir`` first if given)."""
        if out_dir is not None:
            self.save(out_dir)
        d = self.meta.get("saved_to")
        if d is None:
            raise ValueError("save the phantom first (Phantom.save or tissue_params(out_dir))")
        return TissueParams(root_dir=d, chi=self.FILES["chi"], M0=self.FILES["M0"], R1=self.FILES["R1"],
                            R2star=self.FILES["R2star"], mask=self.FILES["mask"], seg=self.FILES["seg"],
                            v1=None, raw_field=None, high_grad=None, white_matter_mask=None)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _inferior_axis(affine):
    """(array axis closest to world z, True if index 0 is the inferior end)."""
    a = np.asarray(affine, float)[:3, :3]
    ax = int(np.argmax(np.abs(a[2])))
    return ax, bool(a[2, ax] > 0)


def build_invivo_phantom(chi: np.ndarray, magnitude: np.ndarray, TEs: Sequence[float], voxel_size,
                         brain_mask: np.ndarray, affine: Optional[np.ndarray] = None,
                         chi_support: Optional[np.ndarray] = None, TR: float = 0.035,
                         flip_angle: float = 15.0, B0_dir=(0.0, 0.0, 1.0),
                         seg: Optional[np.ndarray] = None, R2star: Optional[np.ndarray] = None,
                         clean: Optional[dict] = None, head: Optional[dict] = None,
                         neck_mm: float = 40.0, cut_extend_mm: float = 20.0, name: Optional[str] = None,
                         source: Optional[dict] = None) -> Phantom:
    """Build a phantom whose χ truth inside the brain is a cleaned in vivo χ map.

    Parameters
    ----------
    chi : ndarray
        In vivo χ map (ppm), e.g. a well-regularised reconstruction with a brain mask.
    magnitude : ndarray
        Multi-echo magnitude, echoes last, on the same grid.
    TEs : sequence of float
        Echo times (s).
    voxel_size : sequence of float
        Voxel size (mm).
    brain_mask : ndarray of bool
        Anatomical brain mask (e.g. HD-BET or BET); the truth is defined on it.
    affine : ndarray, optional
        Voxel-to-world affine of the source (diag(voxel_size) if None).
    chi_support : ndarray of bool, optional
        Where the source χ is valid; default its largest finite non-zero component, holes filled.
    TR, flip_angle : float
        Source protocol (s, degrees), to divide the steady-state factor out of M0.
    B0_dir : sequence of float
        B0 direction of the source reconstruction in voxel coordinates.
    seg : ndarray of int, optional
        A brain segmentation to carry into the phantom's seg inside the brain (label values are
        kept; zero inside the brain becomes :data:`LABEL_BRAIN`).
    R2star : ndarray, optional
        A measured R2* map to use instead of fitting the magnitude.
    clean : dict, optional
        Keyword arguments for :func:`clean_chi`.
    head : dict, optional
        Keyword arguments for :func:`head_model`.
    neck_mm : float
        Inferior extension (mm) so the FFT forward model does not see an air interface where the
        field of view cuts through the head. 0 disables it.
    cut_extend_mm : float
        The same for any other face of the field of view that the head crosses (more than 0.5 %
        of the face is not air), e.g. the vertex or the nose; 0 disables it.
    name, source : optional
        Recorded in the phantom's metadata.
    """
    vox = _vox(voxel_size)
    affine = np.diag(list(vox) + [1.0]) if affine is None else np.asarray(affine, float)
    mag = np.asarray(magnitude, np.float32)
    if mag.ndim == 3:
        mag = mag[..., None]
    if mag.shape[:3] != chi.shape:
        raise ValueError(f"magnitude {mag.shape[:3]} and chi {chi.shape} grids differ")
    brain = np.asarray(brain_mask, bool)
    if chi_support is None:
        # holes too: quantised maps (int16 x slope) have exact zeros inside the brain
        support = ndi.binary_fill_holes(_largest_cc(np.isfinite(chi) & (chi != 0)))
    else:
        support = np.asarray(chi_support, bool)
    target = brain | support

    chi_clean, clean_info = clean_chi(chi, support, target, vox, B0_dir=B0_dir, **(clean or {}))

    ax_inf, inf_at_zero = _inferior_axis(affine)
    axial = ax_inf
    labels = head_model(mag[..., 0], target, vox, axial_axis=axial, **(head or {}))

    if R2star is None and mag.shape[-1] >= 2:
        S0, r2s = fit_r2star(mag, TEs)
    else:
        S0 = mag[..., 0] * np.exp(np.asarray(R2star if R2star is not None else 0.0) * float(TEs[0]))
        r2s = np.asarray(R2star if R2star is not None else np.zeros(chi.shape), np.float32)

    seg_full = labels.copy()
    if seg is not None:
        s = np.asarray(seg).astype(np.int16)
        seg_full[target] = np.where(s[target] > 0, s[target], LABEL_BRAIN)

    R1 = np.zeros(chi.shape, np.float32)
    R1[labels == LABEL_BRAIN] = R1_BRAIN
    for lab, r1 in R1_BY_LABEL.items():
        R1[target & (seg_full == lab)] = r1
    R1[labels == LABEL_SOFT_TISSUE] = R1_SOFT_TISSUE
    R1[labels == LABEL_BONE] = R1_BONE
    ss = np.where(R1 > 0, spgr_factor(np.where(R1 > 0, R1, 1.0), TR, flip_angle), 1.0)
    M0 = np.where(R1 > 0, S0 / ss, 0.0).astype(np.float32)
    m0_brain = float(np.median(M0[labels == LABEL_BRAIN]))
    r2 = r2s.copy()
    br = labels == LABEL_BRAIN
    st = labels == LABEL_SOFT_TISSUE
    bo = labels == LABEL_BONE
    r2[br] = np.clip(r2s[br], 5.0, 300.0)
    r2[st] = np.clip(r2s[st], 5.0, 200.0)
    r2[bo] = np.clip(r2s[bo], 100.0, 500.0)
    M0[bo] = np.minimum(M0[bo], 0.15 * m0_brain)
    M0 = np.minimum(M0, 4.0 * float(np.percentile(M0[br], 99.5)))  # inflow / fat outliers
    air = labels == LABEL_AIR
    M0[air] = 0.0
    r2[air] = 0.0

    chi_full = np.zeros(chi.shape, np.float32)
    chi_full[target] = chi_clean[target]
    chi_full[st] = CHI_SOFT_TISSUE_PPM
    chi_full[bo] = CHI_BONE_PPM
    chi_full[air] = CHI_AIR_PPM

    fov = [[0, int(n)] for n in chi.shape]
    arrays = dict(chi=chi_full, M0=M0, R1=R1, R2star=r2.astype(np.float32), mask=target, seg=seg_full,
                  filled=target & ~support)
    n_ext = int(round(neck_mm / vox[ax_inf])) if neck_mm and neck_mm > 0 else 0
    extended = {}
    for ax in range(3):
        for at_zero in (True, False):
            if ax == ax_inf and at_zero == inf_at_zero:
                n = n_ext
            else:
                # any other face the head crosses (top of the head, nose, a narrow in-plane FOV)
                face = arrays["seg"].take(0 if at_zero else -1, axis=ax)
                cut = float(np.mean(face != LABEL_AIR)) > 0.005
                n = int(round(cut_extend_mm / vox[ax])) if cut and cut_extend_mm > 0 else 0
            if n:
                arrays, affine, fov = _extend_face(arrays, affine, ax, at_zero, n, fov)
                extended[f"{ax}{'-' if at_zero else '+'}"] = n

    meta = {"name": name, "kind": "invivo", "source": source or {}, "clean": clean_info,
            "neck_mm": float(neck_mm), "neck_slices": n_ext, "inferior_axis": ax_inf,
            "extended_faces": extended, "cut_extend_mm": float(cut_extend_mm),
            "TEs_source": [float(t) for t in TEs], "TR_source": float(TR), "flip_angle_source": float(flip_angle),
            "B0_dir_source": [float(b) for b in B0_dir],
            "class_volumes_ml": {LABEL_NAMES[l]: float((arrays["seg"] == l).sum() * np.prod(vox) / 1000.0)
                                 for l in (LABEL_BRAIN, LABEL_SOFT_TISSUE, LABEL_BONE, LABEL_AIR)},
            "chi_values_ppm": {"air": CHI_AIR_PPM, "bone": CHI_BONE_PPM, "soft_tissue": CHI_SOFT_TISSUE_PPM},
            "R1_values": {"brain": R1_BRAIN, "soft_tissue": R1_SOFT_TISSUE, "bone": R1_BONE}}
    return Phantom(voxel_size=vox, affine=affine, fov=fov, meta=meta, **arrays)


def _extend_face(arrays, affine, ax, at_zero, n, fov):
    """Replicate the edge slice of one face ``n`` times; brain there becomes soft tissue."""
    pad = [(0, 0)] * 3
    pad[ax] = (n, 0) if at_zero else (0, n)
    out = {}
    for k, a in arrays.items():
        if a is None:
            out[k] = None
            continue
        if k in ("mask", "filled"):
            out[k] = np.pad(a, pad, mode="constant", constant_values=False)
        else:
            out[k] = np.pad(a, pad, mode="edge")
    ext = np.zeros(out["chi"].shape, bool)
    sl = [slice(None)] * 3
    sl[ax] = slice(0, n) if at_zero else slice(-n, None)
    ext[tuple(sl)] = True
    brain_ext = ext & (out["seg"] != LABEL_AIR) & (out["seg"] != LABEL_BONE) & (out["seg"] != LABEL_SOFT_TISSUE)
    st = arrays["seg"] == LABEL_SOFT_TISSUE
    m0_st = float(np.median(arrays["M0"][st])) if st.any() else float(np.median(arrays["M0"][arrays["mask"]]))
    r2_st = float(np.median(arrays["R2star"][st])) if st.any() else 30.0
    out["seg"][brain_ext] = LABEL_SOFT_TISSUE
    out["chi"][brain_ext] = CHI_SOFT_TISSUE_PPM
    out["M0"][brain_ext] = m0_st
    out["R2star"][brain_ext] = r2_st
    out["R1"][brain_ext] = R1_SOFT_TISSUE
    if at_zero:
        affine = affine.copy()
        affine[:3, 3] -= n * affine[:3, ax]
        fov = [list(f) for f in fov]
        fov[ax] = [fov[ax][0] + n, fov[ax][1] + n]
    return out, affine, fov


# ---------------------------------------------------------------------------------------------
# strong sources and pathology

KINDS = ("calcification", "haemorrhage", "microbleed", "ms_lesion", "vein")

# Per-kind defaults. χ is absolute (replaces tissue) for calcification, haemorrhage and microbleed,
# and a change added to tissue for ms_lesion (rim; core in core_chi_ppm) and vein.
KIND_DEFAULTS = {
    # physiological/pathological calcification: strongly diamagnetic, low proton density
    "calcification": dict(radius_mm=(1.0, 3.5), chi_ppm=(-3.0, -1.0), m0_scale=(0.1, 0.3), region="brain"),
    # intraparenchymal haemorrhage (deoxy/methaemoglobin, haemosiderin): irregular blob
    "haemorrhage": dict(radius_mm=(3.0, 9.0), chi_ppm=(0.5, 2.0), m0_scale=(0.7, 1.0), region="brain",
                        irregularity=0.3),
    # cerebral microbleed: 2-10 mm blooming in GRE, a 1-2.5 mm haemosiderin core
    "microbleed": dict(radius_mm=(0.8, 2.5), chi_ppm=(0.8, 2.5), m0_scale=(0.6, 1.0), region="brain"),
    # MS-like lesion: elongated, slightly paramagnetic core, paramagnetic rim (PRL) on some,
    # central vein on some, T2*-hyperintense core
    "ms_lesion": dict(radius_mm=(2.0, 7.0), chi_ppm=(0.04, 0.10), core_chi_ppm=(-0.01, 0.02),
                      m0_scale=(1.05, 1.2), region="wm", rim_mm=1.0, rim_probability=0.5,
                      central_vein_probability=0.5, elongation=(1.0, 1.8)),
    # enlarged/engorged vein: tube of deoxygenated blood
    "vein": dict(radius_mm=(0.8, 2.0), chi_ppm=(0.3, 0.6), m0_scale=(0.9, 1.0), region="brain",
                 length_mm=(20.0, 60.0)),
}


@dataclass
class SourceSpec:
    """One kind of source to insert. ``None`` fields take :data:`KIND_DEFAULTS`; ranges are
    ``(low, high)`` drawn uniformly per source (a scalar fixes the value).

    region: 'brain' (anywhere inside, away from the edge), 'deep' (>= 20 mm from the brain
    surface), 'cortical' (within 6 mm of the surface, still fully inside), 'wm' (seg label 8 if the
    phantom has it, else deep diamagnetic tissue), 'dgm' (seg labels 1-6, else deep paramagnetic
    tissue) or a boolean array of allowed centres.
    """
    kind: str
    count: int = 1
    radius_mm: Optional[Union[float, tuple]] = None
    chi_ppm: Optional[Union[float, tuple]] = None
    region: Optional[Union[str, np.ndarray]] = None
    margin_mm: float = 1.5
    gap_mm: float = 2.0
    m0_scale: Optional[Union[float, tuple]] = None
    core_chi_ppm: Optional[Union[float, tuple]] = None
    rim_mm: Optional[float] = None
    rim_probability: Optional[float] = None
    central_vein_probability: Optional[float] = None
    elongation: Optional[Union[float, tuple]] = None
    irregularity: Optional[float] = None
    length_mm: Optional[Union[float, tuple]] = None
    r2star_per_ppm: float = DR_KERNEL

    def resolved(self) -> "SourceSpec":
        if self.kind not in KIND_DEFAULTS:
            raise ValueError(f"unknown source kind {self.kind!r}; one of {KINDS}")
        out = copy.copy(self)
        for k, v in KIND_DEFAULTS[self.kind].items():
            if getattr(out, k) is None:
                setattr(out, k, v)
        return out


PRESETS = {
    # many strong sources of both signs: calibrates streaking labels
    "streak-heavy": [
        SourceSpec("calcification", count=6, radius_mm=(1.5, 3.5), chi_ppm=(-3.0, -1.5)),
        SourceSpec("haemorrhage", count=3, radius_mm=(4.0, 8.0), chi_ppm=(1.0, 2.5)),
        SourceSpec("microbleed", count=15),
        SourceSpec("vein", count=4, radius_mm=(1.5, 2.5), chi_ppm=(0.4, 0.6)),
    ],
    "calcification": [SourceSpec("calcification", count=4)],
    "microbleeds": [SourceSpec("microbleed", count=25)],
    "haemorrhage": [SourceSpec("haemorrhage", count=1, radius_mm=(6.0, 12.0))],
    "ms": [SourceSpec("ms_lesion", count=12)],
    "mixed-mild": [
        SourceSpec("calcification", count=2, radius_mm=(1.0, 2.0), chi_ppm=(-1.5, -0.8)),
        SourceSpec("microbleed", count=5),
        SourceSpec("ms_lesion", count=4),
    ],
}


def _draw(rng, v):
    if v is None:
        return None
    if np.isscalar(v):
        return float(v)
    lo, hi = v
    return float(rng.uniform(lo, hi))


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def _bbox(center_vox, reach_mm, vox, shape):
    r = np.ceil(reach_mm / vox).astype(int) + 1
    c = np.round(center_vox).astype(int)
    lo = np.maximum(c - r, 0)
    hi = np.minimum(c + r + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _coords_mm(box, vox):
    g = np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij")
    return np.stack([gi * v for gi, v in zip(g, vox)], -1)


def _occupancy_ellipsoid(box, center_mm, radii_mm, R, vox, rng=None, irregularity=0.0):
    """Partial-volume occupancy (0..1) of a (possibly irregular) ellipsoid on a box."""
    x = _coords_mm(box, vox) - center_mm
    u = x @ R  # into the ellipsoid frame
    rho = np.sqrt(((u / radii_mm) ** 2).sum(-1))
    r_mean = float(np.mean(radii_mm))
    d = (rho - 1.0) * r_mean  # approx. signed distance (mm)
    if irregularity and rng is not None:
        noise = ndi.gaussian_filter(rng.normal(size=rho.shape), max(r_mean / 2.5, 1.0) / vox)
        noise /= max(float(np.std(noise)), 1e-9)
        d = d - irregularity * r_mean * noise * np.exp(-np.maximum(rho - 1, 0))
    return np.clip(0.5 - d / float(np.mean(vox)), 0.0, 1.0)


def _occupancy_tube(box, points_mm, radius_mm, vox):
    x = _coords_mm(box, vox).reshape(-1, 3)
    d, _ = cKDTree(points_mm).query(x)
    return np.clip(0.5 - (d.reshape(_coords_mm(box, vox).shape[:3]) - radius_mm) / float(np.mean(vox)), 0.0, 1.0)


def _vein_path(rng, start_mm, length_mm, allowed, vox, step_mm=0.25, persistence=0.97):
    """A smooth random walk inside ``allowed`` (stops at its edge)."""
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    pts = [np.asarray(start_mm, float)]
    shape = np.array(allowed.shape)
    for _ in range(int(length_mm / step_mm)):
        d = persistence * d + (1 - persistence) * 3.0 * rng.normal(size=3) * step_mm
        d /= np.linalg.norm(d)
        p = pts[-1] + d * step_mm
        iv = np.round(p / vox).astype(int)
        if np.any(iv < 0) or np.any(iv >= shape) or not allowed[tuple(iv)]:
            break
        pts.append(p)
    return np.array(pts)


def _region_mask(ph: Phantom, region, dist_mm):
    if isinstance(region, np.ndarray):
        return region.astype(bool) & ph.mask
    chi_s = ndi.gaussian_filter(np.where(ph.mask, ph.chi, 0.0), 1.5 / ph.voxel_size)
    inner = ph.mask & (dist_mm > 0)
    if region == "brain":
        return ph.mask
    if region == "deep":
        return ph.mask & (dist_mm >= 20.0)
    if region == "cortical":
        return ph.mask & (dist_mm <= 9.0)
    if region == "wm":
        if np.any(ph.seg == 8):
            return ph.seg == 8
        return inner & (dist_mm >= 10.0) & (chi_s < np.percentile(chi_s[inner], 30))
    if region == "dgm":
        if np.any((ph.seg >= 1) & (ph.seg <= 6)):
            return (ph.seg >= 1) & (ph.seg <= 6)
        deep = inner & (dist_mm >= 20.0)
        return deep & (chi_s > np.percentile(chi_s[deep], 90))
    raise ValueError(f"unknown region {region!r}")


def insert_sources(phantom: Phantom, specs: Union[str, Sequence[SourceSpec]], seed: int = 0,
                   max_tries: int = 2000) -> Phantom:
    """Insert strong sources / pathology into a copy of ``phantom``.

    ``specs`` is a list of :class:`SourceSpec` or a :data:`PRESETS` name. Every source changes χ
    (replaced for calcification, haemorrhage, microbleed; added for ms_lesion and vein), M0
    (``m0_scale``) and R2* (+ ``r2star_per_ppm`` x |Δχ|, so strong sources become signal voids at
    late echoes) with partial-volume weights, and gets an instance id in
    ``phantom.source_labels`` (voxels with occupancy >= 0.5). ``phantom.sources`` lists each
    source: id, kind, centre (voxel and mm), radius, χ, volume (mL) and moment (ppm·mL, Δχ
    against the tissue it replaced, integrated over the occupancy). Calcifications get seg label
    16, haemorrhages 18, microbleeds 19, lesion cores 20, rims 21, veins 11. Sources do not
    overlap (``gap_mm``) and stay ``margin_mm`` inside the brain mask. Raises if a source cannot be
    placed in ``max_tries`` draws.
    """
    if isinstance(specs, str):
        if specs not in PRESETS:
            raise KeyError(f"unknown preset {specs!r}; one of {sorted(PRESETS)}")
        preset = specs
        specs = PRESETS[specs]
    else:
        preset = None
    ph = phantom.copy()
    rng = np.random.default_rng(seed)
    vox = ph.voxel_size
    shape = np.array(ph.chi.shape)
    dist = ndi.distance_transform_edt(ph.mask, sampling=vox)
    labels = np.zeros(ph.chi.shape, np.int16) if ph.source_labels is None else ph.source_labels.copy()
    occupied = labels > 0
    next_id = int(labels.max()) + 1
    chi, M0, R2 = ph.chi.astype(np.float64), ph.M0.astype(np.float64), ph.R2star.astype(np.float64)
    seg = ph.seg.copy()
    records = list(ph.sources)
    # (centre mm, reach mm) of everything already placed, for the no-overlap test
    placed = []
    for r in records:
        placed.append((np.asarray(r["center_mm"], float), float(r.get("reach_mm", r.get("radius_mm", 0.0)))))
        for p in r.get("path_mm", []):
            placed.append((np.asarray(p, float), float(r.get("radius_mm", 0.0))))

    def apply(box, f, mode, value, m0s, r2pp, seg_label, rec_id):
        c, m, r = chi[box], M0[box], R2[box]
        old = c.copy()
        if mode == "replace":
            new = (1 - f) * c + f * value
        else:
            new = c + f * value
        chi[box] = new
        M0[box] = m * (1 - f + f * m0s)
        R2[box] = r + r2pp * np.abs(new - old)
        core = f >= 0.5
        labels[box][core] = rec_id
        occupied[box] |= f > 0.0
        if seg_label is not None:
            seg[box][core] = seg_label
        return float(np.sum(new - old) * np.prod(vox) / 1000.0), float(np.sum(f) * np.prod(vox) / 1000.0)

    for spec in specs:
        s = spec.resolved()
        allowed_region = _region_mask(ph, s.region, dist)
        for _ in range(int(s.count)):
            radius = _draw(rng, s.radius_mm)
            reach = radius * (max(s.elongation) if isinstance(s.elongation, tuple) else (s.elongation or 1.0))
            need = reach + s.margin_mm
            cand_idx = np.argwhere(allowed_region & (dist >= need))
            if len(cand_idx) == 0:
                raise RuntimeError(f"no room for a {s.kind} of radius {radius:.1f} mm in region {s.region!r}")
            for _try in range(max_tries):
                c_vox = cand_idx[rng.integers(len(cand_idx))].astype(float) + rng.uniform(-0.5, 0.5, 3)
                c_vox = np.clip(c_vox, 0, shape - 1)
                if occupied[tuple(np.round(c_vox).astype(int))]:
                    continue
                if all(np.linalg.norm(c_vox * vox - pc) >= reach + pr + s.gap_mm for pc, pr in placed):
                    break
            else:
                raise RuntimeError(f"could not place {s.kind} #{_ + 1} in {max_tries} draws (lower count or gap_mm)")
            c_mm = c_vox * vox
            tissue_chi = float(np.median(chi[_bbox(c_vox, radius, vox, shape)][ph.mask[_bbox(c_vox, radius, vox, shape)]]))
            rec = {"id": next_id, "kind": s.kind, "center_vox": c_vox.round(2).tolist(),
                   "center_mm": c_mm.round(2).tolist(), "radius_mm": round(radius, 3),
                   "reach_mm": round(reach, 3), "tissue_chi_ppm": round(tissue_chi, 4)}
            m0s = _draw(rng, s.m0_scale)
            if s.kind in ("calcification", "haemorrhage", "microbleed"):
                value = _draw(rng, s.chi_ppm)
                R = _random_rotation(rng)
                radii = radius * np.array([1.0, rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.0)])
                irr = s.irregularity or 0.0
                box = _bbox(c_vox, radius * (1 + 2 * irr) + 2 * vox.max(), vox, shape)
                f = _occupancy_ellipsoid(box, c_mm, radii, R, vox, rng, irr) * ph.mask[box]
                seg_label = {"calcification": LABEL_CALCIFICATION, "haemorrhage": LABEL_HAEMORRHAGE,
                             "microbleed": LABEL_MICROBLEED}[s.kind]
                moment, vol = apply(box, f, "replace", value, m0s, s.r2star_per_ppm, seg_label, next_id)
                rec.update(chi_ppm=round(value, 4), radii_mm=radii.round(3).tolist())
            elif s.kind == "ms_lesion":
                elong = _draw(rng, s.elongation)
                R = _random_rotation(rng)
                radii = radius * np.array([elong, 1.0, rng.uniform(0.8, 1.0)])
                box = _bbox(c_vox, radius * elong + s.rim_mm + 2 * vox.max(), vox, shape)
                core_d = _draw(rng, s.core_chi_ppm)
                f_core = _occupancy_ellipsoid(box, c_mm, radii, R, vox) * ph.mask[box]
                moment, vol = apply(box, f_core, "add", core_d, m0s, s.r2star_per_ppm, LABEL_LESION, next_id)
                rec.update(core_delta_chi_ppm=round(core_d, 4), radii_mm=radii.round(3).tolist(), elongation=round(elong, 3))
                if rng.uniform() < s.rim_probability:
                    rim_d = _draw(rng, s.chi_ppm)
                    f_out = _occupancy_ellipsoid(box, c_mm, radii + s.rim_mm, R, vox) * ph.mask[box]
                    f_rim = np.clip(f_out - f_core, 0.0, 1.0)
                    m_r, _v = apply(box, f_rim, "add", rim_d, 1.0, s.r2star_per_ppm, LABEL_LESION_RIM, next_id)
                    labels[box][(f_rim >= 0.5)] = next_id
                    moment += m_r
                    rec.update(rim_delta_chi_ppm=round(rim_d, 4), rim_mm=s.rim_mm)
                if rng.uniform() < s.central_vein_probability:
                    axis = R[:, 0]
                    L = radius * elong * 1.6
                    pts = c_mm + np.outer(np.linspace(-L, L, int(2 * L / 0.25) + 1), axis)
                    vb = _bbox(c_vox, L + 2 * vox.max(), vox, shape)
                    f_v = _occupancy_tube(vb, pts, 0.5, vox) * ph.mask[vb]
                    m_v, _v = apply(vb, f_v, "add", 0.35, 0.95, s.r2star_per_ppm, None, next_id)
                    moment += m_v
                    rec.update(central_vein=True)
            elif s.kind == "vein":
                value = _draw(rng, s.chi_ppm)
                length = _draw(rng, s.length_mm)
                ok_path = ph.mask & (dist >= radius + s.margin_mm)
                pts = _vein_path(rng, c_mm, length, ok_path, vox)
                # extend backwards too, so the start is not always an end
                pts = np.concatenate([_vein_path(rng, c_mm, length / 2, ok_path, vox)[::-1], pts], 0)
                lo = np.floor(pts.min(0) / vox - radius / vox - 2).astype(int)
                hi = np.ceil(pts.max(0) / vox + radius / vox + 3).astype(int)
                box = tuple(slice(int(max(a, 0)), int(min(b, n))) for a, b, n in zip(lo, hi, shape))
                f = _occupancy_tube(box, pts, radius, vox) * ph.mask[box]
                moment, vol = apply(box, f, "add", value, m0s, s.r2star_per_ppm, LABEL_VEIN, next_id)
                path = pts[:: max(int(2.0 / 0.25), 1)]
                rec.update(delta_chi_ppm=round(value, 4), length_mm=round(float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))), 2),
                           path_mm=path.round(2).tolist())
                placed.extend((p, radius) for p in path)
            rec.update(volume_ml=round(vol, 5), moment_ppm_ml=round(moment, 6), m0_scale=round(m0s, 3))
            records.append(rec)
            placed.append((c_mm, reach))
            next_id += 1

    ph.chi = chi.astype(np.float32)
    ph.M0 = M0.astype(np.float32)
    ph.R2star = np.clip(R2, 0.0, 2000.0).astype(np.float32)
    ph.seg = seg
    ph.source_labels = labels
    ph.sources = records
    ph.meta = dict(ph.meta)
    ph.meta.setdefault("insertions", []).append(
        {"seed": int(seed), "preset": preset, "specs": [asdict(sp) for sp in specs]})
    ph.meta.pop("saved_to", None)
    return ph


def source_kind_map(phantom: Phantom) -> np.ndarray:
    """Per-voxel kind index (1-based position in :data:`KINDS`, 0 = none) from the instance labels."""
    out = np.zeros(phantom.chi.shape, np.int16)
    if phantom.source_labels is None:
        return out
    lut = np.zeros(int(phantom.source_labels.max()) + 1, np.int16)
    for r in phantom.sources:
        lut[r["id"]] = KINDS.index(r["kind"]) + 1
    return lut[phantom.source_labels]


# ---------------------------------------------------------------------------------------------
# streak labels


def cone_restricted_error(recon: np.ndarray, truth: np.ndarray, mask: np.ndarray, voxel_size,
                          B0_dir=(0.0, 0.0, 1.0), halfwidth_deg: float = 6.0,
                          kmin: float = 0.0):
    """Reconstruction error restricted to the magic-angle cone: the exact streak label.

    Both maps are demeaned in ``mask`` and the error is zeroed outside it; its k-space is kept
    within ``halfwidth_deg`` of the cone (raised-cosine taper over the second half) for
    |k| >= ``kmin`` (cycles/mm). Returns ``(streak_map, fraction)`` where ``fraction`` is the
    share of the error energy in the cone band; the map is zero outside ``mask``.
    """
    m = np.asarray(mask, bool)
    e = np.where(m, (recon - recon[m].mean()) - (truth - truth[m].mean()), 0.0)
    E = np.fft.fftn(e)
    kr, dev = _cone_deviation(e.shape, voxel_size, B0_dir)
    w = np.where(dev < halfwidth_deg, 0.5 * (1 + np.cos(np.pi * np.clip(2 * dev / halfwidth_deg - 1, 0, 1))), 0.0)
    w[kr < kmin] = 0.0
    s = np.real(np.fft.ifftn(E * w))
    tot = float(np.sum(np.abs(E) ** 2))
    frac = float(np.sum(np.abs(E * w) ** 2) / tot) if tot > 0 else 0.0
    return np.where(m, s, 0.0).astype(np.float32), frac


# ---------------------------------------------------------------------------------------------
# simulation


def simulate(phantom: Union[Phantom, str], recon_params: ReconParams, bids_dir: str,
             phantom_dir: Optional[str] = None, crop: bool = True, **generate_bids_kwargs) -> dict:
    """Forward-simulate a phantom into a BIDS directory with the true fields saved.

    Saves the phantom to ``phantom_dir`` if it is not saved yet, runs
    :func:`qsm_forward.generate_bids` with ``save_field=True`` (true total field
    ``*_fieldmap.nii`` and true local field ``*_fieldmap-local.nii``) and
    ``save_shimmed_field=True`` unless overridden, writes the source labels as
    ``*_desc-sources_dseg.nii`` and the phantom metadata as ``*_phantom.json`` in the derivatives,
    and with ``crop`` cuts every output back to the source field of view (``phantom.fov``).
    Returns a dict of the main output paths.
    """
    ph = Phantom.load(phantom) if isinstance(phantom, str) else phantom
    if ph.meta.get("saved_to") is None or phantom_dir is not None:
        if phantom_dir is None:
            raise ValueError("phantom_dir is required for an unsaved phantom")
        ph.save(phantom_dir)
    tp = ph.tissue_params()
    kw = dict(save_field=True, save_shimmed_field=True)
    kw.update(generate_bids_kwargs)
    generate_bids(tp, recon_params, bids_dir, **kw)

    rn = f"sub-{recon_params.subject}"
    if recon_params.session:
        rn += f"_ses-{recon_params.session}"
    if recon_params.acq:
        rn += f"_acq-{recon_params.acq}"
    if recon_params.run:
        rn += f"_run-{recon_params.run}"
    sub = os.path.join(bids_dir, f"sub-{recon_params.subject}")
    der = os.path.join(bids_dir, "derivatives", "qsm-forward", f"sub-{recon_params.subject}")
    if recon_params.session:
        sub = os.path.join(sub, f"ses-{recon_params.session}")
        der = os.path.join(der, f"ses-{recon_params.session}")
    anat, danat = os.path.join(sub, "anat"), os.path.join(der, "anat")

    grid = ReconGrid(tissue_shape=ph.chi.shape, tissue_voxel_size=ph.voxel_size,
                     voxel_size=recon_params.voxel_size, slice_gap=getattr(recon_params, "slice_gap", 0.0),
                     slice_axis=getattr(recon_params, "slice_axis", 2))
    if ph.source_labels is not None:
        ref = nib.load(os.path.join(danat, f"{rn}_dseg.nii"))
        lab = grid.resample_map(nib.Nifti1Image(ph.source_labels.astype(np.float32), tp.nii_affine), "nearest")
        nib.save(nib.Nifti1Image(np.round(lab.get_fdata()).astype(np.int16), ref.affine, ref.header),
                 os.path.join(danat, f"{rn}_desc-sources_dseg.nii"))
    with open(os.path.join(danat, f"{rn}_phantom.json"), "w") as fh:
        json.dump({"sources": ph.sources, **{k: v for k, v in ph.meta.items() if k != "saved_to"},
                   "phantom_dir": ph.meta.get("saved_to")}, fh, indent=1, default=_json_default)

    full = tuple(int(n) for n in ph.chi.shape)
    needs_crop = crop and any(f != [0, n] for f, n in zip(ph.fov, full))
    if needs_crop:
        ratio = np.array(grid.shape, float) / np.array(full, float)
        box = tuple(slice(int(round(lo * r)), int(round(hi * r))) for (lo, hi), r in zip(ph.fov, ratio))
        for path in sorted(glob.glob(os.path.join(anat, f"{rn}*.nii")) + glob.glob(os.path.join(danat, f"{rn}*.nii"))):
            img = nib.load(path, mmap=False)
            if tuple(img.shape[:3]) != tuple(grid.shape):
                continue
            data = np.array(np.asanyarray(img.dataobj)[box])  # a copy: the file is overwritten below
            new_aff = img.affine.copy()
            new_aff[:3, 3] = img.affine[:3, :3] @ np.array([b.start for b in box]) + img.affine[:3, 3]
            out = nib.Nifti1Image(data, new_aff, img.header)
            out.header["descrip"] = img.header["descrip"]
            nib.save(out, path)
    return {"bids_dir": bids_dir, "anat": anat, "derivatives": danat, "recon_name": rn,
            "chi": os.path.join(danat, f"{rn}_Chimap.nii"), "mask": os.path.join(danat, f"{rn}_mask.nii"),
            "totalfield": os.path.join(danat, f"{rn}_fieldmap.nii"),
            "localfield": os.path.join(danat, f"{rn}_fieldmap-local.nii"),
            "cropped": bool(needs_crop)}


# ---------------------------------------------------------------------------------------------
# simulating on a grid other than the truth's (no inverse crime)


def _affine_zoom(a, factor, shape, order):
    """Resample ``a`` onto a grid ``factor`` times finer, origin-aligned: output index ``o`` reads
    input position ``o / factor`` (the convention of qsm-forward's ``resize`` and k-space cropping:
    voxel 0 of both grids has the same centre)."""
    return ndi.affine_transform(np.asarray(a, np.float32), np.full(3, 1.0 / factor), offset=0.0,
                                output_shape=tuple(shape), order=order, mode="nearest",
                                prefilter=order > 1).astype(np.float32)


def upsample_phantom(phantom: Phantom, factor: float = 1.5, chi_order: int = 3,
                     label_sigma_vox: float = 0.6) -> Phantom:
    """The phantom on a grid ``factor`` times finer, so a simulation there and a k-space crop
    back to the source resolution has partial-volume and sub-voxel effects that the truth's own
    grid cannot represent (the inverse crime of simulating on the truth's grid).

    - head classes (air, bone, soft tissue, brain): each class indicator is smoothed by
      ``label_sigma_vox`` source voxels, interpolated linearly and the most likely class taken, so
      the air, bone and brain boundaries are smooth curves at the fine resolution rather than
      1 mm staircases;
    - χ inside the brain: cubic spline of the source map extended outward by its nearest brain
      value (no ringing at the edge); outside, the class values (air, bone, soft tissue);
    - brain labels, ``filled`` and source labels: nearest neighbour; M0 and R2*: linear, then air
      set to zero and bone clipped as in :func:`build_invivo_phantom`; R1 from the labels.

    Strong sources should be inserted after upsampling, so their edges have fine partial volume.
    ``factor * shape`` must be whole numbers on every axis (e.g. 1.5 with even shapes).
    """
    f = float(factor)
    ph = phantom
    shape = []
    for n in ph.chi.shape:
        m = n * f
        if abs(m - round(m)) > 1e-6:
            raise ValueError(f"factor {f} x shape {ph.chi.shape} is not a whole grid; pad the phantom first")
        shape.append(int(round(m)))
    vox = ph.voxel_size / f
    brain = ph.mask.astype(bool)
    head_lab = np.where(brain, LABEL_BRAIN, ph.seg)
    head_lab = np.where(np.isin(head_lab, (LABEL_AIR, LABEL_BONE, LABEL_SOFT_TISSUE, LABEL_BRAIN)), head_lab, LABEL_SOFT_TISSUE)
    classes = (LABEL_AIR, LABEL_BONE, LABEL_SOFT_TISSUE, LABEL_BRAIN)
    best = None
    cls = np.zeros(shape, np.int16)
    for c in classes:
        p = _affine_zoom(ndi.gaussian_filter((head_lab == c).astype(np.float32), label_sigma_vox), f, shape, 1)
        if best is None:
            best, cls[:] = p, c
        else:
            better = p > best
            cls[better] = c
            best = np.maximum(best, p)
    del best
    fbrain = cls == LABEL_BRAIN

    # brain values extended outward by their nearest brain voxel before interpolation
    _, idx = ndi.distance_transform_edt(~brain, return_indices=True)
    near = tuple(idx)
    del idx
    chi_b = ph.chi[near]
    seg_b = np.where(brain, ph.seg, 0)[near]
    del near
    chi = _affine_zoom(chi_b, f, shape, chi_order)
    del chi_b
    chi[cls == LABEL_AIR] = CHI_AIR_PPM
    chi[cls == LABEL_BONE] = CHI_BONE_PPM
    chi[cls == LABEL_SOFT_TISSUE] = CHI_SOFT_TISSUE_PPM
    seg_brain = _affine_zoom(seg_b, f, shape, 0).astype(np.int16)
    seg = np.where(fbrain, seg_brain, cls).astype(np.int16)
    del seg_brain

    M0 = np.maximum(_affine_zoom(ph.M0, f, shape, 1), 0.0)
    R2 = np.maximum(_affine_zoom(ph.R2star, f, shape, 1), 0.0)
    m0_brain = float(np.median(ph.M0[brain]))
    bone = cls == LABEL_BONE
    M0[bone] = np.minimum(M0[bone], 0.15 * m0_brain)
    R2[bone] = np.clip(R2[bone], 100.0, 500.0)
    air = cls == LABEL_AIR
    M0[air] = 0.0
    R2[air] = 0.0
    R1 = np.zeros(shape, np.float32)
    R1[fbrain] = R1_BRAIN
    for lab, r1 in R1_BY_LABEL.items():
        R1[fbrain & (seg == lab)] = r1
    R1[cls == LABEL_SOFT_TISSUE] = R1_SOFT_TISSUE
    R1[bone] = R1_BONE

    nearest = lambda a: None if a is None else _affine_zoom(a, f, shape, 0)  # noqa: E731
    filled = nearest(ph.filled)
    src = nearest(ph.source_labels)
    affine = ph.affine.copy()
    affine[:3, :3] = affine[:3, :3] / f
    meta = dict(ph.meta)
    meta.pop("saved_to", None)
    meta["upsampled"] = {"factor": f, "from_shape": list(ph.chi.shape), "from_voxel_size": ph.voxel_size.tolist(),
                         "chi_order": chi_order, "label_sigma_vox": label_sigma_vox}
    fov = [[int(round(lo * f)), int(round(hi * f))] for lo, hi in ph.fov]
    return Phantom(chi=chi, M0=M0, R1=R1, R2star=R2, mask=fbrain, seg=seg, voxel_size=vox, affine=affine,
                   fov=fov, filled=None if filled is None else filled > 0.5,
                   source_labels=None if src is None else src.astype(np.int16), sources=list(ph.sources), meta=meta)


def _pv_matrix(n_src: int, n_dst: int, ratio: float) -> np.ndarray:
    """Overlap weights (n_dst x n_src, rows sum to 1) of origin-aligned 1-D grids: destination voxel
    j spans [j r - r/2, j r + r/2] in source-voxel units, source voxel i spans [i - 1/2, i + 1/2]."""
    lo = np.arange(n_dst) * ratio - ratio / 2.0
    hi = lo + ratio
    s_lo = np.arange(n_src) - 0.5
    W = np.clip(np.minimum(hi[:, None], s_lo[None] + 1.0) - np.maximum(lo[:, None], s_lo[None]), 0.0, None)
    tot = W.sum(1, keepdims=True)
    return (W / np.where(tot > 0, tot, 1.0)).astype(np.float32)


def pv_resample(a: np.ndarray, ratio, shape) -> np.ndarray:
    """Partial-volume (box) average of ``a`` onto a coarser origin-aligned grid of ``shape``,
    ``ratio`` = destination voxel / source voxel per axis. Separable, so memory stays at about one
    copy of the source array."""
    out = np.asarray(a, np.float32)
    for ax in range(3):
        W = _pv_matrix(out.shape[ax], int(shape[ax]), float(np.atleast_1d(ratio)[ax] if np.ndim(ratio) else ratio))
        out = np.moveaxis(np.tensordot(W, np.moveaxis(out, ax, 0), axes=(1, 0)), 0, ax)
    return np.ascontiguousarray(out, np.float32)


def dipole_field(chi: np.ndarray, voxel_size, B0_dir=(0.0, 0.0, 1.0), pad_mm: float = 60.0,
                 pad_value: Optional[float] = None, mask: Optional[np.ndarray] = None, workers: int = 2) -> np.ndarray:
    """Field (ppm) of a χ map (ppm): F^-1[D(k) F[χ]] with D = 1/3 - (k·b)²/|k|², single precision.

    The volume is padded by ``pad_mm`` per side (to an FFT-friendly size) with ``pad_value``
    (default the corner voxel, i.e. outside air for a head phantom), which is what
    :func:`qsm_forward.generate_field` does with a 2x pad, at a fraction of its memory. With
    ``mask`` the mean over the mask is subtracted."""
    from scipy import fft as sfft
    vox = _vox(voxel_size)
    b = np.asarray(B0_dir, float)
    b = b / np.linalg.norm(b)
    n = np.array(chi.shape)
    shp = [sfft.next_fast_len(int(k + 2 * math.ceil(pad_mm / v)), real=True) for k, v in zip(n, vox)]
    pv = float(chi[-1, -1, -1]) if pad_value is None else float(pad_value)
    buf = np.full(shp, pv, np.float32)
    buf[:n[0], :n[1], :n[2]] = chi
    F = sfft.rfftn(buf, workers=workers)
    del buf
    kx = np.fft.fftfreq(shp[0], d=vox[0]).astype(np.float32)[:, None, None]
    ky = np.fft.fftfreq(shp[1], d=vox[1]).astype(np.float32)[None, :, None]
    kz = np.fft.rfftfreq(shp[2], d=vox[2]).astype(np.float32)[None, None, :]
    for i in range(shp[0]):  # plane by plane: no full-size k-space temporaries
        k2 = kx[i] ** 2 + ky[0] ** 2 + kz[0] ** 2
        kb = kx[i] * b[0] + ky[0] * b[1] + kz[0] * b[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            D = np.where(k2 > 0, 1.0 / 3.0 - kb ** 2 / k2, 1.0 / 3.0).astype(np.float32)
        F[i] *= D
    out = sfft.irfftn(F, s=shp, workers=workers)[:n[0], :n[1], :n[2]].astype(np.float32)
    del F
    if mask is not None:
        out -= np.float32(np.mean(out[np.asarray(mask, bool)]))
    return out


def _shim_fit(field, mask, order=2, stride=3):
    """Order-``order`` polynomial (in normalised voxel coordinates) least-squares fitted to
    ``field`` in ``mask`` (on every ``stride``-th voxel), evaluated everywhere; the scanner's shim."""
    shape = field.shape
    sub = tuple(slice(None, None, stride) for _ in range(3))
    idx = np.nonzero(np.asarray(mask, bool)[sub])
    coords = [(i * stride - (n - 1) / 2.0) / (n / 2.0) for i, n in zip(idx, shape)]
    terms = [(p, q, r) for p in range(order + 1) for q in range(order + 1) for r in range(order + 1) if p + q + r <= order]
    A = np.stack([coords[0] ** p * coords[1] ** q * coords[2] ** r for p, q, r in terms], 1)
    c, *_ = np.linalg.lstsq(A, field[sub][idx].astype(np.float64), rcond=None)
    ax = [((np.arange(n) - (n - 1) / 2.0) / (n / 2.0)).astype(np.float32) for n in shape]
    fit = np.zeros(shape, np.float32)
    for (p, q, r), ci in zip(terms, c):
        fit += np.float32(ci) * (ax[0][:, None, None] ** p) * (ax[1][None, :, None] ** q) * (ax[2][None, None, :] ** r)
    return fit


def _phase_offset(M0, mask):
    """The quadratic receive-phase offset of :func:`qsm_forward.generate_phase_offset`, without
    full-size coordinate grids."""
    a = np.abs(M0).astype(np.float64)
    tot = a.sum()
    c, w = [], []
    for ax in range(3):
        prof = a.sum(axis=tuple(i for i in range(3) if i != ax))
        x = np.arange(1, a.shape[ax] + 1)
        ck = float((prof * x).sum() / tot)
        c.append(ck)
        w.append(float(np.sqrt((prof * (x - ck) ** 2).sum() / tot)))
    g = [((np.arange(1, n + 1) - ck) / wk).astype(np.float32) ** 2 for n, ck, wk in zip(a.shape, c, w)]
    temp = g[0][:, None, None] + g[1][None, :, None] + g[2][None, None, :]
    m = np.asarray(mask, bool)
    rng_ = float(temp[m].max() - temp[m].min())
    return (-temp / rng_ * np.pi).astype(np.float32)


def _kspace_crop(sig, shape, workers=2):
    """Centred k-space crop of a complex volume to ``shape`` (qsm-forward's crop_kspace convention
    and energy scaling), single precision."""
    from scipy import fft as sfft
    from .qsm_forward import crop_imagespace
    F = sfft.fftshift(sfft.fftn(sig, workers=workers))
    F = np.ascontiguousarray(crop_imagespace(F, shape))
    out = sfft.ifftn(sfft.ifftshift(F), workers=workers)
    return (out * np.float32(np.prod(shape) / np.prod(sig.shape))).astype(np.complex64)


def simulate_acquisition(phantom: Union[Phantom, str], bids_dir: str, TEs: Sequence[float], B0: float = 3.0,
                         TR: float = 0.035, flip_angle: float = 15.0, voxel_size=(1.0, 1.0, 1.0),
                         B0_dir=(0.0, 0.0, 1.0), noise_sigma: Optional[float] = None, seed: int = 0,
                         subject: str = "1", shim_order: Optional[int] = 2, phase_offset: bool = True,
                         crop: bool = True, pad_mm: float = 60.0, workers: int = 2) -> dict:
    """Simulate a multi-echo GRE acquisition of a (fine-grid) phantom on a coarser acquisition grid.

    The signal is computed on the phantom's grid and reduced to the acquisition matrix by k-space
    cropping, as :class:`qsm_forward.ReconGrid` does; the reference maps on the acquisition grid
    are partial-volume averages of the phantom's (χ, true total/shimmed/local field), so they are
    what a voxel of that size contains, not samples of the simulation grid. With an upsampled phantom
    (:func:`upsample_phantom`) this avoids the inverse crime of simulating on the truth's grid.

    Signal: M0 · SPGR(R1, TR, flip) · exp(-TE R2*) · exp(i (2π γ B0 TE Δf_shimmed + φ0)), with the
    total field from :func:`dipole_field` (single precision, padded by ``pad_mm``), a fitted
    order-``shim_order`` shim removed, and the quadratic receive-phase offset of
    :func:`qsm_forward.generate_phase_offset` if ``phase_offset``.

    Noise: complex Gaussian with standard deviation ``noise_sigma`` per channel, in the signal's
    own units (M0 from :func:`build_invivo_phantom` is in the source magnitude's units, so a σ
    measured on the source data applies directly); None writes noiseless data, and
    :func:`add_bids_noise` can add it afterwards.

    Writes the layout of :func:`simulate` (``sub-<subject>/anat/*_echo-N_part-{mag,phase}_MEGRE.nii``
    with sidecars; derivatives ``*_Chimap``, ``*_mask``, ``*_dseg``, ``*_fieldmap`` (total),
    ``*_desc-shimmed_fieldmap``, ``*_fieldmap-local``, ``*_desc-sources_dseg``, ``*_phantom.json``)
    and crops to the source field of view. χ on the acquisition grid is the mean over the brain part
    of each voxel inside the acquisition mask (partial volume >= 0.5); the true local field is that
    of χ inside the phantom's brain mask.
    """
    ph = Phantom.load(phantom) if isinstance(phantom, str) else phantom
    TEs = [float(t) for t in TEs]
    fvox = ph.voxel_size
    fshape = np.array(ph.chi.shape)
    acq_req = _vox(voxel_size)
    ashape = np.maximum(np.round(fshape * fvox / acq_req).astype(int), 1)
    avox = fshape * fvox / ashape
    ratio = avox / fvox
    mask = ph.mask.astype(bool)

    total = dipole_field(ph.chi, fvox, B0_dir, pad_mm=pad_mm, mask=mask, workers=workers)
    local = dipole_field(np.where(mask, ph.chi, 0.0).astype(np.float32), fvox, B0_dir, pad_mm=pad_mm,
                         pad_value=0.0, mask=mask, workers=workers)
    shim = _shim_fit(total, mask, shim_order) if shim_order is not None else np.zeros_like(total)
    sig_field = total - shim
    phi0 = _phase_offset(ph.M0, mask) if phase_offset else np.float32(0.0)

    # reference maps on the acquisition grid
    pvm = pv_resample(mask.astype(np.float32), ratio, ashape)
    amask = pvm >= 0.5
    chi_b = pv_resample(np.where(mask, ph.chi, 0.0), ratio, ashape) / np.maximum(pvm, 1e-6)
    chi_all = pv_resample(ph.chi, ratio, ashape)
    chi_acq = np.where(amask, chi_b, chi_all).astype(np.float32)
    del chi_b, chi_all
    maps = {"Chimap": chi_acq, "mask": amask.astype(np.uint8),
            "fieldmap": pv_resample(total, ratio, ashape),
            "desc-shimmed_fieldmap": pv_resample(sig_field, ratio, ashape),
            "fieldmap-local": pv_resample(local, ratio, ashape)}
    del total, local, shim
    near = tuple(np.clip(np.round(np.arange(n) * r).astype(int), 0, m - 1) for n, r, m in zip(ashape, ratio, fshape))
    ix = np.ix_(*near)
    maps["dseg"] = ph.seg[ix].astype(np.int16)
    if ph.source_labels is not None:
        maps["desc-sources_dseg"] = ph.source_labels[ix].astype(np.int16)

    aff = ph.affine.copy()
    aff[:3, :3] = ph.affine[:3, :3] * ratio[None, :]
    box = tuple(slice(0, int(n)) for n in ashape)
    if crop:
        box = tuple(slice(int(round(lo / r)), int(round(hi / r))) for (lo, hi), r in zip(ph.fov, ratio))
        aff[:3, 3] = aff[:3, :3] @ np.array([b.start for b in box]) + aff[:3, 3]

    rn = f"sub-{subject}"
    anat = os.path.join(bids_dir, rn, "anat")
    danat = os.path.join(bids_dir, "derivatives", "qsm-forward", rn, "anat")
    os.makedirs(anat, exist_ok=True)
    os.makedirs(danat, exist_ok=True)

    def save(a, path, dtype=np.float32):
        img = nib.Nifti1Image(np.asarray(a[box]).astype(dtype), aff)
        img.header.set_zooms(tuple(float(v) for v in avox))
        nib.save(img, path)

    for name, a in maps.items():
        save(a, os.path.join(danat, f"{rn}_{name}.nii"), np.int16 if "dseg" in name else (np.uint8 if name == "mask" else np.float32))

    ss = np.where(ph.R1 > 0, spgr_factor(np.where(ph.R1 > 0, ph.R1, 1.0), TR, flip_angle), 0.0).astype(np.float32)
    base = (ph.M0 * ss).astype(np.float32)
    del ss
    rng = np.random.default_rng(seed)
    gamma = 42.58  # MHz/T, as generate_signal
    echo_stats = []
    for i, te in enumerate(TEs, 1):
        sig = base * np.exp(np.float32(-te) * ph.R2star)
        sig = sig * np.exp(1j * (np.float32(2 * np.pi * gamma * B0 * te) * sig_field + phi0)).astype(np.complex64)
        acq = _kspace_crop(sig.astype(np.complex64), tuple(int(n) for n in ashape), workers)
        del sig
        acq = acq[box]
        if noise_sigma:
            acq = acq + (rng.standard_normal(acq.shape, np.float32) + 1j * rng.standard_normal(acq.shape, np.float32)) * np.float32(noise_sigma)
        echo_stats.append(float(np.median(np.abs(acq)[amask[box]])))
        side = {"EchoTime": te, "MagneticFieldStrength": B0, "EchoNumber": i, "RepetitionTime": TR,
                "FlipAngle": flip_angle, "B0_dir": [float(v) for v in B0_dir], "VoxelSize": avox.tolist(),
                "ConversionSoftware": "qsm-forward phantoms.simulate_acquisition",
                "NoiseSigma": float(noise_sigma or 0.0), "PhantomVoxelSize": fvox.tolist()}
        for part, v in (("mag", np.abs(acq)), ("phase", np.angle(acq))):
            img = nib.Nifti1Image(v.astype(np.float32), aff)
            img.header.set_zooms(tuple(float(x) for x in avox))
            nib.save(img, os.path.join(anat, f"{rn}_echo-{i}_part-{part}_MEGRE.nii"))
            with open(os.path.join(anat, f"{rn}_echo-{i}_part-{part}_MEGRE.json"), "w") as fh:
                json.dump(dict(side, ImageType=["P", "PHASE"] if part == "phase" else ["M", "MAGNITUDE"]), fh, indent=1)
    dd = os.path.join(bids_dir, "dataset_description.json")
    if not os.path.exists(dd):
        with open(dd, "w") as fh:
            json.dump({"Name": "qsm-forward phantom", "BIDSVersion": "1.9.0"}, fh)
    sim = {"acq_shape": [int(b.stop - b.start) for b in box], "acq_voxel_size": avox.tolist(),
           "phantom_voxel_size": fvox.tolist(), "phantom_shape": fshape.tolist(), "TEs": TEs, "B0": B0,
           "TR": TR, "flip_angle": flip_angle, "B0_dir": [float(v) for v in B0_dir],
           "noise_sigma": float(noise_sigma or 0.0), "seed": int(seed), "shim_order": shim_order,
           "phase_offset": bool(phase_offset), "median_brain_magnitude_per_echo": echo_stats,
           "truth": "partial-volume mean over the acquisition voxel (brain part inside the mask)"}
    with open(os.path.join(danat, f"{rn}_phantom.json"), "w") as fh:
        json.dump({"sources": ph.sources, "simulation": sim,
                   **{k: v for k, v in ph.meta.items() if k != "saved_to"},
                   "phantom_dir": ph.meta.get("saved_to")}, fh, indent=1, default=_json_default)
    return {"bids_dir": bids_dir, "anat": anat, "derivatives": danat, "recon_name": rn,
            "chi": os.path.join(danat, f"{rn}_Chimap.nii"), "mask": os.path.join(danat, f"{rn}_mask.nii"),
            "totalfield": os.path.join(danat, f"{rn}_fieldmap.nii"),
            "localfield": os.path.join(danat, f"{rn}_fieldmap-local.nii"), "simulation": sim}


def add_bids_noise(anat_dir: str, sigma: float, seed: int = 0, subject: str = "1") -> int:
    """Add complex Gaussian noise (σ per channel, signal units) in place to the echoes that
    :func:`simulate_acquisition` wrote without noise; records ``NoiseSigma`` in the sidecars.
    Returns the number of echoes changed."""
    rng = np.random.default_rng(seed)
    n = 0
    for mp in sorted(glob.glob(os.path.join(anat_dir, f"sub-{subject}_echo-*_part-mag_MEGRE.nii")),
                     key=lambda p: int(p.split("_echo-")[1].split("_")[0])):
        pp = mp.replace("part-mag", "part-phase")
        mi, pi = nib.load(mp), nib.load(pp)
        z = np.asarray(mi.dataobj, np.float32) * np.exp(1j * np.asarray(pi.dataobj, np.float32))
        z = z + (rng.standard_normal(z.shape, np.float32) + 1j * rng.standard_normal(z.shape, np.float32)) * np.float32(sigma)
        for path, v, img in ((mp, np.abs(z), mi), (pp, np.angle(z), pi)):
            out = nib.Nifti1Image(v.astype(np.float32), img.affine, img.header)
            nib.save(out, path)
            js = path.replace(".nii", ".json")
            if os.path.exists(js):
                side = json.load(open(js))
                side["NoiseSigma"] = float(sigma)
                json.dump(side, open(js, "w"), indent=1)
        n += 1
    return n
