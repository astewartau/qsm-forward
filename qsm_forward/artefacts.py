"""Artefact injection with known labels, for reference-free QSM quality assessment (qsm-qc).

Each effect is a small function that takes clean data and returns an :class:`EffectResult`: the
modified data, a label dict (what was applied, its severity in physical units, the seed, the
taxonomy rows and the sub-score it feeds) and, where the effect is local, a voxelwise ``affected``
map. The rows (``A4``, ``C2``, ``H1``…) and sub-scores (``GROSS``, ``EDGE``, ``LOCAL``…) are those of
qsm-qc's ``docs/taxonomy.md``. :data:`EFFECTS` lists every effect with a severity ladder from
negligible to gross; :func:`write_labels` writes a JSON sidecar of everything applied; and
:func:`inject_bids` applies signal-, field- and mask-domain effects to a qsm-forward BIDS phantom
so that a real pipeline (QSMxT) can be run on the result.

Data conventions:

- Volumes are numpy arrays indexed (x, y, z); ``voxel`` is the voxel size in mm. Multi-echo data
  carry the echo axis last.
- *signal*: complex multi-echo data (magnitude · exp(i · phase)); :func:`to_complex` and
  :func:`from_complex` convert. Every geometric operation on signal resamples the real and
  imaginary parts (never wrapped phase, which cannot be interpolated).
- *phase*: unwrapped phase in rad (3D, or echo axis last). *field*: a field map in Hz, ppm or rad/s.
- *chi*: a susceptibility map in ppm. *mask*: boolean.

Groups:

1. Gross errors on χ (GROSS; the offset feeds CALIB): :func:`gross_sign`, :func:`gross_scale`,
   :func:`gross_axes`, :func:`gross_wrong_b0`, :func:`gross_quantise`, :func:`gross_nonfinite`,
   :func:`chi_offset`.
2. Mask faults (EDGE): :func:`mask_dilate`, :func:`mask_erode`, :func:`mask_holes`,
   :func:`mask_plane_cut`, :func:`mask_islands`, :func:`mask_misregister`, and
   :func:`chi_under_mask` to see a faulty mask on a χ map.
3. Unwrapping failures (LOCAL): :func:`unwrap_patch` (blobs or rings, smooth or ragged borders),
   :func:`unwrap_slicewise`, :func:`unwrap_echo_error`; on unwrapped phase or on a field map.
4. Motion (LOCAL): :func:`interecho_motion` (per-echo rigid transforms), :func:`slicewise_motion`
   (2D per-slice shifts), :func:`within_echo_motion` (k-space line mixing).
5. Bipolar readout (BKG): :func:`bipolar_readout`.
6. k-space effects: :func:`kspace_spike` (NOISE), :func:`partial_fourier` (RES),
   :func:`phase_highpass` (CALIB).

Known limits of the forward model, stated in each function: when the head moves, the field moves
with it rigidly (the change of the dipole field with the head's orientation relative to B0, and the
scanner-fixed shim field, are not re-simulated); unwrapping errors cannot be put into wrapped phase,
so they go into unwrapped phase or the field map and a pipeline must start from there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage

GAMMA_BAR = 42.58  # MHz/T
SUBSCORES = ("NOISE", "RES", "CONE", "CALIB", "BKG", "EDGE", "LOCAL", "GROSS")


@dataclass
class EffectResult:
    """``data``: the modified data (same kind as the input). ``label``: what was applied (see
    :func:`_label`). ``affected``: voxelwise map of where the effect acts (bool or a magnitude in the
    label's unit), or None when the effect is global."""
    data: object
    label: dict
    affected: Optional[np.ndarray] = None


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _label(effect, rows, subscore, quantity, value, unit, params, seed=None, **extra):
    """The label every effect returns: ``effect`` (function name), ``taxonomy`` (rows of qsm-qc's
    docs/taxonomy.md), ``subscore``, ``severity`` ({quantity, value, unit}: the physical size of the
    effect, 0 = none), ``params`` (every input), ``seed`` and effect-specific measurements."""
    return _jsonable({"effect": effect, "taxonomy": list(rows), "subscore": subscore,
                      "severity": {"quantity": quantity, "value": value, "unit": unit},
                      "params": params, "seed": seed, **extra})


def _ml(n_vox, voxel):
    return float(n_vox) * float(np.prod(voxel)) / 1000.0


def _voxel(voxel):
    return np.asarray(voxel if voxel is not None else (1.0, 1.0, 1.0), float)


def to_complex(magnitude, phase):
    """magnitude · exp(i · phase)."""
    return np.asarray(magnitude, float) * np.exp(1j * np.asarray(phase, float))


def from_complex(signal):
    """(magnitude, wrapped phase) of complex data."""
    return np.abs(signal), np.angle(signal)


# ---------------------------------------------------------------- shared region builders

def _smooth_noise(shape, sigma_vox, rng):
    """Gaussian-smoothed white noise, zero mean, unit SD."""
    v = ndimage.gaussian_filter(rng.standard_normal(shape), sigma_vox)
    v -= v.mean()
    return v / max(v.std(), 1e-30)


def _random_point(mask, rng, inset_vox=0):
    """A random voxel of ``mask`` (of its erosion by ``inset_vox`` if that is not empty)."""
    m = mask
    if inset_vox > 0:
        e = ndimage.binary_erosion(mask, iterations=int(inset_vox))
        if e.any():
            m = e
    idx = np.argwhere(m)
    return idx[rng.integers(len(idx))].astype(float)


def _coords_mm(shape, voxel):
    return np.meshgrid(*[np.arange(n) * v for n, v in zip(shape, voxel)], indexing="ij", sparse=True)


def blob_region(shape, voxel, centre_vox, radius_mm, border="smooth", roughness=0.3, rng=None,
                ring_width_mm=None):
    """A blob (or, with ``ring_width_mm``, a spherical shell of that thickness at ``radius_mm``)
    around ``centre_vox`` whose border is deformed by noise: ``smooth`` uses noise smoothed at a
    third of the radius (a rounded lobe), ``ragged`` adds noise at the voxel scale (a jagged border,
    as region-growing unwrappers leave). ``roughness`` is the deformation relative to the radius."""
    rng = rng if rng is not None else np.random.default_rng(0)
    voxel = _voxel(voxel)
    c = np.asarray(centre_vox, float) * voxel
    x = _coords_mm(shape, voxel)
    r = np.sqrt(sum((xi - ci) ** 2 for xi, ci in zip(x, c)))
    if border not in ("smooth", "ragged"):
        raise ValueError("border must be 'smooth' or 'ragged'")
    pert = _smooth_noise(shape, np.maximum(radius_mm / 3.0 / voxel, 0.5), rng)
    if border == "ragged":
        pert = 0.6 * pert + 0.8 * _smooth_noise(shape, 0.7, rng)
    if ring_width_mm is None:
        return r / radius_mm + roughness * pert < 1.0
    d = np.abs(r - radius_mm) / (ring_width_mm / 2.0)
    return d + roughness * pert < 1.0


def signed_distance(mask, voxel):
    """Signed distance (mm) to the boundary of ``mask``: negative inside, positive outside."""
    voxel = _voxel(voxel)
    m = np.asarray(mask, bool)
    return ndimage.distance_transform_edt(~m, sampling=voxel) - ndimage.distance_transform_edt(m, sampling=voxel)


def mask_metrics(fault, ref, voxel):
    """Excess and missing volume (mL), Dice, and the mean signed distance (mm, positive outward)
    of ``fault``'s boundary from ``ref``'s."""
    voxel = _voxel(voxel)
    f = np.asarray(fault, bool)
    r = np.asarray(ref, bool)
    sd = signed_distance(r, voxel)
    edge = f & ~ndimage.binary_erosion(f)
    inter = (f & r).sum()
    # an empty mask has no surface: report it as eroded to the deepest point of the reference
    msd = float(sd[edge].mean()) if edge.any() else (float(sd[r].min()) if r.any() else 0.0)
    return {"excess_ml": _ml((f & ~r).sum(), voxel), "missing_ml": _ml((r & ~f).sum(), voxel),
            "dice": float(2 * inter / max(f.sum() + r.sum(), 1)), "mean_signed_surface_distance_mm": msd}


# ---------------------------------------------------------------- 1. gross errors on χ

def gross_sign(chi):
    """χ → −χ: wrong phase or χ sign convention (taxonomy H1). Severity: 1 (applied)."""
    return EffectResult(-np.asarray(chi, float),
                        _label("gross_sign", ["H1"], "GROSS", "sign flipped", 1, "categorical", {}))


def scale_causes(b0=3.0, te=0.02):
    """Scale factors of common unit and bookkeeping mistakes, for :func:`gross_scale`."""
    return {
        "ppb_as_ppm": 1000.0,               # χ written in ppb, read as ppm
        "hz_as_ppm": GAMMA_BAR * b0,        # field in Hz not divided by γB0
        "rad_per_s_as_hz": 2 * np.pi,       # angular frequency taken as Hz
        "hz_as_rad_per_s": 1 / (2 * np.pi),
        "missing_te": te,                   # phase/TE omitted: values multiplied by TE (s)
        "te_ms_as_s": 1e-3,                 # TE given in ms where s was expected
        "te_s_as_ms": 1e3,
        "b0_7t_as_3t": 7.0 / 3.0,           # sidecar field strength wrong
        "b0_3t_as_7t": 3.0 / 7.0,
        "b0_2p89_as_3": 3.0 / 2.89,         # nominal 3 T against the true 2.89 T (subtle)
    }


def gross_scale(chi, factor=None, cause=None, b0=3.0, te=0.02):
    """χ × ``factor``, or the factor of a named ``cause`` (:func:`scale_causes`: Hz/ppm/rad
    confusion, missing TE or γB0 division, wrong B0) (taxonomy H2). Severity: |log10 factor|.
    Sub-score GROSS for factor ≥ 1.5 or ≤ 1/1.5, else CALIB."""
    if (factor is None) == (cause is None):
        raise ValueError("give exactly one of factor and cause")
    if cause is not None:
        causes = scale_causes(b0, te)
        if cause not in causes:
            raise ValueError(f"cause must be one of {sorted(causes)}")
        factor = causes[cause]
    factor = float(factor)
    if factor == 0:
        raise ValueError("factor must be non-zero (use gross_sign for -1)")
    sub = "GROSS" if abs(factor) >= 1.5 or abs(factor) <= 1 / 1.5 else "CALIB"
    return EffectResult(np.asarray(chi, float) * factor,
                        _label("gross_scale", ["H2", "C6"], sub, "|log10 scale factor|",
                               abs(np.log10(abs(factor))), "decades",
                               {"factor": factor, "cause": cause, "b0": b0, "te": te}))


def gross_axes(chi, flip=(), permute=None):
    """Flip the axes in ``flip`` and/or transpose by ``permute`` (header and data disagree: L–R
    swap, permuted axes) (taxonomy H4). Severity: number of axes flipped plus 1 if permuted."""
    out = np.asarray(chi)
    if permute is not None:
        if sorted(permute) != list(range(out.ndim)):
            raise ValueError("permute must be a permutation of the axes")
        out = np.transpose(out, permute)
    if flip:
        out = np.flip(out, axis=tuple(flip))
    sev = len(tuple(flip)) + (0 if permute is None or list(permute) == sorted(permute) else 1)
    return EffectResult(np.ascontiguousarray(out),
                        _label("gross_axes", ["H4"], "GROSS", "axes flipped or permuted", sev, "count",
                               {"flip": list(flip), "permute": None if permute is None else list(permute)}))


def _dipole(kx, ky, kz, b):
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    kb = kx * b[0] + ky * b[1] + kz * b[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1.0 / 3.0 - np.where(k2 > 0, kb ** 2 / k2, 0.0)
    return np.where(k2 > 0, d, 0.0)


def gross_wrong_b0(chi, voxel, b0_true=(0, 0, 1), b0_assumed=(1, 0, 0), mask=None, threshold=0.15, pad=0.5):
    """The χ a pipeline returns when the field of ``chi`` (made with the true B0 direction) is
    inverted with the kernel of a wrong direction (axes permuted while the affine is kept, oblique
    data reconstructed along voxel z) (taxonomy H3, H5). Both inversions use the same truncated
    k-space division (TKD at ``threshold``), so the label's error RMS is that of the B0 direction
    alone, against the correct-direction TKD. Severity: angle between the directions (degrees).
    Sub-score GROSS for ≥ 30°, CONE below (H5's mild obliquity)."""
    voxel = _voxel(voxel)
    chi = np.asarray(chi, float)
    bt = np.asarray(b0_true, float) / np.linalg.norm(b0_true)
    ba = np.asarray(b0_assumed, float) / np.linalg.norm(b0_assumed)
    angle = float(np.degrees(np.arccos(np.clip(abs(bt @ ba), -1, 1))))
    sp = chi.shape
    ps = tuple(int(np.ceil(n * (1 + pad))) for n in sp)
    k = np.meshgrid(*[np.fft.fftfreq(n, d=v) for n, v in zip(ps, voxel)], indexing="ij", sparse=True)
    dt = _dipole(*k, bt)
    da = _dipole(*k, ba)
    src = chi if mask is None else chi * mask
    ax = (0, 1, 2)
    f = np.fft.ifftn(dt * np.fft.fftn(src, s=ps, axes=ax)).real[:sp[0], :sp[1], :sp[2]]
    if mask is not None:
        f = f * mask
    F = np.fft.fftn(f, s=ps, axes=ax)

    def tkd(d):
        dd = np.where(np.abs(d) < threshold, np.where(d >= 0, threshold, -threshold), d)
        x = np.fft.ifftn(F / dd).real[:sp[0], :sp[1], :sp[2]]
        return x * mask if mask is not None else x

    ref = tkd(dt)
    out = tkd(da)
    m = np.ones(sp, bool) if mask is None else np.asarray(mask, bool)
    err = out - ref
    return EffectResult(out, _label("gross_wrong_b0", ["H3", "H5"], "GROSS" if angle >= 30 else "CONE",
                                    "B0 direction error", angle, "deg",
                                    {"b0_true": bt, "b0_assumed": ba, "threshold": threshold},
                                    error_rms_ppb=float(np.sqrt(np.mean(err[m] ** 2)) * 1e3)),
                        affected=np.abs(err))


def gross_quantise(chi, step_ppm=0.01, clip_ppm=None, mask=None):
    """Round χ to multiples of ``step_ppm`` and clip to ±``clip_ppm`` (integer datatype, lost
    ``scl_slope``) (taxonomy H9). Severity: the step in ppb. Label also gives the clipped fraction."""
    chi = np.asarray(chi, float)
    out = np.round(chi / step_ppm) * step_ppm if step_ppm > 0 else chi.copy()
    clipped = np.zeros(chi.shape, bool)
    if clip_ppm is not None:
        clipped = np.abs(out) > clip_ppm
        out = np.clip(out, -clip_ppm, clip_ppm)
    m = np.ones(chi.shape, bool) if mask is None else np.asarray(mask, bool)
    return EffectResult(out, _label("gross_quantise", ["H9"], "GROSS", "quantisation step", step_ppm * 1e3, "ppb",
                                    {"step_ppm": step_ppm, "clip_ppm": clip_ppm},
                                    clipped_fraction=float(clipped[m].mean())),
                        affected=clipped if clip_ppm is not None else None)


def gross_nonfinite(chi, mask, voxel, n_patches=1, radius_mm=4.0, value="nan", seed=0):
    """Put NaN (``value='nan'``), ±Inf (``'inf'``) or a mix (``'mixed'``) in ``n_patches`` blobs
    of ``radius_mm`` inside the mask (radius 0: single voxels) (taxonomy C9, I1). Severity: number
    of non-finite voxels. Affected: those voxels."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    out = np.asarray(chi, float).copy()
    hit = np.zeros(mask.shape, bool)
    for _ in range(int(n_patches)):
        c = _random_point(mask, rng)
        if radius_mm <= 0:
            hit[tuple(c.astype(int))] = True
        else:
            hit |= blob_region(mask.shape, voxel, c, radius_mm, "smooth", 0.2, rng) & mask
    if value == "nan":
        out[hit] = np.nan
    elif value == "inf":
        out[hit] = np.where(rng.random(hit.sum()) < 0.5, np.inf, -np.inf)
    elif value == "mixed":
        out[hit] = rng.choice([np.nan, np.inf, -np.inf], hit.sum())
    else:
        raise ValueError("value must be 'nan', 'inf' or 'mixed'")
    return EffectResult(out, _label("gross_nonfinite", ["C9", "I1"], "GROSS", "non-finite voxels", int(hit.sum()),
                                    "voxels", {"n_patches": n_patches, "radius_mm": radius_mm, "value": value},
                                    seed, volume_ml=_ml(hit.sum(), voxel)), affected=hit)


def chi_offset(chi, offset_ppm, mask=None):
    """Add a constant (reference choice) (taxonomy G1). XSIM ignores it; PLAN reports it rather than
    scoring it. Sub-score CALIB. Severity: |offset| in ppb. Zero outside ``mask`` stays zero."""
    chi = np.asarray(chi, float)
    out = chi + offset_ppm if mask is None else np.where(mask, chi + offset_ppm, chi)
    return EffectResult(out, _label("chi_offset", ["G1", "G2"], "CALIB", "|offset|", abs(offset_ppm) * 1e3, "ppb",
                                    {"offset_ppm": offset_ppm}))


# ---------------------------------------------------------------- 2. mask faults

def _ball_distance_threshold(dist, mm, irregularity, voxel, rng):
    if irregularity <= 0:
        return dist <= mm
    pert = _smooth_noise(dist.shape, np.maximum(10.0 / voxel, 1.0), rng)
    return dist <= mm * np.clip(1.0 + irregularity * pert, 0.0, None)


def mask_dilate(mask, voxel, mm=3.0, irregularity=0.0, within=None, seed=0):
    """Grow the mask by ``mm`` (a mask that takes in skull, scalp, dura) (taxonomy D1, D4).
    ``irregularity`` > 0 varies the growth smoothly over the surface (relative SD; 0.5 = ±50 %),
    as threshold and BET masks leak more in some places. ``within`` (e.g. the phantom's head
    support, magnitude > threshold) limits the growth to where there is signal. Severity: mean
    signed surface distance (mm). Affected: the added voxels.

    The mask alone carries no data. To see what a real pipeline does with the added voxels, run it
    on the complex data with this mask (the head phantom has scalp and skull signal and whole-head
    χ, so the field there is real); on a χ map, :func:`chi_under_mask` fills them with a stand-in."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    dist = ndimage.distance_transform_edt(~mask, sampling=voxel)
    out = mask | _ball_distance_threshold(dist, mm, irregularity, voxel, rng)
    if within is not None:
        out &= np.asarray(within, bool) | mask
    met = mask_metrics(out, mask, voxel)
    return EffectResult(out, _label("mask_dilate", ["D1", "D4"], "EDGE", "mean signed surface distance",
                                    met["mean_signed_surface_distance_mm"], "mm",
                                    {"mm": mm, "irregularity": irregularity, "within": within is not None},
                                    seed, **met), affected=out & ~mask)


def mask_erode(mask, voxel, mm=3.0, irregularity=0.0, seed=0):
    """Shrink the mask by ``mm`` (cortex cut; the erosion of SMV-type background removal)
    (taxonomy D2, E4). ``irregularity`` as in :func:`mask_dilate`. Severity: mean signed surface
    distance (mm, negative). Affected: the removed voxels."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    dist = ndimage.distance_transform_edt(mask, sampling=voxel)
    out = mask & ~_ball_distance_threshold(dist, mm, irregularity, voxel, rng)
    met = mask_metrics(out, mask, voxel)
    return EffectResult(out, _label("mask_erode", ["D2", "E4"], "EDGE", "mean signed surface distance",
                                    met["mean_signed_surface_distance_mm"], "mm",
                                    {"mm": mm, "irregularity": irregularity}, seed, **met),
                        affected=mask & ~out)


def mask_holes(mask, voxel, n_holes=3, radius_mm=4.0, prior=None, border="smooth", seed=0):
    """Punch ``n_holes`` blobs of ``radius_mm`` out of the mask's interior (veins, high-R2* deep
    grey matter or haemorrhage dropped by a threshold mask) (taxonomy D3). ``prior`` (e.g. R2* or
    a vein/DGM label mask) weights where the holes go. Severity: hole volume (mL)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    inner = ndimage.binary_erosion(mask, iterations=max(int(np.ceil(radius_mm / voxel.min())), 1))
    inner = inner if inner.any() else mask
    hole = np.zeros(mask.shape, bool)
    cand = np.argwhere(inner)
    w = None
    if prior is not None:
        w = np.clip(np.asarray(prior, float)[inner], 0, None)
        w = w / w.sum() if w.sum() > 0 else None
    centres = []
    for _ in range(int(n_holes)):
        c = cand[rng.choice(len(cand), p=w)].astype(float)
        centres.append(c)
        hole |= blob_region(mask.shape, voxel, c, radius_mm, border, 0.3, rng)
    hole &= mask
    out = mask & ~hole
    return EffectResult(out, _label("mask_holes", ["D3"], "EDGE", "hole volume", _ml(hole.sum(), voxel), "mL",
                                    {"n_holes": n_holes, "radius_mm": radius_mm, "border": border,
                                     "prior": prior is not None}, seed, centres_vox=centres),
                        affected=hole)


def mask_plane_cut(mask, voxel, depth_mm=10.0, normal=(0, 0, -1)):
    """Remove the part of the mask within ``depth_mm`` of its extreme along ``normal`` (default
    inferior, -z: brainstem or cerebellum cut-off; +z: slab missing the vertex) (taxonomy D5,
    A19). Severity: missing volume (mL). Affected: the removed voxels."""
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    x = _coords_mm(mask.shape, voxel)
    proj = sum(xi * ni for xi, ni in zip(x, n))
    proj = np.broadcast_to(proj, mask.shape)
    top = proj[mask].max() if mask.any() else 0.0
    cut = mask & (proj > top - depth_mm)
    out = mask & ~cut
    return EffectResult(out, _label("mask_plane_cut", ["D5", "A19"], "EDGE", "missing volume",
                                    _ml(cut.sum(), voxel), "mL", {"depth_mm": depth_mm, "normal": n}),
                        affected=cut)


def mask_islands(mask, voxel, n_islands=3, radius_mm=4.0, gap_mm=(4.0, 15.0), within=None, seed=0):
    """Add ``n_islands`` disconnected blobs of ``radius_mm`` at ``gap_mm`` (min, max) outside the
    mask (eyes, scalp patches, neck kept by a threshold mask) (taxonomy D6). ``within`` (head
    support) keeps them where there is tissue. Severity: island volume (mL)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    dist = ndimage.distance_transform_edt(~mask, sampling=voxel)
    shell = (dist >= gap_mm[0] + radius_mm) & (dist <= gap_mm[1] + radius_mm)
    if within is not None:
        shell &= np.asarray(within, bool)
    if not shell.any():
        raise ValueError("no room for islands at that gap (grid too small, or 'within' too tight)")
    isl = np.zeros(mask.shape, bool)
    for _ in range(int(n_islands)):
        c = _random_point(shell, rng)
        isl |= blob_region(mask.shape, voxel, c, radius_mm, "smooth", 0.3, rng)
    isl &= dist >= gap_mm[0]
    if within is not None:
        isl &= np.asarray(within, bool)
    out = mask | isl
    return EffectResult(out, _label("mask_islands", ["D6"], "EDGE", "island volume", _ml(isl.sum(), voxel), "mL",
                                    {"n_islands": n_islands, "radius_mm": radius_mm, "gap_mm": list(gap_mm),
                                     "within": within is not None}, seed), affected=isl)


def mask_misregister(mask, voxel, shift_mm=(0.0, 0.0, 0.0), rot_deg=0.0, rot_axis=(0, 0, 1), flip_axis=None):
    """Shift (mm), rotate (degrees about the mask's centre) and/or flip the mask relative to the data
    (header mismatch, the mask of another run, BET orientation wrong) (taxonomy D7). Severity: the
    maximum displacement of a mask voxel (mm); the label also has Dice with the correct mask.
    Affected: voxels where the masks disagree."""
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    m = mask
    if flip_axis is not None:
        m = np.flip(m, axis=flip_axis)
    centre = np.argwhere(mask).mean(0) * voxel if mask.any() else np.array(mask.shape) * voxel / 2
    T = rigid_matrix(rot_axis, rot_deg, shift_mm, centre)
    out = _resample_real(m.astype(float), T, voxel, order=1) > 0.5
    idx = np.argwhere(mask).astype(float)
    src = idx.copy()
    if flip_axis is not None:
        src[:, flip_axis] = mask.shape[flip_axis] - 1 - src[:, flip_axis]
    # where each mask voxel's label ends up (flip, then T), against where it should be
    dest = (src * voxel) @ T[:3, :3].T + T[:3, 3]
    sev = float(np.linalg.norm(dest - idx * voxel, axis=1).max()) if len(idx) else 0.0
    met = mask_metrics(out, mask, voxel)
    return EffectResult(out, _label("mask_misregister", ["D7"], "EDGE", "max displacement", sev, "mm",
                                    {"shift_mm": list(shift_mm), "rot_deg": rot_deg, "rot_axis": list(rot_axis),
                                     "flip_axis": flip_axis}, **met), affected=out ^ mask)


def chi_under_mask(chi, true_mask, fault_mask, voxel, magnitude=None, junk_ppm=0.1, junk_smooth_mm=6.0,
                   noise_ppm=0.05, seed=0):
    """A χ map seen through a faulty mask: zero where ``fault_mask`` is off; where it is on but
    ``true_mask`` is off, a stand-in for the junk a pipeline produces outside the brain (smooth
    random field of RMS ``junk_ppm`` and length ``junk_smooth_mm`` plus white noise of SD
    ``noise_ppm``, scaled by 1/magnitude when ``magnitude`` is given, as phase noise does). This is
    a χ-domain approximation only: the real junk depends on the background removal and inversion,
    so prefer running a pipeline on the complex data with ``fault_mask``. Affected: the junk voxels."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    t = np.asarray(true_mask, bool)
    f = np.asarray(fault_mask, bool)
    out = np.where(f & t, np.asarray(chi, float), 0.0)
    junk = f & ~t
    if junk.any():
        smooth = _smooth_noise(t.shape, np.maximum(junk_smooth_mm / voxel, 0.5), rng) * junk_ppm
        noise = rng.standard_normal(t.shape) * noise_ppm
        if magnitude is not None:
            mag = np.asarray(magnitude, float)
            ref = np.median(mag[t]) if t.any() else mag.max()
            noise = noise * np.clip(ref / np.maximum(mag, 1e-6 * ref), 1.0, 20.0)
        out[junk] = (smooth + noise)[junk]
    return EffectResult(out, _label("chi_under_mask", ["D1", "D6", "D7"], "EDGE", "junk volume",
                                    _ml(junk.sum(), voxel), "mL",
                                    {"junk_ppm": junk_ppm, "junk_smooth_mm": junk_smooth_mm, "noise_ppm": noise_ppm,
                                     "magnitude": magnitude is not None}, seed,
                                    missing_ml=_ml((t & ~f).sum(), voxel)), affected=junk)


# ---------------------------------------------------------------- 3. unwrapping failures

def _jump(k, domain, te=None, b0=None, field_unit="hz"):
    """The value of a k·2π phase error in ``domain``: 2πk rad for phase; for a field map the
    frequency error of a 2πk phase error at echo time ``te`` (k/te Hz, 2πk/te rad/s, or
    k/(te γ B0) ppm)."""
    if domain == "phase":
        return 2 * np.pi * k
    if domain != "field":
        raise ValueError("domain must be 'phase' or 'field'")
    if te is None:
        raise ValueError("a field-domain unwrapping error needs the echo time te (s) at which it happened")
    hz = k / te
    if field_unit == "hz":
        return hz
    if field_unit == "rad/s":
        return 2 * np.pi * hz
    if field_unit == "ppm":
        if b0 is None:
            raise ValueError("ppm needs b0 (T)")
        return hz / (GAMMA_BAR * b0)
    raise ValueError("field_unit must be 'hz', 'rad/s' or 'ppm'")


def _ols_field_error_hz(region_jumps_rad, tes):
    """Field error (Hz) of per-echo phase errors through an unweighted fit through the origin."""
    tes = np.asarray(tes, float)
    return float(np.sum(tes * region_jumps_rad) / (2 * np.pi * np.sum(tes ** 2)))


def unwrap_patch(data, mask, voxel, k=1, radius_mm=15.0, n_patches=1, border="smooth", roughness=0.3,
                 ring_width_mm=None, echoes=None, domain="phase", te=None, tes=None, b0=None, field_unit="hz",
                 seed=0):
    """Add ±k·2π to ``n_patches`` regions inside ``mask`` (residual 2π jumps of spatial unwrapping)
    (taxonomy C2). Regions are blobs of ``radius_mm``, or rings (ROMEO ring artefacts) when
    ``ring_width_mm`` is given; ``border`` 'smooth' or 'ragged' (:func:`blob_region`). ``k`` may be
    an int or a sequence (one per patch; signs random when an int).

    ``domain='phase'``: ``data`` is unwrapped phase (rad), 3D or echo axis last; ``echoes`` (indices,
    None = all) are the echoes that get the jump (one region per patch shared by those echoes, as a
    template unwrapper spreads one echo's error). Give ``tes`` to have the field error of an
    unweighted fit through the origin in the label. ``domain='field'``: ``data`` is a field map and
    the jump is the field error of a 2π·k phase error at echo time ``te`` (:func:`_jump`).

    Severity: mis-unwrapped volume (mL); the label also has |k| max and the jump size. Affected: the
    signed jump in units of 2π (per voxel, before echoes)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    mask = np.asarray(mask, bool)
    out = np.asarray(data, float).copy()
    ks = list(k) if isinstance(k, (list, tuple, np.ndarray)) else [int(k) * rng.choice([-1, 1]) for _ in range(n_patches)]
    if len(ks) != n_patches:
        raise ValueError("one k per patch")
    cyc = np.zeros(mask.shape)
    centres = []
    for kk in ks:
        c = _random_point(mask, rng, inset_vox=2)
        centres.append(c)
        reg = blob_region(mask.shape, voxel, c, radius_mm, border, roughness, rng, ring_width_mm) & mask
        cyc[reg] += kk
    jump_unit = _jump(1, domain, te, b0, field_unit)
    extra = {}
    if domain == "phase" and out.ndim == 4:
        sel = list(range(out.shape[-1])) if echoes is None else list(echoes)
        for e in sel:
            out[..., e] += cyc * jump_unit
        if tes is not None:
            per = np.zeros(out.shape[-1])
            per[sel] = 2 * np.pi
            extra["field_error_hz_per_cycle"] = _ols_field_error_hz(per, tes)
    else:
        out += cyc * jump_unit
    vol = _ml((cyc != 0).sum(), voxel)
    return EffectResult(out, _label("unwrap_patch", ["C2"], "LOCAL", "mis-unwrapped volume", vol, "mL",
                                    {"k": ks, "radius_mm": radius_mm, "n_patches": n_patches, "border": border,
                                     "roughness": roughness, "ring_width_mm": ring_width_mm, "echoes": echoes,
                                     "domain": domain, "te": te, "b0": b0, "field_unit": field_unit},
                                    seed, max_abs_k=int(np.max(np.abs(ks))) if ks else 0,
                                    jump_per_cycle=jump_unit,
                                    jump_unit="rad" if domain == "phase" else field_unit,
                                    centres_vox=centres, **extra), affected=cyc)


def unwrap_slicewise(data, mask, voxel=None, axis=2, fraction=0.2, k_max=1, domain="phase", echoes=None, te=None,
                     b0=None, field_unit="hz", seed=0):
    """Constant ±k·2π (k in 1..``k_max``, random sign) over the in-mask part of a random
    ``fraction`` of the slices along ``axis`` (slice-wise or 2D unwrapping of 3D data)
    (taxonomy C3). Domains as :func:`unwrap_patch`. Severity: mis-unwrapped volume (mL)."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, bool)
    out = np.asarray(data, float).copy()
    other = tuple(i for i in range(3) if i != axis)
    slices = np.nonzero(mask.any(axis=other))[0]
    n = int(round(fraction * len(slices)))
    chosen = np.sort(rng.choice(slices, n, replace=False)) if n else np.array([], int)
    cyc = np.zeros(mask.shape)
    ks = []
    for s in chosen:
        kk = int(rng.integers(1, k_max + 1)) * int(rng.choice([-1, 1]))
        ks.append(kk)
        idx = [slice(None)] * 3
        idx[axis] = s
        cyc[tuple(idx)] = np.where(mask[tuple(idx)], kk, 0)
    ju = _jump(1, domain, te, b0, field_unit)
    if domain == "phase" and out.ndim == 4:
        for e in (range(out.shape[-1]) if echoes is None else echoes):
            out[..., e] += cyc * ju
    else:
        out += cyc * ju
    return EffectResult(out, _label("unwrap_slicewise", ["C3"], "LOCAL", "mis-unwrapped volume",
                                    _ml((cyc != 0).sum(), _voxel(voxel)), "mL",
                                    {"axis": axis, "fraction": fraction, "k_max": k_max, "domain": domain,
                                     "echoes": echoes, "te": te, "b0": b0, "field_unit": field_unit}, seed,
                                    slices=chosen.tolist(), k=ks, jump_per_cycle=ju), affected=cyc)


def unwrap_echo_error(phase, mask, voxel, echo=-1, k=1, radius_mm=None, border="smooth", tes=None, seed=0):
    """A 2π·k error in one echo only (temporal, echo-to-echo unwrapping) (taxonomy C4): over the
    whole mask when ``radius_mm`` is None (a global per-echo error, mostly removed by background
    field removal), else in one region (as around veins and deep grey matter at late echoes).
    ``phase``: unwrapped, echo axis last. Severity: affected volume (mL)."""
    phase = np.asarray(phase, float)
    if phase.ndim != 4:
        raise ValueError("phase needs an echo axis")
    e = echo % phase.shape[-1]
    mask = np.asarray(mask, bool)
    if radius_mm is None:
        out = phase.copy()
        out[..., e] += 2 * np.pi * k * mask
        voxel = _voxel(voxel)
        extra = {"field_error_hz": _ols_field_error_hz(np.eye(phase.shape[-1])[e] * 2 * np.pi * k, tes)} if tes is not None else {}
        return EffectResult(out, _label("unwrap_echo_error", ["C4"], "LOCAL", "affected volume",
                                        _ml(mask.sum(), voxel), "mL",
                                        {"echo": e, "k": k, "radius_mm": None}, seed, **extra),
                            affected=mask * float(k))
    r = unwrap_patch(phase, mask, voxel, k=[k], radius_mm=radius_mm, n_patches=1, border=border,
                     echoes=[e], tes=tes, seed=seed)
    lab = dict(r.label, effect="unwrap_echo_error", taxonomy=["C4"])
    lab["params"] = dict(lab["params"], echo=e)
    return EffectResult(r.data, lab, r.affected)


# ---------------------------------------------------------------- 4. motion

def rigid_matrix(axis=(0, 0, 1), angle_deg=0.0, translation_mm=(0.0, 0.0, 0.0), centre_mm=(0.0, 0.0, 0.0)):
    """4×4 rigid transform (mm, voxel axes as world axes): rotation by ``angle_deg`` about ``axis``
    through ``centre_mm`` (Rodrigues), then translation. Maps where tissue was to where it is."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 0 else np.array([0.0, 0.0, 1.0])
    t = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    c = np.asarray(centre_mm, float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + np.asarray(translation_mm, float)
    return T


def _resample_real(vol, T, voxel, order=1):
    """vol moved by T (mm): out(x) = vol(T⁻¹ x), linear (order 1) or cubic, zero outside."""
    voxel = _voxel(voxel)
    Ti = np.linalg.inv(T)
    S = np.diag(voxel)
    A = np.diag(1 / voxel) @ Ti[:3, :3] @ S
    b = np.diag(1 / voxel) @ Ti[:3, 3]
    return ndimage.affine_transform(vol, A, offset=b, order=order, mode="constant", cval=0.0,
                                    prefilter=order > 1)


def resample_complex(signal, T, voxel, order=1):
    """Complex data moved by the rigid transform ``T`` (:func:`rigid_matrix`): real and imaginary
    parts interpolated separately (magnitude and wrapped phase move together), any trailing echo
    axis moved alike."""
    s = np.asarray(signal)
    if s.ndim == 4:
        return np.stack([resample_complex(s[..., e], T, voxel, order) for e in range(s.shape[-1])], -1)
    return _resample_real(s.real, T, voxel, order) + 1j * _resample_real(s.imag, T, voxel, order)


def _displacement(T, pts):
    return np.linalg.norm(pts @ T[:3, :3].T + T[:3, 3] - pts, axis=1)


def interecho_motion(signal, voxel, mask=None, rot_deg=2.5, trans_mm=1.0, mode="progressive", axis=None,
                     reference_echo=0, order=1, seed=0):
    """Per-echo rigid head motion of complex multi-echo data (echo axis last) (taxonomy A4), as
    QSM.rs ``test_motion_correction`` (``rigid_axis_angle`` + ``resample_complex_onto``).

    ``mode='progressive'``: the head drifts steadily; echo e is rotated by ``rot_deg``·(e − ref)
    about one ``axis`` (random oblique unit vector if None) through the mask's centre and
    translated by ``trans_mm``·(e − ref) along one random direction. ``mode='random'``: each echo
    other than the reference gets its own random axis, a rotation ~ N(0, ``rot_deg``) and a
    translation of norm ~ |N(0, ``trans_mm``)| in a random direction.

    The field pattern moves rigidly with the head; the change of the dipole field with the head's
    orientation to B0 and the scanner-fixed shim field are not re-simulated (second order for a few
    degrees). Severity: RMS over the mask and the echoes of each voxel's displacement from the
    reference echo (mm). Label: per-echo axis, angle, translation, 4×4 matrix. Affected: per-voxel
    maximum displacement over the echoes (mm)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    s = np.asarray(signal)
    if s.ndim != 4:
        raise ValueError("signal needs an echo axis")
    ne = s.shape[-1]
    m = np.ones(s.shape[:3], bool) if mask is None else np.asarray(mask, bool)
    centre = np.argwhere(m).mean(0) * voxel
    pts = np.argwhere(m) * voxel
    if axis is None:
        axis = rng.standard_normal(3)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    tdir = rng.standard_normal(3)
    tdir /= np.linalg.norm(tdir)
    out = np.empty_like(s, dtype=complex)
    per = []
    disp2 = []
    dmax = np.zeros(m.shape)
    for e in range(ne):
        if mode == "progressive":
            ax, ang, tr = axis, rot_deg * (e - reference_echo), trans_mm * (e - reference_echo) * tdir
        elif mode == "random":
            if e == reference_echo:
                ax, ang, tr = axis, 0.0, np.zeros(3)
            else:
                ax = rng.standard_normal(3)
                ax /= np.linalg.norm(ax)
                d = rng.standard_normal(3)
                ang = float(rng.normal(0, rot_deg))
                tr = abs(rng.normal(0, trans_mm)) * d / np.linalg.norm(d)
        else:
            raise ValueError("mode must be 'progressive' or 'random'")
        T = rigid_matrix(ax, ang, tr, centre)
        out[..., e] = s[..., e] if (ang == 0 and not np.any(tr)) else resample_complex(s[..., e], T, voxel, order)
        d = _displacement(T, pts)
        disp2.append(d ** 2)
        dm = np.zeros(m.shape)
        dm[m] = d
        dmax = np.maximum(dmax, dm)
        per.append({"echo": e, "axis": ax, "angle_deg": ang, "translation_mm": tr, "matrix": T,
                    "rms_displacement_mm": float(np.sqrt(np.mean(d ** 2))), "max_displacement_mm": float(d.max())})
    rms = float(np.sqrt(np.mean(np.concatenate(disp2))))
    return EffectResult(out, _label("interecho_motion", ["A4"], "LOCAL", "RMS displacement", rms, "mm",
                                    {"rot_deg": rot_deg, "trans_mm": trans_mm, "mode": mode,
                                     "reference_echo": reference_echo, "order": order}, seed,
                                    echoes=per, centre_mm=centre,
                                    note="field moves rigidly with the head; B0-orientation change not re-simulated"),
                        affected=dmax)


def _fourier_shift(vol, shifts_vox, axes):
    """Sub-voxel circular shift of (complex) ``vol`` by ``shifts_vox`` along ``axes``."""
    F = np.fft.fftn(vol, axes=axes)
    ph = 0.0
    for s, ax in zip(shifts_vox, axes):
        k = np.fft.fftfreq(vol.shape[ax])
        sh = [1] * vol.ndim
        sh[ax] = -1
        ph = ph + (k * s).reshape(sh)
    return np.fft.ifftn(F * np.exp(-2j * np.pi * ph), axes=axes)


def slicewise_motion(signal, voxel, mask=None, axis=2, max_shift_mm=1.0, fraction=0.3, seed=0):
    """In-plane translation of a random ``fraction`` of the slices along ``axis`` (2D multi-slice
    acquired slice by slice), the same for every echo (taxonomy A6). Shifts uniform in
    ±``max_shift_mm`` per in-plane axis. Severity: RMS in-plane displacement over the moved slices
    (mm). Affected: per-voxel displacement (mm)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    s = np.asarray(signal, complex)
    multi = s.ndim == 4
    s4 = s if multi else s[..., None]
    out = s4.copy()
    inplane = [i for i in range(3) if i != axis]
    m = np.ones(s4.shape[:3], bool) if mask is None else np.asarray(mask, bool)
    slices = np.nonzero(m.any(axis=tuple(inplane)))[0]
    n = int(round(fraction * len(slices)))
    chosen = np.sort(rng.choice(slices, n, replace=False)) if n else np.array([], int)
    aff = np.zeros(m.shape)
    rec = []
    for sl in chosen:
        sh_mm = rng.uniform(-max_shift_mm, max_shift_mm, 2)
        sh_vox = sh_mm / voxel[inplane]
        idx = [slice(None)] * 4
        idx[axis] = sl
        plane = s4[tuple(idx)]  # 2 in-plane axes + echo
        out[tuple(idx)] = _fourier_shift(plane, sh_vox, axes=(0, 1))
        idx3 = [slice(None)] * 3
        idx3[axis] = sl
        aff[tuple(idx3)] = np.linalg.norm(sh_mm)
        rec.append({"slice": int(sl), "shift_mm": sh_mm})
    sev = float(np.sqrt(np.mean([np.sum(r["shift_mm"] ** 2) for r in rec]))) if rec else 0.0
    return EffectResult(out if multi else out[..., 0],
                        _label("slicewise_motion", ["A6"], "LOCAL", "RMS in-plane displacement", sev, "mm",
                               {"axis": axis, "max_shift_mm": max_shift_mm, "fraction": fraction}, seed,
                               slices=rec), affected=aff * m)


def within_echo_motion(signal, voxel, mask=None, pe_axis=1, onset=0.7, rot_deg=3.0, trans_mm=2.0, axis=None,
                       order=1, seed=0):
    """Motion during the acquisition (taxonomy A5): with sequential ordering along the phase-encode
    axis ``pe_axis``, the k-space lines acquired after ``onset`` (fraction of the scan) come from a
    copy moved by a rigid transform (rotation ``rot_deg`` about ``axis`` through the mask centre,
    translation ``trans_mm`` in a random direction); every echo shares the same line split. Gives
    ghosts along the phase-encode axis. Severity: ghost energy ‖S' − S‖² / ‖S‖² over all echoes (%).
    Affected: |S' − S| of the first echo relative to its in-mask median magnitude."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    s = np.asarray(signal, complex)
    multi = s.ndim == 4
    s4 = s if multi else s[..., None]
    m = np.ones(s4.shape[:3], bool) if mask is None else np.asarray(mask, bool)
    centre = np.argwhere(m).mean(0) * voxel
    if axis is None:
        axis = rng.standard_normal(3)
    d = rng.standard_normal(3)
    T = rigid_matrix(axis, rot_deg, trans_mm * d / np.linalg.norm(d), centre)
    n = s4.shape[pe_axis]
    kpos = np.fft.fftfreq(n) * n  # array position -> signed k index
    t_acq = (kpos + n // 2) / n  # acquisition time (fraction of the scan), sequential from -kmax to +kmax
    moved_lines = t_acq >= onset
    sh = [1, 1, 1]
    sh[pe_axis] = -1
    sel = moved_lines.reshape(sh)
    out = np.empty_like(s4)
    for e in range(s4.shape[-1]):
        mv = resample_complex(s4[..., e], T, voxel, order)
        K = np.fft.fftn(s4[..., e])
        Km = np.fft.fftn(mv)
        out[..., e] = np.fft.ifftn(np.where(sel, Km, K))
    diff = out - s4
    ghost = float(np.sum(np.abs(diff) ** 2) / max(np.sum(np.abs(s4) ** 2), 1e-30) * 100)
    ref = np.median(np.abs(s4[..., 0])[m]) if m.any() else 1.0
    return EffectResult(out if multi else out[..., 0],
                        _label("within_echo_motion", ["A5"], "LOCAL", "ghost energy", ghost, "%",
                               {"pe_axis": pe_axis, "onset": onset, "rot_deg": rot_deg, "trans_mm": trans_mm,
                                "order": order}, seed, matrix=T, moved_line_fraction=float(moved_lines.mean()),
                               max_displacement_mm=float(_displacement(T, np.argwhere(m) * voxel).max())),
                        affected=np.abs(diff[..., 0]) / max(ref, 1e-30))


# ---------------------------------------------------------------- 5. bipolar readout

def bipolar_readout(signal, voxel, mask=None, readout_axis=0, gradient_rad_per_mm=0.02, offset_rad=0.0,
                    shift_voxels=0.0, echoes="even"):
    """Bipolar (alternating-polarity) readout errors (taxonomy A9): the echoes read with the
    opposite gradient (``echoes='even'``: echoes 2, 4, … ; ``'odd'``: 1, 3, …; or a list of 0-based
    indices) get a linear phase ramp ``gradient_rad_per_mm`` along ``readout_axis`` (a k-space
    centre shift from gradient delays and eddy currents) plus a constant ``offset_rad``, and are
    displaced by ``shift_voxels`` along the readout (opposite-polarity off-resonance and delay
    shift; Fourier shift of the complex image). Monopolar data are the input; the ramp is centred
    on the mask's centre. Severity: phase-ramp gradient (rad/mm); label also gives the peak ramp
    phase over the mask and the shift. Affected: the added phase (rad) on the affected echoes."""
    voxel = _voxel(voxel)
    s = np.asarray(signal, complex)
    if s.ndim != 4:
        raise ValueError("signal needs an echo axis")
    ne = s.shape[-1]
    if echoes == "even":
        sel = list(range(1, ne, 2))
    elif echoes == "odd":
        sel = list(range(0, ne, 2))
    else:
        sel = list(echoes)
    m = np.ones(s.shape[:3], bool) if mask is None else np.asarray(mask, bool)
    c = np.argwhere(m).mean(0)[readout_axis] * voxel[readout_axis]
    x = np.arange(s.shape[readout_axis]) * voxel[readout_axis] - c
    shp = [1, 1, 1]
    shp[readout_axis] = -1
    ramp = np.broadcast_to((gradient_rad_per_mm * x + offset_rad).reshape(shp), s.shape[:3])
    out = s.copy()
    for e in sel:
        v = s[..., e]
        if shift_voxels:
            v = _fourier_shift(v, [shift_voxels], axes=(readout_axis,))
        out[..., e] = v * np.exp(1j * ramp)
    return EffectResult(out, _label("bipolar_readout", ["A9"], "BKG", "odd-even phase gradient",
                                    abs(gradient_rad_per_mm), "rad/mm",
                                    {"readout_axis": readout_axis, "gradient_rad_per_mm": gradient_rad_per_mm,
                                     "offset_rad": offset_rad, "shift_voxels": shift_voxels, "echoes": sel},
                                    peak_ramp_rad=float(np.abs(ramp[m]).max()) if m.any() else 0.0,
                                    shift_mm=float(shift_voxels * voxel[readout_axis])),
                        affected=np.where(m, np.asarray(ramp), 0.0))


# ---------------------------------------------------------------- 6. k-space effects

def kspace_spike(signal, voxel, amplitude=0.05, position=None, echoes=None, seed=0):
    """An RF spike: one k-space sample (``position``: signed k indices, random if None, away from the
    centre) gets ``amplitude`` × |S(k=0)| with a random phase, in ``echoes`` (None = all; a spike
    usually hits one) (taxonomy A23). Gives a stripe (herringbone) over the whole image. Severity:
    spike amplitude relative to the k-space centre. Label: the stripe's spatial frequency
    (cycles/mm) and wavelength (mm)."""
    rng = np.random.default_rng(seed)
    voxel = _voxel(voxel)
    s = np.asarray(signal, complex)
    multi = s.ndim == 4
    s4 = s if multi else s[..., None]
    shp = s4.shape[:3]
    if position is None:
        position = [int(rng.integers(n // 8, n // 3)) * int(rng.choice([-1, 1])) for n in shp]
    pos = tuple(int(p) % n for p, n in zip(position, shp))
    sel = range(s4.shape[-1]) if echoes is None else echoes
    phase = float(rng.uniform(0, 2 * np.pi))
    out = s4.copy()
    for e in sel:
        K = np.fft.fftn(s4[..., e])
        K[pos] += amplitude * abs(K[0, 0, 0]) * np.exp(1j * phase)
        out[..., e] = np.fft.ifftn(K)
    kf = np.array([p / (n * v) for p, n, v in zip(position, shp, voxel)])
    kn = float(np.linalg.norm(kf))
    return EffectResult(out if multi else out[..., 0],
                        _label("kspace_spike", ["A23"], "NOISE", "spike amplitude / |S(k=0)|", amplitude, "ratio",
                               {"amplitude": amplitude, "position": list(position),
                                "echoes": None if echoes is None else list(echoes)}, seed,
                               spike_phase_rad=phase, stripe_k_cycles_per_mm=kf,
                               stripe_wavelength_mm=1 / kn if kn > 0 else None))


def partial_fourier(signal, axis=1, fraction=0.75, method="zerofill"):
    """Partial Fourier along ``axis``: only ``fraction`` (0.5–1) of k-space is acquired (the low
    side is missing) (taxonomy A14). ``method='zerofill'`` zero-fills (blur along the axis and
    ringing); ``'homodyne'`` is Noll's homodyne reconstruction (ramp weighting of the asymmetric
    part, phase from the symmetric centre), so the image phase becomes the low-resolution phase.
    Severity: 1 − fraction (missing k-space fraction along the axis)."""
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0.5, 1]")
    if method not in ("zerofill", "homodyne"):
        raise ValueError("method must be 'zerofill' or 'homodyne'")
    s = np.asarray(signal, complex)
    multi = s.ndim == 4
    s4 = s if multi else s[..., None]
    n = s4.shape[axis]
    kidx = np.fft.fftfreq(n) * n  # signed k index of each array position
    n_low = int(round(fraction * n)) - n // 2  # acquired: k >= -n_low
    acquired = kidx >= -n_low
    shp = [1, 1, 1]
    shp[axis] = -1
    acq = acquired.reshape(shp)
    out = np.empty_like(s4)
    for e in range(s4.shape[-1]):
        K = np.fft.fftn(s4[..., e])
        Kp = np.where(acq, K, 0)
        if method == "zerofill":
            out[..., e] = np.fft.ifftn(Kp)
            continue
        sym = (np.abs(kidx) <= n_low).reshape(shp)
        w = np.where(kidx > n_low, 2.0, np.where(acquired, 1.0, 0.0))
        # smooth transition across the symmetric band (Noll's ramp)
        band = np.abs(kidx) <= n_low
        if n_low > 0:
            w = np.where(band, 1.0 + kidx / max(n_low, 1), w)
        img_w = np.fft.ifftn(Kp * w.reshape(shp))
        low = np.fft.ifftn(np.where(sym, K, 0))
        ph = np.exp(1j * np.angle(low))
        out[..., e] = np.real(img_w * np.conj(ph)) * ph
    return EffectResult(out if multi else out[..., 0],
                        _label("partial_fourier", ["A14"], "RES", "missing k-space fraction", 1 - fraction, "fraction",
                               {"axis": axis, "fraction": fraction, "method": method}))


def phase_highpass(signal, voxel, sigma_mm=20.0):
    """SWI-style homodyne high-pass of the phase (scanner SWI phase, Bruker filtered phase fed to
    QSM) (taxonomy H10): S · conj(L)/|L|, L the complex image low-passed by a Gaussian of SD
    ``sigma_mm`` (image space). Severity: the cut-off spatial frequency 1/(2π σ) (cycles/mm), where
    the low-pass falls to e^(-1/2); larger cut-off removes more."""
    voxel = _voxel(voxel)
    s = np.asarray(signal, complex)
    multi = s.ndim == 4
    s4 = s if multi else s[..., None]
    sig = np.asarray(sigma_mm, float) / voxel
    out = np.empty_like(s4)
    for e in range(s4.shape[-1]):
        L = ndimage.gaussian_filter(s4[..., e].real, sig) + 1j * ndimage.gaussian_filter(s4[..., e].imag, sig)
        out[..., e] = s4[..., e] * np.exp(-1j * np.angle(L))
    kc = 1.0 / (2 * np.pi * float(sigma_mm))
    return EffectResult(out if multi else out[..., 0],
                        _label("phase_highpass", ["H10"], "CALIB", "high-pass cut-off", kc, "cycles/mm",
                               {"sigma_mm": sigma_mm}))


# ---------------------------------------------------------------- registry

@dataclass(frozen=True)
class Effect:
    """One injectable effect: its function, the data it acts on, the taxonomy rows and sub-score,
    and a severity ladder (keyword arguments, negligible → gross) used by the tests and demos."""
    fn: Callable
    domain: str            # 'chi' | 'mask' | 'phase' | 'field' | 'signal'
    rows: tuple
    subscore: str
    ladder: tuple = dc_field(default_factory=tuple)


EFFECTS = {
    # 1. gross
    "gross_sign": Effect(gross_sign, "chi", ("H1",), "GROSS", ({},)),
    "gross_scale": Effect(gross_scale, "chi", ("H2", "C6"), "GROSS",
                          ({"factor": 1.04}, {"factor": 1.5}, {"cause": "rad_per_s_as_hz"}, {"cause": "ppb_as_ppm"})),
    "gross_axes": Effect(gross_axes, "chi", ("H4",), "GROSS",
                         ({"flip": (0,)}, {"permute": (1, 0, 2)}, {"flip": (0, 2), "permute": (2, 1, 0)})),
    "gross_wrong_b0": Effect(gross_wrong_b0, "chi", ("H3", "H5"), "GROSS",
                             ({"b0_assumed": (0, np.sin(np.radians(5)), np.cos(np.radians(5)))},
                              {"b0_assumed": (0, np.sin(np.radians(20)), np.cos(np.radians(20)))},
                              {"b0_assumed": (1, 0, 0)})),
    "gross_quantise": Effect(gross_quantise, "chi", ("H9",), "GROSS",
                             ({"step_ppm": 0.001}, {"step_ppm": 0.01}, {"step_ppm": 0.05, "clip_ppm": 0.1})),
    "gross_nonfinite": Effect(gross_nonfinite, "chi", ("C9", "I1"), "GROSS",
                              ({"radius_mm": 0, "n_patches": 1}, {"radius_mm": 4}, {"radius_mm": 10, "n_patches": 3})),
    "chi_offset": Effect(chi_offset, "chi", ("G1",), "CALIB",
                         ({"offset_ppm": 0.002}, {"offset_ppm": 0.02}, {"offset_ppm": 0.1})),
    # 2. mask
    "mask_dilate": Effect(mask_dilate, "mask", ("D1", "D4"), "EDGE",
                          ({"mm": 1.0}, {"mm": 4.0, "irregularity": 0.5}, {"mm": 8.0, "irregularity": 0.5})),
    "mask_erode": Effect(mask_erode, "mask", ("D2", "E4"), "EDGE",
                         ({"mm": 1.0}, {"mm": 3.0, "irregularity": 0.5}, {"mm": 7.0})),
    "mask_holes": Effect(mask_holes, "mask", ("D3",), "EDGE",
                         ({"n_holes": 1, "radius_mm": 2.0}, {"n_holes": 3, "radius_mm": 4.0},
                          {"n_holes": 6, "radius_mm": 8.0, "border": "ragged"})),
    "mask_plane_cut": Effect(mask_plane_cut, "mask", ("D5", "A19"), "EDGE",
                             ({"depth_mm": 2.0}, {"depth_mm": 10.0}, {"depth_mm": 30.0})),
    "mask_islands": Effect(mask_islands, "mask", ("D6",), "EDGE",
                           ({"n_islands": 1, "radius_mm": 2.0}, {"n_islands": 3, "radius_mm": 4.0},
                            {"n_islands": 8, "radius_mm": 7.0})),
    "mask_misregister": Effect(mask_misregister, "mask", ("D7",), "EDGE",
                               ({"shift_mm": (1.0, 0, 0)}, {"shift_mm": (3.0, 2.0, 0), "rot_deg": 3.0},
                                {"shift_mm": (8.0, 0, 4.0), "rot_deg": 8.0})),
    # 3. unwrapping
    "unwrap_patch": Effect(unwrap_patch, "phase", ("C2",), "LOCAL",
                           ({"radius_mm": 3.0}, {"radius_mm": 10.0, "border": "ragged"},
                            {"radius_mm": 20.0, "n_patches": 3, "k": 2})),
    "unwrap_ring": Effect(unwrap_patch, "phase", ("C2",), "LOCAL",
                          ({"radius_mm": 6.0, "ring_width_mm": 2.0}, {"radius_mm": 12.0, "ring_width_mm": 4.0},
                           {"radius_mm": 20.0, "ring_width_mm": 6.0, "border": "ragged"})),
    "unwrap_slicewise": Effect(unwrap_slicewise, "phase", ("C3",), "LOCAL",
                               ({"fraction": 0.02}, {"fraction": 0.1}, {"fraction": 0.4, "k_max": 2})),
    "unwrap_echo_error": Effect(unwrap_echo_error, "phase", ("C4",), "LOCAL",
                                ({"radius_mm": 5.0}, {"radius_mm": 15.0}, {"radius_mm": None})),
    # 4. motion
    "interecho_motion": Effect(interecho_motion, "signal", ("A4",), "LOCAL",
                               ({"rot_deg": 0.2, "trans_mm": 0.1}, {"rot_deg": 1.0, "trans_mm": 0.5},
                                {"rot_deg": 2.5, "trans_mm": 1.0})),
    "slicewise_motion": Effect(slicewise_motion, "signal", ("A6",), "LOCAL",
                               ({"max_shift_mm": 0.2}, {"max_shift_mm": 1.0}, {"max_shift_mm": 3.0, "fraction": 0.5})),
    "within_echo_motion": Effect(within_echo_motion, "signal", ("A5",), "LOCAL",
                                 ({"rot_deg": 0.5, "trans_mm": 0.3}, {"rot_deg": 2.0, "trans_mm": 1.0},
                                  {"rot_deg": 5.0, "trans_mm": 3.0, "onset": 0.55})),
    # 5. bipolar
    "bipolar_readout": Effect(bipolar_readout, "signal", ("A9",), "BKG",
                              ({"gradient_rad_per_mm": 0.002}, {"gradient_rad_per_mm": 0.02, "shift_voxels": 0.25},
                               {"gradient_rad_per_mm": 0.06, "offset_rad": 0.5, "shift_voxels": 0.5})),
    # 6. k-space
    "kspace_spike": Effect(kspace_spike, "signal", ("A23",), "NOISE",
                           ({"amplitude": 0.002}, {"amplitude": 0.02}, {"amplitude": 0.1})),
    "partial_fourier": Effect(partial_fourier, "signal", ("A14",), "RES",
                              ({"fraction": 0.875}, {"fraction": 0.75}, {"fraction": 0.625, "method": "homodyne"})),
    "phase_highpass": Effect(phase_highpass, "signal", ("H10",), "CALIB",
                             ({"sigma_mm": 60.0}, {"sigma_mm": 20.0}, {"sigma_mm": 6.0})),
}


def apply_effect(name, data, voxel=None, mask=None, seed=0, **kwargs):
    """Call effect ``name`` of :data:`EFFECTS` with the arguments it takes (``voxel``, ``mask``,
    ``seed`` passed when its signature has them)."""
    import inspect
    eff = EFFECTS[name]
    sig = list(inspect.signature(eff.fn).parameters)
    args = dict(kwargs)
    if "voxel" in sig:
        args.setdefault("voxel", voxel)
    if "mask" in sig[1:]:  # mask-domain effects take the mask itself as ``data``
        args.setdefault("mask", mask)
    if "seed" in sig:
        args.setdefault("seed", seed)
    return eff.fn(data, **args)


def write_labels(path, labels, extra=None):
    """Write the labels of every applied effect (a list, in order) as JSON, with the qsm-forward
    version. Returns ``path``."""
    from .qsm_forward import get_version
    if isinstance(labels, dict):
        labels = [labels]
    doc = {"QsmForwardVersion": get_version(), "Effects": _jsonable(list(labels)), **_jsonable(extra or {})}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


# ---------------------------------------------------------------- BIDS injection

def _sidecar_path(nii):
    return nii[: -len(".nii.gz")] + ".json" if nii.endswith(".nii.gz") else nii[: -len(".nii")] + ".json"


def inject_bids(bids_dir, out_dir, effects, subject="1", seed=0):
    """Copy a qsm-forward BIDS phantom to ``out_dir`` and apply ``effects`` (a list of
    ``(name, kwargs)``) in order:

    - ``signal`` effects act on the multi-echo GRE (``sub-<s>/anat/*_part-{mag,phase}_*``), which is
      rewritten (magnitude and wrapped phase);
    - ``mask`` effects act on the ground-truth mask and write ``*_desc-fault_mask.nii`` (pass it to the
      pipeline as its mask);
    - ``field`` effects (unwrapping failures with ``domain='field'``) act on the total field map
      (``*_fieldmap.nii``, ppm; written by ``--save-field``) and write ``*_desc-fault_fieldmap.nii``.
      Wrapped phase cannot carry an unwrapping error, so a pipeline must start from this field map.
    - ``sidecar`` edits: ``("sidecar", {"te_factor": f, "b0": T, "b0_dir": [x, y, z]})`` scale every
      ``EchoTime`` or set ``MagneticFieldStrength`` / ``B0_dir`` (gross scale and B0 errors through
      a real pipeline; taxonomy H2, H3).

    Writes ``<out_dir>/artefacts.json`` (:func:`write_labels`). Returns that path."""
    import glob
    import shutil
    import nibabel as nib
    from .degrade import _echo_files

    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    shutil.copytree(bids_dir, out_dir)
    anat = os.path.join(out_dir, f"sub-{subject}", "anat")
    deriv = os.path.join(out_dir, "derivatives", "qsm-forward", f"sub-{subject}", "anat")
    mags = _echo_files(anat, "MEGRE")
    phs = [f.replace("_part-mag_", "_part-phase_") for f in mags]
    ref = nib.load(mags[0])
    voxel = [float(v) for v in ref.header.get_zooms()[:3]]
    mask_files = [f for f in glob.glob(os.path.join(deriv, "*_mask.nii*")) if "_desc-" not in os.path.basename(f)]
    mask = nib.load(mask_files[0]).get_fdata() > 0.5 if mask_files else None
    stem = os.path.basename(mask_files[0]).split("_mask")[0] if mask_files else f"sub-{subject}"
    signal = None
    fault_mask = None
    field = None
    labels = []
    for i, (name, kw) in enumerate(effects):
        s = seed + i
        if name == "sidecar":
            for f in mags + phs:
                j = _sidecar_path(f)
                d = json.load(open(j))
                if "te_factor" in kw:
                    d["EchoTime"] = d["EchoTime"] * kw["te_factor"]
                if "b0" in kw:
                    d["MagneticFieldStrength"] = kw["b0"]
                if "b0_dir" in kw:
                    d["B0_dir"] = list(kw["b0_dir"])
                json.dump(d, open(j, "w"), indent=2)
            labels.append(_label("sidecar", ["H2", "H3"], "GROSS", "sidecar edit", 1, "categorical", dict(kw)))
            continue
        dom = EFFECTS[name].domain
        if dom == "signal":
            if signal is None:
                signal = np.stack([to_complex(nib.load(m).get_fdata(), nib.load(p).get_fdata())
                                   for m, p in zip(mags, phs)], -1)
            r = apply_effect(name, signal, voxel=voxel, mask=mask, seed=s, **kw)
            signal = r.data
        elif dom == "mask":
            r = apply_effect(name, mask if fault_mask is None else fault_mask, voxel=voxel, seed=s, **kw)
            fault_mask = r.data
        elif dom == "phase" and kw.get("domain") == "field":
            if field is None:
                ff = os.path.join(deriv, f"{stem}_fieldmap.nii")
                if not os.path.exists(ff):
                    raise FileNotFoundError("field-domain effects need *_fieldmap.nii (generate with --save-field)")
                field = nib.load(ff).get_fdata()
            b0 = float(json.load(open(_sidecar_path(mags[0])))["MagneticFieldStrength"])
            kw = dict(kw, field_unit="ppm", b0=b0)
            r = apply_effect(name, field, voxel=voxel, mask=mask, seed=s, **kw)
            field = r.data
        else:
            raise ValueError(f"{name}: inject_bids handles signal, mask and field-domain effects; "
                             "apply chi- and phase-domain effects with the functions directly")
        labels.append(r.label)
    if signal is not None:
        for e, (m, p) in enumerate(zip(mags, phs)):
            mg, ph = from_complex(signal[..., e])
            for path, arr in ((m, mg), (p, ph)):
                img = nib.load(path)
                nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header), path)
    if fault_mask is not None:
        nib.save(nib.Nifti1Image(fault_mask.astype(np.uint8), ref.affine), os.path.join(deriv, f"{stem}_desc-fault_mask.nii"))
    if field is not None:
        nib.save(nib.Nifti1Image(field.astype(np.float32), ref.affine), os.path.join(deriv, f"{stem}_desc-fault_fieldmap.nii"))
    return write_labels(os.path.join(out_dir, "artefacts.json"), labels,
                        {"Source": os.path.abspath(bids_dir), "Seed": seed})


# ---------------------------------------------------------------- CLI

def _parse_effect(spec):
    """``name`` or ``name:key=value,key=value`` (values parsed as JSON, else kept as strings)."""
    name, _, rest = spec.partition(":")
    kw = {}
    items, depth, cur = [], 0, ""
    for ch in rest:  # split on commas outside brackets
        depth += (ch in "[{") - (ch in "]}")
        if ch == "," and depth == 0:
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    for item in filter(None, items):
        k, _, v = item.partition("=")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    if name != "sidecar" and name not in EFFECTS:
        raise SystemExit(f"unknown effect {name!r}; choose from {sorted(EFFECTS)} or 'sidecar'")
    return name, kw


def add_subparser(subparsers):
    """Register ``qsm-forward artefacts``."""
    p = subparsers.add_parser("artefacts", help="Copy a phantom and inject artefacts with known labels (qsm_forward.artefacts)")
    p.add_argument("bids", help="BIDS directory of a qsm-forward phantom")
    p.add_argument("out", help="Output BIDS directory (must not exist)")
    p.add_argument("--subject", default="1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--effect", action="append", required=True, metavar="NAME[:k=v,...]",
                   help='Effect to apply, in order (repeatable), e.g. interecho_motion:rot_deg=2.5,trans_mm=1 '
                        'or sidecar:te_factor=0.001. Values are JSON (lists as [0,0,1]).')
    return p


def run_cli(args):
    path = inject_bids(args.bids, args.out, [_parse_effect(s) for s in args.effect], subject=args.subject,
                       seed=args.seed)
    print(f"  labels: {path}")
