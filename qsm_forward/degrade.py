"""Controlled degradation of the derived χ-separation inputs (local field, χ, R2', per-echo χ).

A χ-separation method is normally fed inputs that a QSM pipeline derived from the measured signal:
a local field, a susceptibility map, R2' and (for DECOMPOSE-type methods) per-echo χ. Oracle inputs
(the phantom's ground truth) favour methods that trust their inputs; one specific QSM pipeline's
outputs bake that pipeline's own errors into the benchmark. This module writes inputs in between:
the ground truth with the error traits of real pipelines added term by term, each term a documented,
calibrated parameter, seeded and reproducible.

Levels (:data:`LEVELS`, parameters in :data:`LEVEL_PARAMS`):

  L0  oracle: the ground truth (nothing is written; the phantom's own files are L0).
  L1  measurement noise only. One realisation of per-echo phase noise, drawn from the dataset's
      SNR (sigma_i = max|S_i| / PeakSNR, qsm-forward's ``add_noise``) and the shipped magnitudes, is
      propagated analytically through the inverse-variance-weighted multi-echo field fit (through
      the origin): the ideal field noise n_f. Real field steps leave more, and correlated, noise
      (phase-offset removal and echo combination spread each voxel's phase noise), so n_f is shaped
      in k-space to the measured spectrum, W(|k|)^2 = eta_f^2 + A / (1 + (|k|/k_c)^2). χ gets the
      SAME shaped noise through the inversion's noise transfer
      G(k) = eta_chi * exp(-(|k|/k_hn)^2) * sign(D) / max(|D|, t), D = 1/3 - cos^2(psi), so methods
      reading the field and methods reading χ see one consistent noise realisation. R2' is fitted
      (mono-exponential, ARLO for evenly spaced echoes as QSMxT's ``r2star``) from the dataset's
      own noisy GRE and spin-echo magnitudes.
  L2  + what real pipelines do to the signal: the field step measures the phantom's sub-voxel
      (non-dipolar) frequency on top of the dipole field (``*_desc-subvoxel_fieldmap.nii``, written
      by :mod:`qsm_forward.chisep` for the LUT signal models) and loses the lowest spatial
      frequencies, R_f(|k|) = 1 - a_f exp(-(|k|/k_f)^2); χ gets a generic inversion transfer
      H(k) = g * D^2/(D^2 + lambda) * (1 - a exp(-(|k|/k_0)^2)) * exp(-(|k|/k_h)^2) of the ground
      truth (magic-cone attenuation, low-|k| loss, high-|k| smoothing) plus the sub-voxel frequency
      through the same inversion, H/D.
  L3  + additive errors: residual smooth polynomial backgrounds (field, χ), a smooth χ error
      that decays from the mask boundary inwards, the brain-mask erosion of background-field
      removal, and χ referenced to its mean over the (eroded) mask, plus an optional offset.

The per-echo χ (``Chimap-echoes``) gets the same transfer and additive terms and its own echo's
phase noise: the ground truth's echo dependence passes the same transfer as χ (unchanged where the
transfer is the identity) and the noise grows at short echo times, as in per-echo QSM.

k is in cycles/mm, psi is the angle between k and B0. The level parameters were calibrated on real
QSM pipelines (QSMxT: iLSQR, TKD, NLTV, MEDI, TFI, iR2QSM; the median across pipelines) run on the
meso-3t phantom's noisy signal, and validated on held-out pipelines and phantoms (wm-micro-7t, the
2019-challenge-based in-silico head); the ranges across pipelines are in :data:`CALIBRATION_SPREAD`.

Known limits of the model: every term is linear or additive, so the non-linear errors of real
inversions (streaking from strong calcification and vessel sources, solver-specific texture) are
not reproduced, and L3 inputs are somewhat cleaner than the median pipeline (higher XSIM, lighter
error tails). The field-step terms describe V-SHARP (four of the six pipelines) and PDF (MEDI).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import asdict, dataclass, replace
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import fft as sfft
from scipy import ndimage

GAMMA_BAR = 42.58  # MHz/T
LEVELS = ("L0", "L1", "L2", "L3")
R2_FITS = ("arlo", "loglinear")
REFERENCES = ("none", "mean")


@dataclass(frozen=True)
class DegradationParams:
    """Every term of the degradation model. Lengths in mm, k in cycles/mm, maps in ppm."""
    noise: bool = True                 # measurement noise from the dataset's SNR (L1+)
    subvoxel_frequency: bool = False   # add the phantom's sub-voxel (non-dipolar) frequency (L2+)
    noise_scale_field: float = 1.0     # eta_f: scales the white part of the ideal field-fit noise
    field_noise_lowk: float = 0.0      # A: low-|k| correlated excess power, relative to the white power
    field_noise_lowk_width: float = 0.03  # k_c of the excess A / (1 + (|k|/k_c)^2)
    noise_scale_chi: float = 1.0       # eta_chi: scales the field noise propagated to χ
    noise_cone_threshold: float = 0.1  # t of the noise transfer sign(D)/max(|D|, t)
    noise_highk_cutoff: Optional[float] = None  # k_hn: high-|k| roll-off of the χ noise (denoising)
    field_lowk_loss: float = 0.0       # a_f of R_f
    field_lowk_width: float = 0.02     # k_f of R_f
    chi_gain: float = 1.0              # g
    cone_lambda: float = 0.0           # lambda of D^2/(D^2 + lambda); 0: no cone attenuation
    chi_lowk_loss: float = 0.0         # a of L
    chi_lowk_width: float = 0.02       # k_0 of L
    chi_highk_cutoff: Optional[float] = None  # k_h of S; None: no high-|k| roll-off
    reference: str = "none"            # "none" | "mean" (χ zero pinned to its mean over the mask)
    chi_offset_ppm: float = 0.0        # extra constant added to χ after referencing
    background_poly_order: int = 0     # order of the random residual-background polynomials
    background_field_ppm: float = 0.0  # RMS of the field's residual background over the mask
    background_chi_ppm: float = 0.0    # RMS of χ's residual low-order error over the mask
    mask_erosion_mm: float = 0.0       # erosion of the brain mask (background-field removal)
    edge_error_ppm: float = 0.0        # RMS of the extra χ error at the (eroded) mask boundary
    edge_error_width_mm: float = 3.0   # its e-folding distance into the brain
    r2_fit: str = "arlo"               # R2* and R2 fit: "arlo" | "loglinear"

    def __post_init__(self):
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")
        if self.r2_fit not in R2_FITS:
            raise ValueError(f"r2_fit must be one of {R2_FITS}, got {self.r2_fit!r}")
        if self.noise and self.noise_cone_threshold <= 0:
            raise ValueError("noise needs a positive noise_cone_threshold (χ noise passes a regularised inverse)")
        for name in ("field_lowk_loss", "chi_lowk_loss"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cone_lambda < 0 or self.mask_erosion_mm < 0 or self.background_poly_order < 0:
            raise ValueError("cone_lambda, mask_erosion_mm and background_poly_order must be >= 0")


# Calibrated on real pipelines (see the module docstring): the median across iLSQR, TKD, NLTV, MEDI,
# TFI and iR2QSM (QSMxT main 9cfe3f9, QSM.rs 651eb5d) on meso-3t's noisy signal and its noiseless twin.
# L1 keeps only the noise terms of L2; L2 adds the transfer; L3 the additive terms. chi_lowk_width (k_0) is
# the stage-2 value: the median of the per-pipeline k_0 that reproduces each pipeline's deep-grey-matter
# contrast ratio (the joint transfer fit trades k_0 against the gain and the cone term).
_NOISE = dict(noise=True, noise_scale_field=1.924, field_noise_lowk=175.2, field_noise_lowk_width=0.02112,
              noise_scale_chi=1.066, noise_cone_threshold=0.1686, noise_highk_cutoff=0.7841)
_TRANSFER = dict(subvoxel_frequency=True, field_lowk_loss=0.958, field_lowk_width=0.05284, chi_gain=1.055,
                 cone_lambda=0.01782, chi_lowk_loss=1.0, chi_lowk_width=0.02656, chi_highk_cutoff=0.9557)
_ADDITIVE = dict(reference='mean', chi_offset_ppm=0.0, background_poly_order=2,
                 background_field_ppm=0.0005975, background_chi_ppm=0.001379, mask_erosion_mm=2.0,
                 edge_error_ppm=0.02283, edge_error_width_mm=3.021)
LEVEL_PARAMS = {
    "L0": DegradationParams(noise=False),
    "L1": DegradationParams(**_NOISE),
    "L2": DegradationParams(**_NOISE, **_TRANSFER),
    "L3": DegradationParams(**_NOISE, **_TRANSFER, **_ADDITIVE),
}
# Range of each calibrated parameter across the six pipelines (levels can bracket it).
CALIBRATION_SPREAD = {
    "noise_scale_field": {"min": 1.741, "max": 1.924},
    "field_noise_lowk": {"min": 34.02, "max": 175.2},
    "field_noise_lowk_width": {"min": 0.02112, "max": 0.02563},
    "noise_scale_chi": {"min": 0.7588, "max": 1.481},
    "noise_cone_threshold": {"min": 0.1438, "max": 0.6528},
    "noise_highk_cutoff": {"min": 0.4341, "max": 20.0},
    "field_lowk_loss": {"min": 0.958, "max": 1.0},
    "field_lowk_width": {"min": 0.04053, "max": 0.05284},
    "chi_gain": {"min": 0.8888, "max": 1.19},
    "cone_lambda": {"min": 0.003062, "max": 0.04988},
    "chi_lowk_loss": {"min": 0.9148, "max": 1.0},
    "chi_lowk_width": {"min": 0.02416, "max": 0.02894},
    "chi_highk_cutoff": {"min": 0.4095, "max": 20.0},
    "background_field_ppm": {"min": 0.0002776, "max": 0.0005975},
    "background_chi_ppm": {"min": 0.001052, "max": 0.002694},
    "mask_erosion_mm": {"min": 0.0, "max": 7.0},
    "edge_error_ppm": {"min": 0.0, "max": 0.02546},
    "edge_error_width_mm": {"min": 1.959, "max": 8.798},
}
CALIBRATION_SOURCE = ("median of QSMxT iLSQR/TKD/NLTV/MEDI/TFI/iR2QSM on qsm-forward meso-3t (noisy + noiseless "
                      "twin), QSMxT main 9cfe3f9; validated on wm-micro-7t and the 2019-challenge head")


def _active_spread(p: DegradationParams) -> dict:
    """The across-pipeline range of every calibrated term ``p`` switches on (written to the sidecar)."""
    off = DegradationParams(noise=False)
    return {k: v for k, v in CALIBRATION_SPREAD.items() if getattr(p, k) != getattr(off, k)}


def level_params(level: str, **overrides) -> DegradationParams:
    """The calibrated parameters of ``level`` (:data:`LEVELS`), with any field overridden."""
    if level not in LEVEL_PARAMS:
        raise KeyError(f"unknown degradation level {level!r}; choose from {LEVELS}")
    return replace(LEVEL_PARAMS[level], **overrides)


# ---------------------------------------------------------------- k-space model

def kspace_grid(shape, voxel, b0_dir=(0.0, 0.0, 1.0)):
    """|k| (cycles/mm) and cos^2(psi) on the ``rfftn`` grid of ``shape``."""
    b = np.asarray(b0_dir, float)
    b = b / np.linalg.norm(b)
    ks = [sfft.fftfreq(n, d=v) for n, v in zip(shape[:-1], voxel[:-1])] + [sfft.rfftfreq(shape[-1], d=voxel[-1])]
    kx, ky, kz = np.meshgrid(*ks, indexing="ij", sparse=True)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    kb = kx * b[0] + ky * b[1] + kz * b[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos2 = np.where(k2 > 0, kb ** 2 / k2, 0.0)
    return np.sqrt(k2), cos2


def _lowk(kmag, loss, width):
    return 1.0 - loss * np.exp(-(kmag / width) ** 2) if loss > 0 else np.ones_like(kmag)


def _highk(kmag, cutoff):
    return np.exp(-(kmag / cutoff) ** 2) if cutoff else np.ones_like(kmag)


def field_transfer(kmag, p: DegradationParams):
    """R_f(|k|): the field step's low-|k| loss."""
    return _lowk(kmag, p.field_lowk_loss, p.field_lowk_width)


def chi_transfer(kmag, cos2, p: DegradationParams):
    """H(k): what reaches χ of the ground truth's spatial frequency k."""
    d2 = (1.0 / 3.0 - cos2) ** 2
    cone = d2 / (d2 + p.cone_lambda) if p.cone_lambda > 0 else np.ones_like(d2)
    return p.chi_gain * cone * _lowk(kmag, p.chi_lowk_loss, p.chi_lowk_width) * _highk(kmag, p.chi_highk_cutoff)


def data_transfer(kmag, cos2, p: DegradationParams):
    """What reaches χ of a field component that is not the dipole field of χ (e.g. the sub-voxel
    frequency): the inversion's data transfer H/D = g L S D/(D^2 + lambda), lambda the signal
    cone_lambda (noise_cone_threshold^2 when that is 0)."""
    d = 1.0 / 3.0 - cos2
    lam = p.cone_lambda if p.cone_lambda > 0 else p.noise_cone_threshold ** 2
    return (p.chi_gain * d / (d ** 2 + lam) * _lowk(kmag, p.chi_lowk_loss, p.chi_lowk_width)
            * _highk(kmag, p.chi_highk_cutoff))


def noise_transfer(kmag, cos2, p: DegradationParams):
    """G(k): what reaches χ of the field noise at k: a truncated inverse sign(D)/max(|D|, t), largest
    on the magic cone (streaks), with its own high-|k| roll-off (regularised and learned inversions
    suppress noise more than signal). A Tikhonov D/(D^2 + lambda) would vanish on the cone, where
    the measured noise of real pipelines peaks. No low-|k| loss: background-field removal takes out
    harmonic fields, not noise."""
    d = 1.0 / 3.0 - cos2
    sgn = np.where(d >= 0, 1.0, -1.0)
    return p.noise_scale_chi * sgn / np.maximum(np.abs(d), p.noise_cone_threshold) * _highk(kmag, p.noise_highk_cutoff)


def field_noise_shaping(kmag, p: DegradationParams):
    """W(|k|): amplitude spectrum that turns the ideal (white) field-fit noise into the field step's
    noise: eta_f^2 white plus a low-|k| correlated excess A / (1 + (|k|/k_c)^2) (phase-offset removal
    and echo combination spread each voxel's phase noise over a neighbourhood)."""
    w2 = p.noise_scale_field ** 2 + (p.field_noise_lowk / (1.0 + (kmag / p.field_noise_lowk_width) ** 2)
                                     if p.field_noise_lowk > 0 else 0.0)
    return np.sqrt(w2)


def _padded_shape(shape, pad=0.5):
    return tuple(sfft.next_fast_len(int(np.ceil(n * (1 + pad))), real=True) for n in shape)


def apply_transfer(arr, transfer_fn, voxel, b0_dir=(0.0, 0.0, 1.0), pad=0.5):
    """``ifft(T(k) * fft(arr))`` on a zero-padded grid (``pad`` × each side's length added), cropped back.
    ``transfer_fn(kmag, cos2)`` builds T on the padded grid; ``arr`` may carry trailing echo axes."""
    arr = np.asarray(arr, float)
    sp = arr.shape[:3]
    ps = _padded_shape(sp, pad)
    kmag, cos2 = kspace_grid(ps, voxel, b0_dir)
    t = transfer_fn(kmag, cos2)
    out = np.empty_like(arr)
    flat = arr.reshape(sp + (-1,))
    oflat = out.reshape(sp + (-1,))
    for i in range(flat.shape[-1]):
        a = sfft.rfftn(flat[..., i], s=ps, workers=-1)
        oflat[..., i] = sfft.irfftn(a * t, s=ps, workers=-1)[:sp[0], :sp[1], :sp[2]]
    return out


def uniform_field(mask, voxel, b0_dir=(0.0, 0.0, 1.0)):
    """Field (ppm per ppm) of a uniform χ = 1 over ``mask`` (Lorentz-corrected dipole kernel):
    harmonic inside the mask, so background-field removal takes it out with χ's mean."""
    def dip(kmag, cos2):
        d = 1.0 / 3.0 - cos2
        d = np.where(kmag > 0, d, 0.0)
        return d
    return apply_transfer(mask.astype(float), dip, voxel, b0_dir, pad=1.0)


# ---------------------------------------------------------------- measurement noise

def echo_sigmas(magnitudes, peak_snr):
    """Per-echo complex noise SD, sigma_i = max|S_i| / PeakSNR (qsm-forward's ``add_noise``),
    estimated from the shipped (noisy) magnitude. 0 for a noiseless dataset."""
    if not np.isfinite(peak_snr):
        return np.zeros(len(magnitudes))
    return np.array([float(np.max(m)) / peak_snr for m in magnitudes])


def field_noise(magnitudes, tes, b0, sigmas, mask, rng):
    """One realisation of per-echo phase noise and its propagation through the inverse-variance
    weighted phase-vs-TE fit through the origin. Returns (n_f, n_e): the combined field noise and the
    per-echo field noise (echo axis last), ppm, zero outside ``mask``.

    Phase noise SD at echo i is sigma_i/|S_i| with |S_i|^2 ~ M_i^2 - 2 sigma_i^2 (Rician bias
    removed; floored at sigma_i^2)."""
    tes = np.asarray(tes, float)
    w2pi = 2 * np.pi * GAMMA_BAR * b0  # rad/s per ppm
    m = mask > 0
    num = np.zeros(int(m.sum()))
    den = np.zeros_like(num)
    n_e = np.zeros(mask.shape + (len(tes),), np.float32)
    for i, (mag, te, s) in enumerate(zip(magnitudes, tes, sigmas)):
        if s <= 0:
            continue
        a2 = np.maximum(np.asarray(mag, float)[m] ** 2 - 2 * s * s, s * s)
        n = rng.standard_normal(a2.size) * (s / np.sqrt(a2))
        w = a2 / (s * s)
        num += w * te * n
        den += w * te * te
        n_e[..., i][m] = n / (te * w2pi)
    n_f = np.zeros(mask.shape)
    ok = den > 0
    vals = np.zeros_like(num)
    vals[ok] = num[ok] / den[ok] / w2pi
    n_f[m] = vals
    return n_f, n_e


# ---------------------------------------------------------------- relaxation fits

def decay_rate(magnitudes, tes, method="arlo"):
    """Mono-exponential decay rate (s^-1) of a multi-echo magnitude series. ``arlo`` (Pei et al.
    2015, as QSMxT's r2star) needs >= 3 evenly spaced echoes and falls back to ``loglinear``."""
    tes = np.asarray(tes, float)
    mags = [np.asarray(m, float) for m in magnitudes]
    dt = np.diff(tes)
    if method == "arlo" and len(tes) >= 3 and np.allclose(dt, dt[0], rtol=1e-3):
        d = dt[0]
        ss = sd = dd = 0.0
        for i in range(len(mags) - 2):
            s = d / 3.0 * (mags[i] + 4 * mags[i + 1] + mags[i + 2])
            dl = mags[i] - mags[i + 2]
            ss = ss + s * s
            sd = sd + s * dl
            dd = dd + dl * dl
        num = ss + d / 3.0 * sd
        den = d / 3.0 * dd + sd
        with np.errstate(divide="ignore", invalid="ignore"):
            t2 = np.where(den != 0, num / den, np.inf)
            r = np.where((t2 > 0) & np.isfinite(t2), 1.0 / t2, 0.0)
        return r
    t = tes - tes.mean()
    acc = 0.0
    for m, ti in zip(mags, t):
        acc = acc + np.log(np.maximum(m, 1e-12)) * ti
    return -acc / (t ** 2).sum()


def fit_r2prime(gre_mags, gre_tes, se_mags, se_tes, mask, method="arlo"):
    """R2' = max(R2* - R2, 0) in the mask (s^-1), R2*/R2 mono-exponential fits."""
    r2s = decay_rate(gre_mags, gre_tes, method)
    r2 = decay_rate(se_mags, se_tes, method)
    return np.where(mask > 0, np.clip(np.maximum(r2s, 0) - np.maximum(r2, 0), 0, None), 0.0)


# ---------------------------------------------------------------- additive terms

def random_polynomial(mask, order, rms, rng):
    """A random polynomial of degree 1..``order`` in the mask's normalised coordinates, zero-mean and
    of RMS ``rms`` over the mask (0 outside)."""
    out = np.zeros(mask.shape)
    m = mask > 0
    if order < 1 or rms <= 0 or not m.any():
        return out
    idx = np.nonzero(m)
    coords = []
    for ax in idx:
        lo, hi = ax.min(), ax.max()
        coords.append((ax - (lo + hi) / 2) / max((hi - lo) / 2, 1))
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                if 0 < i + j + k:
                    terms.append(coords[0] ** i * coords[1] ** j * coords[2] ** k)
    v = np.stack(terms, -1) @ rng.standard_normal(len(terms))
    v -= v.mean()
    v *= rms / max(np.sqrt(np.mean(v ** 2)), 1e-30)
    out[idx] = v
    return out


def edge_error(mask, voxel, rms, width_mm, rng, smooth_mm=2.0):
    """Smooth random error (Gaussian-smoothed white noise, ``smooth_mm`` FWHM-ish sigma) whose RMS
    is ``rms`` at the mask boundary and decays as exp(-d/``width_mm``) with the distance d inside."""
    out = np.zeros(mask.shape)
    m = mask > 0
    if rms <= 0 or not m.any():
        return out
    v = ndimage.gaussian_filter(rng.standard_normal(mask.shape), smooth_mm / np.asarray(voxel, float))
    v /= max(v[m].std(), 1e-30)
    d = ndimage.distance_transform_edt(m, sampling=voxel)
    out[m] = rms * v[m] * np.exp(-(d[m] - d[m].min()) / width_mm)
    return out


def erode_mask(mask, mm, voxel):
    """Erode by a ball of radius ``mm``."""
    m = mask > 0
    if mm <= 0:
        return m
    r = np.ceil(mm / np.asarray(voxel, float)).astype(int)
    g = np.ogrid[tuple(slice(-ri, ri + 1) for ri in r)]
    ball = sum((gi * v / mm) ** 2 for gi, v in zip(g, voxel)) <= 1.0
    return ndimage.binary_erosion(m, structure=ball)


# ---------------------------------------------------------------- the model

def degrade(gt: dict, acq: dict, p: DegradationParams, seed: int = 0) -> dict:
    """Degraded inputs from the ground truth.

    ``gt``: ``localfield``, ``chimap``, ``mask`` (3D), optional ``chimap_echoes`` (3D + echo axis)
    and ``r2prime`` (the oracle, returned unchanged without noise).
    ``acq``: ``magnitudes`` (GRE, list of 3D), ``tes`` (s), ``b0`` (T), ``b0_dir``, ``voxel`` (mm),
    ``peak_snr``, optional ``se_magnitudes``/``se_tes``.
    Returns a dict with ``localfield``, ``chimap``, ``mask``, and ``r2prime`` / ``chimap_echoes`` when
    they can be made. The same ``seed`` gives the same noise at every level."""
    voxel = np.asarray(acq["voxel"], float)
    b0dir = acq.get("b0_dir", (0.0, 0.0, 1.0))
    mask = gt["mask"] > 0
    out_mask = erode_mask(mask, p.mask_erosion_mm, voxel)
    tes = np.asarray(acq["tes"], float)
    rng_noise = np.random.default_rng([seed, 1])
    rng_bg = np.random.default_rng([seed, 2])

    n_f = np.zeros(mask.shape)
    n_e = np.zeros(mask.shape + (len(tes),))
    if p.noise:
        sig = echo_sigmas(acq["magnitudes"], acq.get("peak_snr", np.inf))
        n_f, n_e = field_noise(acq["magnitudes"], tes, acq["b0"], sig, mask, rng_noise)

    def chi_part(x):
        return apply_transfer(x * mask[..., None] if x.ndim == 4 else x * mask,
                              lambda k, c: chi_transfer(k, c, p), voxel, b0dir)

    def noise_part(x):
        return apply_transfer(x, lambda k, c: noise_transfer(k, c, p), voxel, b0dir)

    sub = gt.get("subvoxel_field") if p.subvoxel_frequency else None
    f_in = np.asarray(gt["localfield"], float) * mask
    if sub is not None:
        f_in = f_in + np.asarray(sub, float) * mask
    field = apply_transfer(f_in, lambda k, c: field_transfer(k, p), voxel, b0dir) \
        if p.field_lowk_loss > 0 else f_in
    if p.noise:
        shape_fn = lambda k, c: field_noise_shaping(k, p)
        n_f = apply_transfer(n_f, shape_fn, voxel, b0dir) * mask
        n_e = apply_transfer(n_e, shape_fn, voxel, b0dir) * mask[..., None]
    field = field + n_f
    chi = chi_part(np.asarray(gt["chimap"], float))
    chi_sub = None
    if sub is not None:
        chi_sub = apply_transfer(np.asarray(sub, float) * mask, lambda k, c: data_transfer(k, c, p), voxel, b0dir)
        chi = chi + chi_sub
    if p.noise:
        chi = chi + noise_part(n_f)
    echoes = None
    if "chimap_echoes" in gt or p.noise:
        ge = gt.get("chimap_echoes")
        if ge is None:
            ge = np.repeat(np.asarray(gt["chimap"], float)[..., None], len(tes), -1)
        echoes = chi_part(np.asarray(ge, float))
        if chi_sub is not None:
            echoes = echoes + chi_sub[..., None]
        if p.noise:
            echoes = echoes + noise_part(n_e)

    # L3: additive low-order errors, then the reference (pipelines pin χ's zero last)
    if p.background_poly_order and (p.background_field_ppm > 0 or p.background_chi_ppm > 0):
        bf = random_polynomial(out_mask, p.background_poly_order, p.background_field_ppm, rng_bg)
        bc = random_polynomial(out_mask, p.background_poly_order, p.background_chi_ppm, rng_bg)
        field = field + bf
        chi = chi + bc
        if echoes is not None:
            echoes = echoes + bc[..., None]
    if p.edge_error_ppm > 0:
        ee = edge_error(out_mask, voxel, p.edge_error_ppm, p.edge_error_width_mm, rng_bg)
        chi = chi + ee
        if echoes is not None:
            echoes = echoes + ee[..., None]
    if p.reference == "mean":
        mref = float(chi[out_mask].mean())
        chi = chi - mref
        # the field needs no matching term: the harmonic field of χ's mean is low-|k| and the field
        # step's R_f (L2) already removes it (measured: < 3 % of the real field steps' error)
        if echoes is not None:
            echoes = echoes - echoes[out_mask].mean(axis=0)
    if p.chi_offset_ppm:
        chi = chi + p.chi_offset_ppm
        if echoes is not None:
            echoes = echoes + p.chi_offset_ppm

    res = {"localfield": np.where(out_mask, field, 0.0), "chimap": np.where(out_mask, chi, 0.0),
           "mask": out_mask.astype(np.uint8)}
    if echoes is not None:
        res["chimap_echoes"] = np.where(out_mask[..., None], echoes, 0.0)
    if p.noise and acq.get("se_magnitudes") is not None:
        res["r2prime"] = np.where(out_mask, fit_r2prime(acq["magnitudes"], tes, acq["se_magnitudes"],
                                                        acq["se_tes"], out_mask, p.r2_fit), 0.0)
    elif gt.get("r2prime") is not None:
        res["r2prime"] = np.where(out_mask, gt["r2prime"], 0.0)
    return res


# ---------------------------------------------------------------- measurement (calibration)

def measure_transfer(est, gt, mask, voxel, b0_dir=(0.0, 0.0, 1.0), k_edges=None, n_cos=10):
    """Effective transfer of ``est`` relative to ``gt`` over ``mask``: both demeaned over the mask and
    zeroed outside, then H = sum Re(E G*) / sum |G|^2 per (|k|, |cos psi|) bin. Returns a dict with
    the bin edges, H, the GT power per bin and the radial / angular marginals."""
    m = mask > 0
    e = np.where(m, est - est[m].mean(), 0.0)
    g = np.where(m, gt - gt[m].mean(), 0.0)
    E = sfft.rfftn(e, workers=-1)
    G = sfft.rfftn(g, workers=-1)
    kmag, cos2 = kspace_grid(e.shape, voxel, b0_dir)
    kmag = np.broadcast_to(kmag, E.shape)
    acos = np.broadcast_to(np.sqrt(cos2), E.shape)
    # rfft stores half the spectrum: weight the non-redundant columns twice
    wt = np.full(E.shape[-1], 2.0)
    wt[0] = 1.0
    if e.shape[-1] % 2 == 0:
        wt[-1] = 1.0
    wt = np.broadcast_to(wt, E.shape)
    if k_edges is None:
        k_edges = np.array([0.0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    c_edges = np.linspace(0, 1, n_cos + 1)
    cross = (E * np.conj(G)).real * wt
    pg = (np.abs(G) ** 2) * wt
    pe = (np.abs(E) ** 2) * wt
    ki = np.clip(np.digitize(kmag, k_edges) - 1, 0, len(k_edges) - 2)
    ci = np.clip(np.digitize(acos, c_edges) - 1, 0, n_cos - 1)
    ok = (kmag > 0) & (kmag <= k_edges[-1])
    flat = (ki * n_cos + ci)[ok]
    nb = (len(k_edges) - 1) * n_cos
    sx = np.bincount(flat, cross[ok], nb).reshape(-1, n_cos)
    sg = np.bincount(flat, pg[ok], nb).reshape(-1, n_cos)
    se = np.bincount(flat, pe[ok], nb).reshape(-1, n_cos)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = sx / sg
        hr = sx.sum(1) / sg.sum(1)
        ha = sx.sum(0) / sg.sum(0)
    return {"k_edges": k_edges.tolist(), "cos_edges": c_edges.tolist(), "H": h, "gt_power": sg,
            "est_power": se, "cross": sx, "H_radial": hr, "H_angular": ha}


_FIT_KEYS = ("chi_gain", "cone_lambda", "chi_lowk_loss", "chi_lowk_width", "chi_highk_cutoff")


def fit_chi_transfer(meas: dict, fix: Optional[dict] = None):
    """Fit g, lambda, a, k_0, k_h of :func:`chi_transfer` to a :func:`measure_transfer` table, each bin
    weighted by its GT power. Returns (params dict, weighted R^2 of the fit)."""
    from scipy.optimize import least_squares
    ke = np.asarray(meas["k_edges"])
    ce = np.asarray(meas["cos_edges"])
    kc = 0.5 * (ke[1:] + ke[:-1])
    cc = 0.5 * (ce[1:] + ce[:-1])
    K, C = np.meshgrid(kc, cc ** 2, indexing="ij")
    h = meas["H"]
    w = np.sqrt(np.nan_to_num(meas["gt_power"]))
    ok = np.isfinite(h) & (w > 0)
    w = w / w[ok].max()
    fix = fix or {}
    x0 = {"chi_gain": 1.0, "cone_lambda": 0.01, "chi_lowk_loss": 0.4, "chi_lowk_width": 0.03, "chi_highk_cutoff": 0.6}
    lo = {"chi_gain": 0.1, "cone_lambda": 1e-5, "chi_lowk_loss": 0.0, "chi_lowk_width": 0.003, "chi_highk_cutoff": 0.05}
    hi = {"chi_gain": 3.0, "cone_lambda": 1.0, "chi_lowk_loss": 1.0, "chi_lowk_width": 0.3, "chi_highk_cutoff": 50.0}
    free = [k for k in _FIT_KEYS if k not in fix]

    def model(x):
        d = dict(fix)
        d.update(dict(zip(free, x)))
        return chi_transfer(K, C, DegradationParams(noise=False, **d)), d

    def resid(x):
        return ((model(x)[0] - h) * w)[ok]

    r = least_squares(resid, [x0[k] for k in free], bounds=([lo[k] for k in free], [hi[k] for k in free]))
    pred, d = model(r.x)
    ss = np.sum(((h - pred) * w)[ok] ** 2)
    st = np.sum(((h - np.average(h[ok], weights=w[ok] ** 2)) * w)[ok] ** 2)
    return {k: float(v) for k, v in d.items()}, float(1 - ss / st) if st > 0 else float("nan")


# ---------------------------------------------------------------- BIDS I/O

def _echo_files(anat, suffix):
    files = glob.glob(os.path.join(anat, f"*_echo-*_part-mag_{suffix}.nii*"))
    return sorted(files, key=lambda f: int(re.search(r"_echo-(\d+)_", f).group(1)))


def _sidecar(nii_path):
    return json.load(open(re.sub(r"\.nii(\.gz)?$", ".json", nii_path)))


def load_bids_inputs(bids_dir, subject="1"):
    """(gt, acq, stem, deriv_anat, ref_nii) of a qsm-forward phantom: the ground truth from
    ``derivatives/qsm-forward/sub-<subject>/anat`` and the acquisition from ``sub-<subject>/anat``."""
    deriv = os.path.join(bids_dir, "derivatives", "qsm-forward", f"sub-{subject}", "anat")
    raw = os.path.join(bids_dir, f"sub-{subject}", "anat")
    chis = [f for f in glob.glob(os.path.join(deriv, "*_Chimap.nii*")) if "_desc-" not in os.path.basename(f)]
    if not chis:
        raise FileNotFoundError(f"no ground-truth *_Chimap.nii in {deriv}")
    ref = nib.load(chis[0])
    stem = os.path.basename(chis[0]).split("_Chimap")[0]
    L = lambda name: np.asarray(nib.load(os.path.join(deriv, f"{stem}_{name}.nii")).dataobj, float)
    gt = {"chimap": ref.get_fdata(), "localfield": L("fieldmap-local"), "mask": L("mask") > 0.5}
    if os.path.exists(os.path.join(deriv, f"{stem}_R2prime.nii")):
        gt["r2prime"] = L("R2prime")
    if os.path.exists(os.path.join(deriv, f"{stem}_Chimap-echoes.nii")):
        gt["chimap_echoes"] = L("Chimap-echoes")
    if os.path.exists(os.path.join(deriv, f"{stem}_desc-subvoxel_fieldmap.nii")):
        gt["subvoxel_field"] = L("desc-subvoxel_fieldmap")
    gre = _echo_files(raw, "MEGRE")
    if not gre:
        raise FileNotFoundError(f"no multi-echo GRE magnitude in {raw}")
    side = [_sidecar(f) for f in gre]
    snr = side[0].get("PeakSNR", "inf")
    acq = {"magnitudes": [np.asarray(nib.load(f).dataobj, float) for f in gre],
           "tes": [s["EchoTime"] for s in side], "b0": float(side[0]["MagneticFieldStrength"]),
           "b0_dir": side[0].get("B0_dir", [0.0, 0.0, 1.0]),
           "voxel": [float(v) for v in ref.header.get_zooms()[:3]],
           "peak_snr": float("inf") if snr in ("inf", None) else float(snr)}
    se = _echo_files(raw, "MESE")
    if se:
        acq["se_magnitudes"] = [np.asarray(nib.load(f).dataobj, float) for f in se]
        acq["se_tes"] = [_sidecar(f)["EchoTime"] for f in se]
    return gt, acq, stem, deriv, ref


def degrade_bids(bids_dir, levels=("L1", "L2", "L3"), subject="1", seed=None, overrides=None) -> dict:
    """Write the degraded inputs of each level next to the ground truth, as
    ``<stem>_desc-degraded<L>_{fieldmap-local,Chimap,Chimap-echoes,R2prime,mask}.nii`` and
    ``<stem>_desc-degraded<L>_degradation.json`` (every parameter, the seed, the version and the
    calibration provenance). ``seed`` defaults to the phantom's own (``*_chisep.json``) or 42; every
    level shares it, so levels differ only by their added terms. ``overrides`` (dict) replaces
    fields of every level's parameters. Returns {level: {name: path}}."""
    from .qsm_forward import get_version
    gt, acq, stem, deriv, ref = load_bids_inputs(bids_dir, subject)
    if seed is None:
        meta = os.path.join(deriv, f"{stem}_chisep.json")
        seed = json.load(open(meta))["ChiSeparationPhantom"].get("seed", 42) if os.path.exists(meta) else 42
    written = {}
    for level in levels:
        p = level_params(level, **(overrides or {}))
        if level == "L0":
            written[level] = {}
            continue
        res = degrade(gt, acq, p, seed=seed)
        tag = f"desc-degraded{level}"
        paths = {}
        names = {"localfield": "fieldmap-local", "chimap": "Chimap", "chimap_echoes": "Chimap-echoes",
                 "r2prime": "R2prime", "mask": "mask"}
        for key, name in names.items():
            if key not in res:
                continue
            path = os.path.join(deriv, f"{stem}_{tag}_{name}.nii")
            dtype = np.uint8 if key == "mask" else np.float32
            nib.save(nib.Nifti1Image(res[key].astype(dtype), ref.affine), path)
            paths[key] = path
        side = {"DegradationLevel": level, "Seed": int(seed), "Parameters": asdict(p),
                "QsmForwardVersion": get_version(), "Calibration": CALIBRATION_SOURCE,
                "CalibrationSpread": _active_spread(p),
                "NoiseSD": {"PeakSNR": acq["peak_snr"],
                            "FieldPpmMedian": _median_field_sd(acq, gt["mask"]) if p.noise else 0.0},
                "Inputs": {k: os.path.basename(v) for k, v in paths.items()},
                "Description": _describe(level, p)}
        jp = os.path.join(deriv, f"{stem}_{tag}_degradation.json")
        with open(jp, "w") as f:
            json.dump(side, f, indent=2)
        paths["metadata"] = jp
        written[level] = paths
    return written


def _median_field_sd(acq, mask):
    """Median over the mask of the analytic field-fit noise SD (ppm), before ``noise_scale_field``."""
    sig = echo_sigmas(acq["magnitudes"], acq["peak_snr"])
    if not np.any(sig > 0):
        return 0.0
    m = mask > 0
    den = np.zeros(int(m.sum()))
    for mag, te, s in zip(acq["magnitudes"], acq["tes"], sig):
        a2 = np.maximum(np.asarray(mag, float)[m] ** 2 - 2 * s * s, s * s)
        den += a2 / (s * s) * te * te
    return float(np.median(1.0 / np.sqrt(den) / (2 * np.pi * GAMMA_BAR * acq["b0"])))


def _describe(level, p):
    parts = {"L1": "measurement noise only (field-fit noise from the dataset's SNR; the same noise through "
                   "a regularised inverse in χ; R2' fitted from the noisy GRE and spin echo)",
             "L2": "L1 + the linear transfer of real QSM pipelines (low-|k| loss of the field step; "
                   "magic-cone attenuation, low-|k| loss and smoothing of the inversion)",
             "L3": "L2 + low-order additive errors (χ mean reference and the matching harmonic field, "
                   "residual smooth backgrounds, brain-mask erosion)"}
    return parts.get(level, "custom")
