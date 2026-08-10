"""
Multi-compartment hollow-cylinder white-matter GRE signal model
================================================================

Standalone research prototype for the QSM-CI chi-separation phantom. De-risking
exercise: does a *biophysical* multi-compartment signal make the fibre-to-B0
angle theta recoverable from a SINGLE-orientation multi-echo GRE acquisition?
(The shipped phantom's orientation-modulated *scalar* chi + mono-exponential
decay does NOT -- theta is inert in the signal shape there.)

Do NOT wire this into qsm_forward/ yet. Prototype only.

----------------------------------------------------------------------------
THE MODEL (three water pools in a myelinated-axon voxel)
----------------------------------------------------------------------------
Wharton & Bowtell (2012, PNAS 109:18559) model the myelin sheath as an infinite
hollow cylinder of material with a cylindrically symmetric, radially-oriented
anisotropic susceptibility tensor, decomposed into an isotropic part chi_I and
an anisotropic part chi_A. Three water pools each get a volume fraction f, a
transverse relaxation time T2, and a theta-dependent resonance frequency offset:

    S(TE) = sum_p  f_p * exp(-TE / T2_p) * exp(i * 2*pi * df_p(theta) * TE)

theta = angle between the fibre axis and B0.

FREQUENCY OFFSETS (Hz), hollow-cylinder closed forms.
w0 = gamma_bar * B0, gamma_bar = 42.577 MHz/T (so w0 is Hz per unit susceptibility).

1) INTRA-AXONAL (interior of the hollow cylinder). The field inside an infinite
   cylinder is spatially UNIFORM, so the axonal pool has a single offset driven
   ENTIRELY by the anisotropic susceptibility:

       df_A(theta) = w0 * (3/4) * chi_A * ln(1/g) * sin^2(theta)          [Hz]

   sin^2(theta) dependence; vanishes as g->1 (no myelin) or chi_A->0. This is the
   term that makes theta recoverable from the long-T2 axonal pool.
   SOURCE: Wharton & Bowtell 2012 hollow-cylinder interior result (SI eqs.),
   restated by Yablonskiy & Sukstanskii 2014 (MRM 71:2059) and Nam et al. 2015
   (NeuroImage 116:214). ASSUMPTION: g enters via ln(1/g); g=0.7 -> ln(1/0.7)=0.357.

2) EXTRA-AXONAL (extracellular) water. Mean field over the extra-axonal space is
   ~0 (isotropic part cancels outside an infinite cylinder; anisotropic part is
   higher order). Reference frequency:  df_E(theta) = 0.
   SOURCE: Wharton & Bowtell 2012.

3) MYELIN water (thin aqueous layers in the sheath). Large, strongly
   orientation-dependent shift. Sheath-averaged annular field (W&B 2012, Table 1
   annular region + Lorentzian sphere correction) plus isotropic exchange E:

       df_M(theta) = w0 * [ (chi_I/2)(2/3 - sin^2 theta)
                            + chi_A(1/12 - (5/12) sin^2 theta)
                            + (3/4) chi_A ln(1/g) sin^2 theta
                            + E ]                                          [Hz]

   Strong sin^2(theta) dependence; proportional to chi_I and chi_A; a constant
   exchange offset E (~0.02 ppm) survives at all theta; anisotropic part vanishes
   as chi_A->0 and g->1. SOURCE: Wharton & Bowtell 2012 PNAS Table 1; E from
   their Table 3.

----------------------------------------------------------------------------
SEED PARAMETERS (verified vs literature; see REPORT.md)
----------------------------------------------------------------------------
  chi_I = -0.10 ppm   (W&B Table 3: -0.06+-0.02; chi-sep/Lee sims use -0.1)
  chi_A = -0.10 ppm   (W&B Table 3: -0.12+-0.02; seed -0.1)
  E     =  0.02 ppm   (W&B Table 3: 0.01; sims use 0.02)
  g     =  0.7        (W&B 0.7-0.8)
  T2_M  =  10 ms      (W&B ~10 ms)
  T2_A  =  64 ms ; T2_E = 48 ms  (long pool 36-64 ms)
  MWF   =  0.12       (W&B 12.6+-2%)
  gamma_bar = 42.577 MHz/T
All frequencies scale linearly with B0; tested at 3 T and 7 T.
"""

from __future__ import annotations
import numpy as np

GAMMA_BAR = 42.577e6  # Hz/T (1H gyromagnetic ratio / 2pi)


class WMParams:
    """White-matter hollow-cylinder parameters (single fibre population)."""

    def __init__(
        self,
        chi_I=-0.06e-6,
        chi_A=-0.10e-6,
        E=0.02e-6,
        g=0.7,
        T2_M=10e-3,
        T2_A=64e-3,
        T2_E=48e-3,
        MWF=0.12,
        f_axon=0.55,      # axonal fraction of the NON-myelin water
        R2p_meso=0.0,     # optional common mesoscopic R2' (Hz)
    ):
        self.chi_I = chi_I
        self.chi_A = chi_A
        self.E = E
        self.g = g
        self.T2_M = T2_M
        self.T2_A = T2_A
        self.T2_E = T2_E
        self.MWF = MWF
        self.f_axon = f_axon
        self.R2p_meso = R2p_meso

    def fractions(self):
        fM = self.MWF
        rest = 1.0 - fM
        fA = rest * self.f_axon
        fE = rest * (1.0 - self.f_axon)
        return fM, fA, fE


def _w0(B0):
    return GAMMA_BAR * B0


def freq_axon(theta, B0, p: WMParams):
    """Intra-axonal water frequency offset (Hz): w0*(3/4)*chi_A*ln(1/g)*sin^2(theta)."""
    s2 = np.sin(theta) ** 2
    return _w0(B0) * 0.75 * p.chi_A * np.log(1.0 / p.g) * s2


def freq_extra(theta, B0, p: WMParams):
    """Extra-axonal frequency offset (Hz): reference (~0)."""
    theta = np.asarray(theta, dtype=float)
    return np.zeros_like(theta)


def freq_myelin(theta, B0, p: WMParams):
    """Myelin (sheath) water frequency offset (Hz), hollow-cylinder annular avg + E."""
    s2 = np.sin(theta) ** 2
    w0 = _w0(B0)
    iso_term = p.chi_I * (2.0 / 3.0 - s2) / 2.0
    aniso_term = p.chi_A * (1.0 / 12.0 - 5.0 / 12.0 * s2)
    ln_term = 0.75 * p.chi_A * np.log(1.0 / p.g) * s2
    return w0 * (iso_term + aniso_term + ln_term + p.E)


def compartment_freqs(theta, B0, p: WMParams):
    return (freq_myelin(theta, B0, p),
            freq_axon(theta, B0, p),
            freq_extra(theta, B0, p))


def gre_signal(TEs, theta, B0, p: WMParams, S0=1.0, bulk_freq=0.0):
    """Complex multi-echo GRE signal (single voxel)."""
    TEs = np.asarray(TEs, dtype=float)
    fM, fA, fE = p.fractions()
    dfM, dfA, dfE = compartment_freqs(theta, B0, p)

    def pool(f, T2, df):
        return f * np.exp(-TEs / T2) * np.exp(2j * np.pi * df * TEs)

    S = pool(fM, p.T2_M, dfM) + pool(fA, p.T2_A, dfA) + pool(fE, p.T2_E, dfE)
    S = S * np.exp(-p.R2p_meso * TEs) * np.exp(2j * np.pi * bulk_freq * TEs)
    return S0 * S


def gre_magnitude(TEs, theta, B0, p: WMParams, S0=1.0):
    return np.abs(gre_signal(TEs, theta, B0, p, S0=S0))


def spin_echo_magnitude(TEs, p: WMParams, S0=1.0):
    """Spin-echo magnitude: only T2 decay, NO frequency terms (refocused)."""
    TEs = np.asarray(TEs, dtype=float)
    fM, fA, fE = p.fractions()
    S = (fM * np.exp(-TEs / p.T2_M)
         + fA * np.exp(-TEs / p.T2_A)
         + fE * np.exp(-TEs / p.T2_E))
    return S0 * S


def effective_R2star(TEs, theta, B0, p: WMParams):
    """Fit log|S| ~ a - R2*_eff*TE; return R2*_eff (Hz)."""
    mag = gre_magnitude(TEs, theta, B0, p)
    A = np.vstack([TEs, np.ones_like(TEs)]).T
    slope, _ = np.linalg.lstsq(A, np.log(mag), rcond=None)[0]
    return -slope
