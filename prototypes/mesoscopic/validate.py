"""Static-dephasing checks of the mesoscopic simulator against analytic limits.

1. Poisson spheres (3D):   S(t) = exp(-zeta f_s(dw t)), long-time R2' (Y&H 1994),
                           short-time -ln S = (2/5) zeta (dw t)^2.
2. Poisson parallel solid cylinders (2D), theta sweep: S(t) vs Y&H exterior f_c plus
   the uniform interior term; long-time R2' = zeta dw sin^2(theta)/2;
   short-time -ln|S| = zeta (dw_c t)^2 / 4.
3. Isolated hollow cylinder (Wharton & Bowtell 2012): compartment mean frequency
   differences myelin-extra and axon-extra vs the closed forms in qsm_forward.

Usage:  python validate.py [--out DIR] [--quick]
Writes validation.json (+ printed table) into DIR.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso  # noqa: E402

GBAR = meso.GAMMA / (2 * np.pi) * 1e3  # Hz/T


def hist_signal(values, w_t, nbins=40000):
    """mean exp(i*v*w_t) via a fine histogram of v (exact up to bin width)."""
    lo, hi = values.min(), values.max()
    h, e = np.histogram(values, bins=nbins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    h = h / h.sum()
    return np.exp(1j * np.outer(w_t, c)) @ h


def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def check_spheres(n=256, rad_px=4.0, zeta=0.02, seeds=(0, 1), dchi=1.0, B0=3.0):
    L = 1.0
    R = rad_px * L / n
    dw = meso.dw_sphere(dchi, B0)
    x = np.concatenate([np.linspace(0.05, 1.5, 15), np.linspace(2, 30, 15)])
    t = x / dw
    S = 0
    for sd in seeds:
        c = meso.place_spheres(L, zeta, R, sd)
        Sph = meso.raster_spheres(c, R, L, n)
        zeta_eff = float(Sph.mean())
        Gzz = meso.sphere_basis(Sph, L)[2]
        S = S + hist_signal(Gzz.ravel().astype(np.float64) * dchi, meso.GAMMA * B0 * 1e-6 * t)
    S = S / len(seeds)
    f = meso.yh_f_sphere(x)
    S_th = np.exp(-zeta_eff * f)
    # long-time slope over x in [10, 30]
    m = x >= 10
    slope_sim = -fit_line(t[m], np.log(np.abs(S[m])))[0]
    slope_th = -fit_line(t[m], np.log(np.abs(S_th[m])))[0]
    r2p_yh = meso.yh_r2p_sphere(zeta_eff, dchi, B0)
    # short-time quadratic coefficient: -ln|S| / (zeta (dw t)^2) -> 2/5
    ms = x <= 0.3
    quad_sim = float(np.mean(-np.log(np.abs(S[ms])) / (zeta_eff * x[ms] ** 2)))
    quad_th = float(np.mean(-np.log(np.abs(S_th[ms])) / (zeta_eff * x[ms] ** 2)))
    err = np.abs(S - S_th) / np.abs(1 - S_th).max()
    return dict(zeta_target=zeta, zeta_eff=zeta_eff, n_grid=n, radius_px=rad_px,
                R2p_sim_per_s=slope_sim * 1e3, R2p_YH_exactcurve_per_s=slope_th * 1e3,
                R2p_YH_formula_per_s=r2p_yh * 1e3,
                rel_err_R2p_vs_formula=slope_sim / r2p_yh - 1,
                short_time_coef_sim=quad_sim, short_time_coef_exactcurve=quad_th,
                short_time_coef_asymptote=0.4,
                max_abs_dev_S_over_signal_loss=float(err.max()),
                phase_sim_max_rad=float(np.abs(np.angle(S)).max()),
                phase_theory_max_rad=float(np.abs(np.angle(S_th)).max()))


def check_cylinders(n=4096, rad_px=6.0, zeta=0.02, thetas_deg=(90, 60, 30), dchi=1.0,
                    B0=3.0, seed=0):
    L = 1.0
    R = rad_px * L / n
    rng = np.random.default_rng(seed)
    N = int(round(zeta / (np.pi * R ** 2)))
    xy = rng.uniform(0, L, (N, 2))
    pk = meso.AxonPacking(L, xy[:, 0], xy[:, 1], np.full(N, R), np.zeros(N))
    # solid isotropic cylinders = "sheath" with r_in = 0, chi_perp = dchi, chi_a = 0
    cxx, cxy, czz, lab = meso.raster_axons(pk, n, dchi, 0.0)
    zeta_eff = float(czz.mean() / dchi)
    M1, M2 = meso.myelin_basis(cxx, cxy, czz, L)
    out = []
    x = np.concatenate([np.linspace(0.05, 1.0, 12), np.linspace(2, 30, 15)])
    fx = meso.yh_f_cylinder(x)
    for th in np.deg2rad(thetas_deg):
        s2 = np.sin(th) ** 2
        fld = (s2 * M1 + M2).ravel().astype(np.float64)   # ppm
        dw = meso.dw_cylinder(dchi, B0, th)
        t = x / dw
        S = hist_signal(fld, meso.GAMMA * B0 * 1e-6 * t)
        # theory: Poisson exterior * interior term * global mean shift (D(0)=0)
        w_in = meso.GAMMA * B0 * 1e-6 * dchi * (2 - 3 * s2) / 6
        f_tot = fx + (1 - np.exp(1j * w_in * t))
        S_th = np.exp(-zeta_eff * f_tot) * np.exp(-1j * zeta_eff * w_in * t)
        m = x >= 10
        slope = -fit_line(t[m], np.log(np.abs(S[m])))[0]
        r2p = meso.yh_r2p_cylinder(zeta_eff, dchi, B0, th)
        ms = x <= 0.3
        quad = float(np.mean(-np.log(np.abs(S[ms])) / (zeta_eff * x[ms] ** 2)))
        quad_th = float(np.mean(-np.log(np.abs(S_th[ms])) / (zeta_eff * x[ms] ** 2)))
        err = np.abs(S - S_th) / np.abs(1 - S_th).max()
        out.append(dict(theta_deg=float(np.rad2deg(th)), zeta_eff=zeta_eff,
                        R2p_sim_per_s=slope * 1e3, R2p_YH_per_s=r2p * 1e3,
                        rel_err_R2p=slope / r2p - 1,
                        short_time_coef_sim=quad, short_time_coef_exactcurve=quad_th,
                        short_time_coef_exterior_asymptote=0.25,
                        max_abs_dev_S_over_signal_loss=float(err.max())))
    return out


def wb_freqs(theta, B0, chi_i, chi_a, g):
    """Closed forms from qsm_forward (E = 0), Hz."""
    w0 = GBAR * B0 * 1e-6
    s2 = np.sin(theta) ** 2
    fM = w0 * (chi_i * (2 / 3 - s2) / 2 + chi_a * (1 / 12 - 5 / 12 * s2)
               + 0.75 * chi_a * np.log(1 / g) * s2)
    fA = w0 * 0.75 * chi_a * np.log(1 / g) * s2
    return fM, fA, 0.0


def check_hollow(n=4096, r_out=0.08, g=0.7, chi_i=-0.1, chi_a=-0.1, B0=7.0,
                 thetas_deg=(0, 30, 45, 60, 90), extra_ring=(1.5, 4.0)):
    """Single hollow cylinder in a periodic unit box. Extra-axonal reference = mean
    over an annulus r_out*[1.5, 4] (the 2D-dipole exterior field averages to zero over
    any centred annulus; avoids the periodic-image / D(0) offset)."""
    L = 1.0
    pk = meso.AxonPacking(L, np.array([0.5]), np.array([0.5]), np.array([r_out]),
                          np.array([g * r_out]))
    res = []
    for conv in ("wb_diag(1,-1/2,-1/2)", "perp_plus_radial"):
        if conv == "perp_plus_radial":       # chi_rr = chi_I + chi_A, others chi_I
            cp, ca = chi_i, chi_a
        else:  # W&B 2012: chi = chi_I*I + chi_A*diag(1, -1/2, -1/2) in (r, phi, z)
            cp, ca = chi_i - chi_a / 2, 1.5 * chi_a
        cxx, cxy, czz, lab = meso.raster_axons(pk, n, cp, ca)
        M1, M2 = meso.myelin_basis(cxx, cxy, czz, L)
        c = (np.arange(n) + 0.5) / n - 0.5
        X, Y = np.meshgrid(c, c, indexing="ij")
        d = np.hypot(X, Y)
        ring = (d > extra_ring[0] * r_out) & (d < extra_ring[1] * r_out)
        my = lab == meso.MYELIN
        ax = lab == meso.AXON
        for thd in thetas_deg:
            th = np.deg2rad(thd)
            f = (np.sin(th) ** 2 * M1 + M2) * GBAR * B0 * 1e-6  # Hz
            fe = f[ring].mean()
            sim_M, sim_A = f[my].mean() - fe, f[ax].mean() - fe
            fM, fA, _ = wb_freqs(th, B0, chi_i, chi_a, g)
            res.append(dict(convention=conv, theta_deg=thd,
                            dfM_minus_E_sim_Hz=float(sim_M), dfM_minus_E_WB_Hz=float(fM),
                            dfA_minus_E_sim_Hz=float(sim_A), dfA_minus_E_WB_Hz=float(fA),
                            axon_field_sd_Hz=float(f[ax].std())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/user/uqaste15/chisep-meso/validation")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = {}
    if a.quick:
        res["spheres"] = check_spheres(n=128, rad_px=3, seeds=(0,))
        res["cylinders"] = check_cylinders(n=1024, rad_px=4)
        res["hollow"] = check_hollow(n=1024, r_out=0.1)
    else:
        res["spheres"] = check_spheres()
        print("spheres", json.dumps(res["spheres"], indent=1), flush=True)
        res["cylinders"] = check_cylinders()
        print("cylinders", json.dumps(res["cylinders"], indent=1), flush=True)
        res["hollow"] = check_hollow()
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(a.out, "validation.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
