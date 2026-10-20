"""Iron-cluster calibration sweep: emergent R2', R2(SE) and relaxivities per ppm of
chi+ as a function of cluster radius and cluster susceptibility, with diffusion.

The static-dephasing R2' is size-independent (Y&H); with diffusion, both the GRE
decay and the spin-echo decay depend on R^2/D vs 1/dw. Literature targets to compare
against: D_r+ ~ 137 Hz/ppm at 3T for R2' (Shin et al. 2021); the irreversible part
shows up as SE decay.

Usage: python calibrate_iron.py --radius R --dchi X [--chi 0.1] [--B0 3] [--out DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso  # noqa: E402


def run(radius, dchi, chi=0.1, B0=3.0, D=1.0, n3d=128, box_r=24, walkers=4000,
        dt=None, tmax=60.0, seed=0):
    L = box_r * radius
    zeta = chi / dchi
    rng = np.random.default_rng(seed)
    vox = meso.build_voxel(L, 0.0, [zeta], radius, n2d=16, n3d=n3d, rng=rng)
    nsph = int(round(zeta * L ** 3 / (4 / 3 * np.pi * radius ** 3)))
    zeta_eff = nsph * 4 / 3 * np.pi * radius ** 3 / L ** 3
    pos, comp, _ = meso.init_walkers(vox, walkers, rng)
    if dt is None:   # step sigma <= R/15, capped at 2e-3 ms
        dt = min(2e-3, (radius / 15) ** 2 / (2 * D))
    TE = np.arange(0, tmax + 1e-9, 1.0)
    t0 = time.time()
    coef = meso.phase_coeffs(vox, 0.0, 0, chi / zeta_eff, 0.0)
    res = {}
    for name, Dv in (("static", 0.0), ("diffusion", D)):
        I = meso.walk(vox, pos, comp, TE, D=Dv, dt=dt, rng=rng)
        kt = np.arange(0, len(TE), 2)
        kh = kt // 2
        g, s = meso.compartment_signals(I, comp, coef, B0, (kt, kh))
        gre, se = g["extra"], s["extra"]
        m = (TE >= 10) & (TE <= 40)
        r2s = -np.polyfit(TE[m], np.log(np.abs(gre[m])), 1)[0] * 1e3
        ms = (TE[kt] >= 10) & (TE[kt] <= 40)
        r2 = -np.polyfit(TE[kt][ms], np.log(np.abs(se[ms]) + 1e-12), 1)[0] * 1e3
        res[name] = dict(R2star_per_s=r2s, R2_SE_per_s=r2, R2prime_per_s=r2s - r2,
                         Dr_prime_Hz_per_ppm=(r2s - r2) / chi, Dr_star_Hz_per_ppm=r2s / chi,
                         Dr_SE_Hz_per_ppm=r2 / chi)
    dw = meso.dw_sphere(dchi, B0)
    res.update(radius_um=radius, dchi_ppm=dchi, chi_pos_ppm=chi, B0=B0, D=D, dt_ms=dt,
               zeta=zeta_eff, n_spheres=nsph,
               alpha_dw_R2_over_D=dw * radius ** 2 / D,
               R2p_YH_per_s=meso.yh_r2p_sphere(zeta_eff, dchi, B0) * 1e3,
               elapsed_s=time.time() - t0)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--dchi", type=float, required=True)
    ap.add_argument("--chi", type=float, default=0.1)
    ap.add_argument("--B0", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out", default="/scratch/user/uqaste15/chisep-meso/calib")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = run(a.radius, a.dchi, a.chi, a.B0, dt=a.dt)
    tag = f"R{a.radius}_X{a.dchi}_B{a.B0}" + (f"_dt{a.dt}" if a.dt else "")
    with open(os.path.join(a.out, f"calib_{tag}.json"), "w") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps(r, indent=1))
