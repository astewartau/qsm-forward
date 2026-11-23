"""Iron calibration: emergent R2' and R2 per ppm chi+ for a mix of sphere populations,
static and with diffusion, over the 3 T protocol of meso.PROTOCOL_3T.

Dr+ = R2'/chi+ with R2' = R2*(GRE fit) - R2(Hahn SE fit). Target (Shin et al. 2021,
SNU chi-separation toolbox): Dr+ = 137 * B0/2.9 Hz/ppm, i.e. ~142 at 3 T.

A population is NAME:RADIUS_um:DCHI_ppm:SHARE, SHARE = fraction of chi+ it carries.
Each population lives in its own periodic box (grid_for), as in the LUT.

Usage:
  python calibrate_iron.py --pops cluster:3:10:1 --seed 0 --out DIR
  python calibrate_iron.py --pops cluster:3:10:0.8,ferritin:0.5:3:0.2 --chi 0.1 --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso  # noqa: E402


def parse_pops(spec):
    out = []
    for item in spec.split(","):
        name, r, x, sh = item.split(":")
        out.append(dict(name=name, radius=float(r), dchi=float(x), share=float(sh)))
    return out


def grid_for(radius):
    """Box side and grid for a population, as in the LUT (build_lut.IRON): micron
    clusters in a 120 um box on 256^3 (0.47 um), sub-micron spheres in a 12 um box on
    128^3 (0.094 um)."""
    return (12.0, 128) if radius < 1 else (120.0, 256)


def run(pops, chi=0.1, B0s=(3.0, 7.0), D=1.0, dt=2e-3, walkers=6000, seed=0):
    rng = np.random.default_rng(seed)
    spec = []
    for p in pops:
        L, n = grid_for(p["radius"])
        spec.append(dict(name=p["name"], radius=p["radius"], L=L, n=n,
                         zetas=[chi * p["share"] / p["dchi"]]))
    t0 = time.time()
    vox = meso.build_voxel(8.0, 0.0, spec, n2d=16, rng=rng)
    iron = {}
    for p, sp in zip(pops, vox.pops):
        iron[sp.name] = (0, chi * p["share"] / sp.zeta[0])   # exact chi share
    pos, comp, _ = meso.init_walkers(vox, walkers, rng)
    P = meso.PROTOCOL_3T
    TE = np.arange(0, 40.001, 1.0)
    kg = np.searchsorted(TE, P["gre_ms"])
    kt = np.searchsorted(TE, P["se_ms"])
    kh = np.searchsorted(TE, P["se_ms"] / 2)
    coef = meso.phase_coeffs(vox, 0.0, 0.0, iron)
    res = dict(pops=pops, chi_pos_ppm=chi, D=D, dt_ms=dt, walkers=walkers, seed=seed,
               zeta={sp.name: float(sp.zeta[0]) for sp in vox.pops}, protocol={
                   k: v.tolist() for k, v in P.items()})
    for mode, Dv in (("static", 0.0), ("diffusion", D)):
        I = meso.walk(vox, pos, comp, TE, D=Dv, dt=dt, rng=rng)
        for B0 in B0s:
            g, s = meso.compartment_signals(I, comp, coef, B0, (kt, kh))
            gre, se = g["extra"], s["extra"]
            r2s = float(meso.fit_rate(TE[kg], gre[kg]))
            r2 = float(meso.fit_rate(P["se_ms"], se)) if Dv > 0 else 0.0
            res[f"{mode}_B{B0:g}"] = dict(
                R2star_per_s=r2s, R2_SE_per_s=r2, R2prime_per_s=r2s - r2,
                Dr_prime_Hz_per_ppm=(r2s - r2) / chi, Dr_SE_Hz_per_ppm=r2 / chi,
                gre_mag=np.abs(gre).tolist(), se_mag=np.abs(se).tolist())
    res["elapsed_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", required=True)
    ap.add_argument("--chi", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--walkers", type=int, default=6000)
    ap.add_argument("--dt", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = run(parse_pops(a.pops), a.chi, seed=a.seed, walkers=a.walkers, dt=a.dt)
    tag = a.pops.replace(":", "_").replace(",", "+") + f"_c{a.chi:g}_s{a.seed}"
    with open(os.path.join(a.out, f"calib_{tag}.json"), "w") as fh:
        json.dump(r, fh)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict) or "R2star_per_s" not in v}))
    for k, v in r.items():
        if isinstance(v, dict) and "R2star_per_s" in v:
            print(k, {kk: round(vv, 1) for kk, vv in v.items() if not isinstance(vv, list)})
