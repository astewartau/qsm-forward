"""Build one geometry's slice of the mesoscopic signal look-up table.

One job = one myelin geometry (fibre volume fraction) x one random seed x one mode
(static or diffusion). A single walk per job serves every theta, B0, chi+ level and
every chi- node that uses that geometry (field linearity, see meso.py).

LUT axes (module constants): CHI_POS (ppm), CHI_NEG (ppm), THETA (deg), B0 (T),
TE (ms; GRE) and TE_SE (ms; spin echo), compartment (extra, myelin, axon).

Ground truth is exact per node:  chi+ = sum over iron populations of dchi * (sphere
volume fraction), chi- = chi_I * (myelin volume fraction) * sheath_scale (tr/3 of the
sheath tensor).

Iron (IRON): each chi+ node is split between micron-scale clusters (share f_c; static-
to-intermediate diffusion regime, they produce most of R2') and small ferritin-scale
spheres (share 1 - f_c; motional-narrowing regime, they raise R2 and hardly R2'). f_c is
tuned (calibrate_iron.py) so that Dr+ = R2'/chi+ from mono-exponential fits over
meso.PROTOCOL_3T (GRE - Hahn SE) is ~140 Hz/ppm at 3 T with diffusion (Shin et al.
2021 empirical 137*B0/2.9).

Myelin: Wharton & Bowtell (2012) constants chi_I = chi_A = -0.10 ppm, g = 0.7, FVF <= 0.6,
myelin water density 0.5 (applied at read, lut.MesoLUT). Each chi- node beyond what
FVF 0.6 gives at chi_I = -0.1 scales the whole sheath tensor (sheath_scale), so every
WM voxel's chi- equals the input chi- map.

Usage:
  python build_lut.py --geom K --seed S --mode {static,diffusion} --out DIR
  python build_lut.py --list        # print geometry table
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meso  # noqa: E402

CHI_POS = np.array([0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
CHI_NEG = np.array([0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08])
THETA = np.arange(0, 91, 10.0)
B0S = np.array([3.0, 7.0])
TE = np.arange(0, 80.0001, 0.5)          # ms
TE_SE = np.arange(0, 80.0001, 1.0)       # ms (TE/2 on the GRE grid)

PARAMS = dict(
    L=60.0,               # myelin box side, um
    n2d=4096,             # myelin grid (0.0146 um)
    g=0.7,                # g-ratio
    chi_i=-0.10,          # sheath mean susceptibility, ppm (W&B chi_I)
    chi_a=-0.10,          # sheath anisotropy, ppm (W&B chi_A)
    fvf_max=0.6,          # max fibre volume fraction reachable by the packing
    walker_box=120.0,     # walkers live in a 2L periodic box (myelin pattern tiled 2x2)
    D=1.0,                # water diffusivity, um^2/ms (diffusion mode)
    dt=2e-3,              # MC time step, ms
    walkers_static=100000,  # per compartment
    walkers_mc=8000,        # per compartment
)


# Iron populations. share = fraction of chi+ carried; box = periodic box side (um);
# n = grid points per box side. Cluster share tuned by calibrate_iron.py (NOTES.md).
IRON = dict(
    cluster=dict(radius=3.0, dchi=20.0, share=0.83, box=120.0, n=256),   # 0.47 um grid
    ferritin=dict(radius=0.5, dchi=3.0, share=0.17, box=12.0, n=128),    # 0.094 um grid
)


def geometries(p=PARAMS):
    """Distinct myelin geometries and the chi- nodes each one serves."""
    mvf_per_fvf = 1 - p["g"] ** 2
    fvf_needed = np.abs(CHI_NEG) / (abs(p["chi_i"]) * mvf_per_fvf)
    fvf = np.minimum(fvf_needed, p["fvf_max"])
    out = []
    for f in np.unique(np.round(fvf, 4)):
        idx = np.where(np.isclose(np.round(fvf, 4), f))[0]
        out.append(dict(fvf=float(f), chi_neg_idx=idx.tolist()))
    return out


def run(geom, seed, mode, outdir, p=PARAMS, iron=IRON):
    os.makedirs(outdir, exist_ok=True)
    G = geometries(p)[geom]
    t0 = time.time()
    rng = np.random.default_rng([geom, seed, 0 if mode == "static" else 1])
    cpos = CHI_POS[CHI_POS > 0]
    spec = [dict(name=k, radius=v["radius"], L=v["box"], n=v["n"],
                 zetas=list(cpos * v["share"] / v["dchi"])) for k, v in iron.items()
            if v["share"] > 0]
    vox = meso.build_voxel(p["L"], G["fvf"], spec, n2d=p["n2d"], g=p["g"],
                           chi_i=p["chi_i"], chi_a=p["chi_a"], rng=rng,
                           walker_box=p["walker_box"])
    # exact achieved fractions -> per-population dchi giving the exact chi+ share
    zeta_eff = {sp.name: sp.zeta for sp in vox.pops}
    mvf = vox.packing.mvf
    print(f"geom {geom} fvf={vox.packing.fvf:.3f} mvf={mvf:.3f} build {time.time()-t0:.1f}s",
          flush=True)
    nw = p["walkers_static"] if mode == "static" else p["walkers_mc"]
    pos, comp, frac = meso.init_walkers(vox, nw, rng)
    present = [c for c in meso.COMPARTMENTS if frac[c] > 0]

    nneg = len(G["chi_neg_idx"])
    shape = (nneg, len(CHI_POS), len(THETA), len(B0S), 3)
    gre = np.full(shape + (len(TE),), np.nan, np.complex64)
    se = np.full(shape + (len(TE_SE),), np.nan, np.complex64)
    sheath_scale = np.zeros(nneg)
    for a, ineg in enumerate(G["chi_neg_idx"]):
        sheath_scale[a] = abs(CHI_NEG[ineg]) / (mvf * abs(p["chi_i"])) if mvf > 0 else 0.0

    if mode == "static":
        v = meso._basis_values(vox, pos)
    else:
        I = meso.walk(vox, pos, comp, TE, D=p["D"], dt=p["dt"], rng=rng, progress=True)
        kt = np.round(TE_SE / 0.5).astype(int)
        kh = np.round(TE_SE / 1.0).astype(int)
    print(f"  walkers ready {time.time()-t0:.1f}s", flush=True)

    for a in range(nneg):
        for ip, cp in enumerate(CHI_POS):
            lev = {} if cp == 0 else {
                k: (ip - 1, cp * iron[k]["share"] / z[ip - 1]) for k, z in zeta_eff.items()}
            for it, th in enumerate(np.deg2rad(THETA)):
                coef = meso.phase_coeffs(vox, th, sheath_scale[a], lev)
                for ib, B0 in enumerate(B0S):
                    if mode == "static":
                        w = (v @ coef) * (meso.GAMMA * B0 * 1e-6)   # rad/ms
                        for ci, c in enumerate(meso.COMPARTMENTS):
                            m = comp == ci
                            if not m.any():
                                continue
                            gre[a, ip, it, ib, ci] = _hist_sig(w[m], TE)
                            se[a, ip, it, ib, ci] = 1.0
                    else:
                        g_, s_ = meso.compartment_signals(I, comp, coef, B0, (kt, kh))
                        for ci, c in enumerate(meso.COMPARTMENTS):
                            if c in g_:
                                gre[a, ip, it, ib, ci] = g_[c]
                                se[a, ip, it, ib, ci] = s_[c]
    fn = os.path.join(outdir, f"lut_{mode}_g{geom}_s{seed}.npz")
    np.savez_compressed(
        fn, gre=gre, se=se, chi_neg_idx=np.array(G["chi_neg_idx"]),
        chi_neg=CHI_NEG[G["chi_neg_idx"]], chi_pos=CHI_POS, theta=THETA, B0=B0S, TE=TE,
        TE_SE=TE_SE, frac=np.array([frac[c] for c in meso.COMPARTMENTS]),
        fvf=vox.packing.fvf, mvf=mvf, sheath_scale=sheath_scale,
        zeta_eff=json.dumps({k: v.tolist() for k, v in zeta_eff.items()}),
        params=json.dumps(dict(p, iron=iron)), mode=mode, seed=seed)
    print(f"  wrote {fn} total {time.time()-t0:.1f}s", flush=True)


def _hist_sig(w, t, nbins=4000):
    lo, hi = float(w.min()), float(w.max())
    if hi - lo < 1e-12:
        return np.exp(1j * lo * t)
    h, e = np.histogram(w, bins=nbins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    return np.exp(1j * np.outer(t, c)) @ (h / h.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["static", "diffusion"], default="static")
    ap.add_argument("--out", default="/scratch/user/uqaste15/chisep-meso/lut_parts")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", nargs="*", default=[],
                    help="override PARAMS (k=v) or IRON (pop.k=v)")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        if "." in k:
            pop, kk = k.split(".")
            IRON[pop][kk] = type(IRON[pop][kk])(v)
        else:
            PARAMS[k] = type(PARAMS[k])(v)
    if a.list:
        for i, g in enumerate(geometries()):
            print(i, g)
    else:
        run(a.geom, a.seed, a.mode, a.out)
