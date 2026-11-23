"""Apply a merged LUT to one slice of the head phantom and report, per tissue label,
the R2' a mono-exponential fit would see (GRE log-linear fit minus SE fit), compared
with the single-kernel relaxivity model R2' = Dr (chi+ + |chi-|).

Usage: python demo_slice.py --lut lut_diffusion.npz [--B0 3] [--slice 160]
"""
import argparse
import json

import nibabel as nib
import numpy as np

import meso
from lut import MesoLUT, chi_neg_apparent, chi_neg_ground_truth

DATA = "/scratch/user/uqaste15/qsm-forward-dev/data"
CHI = "/scratch/user/uqaste15/ridani-ref/data/chimodel"
NAMES = {1: "caudate", 2: "pallidus", 3: "putamen", 4: "red_nucleus", 5: "dentate",
         6: "s_nigra", 7: "thalamus", 8: "wm", 9: "gm", 10: "csf", 11: "blood"}


def fit_rate(TE, S):
    t = TE - TE.mean()
    y = np.log(np.maximum(np.abs(S), 1e-12))
    return -(y * t[:, None]).sum(0) / (t ** 2).sum() * 1e3   # 1/s


def iron_relaxivity(L, B0, chis=(0.05, 0.10, 0.15, 0.20)):
    """Dr+ (R2'/chi+) and R2/chi+ of iron alone (chi- = 0) from the LUT, mono-exp fits
    over meso.PROTOCOL_3T (GRE minus Hahn SE), single extra-cellular compartment, no
    intrinsic T2 (it cancels in R2')."""
    P = meso.PROTOCOL_3T
    cp = np.array(chis)
    z = np.zeros_like(cp)
    g = np.array([L.decay(cp, z, z, B0, t, T2_ms=(1e9,) * 3, rho=(1, 0, 0), powder=True)
                  for t in P["gre_ms"]])
    e = np.array([L.decay(cp, z, z, B0, t, T2_ms=(1e9,) * 3, rho=(1, 0, 0), se=True,
                          powder=True) for t in P["se_ms"]])
    r2s, r2 = meso.fit_rate(P["gre_ms"], g), meso.fit_rate(P["se_ms"], e)
    return dict(chi_pos=cp.tolist(), Dr_plus_Hz_per_ppm=((r2s - r2) / cp).tolist(),
                R2_per_ppm=(r2 / cp).tolist(), R2star_per_ppm=(r2s / cp).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", required=True)
    ap.add_argument("--B0", type=float, default=3.0)
    ap.add_argument("--slice", type=int, default=None, help="index along axis 2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    seg = nib.load(f"{DATA}/masks/SegmentedModel.nii.gz").get_fdata()
    k = a.slice if a.slice is not None else seg.shape[2] // 2
    sl = (slice(None), slice(None), k)
    seg = seg[sl].astype(int)
    cp = nib.load(f"{CHI}/Chi_positive.nii.gz").get_fdata()[sl]
    cn = nib.load(f"{CHI}/Chi_negative.nii.gz").get_fdata()[sl]
    v1 = nib.load(f"{DATA}/maps/V1.nii.gz").get_fdata()[:, :, k]
    nrm = np.linalg.norm(v1, axis=-1)
    theta = np.arccos(np.clip(np.abs(v1[..., 2]) / np.maximum(nrm, 1e-12), 0, 1))
    m = (seg > 0) & (seg != 10) & (seg != 11)
    L = MesoLUT(a.lut)
    TE = np.arange(5.0, 40.01, 5.0)
    wm = (seg == 8)[m] & (nrm[m] > 0)
    def run(se):
        out = []
        for te in TE:
            Fw = L.decay(cp[m], cn[m], theta[m], a.B0, te, se=se)
            Fp = L.decay(cp[m], cn[m], theta[m], a.B0, te, se=se, powder=True)
            out.append(np.where(wm, Fw, Fp))
        return np.array(out)
    gre, se = run(False), run(True)
    r2s = fit_rate(TE, gre)
    r2 = fit_rate(TE, se)
    r2p = r2s - r2
    dr = 137.0 * (a.B0 / 3.0)
    r2p_kernel = dr * (np.maximum(cp[m], 0) + np.abs(np.minimum(cn[m], 0)))
    res = {"iron_relaxivity": iron_relaxivity(L, a.B0)}
    gt = chi_neg_ground_truth(cn[m])
    app = np.where(wm, chi_neg_apparent(cn[m], theta[m]), gt)
    for lab, name in NAMES.items():
        mm = seg[m] == lab
        if not mm.any():
            continue
        res[name] = dict(n=int(mm.sum()),
                         R2p_meso=float(np.median(r2p[mm])),
                         R2_SE_meso=float(np.median(r2[mm])),
                         R2p_single_kernel=float(np.median(r2p_kernel[mm])),
                         chi_pos=float(np.median(cp[m][mm])), chi_neg=float(np.median(cn[m][mm])))
    if wm.any():
        th = np.rad2deg(theta[m][wm])
        for lo, hi in ((0, 30), (30, 60), (60, 90)):
            mm = (th >= lo) & (th < hi)
            res[f"wm_theta_{lo}_{hi}"] = dict(
                n=int(mm.sum()), R2p_meso=float(np.median(r2p[wm][mm])),
                chi_neg_gt=float(np.median(gt[wm][mm])),
                chi_neg_apparent=float(np.median(app[wm][mm])))
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        maps = {}
        for nm, v in (("chi- ground truth (ppm)", gt), ("chi- apparent (ppm)", app),
                      ("R2' meso (1/s)", r2p), ("R2 SE meso (1/s)", r2)):
            im = np.full(seg.shape, np.nan)
            im[m] = v
            maps[nm] = im
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
        for k, (nm, im) in enumerate(maps.items()):
            lim = (-0.08, 0) if "chi" in nm else (0, np.nanpercentile(im, 99))
            h = ax[k].imshow(np.rot90(im), cmap="gray", vmin=lim[0], vmax=lim[1])
            ax[k].set_title(nm)
            ax[k].axis("off")
            fig.colorbar(h, ax=ax[k], fraction=0.046)
        fig.suptitle(f"{a.lut.split('/')[-1]}  B0 = {a.B0:g} T")
        fig.tight_layout()
        fig.savefig(a.png, dpi=90)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
