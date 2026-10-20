"""Apply a merged LUT to one slice of the head phantom and report, per tissue label,
the R2' a mono-exponential fit would see (GRE log-linear fit minus SE fit), compared
with the single-kernel relaxivity model R2' = Dr (chi+ + |chi-|).

Usage: python demo_slice.py --lut lut_diffusion.npz [--B0 3] [--slice 160]
"""
import argparse
import json

import nibabel as nib
import numpy as np

from lut import MesoLUT

DATA = "/scratch/user/uqaste15/qsm-forward-dev/data"
CHI = "/scratch/user/uqaste15/ridani-ref/data/chimodel"
NAMES = {1: "caudate", 2: "pallidus", 3: "putamen", 4: "red_nucleus", 5: "dentate",
         6: "s_nigra", 7: "thalamus", 8: "wm", 9: "gm", 10: "csf", 11: "blood"}


def fit_rate(TE, S):
    t = TE - TE.mean()
    y = np.log(np.maximum(np.abs(S), 1e-12))
    return -(y * t[:, None]).sum(0) / (t ** 2).sum() * 1e3   # 1/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", required=True)
    ap.add_argument("--B0", type=float, default=3.0)
    ap.add_argument("--slice", type=int, default=None, help="index along axis 2")
    ap.add_argument("--out", default=None)
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
    res = {}
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
            res[f"wm_theta_{lo}_{hi}"] = dict(n=int(mm.sum()), R2p_meso=float(np.median(r2p[wm][mm])))
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
