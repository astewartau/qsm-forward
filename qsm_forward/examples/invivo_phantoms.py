#!/usr/bin/env python
"""
Build head phantoms from in vivo reconstructions and simulate them.

For each acquisition: an in vivo χ map (``--chi``, e.g. a QSMxT iLSQR run with a BET mask), the
packed QSM-CI harmonization inputs it came from (``--inputs``: ``magnitude.nii.gz`` with echoes
last, ``mask.nii.gz`` the brain mask, ``params.json`` with TE, TR, flip_angle, B0_dir), optionally
strong sources (``--preset``), then a forward simulation (``--simulate``) at the source protocol or
at a chosen B0 / voxel size / SNR.

    python invivo_phantoms.py --chi chi.nii.gz --inputs inputs/ --out phantoms/s002001 \\
        [--preset streak-heavy --seed 1] [--simulate --B0 3 --peak-snr 100 --voxel-size 1 1 1]

With ``--seg aseg.nii.gz`` (SynthSeg/FreeSurfer labels, e.g. from ``qsmxt segment``) the phantom gets
brain labels; with ``--upsample 1.5`` it is built on a 1.5x finer grid and ``--simulate`` uses
``simulate_acquisition`` (k-space crop to ``--voxel-size``, partial-volume references, no inverse
crime) with ``--noise-sigma`` in the source magnitude's units instead of ``--peak-snr``.

Author: Ashley Stewart (a.stewart.au@gmail.com)
"""

import argparse
import json
import os

import nibabel as nib
import numpy as np

from qsm_forward import ReconParams, phantoms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chi", required=True, help="in vivo chi map (ppm)")
    ap.add_argument("--chi-mask", default=None, help="where --chi is valid (default: its non-zero support)")
    ap.add_argument("--cone", default="clamp", choices=["clamp", "equalise", "none"])
    ap.add_argument("--inputs", required=True, help="dir with magnitude.nii.gz, mask.nii.gz, params.json")
    ap.add_argument("--out", required=True, help="output dir (phantom/ and bids/ are written in it)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--denoise", default="wiener", choices=["wiener", "tv", "none"])
    ap.add_argument("--neck-mm", type=float, default=40.0)
    ap.add_argument("--preset", default=None, help=f"insert sources: one of {sorted(phantoms.PRESETS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--B0", type=float, default=None, help="default: the source's nominal B0")
    ap.add_argument("--TEs", type=float, nargs="+", default=None, help="s; default: the source's")
    ap.add_argument("--voxel-size", type=float, nargs=3, default=None, help="mm; default: the source's")
    ap.add_argument("--B0-dir", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument("--peak-snr", type=float, default=100.0)
    ap.add_argument("--subject", default="1")
    ap.add_argument("--seg", default=None, help="FreeSurfer/SynthSeg label map on the chi grid")
    ap.add_argument("--upsample", type=float, default=None, help="build on a grid this many times finer")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="with --upsample: complex noise SD in the source magnitude's units")
    a = ap.parse_args()

    params = json.load(open(os.path.join(a.inputs, "params.json")))
    chi_img = nib.load(a.chi)
    chi = chi_img.get_fdata(dtype=np.float32)
    mag = np.asarray(nib.load(os.path.join(a.inputs, "magnitude.nii.gz")).dataobj, dtype=np.float32)
    brain = nib.load(os.path.join(a.inputs, "mask.nii.gz")).get_fdata() > 0
    vox = np.array(chi_img.header.get_zooms()[:3], float)
    TEs = params["TE"]

    phantom_dir = os.path.join(a.out, "phantom")
    if os.path.exists(os.path.join(phantom_dir, "phantom.json")) and a.preset is None:
        ph = phantoms.Phantom.load(phantom_dir)
        print(f"loaded {phantom_dir}")
    else:
        ph = phantoms.build_invivo_phantom(
            chi, mag, TEs, vox, brain, affine=chi_img.affine, TR=params.get("TR", 0.035),
            flip_angle=params.get("flip_angle", 15), B0_dir=params.get("B0_dir", [0, 0, 1]),
            chi_support=(nib.load(a.chi_mask).get_fdata() > 0) if a.chi_mask else None,
            seg=phantoms.seg_from_freesurfer(nib.load(a.seg).get_fdata()) if a.seg else None,
            clean={"denoise": None if a.denoise == "none" else a.denoise,
                   "cone": None if a.cone == "none" else a.cone}, neck_mm=a.neck_mm,
            name=a.name or os.path.basename(os.path.normpath(a.out)),
            source={"chi": os.path.abspath(a.chi), "inputs": os.path.abspath(a.inputs),
                    "params": params})
        if a.upsample:
            ph = phantoms.upsample_phantom(ph, a.upsample)
        ph.save(phantom_dir)
        print(f"wrote {phantom_dir}: clean {json.dumps(ph.meta['clean'])}")
    if a.preset:
        ph = phantoms.insert_sources(ph, a.preset, seed=a.seed)
        phantom_dir = os.path.join(a.out, f"phantom-{a.preset}-seed{a.seed}")
        ph.save(phantom_dir)
        print(f"wrote {phantom_dir}: {len(ph.sources)} sources")

    if a.simulate and "upsampled" in ph.meta:
        bids = os.path.join(a.out, "bids" if not a.preset else f"bids-{a.preset}-seed{a.seed}")
        out = phantoms.simulate_acquisition(
            ph, bids, a.TEs or TEs, B0=a.B0 or params.get("B0_nominal", 3), TR=params.get("TR", 0.035),
            flip_angle=params.get("flip_angle", 15), voxel_size=a.voxel_size or ph.meta["upsampled"]["from_voxel_size"],
            B0_dir=a.B0_dir, noise_sigma=a.noise_sigma, seed=a.seed, subject=a.subject)
        print(json.dumps(out, indent=1))
    elif a.simulate:
        rp = ReconParams(
            subject=a.subject, TEs=np.array(a.TEs or TEs), TR=params.get("TR", 0.035),
            flip_angle=params.get("flip_angle", 15), B0=a.B0 or params.get("B0_nominal", 3),
            B0_dir=np.array(a.B0_dir), voxel_size=np.array(a.voxel_size or vox), peak_snr=a.peak_snr,
            random_seed=a.seed)
        bids = os.path.join(a.out, "bids" if not a.preset else f"bids-{a.preset}-seed{a.seed}")
        out = phantoms.simulate(phantom_dir, rp, bids)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
