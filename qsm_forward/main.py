#!/usr/bin/env python3

import argparse
import qsm_forward
import numpy as np


def argparse_bool(user_in):
    if user_in is None: return None
    if isinstance(user_in, bool): return user_in
    user_in = user_in.strip().lower()
    if user_in in ['on', 'true', 'yes']: return True
    if user_in in ['off', 'false', 'no']: return False
    raise ValueError(f"Invalid boolean value {user_in}; use on/yes/true or off/false/no")


def add_common_args(p):
    """Reconstruction / output arguments shared by all subcommands."""
    p.add_argument('bids', help='Output BIDS directory')
    p.add_argument('--subject', default='1')
    p.add_argument('--session', default=None)
    p.add_argument('--acq', default=None)
    p.add_argument('--run', default=None)
    p.add_argument('--TR', default=50e-3, type=float)
    p.add_argument('--TEs', default=[4e-3, 12e-3, 20e-3, 28e-3], type=float, nargs='+')
    p.add_argument('--flip_angle', default=15, type=float)
    p.add_argument('--B0', default=7, type=float)
    p.add_argument('--B0-dir', default=[0., 0., 1.], type=float, nargs=3)
    p.add_argument('--suffix', default='MEGRE')
    p.add_argument('--generate-phase-offset', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--generate-shim-field', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--voxel-size', default=[1., 1., 1.], type=float, nargs=3)
    p.add_argument('--peak-snr', default=np.inf, type=float)
    p.add_argument('--random-seed', default=42, type=int)
    p.add_argument('--save-phase', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--save-chi', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--save-mask', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--save-segmentation', nargs='?', type=argparse_bool, const=True, default=True)
    p.add_argument('--save-r2', nargs='?', type=argparse_bool, const=True, default=False, help='Save computed R2 map')
    p.add_argument('--save-dr-pos', nargs='?', type=argparse_bool, const=True, default=False, help='Save paramagnetic relaxivity (Dr+) map')
    p.add_argument('--save-dr-neg', nargs='?', type=argparse_bool, const=True, default=False, help='Save diamagnetic relaxivity (Dr-) map')
    p.add_argument('--save-t2', nargs='?', type=argparse_bool, const=True, default=False, help='Save computed T2 map')
    p.add_argument('--save-se', nargs='?', type=argparse_bool, const=True, default=False, help='Save a simulated multi-echo spin-echo (SE) acquisition (magnitude decays with R2, not R2*)')
    p.add_argument('--se-TR', default=1.0, type=float, help='Repetition time (s) for the spin-echo acquisition (default: 1.0)')
    p.add_argument('--se-TEs', default=[10e-3, 30e-3, 50e-3, 70e-3], type=float, nargs='+', help='Echo times (s) for the spin-echo acquisition')


def make_recon_params(args):
    return qsm_forward.ReconParams(
        subject=args.subject, session=args.session, acq=args.acq, run=args.run,
        TR=args.TR, TEs=np.array(args.TEs), flip_angle=args.flip_angle,
        B0=args.B0, B0_dir=np.array(args.B0_dir), phase_offset=0,
        generate_phase_offset=args.generate_phase_offset, generate_shim_field=args.generate_shim_field,
        voxel_size=np.array(args.voxel_size), peak_snr=args.peak_snr, random_seed=args.random_seed,
        save_phase=args.save_phase, suffix=args.suffix, se_TR=args.se_TR, se_TEs=np.array(args.se_TEs),
    )


def main():
    parser = argparse.ArgumentParser(description='Simulate magnitude and phase')
    subparsers = parser.add_subparsers(dest='mode')

    head_parser = subparsers.add_parser('head', help='Realistic in-silico head phantom')
    simple_parser = subparsers.add_parser('simple', help='Simple susceptibility sources')
    chisep_parser = subparsers.add_parser('chi-sep', help='Susceptibility source-separation phantom (Ridani et al. 2026)')

    head_parser.add_argument('data', help='Head phantom data directory (download from https://doi.org/10.34973/m20r-jt17)')
    chisep_parser.add_argument('data', help='Head phantom data directory; must also contain maps/V1.nii.gz and masks/white_matter_mask.nii.gz for anisotropy, and raw/rawField.nii.gz')

    for p in (head_parser, simple_parser, chisep_parser):
        add_common_args(p)

    # head / simple chi-sep-aware signal knobs (single-kernel Dr)
    for p in (head_parser, simple_parser):
        p.add_argument('--save-field', nargs='?', type=argparse_bool, const=True, default=False)
        p.add_argument('--save-shimmed-field', nargs='?', type=argparse_bool, const=True, default=False)
        p.add_argument('--save-shimmed-offset-field', nargs='?', type=argparse_bool, const=True, default=False)
        p.add_argument('--save-chi-pos', nargs='?', type=argparse_bool, const=True, default=False, help='Save paramagnetic susceptibility map (chi+)')
        p.add_argument('--save-chi-neg', nargs='?', type=argparse_bool, const=True, default=False, help='Save diamagnetic susceptibility map (chi-)')
        p.add_argument('--save-r2prime', nargs='?', type=argparse_bool, const=True, default=False, help="Save R2' map computed from chi+ and chi-")
        p.add_argument('--dr', default=qsm_forward.DR_KERNEL, type=float, help=f"Magnitude decay kernel (Hz/ppm) relating |chi| to R2' (default: {qsm_forward.DR_KERNEL})")
        p.add_argument('--dr-neg', default=None, type=float, help='Optional separate diamagnetic relaxivity (Hz/ppm) for a split model')
        p.add_argument('--chisep-signal', nargs='?', type=argparse_bool, const=True, default=False, help='Use chi-sep-aware signal model (R2 + Dr*|chi|) instead of R2*')
        p.add_argument('--chisep-multicompartment', nargs='?', type=argparse_bool, const=True, default=False, help='Multi-compartment GRE magnitude (implies --chisep-signal)')
        p.add_argument('--anisotropy', nargs='?', type=argparse_bool, const=True, default=False, help='Enable WM susceptibility anisotropy (requires --v1 / angle map)')

    # head-specific
    head_parser.add_argument('--chi-pos', default=None, help='Path to paramagnetic susceptibility map (chi+)')
    head_parser.add_argument('--chi-neg', default=None, help='Path to diamagnetic susceptibility map (chi-)')
    head_parser.add_argument('--v1', default=None, help='Path to V1 eigenvector map (4D NIfTI) for WM anisotropy')
    head_parser.add_argument('--r2', default=None, help='Path to pre-computed R2 map')
    head_parser.add_argument('--angle-map', default=None, help='Path to fiber-to-field angle map (degrees)')

    # simple-specific
    simple_parser.add_argument('--resolution', default=[100, 100, 100], type=int, nargs=3)
    simple_parser.add_argument('--background', default=0, type=float)
    simple_parser.add_argument('--large-cylinder-val', default=0.005, type=float)
    simple_parser.add_argument('--small-cylinder-radii', default=[4, 4, 4, 7], type=float, nargs='+')
    simple_parser.add_argument('--small-cylinder-vals', default=[0.05, 0.1, 0.2, 0.5], type=float, nargs='+')

    # chi-sep-specific
    chisep_parser.add_argument('--dr-model', choices=['fixed', 'scaled'], default='fixed',
                               help="Dr magnitude: 'fixed' anchors Dr+ = 137 Hz/ppm (Shin et al. 2021), field-independent; "
                                    "'scaled' uses the static-dephasing Dr+ = 107.84*B0 (Yablonskiy & Haacke 1994). Both keep the sin^2(theta) shape.")
    chisep_parser.add_argument('--isotropic', action='store_true', help='Disable WM anisotropy (constant Dr-)')
    chisep_parser.add_argument('--no-brain-mask', action='store_true', help='Do not restrict chi to the brain mask (simulate the whole head, with background field)')

    args = parser.parse_args()

    # ---- chi-separation phantom (Ridani et al. 2026) ----
    if args.mode == 'chi-sep':
        anisotropy = not args.isotropic
        tissue_params = qsm_forward.TissueParams(
            args.data, chisep_anisotropy=anisotropy, chisep_apply_brain_mask=not args.no_brain_mask,
        )
        theta = (qsm_forward.generate_theta_from_v1(tissue_params.v1.get_fdata(), np.array(args.B0_dir))
                 if tissue_params.v1 is not None else None)
        dr_pos_map, dr_neg_map = qsm_forward.generate_dr_maps_ridani(
            tissue_params.seg.get_fdata(), theta, B0=args.B0, anisotropic=anisotropy,
            dr_fixed=(qsm_forward.DR_KERNEL if args.dr_model == 'fixed' else None),
        )
        qsm_forward.generate_bids(
            tissue_params, make_recon_params(args), args.bids,
            save_chi=args.save_chi, save_mask=args.save_mask, save_segmentation=args.save_segmentation,
            save_field=True, save_chi_pos=True, save_chi_neg=True, save_r2prime=True,
            dr_pos_map=dr_pos_map, dr_neg_map=dr_neg_map, chisep_signal=True,
            save_r2=args.save_r2, save_dr_pos=args.save_dr_pos, save_dr_neg=args.save_dr_neg,
            save_t2=args.save_t2, save_se=True,
        )
        return

    # ---- head / simple ----
    if args.mode == 'head':
        tissue_params = qsm_forward.TissueParams(
            args.data, chi_pos=args.chi_pos, chi_neg=args.chi_neg,
            v1=args.v1, R2=args.r2, angle_map=args.angle_map,
        )
    elif args.mode == 'simple':
        tissue_params = qsm_forward.TissueParams(
            chi=qsm_forward.generate_susceptibility_phantom(
                resolution=args.resolution, background=args.background,
                large_cylinder_val=args.large_cylinder_val,
                small_cylinder_radii=args.small_cylinder_radii,
                small_cylinder_vals=args.small_cylinder_vals,
            )
        )
    else:
        parser.print_help()
        return

    qsm_forward.generate_bids(
        tissue_params, make_recon_params(args), args.bids,
        save_chi=args.save_chi, save_mask=args.save_mask, save_segmentation=args.save_segmentation,
        save_field=args.save_field, save_shimmed_field=args.save_shimmed_field,
        save_shimmed_offset_field=args.save_shimmed_offset_field,
        save_chi_pos=args.save_chi_pos, save_chi_neg=args.save_chi_neg, save_r2prime=args.save_r2prime,
        dr=args.dr, dr_neg=args.dr_neg, chisep_signal=args.chisep_signal,
        chisep_multicompartment=args.chisep_multicompartment, anisotropy=args.anisotropy,
        save_r2=args.save_r2, save_dr_pos=args.save_dr_pos, save_dr_neg=args.save_dr_neg,
        save_t2=args.save_t2, save_se=args.save_se,
    )


if __name__ == "__main__":
    main()
