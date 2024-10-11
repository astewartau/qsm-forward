#!/usr/bin/env python3

import argparse
import qsm_forward
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(
        description='Simulate magnitude and phase'
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest='mode')
    headphantom_parser = subparsers.add_parser('head', help='Use realistic in-silico head phantom for simulation')
    simplephantom_parser = subparsers.add_parser('simple', help='Generate simple susceptibility sources for simulation')

    def argparse_bool(user_in):
        if user_in is None: return None
        if isinstance(user_in, bool): return user_in
        user_in = user_in.strip().lower()
        if user_in in ['on', 'true', 'yes']: return True
        if user_in in ['off', 'false', 'no']: return False
        raise ValueError(f"Invalid boolean value {user_in}; use on/yes/true or off/false/no")# Arguments specific to phantom simulation
    
    headphantom_parser.add_argument('data', help='Head phantom data directory (download from https://doi.org/10.34973/m20r-jt17)')
    
    # Arguments common to both subcommands
    for sub_parser in [headphantom_parser, simplephantom_parser]:
        # Add the common arguments here e.g. 
        sub_parser.add_argument('bids', help='Output BIDS directory')
        sub_parser.add_argument('--subject', default='1')
        sub_parser.add_argument('--session', default=None),
        sub_parser.add_argument('--acq', default=None),
        sub_parser.add_argument('--run', default=None),
        sub_parser.add_argument('--TR', default=50e-3, type=float),
        sub_parser.add_argument('--TEs', default=[ 4e-3, 12e-3, 20e-3, 28e-3 ], type=float, nargs='+')
        sub_parser.add_argument('--flip_angle', default=15, type=float)
        sub_parser.add_argument('--B0', default=7, type=float)
        sub_parser.add_argument('--B0-dir', default=[0., 0., 1.], type=float, nargs=3)
        sub_parser.add_argument('--suffix', default='MEGRE')
        sub_parser.add_argument('--generate-phase-offset', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--generate-shim-field', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--voxel-size', default=[1., 1., 1.], type=float, nargs=3)
        sub_parser.add_argument('--peak-snr', default=np.inf, type=float)
        sub_parser.add_argument('--random-seed', default=42, type=int)
        sub_parser.add_argument('--save-phase', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--save-chi', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--save-mask', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--save-segmentation', nargs='?', type=argparse_bool, const=True, default=True)
        sub_parser.add_argument('--save-field', nargs='?', type=argparse_bool, const=False, default=False)
        sub_parser.add_argument('--save-shimmed-field', nargs='?', type=argparse_bool, const=False, default=False)
        sub_parser.add_argument('--save-shimmed-offset-field', nargs='?', type=argparse_bool, const=False, default=False)

    # Arguments specific to susceptibility sources simulation
    simplephantom_parser.add_argument('--resolution', default=[100, 100, 100], type=int, nargs=3)
    simplephantom_parser.add_argument('--background', default=0, type=float)
    simplephantom_parser.add_argument('--large-cylinder-val', default=0.005, type=float)
    simplephantom_parser.add_argument('--small-cylinder-radii', default=[4, 4, 4, 7], type=float, nargs='+')
    simplephantom_parser.add_argument('--small-cylinder-vals', default=[0.05, 0.1, 0.2, 0.5], type=float, nargs='+')

    args = parser.parse_args()

    if args.mode == 'head':
        tissue_params = qsm_forward.TissueParams(args.data)

    elif args.mode == 'simple':
        tissue_params = qsm_forward.TissueParams(
            chi=qsm_forward.generate_susceptibility_phantom(
                resolution=args.resolution,
                background=args.background,
                large_cylinder_val=args.large_cylinder_val,
                small_cylinder_radii=args.small_cylinder_radii,
                small_cylinder_vals=args.small_cylinder_vals
            )
        )

    else:
        parser.print_help()
        return

    # Common code to run for both modes
    recon_params = qsm_forward.ReconParams(
        subject=args.subject,
        session=args.session,
        acq=args.acq,
        run=args.run,
        TR=args.TR,
        TEs=np.array(args.TEs),
        flip_angle=args.flip_angle,
        B0=args.B0,
        B0_dir=np.array(args.B0_dir),
        phase_offset=0,
        generate_phase_offset=True,
        generate_shim_field=True,
        voxel_size=np.array(args.voxel_size),
        peak_snr=args.peak_snr,
        random_seed=args.random_seed,
        save_phase=args.save_phase,
        suffix=args.suffix
    )

    qsm_forward.generate_bids(
        tissue_params,
        recon_params,
        args.bids,
        save_chi=args.save_chi,
        save_mask=args.save_mask,
        save_segmentation=args.save_segmentation,
        save_field=args.save_field,
        save_shimmed_field=args.save_shimmed_field,
        save_shimmed_offset_field=args.save_shimmed_offset_field
    )

if __name__ == "__main__":
    main()

