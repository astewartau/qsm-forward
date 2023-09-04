#!/usr/bin/env python3

import argparse
import qsm_forward
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(
        description='Simulate magnitude and phase'
    )

    def argparse_bool(user_in):
        if user_in is None: return None
        if isinstance(user_in, bool): return user_in
        user_in = user_in.strip().lower()
        if user_in in ['on', 'true', 'yes']: return True
        if user_in in ['off', 'false', 'no']: return False
        raise ValueError(f"Invalid boolean value {user_in}; use on/yes/true or off/false/no")
    
    parser.add_argument('bids', help='Output BIDS directory')
    parser.add_argument('--maps', default=None, help='Head phantom maps directory')
    parser.add_argument('--subject', default='1')
    parser.add_argument('--session', default=None),
    parser.add_argument('--acq', default=None),
    parser.add_argument('--run', default=None),
    parser.add_argument('--TR', default=50e-3, type=float),
    parser.add_argument('--TEs', default=[ 4e-3, 12e-3, 20e-3, 28e-3 ], type=float, nargs='+')
    parser.add_argument('--flip_angle', default=15)
    parser.add_argument('--B0', default=7, type=float)
    parser.add_argument('--B0-dir', default=[0., 0., 1.], type=float, nargs=3)
    parser.add_argument('--generate-phase-offset', nargs='?', type=argparse_bool, const=True, default=True)
    parser.add_argument('--generate-shim-field', nargs='?', type=argparse_bool, const=True, default=True)
    parser.add_argument('--voxel-size', default=[1., 1., 1.], type=float, nargs=3)
    parser.add_argument('--peak-snr', default=np.inf, type=float)
    parser.add_argument('--random-seed', default=None, type=int)
    parser.add_argument('--save-chi', nargs='?', type=argparse_bool, const=True, default=True)
    parser.add_argument('--save-mask', nargs='?', type=argparse_bool, const=True, default=True)
    parser.add_argument('--save-segmentation', nargs='?', type=argparse_bool, const=True, default=True)
    parser.add_argument('--save-field', nargs='?', type=argparse_bool, const=False, default=False)
    parser.add_argument('--save-shimmed-field', nargs='?', type=argparse_bool, const=False, default=False)
    parser.add_argument('--save-shimmed-offset-field', nargs='?', type=argparse_bool, const=False, default=False)

    args = parser.parse_args()

    if args.maps is not None:
        tissue_params = qsm_forward.TissueParams(args.maps)
    else:
        tissue_params = qsm_forward.TissueParams(chi=qsm_forward.simulate_susceptibility_sources())
    
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
        random_seed=args.random_seed
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

