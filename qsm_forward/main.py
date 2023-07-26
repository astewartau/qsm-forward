import argparse
import qsm_forward
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(
        description='Simulate magnitude and phase'
    )
    
    parser.add_argument('bids', help='Output BIDS directory')
    parser.add_argument('--maps', default=None, help='Head phantom maps directory')
    parser.add_argument('--subject', default='1')
    parser.add_argument('--session', default='1'),
    parser.add_argument('--run', default='1'),
    parser.add_argument('--TR', default=50e-3, type=float),
    parser.add_argument('--TEs', default=[ 4e-3, 12e-3, 20e-3, 28e-3 ], type=float, nargs='+')
    parser.add_argument('--flip_angle', default=15)
    parser.add_argument('--B0', default=7, type=float)
    parser.add_argument('--B0-dir', default=[0., 0., 1.], type=float, nargs=3)
    parser.add_argument('--generate-phase-offset', action='store_true', default=True, type=bool)
    parser.add_argument('--generate-shim-field', action='store_true', default=True, type=bool)
    parser.add_argument('--voxel-size', default=[1., 1., 1.], type=float, nargs=3)
    parser.add_argument('--peak-snr', default=np.inf, type=float)
    parser.add_argument('--random-seed', default=None, type=int)
    parser.add_argument('--save-chi', action='store_true', default=True, type=bool)
    parser.add_argument('--save-mask', action='store_true', default=True, type=bool)
    parser.add_argument('--save-segmentation', action='store_true', default=True, type=bool)
    parser.add_argument('--save-field', action='store_true', default=False, type=bool)
    parser.add_argument('--save-shimmed-field', action='store_true', default=False, type=bool)
    parser.add_argument('--save-shimmed-offset-field', action='store_true', default=False, type=bool)

    args = parser.parse_args()

    if args.maps is not None:
        tissue_params = qsm_forward.TissueParams(args.maps)
    else:
        tissue_params = qsm_forward.TissueParams(chi=qsm_forward.simulate_susceptibility_sources())
    
    recon_params = qsm_forward.ReconParams(
        subject=args.subject,
        session=args.session,
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

