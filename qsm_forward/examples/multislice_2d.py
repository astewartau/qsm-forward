#!/usr/bin/env python

"""
Generate 2D multi-slice phantom data, in the three steps that make it progressively
less like a 3D acquisition.

Each session is a separate run over the same tissue model:

  ses-thick    3 mm slices, contiguous, no slice phase offsets. A thick-slab
               acquisition - anisotropic, but still a contiguous 3D volume.
  ses-offsets  the same geometry with a per-slice receive phase offset, which is what
               actually breaks 3D region-growing phase unwrapping.
  ses-gap      3 mm slabs sampled at a 4 mm pitch. The sampled volume is no longer
               contiguous, so the FFT-based dipole kernel does not apply to it at all.
               Generated deliberately, to check that downstream tools refuse it.

The results are saved in the "bids" directory.

Author: Ashley Stewart (a.stewart.au@gmail.com)
"""

import numpy as np

import qsm_forward

if __name__ == "__main__":
    tissue_params = qsm_forward.TissueParams(
        chi=qsm_forward.generate_susceptibility_phantom(
            resolution=[100, 100, 96],
            background=0,
            large_cylinder_val=0.005,
            small_cylinder_radii=[4, 4, 4, 7],
            small_cylinder_vals=[0.05, 0.1, 0.2, 0.5]
        )
    )

    sessions = [
        ("thick",   dict()),
        ("offsets", dict(slice_phase_offsets="interleaved")),
        ("gap",     dict(slice_phase_offsets="interleaved", slice_gap=1.0)),
    ]

    for session, extra in sessions:
        recon_params = qsm_forward.ReconParams(
            subject="multislice",
            session=session,
            voxel_size=np.array([1.0, 1.0, 3.0]),
            peak_snr=100,
            random_seed=42,
            **extra
        )
        qsm_forward.generate_bids(tissue_params, recon_params, "bids", save_field=True)
