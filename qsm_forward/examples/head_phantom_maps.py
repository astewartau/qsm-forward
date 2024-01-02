#!/usr/bin/env python

"""
This script uses the qsm_forward library to generate BIDS-compliant files 
from MRI simulations. 

The simulations are carried out with various reconstruction parameters 
(which include different voxel sizes), and the results are saved in the "bids" directory.

The tissue parameters are loaded from the directory "../data". The data directory can
be downloaded from https://doi.org/10.34973/m20r-jt17.

Author: Ashley Stewart (a.stewart.au@gmail.com)
"""

import qsm_forward
import numpy as np

if __name__ == "__main__":
    tissue_params = qsm_forward.TissueParams(root_dir="~/data")
    
    recon_params_all = [
        qsm_forward.ReconParams(voxel_size=voxel_size, TEs=TEs, TR=TR, flip_angle=flip_angle, suffix=suffix, export_phase=export_phase)
        for (voxel_size, TEs, TR, flip_angle, suffix, export_phase) in [
            (np.array([1.0, 1.0, 1.0]), np.array([0.004, 0.012, 0.02, 0.028]), 0.05, 15, "T2starw", True),
            (np.array([1.0, 1.0, 1.0]), np.array([0.0035]), 0.0075, 40, "T1w", False)
        ]
    ]

    for recon_params in recon_params_all:    
        qsm_forward.generate_bids(tissue_params=tissue_params, recon_params=recon_params, bids_dir="bids")

