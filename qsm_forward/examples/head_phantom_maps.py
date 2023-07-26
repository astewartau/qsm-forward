#!/usr/bin/env python

"""
This script uses the qsm_forward library to generate BIDS-compliant files 
from MRI simulations. 

The simulations are carried out with various reconstruction parameters 
(which include different voxel sizes), and the results are saved in the "bids" directory.

The tissue parameters are loaded from the directory "../head-phantom-maps".

Author: Ashley Stewart (a.stewart.au@gmail.com)
"""

import qsm_forward
import numpy as np

if __name__ == "__main__":
    tissue_params = qsm_forward.TissueParams("../head-phantom-maps", apply_mask=False)
    
    recon_params_all = [
        qsm_forward.ReconParams(voxel_size=voxel_size, peak_snr=100, session=session)
        for (voxel_size, session) in [
            (np.array([0.8, 0.8, 0.8]), "0p8"),
            (np.array([1.0, 1.0, 1.0]), "1p0"),
            (np.array([1.2, 1.2, 1.2]), "1p2")
        ]
    ]

    for recon_params in recon_params_all:    
        qsm_forward.generate_bids(tissue_params=tissue_params, recon_params=recon_params, bids_dir="bids")

