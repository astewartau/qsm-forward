import json
import os
import nibabel as nib
import numpy as np
import qsm_forward

if __name__ == "__main__":
    bids_dir = "bids"
    tissue_params = qsm_forward.default_tissue_params.copy()
    recon_params = qsm_forward.default_recon_params.copy()

    qsm_forward.generate_bids(tissue_params=tissue_params, recon_params=recon_params, bids_dir=bids_dir)

