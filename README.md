# QSM Forward Model

Based on Marques, J. P., et al. (2021). QSM reconstruction challenge 2.0: A realistic in silico head phantom for MRI data simulation and evaluation of susceptibility mapping procedures. Magnetic Resonance in Medicine, 86(1), 526-542. https://doi.org/10.1002/mrm.28716

Includes code for:

 - Field model (forward multiplication with dipole kernel based on chi)
 - Signal model (magnitude and phase simulation based on field/M0/R1/R2star)
 - Phase offset model
 - Noise model
 - Shim field model
 - k-space cropping

## Install

```
pip install qsm-forward
```

## Example using simulated sources

In this example, we simulated susceptibility sources (spheres and rectangles) to generate a BIDS directory:

```python
import qsm_forward

if __name__ == "__main__":
    recon_params = qsm_forward.ReconParams()
    recon_params.subject = "simulated-sources"
    recon_params.peak_snr = 100
    recon_params.random_seed = 42

    tissue_params = qsm_forward.TissueParams(
        chi=qsm_forward.generate_susceptibility_phantom(
            resolution=[100, 100, 100],
            background=0,
            large_cylinder_val=0.005,
            small_cylinder_radii=[4, 4, 4, 7],
            small_cylinder_vals=[0.05, 0.1, 0.2, 0.5]
        )
    )

    qsm_forward.generate_bids(tissue_params, recon_params, "bids")
```

```
bids/
└── sub-simulated-sources
    └── ses-1
        ├── anat
        │   ├── sub-simulated-sources_ses-1_run-1_echo-1_part-mag_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-1_part-mag_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-1_part-phase_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-1_part-phase_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-2_part-mag_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-2_part-mag_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-2_part-phase_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-2_part-phase_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-3_part-mag_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-3_part-mag_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-3_part-phase_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-3_part-phase_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-4_part-mag_MEGRE.json
        │   ├── sub-simulated-sources_ses-1_run-1_echo-4_part-mag_MEGRE.nii
        │   ├── sub-simulated-sources_ses-1_run-1_echo-4_part-phase_MEGRE.json
        │   └── sub-simulated-sources_ses-1_run-1_echo-4_part-phase_MEGRE.nii
        └── extra_data
            ├── sub-simulated-sources_ses-1_run-1_chi.nii
            ├── sub-simulated-sources_ses-1_run-1_mask.nii
            └── sub-simulated-sources_ses-1_run-1_segmentation.nii
```

Some repesentative images including the mask, first and last-echo phase image, and ground truth susceptibility (chi):

![Image](https://i.imgur.com/3zpKbP0.png)

## Example using head phantom data

In this example, we generate a BIDS-compliant dataset based on the [realistic in-silico head phantom](https://doi.org/10.34973/m20r-jt17). If you have access to the head phantom, you need to retain the `data` directory which provides relevant tissue parameters:

```python
import qsm_forward
import numpy as np

if __name__ == "__main__":
    tissue_params = qsm_forward.TissueParams(root_dir="~/data")
    
    recon_params_all = [
        qsm_forward.ReconParams(voxel_size=voxel_size, peak_snr=100, random_seed=42, session=session)
        for (voxel_size, session) in [
            (np.array([0.8, 0.8, 0.8]), "0p8"),
            (np.array([1.0, 1.0, 1.0]), "1p0"),
            (np.array([1.2, 1.2, 1.2]), "1p2")
        ]
    ]

    for recon_params in recon_params_all:    
        qsm_forward.generate_bids(tissue_params=tissue_params, recon_params=recon_params, bids_dir="bids")
```

```
bids/
└── sub-1
    ├── ses-0p8
    │   ├── anat
    │   │   ├── sub-1_ses-0p8_run-1_echo-1_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-1_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-1_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-1_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-2_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-2_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-2_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-2_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-3_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-3_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-3_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-3_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-4_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-0p8_run-1_echo-4_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-0p8_run-1_echo-4_part-phase_MEGRE.json
    │   │   └── sub-1_ses-0p8_run-1_echo-4_part-phase_MEGRE.nii
    │   └── extra_data
    │       ├── sub-1_ses-0p8_run-1_chi.nii
    │       ├── sub-1_ses-0p8_run-1_mask.nii
    │       └── sub-1_ses-0p8_run-1_segmentation.nii
    ├── ses-1p0
    │   ├── anat
    │   │   ├── sub-1_ses-1p0_run-1_echo-1_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-1_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-1_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-1_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-2_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-2_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-2_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-2_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-3_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-3_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-3_part-phase_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-3_part-phase_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-4_part-mag_MEGRE.json
    │   │   ├── sub-1_ses-1p0_run-1_echo-4_part-mag_MEGRE.nii
    │   │   ├── sub-1_ses-1p0_run-1_echo-4_part-phase_MEGRE.json
    │   │   └── sub-1_ses-1p0_run-1_echo-4_part-phase_MEGRE.nii
    │   └── extra_data
    │       ├── sub-1_ses-1p0_run-1_chi.nii
    │       ├── sub-1_ses-1p0_run-1_mask.nii
    │       └── sub-1_ses-1p0_run-1_segmentation.nii
    └── ses-1p2
        ├── anat
        │   ├── sub-1_ses-1p2_run-1_echo-1_part-mag_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-1_part-mag_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-1_part-phase_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-1_part-phase_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-2_part-mag_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-2_part-mag_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-2_part-phase_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-2_part-phase_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-3_part-mag_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-3_part-mag_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-3_part-phase_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-3_part-phase_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-4_part-mag_MEGRE.json
        │   ├── sub-1_ses-1p2_run-1_echo-4_part-mag_MEGRE.nii
        │   ├── sub-1_ses-1p2_run-1_echo-4_part-phase_MEGRE.json
        │   └── sub-1_ses-1p2_run-1_echo-4_part-phase_MEGRE.nii
        └── extra_data
            ├── sub-1_ses-1p2_run-1_chi.nii
            ├── sub-1_ses-1p2_run-1_mask.nii
            └── sub-1_ses-1p2_run-1_segmentation.nii
```

Some repesentative images including the ground truth chi map, first-echo magnitude image, and first and last-echo phase images:

![Image](https://i.imgur.com/cE1cQ3U.png)

## Example including T1-weighted images

```python
import qsm_forward
import numpy as np

if __name__ == "__main__":
    tissue_params = qsm_forward.TissueParams(root_dir="~/data", chi="ChiModelMIX.nii.gz")
    
    recon_params_all = [
        qsm_forward.ReconParams(voxel_size=voxel_size, session=session, TEs=TEs, TR=TR, flip_angle=flip_angle, random_seed=42, suffix=suffix, save_phase=save_phase)
        for (voxel_size, session, TEs, TR, flip_angle, suffix, save_phase) in [
            (np.array([0.64, 0.64, 0.64]), "0p64", np.array([3.5e-3]), 7.5e-3, 40, "T1w", False),
            (np.array([0.64, 0.64, 0.64]), "0p64", np.array([0.004, 0.012, 0.02, 0.028]), 0.05, 15, "T2starw", True),
        ]
    ]

    for recon_params in recon_params_all:    
        qsm_forward.generate_bids(tissue_params=tissue_params, recon_params=recon_params, bids_dir="bids")
```

```
bids/
└── sub-1
    └── ses-0p64
        ├── anat
        │   ├── sub-1_ses-0p64_run-1_echo-1_part-mag_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-1_part-mag_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-1_part-phase_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-1_part-phase_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-2_part-mag_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-2_part-mag_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-2_part-phase_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-2_part-phase_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-3_part-mag_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-3_part-mag_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-3_part-phase_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-3_part-phase_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-4_part-mag_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-4_part-mag_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_echo-4_part-phase_MEGRE.json
        │   ├── sub-1_ses-0p64_run-1_echo-4_part-phase_MEGRE.nii
        │   ├── sub-1_ses-0p64_run-1_T1w.json
        │   └── sub-1_ses-0p64_run-1_T1w.nii
        └── extra_data
            ├── sub-1_ses-0p64_run-1_chi.nii
            ├── sub-1_ses-0p64_run-1_mask.nii
            └── sub-1_ses-0p64_run-1_segmentation.nii
```

Some repesentative images including the T2starw and T1w magnitude images:

![Image](https://i.imgur.com/RVzdhRz.png)

## Example simulating oblique acquisition

In this [example](qsm_forward/examples/simulated_sources_oblique.py), we simulated spherical susceptibility sources to generate a BIDS directory with a range of B0 directions:

![Image](https://i.imgur.com/8GDsqiN.png)

On the left is the phase image with the two sources with an axial B0 direction. On the right is a phase image with the two sources with a B0 direction rotated 30 degrees about the x axis.

