import glob

import os
import nibabel as nib
import numpy as np

import qsm_forward_3d

class TissueParams:
    chi_path = "head-phantom-maps/ChiModelMIX_noCalc.nii.gz"
    M0_path = "head-phantom-maps/M0.nii.gz"
    R1_path = "head-phantom-maps/R1.nii.gz"
    R2star_path = "head-phantom-maps/R2star.nii.gz"
    mask_path = "head-phantom-maps/BrainMask.nii.gz"

class ReconParams:
    TR = 50e-3
    TEs = np.array([ 4e-3, 12e-3, 20e-3, 28e-3 ])
    flip_angle = 15
    B0 = 7
    B0_dir = np.array([0, 0, 1])
    phase_offset = 0
    shimm = False
    voxel_size = np.array([1.0, 1.0, 1.0])
    peak_snr = np.inf
    out_dir = os.path.join(os.getcwd(), "outputs")


def default_recon_params():
    return ReconParams()

def default_tissue_params():
    return TissueParams()


if __name__ == "__main__":
    # get parameters
    print("Loading parameters...")
    recon_params = default_recon_params()
    tissue_params = default_tissue_params()

    # create output directories
    print("Creating output directory...")
    os.makedirs(recon_params.out_dir, exist_ok=True)

    # calculate field
    print("Computing field model...")
    chi_nii = nib.load(tissue_params.chi_path)
    chi = chi_nii.get_fdata()
    field = qsm_forward_3d.forward_convolution_v2(chi)
    del chi

    # simulate shim field
    print("Computing shim fields...")
    brain_mask = nib.load(tissue_params.mask_path).get_fdata()
    _, field, _ = qsm_forward_3d.poly_fit_shim_like(field, brain_mask, order=2)

    # phase offset
    print("Computing phase offset...")
    M0 = nib.load(tissue_params.M0_path).get_fdata()
    phase_offset = qsm_forward_3d.compute_phase_offset1(M0, brain_mask, M0.shape)
    del brain_mask

    # signal model
    print("Computing MR signal...")
    sigHR = qsm_forward_3d.signal_model(
        field=field,
        B0=recon_params.B0,
        TR=recon_params.TR,
        TE=recon_params.TEs[0],
        flip_angle=recon_params.flip_angle,
        phase_offset=phase_offset,
        R1=nib.load(tissue_params.R1_path).get_fdata(),
        R2star=nib.load(tissue_params.R2star_path).get_fdata(),
        M0=nib.load(tissue_params.M0_path).get_fdata()
    )
    nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR), affine=chi_nii.affine, header=chi_nii.header), filename=os.path.join(recon_params.out_dir, "mag.nii"))
    nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR), affine=chi_nii.affine, header=chi_nii.header), filename=os.path.join(recon_params.out_dir, "phs.nii"))

    # image-space cropping of chi
    print("Image-space cropping of chi...")
    chi_cropped_nii = qsm_forward_3d.resize(chi_nii, recon_params.voxel_size)
    nib.save(chi_cropped_nii, filename=os.path.join(recon_params.out_dir, "cropped_qsm.nii"))
    
    # k-space cropping of sigHR
    print("k-space cropping of MR signal")
    resolution = np.array(np.round((np.array(chi_nii.header.get_zooms()) / recon_params.voxel_size) * np.array(chi_nii.header.get_data_shape())), dtype=int)
    sigHR_cropped = qsm_forward_3d.kspace_crop(sigHR, resolution)
    del sigHR
    nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params.out_dir, "cropped_mag.nii"))
    nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params.out_dir, "cropped_phs.nii"))

    # noise
    print("Simulating noise...")
    sigHR_cropped_noisy = qsm_forward_3d.add_noise(sigHR_cropped, peak_snr=recon_params.peak_snr)
    del sigHR_cropped
    nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params.out_dir, "cropped_mag_noisy.nii"))
    nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped_noisy), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params.out_dir, "cropped_phs_noisy.nii"))

    print("Done!")

