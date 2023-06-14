import glob

import json
import os
import nibabel as nib
import numpy as np

import qsm_forward

tissue_params = {
    "chi_path" : "head-phantom-maps/ChiModelMIX_noCalc.nii.gz",
    "M0_path" : "head-phantom-maps/M0.nii.gz",
    "R1_path" : "head-phantom-maps/R1.nii.gz",
    "R2star_path" : "head-phantom-maps/R2star.nii.gz",
    "mask_path" : "head-phantom-maps/BrainMask.nii.gz",
    "seg_path" : "head-phantom-maps/SegmentedModel.nii.gz"
}

recon_params = {
    "Subject" : "1",
    "Session" : "1",
    "Run" : 1,
    "TR" : 50e-3,
    "TEs" : np.array([ 4e-3, 12e-3, 20e-3, 28e-3 ]),
    "flip_angle" : 15,
    "B0" : 7,
    "B0_dir" : np.array([0, 0, 1]),
    "phase_offset" : 0,
    "shimm" : False,
    "voxel_size" : np.array([1.0, 1.0, 1.0]),
    "peak_snr" : np.inf,
    "out_dir" : os.path.join(os.getcwd(), "bids")
}


if __name__ == "__main__":
    # create output directories
    print("Creating output directory...")
    os.makedirs(recon_params['out_dir'], exist_ok=True)
    
    # recon name
    recon_name = f"sub-{recon_params['Subject']}_ses-{recon_params['Session']}_run-{recon_params['Run']}"
    session_dir = os.path.join(recon_params['out_dir'], f"sub-{recon_params['Subject']}", f"ses-{recon_params['Session']}")
    os.makedirs(os.path.join(session_dir, 'anat'))
    os.makedirs(os.path.join(session_dir, 'extra_data'))

    # calculate field
    print("Computing field model...")
    chi_nii = nib.load(tissue_params['chi_path'])
    chi = chi_nii.get_fdata()
    field = qsm_forward.generate_field(chi)
    del chi

    # simulate shim field
    print("Computing shim fields...")
    brain_mask = nib.load(tissue_params['mask_path']).get_fdata()
    _, field, _ = qsm_forward.generate_shimmed_field(field, brain_mask, order=2)

    # phase offset
    print("Computing phase offset...")
    M0 = nib.load(tissue_params['M0_path']).get_fdata()
    phase_offset = qsm_forward.generate_phase_offset(M0, brain_mask, M0.shape)
    del brain_mask

    # image-space resizing of chi
    print("Image-space resizing of chi...")
    chi_downsampled_nii = qsm_forward.resize(chi_nii, recon_params['voxel_size'])
    nib.save(chi_downsampled_nii, filename=os.path.join(session_dir, "extra_data", "chi.nii"))
    print("Image-space cropping of brain mask...")
    mask_downsampled_nii = qsm_forward.resize(nib.load(tissue_params['mask_path']), recon_params['voxel_size'], 'nearest')
    nib.save(mask_downsampled_nii, filename=os.path.join(session_dir, "extra_data", "brain-mask.nii"))
    del mask_downsampled_nii
    print("Image-space cropping of segmentation...")
    seg_downsampled_nii = qsm_forward.resize(nib.load(tissue_params['seg_path']), recon_params['voxel_size'], 'nearest')
    nib.save(seg_downsampled_nii, filename=os.path.join(session_dir, "extra_data", "brain-mask.nii"))
    del seg_downsampled_nii

    # signal model
    multiecho = len(recon_params['TEs']) > 1
    for i in range(len(recon_params['TEs'])):
        print(f"Computing MR signal for echo {i+1}...")
        recon_name_i = f"{recon_name}_echo-{i+1}" if multiecho else recon_name
        sigHR = qsm_forward.generate_signal(
            field=field,
            B0=recon_params['B0'],
            TR=recon_params['TR'],
            TE=recon_params['TEs'][i],
            flip_angle=recon_params['flip_angle'],
            phase_offset=phase_offset,
            R1=nib.load(tissue_params['R1_path']).get_fdata(),
            R2star=nib.load(tissue_params['R2star_path']).get_fdata(),
            M0=nib.load(tissue_params['M0_path']).get_fdata()
        )
        #nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR), affine=chi_nii.affine, header=chi_nii.header), filename=os.path.join(recon_params['out_dir'], "mag.nii"))
        #nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR), affine=chi_nii.affine, header=chi_nii.header), filename=os.path.join(recon_params['out_dir'], "phs.nii"))
    
        # k-space cropping of sigHR
        print(f"k-space cropping of MR signal for echo {i+1}...")
        resolution = np.array(np.round((np.array(chi_nii.header.get_zooms()) / recon_params['voxel_size']) * np.array(chi_nii.header.get_data_shape())), dtype=int)
        sigHR_cropped = qsm_forward.crop_kspace(sigHR, resolution)
        del sigHR
        #nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params['out_dir'], "cropped_mag.nii"))
        #nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped), affine=chi_cropped_nii.affine, header=chi_cropped_nii.header), filename=os.path.join(recon_params['out_dir'], "cropped_phs.nii"))

        # noise
        print(f"Simulating noise for echo {i+1}...")
        sigHR_cropped_noisy = qsm_forward.add_noise(sigHR_cropped, peak_snr=recon_params['peak_snr'])
        del sigHR_cropped

        # save nifti images
        mag_filename = f"{recon_name_i}_part-mag" + ("_MEGRE" if multiecho else "_T2starw")
        phs_filename = f"{recon_name_i}_part-phase" + ("_MEGRE" if multiecho else "_T2starw")
        nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(recon_params['out_dir'], f"{mag_filename}.nii"))
        nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(recon_params['out_dir'], f"{phs_filename}.nii"))

        # json header
        print(f"Creating JSON headers...")
        json_dict = { 'EchoTime': recon_params['TEs'][i], 'MagneticFieldStrength': recon_params['B0'], 'EchoNumber': i+1, 'ProtocolName': 'T2starw', 'ConversionSoftware': 'qsm-forward' }

        json_dict_phs = json_dict.copy()
        json_dict_phs['ImageType'] = ['P', 'PHASE']
        json_dict_mag = json_dict.copy()
        json_dict_mag['ImageType'] = ['M', 'MAGNITUDE']

        with open(os.path.join(recon_params['out_dir'], f"{mag_filename}.json"), 'w') as mag_json_file:
            json.dump(json_dict_mag, mag_json_file)
        with open(os.path.join(recon_params['out_dir'], f"{phs_filename}.json"), 'w') as phs_json_file:
            json.dump(json_dict_phs, phs_json_file)

    print("Done!")

