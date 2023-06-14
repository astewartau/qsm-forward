# qsm_forward


### qsm_forward.add_noise(sig, peak_snr=inf)
Add complex Gaussian noise to a signal.

## Parameters

sig

    The input signal to which noise will be added.

peak_snr

    The peak signal-to-noise ratio, by default np.inf

## Returns

numpy.ndarray

    The input signal with added noise.


### qsm_forward.crop_imagespace(x, shape)
Crop a nD matrix around its center.

## Parameters

x

    The input n-dimensional matrix.

shape

    The desired shape after cropping.

## Returns

numpy.ndarray

    The cropped matrix.


### qsm_forward.crop_kspace(volume, dims, scaling=False, gibbs_correction=True)
Crop a 3D volume in k-space and apply optional scaling and Gibbs ringing correction.

## Parameters

volume

    The input 3D volume.

dims

    The desired dimensions after cropping.

scaling

    Whether to scale the cropped volume to maintain the total energy. Default is False.

gibbs_correction

    Whether to apply Gibbs ringing correction. Default is True.

## Returns

numpy.ndarray

    The cropped volume.


### qsm_forward.generate_bids(tissue_params, recon_params, bids_dir)
Generate T2\*-weighted MRI signals and save the outputs in the BIDS-compliant format.

This function simulates a T2\*-weighted MRI signal based on a ground truth susceptibility map,
computes the necessary parameters, and saves the outputs (images, JSON headers) in the BIDS-compliant
format in the specified directory.

Parameters:
tissue_params (dict): A dictionary containing paths to different tissue parameter files.

> Keys are: “chi_path”, “M0_path”, “R1_path”, “R2star_path”, “mask_path”, and “seg_path”.

recon_params (dict): A dictionary containing the parameters for the reconstruction simulation.

    Keys are: “Subject”, “Session”, “Run”, “TR”, “TEs”, “flip_angle”, “B0”, “B0_dir”,
    “phase_offset”, “shimm”, “voxel_size”, “peak_snr”.

bids_dir (str): The directory where the BIDS-formatted outputs will be saved.

Returns:
None. Outputs are saved as files in the bids_dir directory.


### qsm_forward.generate_field(chi)
Perform the forward convolution operation.

This function performs the forward convolution step of the QSM simulation.

## Parameters

chi

    The susceptibility distribution array.

## Returns

numpy.ndarray

    The resulting magnetic field array after the forward convolution operation.


### qsm_forward.generate_phase_offset(M0, mask, dims)
Generate a suitable phase offset.

## Parameters

M0

    The initial magnetization.

mask

    A binary mask that indicates the internal region of interest.

dims

    The dimensions of the input image.

## Returns

numpy.ndarray

    The phase offset of the input image.


### qsm_forward.generate_shimmed_field(field, mask, order=2)
Simulate field shimming by fitting the field with second- and third-order Legendre polynomials.

## Parameters

field

    3D array representing the magnetic field to fit.

mask

    3D binary array. Must be the same shape as field. A True value at a coordinate will 
    include that point in the fit.

order

    The order of the polynomial to fit. Must be 0, 1, or 2. Default is 2.

## Returns

FIT3D

    3D array representing the fitted field.

Residuals

    3D array representing the residuals of the fit.

b

    1D array representing the coefficients of the fitted polynomial.

## Raises

ValueError

    If field and mask shapes are not the same.


### qsm_forward.generate_signal(field, B0=3, TR=1, TE=0.03, flip_angle=90, phase_offset=0, R1=1, R2star=50, M0=1)
Compute the MRI signal based on the given parameters.

## Parameters

field

    The magnetic field distribution.

B0

    The main magnetic field strength. Default is 3.

TR

    The repetition time. Default is 1.

TE

    The echo time. Default is 30e-3.

flip_angle

    The flip angle in degrees. Default is 90.

phase_offset

    The phase offset. Default is 0.

R1

    The longitudinal relaxation rate. Can be a single value or a 3D numpy array. Default is 1.

R2star

    The effective transverse relaxation rate. Can be a single value or a 3D numpy array. Default is 50.

M0

    The equilibrium magnetization. Can be a single value or a 3D numpy array. Default is 1.

## Returns

numpy.ndarray

    The computed MRI signal.


### qsm_forward.resize(nii, voxel_size, interpolation='continuous')
Resize a Nifti image to a voxel size.

## Parameters

nii

    The input Nifti image.

voxel_size

    The desired voxel size after resizing.

interpolation

    Can be ‘continuous’, ‘linear’, or ‘nearest’. Indicates the resample method. Default=’continuous’.

## Returns

nibabel.nifti1.Nifti1Image

    The resized Nifti image.
