# qsm_forward


### _class_ qsm_forward.ReconParams(subject='1', session='1', run='1', TR=0.05, TEs=array([0.004, 0.012, 0.02, 0.028]), flip_angle=15, B0=7, B0_dir=array([0, 0, 1]), phase_offset=0, generate_phase_offset=True, generate_shim_field=True, voxel_size=array([1., 1., 1.]), peak_snr=inf)
A class used to represent reconstruction parameters.

## Attributes

subject

    The ID of the subject.

session

    The ID of the session.

run

    The run number.

TR

    Repetition time (in seconds).

TEs

    Echo times (in seconds).

flip_angle

    Flip angle (in degrees).

B0

    Magnetic field strength (in Tesla).

B0_dir

    B0 field direction.

phase_offset

    Phase offset (in radians).

generate_phase_offset

    Boolean to control phase offset generation.

generate_shim_field

    Boolean to control shim field generation.

voxel_size

    Voxel size (in mm).

peak_snr

    Peak signal-to-noise ratio.


### _class_ qsm_forward.TissueParams(root_dir='', chi='ChiModelMIX_noCalc.nii.gz', M0='M0.nii.gz', R1='R1.nii.gz', R2star='R2star.nii.gz', mask='BrainMask.nii.gz', seg='SegmentedModel.nii.gz')
A class used to represent tissue parameters.

## Attributes

chi_path

    The path to the Chi file or a 3D numpy array containing Chi values.

M0_path

    The path to the M0 file or a 3D numpy array containing M0 values.

R1_path

    The path to the R1 file or a 3D numpy array containing R1 values.

R2star_path

    The path to the R2\* file or a 3D numpy array containing R2\* values.

mask_path

    The path to the brain mask file or a 3D numpy array containing brain mask values.

seg_path

    The path to the segmentation file or a 3D numpy array containing segmentation values.


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


### qsm_forward.generate_bids(tissue_params: TissueParams, recon_params: ReconParams, bids_dir)
Simulate T2\*-weighted magnitude and phase images and save the outputs in the BIDS-compliant format.

This function simulates a T2\*-weighted MRI signal based on a ground truth susceptibility map,
and saves the outputs (images, JSON headers) in the BIDS-compliant format in the specified
directory.

## Parameters

tissue_params

    Provides paths to different tissue parameter files or the 3D numpy arrays themselves.

recon_params

    Provides parameters for the simulated reconstruction.

bids_dir

    The directory where the BIDS-formatted outputs will be saved.

## Returns

None

    Outputs are saved as files in the bids_dir directory.


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


### qsm_forward.simulate_susceptibility_sources(simulation_dim=160, rectangles_total=50, spheres_total=50, sus_std=1, shape_size_min_factor=0.01, shape_size_max_factor=0.5)
This function simulates susceptibility sources by generating a three-dimensional numpy array, 
and populating it with a certain number of randomly generated and positioned rectangular prisms and spheres.

## Parameters

simulation_dim

    The size of the simulation space in each dimension (i.e., the simulation space is simulation_dim^3).

rectangles_total

    The total number of rectangular prisms to generate in the simulation space.

spheres_total

    The total number of spheres to generate in the simulation space.

sus_std

    The standard deviation of the Gaussian distribution from which susceptibility values are drawn.

shape_size_min_factor

    A factor to determine the minimum size of the shapes (both rectangular prisms and spheres). 
    The actual minimum size in each dimension is calculated as simulation_dim \* shape_size_min_factor.

shape_size_max_factor

    A factor to determine the maximum size of the shapes (both rectangular prisms and spheres). 
    The actual maximum size in each dimension is calculated as simulation_dim \* shape_size_max_factor.

## Returns

temp_sources

    A three-dimensional numpy array of size (simulation_dim, simulation_dim, simulation_dim) 
    that contains the simulated susceptibility sources. Rectangular prisms and spheres have susceptibility 
    values drawn from a Gaussian distribution, while all other points are set to zero.
