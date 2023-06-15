"""
Author: Ashley Stewart <a.stewart.au@gmail.com>

"""

import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from nilearn.image import resample_img

import json
import os
import nibabel as nib
import numpy as np

class TissueParams:
    """
    A class used to represent tissue parameters.

    Attributes
    ----------
    chi_path : str
        The path to the Chi file.
    M0_path : str
        The path to the M0 file.
    R1_path : str
        The path to the R1 file.
    R2star_path : str
        The path to the R2* file.
    mask_path : str
        The path to the brain mask file.
    seg_path : str
        The path to the segmentation file.
    """

    def __init__(
            self,
            root_dir = "",
            chi_fname = "ChiModelMIX_noCalc.nii.gz",
            M0_fname = "M0.nii.gz",
            R1_fname = "R1.nii.gz",
            R2star_fname = "R2star.nii.gz",
            mask_fname = "BrainMask.nii.gz",
            seg_fname = "SegmentedModel.nii.gz"
    ):
        self.chi_path = os.path.join(root_dir, chi_fname)
        self.M0_path = os.path.join(root_dir, M0_fname)
        self.R1_path = os.path.join(root_dir, R1_fname)
        self.R2star_path = os.path.join(root_dir, R2star_fname)
        self.mask_path = os.path.join(root_dir, mask_fname)
        self.seg_path = os.path.join(root_dir, seg_fname)
        

class ReconParams:
    """
    A class used to represent reconstruction parameters.

    Attributes
    ----------
    subject : str
        The ID of the subject.
    session : str
        The ID of the session.
    run : int
        The run number.
    TR : float
        Repetition time (in seconds).
    TEs : np.array
        Echo times (in seconds).
    flip_angle : int
        Flip angle (in degrees).
    B0 : int
        Magnetic field strength (in Tesla).
    B0_dir : np.array
        B0 field direction.
    phase_offset : int
        Phase offset (in radians).
    generate_phase_offset : bool
        Boolean to control phase offset generation.
    generate_shim_field : bool
        Boolean to control shim field generation.
    voxel_size : np.array
        Voxel size (in mm).
    peak_snr : float
        Peak signal-to-noise ratio.
    """

    def __init__(
            self,
            subject="1",
            session="1",
            run="1",
            TR=50e3,
            TEs=np.array([ 4e-3, 12e-3, 20e-3, 28e-3 ]),
            flip_angle=15,
            B0=7,
            B0_dir=np.array([0, 0, 1]),
            phase_offset=0,
            generate_phase_offset=True,
            generate_shim_field=True,
            voxel_size=np.array([1.0, 1.0, 1.0]),
            peak_snr=np.inf
        ):
        self.subject = subject
        self.session = session
        self.run = run
        self.TR = TR
        self.TEs = TEs
        self.flip_angle = flip_angle
        self.B0 = B0
        self.B0_dir = B0_dir
        self.phase_offset = phase_offset
        self.generate_phase_offset = generate_phase_offset
        self.generate_shim_field = generate_shim_field
        self.voxel_size = voxel_size
        self.peak_snr = peak_snr

def generate_bids(tissue_params, recon_params, bids_dir):
    """
    Simulate T2*-weighted magnitude and phase images and save the outputs in the BIDS-compliant format.

    This function simulates a T2*-weighted MRI signal based on a ground truth susceptibility map,
    and saves the outputs (images, JSON headers) in the BIDS-compliant format in the specified
    directory.

    Parameters:
    tissue_params (TissueParams): Provides paths to different tissue parameter files.

    recon_params (ReconParams): Provides parameters for the simulated reconstruction.

    bids_dir (str): The directory where the BIDS-formatted outputs will be saved.

    Returns:
    None. Outputs are saved as files in the bids_dir directory.

    """

    # create output directories
    print("Creating output directory...")
    os.makedirs(bids_dir, exist_ok=True)
    
    # recon name
    recon_name = f"sub-{recon_params.subject}_ses-{recon_params.session}_run-{recon_params.run}"
    session_dir = os.path.join(bids_dir, f"sub-{recon_params.subject}", f"ses-{recon_params.session}")
    os.makedirs(os.path.join(session_dir, 'anat'), exist_ok=True)
    os.makedirs(os.path.join(session_dir, 'extra_data'), exist_ok=True)

    # image-space resizing
    chi_nii = nib.load(tissue_params.chi_path)
    print("Image-space resizing of chi...")
    chi_downsampled_nii = resize(chi_nii, recon_params.voxel_size)
    nib.save(chi_downsampled_nii, filename=os.path.join(session_dir, "extra_data", f"{recon_name}_chi.nii"))
    if os.path.exists(str(tissue_params.mask_path)):
        print("Image-space cropping of mask...")
        mask_downsampled_nii = resize(nib.load(tissue_params.mask_path), recon_params.voxel_size, 'nearest')
        nib.save(mask_downsampled_nii, filename=os.path.join(session_dir, "extra_data", f"{recon_name}_mask.nii"))
        del mask_downsampled_nii
    if os.path.exists(str(tissue_params.seg_path)):
        print("Image-space cropping of segmentation...")
        seg_downsampled_nii = resize(nib.load(tissue_params.seg_path), recon_params.voxel_size, 'nearest')
        nib.save(seg_downsampled_nii, filename=os.path.join(session_dir, "extra_data", f"{recon_name}_segmentation.nii"))
        del seg_downsampled_nii

    # calculate field
    print("Computing field model...")
    chi = chi_nii.get_fdata()
    field = generate_field(chi)
    del chi

    # simulate shim field
    mask = nib.load(tissue_params.mask_path).get_fdata() if os.path.exists(tissue_params.mask_path) else np.ones(field.shape)
    if recon_params.generate_shim_field:
        print("Computing shim fields...")
        _, field, _ = generate_shimmed_field(field, mask, order=2)

    # phase offset
    M0 = nib.load(tissue_params.M0_path).get_fdata() if os.path.exists(tissue_params.M0_path) else np.ones(field.shape) * mask
    if recon_params.generate_phase_offset:
        print("Computing phase offset...")
        phase_offset = recon_params.phase_offset + generate_phase_offset(M0, mask, M0.shape)

    # signal model
    multiecho = len(recon_params.TEs) > 1
    R1 = nib.load(tissue_params.R1_path).get_fdata() if os.path.exists(tissue_params.R1_path) else np.ones(field.shape) * mask
    R2star = nib.load(tissue_params.R2star_path).get_fdata() if os.path.exists(tissue_params.R2star_path) else np.ones(field.shape) * mask
    del mask
    for i in range(len(recon_params.TEs)):
        print(f"Computing MR signal for echo {i+1}...")
        recon_name_i = f"{recon_name}_echo-{i+1}" if multiecho else recon_name

        sigHR = generate_signal(
            field=field,
            B0=recon_params.B0,
            TR=recon_params.TR,
            TE=recon_params.TEs[i],
            flip_angle=recon_params.flip_angle,
            phase_offset=phase_offset,
            R1=R1,
            R2star=R2star,
            M0=M0
        )
    
        # k-space cropping of sigHR
        print(f"k-space cropping of MR signal for echo {i+1}...")
        resolution = np.array(np.round((np.array(chi_nii.header.get_zooms()) / recon_params.voxel_size) * np.array(chi_nii.header.get_data_shape())), dtype=int)
        sigHR_cropped = crop_kspace(sigHR, resolution)
        del sigHR

        # noise
        print(f"Simulating noise for echo {i+1}...")
        sigHR_cropped_noisy = add_noise(sigHR_cropped, peak_snr=recon_params.peak_snr)
        del sigHR_cropped

        # save nifti images
        mag_filename = f"{recon_name_i}_part-mag" + ("_MEGRE" if multiecho else "_T2starw")
        phs_filename = f"{recon_name_i}_part-phase" + ("_MEGRE" if multiecho else "_T2starw")
        nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(session_dir, "anat", f"{mag_filename}.nii"))
        nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(session_dir, "anat", f"{phs_filename}.nii"))

        # json header
        print(f"Creating JSON headers...")
        json_dict = { 'EchoTime': recon_params.TEs[i], 'MagneticFieldStrength': recon_params.B0, 'EchoNumber': i+1, 'ProtocolName': 'T2starw', 'ConversionSoftware': 'qsm-forward' }

        json_dict_phs = json_dict.copy()
        json_dict_phs['ImageType'] = ['P', 'PHASE']
        json_dict_mag = json_dict.copy()
        json_dict_mag['ImageType'] = ['M', 'MAGNITUDE']

        with open(os.path.join(session_dir, "anat", f"{mag_filename}.json"), 'w') as mag_json_file:
            json.dump(json_dict_mag, mag_json_file)
        with open(os.path.join(session_dir, "anat", f"{phs_filename}.json"), 'w') as phs_json_file:
            json.dump(json_dict_phs, phs_json_file)

    print("Done!")


def generate_field(chi):
    """
    Perform the forward convolution operation.

    This function performs the forward convolution step of the QSM simulation.

    Parameters
    ----------
    chi : numpy.ndarray
        The susceptibility distribution array.

    Returns
    -------
    numpy.ndarray
        The resulting magnetic field array after the forward convolution operation.

    """
    dims = np.array(chi.shape)
    D = _generate_3d_dipole_kernel(data_shape=dims, voxel_size=[1, 1, 1], b0_dir=[0, 0, 1])
    
    chitemp = np.ones(2 * dims) * chi[-1, -1, -1]
    chitemp[:dims[0], :dims[1], :dims[2]] = chi
    field = np.real(np.fft.ifftn(np.fft.fftn(chitemp) * D))
    field = field[:dims[0], :dims[1], :dims[2]]

    return field

def generate_phase_offset(M0, mask, dims):
    """
    Generate a suitable phase offset.

    Parameters
    ----------
    M0 : numpy.ndarray
        The initial magnetization.
    mask : numpy.ndarray
        A binary mask that indicates the internal region of interest.
    dims : tuple of int
        The dimensions of the input image.

    Returns
    -------
    numpy.ndarray
        The phase offset of the input image.

    """

    c, w = _center_of_mass(M0)
    
    x, y, z = np.meshgrid(
        np.arange(1, dims[1]+1)-c[1],
        np.arange(1, dims[0]+1)-c[0],
        np.arange(1, dims[2]+1)-c[2]
    )
    
    temp = (x/w[1])**2 + (y/w[0])**2 + (z/w[2])**2
    
    max_temp = np.max(temp[mask != 0])
    min_temp = np.min(temp[mask != 0])
    
    phase_offset = -temp / (max_temp - min_temp) * np.pi

    return phase_offset


def generate_shimmed_field(field, mask, order=2):
    """
    Simulate field shimming by fitting the field with second- and third-order Legendre polynomials.

    Parameters
    ----------
    field : numpy.ndarray
        3D array representing the magnetic field to fit.
    mask : numpy.ndarray
        3D binary array. Must be the same shape as `field`. A True value at a coordinate will 
        include that point in the fit.
    order : int, optional
        The order of the polynomial to fit. Must be 0, 1, or 2. Default is 2.

    Returns
    -------
    FIT3D : numpy.ndarray
        3D array representing the fitted field.
    Residuals : numpy.ndarray
        3D array representing the residuals of the fit.
    b : numpy.ndarray
        1D array representing the coefficients of the fitted polynomial.

    Raises
    ------
    ValueError
        If `field` and `mask` shapes are not the same.
    """

    dim = field.shape
    
    ## for volume fitting
    #mask = np.ones(mask.shape)
    indices = np.nonzero(mask)
    x1, y1, z1 = indices
    R = field[indices]
    b = None
    
    if len(indices[0]) > (3*order)**2:
        model = _create_model(x1, y1, z1, dim, order)
        b = np.linalg.pinv(model) @ R
        temp = R - model @ b
        del model, R
        
        indices = np.meshgrid(*[range(d) for d in dim], indexing='ij')
        x1, y1, z1 = [ind.flatten() for ind in indices]
        model = _create_model(x1, y1, z1, dim, order)
        
        Fit = model @ b
        del model
        
        FIT3D = Fit.reshape(dim)
        Residuals = (field-FIT3D)
    else:
        FIT3D = np.zeros_like(field)
        Residuals = (field-FIT3D) * mask
    
    return FIT3D, Residuals, b

def generate_signal(field, B0=3, TR=1, TE=30e-3, flip_angle=90, phase_offset=0, R1=1, R2star=50, M0=1):
    """
    Compute the MRI signal based on the given parameters.

    Parameters
    ----------
    field : numpy.ndarray
        The magnetic field distribution.
    B0 : float, optional
        The main magnetic field strength. Default is 3.
    TR : float, optional
        The repetition time. Default is 1.
    TE : float, optional
        The echo time. Default is 30e-3.
    flip_angle : float, optional
        The flip angle in degrees. Default is 90.
    phase_offset : float, optional
        The phase offset. Default is 0.
    R1 : float or numpy.ndarray, optional
        The longitudinal relaxation rate. Can be a single value or a 3D numpy array. Default is 1.
    R2star : float or numpy.ndarray, optional
        The effective transverse relaxation rate. Can be a single value or a 3D numpy array. Default is 50.
    M0 : float or numpy.ndarray, optional
        The equilibrium magnetization. Can be a single value or a 3D numpy array. Default is 1.

    Returns
    -------
    numpy.ndarray
        The computed MRI signal.

    """

    sigHR = M0 * np.exp(1j * (2 * np.pi * field * B0 * 42.58 * TE + phase_offset)) * np.exp(-TE * R2star) \
        * (1 - np.exp(-TR * R1)) * np.sin(np.deg2rad(flip_angle)) / (1 - np.cos(np.deg2rad(flip_angle)) * np.exp(-TR * R1))
    sigHR[np.isnan(sigHR)] = 0

    return sigHR

def add_noise(sig, peak_snr=np.inf):
    """
    Add complex Gaussian noise to a signal.

    Parameters
    ----------
    sig : numpy.ndarray
        The input signal to which noise will be added.
    peak_snr : float, optional
        The peak signal-to-noise ratio, by default np.inf

    Returns
    -------
    numpy.ndarray
        The input signal with added noise.

    """

    noise = np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    sig_noisy = sig + (noise * np.max(np.abs(sig))) / peak_snr
    return sig_noisy


def resize(nii, voxel_size, interpolation='continuous'):
    """
    Resize a Nifti image to a voxel size.

    Parameters
    ----------
    nii : nibabel.nifti1.Nifti1Image
        The input Nifti image.
    voxel_size : list of float
        The desired voxel size after resizing.
    interpolation : str
        Can be 'continuous', 'linear', or 'nearest'. Indicates the resample method. Default='continuous'.

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        The resized Nifti image.

    """
    
    target_shape = np.array(np.round((np.array(nii.header.get_zooms()) / voxel_size) * np.array(nii.header.get_data_shape())), dtype=int)
    target_affine = np.diag(list(voxel_size) + [1])
    
    return resample_img(
        nii,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation=interpolation
    )


def crop_imagespace(x, shape):
    """
    Crop a nD matrix around its center.

    Parameters
    ----------
    x : numpy.ndarray
        The input n-dimensional matrix.
    shape : tuple of int
        The desired shape after cropping.

    Returns
    -------
    numpy.ndarray
        The cropped matrix.

    """
        
    m = np.array(x.shape)
    s = np.array(shape)
    if s.size < m.size:
        s = np.concatenate((s, np.ones(m.size - s.size, dtype=int)))
    if np.array_equal(m, s):
        res = x
        return res
    idx = []
    for n in range(s.size):
        start = np.floor_divide(m[n], 2) + np.ceil(-s[n] / 2)
        end = np.floor_divide(m[n], 2) + np.ceil(s[n] / 2)
        idx.append(slice(int(start), int(end)))
    res = x[tuple(idx)]
    return res

def crop_kspace(volume, dims, scaling=False, gibbs_correction=True):
    """
    Crop a 3D volume in k-space and apply optional scaling and Gibbs ringing correction.

    Parameters
    ----------
    volume : numpy.ndarray
        The input 3D volume.
    dims : tuple of int
        The desired dimensions after cropping.
    scaling : bool, optional
        Whether to scale the cropped volume to maintain the total energy. Default is False.
    gibbs_correction : bool, optional
        Whether to apply Gibbs ringing correction. Default is True.

    Returns
    -------
    numpy.ndarray
        The cropped volume.

    """

    working_volume = np.fft.ifftn(np.fft.ifftshift(crop_imagespace(np.fft.fftshift(np.fft.fftn(volume)), dims)))
    
    # gibbs correction is only needed for non-complex volumes
    if not np.iscomplexobj(volume):
        working_volume = np.real(working_volume)
        
        if gibbs_correction:
            working_volume = gibbs_removal(gibbs_removal(working_volume, slice_axis=2), slice_axis=1)

    if scaling:
        working_volume *= np.prod(dims) / np.prod(volume.shape)
    
    return working_volume


def _generate_3d_dipole_kernel(data_shape, voxel_size, b0_dir):
    """
    Generate a 3D dipole kernel.

    This function generates a 3D dipole kernel used in the forward convolution step of the QSM simulation.

    Parameters
    ----------
    data_shape : tuple of int
        The shape of the data array (nx, ny, nz).
    voxel_size : list of float
        The size of a voxel in each direction (dx, dy, dz).
    b0_dir : list of float
        The direction of the B0 field (B0x, B0y, B0z).

    Returns
    -------
    numpy.ndarray
        A 3D array representing the dipole kernel.

    """
    kx, ky, kz = np.meshgrid(
        np.arange(-data_shape[1], data_shape[1]),
        np.arange(-data_shape[0], data_shape[0]),
        np.arange(-data_shape[2], data_shape[2])
    )

    kx = kx / (2 * voxel_size[0] * np.max(np.abs(kx)))
    ky = ky / (2 * voxel_size[1] * np.max(np.abs(ky)))
    kz = kz / (2 * voxel_size[2] * np.max(np.abs(kz)))

    k2 = kx**2 + ky**2 + kz**2
    k2[k2 == 0] = np.finfo(float).eps
    D = np.fft.fftshift(1 / 3 - ((kx * b0_dir[0] + ky * b0_dir[1] + kz * b0_dir[2])**2 / k2))
    
    return D


def _center_of_mass(data):
    """
    Compute the center of mass of a 3D array.

    Parameters
    ----------
    data : numpy.ndarray
        The input 3D array.

    Returns
    -------
    tuple
        A tuple containing two arrays:
        1) The coordinates of the center of mass.
        2) The standard deviation along each axis.

    """

    data = np.abs(data)
    dims = np.shape(data)
    coord = np.zeros(len(dims))
    width = np.zeros(len(dims))

    for k in range(len(dims)):
        dimsvect = np.ones(len(dims), dtype=int)
        dimsvect[k] = dims[k]
        temp = np.multiply(data, np.reshape(np.arange(1, dims[k]+1), dimsvect))
        coord[k] = np.sum(temp)/np.sum(data)
        temp = np.multiply(data, np.power(np.reshape(np.arange(1, dims[k]+1), dimsvect)-coord[k], 2))
        width[k] = np.sqrt(np.sum(temp)/np.sum(data))

    return coord, width

def _create_model(x1, y1, z1, dim, order):
    """
    Creates a model based on x, y, z coordinates and the specified order.

    Parameters
    ----------
    x1, y1, z1 : numpy.ndarray
        1D arrays of the x, y, z coordinates respectively.
    dim : tuple of int
        The shape of the 3D space.
    order : int
        The order of the model to create. Must be 0, 1, or 2.

    Returns
    -------
    model : numpy.ndarray
        2D array where each row is the model for the corresponding point.

    Raises
    ------
    ValueError
        If order is not 0, 1, or 2.
    """
    Nsize = [1, 4, 10, 20, 35]
    N = Nsize[order+1]
    model = np.zeros((len(x1), N), dtype=float)
    
    # zeroth order
    if order >= 0:
        model[:, 0] = 1
    
    # first order
    if order >= 1:
        model[:, 1] = np.reshape(x1 - dim[0]/2, (len(x1),))
        model[:, 2] = np.reshape(y1 - dim[1]/2, (len(x1),))
        model[:, 3] = np.reshape(z1 - dim[2]/2, (len(x1),))
    
    # second order
    if order >= 2:
        model[:, 4] = model[:, 1] * model[:, 1] - model[:, 2] * model[:, 2] # siemens x^2 - y^2
        model[:, 5] = model[:, 1] * model[:, 2] # x^1 y^1 z^0 - siemens xy
        model[:, 6] = 2 * model[:, 3] * model[:, 3] - model[:, 2] * model[:, 2] - model[:, 1] * model[:, 1] # 2 z^2 - x^2 - y^2
        model[:, 7] = model[:, 2] * model[:, 3] # x^0 y^1 z^1 - siemens yz
        model[:, 8] = model[:, 3] * model[:, 1] # x^1 y^0 z^1 - siemens xz

    return model


