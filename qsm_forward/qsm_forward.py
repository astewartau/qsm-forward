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
    chi_path : str or ndarray
        The path to the Chi file or a 3D numpy array containing Chi values.
    M0_path : str or ndarray
        The path to the M0 file or a 3D numpy array containing M0 values.
    R1_path : str or ndarray
        The path to the R1 file or a 3D numpy array containing R1 values.
    R2star_path : str or ndarray
        The path to the R2* file or a 3D numpy array containing R2* values.
    mask_path : str or ndarray
        The path to the brain mask file or a 3D numpy array containing brain mask values.
    seg_path : str or ndarray
        The path to the segmentation file or a 3D numpy array containing segmentation values.
    """

    def __init__(
            self,
            root_dir = "",
            chi = "ChiModelMIX_noCalc.nii.gz",
            M0 = "M0.nii.gz",
            R1 = "R1.nii.gz",
            R2star = "R2star.nii.gz",
            mask = "BrainMask.nii.gz",
            seg = "SegmentedModel.nii.gz",
            apply_mask = False
    ):
        if isinstance(chi, str) and not os.path.exists(os.path.join(root_dir, chi)):
            raise ValueError(f"Path to chi is invalid! ({os.path.join(root_dir, chi)})")
        self._chi = os.path.join(root_dir, chi) if isinstance(chi, str) and os.path.exists(os.path.join(root_dir, chi)) else chi if not isinstance(chi, str) else None
        self._M0 = os.path.join(root_dir, M0) if isinstance(M0, str) and os.path.exists(os.path.join(root_dir, M0)) else M0 if not isinstance(M0, str) else None
        self._R1 = os.path.join(root_dir, R1) if isinstance(R1, str) and os.path.exists(os.path.join(root_dir, R1)) else R1 if not isinstance(R1, str) else None
        self._R2star = os.path.join(root_dir, R2star) if isinstance(R2star, str) and os.path.exists(os.path.join(root_dir, R2star)) else R2star if not isinstance(R2star, str) else None
        self._mask = os.path.join(root_dir, mask) if isinstance(mask, str) and os.path.exists(os.path.join(root_dir, mask)) else mask if not isinstance(mask, str) else None
        self._seg = os.path.join(root_dir, seg) if isinstance(seg, str) and os.path.exists(os.path.join(root_dir, seg)) else seg if not isinstance(seg, str) else None
        self._apply_mask = apply_mask

    @property
    def voxel_size(self):
        zooms = self.nii_header.get_zooms()
        return zooms if len(zooms) == 3 else np.array([zooms[0] for i in range(3)])

    @property
    def nii_header(self):
        if isinstance(self._chi, str):
            return nib.load(self._chi).header
        header = nib.Nifti1Header()
        return header
    
    @property
    def nii_affine(self):
        if isinstance(self._chi, str):
            return nib.load(self._chi).affine
        return np.eye(4)

    def _do_apply_mask(self, nii): return nib.Nifti1Image(dataobj=nii.get_fdata() * self.mask.get_fdata(), affine=nii.affine, header=nii.header) if self._apply_mask else nii

    @property
    def chi(self): return self._do_apply_mask(nib.load(self._chi) if isinstance(self._chi, str) else nib.Nifti1Image(self._chi, affine=self.nii_affine, header=self.nii_header))

    @property
    def mask(self): return nib.load(self._mask) if isinstance(self._mask, str) else nib.Nifti1Image(self._mask or np.array(self._chi != 0), affine=self.nii_affine, header=self.nii_header)

    @property
    def M0(self): return self._do_apply_mask(nib.load(self._M0) if isinstance(self._M0, str) else nib.Nifti1Image(self._M0 or np.array(self.mask.get_fdata() * 1), affine=self.nii_affine, header=self.nii_header))

    @property
    def R1(self): return self._do_apply_mask(nib.load(self._R1) if isinstance(self._R1, str) else nib.Nifti1Image(self._R1 or np.array(self.mask.get_fdata() * 1), affine=self.nii_affine, header=self.nii_header))
    
    @property
    def R2star(self): return self._do_apply_mask(nib.load(self._R2star) if isinstance(self._R2star, str) else nib.Nifti1Image(self._R2star or np.array(self.mask.get_fdata() * 50), affine=self.nii_affine, header=self.nii_header))
    
    @property
    def seg(self): return nib.load(self._seg) if isinstance(self._seg, str) else nib.Nifti1Image(self._seg or self.mask.get_fdata(), affine=self.nii_affine, header=self.nii_header)
    

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
    random_seed : int
        Random seed to use for noise.
    """

    def __init__(
            self,
            subject="1",
            session="1",
            run="1",
            TR=50e-3,
            TEs=np.array([ 4e-3, 12e-3, 20e-3, 28e-3 ]),
            flip_angle=15,
            B0=7,
            B0_dir=np.array([0, 0, 1]),
            phase_offset=0,
            generate_phase_offset=True,
            generate_shim_field=True,
            voxel_size=np.array([1.0, 1.0, 1.0]),
            peak_snr=np.inf,
            random_seed=None
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
        self.random_seed = random_seed

def generate_bids(tissue_params: TissueParams, recon_params: ReconParams, bids_dir, save_chi=True, save_mask=True, save_segmentation=True, save_field=False, save_shimmed_field=False, save_shimmed_offset_field=False):
    """
    Simulate T2*-weighted magnitude and phase images and save the outputs in the BIDS-compliant format.

    This function simulates a T2*-weighted MRI signal based on a ground truth susceptibility map,
    and saves the outputs (images, JSON headers) in the BIDS-compliant format in the specified
    directory.

    Parameters
    ----------
    tissue_params : TissueParams
        Provides paths to different tissue parameter files or the 3D numpy arrays themselves.
    recon_params : ReconParams
        Provides parameters for the simulated reconstruction.
    bids_dir : str
        The directory where the BIDS-formatted outputs will be saved.
    save_chi : bool
        Whether to save the cropped chi map to the BIDS directory. Default is True.
    save_mask : bool
        Whether to save the cropped mask to the BIDS directory. Default is True.
    save_segmentation : bool
        Whether to save the cropped segmentation to the BIDS directory. Default is True.
    save_field : bool
        Whether to save the cropped field map to the BIDS directory. Default is False.
    save_shimmed_field : bool
        Whether to save the cropped and shimmed field map to the BIDS directory. Default is False.
    save_shimmed_offset_field : bool
        Whether to save the cropped, shimmed and offset field map to the BIDS directory. Default is False.

    Returns
    -------
    None
        Outputs are saved as files in the bids_dir directory.

    """

    # create output directories
    print("Creating output directory...")
    os.makedirs(bids_dir, exist_ok=True)
    
    # recon name
    recon_name = f"sub-{recon_params.subject}_ses-{recon_params.session}_run-{recon_params.run}"
    session_dir = os.path.join(bids_dir, f"sub-{recon_params.subject}", f"ses-{recon_params.session}")
    os.makedirs(os.path.join(session_dir, 'anat'), exist_ok=True)
    os.makedirs(os.path.join(session_dir, 'extra_data'), exist_ok=True)

    # random number generator for noise etc.
    rng = np.random.default_rng(recon_params.random_seed)

    # image-space resizing
    print("Image-space resizing of chi...")
    chi_downsampled_nii = resize(tissue_params.chi, recon_params.voxel_size)
    if save_chi: nib.save(chi_downsampled_nii, filename=os.path.join(session_dir, "extra_data", f"{recon_name}_chi.nii"))
    print("Image-space cropping of mask...")
    if save_mask: nib.save(resize(tissue_params.mask, recon_params.voxel_size, 'nearest'), filename=os.path.join(session_dir, "extra_data", f"{recon_name}_mask.nii"))
    print("Image-space cropping of segmentation...")
    if save_segmentation: nib.save(resize(tissue_params.seg, recon_params.voxel_size, 'nearest'), filename=os.path.join(session_dir, "extra_data", f"{recon_name}_segmentation.nii"))

    # calculate field
    print("Computing field model...")
    field = generate_field(tissue_params.chi.get_fdata(), voxel_size=tissue_params.voxel_size, B0_dir=recon_params.B0_dir)
    if save_field: nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(session_dir, "extra_data", f"{recon_name}_field.nii"))

    # simulate shim field
    if recon_params.generate_shim_field:
        print("Computing shim fields...")
        _, field, _ = generate_shimmed_field(field, tissue_params.mask.get_fdata(), order=2)
        if save_shimmed_field: nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(session_dir, "extra_data", f"{recon_name}_field-shimmed.nii"))

    # phase offset
    phase_offset = recon_params.phase_offset
    if recon_params.generate_phase_offset:
        print("Computing phase offset...")
        phase_offset = recon_params.phase_offset + generate_phase_offset(tissue_params.M0.get_fdata(), tissue_params.mask.get_fdata(), tissue_params.M0.get_fdata().shape)
        if save_shimmed_offset_field: nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(session_dir, "extra_data", f"{recon_name}_field-shimmed-offset.nii"))

    # signal model
    multiecho = len(recon_params.TEs) > 1
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
            R1=tissue_params.R1.get_fdata(),
            R2star=tissue_params.R2star.get_fdata(),
            M0=tissue_params.M0.get_fdata()
        )
    
        # k-space cropping of sigHR
        print(f"k-space cropping of MR signal for echo {i+1}...")
        resolution = np.array(np.round((np.array(tissue_params.voxel_size) / recon_params.voxel_size) * np.array(tissue_params.nii_header.get_data_shape())), dtype=int)
        sigHR_cropped = crop_kspace(sigHR, resolution)
        del sigHR

        # noise
        print(f"Simulating noise for echo {i+1}...")
        sigHR_cropped_noisy = add_noise(sigHR_cropped, peak_snr=recon_params.peak_snr, rng=rng)
        del sigHR_cropped

        # save nifti images
        mag_filename = f"{recon_name_i}_part-mag" + ("_MEGRE" if multiecho else "_T2starw")
        phs_filename = f"{recon_name_i}_part-phase" + ("_MEGRE" if multiecho else "_T2starw")
        nib.save(nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy)*500, affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(session_dir, "anat", f"{mag_filename}.nii"))
        nib.save(nib.Nifti1Image(dataobj=np.angle(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header), filename=os.path.join(session_dir, "anat", f"{phs_filename}.nii"))

        # json header
        print(f"Creating JSON headers...")
        json_dict = { 
            'EchoTime': recon_params.TEs[i],
            'MagneticFieldStrength': recon_params.B0,
            'EchoNumber': i+1,
            'ProtocolName': 'T2starw',
            'ConversionSoftware': 'qsm-forward',
            'RepetitionTime': recon_params.TR,
            'FlipAngle': recon_params.flip_angle,
            'B0_dir': recon_params.B0_dir.tolist(),
            'PhaseOffset': recon_params.generate_phase_offset or phase_offset != 0,
            'ShimmField': recon_params.generate_shim_field,
            'VoxelSize': recon_params.voxel_size.tolist(),
            'PeakSNR': recon_params.peak_snr if recon_params.peak_snr != np.inf else "inf"
        }

        json_dict_phs = json_dict.copy()
        json_dict_phs['ImageType'] = ['P', 'PHASE']
        json_dict_mag = json_dict.copy()
        json_dict_mag['ImageType'] = ['M', 'MAGNITUDE']

        with open(os.path.join(session_dir, "anat", f"{mag_filename}.json"), 'w') as mag_json_file:
            json.dump(json_dict_mag, mag_json_file)
        with open(os.path.join(session_dir, "anat", f"{phs_filename}.json"), 'w') as phs_json_file:
            json.dump(json_dict_phs, phs_json_file)

    print("Done!")


def generate_field(chi, voxel_size=[1, 1, 1], B0_dir=[0, 0, 1]):
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
    D = _generate_3d_dipole_kernel(data_shape=dims, voxel_size=voxel_size, B0_dir=B0_dir)
    
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
    sigHR[np.isnan(sigHR)]

    return sigHR

def add_noise(sig, peak_snr=np.inf, rng=None):
    """
    Add complex Gaussian noise to a signal.

    Parameters
    ----------
    sig : numpy.ndarray
        The input signal to which noise will be added.
    peak_snr : float, optional
        The peak signal-to-noise ratio, by default np.inf
    rng : numpy.random.Generator, optional
        A random number Generator. If None, a new Generator will be created.

    Returns
    -------
    numpy.ndarray
        The input signal with added noise.
    """

    # Create a new RNG if one was not provided
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)
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
    original_shape = np.array(nii.header.get_data_shape())
    target_shape = np.array(np.round((np.array(nii.header.get_zooms()) / voxel_size) * np.array(nii.header.get_data_shape())), dtype=int)

    if np.array_equal(original_shape, target_shape):
        return nii

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

    if np.array_equal(x.shape, np.array(shape)):
        return x
        
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

def crop_kspace(volume, dims, scaling=True, gibbs_correction=True):
    """
    Crop a 3D volume in k-space and apply optional scaling and Gibbs ringing correction.

    Parameters
    ----------
    volume : numpy.ndarray
        The input 3D volume.
    dims : tuple of int
        The desired dimensions after cropping.
    scaling : bool, optional
        Whether to scale the cropped volume to maintain the total energy. Default is True.
    gibbs_correction : bool, optional
        Whether to apply Gibbs ringing correction. Default is True.

    Returns
    -------
    numpy.ndarray
        The cropped volume.

    """

    if np.array_equal(volume.shape, dims):
        return volume

    working_volume = np.fft.ifftn(np.fft.ifftshift(crop_imagespace(np.fft.fftshift(np.fft.fftn(volume)), dims)))
    
    # gibbs correction is only needed for non-complex volumes
    if not np.iscomplexobj(volume):
        working_volume = np.real(working_volume)
        
        if gibbs_correction:
            working_volume = gibbs_removal(gibbs_removal(working_volume, slice_axis=2), slice_axis=1)

    if scaling:
        working_volume *= np.prod(dims) / np.prod(volume.shape)
    
    return working_volume


def _generate_3d_dipole_kernel(data_shape, voxel_size, B0_dir):
    """
    Generate a 3D dipole kernel.

    This function generates a 3D dipole kernel used in the forward convolution step of the QSM simulation.

    Parameters
    ----------
    data_shape : tuple of int
        The shape of the data array (nx, ny, nz).
    voxel_size : list of float
        The size of a voxel in each direction (dx, dy, dz).
    B0_dir : list of float
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
    D = np.fft.fftshift(1 / 3 - ((kx * B0_dir[1] + ky * B0_dir[0] + kz * B0_dir[2])**2 / k2))
    
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

def generate_susceptibility_phantom(resolution, background, large_cylinder_val, small_cylinder_radii, small_cylinder_vals):
    assert len(small_cylinder_radii) == len(small_cylinder_vals), "Number of small cylinders and their values should be the same"
    
    # Initialize the 3D array with the background value
    array = np.full(resolution, fill_value=background, dtype=float)

    # Calculate the center and the large radius
    center = [res//2 for res in resolution]
    large_radius = min(center[1:]) * 0.75

    # Create coordinates for the 3D array
    z,y,x = np.indices(resolution)
    
    # Calculate the lower and upper limit for the height
    lower_limit1 = (1 - 0.75) / 2 * resolution[0]
    upper_limit2 = (1 + 0.75) / 2 * resolution[0]

    lower_limit3 = (1 - 0.6) / 2 * resolution[0]
    upper_limit4 = (1 + 0.6) / 2 * resolution[0]
    
    # Create the large cylinder along x-axis
    large_cylinder = ((z-center[2])**2 + (y-center[1])**2 < large_radius**2) & (x >= lower_limit1) & (x < upper_limit2)
    array[large_cylinder] = large_cylinder_val

    # Calculate angle between each small cylinder
    angle = 2*np.pi/len(small_cylinder_radii)
    
    # Create the small cylinders
    for i, (small_radius, small_val) in enumerate(zip(small_cylinder_radii, small_cylinder_vals)):
        # Calculate center of the small cylinder
        small_center_z = center[2] + large_radius/2 * np.cos(i*angle)
        small_center_y = center[1] + large_radius/2 * np.sin(i*angle)
        
        small_cylinder = ((z-small_center_z)**2 + (y-small_center_y)**2 < small_radius**2) & (x >= lower_limit3) & (x < upper_limit4)
        array[small_cylinder] = small_val
    
    return array

def simulate_susceptibility_sources(
    simulation_dim=160,
    rectangles_total=50,
    spheres_total=50,
    sus_std=1,
    shape_size_min_factor=0.01,
    shape_size_max_factor=0.5,
    seed=None
):
    """
    This function simulates susceptibility sources by generating a three-dimensional numpy array, 
    and populating it with a certain number of randomly generated and positioned rectangular prisms and spheres.
    
    Parameters
    ----------
    simulation_dim : int
        The size of the simulation space in each dimension (i.e., the simulation space is simulation_dim^3).
        
    rectangles_total : int
        The total number of rectangular prisms to generate in the simulation space.
        
    spheres_total : int
        The total number of spheres to generate in the simulation space.
        
    sus_std : float
        The standard deviation of the Gaussian distribution from which susceptibility values are drawn.
        
    shape_size_min_factor : float
        A factor to determine the minimum size of the shapes (both rectangular prisms and spheres). 
        The actual minimum size in each dimension is calculated as simulation_dim * shape_size_min_factor.
        
    shape_size_max_factor : float
        A factor to determine the maximum size of the shapes (both rectangular prisms and spheres). 
        The actual maximum size in each dimension is calculated as simulation_dim * shape_size_max_factor.

    seed : int, optional
        A seed for the random number generator. If None, a random seed will be used.
        
    Returns
    -------
    temp_sources : ndarray
        A three-dimensional numpy array of size (simulation_dim, simulation_dim, simulation_dim) 
        that contains the simulated susceptibility sources. Rectangular prisms and spheres have susceptibility 
        values drawn from a Gaussian distribution, while all other points are set to zero.
    """

    temp_sources = np.zeros((simulation_dim, simulation_dim, simulation_dim))

    # Create a new generator instance with the provided seed if one was given
    rng = np.random.default_rng(seed)

    # Generate rectangles
    for shapes in range(rectangles_total):
        shrink_factor = 1 / ((shapes / rectangles_total + 1))
        shape_size_min = np.floor(
            simulation_dim * shrink_factor * shape_size_min_factor
        )
        shape_size_max = np.floor(
            simulation_dim * shrink_factor * shape_size_max_factor
        )

        susceptibility_value = rng.normal(loc=0.0, scale=sus_std)
        random_sizex = rng.integers(low=shape_size_min, high=shape_size_max)
        random_sizey = rng.integers(low=shape_size_min, high=shape_size_max)
        random_sizez = rng.integers(low=shape_size_min, high=shape_size_max)
        x_pos = rng.integers(simulation_dim)
        y_pos = rng.integers(simulation_dim)
        z_pos = rng.integers(simulation_dim)

        x_pos_max = x_pos + random_sizex
        if x_pos_max >= simulation_dim:
            x_pos_max = simulation_dim

        y_pos_max = y_pos + random_sizey
        if y_pos_max >= simulation_dim:
            y_pos_max = simulation_dim

        z_pos_max = z_pos + random_sizez
        if z_pos_max >= simulation_dim:
            z_pos_max = simulation_dim

        temp_sources[
            x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max
        ] = susceptibility_value

    # Generate spheres
    for sphere in range(spheres_total):
        susceptibility_value = rng.normal(loc=0.0, scale=sus_std)
        sphere_radius = rng.integers(low=shape_size_min//2, high=shape_size_max//2)
        x_center = rng.integers(simulation_dim)
        y_center = rng.integers(simulation_dim)
        z_center = rng.integers(simulation_dim)

        # Iterate over the 3D array
        for x in range(max(0, x_center-sphere_radius), min(simulation_dim, x_center+sphere_radius)):
            for y in range(max(0, y_center-sphere_radius), min(simulation_dim, y_center+sphere_radius)):
                for z in range(max(0, z_center-sphere_radius), min(simulation_dim, z_center+sphere_radius)):
                    # Determine if this point is inside the sphere
                    if (x - x_center) ** 2 + (y - y_center) ** 2 + (z - z_center) ** 2 <= sphere_radius ** 2:
                        temp_sources[x, y, z] = susceptibility_value

    return temp_sources


