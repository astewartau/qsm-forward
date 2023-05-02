import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from nilearn.image import resample_img

def simulate_susceptibility_sources(
    simulation_dim=160,
    rectangles_total=100,
    spheres_total=80,
    sus_std=1,  # standard deviation of susceptibility values
    shape_size_min_factor=0.01,
    shape_size_max_factor=0.5,
):

    temp_sources = np.zeros((simulation_dim, simulation_dim, simulation_dim))

    for shapes in range(rectangles_total):
        shrink_factor = 1 / ((shapes / rectangles_total + 1))
        shape_size_min = np.floor(
            simulation_dim * shrink_factor * shape_size_min_factor
        )
        shape_size_max = np.floor(
            simulation_dim * shrink_factor * shape_size_max_factor
        )

        susceptibility_value = np.random.normal(loc=0.0, scale=sus_std)
        random_sizex = np.random.randint(low=shape_size_min, high=shape_size_max)
        random_sizey = np.random.randint(low=shape_size_min, high=shape_size_max)
        random_sizez = np.random.randint(low=shape_size_min, high=shape_size_max)
        x_pos = np.random.randint(simulation_dim)
        y_pos = np.random.randint(simulation_dim)
        z_pos = np.random.randint(simulation_dim)

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

    return temp_sources


def generate_3d_dipole_kernel_v1(data_shape, voxel_size, b0_dir):

    ky, kx, kz = np.meshgrid(
        np.arange(-data_shape[1] // 2, data_shape[1] // 2),
        np.arange(-data_shape[0] // 2, data_shape[0] // 2),
        np.arange(-data_shape[2] // 2, data_shape[2] // 2),
    )

    kx = kx / (data_shape[0] * voxel_size[0])
    ky = ky / (data_shape[1] * voxel_size[1])
    kz = kz / (data_shape[2] * voxel_size[2])

    k2 = kx**2 + ky**2 + kz**2
    k2[k2 == 0] = 1e-6
    D = 1 / 3 - ((kx * b0_dir[0] + ky * b0_dir[1] + kz * b0_dir[2])**2 / k2)

    return D


def generate_3d_dipole_kernel_v2(data_shape, voxel_size, b0_dir):
    # TODO Maybe increase size further (e.g. 2.5x)
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


def forward_convolution_v1(chi):
    dims = np.array(chi.shape)
    D = generate_3d_dipole_kernel_v1(data_shape=dims, voxel_size=[1, 1, 1], b0_dir=[0, 0, 1])

    scaling = np.sqrt(chi.size)
    chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chi))) / scaling
    chi_fft_t_kernel = chi_fft * D
    tissue_phase = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = np.real(tissue_phase * scaling)

    return tissue_phase


def forward_convolution_v2(chi):
    dims = np.array(chi.shape)
    D = generate_3d_dipole_kernel_v2(data_shape=dims, voxel_size=[1, 1, 1], b0_dir=[0, 0, 1])
    
    chitemp = np.ones(2 * dims) * chi[-1, -1, -1]
    chitemp[:dims[0], :dims[1], :dims[2]] = chi
    field = np.real(np.fft.ifftn(np.fft.fftn(chitemp) * D))
    field = field[:dims[0], :dims[1], :dims[2]]

    return field



### ===================== FORWARD MODEL CONVERSION FROM MATLAB ======================================
def add_noise(sig, peak_snr=np.inf):
    noise = np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    sig_noisy = sig + (noise * np.max(np.abs(sig))) / peak_snr
    return sig_noisy

def center_of_mass(data):
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


def compute_phase_offset1(M0, brain_mask, dims):
    c, w = center_of_mass(M0)
    
    x, y, z = np.meshgrid(
        np.arange(1, dims[1]+1)-c[1],
        np.arange(1, dims[0]+1)-c[0],
        np.arange(1, dims[2]+1)-c[2]
    )
    
    temp = (x/w[1])**2 + (y/w[0])**2 + (z/w[2])**2
    
    max_temp = np.max(temp[brain_mask != 0])
    min_temp = np.min(temp[brain_mask != 0])
    
    phase_offset = -temp / (max_temp - min_temp) * np.pi

    return phase_offset


def compute_phase_offset(M0, brain_mask, dims):
    c, w = center_of_mass(M0)
    
    y, x, z = np.meshgrid(
        np.arange(1, dims[1] + 1) - c[1],
        np.arange(1, dims[0] + 1) - c[0],
        np.arange(1, dims[2] + 1) - c[2]
    )
    
    temp = (x/w[0])**2 + (y/w[1])**2 + (z/w[2])**2
    temp_brain = temp[brain_mask == 1]
    
    temp_min = np.min(temp_brain)
    temp_max = np.max(temp_brain)

    phase_offset = -temp / (temp_max - temp_min) * np.pi
    
    return phase_offset

def resize(nii, voxel_size):
    return resample_img(nii, target_affine=np.diag(voxel_size), interpolation='nearest')


def crop(x, shape):
    # crops a nD matrix around its center
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

def kspace_crop(volume, dims, scaling=False, gibbs_correction=True):
    #chi_cropped = gibbs_removal(gibbs_removal(np.real(qsm_forward_3d.kspace_crop(chi, np.round(np.array(chi.shape) * 0.9))) - np.min(chi), slice_axis=2), slice_axis=1) + np.min(chi)
    
    working_volume = np.fft.ifftn(np.fft.ifftshift(crop(np.fft.fftshift(np.fft.fftn(volume)), dims)))
    
    # gibbs correction is only needed for non-complex volumes
    if not np.iscomplexobj(volume):
        working_volume = np.real(working_volume)
        
        if gibbs_correction:
            working_volume = gibbs_removal(gibbs_removal(working_volume, slice_axis=2), slice_axis=1)

    if scaling:
        working_volume *= np.prod(dims) / np.prod(volume.shape)
    
    return working_volume

def signal_model(field, B0=3, TR=1, TE=30e-3, flip_angle=90, phase_offset=0, R1=1, R2star=50, M0=1):
    sigHR = M0 * np.exp(1j * (2 * np.pi * field * B0 * 42.58 * TE + phase_offset)) * np.exp(-TE * R2star) \
        * (1 - np.exp(-TR * R1)) * np.sin(np.deg2rad(flip_angle)) / (1 - np.cos(np.deg2rad(flip_angle)) * np.exp(-TR * R1))
    sigHR[np.isnan(sigHR)] = 0

    return sigHR

def create_model(x1, y1, z1, dim, order):
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

def poly_fit_shim_like(volume, brain_mask, order=2):
    # field shimming was simulated by fitting the frequency map with second-order and third-order Legendre polynomials
    dim = volume.shape
    
    ## for volume fitting
    #brain_mask = np.ones(brain_mask.shape)
    indices = np.nonzero(brain_mask)
    x1, y1, z1 = indices
    R = volume[indices]
    b = None
    
    if len(indices[0]) > (3*order)**2:
        model = create_model(x1, y1, z1, dim, order)
        b = np.linalg.pinv(model) @ R
        temp = R - model @ b
        del model, R
        
        indices = np.meshgrid(*[range(d) for d in dim], indexing='ij')
        x1, y1, z1 = [ind.flatten() for ind in indices]
        model = create_model(x1, y1, z1, dim, order)
        
        Fit = model @ b
        del model
        
        FIT3D = Fit.reshape(dim)
        Residuals = (volume-FIT3D)
    else:
        FIT3D = np.zeros_like(volume)
        Residuals = (volume-FIT3D) * brain_mask
    
    return FIT3D, Residuals, b

