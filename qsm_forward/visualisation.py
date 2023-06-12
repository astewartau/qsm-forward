import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

def get_slice(volume, dim=0, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[dim] // 2
    idx_tuple = tuple(slice_idx if i == dim else slice(None) for i in range(len(volume.shape)))
    return volume[idx_tuple], slice_idx

def show_histogram(data, mask=None, title=None, dim=0, n_bins=100, n_ticks=10, n_rotations=0, vrange_hist=None, vrange_ylim=None, vrange_imshow=None, show_range=True, show_mean=True, show_stds=True, cmap='gray', units=None, precision=3, **args):
    # if str
    if isinstance(data, str): data = torch.tensor(nib.load(data).get_fdata())
    
    # if not tensor
    if not isinstance(data, torch.Tensor): data = torch.tensor(data)

    # get slice
    slice, slice_idx = get_slice(data, dim=dim) if data.dim() == 3 else data
    if mask is not None:
        mask_slice = get_slice(mask)[0] if mask.dim() == 3 else mask
    
    # rotate slice
    if n_rotations:
        slice = np.rot90(slice, k=n_rotations)
        if mask is not None:
            mask_slice = np.rot90(mask_slice, k=n_rotations)
    
    # superfigure
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figwidth(15)
    if title: fig.suptitle(title)#fontsize=29

    # data subplot
    if vrange_imshow is None: vrange_imshow = (float(data.min()), float(data.max()))
    ax1.set_title(f"volume[{slice_idx},:,:]" if dim == 0 else (f"volume[:,{slice_idx},:]" if dim == 1 else f"volume[:,:,{slice_idx}]"))
    im = ax1.imshow(slice, vmin=vrange_imshow[0], vmax=vrange_imshow[1], cmap=cmap, **args)
    if mask is not None: ax1.imshow(mask_slice, cmap='jet', interpolation='none', alpha=1.0*(mask_slice>0))
    ax1.set_axis_off()
    plt.colorbar(im, ax=ax1)#, orientation='horizontal')#, pad=0.2)

    # choose bins and ticks
    if n_bins <= 20:
        n_ticks = n_bins

    # histogram subplot
    if vrange_hist is None: vrange_hist = (float(data.min()), float(data.max()))
    hist = torch.histc(data, bins=n_bins, min=vrange_hist[0], max=vrange_hist[1])
    ax2.set_title("Histogram")
    ax2.bar(range(len(hist)), hist, align='center', color='skyblue')
    if units: ax2.set_xlabel(units)
    if vrange_ylim: ax2.set_ylim(vrange_ylim[0], vrange_ylim[1])
    if mask is not None:
        hist2 = torch.histc(data*mask, bins=n_bins, min=vrange_hist[0], max=vrange_hist[1])
        hist2 *= (max(hist) / max(hist2)) * 0.1
        ax2.bar(range(len(hist2)), hist2, align='center', color='red')
    
    ticks = np.array(np.linspace(start=0, stop=len(hist), num=n_ticks, endpoint=False))
    labels = [round(float(vrange_hist[0] + i*((vrange_hist[1] - vrange_hist[0]) / n_ticks)), precision) for i in range(n_ticks)]
    ax2.set_xticks(ticks=ticks, labels=labels, rotation=45, ha='right')

    def get_closest_tick(val):
        return np.argmin(np.abs(np.array(actual_ticks)-np.array(val)))
    
    def get_tick_from_val(val):
        return np.interp(val, (actual_ticks[0], actual_ticks[-1]), (ticks[0], ticks[-1]))

    actual_ticks = np.linspace(start=vrange_hist[0], stop=vrange_hist[1], num=n_bins)#.tolist()
    trans = ax2.get_xaxis_transform()
    if show_range:
        l1 = ax2.axvline(x=get_tick_from_val(vrange_imshow[0]), color='b')
        l2 = ax2.axvline(x=get_tick_from_val(vrange_imshow[1]), color='b')
        #l1 = ax2.axvline(x=get_closest_tick(vrange_imshow[0]), color='b')
        #l2 = ax2.axvline(x=get_closest_tick(vrange_imshow[1]), color='b')
        plt.text(get_tick_from_val(vrange_imshow[0]), .85, f"{round(vrange_imshow[0],precision)}", transform=trans, horizontalalignment='left' if vrange_imshow[0] <= vrange_hist[0] else 'right')
        plt.text(get_tick_from_val(vrange_imshow[1]), .85, f"{round(vrange_imshow[1],precision)}", transform=trans, horizontalalignment='right' if vrange_imshow[1] >= vrange_hist[1] else 'left')
    if show_mean:
        l3 = ax2.axvline(x=get_tick_from_val(data.mean()), color='r')
        plt.text(get_tick_from_val(data.mean()), .9, f"{round(float(data.mean()),precision)}±{round(float(data.std()),precision)}", transform=trans, horizontalalignment='left')
    if show_stds:
        l4 = ax2.axvline(x=get_tick_from_val(data.mean() + data.std()), color='darkred')
        l5 = ax2.axvline(x=get_tick_from_val(data.mean() - data.std()), color='darkred')

    plt.show()
    plt.close()

def show_image(
        nii_path=None,
        nii_object=None,
        numpy_array=None,
        slice_idx=None,
        dim_idx=0,
        rotations=0,
        title=None,
        caption=None,
        cmap='gray',
        interpolation='nearest',
        cbar=False,
        cbar_title=None,
        cbar_ticks=2,
        discrete=False,
        **kwargs
    ):

    # get image data
    if nii_path is not None:
        img_data = nib.load(nii_path).get_fdata()
    elif nii_object is not None:
        img_data = nii_object.get_fdata()
    elif numpy_array is not None:
        img_data = numpy_array
    else:
        raise ValueError()
    
    # only works with 3d data
    assert(len(img_data.shape) == 3)
    
    # dim_idx must be valid
    assert(0 <= dim_idx <= len(img_data.shape) - 1)
    
    # slice_idx must be valid
    if slice_idx is not None:
        assert(0 <= slice_idx <= img_data.shape[dim_idx])
    else:
        slice_idx = img_data.shape[dim_idx] // 2

    # get slice data
    if dim_idx == 0:
        slice_data = img_data[slice_idx,:,:]
    elif dim_idx == 1:
        slice_data = img_data[:,slice_idx,:]
    elif dim_idx == 2:
        slice_data = img_data[:,:,slice_idx]

    # rotate slice data
    if rotations != 0:
        slice_data = np.rot90(m=slice_data, k=1)

    fig, ax = plt.subplots()

    if title: ax.set_title(title)

    # Define the colormap
    num_colors = len(np.unique(img_data))
    unique_vals = np.unique(img_data)

    if discrete:
        cmap = plt.get_cmap(cmap, num_colors)
        midpoints = np.diff(unique_vals) / 2 + unique_vals[:-1]
        boundaries = np.concatenate(([unique_vals[0] - 0.5], midpoints, [unique_vals[-1] + 0.5]))
        norm = matplotlib.colors.BoundaryNorm(boundaries, num_colors)
        im = ax.imshow(X=slice_data, cmap=cmap, interpolation=interpolation, norm=norm, **kwargs)
    else:
        im = ax.imshow(X=slice_data, cmap=cmap, interpolation=interpolation, **kwargs)
    
    ax.set_xticks([])
    ax.set_yticks([])

    if cbar:
        im_ratio = slice_data.shape[0]/slice_data.shape[1]
        if discrete:
            cbar = ax.figure.colorbar(im, ax=ax, ticks=sorted(np.round(np.unique(img_data), decimals=3)), fraction=0.046*im_ratio, pad=0.04)# shrink=0.88)
            #cbar = fig.colorbar(im, fraction=1.0)
            cbar.ax.set_yticklabels(sorted(np.round(np.unique(img_data), decimals=3)))
        else:
            if all(x in kwargs for x in ['vmin', 'vmax']):
                cbar = ax.figure.colorbar(im, ax=ax, ticks=np.linspace(kwargs['vmin'], kwargs['vmax'], cbar_ticks), fraction=0.046*im_ratio, pad=0.04)
            else:
                cbar = ax.figure.colorbar(im, ax=ax, ticks=np.linspace(np.min(img_data), np.max(img_data), cbar_ticks), fraction=0.046*im_ratio, pad=0.04)
            #cbar = plt.colorbar(im, shrink=0.88)
        if cbar_title:
            cbar.ax.set_ylabel(cbar_title, rotation=0, ha='left')

    if caption: ax.set_xlabel(caption)

    plt.show()
    plt.close()

def view_slices_3d(image_3d, slice_nbr, vmin, vmax, title=""):
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontsize=10)

    plt.subplot(131)
    plt.imshow(np.take(image_3d, slice_nbr, 2), vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis('off')
    plt.title("Axial")

    plt.subplot(132)
    image_rot = ndimage.rotate(np.take(image_3d, slice_nbr, 1), 90)
    plt.imshow(image_rot, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis('off')
    plt.title("Coronal")

    plt.subplot(133)
    image_rot = ndimage.rotate(np.take(image_3d, slice_nbr, 0), 90)
    plt.imshow(image_rot, vmin=vmin, vmax=vmax, cmap="gray")
    plt.axis('off')
    plt.title("Sagittal")
    cbar = plt.colorbar(shrink=0.5)

