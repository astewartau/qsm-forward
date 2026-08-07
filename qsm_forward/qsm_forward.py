"""
Author: Ashley Stewart <a.stewart.au@gmail.com>

Please cite the following work, which provided the original MATLAB implementation
of the base head phantom (total susceptibility, field/signal/relaxation models):

Marques, J. P., et al. (2021). QSM reconstruction challenge 2.0: A realistic in silico head
phantom for MRI data simulation and evaluation of susceptibility mapping procedures.
Magnetic Resonance in Medicine, 86(1), 526-542. https://doi.org/10.1002/mrm.28716.

The chi-separation model implemented here (per-tissue paramagnetic/diamagnetic
susceptibility construction with R1/R2* modulation, white-matter susceptibility
anisotropy, the Dr/R2' relaxivity maps, T2/R2 and 3T-scaling maps, and the
chi-sep-aware GRE signal model) is a Python port of the
Susceptibility-Separation-Phantom
(https://github.com/neuropoly/Susceptibility-Separation-Phantom), MIT licensed,
(c) NeuroPoly 2024. ``generate_chisep_maps`` implements PhantomCreationFunction.m:
wavelet-denoised R1/R2*, per-tissue
``chitemp = R2star2chi*(R2*-mean) + R12chi*(R1-mean)``, the NSTD=3 outlier clamp,
the smooth-probability blend via ``smooth3D``, and the two-model
``Chimap5 = W^4*Chimap3 + (1-W^4)*Chimap4`` field-gradient weighting, building
chi+ and chi- per tissue from the base values. Per-tissue chi+/chi- values
(CHISEP_TISSUE_PARAMS), white-matter tract anisotropy values
(WM_TRACT_ANISOTROPY_PARAMS), and the Dr relaxivities are taken from that
phantom's data/chimodel/SusceptibilityValues.mat, PhantomCreation.m and
calculate_Dr.m. Please additionally cite:

Ridani, S., De Leener, B., & Alonso-Ortiz, E. (2026). A realistic in-silico brain
phantom for quantifying susceptibility anisotropy-induced error in susceptibility
separation. bioRxiv. https://doi.org/10.64898/2026.04.07.716972

You may also cite the repository https://github.com/astewartau/qsm-forward.

"""

from dipy.denoise.gibbs import gibbs_removal
from nilearn.image import resample_img
from scipy.ndimage import gaussian_filter

import json
import os
import nibabel as nib
import numpy as np
from importlib.metadata import version as _get_version
import site
import datetime


# Per-tissue chi-separation parameters (FAITHFUL to Ridani et al. 2026).
# For each tissue label:
#   (chipos, chineg, R12chipos, R2star2chipos, R12chineg, R2star2chineg)
#     chipos/chineg      : base paramagnetic/diamagnetic susceptibility (ppm)
#     R12chi*, R2star2chi*: per-tissue R1 / R2* modulation coefficients that add
#                           anatomically-valid intra-tissue texture (Marques-style):
#                             chi_component = base + R2star2chi*(R2*-mean) + R12chi*(R1-mean)
#
# ALL values are the AUTHORITATIVE per-tissue values from the
# Susceptibility-Separation-Phantom (Ridani, De Leener & Alonso-Ortiz, 2026; MIT,
# (c) NeuroPoly 2024), extracted at full precision from the `label` struct array in
#   data/chimodel/SusceptibilityValues.mat
# (fields: chipos, chineg, R12chipos, R2star2chipos, R12chineg, R2star2chineg).
# The label indices align 1:1 with SegmentedModel / label.json (1=Caudate ...
# 16=Calcification).
#
# chi_total(tissue) = chipos + chineg (PhantomCreationFunction.m "Create Chi total").
# chi+ and chi- are built per tissue from these base values plus the R1/R2*
# modulation coefficients above (see generate_chisep_maps).
CHISEP_TISSUE_PARAMS = {
    1:  (0.05265,      -0.00865,      -0.0119,    0.0011,                 -0.0119,    0.0),                     # Caudate nucleus
    2:  (0.143722992,  -0.013222992,  -0.0257,    0.0008428,              -0.0257,    0.0),                     # Globus pallidus
    3:  (0.047061074,  -0.009061074,  -0.0249,    0.0019,                  0.0249,    0.0),                     # Putamen
    4:  (0.110905893,  -0.010905893,  -0.0444,    0.00178,                -0.0444,    0.00178),                 # Red nucleus
    5:  (0.168411781,  -0.016411781,  -0.0639,    0.0017077,              -0.0639,    0.0017077),               # Dentate nucleus
    6:  (0.122433878,  -0.011433878,  -0.075,     0.0014913,              -0.075,     0.0014913),               # Substantia nigra & subthalamic nucleus
    7:  (0.050903658,  -0.030903658,  -0.0861799, 0.0012748418375849724,  -0.0861799, 0.0012748418375849724),  # Thalamus
    8:  (0.0059,       -0.0359,       -0.078,     0.0,                    -0.078,     0.0011),                   # White matter
    9:  (0.039182053,  -0.019182053,  -0.0947979, 0.001402326044626534,   -0.0947979, 0.001402326044626534),   # Gray matter
    10: (0.027511641,  -0.008511641,  -0.0063624, 6.671431037830189e-05,  -0.0063624, 6.671431037830189e-05),  # CSF
    11: (0.19,          0.0,          -0.0582219, 4.701104626292363e-05,  -0.0582219, 4.701104626292363e-05),   # Blood — deoxyhemoglobin
    12: (0.038,        -0.019,         0.0,       0.0,                     0.0,       0.0),                      # Fat
    13: (-2.1,         -4.2,           1.2690799849224277e-05, 2.8707299293273536e-07, 1.2690799849224277e-05, 2.8707299293273536e-07),  # Bone
    14: (18.4,         -9.2,          -3.100334140526684e-07, 3.241496404626787e-09, -3.100334140526684e-07, 3.241496404626787e-09),     # Air
    15: (0.0,           0.0,          -1.5497382264584303e-05, 4.508940776304371e-07, -1.5497382264584303e-05, 4.508940776304371e-07),   # Muscle
    16: (0.0,           0.0,           0.0,       0.0,                     0.0,       0.0),                      # Calcification
}

# -----------------------------------------------------------------------------
# Susceptibility -> R2' relaxivity ("magnitude decay kernel", D_r).
#
# In the static-dephasing regime (Yablonskiy & Haacke, 1994), reversible
# transverse relaxation depends on the magnitude of the susceptibility-induced
# field perturbation, not its sign. A single spatially-invariant kernel then
# relates R2' to the absolute susceptibility of both source types:
#
#     R2' = D_r * (|chi_pos| + |chi_neg|)
#
# so paramagnetic (iron) and diamagnetic (myelin) sources share one relaxivity.
# DR_KERNEL = 137 Hz/ppm is the empirically calibrated value of Shin et al.
# (NeuroImage 2021); the COSMOS-referenced value of 114 Hz/ppm (chi-sepnet) is an
# equivalent single-kernel estimate that differs by the QSM algorithm used to
# calibrate it.
#
# generate_dr_maps_ridani() provides the alternative source-dependent,
# orientation-dependent relaxivities (D_r+ spherical, D_r- cylindrical sin^2(θ);
# Yablonskiy & Haacke 1994, as used by Ridani et al. 2026), which generate_r2prime
# and generate_bids accept as spatially-varying maps. A scalar split (distinct
# D_r+/D_r-) is also available via the ``dr_neg`` argument.
# -----------------------------------------------------------------------------
DR_KERNEL = 137.0  # Hz/ppm; single-kernel susceptibility->R2' relaxivity (Shin et al. 2021, NeuroImage 240:118371)

# Per-tissue T2 values at 7T in milliseconds.
# Values from Kumar et al. (2011, 2012) J Magn Reson Imaging, scaled to 7T.
# Used by generate_t2_map() for simulating R2 maps.
T2_TISSUE_PARAMS_7T = {
    1:  57.46,   # Caudate nucleus
    2:  41.47,   # Globus pallidus
    3:  50.44,   # Putamen
    4:  44.07,   # Red nucleus
    5:  71.71,   # Dentate nucleus
    6:  47.255,  # Substantia nigra
    7:  56.62,   # Thalamus
    8:  45.54,   # White matter
    9:  84.71,   # Gray matter
    10: 1029.6,  # CSF
    11: 97.5,    # Blood
}

# Per-tissue R1 scaling factors for 7T-to-3T conversion.
# R1_3T = R1_7T / factor for each tissue label.
# From NeuroPoly Susceptibility-Separation-Phantom Map_creation_3T.m.
R1_3T_DIVISION_FACTORS = {
    1:  0.75929,  # Caudate
    2:  0.73274,  # Globus pallidus
    3:  0.74212,  # Putamen
    4:  0.65,     # Red nucleus
    5:  0.65,     # Dentate nucleus
    6:  0.65,     # Substantia nigra
    7:  0.73898,  # Thalamus
    8:  0.72472,  # White matter
    9:  0.73648,  # Gray matter
    10: 1.0051,   # CSF
    11: 0.75672,  # Blood
}

# White matter susceptibility anisotropy parameters (per WM tract).
# For each WM-tract label in white_matter_mask.nii.gz: (delta_chi, chi_0) in ppm,
# where the apparent (orientation-dependent) diamagnetic susceptibility is
#   chi_neg_aniso = delta_chi * cos^2(theta) + chi_0
# and theta is the angle between the fibre orientation (V1) and B0.
#
# delta_chi (= chi_parallel - chi_perp) and chi_0 are the per-WM-tract values from
# the Susceptibility-Separation-Phantom (Ridani, De Leener & Alonso-Ortiz, 2026;
# MIT, (c) NeuroPoly 2024), Table 1 of that phantom's README, encoded numerically
# in PhantomCreation.m as:
#   deltaX_values = [0.032, 0.024, 0.014, 0.016, 0.016, 0.005, 0.008, 0.006, -0.015, -0.015, 0.0091]
#   Xzero_values  = [-0.0512,-0.0522,-0.0382,-0.0512,-0.0592,-0.0442,-0.0542,-0.0462,-0.0382,-0.0372,-0.0385]
# applied over white_matter_mask labels 1..11. WM_TRACT_ANISOTROPY_PARAMS below is
# a labelled view of these values; WM_TRACT_ANISOTROPY_ARRAYS is the index-keyed
# (label 1..11) form used by apply_wm_anisotropy, which applies them per-tract over
# the supplied white_matter_mask as in PhantomCreation.m.
WM_TRACT_ANISOTROPY_PARAMS = {
    'body_corpus_callosum':            (0.032, -0.0512),
    'splenium_corpus_callosum':        (0.024, -0.0522),
    'genu_corpus_callosum':            (0.014, -0.0382),
    'anterior_limb_internal_capsule':  (0.016, -0.0512),
    'posterior_thalamic_radiations':   (0.016, -0.0592),
    'superior_corona_radiata':         (0.005, -0.0442),
    'posterior_corona_radiata':        (0.008, -0.0542),
    'anterior_corona_radiata':         (0.006, -0.0462),
    'posterior_limb_internal_capsule': (-0.015, -0.0382),
    'superior_longitudinal_fascicle':  (-0.015, -0.0372),
    'tract_11':                        (0.0091, -0.0385),
}

# Index-keyed (white_matter_mask label 1..11 -> (delta_chi, chi_0)) view of the
# authoritative deltaX_values/Xzero_values arrays from PhantomCreation.m.
WM_TRACT_ANISOTROPY_ARRAYS = {
    1:  (0.032, -0.0512),
    2:  (0.024, -0.0522),
    3:  (0.014, -0.0382),
    4:  (0.016, -0.0512),
    5:  (0.016, -0.0592),
    6:  (0.005, -0.0442),
    7:  (0.008, -0.0542),
    8:  (0.006, -0.0462),
    9:  (-0.015, -0.0382),
    10: (-0.015, -0.0372),
    11: (0.0091, -0.0385),
}


def is_editable_package(package_name):
    """
    Determine if a package was installed in "editable" mode.
    
    :param package_name: The name of the package.
    :return: True if the package was installed in editable mode, False otherwise.
    """
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Look for the package's metadata directory
    for item in os.listdir(site_packages):
        if item.startswith(package_name) and item.endswith(".egg-link"):
            return True
        if item.startswith(package_name) and item.endswith(".dist-info"):
            dist_info_dir = os.path.join(site_packages, item)
            direct_url_path = os.path.join(dist_info_dir, "direct_url.json")
            
            # If direct_url.json exists, parse it and check for "editable"
            if os.path.exists(direct_url_path):
                with open(direct_url_path, 'r') as f:
                    data = json.load(f)
                    return data.get("editable", False)
    
    return False

def get_version():
    return f"{_get_version('qsm-forward')}" + (" (linked installation)" if is_editable_package('qsm-forward') else "")

class TissueParams:
    """
    A class used to represent tissue parameters.

    Attributes
    ----------
    root_dir : str or None
        The path to the root directory containing the tissue parameter files.
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
    chi_pos : str or ndarray or None
        Optional path or array for paramagnetic susceptibility (chi+, >= 0).
        If None, derived from chi as max(0, chi).
    chi_neg : str or ndarray or None
        Optional path or array for diamagnetic susceptibility (chi-, <= 0).
        If None, derived from chi as min(0, chi).
    """

    def __init__(
            self,
            root_dir = "",
            chi = "chimodel/ChiModelMIX.nii",
            M0 = "maps/M0.nii.gz",
            R1 = "maps/R1.nii.gz",
            R2star = "maps/R2star.nii.gz",
            mask = "masks/BrainMask.nii.gz",
            seg = "masks/SegmentedModel.nii.gz",
            chi_pos = None,
            chi_neg = None,
            voxel_size = None,
            apply_mask = False,
            v1 = "maps/V1.nii.gz",
            R2 = None,
            angle_map = None,
            raw_field = "raw/rawField.nii.gz",
            high_grad = "masks/highgrad.nii.gz",
            white_matter_mask = "masks/white_matter_mask.nii.gz",
            chisep_anisotropy = False,
            chisep_apply_brain_mask = True,
    ):
        if isinstance(chi, str) and not os.path.exists(os.path.join(root_dir, chi)):
            raise ValueError(f"Path to chi is invalid! ({os.path.join(root_dir, chi)})")
        self._chi = os.path.join(root_dir, chi) if isinstance(chi, str) and os.path.exists(os.path.join(root_dir, chi)) else chi if not isinstance(chi, str) else None
        self._M0 = os.path.join(root_dir, M0) if isinstance(M0, str) and os.path.exists(os.path.join(root_dir, M0)) else M0 if not isinstance(M0, str) else None
        self._R1 = os.path.join(root_dir, R1) if isinstance(R1, str) and os.path.exists(os.path.join(root_dir, R1)) else R1 if not isinstance(R1, str) else None
        self._R2star = os.path.join(root_dir, R2star) if isinstance(R2star, str) and os.path.exists(os.path.join(root_dir, R2star)) else R2star if not isinstance(R2star, str) else None
        self._mask = os.path.join(root_dir, mask) if isinstance(mask, str) and os.path.exists(os.path.join(root_dir, mask)) else mask if not isinstance(mask, str) else None
        self._seg = os.path.join(root_dir, seg) if isinstance(seg, str) and os.path.exists(os.path.join(root_dir, seg)) else seg if not isinstance(seg, str) else None
        self._chi_pos = os.path.join(root_dir, chi_pos) if isinstance(chi_pos, str) and os.path.exists(os.path.join(root_dir, chi_pos)) else chi_pos if not isinstance(chi_pos, str) else None
        self._chi_neg = os.path.join(root_dir, chi_neg) if isinstance(chi_neg, str) and os.path.exists(os.path.join(root_dir, chi_neg)) else chi_neg if not isinstance(chi_neg, str) else None
        self._v1 = os.path.join(root_dir, v1) if isinstance(v1, str) and os.path.exists(os.path.join(root_dir, v1)) else v1 if not isinstance(v1, str) else None
        self._R2 = os.path.join(root_dir, R2) if isinstance(R2, str) and os.path.exists(os.path.join(root_dir, R2)) else R2 if not isinstance(R2, str) else None
        self._angle_map = os.path.join(root_dir, angle_map) if isinstance(angle_map, str) and os.path.exists(os.path.join(root_dir, angle_map)) else angle_map if not isinstance(angle_map, str) else None
        # Extra maps for the chi-sep build. String paths are
        # resolved relative to root_dir if they exist; otherwise kept as the raw
        # value (arrays) or set to None (missing optional path). Missing required
        # maps are reported by _compute_chisep_maps, not silently defaulted.
        self._raw_field = os.path.join(root_dir, raw_field) if isinstance(raw_field, str) and os.path.exists(os.path.join(root_dir, raw_field)) else raw_field if not isinstance(raw_field, str) else None
        self._high_grad = os.path.join(root_dir, high_grad) if isinstance(high_grad, str) and os.path.exists(os.path.join(root_dir, high_grad)) else high_grad if not isinstance(high_grad, str) else None
        self._white_matter_mask = os.path.join(root_dir, white_matter_mask) if isinstance(white_matter_mask, str) and os.path.exists(os.path.join(root_dir, white_matter_mask)) else white_matter_mask if not isinstance(white_matter_mask, str) else None
        self._chisep_anisotropy = chisep_anisotropy
        self._chisep_apply_brain_mask = chisep_apply_brain_mask
        self._apply_mask = apply_mask
        self._voxel_size = voxel_size
        self._affine = None

    def set_affine(self, affine):
        self._affine = affine

    def _load(self, nii_path):
        nii = nib.load(nii_path)
        if self._affine is not None:
            nii = nib.Nifti1Image(dataobj=nii.get_fdata(), affine=self._affine, header=nii.header)
        return nii

    @property
    def voxel_size(self):
        if self._voxel_size is not None:
            return self._voxel_size
        zooms = self.nii_header.get_zooms()
        return zooms if len(zooms) == 3 else np.array([zooms[0] for i in range(3)])

    @property
    def nii_header(self):
        if isinstance(self._chi, str):
            return self._load(self._chi).header
        header = nib.Nifti1Header()
        header.set_data_shape(self._chi.shape)
        return header
    
    @property
    def nii_affine(self):
        if self._affine is not None:
            return self._affine
        if isinstance(self._chi, str):
            return self._load(self._chi).affine
        return np.eye(4)

    def _do_apply_mask(self, nii): return nib.Nifti1Image(dataobj=nii.get_fdata() * self.mask.get_fdata(), affine=self.nii_affine, header=nii.header) if self._apply_mask else nii

    @property
    def chi(self): return self._do_apply_mask(self._load(self._chi) if isinstance(self._chi, str) else nib.Nifti1Image(self._chi, affine=self.nii_affine, header=self.nii_header))

    @property
    def mask(self): return self._load(self._mask) if isinstance(self._mask, str) else nib.Nifti1Image(self._mask or np.array(self._chi != 0), affine=self.nii_affine, header=self.nii_header)

    @property
    def M0(self): return self._do_apply_mask(self._load(self._M0) if isinstance(self._M0, str) else nib.Nifti1Image(self._M0 or np.array(self.mask.get_fdata() * 1), affine=self.nii_affine, header=self.nii_header))

    @property
    def R1(self): return self._do_apply_mask(self._load(self._R1) if isinstance(self._R1, str) else nib.Nifti1Image(self._R1 or np.array(self.mask.get_fdata() * 1), affine=self.nii_affine, header=self.nii_header))
    
    @property
    def R2star(self): return self._do_apply_mask(self._load(self._R2star) if isinstance(self._R2star, str) else nib.Nifti1Image(self._R2star or np.array(self.mask.get_fdata() * 50), affine=self.nii_affine, header=self.nii_header))
    
    @property
    def seg(self): return self._load(self._seg) if isinstance(self._seg, str) else nib.Nifti1Image(self._seg or self.mask.get_fdata(), affine=self.nii_affine, header=self.nii_header)

    @property
    def v1(self):
        if self._v1 is None:
            return None
        return self._load(self._v1) if isinstance(self._v1, str) else nib.Nifti1Image(self._v1, affine=self.nii_affine, header=self.nii_header)

    @property
    def R2(self):
        if self._R2 is None:
            return None
        return self._do_apply_mask(self._load(self._R2) if isinstance(self._R2, str) else nib.Nifti1Image(self._R2, affine=self.nii_affine, header=self.nii_header))

    @property
    def angle_map(self):
        if self._angle_map is None:
            return None
        return self._load(self._angle_map) if isinstance(self._angle_map, str) else nib.Nifti1Image(self._angle_map, affine=self.nii_affine, header=self.nii_header)

    @property
    def raw_field(self):
        if self._raw_field is None:
            return None
        return self._load(self._raw_field) if isinstance(self._raw_field, str) else nib.Nifti1Image(self._raw_field, affine=self.nii_affine, header=self.nii_header)

    @property
    def high_grad(self):
        if self._high_grad is None:
            return None
        return self._load(self._high_grad) if isinstance(self._high_grad, str) else nib.Nifti1Image(self._high_grad, affine=self.nii_affine, header=self.nii_header)

    @property
    def white_matter_mask(self):
        if self._white_matter_mask is None:
            return None
        return self._load(self._white_matter_mask) if isinstance(self._white_matter_mask, str) else nib.Nifti1Image(self._white_matter_mask, affine=self.nii_affine, header=self.nii_header)

    def _compute_chisep_maps(self):
        """Build and cache chi+ / chi- maps via generate_chisep_maps.

        Requires the R1, R2*, raw field and high-gradient maps (and, for
        anisotropy, V1 + white_matter_mask). Missing required maps raise a clear
        error naming the file — there is no silent fallback.
        """
        missing = []
        if self.raw_field is None:
            missing.append("raw_field (masks/raw/rawField.nii.gz)")
        if self.high_grad is None:
            missing.append("high_grad (masks/highgrad.nii.gz)")
        anisotropy = bool(self._chisep_anisotropy)
        if anisotropy:
            if self.v1 is None:
                missing.append("v1 (maps/V1.nii.gz)")
            if self.white_matter_mask is None:
                missing.append("white_matter_mask (masks/white_matter_mask.nii.gz)")
        if missing:
            raise FileNotFoundError(
                "generate_chisep_maps needs these maps, "
                "which are missing: " + ", ".join(missing)
            )

        seg_data = self.seg.get_fdata()
        mask_data = self.mask.get_fdata()
        R1_data = self.R1.get_fdata()
        R2star_data = self.R2star.get_fdata()
        raw_field_data = self.raw_field.get_fdata()
        high_grad_data = self.high_grad.get_fdata()
        v1_data = self.v1.get_fdata() if (anisotropy and self.v1 is not None) else None
        wm_tract_data = self.white_matter_mask.get_fdata() if (anisotropy and self.white_matter_mask is not None) else None

        cp, cn = generate_chisep_maps(
            seg=seg_data, mask=mask_data, R1=R1_data, R2star=R2star_data,
            raw_field=raw_field_data, high_grad=high_grad_data,
            voxel_size=self.voxel_size,
            v1_map=v1_data, wm_tract_mask=wm_tract_data, anisotropy=anisotropy,
            apply_brain_mask=bool(self._chisep_apply_brain_mask),
        )
        self._cached_chi_pos = nib.Nifti1Image(cp, affine=self.nii_affine, header=self.nii_header)
        self._cached_chi_neg = nib.Nifti1Image(cn, affine=self.nii_affine, header=self.nii_header)

    @property
    def chi_pos(self):
        if self._chi_pos is not None:
            nii = self._do_apply_mask(self._load(self._chi_pos) if isinstance(self._chi_pos, str) else nib.Nifti1Image(self._chi_pos, affine=self.nii_affine, header=self.nii_header))
            return nii
        # Use tissue-informed splitting (cached)
        if not hasattr(self, '_cached_chi_pos'):
            self._compute_chisep_maps()
        return self._cached_chi_pos

    @property
    def chi_neg(self):
        if self._chi_neg is not None:
            nii = self._do_apply_mask(self._load(self._chi_neg) if isinstance(self._chi_neg, str) else nib.Nifti1Image(self._chi_neg, affine=self.nii_affine, header=self.nii_header))
            return nii
        # Use tissue-informed splitting (cached)
        if not hasattr(self, '_cached_chi_neg'):
            self._compute_chisep_maps()
        return self._cached_chi_neg


class ReconParams:
    """
    A class used to represent reconstruction parameters.

    Attributes
    ----------
    subject : str
        The ID of the subject.
    session : str
        The ID of the session.
    acq : str
        The acquisition name.
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
    suffix : string
        The BIDS-compliant suffix that defines the weighting of the images (e.g. T1w, T2starw, PD).
    save_phase : bool
        Boolean to control whether phase images are saved.
    se_TR : float
        Repetition time (in seconds) for the spin-echo acquisition. Default is 1.0.
    se_TEs : np.array
        Echo times (in seconds) for the spin-echo acquisition. Default is a
        4-echo train spanning brain T2 (10-70 ms).
    """

    def __init__(
            self,
            subject="1",
            session=None,
            acq=None,
            run=None,
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
            random_seed=None,
            suffix=None,
            save_phase=True,
            se_TR=1.0,
            se_TEs=np.array([ 10e-3, 30e-3, 50e-3, 70e-3 ])
        ):
        self.subject = subject
        self.session = session
        self.acq = acq
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
        self.save_phase = save_phase
        self.se_TR = se_TR
        self.se_TEs = np.asarray(se_TEs)
        self.suffix = suffix
        if suffix is None:
            self.suffix = "MEGRE" if len(TEs) > 1 else "T2starw"

def rotation_matrix_from_vectors(vec1, vec2):
    """ Compute the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    # Check if vectors are nearly parallel
    if np.isclose(s, 0):
        return np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def adjust_affine_for_B0_direction(affine, B0_dir):
    B0_dir_normalized = B0_dir / np.linalg.norm(B0_dir)
    rotation_matrix = np.linalg.inv(rotation_matrix_from_vectors([0, 0, 1], B0_dir_normalized))
    return affine.dot(np.vstack([np.column_stack([rotation_matrix, [0, 0, 0]]), [0, 0, 0, 1]]))

def generate_bids(tissue_params: TissueParams, recon_params: ReconParams, bids_dir, save_chi=True, save_mask=True, save_segmentation=True, save_field=False, save_shimmed_field=False, save_shimmed_offset_field=False, save_chi_pos=False, save_chi_neg=False, save_r2prime=False, dr=DR_KERNEL, dr_neg=None, dr_pos_map=None, dr_neg_map=None, chisep_signal=False, chisep_multicompartment=False, anisotropy=False, save_r2=False, save_dr_pos=False, save_dr_neg=False, save_t2=False, save_se=False):
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
    save_chi_pos : bool
        Whether to save the paramagnetic susceptibility map (chi+). Default is False.
    save_chi_neg : bool
        Whether to save the diamagnetic susceptibility map (chi-). Default is False.
    save_r2prime : bool
        Whether to save the R2' map computed from chi+ and chi-. Default is False.
    dr : float
        Magnitude decay kernel in Hz/ppm relating |chi| to R2' (single kernel for
        both source types; see the DR_KERNEL note). Default is DR_KERNEL (137.0).
    dr_neg : float or None
        Optional separate diamagnetic relaxivity in Hz/ppm. None (default) uses the
        single kernel ``dr`` for both source types; pass a value to opt into a
        non-standard split model for the R2' map AND the chi-sep signal.
    dr_pos_map, dr_neg_map : numpy.ndarray or None
        Optional caller-supplied spatially-varying paramagnetic/diamagnetic relaxivity
        maps (Hz/ppm), e.g. an anisotropic (orientation-dependent) split Dr. When given
        they take precedence over the scalar ``dr``/``dr_neg`` and over the built-in
        ``anisotropy`` path, and drive both the shipped R2' map and the GRE signal decay
        so the two are consistent. Either may be None to keep the scalar kernel for that
        source type.
    save_se : bool
        Whether to save a simulated multi-echo spin-echo (SE) acquisition, whose
        magnitude decays with R2 (not R2*). Enables deriving R2' = R2* - R2 from
        an SE/GRE pair. Uses recon_params.se_TR and recon_params.se_TEs. Default
        is False.

    Returns
    -------
    None
        Outputs are saved as files in the bids_dir directory.

    """

    # create output directories
    print("Creating output directory...")
    os.makedirs(bids_dir, exist_ok=True)
    
    # recon name
    recon_name = f"sub-{recon_params.subject}"
    if recon_params.session: recon_name += f"_ses-{recon_params.session}"
    if recon_params.acq: recon_name += f"_acq-{recon_params.acq}"
    if recon_params.run: recon_name += f"_run-{recon_params.run}"

    # subject directory
    subject_dir = os.path.join(bids_dir, f"sub-{recon_params.subject}")
    if recon_params.session: subject_dir = os.path.join(subject_dir, f"ses-{recon_params.session}")

    # derivatives directory
    subject_dir_deriv = os.path.join(bids_dir, "derivatives", "qsm-forward", f"sub-{recon_params.subject}")
    if recon_params.session: subject_dir_deriv = os.path.join(subject_dir_deriv, f"ses-{recon_params.session}")

    os.makedirs(os.path.join(subject_dir, 'anat'), exist_ok=True)
    os.makedirs(os.path.join(subject_dir_deriv, 'anat'), exist_ok=True)

    # random number generator for noise etc.
    rng = np.random.default_rng(recon_params.random_seed)

    # adjust affine for B0 direction
    affine = adjust_affine_for_B0_direction(tissue_params.nii_affine.copy(), recon_params.B0_dir)
    tissue_params.set_affine(affine)

    # image-space resizing
    print("Image-space resizing of chi...")
    chi_downsampled_nii = resize(tissue_params.chi, recon_params.voxel_size)
    if save_chi: nib.save(chi_downsampled_nii, filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_Chimap.nii"))
    print("Image-space cropping of mask...")
    if save_mask:
        nib.save(resize(tissue_params.mask, recon_params.voxel_size, 'nearest'), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_mask.nii"))
    print("Image-space cropping of segmentation...")
    if save_segmentation: nib.save(resize(tissue_params.seg, recon_params.voxel_size, 'nearest'), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_dseg.nii"))

    # chi-separation derivatives
    if save_chi_pos or save_chi_neg or save_r2prime:
        print("Computing chi-separation maps...")
        chi_pos_nii = tissue_params.chi_pos
        chi_neg_nii = tissue_params.chi_neg
        if save_chi_pos:
            print("Image-space resizing of chi+...")
            nib.save(resize(chi_pos_nii, recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_Chimap-pos.nii"))
        if save_chi_neg:
            print("Image-space resizing of chi-...")
            chi_neg_abs_nii = nib.Nifti1Image(dataobj=np.abs(chi_neg_nii.get_fdata()), affine=chi_neg_nii.affine, header=chi_neg_nii.header)
            nib.save(resize(chi_neg_abs_nii, recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_Chimap-neg.nii"))
        if save_r2prime:
            print("Computing R2' from chi+ and chi-...")
            # Provided spatially-varying Dr maps (e.g. anisotropic split relaxivity) take precedence
            # over the scalar dr/dr_neg, and are used here AND in the signal below so the shipped R2'
            # and the GRE decay stay consistent.
            rp_pos = dr if dr_pos_map is None else dr_pos_map
            rp_neg = (dr if dr_neg is None else dr_neg) if dr_neg_map is None else dr_neg_map
            r2prime_data = generate_r2prime(chi_pos_nii.get_fdata(), chi_neg_nii.get_fdata(), dr=rp_pos, dr_neg=rp_neg)
            r2prime_nii = nib.Nifti1Image(dataobj=r2prime_data.astype(np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header)
            print("Image-space resizing of R2'...")
            nib.save(resize(r2prime_nii, recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_R2prime.nii"))

    # calculate field
    print("Computing field model...")
    field = generate_field(tissue_params.chi.get_fdata(), tissue_params.mask.get_fdata(),voxel_size=tissue_params.voxel_size, B0_dir=recon_params.B0_dir)
    if save_field:
        nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_fieldmap.nii"))
        local_field = generate_field(tissue_params.chi.get_fdata() * tissue_params.mask.get_fdata(), tissue_params.mask.get_fdata(), voxel_size=tissue_params.voxel_size, B0_dir=recon_params.B0_dir)
        nib.save(resize(nib.Nifti1Image(dataobj=np.array(local_field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_fieldmap-local.nii"))

    # simulate shim field
    if recon_params.generate_shim_field:
        print("Computing shim fields...")
        _, field, _ = generate_shimmed_field(field, tissue_params.mask.get_fdata(), order=2)
        if save_shimmed_field: nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_desc-shimmed_fieldmap.nii"))

    # phase offset
    phase_offset = recon_params.phase_offset
    if recon_params.generate_phase_offset:
        print("Computing phase offset...")
        phase_offset = recon_params.phase_offset + generate_phase_offset(tissue_params.M0.get_fdata(), tissue_params.mask.get_fdata(), tissue_params.M0.get_fdata().shape)
        if save_shimmed_offset_field: nib.save(resize(nib.Nifti1Image(dataobj=np.array(field, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_desc-shimmed-offset_fieldmap.nii"))

    # The multi-compartment GRE model needs the chi-sep maps (chi+/chi-, R2, Dr), so it implies chisep_signal.
    if chisep_multicompartment:
        chisep_signal = True

    # transverse relaxation (R2) — shared by the chi-sep GRE model and the SE signal
    R2_data = None
    dr_pos_data = None
    dr_neg_data = None
    chi_pos_data = None
    chi_neg_data = None

    if chisep_signal or save_se:
        print("Computing transverse relaxation (R2)...")
        # Compute or load R2
        if tissue_params.R2 is not None:
            R2_data = tissue_params.R2.get_fdata()
            print("  Using pre-loaded R2 map")
        else:
            print("  Computing T2/R2 maps from tissue segmentation...")
            T2_data, R2_data = generate_t2_map(
                seg=tissue_params.seg.get_fdata(),
                R2star=tissue_params.R2star.get_fdata(),
                M0=tissue_params.M0.get_fdata(),
                B0=recon_params.B0
            )
            if save_t2:
                nib.save(resize(nib.Nifti1Image(dataobj=T2_data.astype(np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_T2map.nii"))
        if save_r2:
            nib.save(resize(nib.Nifti1Image(dataobj=R2_data.astype(np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_R2map.nii"))

    if chisep_signal:
        print("Setting up chi-sep-aware GRE signal model...")
        # Get chi+/chi-
        chi_pos_data = tissue_params.chi_pos.get_fdata()
        chi_neg_data = tissue_params.chi_neg.get_fdata()

        # Relaxivity (magnitude decay kernel). Default: single scalar kernel
        # applied voxel-wise, which makes the GRE R2' contribution
        # (dr_pos*|chi+| + dr_neg*|chi-|) IDENTICAL to generate_r2prime and, with
        # the SE R2 decay, to the derivable R2' = R2* - R2. anisotropy is an
        # opt-in that swaps in a spatially-varying orientation-dependent map.
        if dr_pos_map is not None or dr_neg_map is not None:
            # Caller-supplied spatially-varying split relaxivity (same maps used for the shipped R2').
            print("  Using provided spatially-varying Dr maps...")
            dr_pos_data = dr if dr_pos_map is None else dr_pos_map
            dr_neg_data = (dr if dr_neg is None else dr_neg) if dr_neg_map is None else dr_neg_map
        elif anisotropy and tissue_params.angle_map is not None:
            print("  Computing orientation-dependent Dr maps (anisotropy=True)...")
            dr_pos_data, dr_neg_data = generate_dr_maps(
                seg=tissue_params.seg.get_fdata(),
                B0=recon_params.B0,
                angle_map=tissue_params.angle_map.get_fdata(),
                anisotropy=True,
                dr=dr
            )
        else:
            dr_pos_data = dr
            dr_neg_data = dr if dr_neg is None else dr_neg

        if save_dr_pos or save_dr_neg:
            # Broadcast scalar kernels to masked constant maps for saving
            mask_data = tissue_params.mask.get_fdata()
            dr_pos_save = dr_pos_data if np.ndim(dr_pos_data) else dr_pos_data * mask_data
            dr_neg_save = dr_neg_data if np.ndim(dr_neg_data) else dr_neg_data * mask_data
            if save_dr_pos:
                nib.save(resize(nib.Nifti1Image(dataobj=np.asarray(dr_pos_save, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_Dr-pos.nii"))
            if save_dr_neg:
                nib.save(resize(nib.Nifti1Image(dataobj=np.asarray(dr_neg_save, dtype=np.float32), affine=tissue_params.nii_affine, header=tissue_params.nii_header), recon_params.voxel_size), filename=os.path.join(subject_dir_deriv, "anat", f"{recon_name}_Dr-neg.nii"))

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
            M0=tissue_params.M0.get_fdata(),
            R2=R2_data,
            dr_pos=dr_pos_data,
            dr_neg=dr_neg_data,
            chi_pos=chi_pos_data,
            chi_neg=chi_neg_data,
            multicompartment=chisep_multicompartment,
        )
    
        # k-space cropping of sigHR
        print(f"k-space cropping of MR signal for echo {i+1}...")
        resolution = np.array(np.round((np.array(tissue_params.voxel_size) / recon_params.voxel_size) * np.array(tissue_params.nii_header.get_data_shape())), dtype=int)
        sigHR_cropped = crop_kspace(sigHR, resolution)
        del sigHR

        # noise
        if recon_params.random_seed is not None:
            print(f"Simulating noise for echo {i+1} with random seed {recon_params.random_seed}...")
            sigHR_cropped_noisy = add_noise(sigHR_cropped, peak_snr=recon_params.peak_snr, rng=rng)
        else:
            sigHR_cropped_noisy = np.array(sigHR_cropped)
        del sigHR_cropped

        # save nifti images
        mag_filename = f"{recon_name_i}" + ("_part-mag" if recon_params.save_phase else "") + f"_{recon_params.suffix}"
        phs_filename = f"{recon_name_i}" + ("_part-phase" if recon_params.save_phase else "") + f"_{recon_params.suffix}"
        description = f"TE={recon_params.TEs[i]}; TR={recon_params.TR}; FlipAngle={recon_params.flip_angle}; B0={recon_params.B0}; B0_dir={recon_params.B0_dir}"
        mag_nii = nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header)
        mag_nii.header['descrip'] = description
        nib.save(mag_nii, filename=os.path.join(subject_dir, "anat", f"{mag_filename}.nii"))
        if recon_params.save_phase:
            phs_nii = nib.Nifti1Image(dataobj=np.angle(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header)
            phs_nii.header['descrip'] = description
            nib.save(phs_nii, filename=os.path.join(subject_dir, "anat", f"{phs_filename}.nii"))

        # json header
        print(f"Creating JSON headers...")
        json_dict = {
            'Subject': recon_params.subject,
            'Session': recon_params.session,
            'Acquisition': recon_params.acq,
            'Run': recon_params.run,
            'EchoTime': recon_params.TEs[i],
            'MagneticFieldStrength': recon_params.B0,
            'EchoNumber': i+1,
            'ProtocolName': recon_params.suffix,
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

        with open(os.path.join(subject_dir, "anat", f"{mag_filename}.json"), 'w') as mag_json_file:
            json.dump(json_dict_mag, mag_json_file)
        if recon_params.save_phase:
            with open(os.path.join(subject_dir, "anat", f"{phs_filename}.json"), 'w') as phs_json_file:
                json.dump(json_dict_phs, phs_json_file)

    # spin-echo signal (magnitude only; decays with R2, not R2*, and carries no susceptibility phase)
    if save_se:
        se_TEs = np.asarray(recon_params.se_TEs)
        se_TR = recon_params.se_TR
        multiecho_se = len(se_TEs) > 1
        se_suffix = "MESE" if multiecho_se else "T2w"
        for i in range(len(se_TEs)):
            print(f"Computing SE signal for echo {i+1}...")
            recon_name_i = f"{recon_name}_echo-{i+1}" if multiecho_se else recon_name

            # cast to complex (zero phase) so k-space cropping takes the fast
            # complex path — matching the GRE pipeline, which skips the expensive
            # Gibbs-unring step that a real-valued input would otherwise trigger
            sigHR = generate_se_signal(
                TR=se_TR,
                TE=se_TEs[i],
                R1=tissue_params.R1.get_fdata(),
                R2=R2_data,
                M0=tissue_params.M0.get_fdata(),
            ).astype(np.complex128)

            # k-space cropping of sigHR
            print(f"k-space cropping of SE signal for echo {i+1}...")
            resolution = np.array(np.round((np.array(tissue_params.voxel_size) / recon_params.voxel_size) * np.array(tissue_params.nii_header.get_data_shape())), dtype=int)
            sigHR_cropped = crop_kspace(sigHR, resolution)
            del sigHR

            # noise (complex noise -> Rician magnitude, matching the GRE path)
            if recon_params.random_seed is not None:
                print(f"Simulating noise for SE echo {i+1} with random seed {recon_params.random_seed}...")
                sigHR_cropped_noisy = add_noise(sigHR_cropped, peak_snr=recon_params.peak_snr, rng=rng)
            else:
                sigHR_cropped_noisy = np.array(sigHR_cropped)
            del sigHR_cropped

            # save nifti (magnitude only; SE phase is refocused and not informative)
            mag_filename = f"{recon_name_i}" + ("_part-mag" if recon_params.save_phase else "") + f"_{se_suffix}"
            description = f"TE={se_TEs[i]}; TR={se_TR}; B0={recon_params.B0}; B0_dir={recon_params.B0_dir}; acq=spin-echo"
            mag_nii = nib.Nifti1Image(dataobj=np.abs(sigHR_cropped_noisy), affine=chi_downsampled_nii.affine, header=chi_downsampled_nii.header)
            mag_nii.header['descrip'] = description
            nib.save(mag_nii, filename=os.path.join(subject_dir, "anat", f"{mag_filename}.nii"))
            del sigHR_cropped_noisy

            # json header
            print(f"Creating SE JSON header for echo {i+1}...")
            json_dict = {
                'Subject': recon_params.subject,
                'Session': recon_params.session,
                'Acquisition': recon_params.acq,
                'Run': recon_params.run,
                'EchoTime': se_TEs[i],
                'MagneticFieldStrength': recon_params.B0,
                'EchoNumber': i+1,
                'ProtocolName': se_suffix,
                'ConversionSoftware': 'qsm-forward',
                'RepetitionTime': se_TR,
                'B0_dir': recon_params.B0_dir.tolist(),
                'VoxelSize': recon_params.voxel_size.tolist(),
                'PeakSNR': recon_params.peak_snr if recon_params.peak_snr != np.inf else "inf",
                'PulseSequenceType': 'Spin Echo',
                'ImageType': ['M', 'MAGNITUDE'],
            }
            with open(os.path.join(subject_dir, "anat", f"{mag_filename}.json"), 'w') as se_json_file:
                json.dump(json_dict, se_json_file)

    print(f"Generating details for BIDS datset_description.json...")
    dataset_description = {
        "Name" : f"qsm-forward BIDS ({datetime.date.today()})",
        "BIDSVersion" : "1.9.0",
        "GeneratedBy" : [{
            "Name" : "qsm-forward",
            "Version": f"{get_version()}",
            "CodeURL" : "https://github.com/astewartau/qsm-forward"
        }],
        "Authors" : ["ADD AUTHORS HERE"]
    }
    print(f"Writing BIDS dataset_description.json...")
    with open(os.path.join(bids_dir, 'dataset_description.json'), 'w', encoding='utf-8') as dataset_json_file:
        json.dump(dataset_description, dataset_json_file)
    with open(os.path.join(bids_dir, 'derivatives', 'qsm-forward', 'dataset_description.json'), 'w', encoding='utf-8') as dataset_json_file:
        json.dump(dataset_description, dataset_json_file)

    print(f"Writing BIDS .bidsignore file...")
    with open(os.path.join(bids_dir, '.bidsignore'), 'w', encoding='utf-8') as bidsignore_file:
        bidsignore_file.write('')

    print(f"Writing BIDS dataset README...")
    with open(os.path.join(bids_dir, 'README'), 'w', encoding='utf-8') as readme_file:
        readme_file.write(f"Generated using qsm-forward ({get_version()})\n")
        readme_file.write(f"\nDescribe your dataset here.\n")
    
    print("Done!")


def generate_field(chi, mask=None, voxel_size=[1, 1, 1], B0_dir=[0, 0, 1]):
    """
    Perform the forward convolution operation.

    This function performs the forward convolution step of the QSM simulation.

    Parameters
    ----------
    chi : numpy.ndarray
        The susceptibility distribution array.
    mask : numpy.ndarray
        A binary mask that indicates the internal region of interest.
    voxel_size : list, optional
        The voxel size. Default is [1, 1, 1].
    B0_dir : list, optional
        The B0 direction. Default is [0, 0, 1].

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
    if mask is not None:
        field = field - np.mean(field[mask != 0])

    return field

def generate_r2prime(chi_pos, chi_neg, dr=DR_KERNEL, dr_neg=None):
    """
    Compute the R2' map from paramagnetic and diamagnetic susceptibility components.

    R2' is the susceptibility-related (reversible) contribution to transverse
    relaxation used in chi-separation (Shin et al., 2021). By default a SINGLE
    magnitude decay kernel is applied to both source types (the standard
    static-dephasing model):

        R2' = dr * (|chi_pos| + |chi_neg|)

    A split model is available as an explicit opt-in by passing ``dr_neg`` (this
    is non-standard -- see the DR_KERNEL note above):

        R2' = dr * |chi_pos| + dr_neg * |chi_neg|

    Parameters
    ----------
    chi_pos : numpy.ndarray
        Paramagnetic susceptibility (>= 0) in ppm.
    chi_neg : numpy.ndarray
        Diamagnetic susceptibility (<= 0) in ppm.
    dr : float, optional
        Magnitude decay kernel in Hz/ppm applied to |chi_pos| (and to |chi_neg|
        too, unless ``dr_neg`` is given). Default is DR_KERNEL (137.0).
    dr_neg : float or None, optional
        Optional separate diamagnetic relaxivity in Hz/ppm. If None (default) the
        single kernel ``dr`` is used for both source types. Pass a value to opt
        into a non-standard split model.

    Returns
    -------
    numpy.ndarray
        R2' map in Hz.
    """
    if dr_neg is None:
        dr_neg = dr
    return dr * np.abs(chi_pos) + dr_neg * np.abs(chi_neg)


def generate_t2_map(seg, R2star, M0, B0=7, t2_params=None, gaussian_sigma=0.2):
    """
    Simulate a T2 map from tissue segmentation and compute R2 = 1/T2.

    Assigns per-tissue T2 values from the literature and modulates them with
    R2* and M0 maps for realistic intra-tissue texture, following the approach
    of the NeuroPoly Susceptibility-Separation-Phantom (T2_simulation.m).

    Parameters
    ----------
    seg : numpy.ndarray
        3D tissue segmentation labels (integer-valued).
    R2star : numpy.ndarray
        3D R2* map in Hz (1/T2*).
    M0 : numpy.ndarray
        3D proton density / net magnetization map.
    B0 : float, optional
        Magnetic field strength in Tesla. Default is 7.
    t2_params : dict or None, optional
        Per-tissue T2 values in ms, keyed by label. Default uses T2_TISSUE_PARAMS_7T.
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of the T2 map. Default is 0.2.

    Returns
    -------
    T2 : numpy.ndarray
        T2 map in milliseconds.
    R2 : numpy.ndarray
        R2 map in Hz (1/T2).
    """
    if t2_params is None:
        t2_params = T2_TISSUE_PARAMS_7T

    # Assign per-tissue T2 values
    T2_uniform = np.zeros_like(seg, dtype=np.float64)
    for label, t2_val in t2_params.items():
        T2_uniform[seg == label] = t2_val

    # Compute intra-tissue texture modulation from R2* and M0
    R2star = np.array(R2star, dtype=np.float64)
    M0 = np.array(M0, dtype=np.float64)
    R2star[np.isnan(R2star)] = 0
    M0[np.isnan(M0)] = 0

    pct_r2star = np.zeros_like(seg, dtype=np.float64)
    pct_m0 = np.zeros_like(seg, dtype=np.float64)
    for label in t2_params:
        region_mask = (seg == label)
        if not region_mask.any():
            continue
        mean_r2star = np.mean(R2star[region_mask])
        mean_m0 = np.mean(M0[region_mask])
        if mean_r2star > 0:
            pct_r2star[region_mask] = R2star[region_mask] / mean_r2star
        if mean_m0 > 0:
            pct_m0[region_mask] = M0[region_mask] / mean_m0

    pct_combined = (pct_r2star + pct_m0) / 2.0
    T2_textured = T2_uniform * pct_combined

    # Gaussian smooth
    T2_smoothed = gaussian_filter(T2_textured, sigma=gaussian_sigma)
    T2_smoothed = np.maximum(T2_smoothed, 0)  # Clamp: smoothing can produce negatives at boundaries

    # Scale for 3T if needed
    if B0 == 3:
        T2_smoothed = T2_smoothed / 0.65

    # Compute R2 = 1000 / T2 (ms to Hz), only where T2 > 0
    R2 = np.zeros_like(T2_smoothed)
    valid = T2_smoothed > 0
    R2[valid] = 1000.0 / T2_smoothed[valid]
    R2[R2 > 300] = 0

    # Scale R2 for 3T
    if B0 == 3:
        R2 = R2 * 0.65

    return T2_smoothed, R2


def generate_dr_maps(seg, B0=7, angle_map=None, anisotropy=False, dr=DR_KERNEL):
    """
    Generate spatially-varying Dr (magnitude decay kernel) maps for the
    paramagnetic and diamagnetic susceptibility components.

    Both source types share the single empirical kernel ``dr`` (see the DR_KERNEL
    note above): paramagnetic sources are assigned ``dr`` in the deep/cortical
    grey-matter regions, diamagnetic sources ``dr`` in white matter. This is only
    needed for the anisotropy opt-in; the default (isotropic) chi-separation
    signal path applies the single scalar kernel voxel-wise instead (see
    generate_bids), which is exactly consistent with generate_r2prime.

    NOTE: this function is an EXPERIMENTAL opt-in used only when anisotropy=True.
    The original NeuroPoly translation used a theoretical static-dephasing sphere
    kernel (~755 Hz/ppm at 7T); it has been rebased onto the empirical single
    kernel ``dr`` so the whole pipeline shares one relaxivity scale. When
    anisotropy=True the diamagnetic (white-matter) kernel is modulated by
    sin^2(theta) of the fibre-to-field angle (maximal perpendicular to B0, zero
    parallel); the amplitude is the empirical kernel rather than the theoretical
    one, so absolute anisotropic values differ from the NeuroPoly original.

    Parameters
    ----------
    seg : numpy.ndarray
        3D tissue segmentation labels.
    B0 : float, optional
        Magnetic field strength in Tesla. Retained for API compatibility; the
        empirical kernel is applied field-agnostically (as chi-sep methods do),
        so B0 does not scale the amplitude. Default is 7.
    angle_map : numpy.ndarray or None, optional
        3D map of fiber-to-field angle in degrees. Required when anisotropy=True.
    anisotropy : bool, optional
        If True, use orientation-dependent Dr_neg for white matter. Default is False.
    dr : float, optional
        Magnitude decay kernel in Hz/ppm. Default is DR_KERNEL (137.0).

    Returns
    -------
    dr_pos : numpy.ndarray
        Paramagnetic relaxivity map in Hz/ppm (non-zero in gray matter regions).
    dr_neg : numpy.ndarray
        Diamagnetic relaxivity map in Hz/ppm (non-zero in white matter).
    """
    # Dr_pos: single kernel in the (non-WM) grey-matter regions
    non_wm_labels = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    dr_pos = np.zeros_like(seg, dtype=np.float64)
    for label in non_wm_labels:
        dr_pos[seg == label] = dr

    # Dr_neg: depends on anisotropy flag
    dr_neg = np.zeros_like(seg, dtype=np.float64)
    wm_mask = (seg == 8)

    if anisotropy and angle_map is not None:
        # Orientation-dependent: single kernel modulated by sin^2(theta)
        angle_rad = np.deg2rad(angle_map.astype(np.float64))
        dr_neg[wm_mask] = dr * np.sin(angle_rad[wm_mask])**2
    else:
        # Isotropic single kernel for WM
        dr_neg[wm_mask] = dr

    dr_neg[np.isnan(dr_neg)] = 0

    return dr_pos, dr_neg


def generate_dr_maps_ridani(seg, theta, B0=7, anisotropic=True, dr_fixed=None):
    """
    Generate the paramagnetic/diamagnetic R2' relaxivity (Dr) maps (calculate_Dr.m).

    Paramagnetic (Dr_pos), non-WM regions [1,2,3,4,5,6,7,9,10,11]:
        Dr_pos = (2*pi * 2*pi * 42.58 * B0) / (9*sqrt(3))
    (0 in WM label 8, 0 elsewhere).

    Diamagnetic (Dr_neg), WM label 8:
        anisotropic=True  : Dr_neg = 0.5 * 42.58 * 2*pi * B0 * sin(theta_rad)^2
        anisotropic=False : Dr_neg = 700.8 (the constant option in calculate_Dr.m)

    NOTE: calculate_Dr.m hard-codes B0=7 in the numeric constants; ``B0`` here
    scales those expressions the same way the formulae are written, so B0=7
    reproduces the MATLAB exactly.

    Parameters
    ----------
    seg : numpy.ndarray
        3D SegmentedModel labels.
    theta : numpy.ndarray
        3D fibre-to-field angle map in degrees (generate_theta_from_v1); used only
        for the anisotropic Dr_neg.
    B0 : float
        Field strength in Tesla. Default 7 (matches calculate_Dr.m constants).
    anisotropic : bool
        If True, WM Dr_neg is the sin^2(theta) variable form; if False, the 700.8
        constant form. Default True.
    dr_fixed : float or None
        If given, both maps are rescaled so the paramagnetic relaxivity equals
        ``dr_fixed`` Hz/ppm (field-independent), preserving the sin^2(theta) shape.
        If None (default), the field-scaled static-dephasing magnitudes are used.

    Returns
    -------
    dr_pos_map : numpy.ndarray
        Paramagnetic relaxivity map (Hz).
    dr_neg_map : numpy.ndarray
        Diamagnetic relaxivity map (Hz).
    """
    seg = np.asarray(seg, dtype=np.float64)
    regions_to_modify = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]

    value_positive = (2 * np.pi * 2 * np.pi * 42.58 * B0) / (9 * np.sqrt(3))
    dr_pos_map = np.zeros_like(seg)
    for region in regions_to_modify:
        dr_pos_map[seg == region] = value_positive
    dr_pos_map[seg == 8] = 0

    wm = (seg == 8)
    dr_neg_map = np.zeros_like(seg)
    if anisotropic:
        theta = np.asarray(theta, dtype=np.float64)
        drneg = 0.5 * 42.58 * 2 * np.pi * B0 * np.sin(np.deg2rad(theta)) ** 2
        drneg[np.isnan(drneg)] = 0
        dr_neg_map[wm] = drneg[wm]
    else:
        dr_neg_map[wm] = 700.8

    if dr_fixed is not None:
        scale = dr_fixed / value_positive
        dr_pos_map = dr_pos_map * scale
        dr_neg_map = dr_neg_map * scale

    return dr_pos_map, dr_neg_map


def scale_maps_to_3t(R2, R2star, R1, seg, T2=None):
    """
    Scale tissue parameter maps from 7T to 3T field strength.

    Translated from NeuroPoly Susceptibility-Separation-Phantom Map_creation_3T.m.

    Parameters
    ----------
    R2 : numpy.ndarray
        R2 map in Hz at 7T.
    R2star : numpy.ndarray
        R2* map in Hz at 7T.
    R1 : numpy.ndarray
        R1 map in Hz at 7T.
    seg : numpy.ndarray
        Tissue segmentation labels.
    T2 : numpy.ndarray or None, optional
        T2 map in ms at 7T. If provided, also scaled.

    Returns
    -------
    dict
        Dictionary with keys 'R2', 'R2star', 'R1', and optionally 'T2',
        each containing the 3T-scaled map.
    """
    result = {
        'R2': np.array(R2, dtype=np.float64) * 0.65,
        'R2star': np.array(R2star, dtype=np.float64) * 0.5,
    }

    R1_3t = np.array(R1, dtype=np.float64).copy()
    for label, factor in R1_3T_DIVISION_FACTORS.items():
        mask = (seg == label)
        if mask.any():
            R1_3t[mask] = R1_3t[mask] / factor
    result['R1'] = R1_3t

    if T2 is not None:
        result['T2'] = np.array(T2, dtype=np.float64) / 0.65

    return result


# -----------------------------------------------------------------------------
# Faithful helpers ported from the Susceptibility-Separation-Phantom MATLAB utils
# (func/utils/{smooth3D.m, gradient_3D.m, sos.m, WaveletDenoising/Wavedec3Denoising.m}).
# -----------------------------------------------------------------------------

def smooth3d_fft(X, fwhm, vox=(1, 1, 1)):
    """
    3D Gaussian smoothing via FFT, port of smooth3D.m.

    dev = FWHM/2.35; the Gaussian is centred on an ndgrid of physical coordinates
    ((-dim/2 : dim/2-1)*vox) and normalised by ``sum(abs(GAUSS))`` before an
    FFT-domain multiply. Both the image and the kernel are transformed with an
    unshifted fftn (``fft3s``/``ifft3s`` in the MATLAB), so the returned array
    matches MATLAB's ``real(smooth3D(...))`` usage.

    Parameters
    ----------
    X : numpy.ndarray
        3D input volume.
    fwhm : float
        Full width at half maximum in mm.
    vox : sequence of float
        Voxel size in mm (3-element).

    Returns
    -------
    numpy.ndarray
        Smoothed volume (real part), same shape as X.
    """
    X = np.asarray(X, dtype=np.float64)
    dim = X.shape
    # ndgrid coordinates: (-n/2 : n/2-1) * vox, matching MATLAB smooth3D.m exactly.
    axes = [(np.arange(-d / 2, d / 2)) * v for d, v in zip(dim, vox)]
    X1, X2, X3 = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
    dev = fwhm / 2.35
    gauss = np.exp(-(X1 ** 2 + X2 ** 2 + X3 ** 2) / (2 * dev ** 2))
    gauss = gauss / np.sum(np.abs(gauss))
    # MATLAB smooth3D uses fft3s = fftshift(fft(ifftshift(x))): the ifftshift moves the
    # centre-peaked Gaussian to the array ORIGIN so the convolution introduces NO spatial
    # shift. Working this through, S = ifftn(fftn(X) * fftn(ifftshift(gauss))). Omitting the
    # ifftshift convolves with a centre-peaked kernel and shifts the result by dim/2 in every
    # axis (the bug that zeroed the smooth-branch tissues); the ifftshift below fixes it.
    S = np.fft.ifftn(np.fft.fftn(X) * np.fft.fftn(np.fft.ifftshift(gauss)))
    return np.real(S)


def gradient_3d_midpoint(I, res=(1, 1, 1)):
    """
    Midpoint (central-difference) 3D gradient, port of gradient_3D.m
    method 0 (the default). One-sided differences at the two boundary planes
    along each axis, central differences in the interior, each divided by the
    corresponding voxel size in ``res``.

    Parameters
    ----------
    I : numpy.ndarray
        3D input volume.
    res : sequence of float
        Voxel size in mm (3-element).

    Returns
    -------
    (dx, dy, dz) : tuple of numpy.ndarray
        Gradient components along each axis.
    """
    I = np.asarray(I, dtype=np.float64)

    def _grad_axis(A, axis):
        g = np.empty_like(A)
        sl = [slice(None)] * A.ndim

        def s(a, b=None):
            idx = list(sl)
            idx[axis] = slice(a, b)
            return tuple(idx)

        def si(i):
            idx = list(sl)
            idx[axis] = i
            return tuple(idx)

        # first plane: I(2)-I(1)
        g[si(0)] = A[si(1)] - A[si(0)]
        # interior: 0.5*(I(k+1)-I(k-1))
        g[s(1, -1)] = 0.5 * (A[s(2, None)] - A[s(0, -2)])
        # last plane: I(end)-I(end-1)
        g[si(-1)] = A[si(-1)] - A[si(-2)]
        return g

    dx = _grad_axis(I, 0) / res[0]
    dy = _grad_axis(I, 1) / res[1]
    dz = _grad_axis(I, 2) / res[2]
    return dx, dy, dz


def sos(x, dim=-1, pnorm=2):
    """
    Square root of the sum of squares along an axis, port of sos.m.

        res = (sum(|x|^pnorm, dim))^(1/pnorm)

    Default reduces over the last axis with pnorm=2.
    """
    x = np.asarray(x, dtype=np.float64)
    return (np.sum(np.abs(x) ** pnorm, axis=dim)) ** (1.0 / pnorm)


def wavelet_denoise(inmat, threshold, level, wavelet, method, mask=None):
    """
    Wavelet-domain denoising, port of Wavedec3Denoising.m.

    Uses pywt.wavedecn / waverecn (db2, ``level`` levels) and applies a
    hard / soft / verysoft threshold to every detail (and approximation) band,
    with the optional mask-weighting logic from the MATLAB (a band coefficient is
    only eligible for thresholding where the all-ones decomposition is NOT much
    larger than the mask decomposition, i.e. away from the mask's own edges).

    NOTE: pywt is algorithmically equivalent to MATLAB wavedec3, not
    byte-identical (different boundary handling / coefficient ordering), so the
    denoised output matches the MATLAB in behaviour but not bit-for-bit.

    Parameters
    ----------
    inmat : numpy.ndarray
        3D input volume.
    threshold : float
        Threshold (same units as the wavelet coefficients).
    level : int
        Number of decomposition levels (MATLAB L; the phantom uses 8).
    wavelet : str
        Wavelet name (the phantom uses 'db2').
    method : str
        'hard', 'soft', or 'verysoft'.
    mask : numpy.ndarray or None
        Optional weighting mask (same shape as inmat). Matches the MATLAB
        varargin{1} mask-weighting term.

    Returns
    -------
    numpy.ndarray
        Denoised volume (complex-capable; callers take abs(real(.))).
    """
    import pywt  # lazy import

    inmat = np.asarray(inmat, dtype=np.float64)
    coeffs = pywt.wavedecn(inmat, wavelet, level=level)

    # Build the per-band mask weighting (w_mask.dec) as in the MATLAB: for each
    # band, compare the wavedec of ones(size) against the wavedec of the mask.
    wmask = None
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        cmask = pywt.wavedecn(mask, wavelet, level=level)
        cones = pywt.wavedecn(np.ones_like(mask), wavelet, level=level)
        wmask = []
        for cm, co in zip(cmask, cones):
            if isinstance(cm, dict):
                wm = {}
                for key in cm:
                    m = cm[key]
                    o = co[key]
                    w = np.ones_like(o)
                    w[np.abs(o) > 1.1 * np.abs(m)] = 0
                    w[np.abs(o) <= 1.1 * np.abs(m)] = 1
                    w[np.abs(m) < 0.01] = 1
                    wm[key] = w
                wmask.append(wm)
            else:
                m = cm
                o = co
                w = np.ones_like(o)
                w[np.abs(o) > 1.1 * np.abs(m)] = 0
                w[np.abs(o) <= 1.1 * np.abs(m)] = 1
                w[np.abs(m) < 0.01] = 1
                wmask.append(w)

    # Global max over all bands (for the 'verysoft' rescale factor). The MATLAB
    # computes max(abs(y_r.dec{k}(:))) per band k.
    def _threshold_band(band, wband):
        band = band.astype(np.complex128)
        w = np.ones_like(band, dtype=bool) if wband is None else (wband != 0)
        below = (np.abs(band) < threshold) & w
        keep = ~((np.abs(band) < threshold) & w)
        if method == 'hard':
            band[below] = 0
        elif method == 'soft':
            band[below] = 0
            ang = np.angle(band[keep])
            band[keep] = band[keep] - np.exp(1j * ang) * threshold
        elif method == 'verysoft':
            band[below] = 0
            mx = np.max(np.abs(band)) if band.size else 0.0
            denom = (mx - threshold)
            if denom != 0:
                ang = np.angle(band[keep])
                band[keep] = mx / denom * (band[keep] - np.exp(1j * ang) * threshold)
        else:
            raise ValueError(f"Unknown wavelet denoising method: {method}")
        return band

    for i, band in enumerate(coeffs):
        wband_group = wmask[i] if wmask is not None else None
        if isinstance(band, dict):
            for key in band:
                wband = wband_group[key] if wband_group is not None else None
                band[key] = _threshold_band(band[key], wband)
        else:
            coeffs[i] = _threshold_band(band, wband_group)

    out = pywt.waverecn(coeffs, wavelet)
    # waverecn may pad by one sample per level; crop back to input shape.
    slices = tuple(slice(0, s) for s in inmat.shape)
    return out[slices]


def generate_theta_from_v1(v1, B0_dir=np.array([0, 0, 1]), mask=None):
    """
    Compute the fibre-to-field angle theta (degrees) from the primary eigenvector
    V1, port of the PhantomCreation.m "Theta" block.

    The MATLAB uses (with B0 along z, i.e. the in-plane components are 1:2):
        numerator   = sum(V1(:,:,:,1:2).^2, 4)
        denominator = sum(V1.^2, 4).^2      % NOTE: the denominator is SQUARED
        sin2        = numerator ./ denominator
        theta       = asin(sqrt(sin2)) * 180/pi

    This reproduces that expression EXACTLY, including the squared denominator
    (a quirk of the original code). The "in-plane" components are taken as those
    perpendicular to ``B0_dir``; for the default z-oriented B0 these are the x,y
    components, matching V1(:,:,:,1:2).

    Parameters
    ----------
    v1 : numpy.ndarray
        4D array (x, y, z, 3) of the primary eigenvector.
    B0_dir : numpy.ndarray
        B0 direction (3-element). Default [0, 0, 1].
    mask : numpy.ndarray or None
        Optional 3D ROI mask; V1 is zeroed outside it before the computation
        (matching the MATLAB roi_mask multiply). If None, the whole volume is used.

    Returns
    -------
    numpy.ndarray
        theta in degrees (3D). NaNs (from zero-norm voxels) are left as produced,
        as in the MATLAB (they are handled downstream).
    """
    v1 = np.asarray(v1, dtype=np.float64)
    B0_dir = np.asarray(B0_dir, dtype=np.float64)
    B0_dir = B0_dir / np.linalg.norm(B0_dir)

    if mask is not None:
        v1 = v1 * (mask[..., None] != 0)

    # component index parallel to B0 (the "z" axis in the default case)
    par_axis = int(np.argmax(np.abs(B0_dir)))
    perp_axes = [a for a in range(3) if a != par_axis]

    numerator = np.sum(v1[..., perp_axes] ** 2, axis=-1)
    denominator = np.sum(v1 ** 2, axis=-1) ** 2  # squared denominator, as in PhantomCreation.m
    with np.errstate(divide='ignore', invalid='ignore'):
        sin2 = numerator / denominator
        theta = np.arcsin(np.sqrt(sin2)) * (180.0 / np.pi)
    return theta


def apply_wm_anisotropy(chi_neg, wm_tract_mask, theta, R1=None, seg=None,
                        tract_params=None, region8_r1_weighting=True,
                        noise_std=0.01, noise_seed=0):
    """
    Apply per-tract white-matter susceptibility anisotropy to chi- (the "Apparent
    susceptibility" block of PhantomCreation.m).

    For each WM tract label (1..11) in ``wm_tract_mask`` the apparent diamagnetic
    susceptibility is
        chi_neg = delta_chi * cos^2(theta) + chi_0
    with (delta_chi, chi_0) taken from the phantom's deltaX_values/Xzero_values
    (WM_TRACT_ANISOTROPY_ARRAYS) and theta the fibre-to-field angle in DEGREES
    (see generate_theta_from_v1). This overwrites chi- in the union of the WM
    tracts, exactly as the MATLAB replaces ``chi_negative_data(segmentation==region)``.

    Parameters
    ----------
    chi_neg : numpy.ndarray
        3D diamagnetic susceptibility map (ppm). Modified in the WM-tract voxels.
    wm_tract_mask : numpy.ndarray
        3D white_matter_mask with tract labels 1..11 (0 elsewhere).
    theta : numpy.ndarray
        3D fibre-to-field angle map in degrees (from generate_theta_from_v1).
    R1 : numpy.ndarray or None
        3D R1 map, used for the region-8 R1-weighting step below (only when
        ``region8_r1_weighting`` and ``seg`` are provided).
    seg : numpy.ndarray or None
        3D SegmentedModel labels; region 8 (generic WM) is where the R1-weighting
        is applied in PhantomCreation.m.
    tract_params : dict or None
        {label: (delta_chi, chi_0)}. Default WM_TRACT_ANISOTROPY_ARRAYS.
    region8_r1_weighting : bool
        If True and R1 and seg are given, apply the R1-percentage texture in
        SegmentedModel region 8.
    noise_std : float
        Standard deviation of the Gaussian noise added to the R1-percentage term
        at each of the three iterations (PhantomCreation.m uses 0.01). Set to 0 to
        apply the deterministic R1-percentage term only.
    noise_seed : int or None
        Seed for the noise generator, so the (otherwise random) texture is
        reproducible. Default 0. Ignored when ``noise_std`` is 0 or None.

    Returns
    -------
    numpy.ndarray
        chi_neg with anisotropic WM susceptibility (float, NaNs -> 0).
    """
    if tract_params is None:
        tract_params = WM_TRACT_ANISOTROPY_ARRAYS

    chi_neg = np.asarray(chi_neg, dtype=np.float64).copy()
    theta = np.asarray(theta, dtype=np.float64)

    # Build the delta_X and Xzero maps and the apparent susceptibility Xapp
    # exactly as PhantomCreation.m: Xapp = deltaX.*cos(theta*pi/180).^2 + Xzero.
    delta_map = np.zeros_like(chi_neg)
    xzero_map = np.zeros_like(chi_neg)
    tract_union = np.zeros_like(chi_neg, dtype=bool)
    for label, (delta_chi, chi_0) in tract_params.items():
        m = (wm_tract_mask == label)
        if not m.any():
            continue
        delta_map[m] = delta_chi
        xzero_map[m] = chi_0
        tract_union |= m

    xapp = delta_map * (np.cos(theta * np.pi / 180.0) ** 2) + xzero_map

    # Region-8 (WM in SegmentedModel) R1-percentage weighting. PhantomCreation.m
    # iterates xapp *= (pct + N(0, noise_std)) three times, where pct = R1/mean(R1)
    # over the region. The noise generator is seeded (noise_seed) so the texture is
    # reproducible.
    region8 = (seg == 8) if seg is not None else tract_union
    if region8_r1_weighting and R1 is not None and seg is not None:
        R1 = np.asarray(R1, dtype=np.float64)
        if region8.any():
            mean_r1 = np.mean(R1[region8])
            if mean_r1 != 0:
                pct = np.zeros_like(chi_neg)
                pct[region8] = R1[region8] / mean_r1
                rng = np.random.default_rng(noise_seed) if noise_std else None
                for _ in range(3):
                    weight = pct if not noise_std else pct + rng.normal(0.0, noise_std, size=pct.shape)
                    xapp = np.where(region8, xapp * weight, xapp)

    # Replace chi- in the SegmentedModel WM (label 8), exactly as PhantomCreation.m
    # (`chi_negative_data(segmentation==8) = Xapp_masked(...)`). The white-matter tract
    # mask supplies the per-tract delta_chi/chi_0; it covers ~99% of WM(8), and the
    # uncovered WM voxels get xapp≈0 there, matching the reference.
    overwrite = region8 if seg is not None else tract_union
    chi_neg[overwrite] = xapp[overwrite]
    chi_neg[np.isnan(chi_neg)] = 0
    return chi_neg


def _build_chisep_component(seg, R1, R2star, weighting_between_models,
                            r12chi, r2star2chi, base_chi, smooth_fwhm, voxel_size,
                            hg_mask_index):
    """
    Build ONE chi component (chi+ or chi-) bottom-up per PhantomCreationFunction.m.

    ``r12chi``/``r2star2chi`` are per-label 1-based lists of the R1 / R2* modulation
    coefficients for this component; ``base_chi`` is the per-label base value
    (chipos or chineg). ``hg_mask_index`` is the boolean LG_mask (HighGrad==0|1).
    Returns Chimap5 = W^4*Chimap3 + (1-W^4)*Chimap4.
    """
    shape = seg.shape
    chimap3 = np.zeros(shape, dtype=np.float64)
    chimap4 = np.zeros(shape, dtype=np.float64)
    prob_acc = np.zeros(shape, dtype=np.float64)

    NSTD = 3
    smoothThresh = 0.9
    smoothThresh2 = 0.05

    labels = sorted(base_chi.keys())
    for k in labels:
        mask_k = (seg == k)
        n = int(mask_k.sum())
        if n <= 1:
            continue

        # FAITHFUL: PhantomCreationFunction.m calls smooth3D(Mask, FWHM_V, [1,1,1]) — i.e. the
        # probability-blend Gaussian FWHM is in VOXEL units ([1,1,1]), NOT the real voxel size.
        # Using the real 0.64 mm here widens the kernel ~1.5x, over-spreading the tissue
        # probabilities (over-reduces the hard-branch tissues like blood and smears deep-GM
        # interiors low). Pass [1,1,1] to match the reference exactly.
        mask_smooth = np.real(smooth3d_fft(mask_k.astype(np.float64), smooth_fwhm, (1.0, 1.0, 1.0)))
        r2s_mean = np.mean(R2star[mask_k])
        r1_mean = np.mean(R1[mask_k])
        chipos_k = base_chi[k]

        if k <= 10:
            prob_acc += mask_smooth
            lg = hg_mask_index  # boolean over full volume
            # chitemp over the LG voxels
            chitemp = (r2star2chi[k] * (R2star[lg] - r2s_mean)
                       + r12chi[k] * (R1[lg] - r1_mean))
            ms_lg = mask_smooth[lg]

            sel_hi = ms_lg > smoothThresh
            if sel_hi.any():
                mod_std = np.std(chitemp[sel_hi], ddof=1) if sel_hi.sum() > 1 else 0.0
                mod_mean = np.mean(chitemp[sel_hi])
            else:
                mod_std = 0.0
                mod_mean = 0.0
            clamp = (ms_lg > smoothThresh2) & (np.abs(chitemp - mod_mean) > NSTD * mod_std)
            chitemp = chitemp.copy()
            chitemp[clamp] = mod_mean

            add3 = np.zeros(shape, dtype=np.float64)
            add3[lg] = ms_lg * (chipos_k + chitemp)
            chimap3 += add3

            # second clamp pass (Chimap4 uses the re-clamped chitemp)
            if sel_hi.any():
                mod_std = np.std(chitemp[sel_hi], ddof=1) if sel_hi.sum() > 1 else 0.0
                mod_mean = np.mean(chitemp[sel_hi])
            else:
                mod_std = 0.0
                mod_mean = 0.0
            clamp = (ms_lg > smoothThresh2) & (np.abs(chitemp - mod_mean) > NSTD * mod_std)
            chitemp[clamp] = mod_mean

            add4 = np.zeros(shape, dtype=np.float64)
            add4[lg] = ms_lg * (chipos_k + chitemp)
            chimap4 += add4
        else:
            idx = mask_k
            chitemp = (r2star2chi[k] * (R2star[idx] - r2s_mean)
                       + r12chi[k] * (R1[idx] - r1_mean))
            # In the MATLAB, Mask(indexes) is all-ones inside the region, so the
            # (Mask>smoothThresh) selection is the whole region and the two
            # inequality-clamps (chitemp>m+3s AND chitemp<m+3s) together set
            # EVERY voxel to modulation_mean.
            # Reproduces the MATLAB clamp, which sets chitemp to its mean over the region.
            mod_mean = np.mean(chitemp)
            chitemp = np.full_like(chitemp, mod_mean)

            prob_here = (1.0 - prob_acc[idx])  # (Mask - ProbabilityAccumulated)
            chimap3[idx] += prob_here * (chipos_k + chitemp)

            # second pass: chitemp already flat -> mean unchanged
            mod_mean = np.mean(chitemp)
            chitemp = np.full_like(chitemp, mod_mean)
            chimap4[idx] += prob_here * (chipos_k + chitemp)

    W4 = weighting_between_models ** 4
    return W4 * chimap3 + (1.0 - W4) * chimap4


def generate_chisep_maps(seg, mask, R1, R2star, raw_field, high_grad,
                         voxel_size=(0.64, 0.64, 0.64), tissue_params=None,
                         boundary_smooth_fwhm=1.2, v1_map=None,
                         wm_tract_mask=None, anisotropy=False,
                         B0_dir=np.array([0, 0, 1]), apply_brain_mask=True,
                         enforce_sign=True):
    """
    Build paramagnetic (chi+) and diamagnetic (chi-) susceptibility maps BOTTOM-UP,
    a port of PhantomCreationFunction.m (+ the anisotropy block of
    PhantomCreation.m).

    chi+ and chi- are each constructed
    independently from the per-tissue base values (chipos / chineg) plus real
    intra-tissue texture derived from the wavelet-denoised R1 and R2* maps:

        chitemp = R2star2chi*(R2* - tissue_mean) + R12chi*(R1 - tissue_mean)

    with a NSTD=3 outlier clamp, a smooth3D(FWHM=1.2) probability blend, and a
    two-model field-gradient weighting
        Chimap5 = W^4 * Chimap3 + (1-W^4) * Chimap4,  W = cos(fieldgradient).
    See PhantomCreationFunction.m for the exact construction.

    Parameters
    ----------
    seg : numpy.ndarray
        3D SegmentedModel labels (1..16), matching CHISEP_TISSUE_PARAMS.
    mask : numpy.ndarray
        3D brain mask (applied to the outputs).
    R1 : numpy.ndarray
        3D R1 map (1/T1). Wavelet-denoised internally (threshold 25/1000, db2,
        'verysoft', mask R1!=0), per the MATLAB.
    R2star : numpy.ndarray
        3D R2* map (1/T2*). Wavelet-denoised internally (threshold 45/10, db2,
        'soft').
    raw_field : numpy.ndarray
        3D raw field map; its gradient magnitude drives the two-model weighting.
    high_grad : numpy.ndarray
        3D high-gradient mask (0/1); defines the LG mask and modulates the field
        gradient.
    voxel_size : tuple of float
        Voxel size in mm. Default (0.64, 0.64, 0.64).
    tissue_params : dict or None
        Per-tissue 6-tuples (chipos, chineg, R12chipos, R2star2chipos, R12chineg,
        R2star2chineg). Default CHISEP_TISSUE_PARAMS.
    boundary_smooth_fwhm : float
        FWHM (mm) for smooth3D probability blending. Default 1.2.
    v1_map : numpy.ndarray or None
        4D (x,y,z,3) primary-eigenvector map; required when anisotropy=True.
    wm_tract_mask : numpy.ndarray or None
        3D white_matter_mask with tract labels 1..11; required when anisotropy=True.
    anisotropy : bool
        If True, overwrite chi- in the WM tracts with the apparent
        (orientation-dependent) susceptibility (apply_wm_anisotropy).
    B0_dir : numpy.ndarray
        B0 direction for the theta computation. Default [0, 0, 1].

    Returns
    -------
    chi_pos : numpy.ndarray (float32)
        Paramagnetic susceptibility map (ppm).
    chi_neg : numpy.ndarray (float32)
        Diamagnetic susceptibility map (ppm).
    """
    if tissue_params is None:
        tissue_params = CHISEP_TISSUE_PARAMS

    seg = np.asarray(seg, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    R2star = np.asarray(R2star, dtype=np.float64)
    raw_field = np.asarray(raw_field, dtype=np.float64)
    high_grad = np.asarray(high_grad, dtype=np.float64)
    vox = tuple(voxel_size)

    # Wavelet denoise R1 (verysoft) and R2* (soft), matching PhantomCreationFunction.m.
    R1_dn = np.abs(np.real(wavelet_denoise(R1, 25.0 / 1000.0, 8, 'db2', 'verysoft', mask=(R1 != 0).astype(np.float64))))
    R2star_dn = np.abs(np.real(wavelet_denoise(R2star, 45.0 / 10.0, 8, 'db2', 'soft', mask=(R1_dn != 0).astype(np.float64))))

    # Field-gradient two-model weighting (WeightingBetweenModels = cos(fieldgradient)).
    gx, gy, gz = gradient_3d_midpoint(raw_field, res=(1, 1, 1))
    fieldgradient = sos(np.stack([gx, gy, gz], axis=-1), dim=-1)
    fieldgradient[seg >= 11] = 0
    fieldgradient = fieldgradient * high_grad
    nz = fieldgradient != 0
    fieldgradient[nz & (fieldgradient > 0.2)] = 0.2
    fieldgradient[nz & (fieldgradient < 0.05)] = 0.05
    nz = fieldgradient != 0
    fieldgradient[nz] = (fieldgradient[nz] - 0.05) / (0.2 - 0.05)
    weighting_between_models = np.cos(fieldgradient)

    # LG mask index: HighGrad == 0 or == 1 (i.e. essentially the whole defined volume).
    hg_mask_index = (high_grad == 0) | (high_grad == 1)

    # Per-label modulation coefficient lookups (1-based labels).
    r12chipos = {k: v[2] for k, v in tissue_params.items()}
    r2star2chipos = {k: v[3] for k, v in tissue_params.items()}
    r12chineg = {k: v[4] for k, v in tissue_params.items()}
    r2star2chineg = {k: v[5] for k, v in tissue_params.items()}
    base_chipos = {k: v[0] for k, v in tissue_params.items()}
    base_chineg = {k: v[1] for k, v in tissue_params.items()}

    chi_pos = _build_chisep_component(
        seg, R1_dn, R2star_dn, weighting_between_models,
        r12chipos, r2star2chipos, base_chipos, boundary_smooth_fwhm, vox, hg_mask_index)
    chi_neg = _build_chisep_component(
        seg, R1_dn, R2star_dn, weighting_between_models,
        r12chineg, r2star2chineg, base_chineg, boundary_smooth_fwhm, vox, hg_mask_index)

    # Optional WM susceptibility anisotropy (chi- only), per PhantomCreation.m.
    if anisotropy:
        if v1_map is None or wm_tract_mask is None:
            raise ValueError("anisotropy=True requires both v1_map and wm_tract_mask")
        theta = generate_theta_from_v1(v1_map, B0_dir=B0_dir, mask=(seg == 8))
        chi_neg = apply_wm_anisotropy(chi_neg, wm_tract_mask, theta, R1=R1_dn, seg=seg)

    # Enforce the physical sign of each source (paramagnetic chi+ >= 0, diamagnetic
    # chi- <= 0). The R1/R2* modulation can otherwise push a few voxels past zero.
    if enforce_sign:
        chi_pos = np.maximum(chi_pos, 0.0)
        chi_neg = np.minimum(chi_neg, 0.0)

    # Restrict to the brain mask unless a whole-head map is requested.
    if apply_brain_mask:
        brain = (mask > 0)
        chi_pos = chi_pos * brain
        chi_neg = chi_neg * brain
    chi_pos[np.isnan(chi_pos)] = 0
    chi_neg[np.isnan(chi_neg)] = 0

    return chi_pos.astype(np.float32), chi_neg.astype(np.float32)


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

def generate_signal(field, B0=3, TR=1, TE=30e-3, flip_angle=90, phase_offset=0, R1=1, R2star=50, M0=1,
                    R2=None, dr_pos=None, dr_neg=None, chi_pos=None, chi_neg=None, multicompartment=False):
    """
    Compute the MRI signal based on the given parameters.

    Supports two decay models:
    - Standard R2* model (default): decay = exp(-TE * R2star)
    - Chi-sep-aware model: decay = exp(-TE * (R2 + dr_pos*|chi+| + dr_neg*|chi-|))
      Activated when R2, dr_pos, dr_neg, chi_pos, and chi_neg are all provided.

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
    R2 : numpy.ndarray or None, optional
        Transverse relaxation rate (exchange-only, no susceptibility). Used with chi-sep model.
    dr_pos : numpy.ndarray or None, optional
        Paramagnetic relaxivity map in Hz/ppm. Used with chi-sep model.
    dr_neg : numpy.ndarray or None, optional
        Diamagnetic relaxivity map in Hz/ppm. Used with chi-sep model.
    chi_pos : numpy.ndarray or None, optional
        Paramagnetic susceptibility map in ppm. Used with chi-sep model.
    chi_neg : numpy.ndarray or None, optional
        Diamagnetic susceptibility map in ppm. Used with chi-sep model.

    Returns
    -------
    numpy.ndarray
        The computed MRI signal.

    """

    # Choose decay model
    chisep = R2 is not None and dr_pos is not None and dr_neg is not None and chi_pos is not None and chi_neg is not None
    if chisep and multicompartment:
        # Multi-compartment (signal-domain) chi-sep model: the voxel magnitude is |Σ compartments|,
        # each a static-dephasing exponential with its OWN decay rate (R2 + Dr·|chi|) and sphere
        # frequency ((2/3)·chi·γ·B0). Paramagnetic and diamagnetic compartments beat against each
        # other, giving the non-mono-exponential magnitude that signal-domain separators (DECOMPOSE)
        # need. The decay rates use the SAME Dr kernel as generate_r2prime, so the effective R2* (and
        # thus R2') is unchanged for the R2'-domain methods — only the *shape* of the decay is enriched.
        # Volume fractions split the susceptibility signal by relative source content (so a pure-para
        # voxel has no diamagnetic compartment), leaving a neutral (non-source) fraction.
        w = 2 * np.pi * (2.0 / 3.0) * 42.58 * B0  # (2/3)·γ·B0 phase coeff (per ppm, per s), matching the bulk-phase 42.58
        tot = np.abs(chi_pos) + np.abs(chi_neg) + 1e-6
        C_pos = 0.5 * np.abs(chi_pos) / tot
        C_neg = 0.5 * np.abs(chi_neg) / tot
        C_0 = 1.0 - C_pos - C_neg
        S = (C_pos * np.exp(-(R2 + dr_pos * np.abs(chi_pos) + 1j * w * chi_pos) * TE)
             + C_neg * np.exp(-(R2 + dr_neg * np.abs(chi_neg) + 1j * w * chi_neg) * TE)
             + C_0 * np.exp(-R2 * TE))
        decay = np.abs(S)  # multi-compartment magnitude envelope; the bulk field carries the phase below
    elif chisep:
        # Chi-sep-aware signal model: S ~ exp(-TE * (R2 + Dr_pos*|chi+| + Dr_neg*|chi-|))
        decay = np.exp(-TE * (R2 + dr_pos * np.abs(chi_pos) + dr_neg * np.abs(chi_neg)))
    else:
        # Standard R2* model
        decay = np.exp(-TE * R2star)

    sigHR = M0 * np.exp(1j * (2 * np.pi * field * B0 * 42.58 * TE + phase_offset)) * decay \
        * (1 - np.exp(-TR * R1)) * np.sin(np.deg2rad(flip_angle)) / (1 - np.cos(np.deg2rad(flip_angle)) * np.exp(-TR * R1))
    sigHR[np.isnan(sigHR)] = 0

    return sigHR

def generate_se_signal(TR=1.0, TE=30e-3, R1=1, R2=50, M0=1):
    """
    Compute the spin-echo (SE) magnitude signal.

        S_SE = M0 * (1 - exp(-TR * R1)) * exp(-TE * R2)

    Unlike the gradient-echo signal, the 180-degree refocusing pulse reverses
    static (susceptibility-induced) dephasing, so the SE magnitude decays with
    the irreversible transverse relaxation rate R2 (not R2*) and carries no
    susceptibility-induced phase. This is the forward-model extension of Stoll
    (2025), Eq. 3.5 — pairing an SE R2 acquisition with the GRE R2* acquisition
    lets a chi-separation method derive R2' = R2* - R2 the same way it would from
    real data, instead of being handed a ground-truth R2' map.

    Parameters
    ----------
    TR : float, optional
        Repetition time in seconds. Default is 1.0.
    TE : float, optional
        Echo time in seconds. Default is 30e-3.
    R1 : float or numpy.ndarray, optional
        Longitudinal relaxation rate in Hz. Default is 1.
    R2 : float or numpy.ndarray, optional
        Transverse relaxation rate in Hz (1/T2), NOT R2*. Default is 50.
    M0 : float or numpy.ndarray, optional
        Equilibrium magnetization. Default is 1.

    Returns
    -------
    numpy.ndarray
        The computed SE magnitude signal (real, non-negative).
    """
    sig = M0 * (1 - np.exp(-TR * R1)) * np.exp(-TE * R2)
    sig = np.asarray(sig, dtype=np.float64)
    sig[np.isnan(sig)] = 0
    return sig

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
    # Store the original dtype
    original_dtype = nii.get_data_dtype()

    original_shape = np.array(nii.header.get_data_shape())
    target_shape = np.array(np.round((np.array(nii.header.get_zooms()) / voxel_size) * original_shape), dtype=int)

    if np.array_equal(original_shape, target_shape):
        return nii

    # Create a new affine matrix that directly sets the diagonal to the new voxel sizes
    new_affine = np.eye(4)
    new_affine[:3, :3] = nii.affine[:3, :3]
    scale_factors = np.divide(nii.header.get_zooms(), voxel_size)

    # Adjust the voxel sizes in the new affine
    for i in range(3):
        new_affine[i, i] = nii.affine[i, i] / scale_factors[i]

    # If using nearest interpolation (binary mask), cast the data to float32 to avoid casting issues
    if interpolation == 'nearest':
        nii = nib.Nifti1Image(nii.get_fdata().astype(np.float32), nii.affine, nii.header)

    # Resample the image
    resampled_nii = resample_img(
        nii,
        target_affine=new_affine,
        target_shape=target_shape,
        interpolation=interpolation
    )

    # Cast back to the original dtype
    resampled_data = resampled_nii.get_fdata().astype(original_dtype)
    resampled_nii = nib.Nifti1Image(resampled_data, resampled_nii.affine, resampled_nii.header)

    return resampled_nii


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

def generate_chisep_phantom(resolution, para_vals=None, dia_vals=None, para_radii=None, dia_radii=None):
    """
    Generate a phantom with separate paramagnetic and diamagnetic susceptibility sources.

    Creates concentric cylinders: inner cylinders are paramagnetic (iron-like),
    surrounded by shells that are diamagnetic (myelin-like). This mimics the
    spatial relationship in brain tissue where iron-rich deep gray matter
    nuclei are surrounded by myelinated white matter.

    Parameters
    ----------
    resolution : list of int
        The dimensions of the phantom [nx, ny, nz].
    para_vals : list of float, optional
        Paramagnetic susceptibility values (ppm, positive) for each source.
        Default is [0.05, 0.10, 0.15, 0.20].
    dia_vals : list of float, optional
        Diamagnetic susceptibility values (ppm, negative) for each source.
        Default is [-0.02, -0.03, -0.04, -0.05].
    para_radii : list of float, optional
        Radii of paramagnetic inner cylinders (voxels). Default is [4, 4, 4, 5].
    dia_radii : list of float, optional
        Outer radii of diamagnetic shells (voxels). Default is [7, 7, 7, 9].

    Returns
    -------
    chi_pos : numpy.ndarray
        Paramagnetic susceptibility map (>= 0).
    chi_neg : numpy.ndarray
        Diamagnetic susceptibility map (<= 0).
    mask : numpy.ndarray
        Binary mask of the phantom region.
    """
    if para_vals is None:
        para_vals = [0.05, 0.10, 0.15, 0.20]
    if dia_vals is None:
        dia_vals = [-0.02, -0.03, -0.04, -0.05]
    if para_radii is None:
        para_radii = [4, 4, 4, 5]
    if dia_radii is None:
        dia_radii = [7, 7, 7, 9]

    assert len(para_vals) == len(dia_vals) == len(para_radii) == len(dia_radii), \
        "All source parameter lists must have the same length"

    chi_pos = np.zeros(resolution, dtype=float)
    chi_neg = np.zeros(resolution, dtype=float)

    center = [res // 2 for res in resolution]
    large_radius = min(center[1:]) * 0.75

    z, y, x = np.indices(resolution)

    # Cylinder height limits
    lower_limit = (1 - 0.6) / 2 * resolution[0]
    upper_limit = (1 + 0.6) / 2 * resolution[0]

    # Large background cylinder (mask region)
    mask = ((z - center[2])**2 + (y - center[1])**2 < large_radius**2) & \
           (x >= (1 - 0.75) / 2 * resolution[0]) & (x < (1 + 0.75) / 2 * resolution[0])

    # Place sources in a circle
    n_sources = len(para_vals)
    angle = 2 * np.pi / n_sources

    for i in range(n_sources):
        cy = center[1] + large_radius / 2 * np.sin(i * angle)
        cz = center[2] + large_radius / 2 * np.cos(i * angle)

        dist2 = (z - cz)**2 + (y - cy)**2
        in_height = (x >= lower_limit) & (x < upper_limit)

        # Diamagnetic shell (outer - inner)
        shell = (dist2 < dia_radii[i]**2) & (dist2 >= para_radii[i]**2) & in_height
        chi_neg[shell] = dia_vals[i]

        # Paramagnetic core
        core = (dist2 < para_radii[i]**2) & in_height
        chi_pos[core] = para_vals[i]

    return chi_pos, chi_neg, mask.astype(float)


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


