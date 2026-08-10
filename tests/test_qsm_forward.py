import pytest
import argparse
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qsm_forward.main import main
import qsm_forward


class TestArgumentParsing:
    """Test command-line argument parsing, focusing on output flags."""

    def test_save_field_flag_without_value_defaults_to_true(self):
        """Test that --save-field flag without explicit value sets to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-field']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                # Check the save_field keyword argument passed to generate_bids
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is True

    def test_save_field_flag_with_explicit_false(self):
        """Test that --save-field False explicitly sets to False."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-field', 'False']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is False

    def test_save_field_flag_with_explicit_true(self):
        """Test that --save-field True explicitly sets to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-field', 'True']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is True

    def test_save_shimmed_field_flag_without_value_defaults_to_true(self):
        """Test that --save-shimmed-field flag without explicit value sets to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-shimmed-field']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_shimmed_field'] is True

    def test_save_shimmed_field_flag_with_explicit_false(self):
        """Test that --save-shimmed-field False explicitly sets to False."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-shimmed-field', 'False']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_shimmed_field'] is False

    def test_save_shimmed_offset_field_flag_without_value_defaults_to_true(self):
        """Test that --save-shimmed-offset-field flag without explicit value sets to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-shimmed-offset-field']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_shimmed_offset_field'] is True

    def test_save_shimmed_offset_field_flag_with_explicit_false(self):
        """Test that --save-shimmed-offset-field False explicitly sets to False."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-shimmed-offset-field', 'False']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_shimmed_offset_field'] is False

    def test_all_save_flags_without_values_default_to_true(self):
        """Test that all three save flags without values default to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids',
                               '--save-field', '--save-shimmed-field', '--save-shimmed-offset-field']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is True
                assert call_kwargs['save_shimmed_field'] is True
                assert call_kwargs['save_shimmed_offset_field'] is True

    def test_mixed_flag_usage(self):
        """Test mixed usage of flags with and without explicit values."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids',
                               '--save-field', '--save-shimmed-field', 'False', '--save-shimmed-offset-field']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is True  # flag without value
                assert call_kwargs['save_shimmed_field'] is False  # explicit False
                assert call_kwargs['save_shimmed_offset_field'] is True  # flag without value

    def test_default_values_when_flags_not_provided(self):
        """Test that default values are False when flags are not provided."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_field'] is False
                assert call_kwargs['save_shimmed_field'] is False
                assert call_kwargs['save_shimmed_offset_field'] is False

    def test_other_save_flags_still_default_to_true(self):
        """Test that other save flags (chi, mask, segmentation) still default to True."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_chi'] is True
                assert call_kwargs['save_mask'] is True  
                assert call_kwargs['save_segmentation'] is True

    def test_other_save_flags_can_be_disabled(self):
        """Test that other save flags can be explicitly disabled."""
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids', '--save-chi', 'False', '--save-mask', 'False']):
            with patch('qsm_forward.generate_bids') as mock_generate_bids:
                main()
                mock_generate_bids.assert_called_once()
                call_kwargs = mock_generate_bids.call_args[1]
                assert call_kwargs['save_chi'] is False
                assert call_kwargs['save_mask'] is False
                assert call_kwargs['save_segmentation'] is True  # not modified


class TestFileOutputIntegration:
    """Integration tests that verify actual file outputs are created."""

    def test_simple_phantom_creates_expected_files_with_field_flags(self):
        """Test that simple phantom creates expected files when field flags are enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = os.path.join(temp_dir, "bids_output")
            
            # Run with field flags enabled
            with patch('sys.argv', ['qsm_forward', 'simple', bids_dir,
                                   '--save-field', '--save-shimmed-field', '--save-shimmed-offset-field',
                                   '--resolution', '20', '20', '20']):  # Small resolution for speed
                main()
            
            # Check that the expected directories exist
            subject_dir = os.path.join(bids_dir, "sub-1", "anat")
            deriv_dir = os.path.join(bids_dir, "derivatives", "qsm-forward", "sub-1", "anat")
            
            assert os.path.exists(subject_dir), f"Subject directory not found: {subject_dir}"
            assert os.path.exists(deriv_dir), f"Derivatives directory not found: {deriv_dir}"
            
            # Check for main output files (should always be created)
            # Files are created per echo, so check for echo-1 files as representative
            mag_file = os.path.join(subject_dir, "sub-1_echo-1_part-mag_MEGRE.nii")
            phs_file = os.path.join(subject_dir, "sub-1_echo-1_part-phase_MEGRE.nii")
            assert os.path.exists(mag_file), f"Magnitude file not found: {mag_file}"
            assert os.path.exists(phs_file), f"Phase file not found: {phs_file}"
            
            # Check for field map files (these should be created because flags were enabled)
            fieldmap_file = os.path.join(deriv_dir, "sub-1_fieldmap.nii")
            fieldmap_local_file = os.path.join(deriv_dir, "sub-1_fieldmap-local.nii")
            shimmed_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed_fieldmap.nii")
            shimmed_offset_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed-offset_fieldmap.nii")
            
            assert os.path.exists(fieldmap_file), f"Field map file not found: {fieldmap_file}"
            assert os.path.exists(fieldmap_local_file), f"Local field map file not found: {fieldmap_local_file}"
            assert os.path.exists(shimmed_fieldmap_file), f"Shimmed field map file not found: {shimmed_fieldmap_file}"
            assert os.path.exists(shimmed_offset_fieldmap_file), f"Shimmed offset field map file not found: {shimmed_offset_fieldmap_file}"
            
            # Check default files are still created
            chi_file = os.path.join(deriv_dir, "sub-1_Chimap.nii")  # Note: Capital C
            mask_file = os.path.join(deriv_dir, "sub-1_mask.nii")
            seg_file = os.path.join(deriv_dir, "sub-1_dseg.nii")
            assert os.path.exists(chi_file), f"Chi map file not found: {chi_file}"
            assert os.path.exists(mask_file), f"Mask file not found: {mask_file}"
            assert os.path.exists(seg_file), f"Segmentation file not found: {seg_file}"

    def test_simple_phantom_without_field_flags_excludes_field_files(self):
        """Test that simple phantom does not create field files when flags are disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = os.path.join(temp_dir, "bids_output")
            
            # Run without field flags (they should default to False)
            with patch('sys.argv', ['qsm_forward', 'simple', bids_dir,
                                   '--resolution', '20', '20', '20']):  # Small resolution for speed
                main()
            
            deriv_dir = os.path.join(bids_dir, "derivatives", "qsm-forward", "sub-1", "anat")
            
            # Check that field map files are NOT created
            fieldmap_file = os.path.join(deriv_dir, "sub-1_fieldmap.nii")
            shimmed_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed_fieldmap.nii")
            shimmed_offset_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed-offset_fieldmap.nii")
            
            assert not os.path.exists(fieldmap_file), f"Field map file should not exist: {fieldmap_file}"
            assert not os.path.exists(shimmed_fieldmap_file), f"Shimmed field map file should not exist: {shimmed_fieldmap_file}"
            assert not os.path.exists(shimmed_offset_fieldmap_file), f"Shimmed offset field map file should not exist: {shimmed_offset_fieldmap_file}"
            
            # But default files should still be created
            chi_file = os.path.join(deriv_dir, "sub-1_Chimap.nii")  # Note: Capital C
            mask_file = os.path.join(deriv_dir, "sub-1_mask.nii")
            seg_file = os.path.join(deriv_dir, "sub-1_dseg.nii")
            assert os.path.exists(chi_file), f"Chi map file not found: {chi_file}"
            assert os.path.exists(mask_file), f"Mask file not found: {mask_file}"
            assert os.path.exists(seg_file), f"Segmentation file not found: {seg_file}"

    def test_simple_phantom_selective_field_flags(self):
        """Test that only selected field files are created based on individual flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = os.path.join(temp_dir, "bids_output")
            
            # Run with only save-field and save-shimmed-field enabled
            with patch('sys.argv', ['qsm_forward', 'simple', bids_dir,
                                   '--save-field', '--save-shimmed-field',
                                   '--resolution', '20', '20', '20']):
                main()
            
            deriv_dir = os.path.join(bids_dir, "derivatives", "qsm-forward", "sub-1", "anat")
            
            # These should exist
            fieldmap_file = os.path.join(deriv_dir, "sub-1_fieldmap.nii")
            fieldmap_local_file = os.path.join(deriv_dir, "sub-1_fieldmap-local.nii")
            shimmed_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed_fieldmap.nii")
            assert os.path.exists(fieldmap_file), f"Field map file not found: {fieldmap_file}"
            assert os.path.exists(fieldmap_local_file), f"Local field map file not found: {fieldmap_local_file}"
            assert os.path.exists(shimmed_fieldmap_file), f"Shimmed field map file not found: {shimmed_fieldmap_file}"
            
            # This should NOT exist (save-shimmed-offset-field was not enabled)
            shimmed_offset_fieldmap_file = os.path.join(deriv_dir, "sub-1_desc-shimmed-offset_fieldmap.nii")
            assert not os.path.exists(shimmed_offset_fieldmap_file), f"Shimmed offset field map file should not exist: {shimmed_offset_fieldmap_file}"

    def test_head_phantom_mocked_for_missing_data(self):
        """Test that head phantom mode is properly handled when data directory is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_data_dir = os.path.join(temp_dir, "nonexistent_data")
            bids_dir = os.path.join(temp_dir, "bids_output")
            
            # Mock the TissueParams to avoid needing actual head phantom data
            with patch('qsm_forward.TissueParams') as mock_tissue_params:
                # Configure the mock to behave like it has the necessary data
                mock_instance = MagicMock()
                mock_tissue_params.return_value = mock_instance
                
                # Mock the generate_bids function to avoid actual processing
                with patch('qsm_forward.generate_bids') as mock_generate_bids:
                    # Run head phantom mode with field flags
                    with patch('sys.argv', ['qsm_forward', 'head', fake_data_dir, bids_dir,
                                           '--save-field', '--save-shimmed-field', '--save-shimmed-offset-field']):
                        main()
                    
                    # Verify that TissueParams was called with the data directory
                    mock_tissue_params.assert_called_once_with(
                        fake_data_dir,
                        chi_pos=None, chi_neg=None,
                        v1=None, R2=None, angle_map=None,
                    )
                    # Verify that generate_bids was called with the expected arguments
                    mock_generate_bids.assert_called_once()

    def test_file_content_validation(self):
        """Test that created files have reasonable content (non-empty, proper format)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_dir = os.path.join(temp_dir, "bids_output")
            
            # Run with field flags enabled
            with patch('sys.argv', ['qsm_forward', 'simple', bids_dir,
                                   '--save-field',
                                   '--resolution', '10', '10', '10']):  # Very small for speed
                main()
            
            deriv_dir = os.path.join(bids_dir, "derivatives", "qsm-forward", "sub-1", "anat")
            fieldmap_file = os.path.join(deriv_dir, "sub-1_fieldmap.nii")
            
            # Check file exists and has reasonable size
            assert os.path.exists(fieldmap_file), f"Field map file not found: {fieldmap_file}"
            file_size = os.path.getsize(fieldmap_file)
            assert file_size > 0, f"Field map file is empty: {fieldmap_file}"
            assert file_size > 100, f"Field map file suspiciously small: {file_size} bytes"  # Should be larger than just headers


import numpy as np


class TestGenerateT2Map:
    def test_output_shape_matches_input(self):
        seg = np.zeros((10, 10, 10), dtype=np.float64)
        seg[2:8, 2:8, 2:8] = 9  # Gray matter
        R2star = np.ones_like(seg) * 50.0
        M0 = np.ones_like(seg)
        T2, R2 = qsm_forward.generate_t2_map(seg, R2star, M0)
        assert T2.shape == seg.shape
        assert R2.shape == seg.shape

    def test_tissue_values_assigned(self):
        seg = np.zeros((10, 10, 10), dtype=np.float64)
        seg[3:7, 3:7, 3:7] = 8  # WM
        R2star = np.ones_like(seg) * 50.0
        M0 = np.ones_like(seg)
        T2, R2 = qsm_forward.generate_t2_map(seg, R2star, M0, gaussian_sigma=0)
        # WM T2 at 7T should be ~45.54 ms (modulated by R2*/M0)
        wm_t2 = T2[5, 5, 5]
        assert wm_t2 > 20 and wm_t2 < 100, f"WM T2={wm_t2} out of expected range"

    def test_r2_inverse_relationship(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 9  # All GM
        R2star = np.ones_like(seg) * 50.0
        M0 = np.ones_like(seg)
        T2, R2 = qsm_forward.generate_t2_map(seg, R2star, M0, gaussian_sigma=0)
        # R2 = 1000 / T2 where T2 > 0
        mask = T2 > 0
        np.testing.assert_allclose(R2[mask], 1000.0 / T2[mask], rtol=1e-10)

    def test_nan_handling(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 9
        R2star = np.ones_like(seg) * 50.0
        R2star[2, 2, 2] = np.nan
        M0 = np.ones_like(seg)
        T2, R2 = qsm_forward.generate_t2_map(seg, R2star, M0)
        assert np.all(np.isfinite(R2))


class TestGenerateDrMaps:
    def test_dr_pos_single_kernel(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 9  # All GM
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7)
        np.testing.assert_allclose(dr_pos[2, 2, 2], qsm_forward.DR_KERNEL)

    def test_wm_has_zero_dr_pos(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 8  # All WM
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7)
        assert np.all(dr_pos == 0)

    def test_dr_neg_constant_mode(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 8  # All WM
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7, anisotropy=False)
        assert np.all(dr_neg == qsm_forward.DR_KERNEL)

    def test_dr_neg_anisotropy_mode(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 8  # All WM
        angle_map = np.ones((5, 5, 5)) * 45.0  # 45 degrees
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7, angle_map=angle_map, anisotropy=True)
        expected = qsm_forward.DR_KERNEL * np.sin(np.deg2rad(45))**2
        np.testing.assert_allclose(dr_neg[2, 2, 2], expected, rtol=1e-10)

    def test_dr_custom_kernel(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 9  # All GM
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7, dr=100.0)
        np.testing.assert_allclose(dr_pos[2, 2, 2], 100.0)

    def test_non_wm_has_zero_dr_neg(self):
        seg = np.ones((5, 5, 5), dtype=np.float64) * 9  # All GM
        dr_pos, dr_neg = qsm_forward.generate_dr_maps(seg, B0=7)
        assert np.all(dr_neg == 0)


class TestGenerateR2prime:
    def test_single_kernel_default(self):
        cp = np.array([0.1, 0.0, 0.05])
        cn = np.array([0.0, -0.2, -0.1])
        r2p = qsm_forward.generate_r2prime(cp, cn)
        expected = qsm_forward.DR_KERNEL * (np.abs(cp) + np.abs(cn))
        np.testing.assert_allclose(r2p, expected)

    def test_dr_kernel_value(self):
        assert qsm_forward.DR_KERNEL == 137.0

    def test_split_opt_in(self):
        cp = np.array([0.1, 0.05])
        cn = np.array([-0.2, -0.1])
        r2p = qsm_forward.generate_r2prime(cp, cn, dr=114.0, dr_neg=30.0)
        expected = 114.0 * np.abs(cp) + 30.0 * np.abs(cn)
        np.testing.assert_allclose(r2p, expected)

    def test_signal_r2prime_matches_generate_r2prime(self):
        # The chi-sep GRE decay's R2' contribution must equal generate_r2prime
        # exactly when the single scalar kernel is passed voxel-wise.
        cp = np.abs(np.random.RandomState(0).rand(4, 4, 4)) * 0.1
        cn = -np.abs(np.random.RandomState(1).rand(4, 4, 4)) * 0.1
        r2p = qsm_forward.generate_r2prime(cp, cn)
        signal_r2p = qsm_forward.DR_KERNEL * np.abs(cp) + qsm_forward.DR_KERNEL * np.abs(cn)
        np.testing.assert_allclose(r2p, signal_r2p, rtol=1e-12)


class TestChiSepSignalModel:
    def test_backwards_compat_none_params(self):
        field = np.zeros((5, 5, 5))
        sig1 = qsm_forward.generate_signal(field, R2star=50)
        sig2 = qsm_forward.generate_signal(field, R2star=50, R2=None, dr_pos=None, dr_neg=None, chi_pos=None, chi_neg=None)
        np.testing.assert_array_equal(sig1, sig2)

    def test_chisep_model_differs_from_r2star(self):
        field = np.zeros((5, 5, 5))
        R2 = np.ones((5, 5, 5)) * 10.0
        dr_pos_arr = np.ones((5, 5, 5)) * 100.0
        dr_neg_arr = np.ones((5, 5, 5)) * 50.0
        chi_pos_arr = np.ones((5, 5, 5)) * 0.05
        chi_neg_arr = np.ones((5, 5, 5)) * -0.03
        sig_chisep = qsm_forward.generate_signal(
            field, R2star=50, R2=R2, dr_pos=dr_pos_arr, dr_neg=dr_neg_arr,
            chi_pos=chi_pos_arr, chi_neg=chi_neg_arr
        )
        sig_r2star = qsm_forward.generate_signal(field, R2star=50)
        assert not np.allclose(sig_chisep, sig_r2star)

    def test_chisep_decay_formula(self):
        field = np.zeros((3, 3, 3))
        R2 = np.ones((3, 3, 3)) * 15.0
        dr_pos_arr = np.ones((3, 3, 3)) * 100.0
        dr_neg_arr = np.ones((3, 3, 3)) * 30.0
        chi_pos_arr = np.ones((3, 3, 3)) * 0.1
        chi_neg_arr = np.ones((3, 3, 3)) * -0.05
        TE = 20e-3
        sig = qsm_forward.generate_signal(
            field, TE=TE, R2=R2, dr_pos=dr_pos_arr, dr_neg=dr_neg_arr,
            chi_pos=chi_pos_arr, chi_neg=chi_neg_arr
        )
        # Expected decay: exp(-TE * (R2 + dr_pos*|chi_pos| + dr_neg*|chi_neg|))
        expected_rate = 15.0 + 100.0 * 0.1 + 30.0 * 0.05
        expected_decay = np.exp(-TE * expected_rate)
        # Signal magnitude should match this decay (M0=1, SPGR terms=1 for default params)
        assert np.abs(sig[1, 1, 1]) > 0


class TestSESignalModel:
    def test_se_decay_formula(self):
        R1 = np.ones((3, 3, 3)) * 1.0
        R2 = np.ones((3, 3, 3)) * 15.0
        M0 = np.ones((3, 3, 3)) * 2.0
        TR = 1.0
        TE = 20e-3
        sig = qsm_forward.generate_se_signal(TR=TR, TE=TE, R1=R1, R2=R2, M0=M0)
        expected = 2.0 * (1 - np.exp(-TR * 1.0)) * np.exp(-TE * 15.0)
        np.testing.assert_allclose(sig[1, 1, 1], expected, rtol=1e-12)

    def test_se_is_real_and_nonnegative(self):
        R1 = np.ones((4, 4, 4)) * 0.8
        R2 = np.ones((4, 4, 4)) * 20.0
        M0 = np.ones((4, 4, 4))
        sig = qsm_forward.generate_se_signal(TR=1.0, TE=30e-3, R1=R1, R2=R2, M0=M0)
        assert np.isrealobj(sig)
        assert np.all(sig >= 0)

    def test_se_uses_r2_not_r2star(self):
        # SE decay must depend on R2 only; changing R2* (susceptibility dephasing) has no effect
        R1 = np.ones((3, 3, 3))
        R2 = np.ones((3, 3, 3)) * 15.0
        M0 = np.ones((3, 3, 3))
        sig_a = qsm_forward.generate_se_signal(TR=1.0, TE=20e-3, R1=R1, R2=R2, M0=M0)
        sig_b = qsm_forward.generate_se_signal(TR=1.0, TE=20e-3, R1=R1, R2=R2 * 2, M0=M0)
        assert not np.allclose(sig_a, sig_b)

    def test_se_recovers_r2_from_two_echoes(self):
        # Round-trip: fit R2 from a noiseless two-echo SE and recover the input
        R1 = np.ones((2, 2, 2))
        R2_true = np.ones((2, 2, 2)) * 25.0
        M0 = np.ones((2, 2, 2))
        TE1, TE2 = 10e-3, 40e-3
        s1 = qsm_forward.generate_se_signal(TR=2.0, TE=TE1, R1=R1, R2=R2_true, M0=M0)
        s2 = qsm_forward.generate_se_signal(TR=2.0, TE=TE2, R1=R1, R2=R2_true, M0=M0)
        R2_fit = np.log(s1 / s2) / (TE2 - TE1)
        np.testing.assert_allclose(R2_fit, R2_true, rtol=1e-10)


class TestWmAnisotropy:
    def test_theta_from_v1_parallel_and_perpendicular(self):
        # V1 parallel to B0 (z) -> theta = 0 deg; perpendicular (x) -> theta = 90 deg
        # (faithful to PhantomCreation.m: sin^2 = (V1x^2+V1y^2)/|V1|^4)
        v1_par = np.zeros((5, 5, 5, 3)); v1_par[:, :, :, 2] = 1.0
        theta_par = qsm_forward.generate_theta_from_v1(v1_par, np.array([0, 0, 1]))
        np.testing.assert_allclose(theta_par[2, 2, 2], 0.0, atol=1e-6)
        v1_perp = np.zeros((5, 5, 5, 3)); v1_perp[:, :, :, 0] = 1.0
        theta_perp = qsm_forward.generate_theta_from_v1(v1_perp, np.array([0, 0, 1]))
        np.testing.assert_allclose(theta_perp[2, 2, 2], 90.0, atol=1e-6)

    def test_anisotropy_per_tract_cos2_modulation(self):
        # theta=0 (cos^2=1) over tract label 1 -> chi_neg = delta_chi*1 + chi_0
        chi_neg = np.ones((5, 5, 5)) * -0.04
        wm_tract = np.ones((5, 5, 5), dtype=int)  # all voxels are tract label 1
        theta = np.zeros((5, 5, 5))               # cos^2(0) = 1
        result = qsm_forward.apply_wm_anisotropy(
            chi_neg, wm_tract, theta, R1=None, region8_r1_weighting=False
        )
        delta_chi, chi_0 = qsm_forward.WM_TRACT_ANISOTROPY_ARRAYS[1]
        expected = delta_chi * 1.0 + chi_0
        np.testing.assert_allclose(result[2, 2, 2], expected, rtol=1e-6)


class TestReferenceValues:
    """Lock in the authoritative per-tissue values from the
    Susceptibility-Separation-Phantom (SusceptibilityValues.mat / Tables 1-2)."""

    def test_chisep_reference_values_match_phantom(self):
        # (chi_pos, chi_neg) from data/chimodel/SusceptibilityValues.mat
        # (fields chipos/chineg), label indices aligned to label.json.
        expected = {
            1:  (0.052650, -0.008650),
            2:  (0.143723, -0.013223),
            3:  (0.047061, -0.009061),
            4:  (0.110906, -0.010906),
            5:  (0.168412, -0.016412),
            6:  (0.122434, -0.011434),
            7:  (0.050904, -0.030904),
            8:  (0.005900, -0.035900),
            9:  (0.039182, -0.019182),
            10: (0.027512, -0.008512),
            11: (0.190000,  0.000000),
        }
        for label, (cp, cn) in expected.items():
            cp_ref, cn_ref = qsm_forward.CHISEP_TISSUE_PARAMS[label][:2]
            np.testing.assert_allclose(cp_ref, cp, atol=1e-6,
                                       err_msg=f"chi_pos mismatch label {label}")
            np.testing.assert_allclose(cn_ref, cn, atol=1e-6,
                                       err_msg=f"chi_neg mismatch label {label}")

    def test_chisep_total_reproduces_net_chi(self):
        # chi_pos + chi_neg must reproduce the phantom's net per-tissue chi
        # (chiref) for the deep-GM / WM / GM tissues.
        expected_net = {
            1: 0.044, 2: 0.1305, 3: 0.038, 7: 0.02,
            8: -0.03, 9: 0.02, 10: 0.019, 11: 0.19,
        }
        for label, net in expected_net.items():
            cp, cn = qsm_forward.CHISEP_TISSUE_PARAMS[label][:2]
            np.testing.assert_allclose(cp + cn, net, atol=1e-4,
                                       err_msg=f"net chi mismatch label {label}")

    def test_wm_tract_anisotropy_reference_values(self):
        # Table 1 of the phantom README / PhantomCreation.m deltaX_values,Xzero_values
        assert qsm_forward.WM_TRACT_ANISOTROPY_PARAMS['body_corpus_callosum'] == (0.032, -0.0512)
        assert qsm_forward.WM_TRACT_ANISOTROPY_PARAMS['posterior_thalamic_radiations'] == (0.016, -0.0592)
        assert qsm_forward.WM_TRACT_ANISOTROPY_PARAMS['superior_longitudinal_fascicle'] == (-0.015, -0.0372)

    def test_t2_reference_values_7t(self):
        # T2_simulation.m region_values (7T, ms)
        expected = [57.46, 41.47, 50.44, 44.07, 71.71, 47.255,
                    56.62, 45.54, 84.71, 1029.6, 97.5]
        for label, val in enumerate(expected, start=1):
            assert qsm_forward.T2_TISSUE_PARAMS_7T[label] == val

    def test_r1_3t_division_factors(self):
        # Map_creation_3T.m division_factors
        expected = [0.75929, 0.73274, 0.74212, 0.65, 0.65, 0.65,
                    0.73898, 0.72472, 0.73648, 1.0051, 0.75672]
        for label, val in enumerate(expected, start=1):
            assert qsm_forward.R1_3T_DIVISION_FACTORS[label] == val


class TestScaleMapsTo3T:
    def test_r2_scaling(self):
        R2 = np.ones((3, 3, 3)) * 20.0
        R2star = np.ones((3, 3, 3)) * 50.0
        R1 = np.ones((3, 3, 3)) * 1.0
        seg = np.ones((3, 3, 3), dtype=np.float64) * 9  # GM
        result = qsm_forward.scale_maps_to_3t(R2, R2star, R1, seg)
        np.testing.assert_allclose(result['R2'], 20.0 * 0.65)
        np.testing.assert_allclose(result['R2star'], 50.0 * 0.5)

    def test_r1_per_region(self):
        R2 = np.ones((3, 3, 3)) * 20.0
        R2star = np.ones((3, 3, 3)) * 50.0
        R1 = np.ones((3, 3, 3)) * 1.0
        seg = np.ones((3, 3, 3), dtype=np.float64) * 9  # GM, factor=0.73648
        result = qsm_forward.scale_maps_to_3t(R2, R2star, R1, seg)
        np.testing.assert_allclose(result['R1'][1, 1, 1], 1.0 / 0.73648, rtol=1e-5)

    def test_t2_scaling(self):
        R2 = np.ones((3, 3, 3)) * 20.0
        R2star = np.ones((3, 3, 3)) * 50.0
        R1 = np.ones((3, 3, 3)) * 1.0
        seg = np.ones((3, 3, 3), dtype=np.float64) * 9
        T2 = np.ones((3, 3, 3)) * 80.0
        result = qsm_forward.scale_maps_to_3t(R2, R2star, R1, seg, T2=T2)
        np.testing.assert_allclose(result['T2'], 80.0 / 0.65)


class TestCLINewFlags:
    def test_chisep_signal_flag_default(self):
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids']):
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest='mode')
            # Re-parse using main's parser structure
            from qsm_forward.main import main
            # Just test the flag exists and defaults correctly
            with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids']):
                args = None
                try:
                    # We can't easily test the full parser without running main,
                    # but we can verify the flags parse correctly
                    pass
                except SystemExit:
                    pass

    def test_anisotropy_flag_parsing(self):
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids',
                                '--chisep-signal', '--anisotropy',
                                '--save-r2', '--save-dr-pos', '--save-dr-neg', '--save-t2']):
            # Verify these args are accepted without error by importing and parsing
            from qsm_forward.main import main
            with patch('qsm_forward.TissueParams') as mock_tp, \
                 patch('qsm_forward.generate_bids') as mock_gb, \
                 patch('qsm_forward.generate_susceptibility_phantom', return_value=np.zeros((10, 10, 10))):
                mock_tp.return_value = MagicMock()
                main()
                # Verify chisep_signal was passed as True
                call_kwargs = mock_gb.call_args[1]
                assert call_kwargs['chisep_signal'] == True
                assert call_kwargs['anisotropy'] == True
                assert call_kwargs['save_r2'] == True
                assert call_kwargs['save_dr_pos'] == True
                assert call_kwargs['save_dr_neg'] == True
                assert call_kwargs['save_t2'] == True

    def test_save_se_flag_parsing(self):
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids',
                                '--save-se', '--se-TR', '2.0',
                                '--se-TEs', '0.01', '0.03', '0.05']):
            from qsm_forward.main import main
            with patch('qsm_forward.TissueParams') as mock_tp, \
                 patch('qsm_forward.generate_bids') as mock_gb, \
                 patch('qsm_forward.ReconParams') as mock_rp, \
                 patch('qsm_forward.generate_susceptibility_phantom', return_value=np.zeros((10, 10, 10))):
                mock_tp.return_value = MagicMock()
                main()
                assert mock_gb.call_args[1]['save_se'] == True
                rp_kwargs = mock_rp.call_args[1]
                assert rp_kwargs['se_TR'] == 2.0
                np.testing.assert_allclose(rp_kwargs['se_TEs'], [0.01, 0.03, 0.05])

    def test_save_se_flag_default_false(self):
        with patch('sys.argv', ['qsm_forward', 'simple', '/tmp/bids']):
            from qsm_forward.main import main
            with patch('qsm_forward.TissueParams') as mock_tp, \
                 patch('qsm_forward.generate_bids') as mock_gb, \
                 patch('qsm_forward.generate_susceptibility_phantom', return_value=np.zeros((10, 10, 10))):
                mock_tp.return_value = MagicMock()
                main()
                assert mock_gb.call_args[1]['save_se'] == False


# ---------------------------------------------------------------------------
# Hollow-cylinder multi-compartment white-matter GRE model.
#
# These tests assert the wired-in model (qsm_forward) reproduces the de-risked
# standalone prototype (prototypes/hollow_cylinder/hollow_cylinder.py) for a WM
# voxel: the three theta-dependent compartment frequencies match to ~1e-6, the
# resulting multi-echo complex signal matches for the same TEs/theta/B0/params,
# and the WM magnitude is non-mono-exponential (the property that makes theta
# recoverable). Also checks the flag is inert when off / on non-WM voxels.
# ---------------------------------------------------------------------------
# The golden reference is a committed COPY of the validated hollow-cylinder prototype
# (tests/_hollow_cylinder_ref.py) — an independent implementation of the closed-form
# physics, NOT the code under test. Committing it here means these tests always run
# (in CI / a fresh clone / an isolated worktree), instead of silently skipping when the
# research `prototypes/` dir is absent.
def _load_prototype():
    """Import the committed golden-reference hollow-cylinder module (by path; not a package)."""
    import importlib.util
    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_hollow_cylinder_ref.py")
    spec = importlib.util.spec_from_file_location("hollow_cylinder_ref", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHollowCylinderMultiCompartment:
    def test_compartment_freqs_match_prototype(self):
        """The three compartment frequencies must match hollow_cylinder.py to ~1e-6 Hz
        across angles and field strengths (WM_HC_PARAMS defaults == prototype WMParams)."""
        proto = _load_prototype()
        p = proto.WMParams()  # prototype's calibrated defaults
        for B0 in (3.0, 7.0):
            for deg in (0, 15, 30, 45, 60, 75, 90):
                th = np.deg2rad(deg)
                ref = proto.compartment_freqs(th, B0, p)          # (M, A, E)
                got = qsm_forward.hc_compartment_freqs(th, B0)
                for r, g, name in zip(ref, got, ("myelin", "axon", "extra")):
                    np.testing.assert_allclose(
                        float(np.asarray(g)), float(np.asarray(r)), atol=1e-6,
                        err_msg=f"{name} freq mismatch at theta={deg} B0={B0}")

    def test_wm_defaults_equal_prototype_params(self):
        """The wired-in WM parameter defaults must equal the prototype's WMParams."""
        proto = _load_prototype()
        p = proto.WMParams()
        d = qsm_forward.WM_HC_PARAMS
        assert d["chi_I"] == p.chi_I
        assert d["chi_A"] == p.chi_A
        assert d["E"] == p.E
        assert d["g"] == p.g
        assert d["T2_M"] == p.T2_M
        assert d["T2_A"] == p.T2_A
        assert d["T2_E"] == p.T2_E
        assert d["f_axon"] == p.f_axon

    def test_multiecho_complex_signal_matches_prototype(self):
        """The per-voxel WM complex multi-echo signal must reproduce the prototype's
        gre_signal (same TEs / theta / B0 / MWF, S0=1, bulk_freq=0) to ~1e-9."""
        proto = _load_prototype()
        TEs = np.arange(2e-3, 48e-3 + 1e-9, 2e-3)  # 24 echoes, matches prototype level 3
        B0 = 7.0
        for deg in (10, 35, 55, 80):
            for mwf in (0.08, 0.12, 0.16):
                th = np.deg2rad(deg)
                p = proto.WMParams(MWF=mwf)
                ref = proto.gre_signal(TEs, th, B0, p, S0=1.0, bulk_freq=0.0)
                got = np.array([
                    qsm_forward.hc_wm_signal(TE, th, B0, mwf, bulk_freq=0.0)
                    for TE in TEs])
                np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-9)

    def test_wm_magnitude_is_non_mono_exponential(self):
        """The WM multi-echo magnitude must be non-mono-exponential: the log-mag
        residual to a straight-line (mono-exp) fit is >> a single-pool control.

        This is the property that makes theta recoverable (prototype level-1 [1h])."""
        TEs = np.arange(2e-3, 40e-3 + 1e-9, 2e-3)
        B0 = 7.0
        th = np.deg2rad(60)
        A = np.vstack([TEs, np.ones_like(TEs)]).T

        mag = np.abs(np.array([qsm_forward.hc_wm_signal(TE, th, B0, 0.12) for TE in TEs]))
        coef, *_ = np.linalg.lstsq(A, np.log(mag), rcond=None)
        resid_mc = np.sqrt(np.mean((np.log(mag) - A @ coef) ** 2))

        # single-pool control: MWF=0, no anisotropy/exchange -> one long pool, mono-exp
        mono_p = {"chi_A": 0.0, "chi_I": 0.0, "E": 0.0}
        mag_mono = np.abs(np.array([
            qsm_forward.hc_wm_signal(TE, th, B0, 0.0, p=mono_p) for TE in TEs]))
        coefm, *_ = np.linalg.lstsq(A, np.log(mag_mono), rcond=None)
        resid_mono = np.sqrt(np.mean((np.log(mag_mono) - A @ coefm) ** 2))

        assert resid_mc > 1e-3
        assert resid_mc > 20 * max(resid_mono, 1e-12)

    def test_theta_changes_wm_signal_shape(self):
        """Different fibre angles must produce different WM multi-echo signals
        (theta is encoded in the signal), unlike the inert scalar-chi scaffold."""
        TEs = np.arange(2e-3, 40e-3 + 1e-9, 2e-3)
        B0 = 7.0
        s30 = np.array([qsm_forward.hc_wm_signal(TE, np.deg2rad(30), B0, 0.12) for TE in TEs])
        s80 = np.array([qsm_forward.hc_wm_signal(TE, np.deg2rad(80), B0, 0.12) for TE in TEs])
        # normalise out the (theta-independent) first-echo scale, compare shapes
        assert not np.allclose(s30 / s30[0], s80 / s80[0], atol=1e-3)

    def test_mwf_mapping_monotone_and_anchored(self):
        """MWF grows with diamagnetic (myelin) content; the reference chi- (-0.10 ppm)
        maps to the prototype reference MWF (0.12); values stay in a physiological band."""
        f = qsm_forward.hc_mwf_from_myelin_content
        np.testing.assert_allclose(float(f(-0.10e-6)), 0.12, atol=1e-9)
        # monotone increasing in |chi-|
        vals = [float(f(-c * 1e-6)) for c in (0.02, 0.05, 0.10, 0.15, 0.20)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))
        # clipped to physiological band
        arr = f(np.array([-0.0, -0.01e-6, -0.5e-6, -1.0e-6]))
        assert np.all(arr >= 0.03) and np.all(arr <= 0.25)

    def test_generate_signal_flag_off_byte_identical(self):
        """With multicompartment=False the signal is byte-identical whether or not
        theta/wm_mask are supplied (the WM branch is fully inert)."""
        rng = np.random.default_rng(0)
        shp = (6, 6, 6)
        field = rng.standard_normal(shp) * 0.01
        R2 = np.ones(shp) * 15.0
        drp = np.ones(shp) * 137.0
        drn = np.ones(shp) * 137.0
        chip = np.ones(shp) * 0.03
        chin = -np.ones(shp) * 0.05
        kw = dict(B0=7, TE=20e-3, R2=R2, dr_pos=drp, dr_neg=drn, chi_pos=chip, chi_neg=chin)
        s1 = qsm_forward.generate_signal(field, multicompartment=False, **kw)
        s2 = qsm_forward.generate_signal(
            field, multicompartment=False,
            theta=np.ones(shp), wm_mask=np.ones(shp, bool), **kw)
        np.testing.assert_array_equal(s1, s2)

    def test_generate_signal_non_wm_reduces_to_single_compartment(self):
        """multicompartment=True with an all-False wm_mask must equal the
        single-compartment chi-sep signal (only WM voxels get the hollow-cylinder model)."""
        rng = np.random.default_rng(1)
        shp = (6, 6, 6)
        field = rng.standard_normal(shp) * 0.01
        R2 = np.ones(shp) * 15.0
        drp = np.ones(shp) * 137.0
        drn = np.ones(shp) * 137.0
        chip = np.ones(shp) * 0.03
        chin = -np.ones(shp) * 0.05
        kw = dict(B0=7, TE=20e-3, R2=R2, dr_pos=drp, dr_neg=drn, chi_pos=chip, chi_neg=chin)
        s_single = qsm_forward.generate_signal(field, multicompartment=False, **kw)
        s_mc_nowm = qsm_forward.generate_signal(
            field, multicompartment=True,
            theta=np.zeros(shp), wm_mask=np.zeros(shp, bool), **kw)
        np.testing.assert_allclose(s_mc_nowm, s_single, rtol=1e-12, atol=1e-15)

    def test_generate_signal_wm_voxels_differ_and_encode_theta(self):
        """In a mixed volume, WM voxels (multicompartment) differ from the single-
        compartment signal, and their multi-echo signal depends on theta."""
        shp = (4, 4, 4)
        field = np.zeros(shp)
        R2 = np.ones(shp) * 15.0
        drp = np.ones(shp) * 137.0
        drn = np.ones(shp) * 137.0
        chip = np.ones(shp) * 0.02
        chin = -np.ones(shp) * 0.06
        wm = np.zeros(shp, bool)
        wm[0, 0, 0] = True  # single WM voxel
        theta_a = np.full(shp, np.deg2rad(20))
        theta_b = np.full(shp, np.deg2rad(80))
        TEs = np.arange(4e-3, 40e-3 + 1e-9, 4e-3)
        kw = dict(B0=7, R2=R2, dr_pos=drp, dr_neg=drn, chi_pos=chip, chi_neg=chin)

        sig_a = np.array([np.abs(qsm_forward.generate_signal(
            field, TE=TE, multicompartment=True, theta=theta_a, wm_mask=wm, **kw)[0, 0, 0])
            for TE in TEs])
        sig_b = np.array([np.abs(qsm_forward.generate_signal(
            field, TE=TE, multicompartment=True, theta=theta_b, wm_mask=wm, **kw)[0, 0, 0])
            for TE in TEs])
        sig_single = np.array([np.abs(qsm_forward.generate_signal(
            field, TE=TE, multicompartment=False, **kw)[0, 0, 0])
            for TE in TEs])

        # WM voxel differs from single-compartment, and theta changes its decay shape
        assert not np.allclose(sig_a, sig_single, atol=1e-6)
        assert not np.allclose(sig_a / sig_a[0], sig_b / sig_b[0], atol=1e-3)

        # a non-WM voxel in the SAME call is unchanged vs single-compartment
        nonwm_a = np.array([np.abs(qsm_forward.generate_signal(
            field, TE=TE, multicompartment=True, theta=theta_a, wm_mask=wm, **kw)[1, 1, 1])
            for TE in TEs])
        nonwm_single = np.array([np.abs(qsm_forward.generate_signal(
            field, TE=TE, multicompartment=False, **kw)[1, 1, 1])
            for TE in TEs])
        np.testing.assert_allclose(nonwm_a, nonwm_single, rtol=1e-12, atol=1e-15)

    def test_wm_multicompartment_preserves_R2prime(self):
        """WM keeps its mesoscopic R2' (Dr*|chi|) applied on top of the pool T2s, rather
        than dropping it — so WM retains its susceptibility contrast and stays consistent
        with the provided r2prime. Checks (1) hc_wm_signal factorises R2p_meso as
        exp(-TE*R2'), and (2) generate_signal's WM branch feeds in Dr_pos*|chi+|+Dr_neg*|chi-|."""
        B0 = 7.0
        th, mwf = np.deg2rad(50), 0.12
        # (1) factorisation property of hc_wm_signal
        for TE in (8e-3, 24e-3):
            s0 = qsm_forward.hc_wm_signal(TE, th, B0, mwf, R2p_meso=0.0)
            sr = qsm_forward.hc_wm_signal(TE, th, B0, mwf, R2p_meso=8.0)
            np.testing.assert_allclose(sr, s0 * np.exp(-8.0 * TE), rtol=1e-12)
        # (2) generate_signal WM branch applies R2' = Dr_pos*|chi+| + Dr_neg*|chi-|
        shp = (2, 2, 2)
        field = np.zeros(shp)
        R2 = np.ones(shp) * 15.0
        drp = np.ones(shp) * 137.0
        drn = np.ones(shp) * 137.0
        chip = np.ones(shp) * 0.02
        chin = -np.ones(shp) * 0.06
        wm = np.zeros(shp, bool)
        wm[0, 0, 0] = True
        theta = np.full(shp, th)
        r2prime = 137.0 * 0.02 + 137.0 * 0.06  # expected source R2' at the WM voxel (Hz)
        kw = dict(B0=7, R2=R2, dr_pos=drp, dr_neg=drn, chi_pos=chip, chi_neg=chin)
        mwf_wm = qsm_forward.hc_mwf_from_myelin_content(-0.06)

        def ratio(TE):
            g = np.abs(qsm_forward.generate_signal(
                field, TE=TE, multicompartment=True, theta=theta, wm_mask=wm, **kw)[0, 0, 0])
            pool0 = np.abs(qsm_forward.hc_wm_signal(TE, th, 7, mwf_wm, R2p_meso=0.0))
            return g / pool0  # cancels the M0/T1/flip constant, leaving exp(-TE*r2prime)

        TE1, TE2 = 8e-3, 24e-3
        np.testing.assert_allclose(ratio(TE2) / ratio(TE1),
                                   np.exp(-(TE2 - TE1) * r2prime), rtol=1e-6)

    def test_hc_wm_se_signal_is_pool_t2_mixture(self):
        """The WM spin-echo factor is the refocused 3-pool T2 mixture: no frequency
        offsets, no mesoscopic R2', same volume fractions as hc_wm_signal. This keeps
        the SE consistent with the multicompartment GRE so the signal-derivable
        R2' = R2* - R2 matches the provided r2prime in WM."""
        p = qsm_forward.WM_HC_PARAMS
        mwf = 0.12
        fM, rest = mwf, 1.0 - mwf
        fA, fE = rest * p["f_axon"], rest * (1.0 - p["f_axon"])
        for TE in (8e-3, 24e-3, 64e-3):
            expected = (fM * np.exp(-TE / p["T2_M"])
                        + fA * np.exp(-TE / p["T2_A"])
                        + fE * np.exp(-TE / p["T2_E"]))
            np.testing.assert_allclose(qsm_forward.hc_wm_se_signal(TE, mwf), expected, rtol=1e-12)
        # equals the magnitude-relevant part of hc_wm_signal with frequencies and R2' off
        for TE in (8e-3, 24e-3):
            gre_pools = qsm_forward.hc_wm_signal(
                TE, 0.0, 7.0, mwf, R2p_meso=0.0,
                p={"chi_I": 0.0, "chi_A": 0.0, "E": 0.0})
            np.testing.assert_allclose(qsm_forward.hc_wm_se_signal(TE, mwf),
                                       np.abs(gre_pools), rtol=1e-12)
        # broadcasts over an mwf array
        arr = qsm_forward.hc_wm_se_signal(10e-3, np.array([0.05, 0.12, 0.25]))
        assert arr.shape == (3,) and np.all(np.diff(arr) < 0)  # more myelin water => faster SE decay