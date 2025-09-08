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
                    mock_tissue_params.assert_called_once_with(fake_data_dir)
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