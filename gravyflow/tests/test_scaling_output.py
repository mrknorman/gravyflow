"""
Tests for ScalingTypes (SNR, HPEAK, HRSS) as return variables.

Verifies that ScalingTypes can be requested as output/input variables
and that values are calculated correctly from scaled injections.
"""

import pytest
import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.dataset import create_variable_dictionary


class TestScalingTypesOutput:
    """Tests for ScalingTypes as return variables."""
    
    def test_snr_returned_when_requested(self):
        """SNR should be calculated and returned when ScalingTypes.SNR is in return_variables."""
        # Create mock data
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        # Create mock scaled injections (batch, det, samples)
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        # Create mock offsource (batch, det, samples)
        raw_offsource = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [gf.ScalingTypes.SNR]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=raw_offsource
        )
        
        assert 'SNR' in result
        assert result['SNR'] is not None
        # SNR should be per-batch (batch_size,)
        assert len(result['SNR']) == batch_size
    
    def test_hrss_returned_when_requested(self):
        """HRSS should be calculated and returned when ScalingTypes.HRSS is in return_variables."""
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [gf.ScalingTypes.HRSS]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=None
        )
        
        assert 'HRSS' in result
        assert result['HRSS'] is not None
        # HRSS should be per-batch
        assert len(result['HRSS']) == batch_size
    
    def test_hpeak_returned_when_requested(self):
        """HPEAK should be calculated and returned when ScalingTypes.HPEAK is in return_variables."""
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [gf.ScalingTypes.HPEAK]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=None
        )
        
        assert 'HPEAK' in result
        assert result['HPEAK'] is not None
        # HPEAK should be per-batch
        assert len(result['HPEAK']) == batch_size
    
    def test_multiple_scaling_types_simultaneously(self):
        """Multiple ScalingTypes can be requested at once."""
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        raw_offsource = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [
            gf.ScalingTypes.SNR,
            gf.ScalingTypes.HRSS,
            gf.ScalingTypes.HPEAK
        ]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=raw_offsource
        )
        
        assert 'SNR' in result
        assert 'HRSS' in result
        assert 'HPEAK' in result
    
    def test_scaling_types_with_return_variables(self):
        """ScalingTypes can be mixed with regular ReturnVariables."""
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        onsource = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        raw_offsource = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [
            gf.ReturnVariables.ONSOURCE,
            gf.ScalingTypes.SNR
        ]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=onsource,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=raw_offsource
        )
        
        assert 'ONSOURCE' in result
        assert 'SNR' in result
    
    def test_snr_not_calculated_without_offsource(self):
        """SNR should not be calculated if offsource is not available."""
        batch_size = 4
        num_detectors = 2
        num_samples = 2048
        
        scaled_injections = np.random.randn(batch_size, num_detectors, num_samples).astype(np.float32)
        
        return_variables = [gf.ScalingTypes.SNR]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=scaled_injections,
            sample_rate_hertz=2048.0,
            raw_offsource=None  # No offsource
        )
        
        # SNR should not be in result because offsource is missing
        assert 'SNR' not in result
    
    def test_hrss_calculation_correctness(self):
        """HRSS should equal sqrt(sum(x^2)) for a simple signal."""
        batch_size = 1
        num_samples = 100
        
        # Simple signal with known HRSS
        signal = np.ones((batch_size, num_samples), dtype=np.float32)  # Shape: (1, 100)
        expected_hrss = np.sqrt(num_samples)  # sqrt(100 * 1^2) = 10
        
        return_variables = [gf.ScalingTypes.HRSS]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=signal,
            sample_rate_hertz=2048.0,
            raw_offsource=None
        )
        
        np.testing.assert_allclose(result['HRSS'], expected_hrss, rtol=0.01)
    
    def test_hpeak_calculation_correctness(self):
        """HPEAK should equal max(abs(x)) for a simple signal."""
        batch_size = 1
        num_samples = 100
        
        # Signal with known peak
        signal = np.zeros((batch_size, num_samples), dtype=np.float32)
        signal[0, 50] = 5.0  # Peak at index 50
        expected_hpeak = 5.0
        
        return_variables = [gf.ScalingTypes.HPEAK]
        
        result = create_variable_dictionary(
            return_variables=return_variables,
            onsource=None,
            whitened_onsource=None,
            offsource=None,
            gps_times=None,
            injections=None,
            whitened_injections=None,
            mask=None,
            rolling_pearson_onsource=None,
            spectrogram_onsource=None,
            injection_parameters={},
            scaled_injections=signal,
            sample_rate_hertz=2048.0,
            raw_offsource=None
        )
        
        np.testing.assert_allclose(result['HPEAK'], expected_hpeak, rtol=0.01)
