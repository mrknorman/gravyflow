import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import gravyflow as gf
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.tools import psd as gf_psd

def test_fftfreq():
    n = 10
    d = 0.1
    freqs = gf_psd.fftfreq(n, d)
    expected = np.fft.rfftfreq(n, d)
    np.testing.assert_allclose(freqs, expected)

def test_detrend_constant():
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    detrended = gf_psd.detrend(data, type='constant')
    expected = np.array([[-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(detrended, expected)

def test_detrend_linear():
    # y = 2x + 1 -> [1, 3, 5]
    data = ops.convert_to_tensor([[1.0, 3.0, 5.0]], dtype="float32")
    detrended = gf_psd.detrend(data, type='linear')
    # Should be all zeros
    np.testing.assert_allclose(detrended, np.zeros((1, 3)), atol=1e-5)

def test_psd():
    sample_rate = 100.0
    t = np.linspace(0, 1, int(sample_rate), endpoint=False)
    freq = 10.0
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, -1))
    
    freqs, pxx = gf_psd.psd(signal, nperseg=100, sample_rate_hertz=sample_rate)
    
    # Peak should be at 10Hz
    # pxx is (Batch, Freqs)
    peak_idx = np.argmax(pxx[0])
    peak_freq = freqs[peak_idx]
    assert np.isclose(peak_freq, freq, atol=1.0)

def test_psd_vs_scipy():
    import scipy.signal
    
    sample_rate = 1024.0
    duration = 4.0
    nperseg = int(sample_rate) # 1 second window
    noverlap = int(sample_rate / 2) # 50% overlap
    
    # Generate White Noise
    np.random.seed(42)
    noise = np.random.normal(0, 1, int(sample_rate * duration)).astype(np.float32)
    noise_tensor = ops.convert_to_tensor(noise.reshape(1, -1))
    
    # Gravyflow PSD
    gf_freqs, gf_pxx = gf_psd.psd(
        noise_tensor, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        sample_rate_hertz=sample_rate
    )
    
    # Scipy PSD
    scipy_freqs, scipy_pxx = scipy.signal.welch(
        noise, 
        fs=sample_rate, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap,
        scaling='density' # Default for scipy, check gf default
    )
    
    # Compare Frequencies
    np.testing.assert_allclose(gf_freqs, scipy_freqs, atol=1e-5, err_msg="Frequency mismatch")
    
    # Compare PSD values
    # Note: Scipy returns (Freqs,), GF returns (Batch, Freqs)
    # Allow some tolerance for implementation differences (e.g. float32 vs float64)
    # Gravyflow might use a different normalization or window correction factor?
    # Let's check if they are close.
    
    # Check relative error
    rel_error = np.abs(gf_pxx[0] - scipy_pxx) / (scipy_pxx + 1e-10)
    mean_rel_error = np.mean(rel_error)
    
    print(f"Mean Relative Error: {mean_rel_error}")
    
    # Assert mean relative error is small (< 1%)
    assert mean_rel_error < 0.01, f"PSD mismatch. Mean Rel Error: {mean_rel_error}"
    
    # Also check a sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq = 60.0
    sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
    sine_tensor = ops.convert_to_tensor(sine.reshape(1, -1))
    
    gf_freqs_sine, gf_pxx_sine = gf_psd.psd(
        sine_tensor, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        sample_rate_hertz=sample_rate
    )
    
    scipy_freqs_sine, scipy_pxx_sine = scipy.signal.welch(
        sine, 
        fs=sample_rate, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    # Compare peak height
    peak_idx = np.argmax(scipy_pxx_sine)
    gf_peak = gf_pxx_sine[0, peak_idx]
    scipy_peak = scipy_pxx_sine[peak_idx]
    
    assert np.isclose(gf_peak, scipy_peak, rtol=0.05), f"Sine Peak Mismatch: GF={gf_peak}, Scipy={scipy_peak}"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_detrend_invalid_type():
    """Verify detrend raises error for invalid type parameter."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValueError, match="Trend type must be"):
        gf_psd.detrend(data, type='invalid')


def test_detrend_breakpoints_not_supported():
    """Verify detrend raises NotImplementedError for breakpoints."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype="float32")
    
    with pytest.raises(NotImplementedError, match="Breakpoints not yet supported"):
        gf_psd.detrend(data, type='linear', bp=1)


def test_psd_invalid_mode():
    """Verify psd raises error for unsupported mode."""
    signal = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0] * 100], dtype="float32")
    
    with pytest.raises(ValueError, match="Mode not supported"):
        gf_psd.psd(signal, nperseg=100, sample_rate_hertz=100.0, mode='invalid')


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_psd_1d_input():
    """Test PSD with 1D input (not batched)."""
    sample_rate = 100.0
    t = np.linspace(0, 1, int(sample_rate), endpoint=False)
    freq = 10.0
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    signal_1d = ops.convert_to_tensor(signal)  # 1D, no batch dim
    
    freqs, pxx = gf_psd.psd(signal_1d, nperseg=100, sample_rate_hertz=sample_rate)
    
    # Should still work and find peak at 10Hz
    peak_freq = float(freqs[np.argmax(np.array(pxx))])
    assert np.isclose(peak_freq, freq, atol=1.0)


def test_psd_median_mode():
    """Test PSD with median mode instead of mean."""
    sample_rate = 1024.0
    duration = 2.0
    nperseg = 256
    
    # Generate white noise
    np.random.seed(42)
    noise = np.random.normal(0, 1, int(sample_rate * duration)).astype(np.float32)
    noise_tensor = ops.convert_to_tensor(noise.reshape(1, -1))
    
    # Median mode
    freqs_median, pxx_median = gf_psd.psd(
        noise_tensor, nperseg=nperseg, sample_rate_hertz=sample_rate, mode='median'
    )
    
    # Mean mode for comparison
    freqs_mean, pxx_mean = gf_psd.psd(
        noise_tensor, nperseg=nperseg, sample_rate_hertz=sample_rate, mode='mean'
    )
    
    # Both should have same shape
    assert ops.shape(pxx_median) == ops.shape(pxx_mean)
    
    # Both should be positive
    assert float(ops.min(pxx_median)) >= 0
    assert float(ops.min(pxx_mean)) >= 0


def test_detrend_type_aliases():
    """Test detrend accepts 'c' and 'l' type aliases."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    
    # 'c' should work like 'constant'
    detrended_c = gf_psd.detrend(data, type='c')
    detrended_constant = gf_psd.detrend(data, type='constant')
    np.testing.assert_allclose(detrended_c, detrended_constant)
    
    # 'l' should work like 'linear'
    data_linear = ops.convert_to_tensor([[1.0, 3.0, 5.0]], dtype="float32")
    detrended_l = gf_psd.detrend(data_linear, type='l')
    detrended_linear = gf_psd.detrend(data_linear, type='linear')
    np.testing.assert_allclose(detrended_l, detrended_linear, atol=1e-5)


# ============================================================================
# NUMERICAL CORRECTNESS TESTS
# ============================================================================

def test_psd_pure_sine_peak_location():
    """Verify PSD correctly identifies peak frequency for pure sine wave."""
    sample_rate = 1024.0
    duration = 4.0
    test_freqs = [50.0, 100.0, 200.0]
    
    for freq in test_freqs:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
        signal_tensor = ops.convert_to_tensor(signal.reshape(1, -1))
        
        freqs, pxx = gf_psd.psd(
            signal_tensor, 
            nperseg=1024, 
            sample_rate_hertz=sample_rate
        )
        
        # Find peak
        peak_idx = np.argmax(np.array(pxx[0]))
        peak_freq = float(freqs[peak_idx])
        
        # Peak should be at the expected frequency (within 1 Hz)
        assert np.isclose(peak_freq, freq, atol=1.0), f"Expected {freq}Hz, got {peak_freq}Hz"


def test_psd_white_noise_flatness():
    """Verify PSD of white noise is approximately flat."""
    sample_rate = 1024.0
    duration = 8.0
    nperseg = 1024
    
    np.random.seed(42)
    noise = np.random.normal(0, 1, int(sample_rate * duration)).astype(np.float32)
    noise_tensor = ops.convert_to_tensor(noise.reshape(1, -1))
    
    freqs, pxx = gf_psd.psd(noise_tensor, nperseg=nperseg, sample_rate_hertz=sample_rate)
    
    pxx_values = np.array(pxx[0])
    
    # Exclude DC and Nyquist bins
    pxx_middle = pxx_values[5:-5]
    
    # For white noise, variance should be low relative to mean (flat spectrum)
    mean_psd = np.mean(pxx_middle)
    std_psd = np.std(pxx_middle)
    
    # Coefficient of variation should be reasonable (< 1 for reasonably flat)
    cv = std_psd / mean_psd
    assert cv < 1.0, f"PSD of white noise too variable: CV={cv}"


def test_detrend_removes_dc():
    """Verify constant detrending removes DC offset."""
    dc_offset = 5.0
    data = ops.convert_to_tensor([[dc_offset + 1.0, dc_offset + 2.0, dc_offset + 3.0]])
    
    detrended = gf_psd.detrend(data, type='constant')
    
    # Mean should now be approximately zero
    mean = float(ops.mean(detrended))
    assert np.isclose(mean, 0.0, atol=1e-6)


def test_detrend_linear_non_default_axis():
    """Test linear detrend with non-default axis (covers axis movement logic)."""
    # Create 3D data (batch, channels, time) with trend along axis=1
    # Shape: (2, 5, 3) - detrend along axis=1 (middle dim with 5 elements)
    # Each column of axis=1 should have a linear trend
    data = np.zeros((2, 5, 3), dtype=np.float32)
    
    # Add linear trend along axis 1: values = [0, 1, 2, 3, 4] * slope
    for i in range(2):
        for j in range(3):
            data[i, :, j] = np.arange(5) * 2.0 + 1.0  # y = 2x + 1
    
    data_tensor = ops.convert_to_tensor(data)
    
    # Detrend along axis 1 (not the last axis)
    detrended = gf_psd.detrend(data_tensor, type='linear', axis=1)
    
    # Shape should be preserved
    assert ops.shape(detrended) == (2, 5, 3)
    
    # After linear detrend, all values should be near zero
    np.testing.assert_allclose(np.array(detrended), np.zeros((2, 5, 3)), atol=1e-5)