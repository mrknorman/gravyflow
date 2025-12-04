import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
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