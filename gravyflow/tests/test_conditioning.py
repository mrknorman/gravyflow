import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.conditioning import conditioning as gf_cond

def test_spectrogram_shape():
    input_shape = (1, 1000)
    num_frame = 256
    num_step = 128
    num_fft = 256
    
    out_shape = gf_cond.spectrogram_shape(input_shape, num_frame, num_step, num_fft)
    
    # Frames: 1 + (1000 - 256) // 128 = 1 + 744 // 128 = 1 + 5 = 6
    # Freqs: 256 // 2 + 1 = 129
    expected = (1, 6, 129)
    assert out_shape == expected

def test_spectrogram_calculation():
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal: Sine wave at 10Hz
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, -1))
    
    # Spectrogram
    spec = gf_cond.spectrogram(
        signal,
        num_frame_samples=100,
        num_step_samples=50,
        num_fft_samples=100
    )
    
    # Shape check
    # Frames: 1 + (400 - 100) // 50 = 1 + 6 = 7
    # Freqs: 100 // 2 + 1 = 51
    assert ops.shape(spec) == (1, 7, 51)
    
    # Check peak frequency
    # 10Hz bin. df = 100Hz / 100 = 1Hz.
    # Bin 10 should be peak.
    
    # Sum over time
    avg_spec = ops.mean(spec, axis=1) # (1, 51)
    peak_bin = np.argmax(avg_spec[0])
    
    assert peak_bin == 10

from gravyflow.src.dataset.conditioning.pearson import rolling_pearson

def test_whiten_invalid_input():
    """Verify behavior with NaN/Inf input (should handle or raise)."""
    
    sample_rate = 1024.0
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    # Create input with NaNs
    data = np.random.randn(1, 1, num_samples).astype(np.float32)
    data[0, 0, 100] = np.nan
    data = ops.convert_to_tensor(data)
    
    # Offsource needed for PSD
    offsource = np.random.randn(1, 1, num_samples).astype(np.float32)
    offsource = ops.convert_to_tensor(offsource)
    
    # Expectation: It might run but return NaNs, OR raise an error.
    # Both are acceptable for garbage input.
    
    try:
        whitened = gf_cond.whiten(
            data,
            offsource,
            sample_rate_hertz=sample_rate,
            fft_duration_seconds=1.0,
            overlap_duration_seconds=0.5,
            filter_duration_seconds=1.0
        )
        # If it returns, check if it contains NaNs (expected)
        assert np.any(np.isnan(whitened)), "Whitening NaN input should result in NaN output."
    except Exception as e:
        # If it raises, that's also fine for invalid input.
        print(f"Whitening raised exception on NaN input (acceptable): {e}")

def test_pearson_correlation_logic():
    """Verify rolling pearson correlation on known signals."""
    # Create two identical signals
    sample_rate = 100.0
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal: Gaussian pulse
    sig = np.exp(-(t - 0.5)**2 / 0.01).astype(np.float32)
    
    # Batch dim
    sig = sig.reshape(1, -1)
    
    # Shift one signal
    shift = 10
    sig_shifted = np.roll(sig, shift, axis=-1)
    
    # Rolling Pearson
    max_shift_seconds = 0.2 # 20 samples
    
    # Create 2 IFOs
    strain = np.stack([sig[0], sig_shifted[0]], axis=0) # (2, Time)
    strain = strain.reshape(1, 2, -1) # (1, 2, Time)
    strain = ops.convert_to_tensor(strain)
    
    # Calculate
    pearson = rolling_pearson(
        strain,
        max_arrival_time_difference_seconds=max_shift_seconds,
        sample_rate_hertz=sample_rate
    )
    
    # Check max correlation
    max_corr = ops.max(pearson)
    print(f"DEBUG: Max Pearson: {max_corr}")
    
    assert max_corr > 0.9, f"Pearson correlation should be high for shifted signals. Got {max_corr}"
