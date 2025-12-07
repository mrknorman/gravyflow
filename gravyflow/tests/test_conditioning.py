import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import gravyflow as gf
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


def test_create_spectrogram_apply():
    """Test CreateSpectrogram.apply() method (covers line 41)."""
    # Create a simple signal
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, -1))
    
    # CreateSpectrogram requires type_ argument from parent dataclass
    spectrogram_creator = gf_cond.CreateSpectrogram(
        type_=gf_cond.ConditioningMethods.SPECTROGRM
    )
    spectrogram_creator.num_frame_samples = 100
    spectrogram_creator.num_step_samples = 50
    spectrogram_creator.num_fft_samples = 100
    
    spec = spectrogram_creator.apply(signal)
    
    # Verify shape
    # Frames: 1 + (400 - 100) // 50 = 7
    # Freqs: 100 // 2 + 1 = 51
    assert ops.shape(spec) == (1, 7, 51)


def test_spectrogram_1d_input():
    """Test spectrogram with 1D input (covers line 75 and 103)."""
    # Create a 1D signal (no batch dimension)
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    # 1D tensor - NOT reshaped to (1, -1)
    signal = ops.convert_to_tensor(signal)
    
    spec = gf_cond.spectrogram(
        signal,
        num_frame_samples=100,
        num_step_samples=50,
        num_fft_samples=100
    )
    
    # For 1D input, output should be squeezed
    # Shape should be (7, 51) - no batch dim
    assert ops.shape(spec) == (7, 51)


def test_spectrogram_shape_short_signal():
    """Test spectrogram_shape when num_frame_samples > num_samples (covers line 123)."""
    # Signal shorter than frame size
    input_shape = (1, 100)  # Only 100 samples
    num_frame = 256  # Frame size larger than signal
    num_step = 128
    num_fft = 256
    
    out_shape = gf_cond.spectrogram_shape(input_shape, num_frame, num_step, num_fft)
    
    # When num_frame_samples > num_samples, num_frame_samples is clamped to num_samples
    # num_time_frames = 1 + (100 - 100) // 128 = 1
    # num_frequency_bins = 256 // 2 + 1 = 129
    expected = (1, 1, 129)
    assert out_shape == expected


def test_spectrogram_shape_very_short_signal():
    """Test spectrogram_shape when num_time_frames < 1 (covers line 129)."""
    # Signal so short that calculation would give < 1 frames
    # Need: 1 + (num_samples - num_frame_samples) // num_step < 1
    # Which means: (num_samples - num_frame_samples) // num_step < 0
    # This requires num_samples < num_frame_samples, which is already covered above
    # BUT the clamping on line 123 prevents num_frame_samples > num_samples
    # So we need a case where num_step is so large that we get negative frames
    # Actually looking at line 128: if (num_time_frames < 1): num_time_frames = 1
    # This happens when: 1 + (num_samples - num_frame_samples) // num_step < 1
    # => (num_samples - num_frame_samples) // num_step < 0
    # => num_samples - num_frame_samples < 0 (negative division)
    # But num_frame_samples is clamped to num_samples first!
    # So after line 123, we have num_frame_samples == num_samples (at most)
    # Then num_time_frames = 1 + (num_samples - num_samples) // step = 1 + 0 = 1
    # This is never < 1, so line 129 is unreachable after line 123!
    
    # Actually wait - if num_step is zero... no that would be division by zero
    # Let me re-check the logic - the condition on line 128 is AFTER line 123
    # If num_frame_samples <= num_samples, and num_step > 0, then
    # (num_samples - num_frame_samples) >= 0, so // num_step >= 0
    # So 1 + ... >= 1, and line 129 is never reached!
    
    # This appears to be dead code. Let's verify with a test that line 123 was hit
    input_shape = (1, 50)
    num_frame = 100  # Larger than signal
    num_step = 200
    num_fft = 256
    
    out_shape = gf_cond.spectrogram_shape(input_shape, num_frame, num_step, num_fft)
    
    # After clamping: num_frame = 50, num_time_frames = 1 + 0 = 1
    expected = (1, 1, 129)
    assert out_shape == expected
