import pytest

import gravyflow as gf
import numpy as np
from keras import ops
from gravyflow.src.dataset.conditioning import conditioning as gf_cond
from gravyflow.src.dataset.conditioning.pearson import rolling_pearson
from gravyflow.tests._helpers import representation_tolerance

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



@pytest.mark.parametrize("window", ["onsource", "offsource"])
def test_whitening_a_non_finite_window_gives_a_non_finite_example(window):
    """
    A NaN in an example's onsource, or in the offsource its PSD is estimated
    from, makes that whole whitened example NaN: it is never silently
    whitened into finite numbers. The batch's other examples are untouched.
    """
    rng = np.random.default_rng(0)
    rate = 1024.0
    onsource = rng.standard_normal((2, 1, int(rate))).astype(np.float32)
    offsource = rng.standard_normal((2, 1, int(4 * rate))).astype(np.float32)
    settings = dict(sample_rate_hertz=rate, fft_duration_seconds=1.0, overlap_duration_seconds=0.5,
                    filter_duration_seconds=1.0)
    clean = np.asarray(gf.whiten(onsource, offsource, **settings))
    (onsource if window == "onsource" else offsource)[0, 0, 100] = np.nan
    whitened = np.asarray(gf.whiten(onsource, offsource, **settings))
    assert np.all(np.isnan(whitened[0]))
    np.testing.assert_array_equal(whitened[1], clean[1])


def test_pearson_correlation_logic():
    """A pulse and the same pulse 10 samples later correlate fully at the lag of 10 samples, exactly."""
    sample_rate, max_lag_seconds, shift = 100.0, 0.2, 10
    t = np.arange(int(sample_rate)) / sample_rate
    pulse = np.exp(-(t - 0.5) ** 2 / 0.01).astype(np.float32)
    strain = ops.convert_to_tensor(np.stack([pulse, np.roll(pulse, shift)])[None])  # (1, 2, T)
    pearson = np.asarray(rolling_pearson(strain, max_lag_seconds=max_lag_seconds, sample_rate_hertz=sample_rate))
    # Index i compares the first detector at t with the second at t + i - M.
    max_lag = int(np.ceil(max_lag_seconds * sample_rate))
    assert pearson.shape == (1, 1, 2 * max_lag + 1)
    assert int(np.argmax(pearson[0, 0])) == max_lag + shift
    # Representation: float32 sums of N - 2M products, bounded by N - 2M roundings.
    np.testing.assert_allclose(pearson[0, 0, max_lag + shift], 1.0, rtol=0.0,
                               atol=representation_tolerance(np.float32, 1.0, t.size - 2 * max_lag))


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
