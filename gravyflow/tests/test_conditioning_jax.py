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
