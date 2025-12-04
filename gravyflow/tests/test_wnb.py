import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.features.waveforms import wnb as gf_wnb

def test_wnb_generation():
    num_waveforms = 2
    sample_rate = 100.0
    max_duration = 1.0
    duration = ops.convert_to_tensor([0.5, 0.8])
    min_freq = ops.convert_to_tensor([10.0, 20.0])
    max_freq = ops.convert_to_tensor([40.0, 45.0])
    seed = 42
    
    waveforms = gf_wnb.wnb(
        num_waveforms,
        sample_rate,
        max_duration,
        duration,
        min_freq,
        max_freq,
        seed
    )
    
    # Shape: (Batch, 2, Time)
    # Time = 1.0 * 100 = 100 samples
    assert ops.shape(waveforms) == (2, 2, 100)
    
    # Check that it's not all zeros
    assert ops.max(ops.abs(waveforms)) > 0.0
    
    # Check centered alignment
    # Duration 0.5s = 50 samples. Max 100.
    # Start index = (100 - 50) // 2 = 25.
    # Signal should be present in [25, 75).
    # Zeros in [0, 25) and [75, 100).
    
    w0 = waveforms[0, 0] # Real part
    
    # Check zeros at start
    assert np.allclose(w0[:25], 0.0, atol=1e-6)
    
    # Check zeros at end
    assert np.allclose(w0[75:], 0.0, atol=1e-6)
    
    # Check signal presence in middle
    assert np.max(np.abs(w0[25:75])) > 0.0

def test_generate_envelopes():
    num_samples = ops.convert_to_tensor([50])
    max_samples = 100
    
    env = gf_wnb.generate_envelopes(num_samples, max_samples)
    
    assert ops.shape(env) == (1, 100)
    
    # Check centered alignment
    # Start index = 25
    assert np.allclose(env[0, :25], 0.0)
    assert np.allclose(env[0, 75:], 0.0)
    
    # Peak should be around center (index 50)
    assert env[0, 50] > 0.9 
    
    # Edges of the window should be near zero (Hann window property)
    # env[0, 25] and env[0, 74] should be small but non-zero?
    # Hann starts at 0.
    assert env[0, 25] == 0.0
    assert env[0, 74] < 0.1 # Just check it's small at the very edge
