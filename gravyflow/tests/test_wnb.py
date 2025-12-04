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
    
    # Check end-alignment (start should be zero for shorter duration)
    # Duration 0.5s = 50 samples. Max 100.
    # First 50 samples should be zero (or close to zero due to windowing/filtering).
    # Actually, Gaussian noise is masked by `mask = ops.flip(mask)`.
    # Mask is [0...0, 1...1].
    # So first 50 samples of waveform 0 should be exactly zero before filtering?
    # Filtering involves FFT/IFFT which spreads energy.
    # But then we apply envelope which is also end-aligned [0...0, Hann].
    # So first 50 samples should be zero.
    
    w0 = waveforms[0, 0] # Real part
    # First 40 samples should be zero (allowing for some edge effects?)
    # Envelope is 0 for first 50 samples.
    # So result should be 0.
    
    assert np.allclose(w0[:50], 0.0, atol=1e-6)
    
    # Check signal presence in second half
    assert np.max(np.abs(w0[50:])) > 0.0

def test_generate_envelopes():
    num_samples = ops.convert_to_tensor([50])
    max_samples = 100
    
    env = gf_wnb.generate_envelopes(num_samples, max_samples)
    
    assert ops.shape(env) == (1, 100)
    
    # Check end alignment
    assert np.allclose(env[0, :50], 0.0)
    assert env[0, 50] == 0.0 # Hann starts at 0
    assert env[0, 75] > 0.9 # Peak around middle of window
    assert env[0, 99] == 0.0 # Hann ends at 0 (or close)
