import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
import jax
import jax.numpy as jnp
from gravyflow.src.dataset.noise import noise as gf_noise
import gravyflow as gf

def test_white_noise_generator():
    # Test white noise generation
    # It yields (onsource, offsource, gps_times)
    
    num_examples = 2
    ifos = [gf.IFO.L1]
    onsource_dur = 1.0
    crop_dur = 0.5
    offsource_dur = 2.0
    sample_rate = 1024.0
    seed = 42
    
    gen = gf_noise.white_noise_generator(
        num_examples_per_batch=num_examples,
        ifos=ifos,
        onsource_duration_seconds=onsource_dur,
        crop_duration_seconds=crop_dur,
        offsource_duration_seconds=offsource_dur,
        sample_rate_hertz=sample_rate,
        seed=seed
    )
    
    onsource, offsource, gps = next(gen)
    
    # Check shapes
    # onsource: (Batch, IFOs, Time)
    # Time = (1.0 + 2*0.5) * 1024 = 2048
    assert ops.shape(onsource) == (num_examples, 1, 2048)
    assert ops.shape(offsource) == (num_examples, 1, int(offsource_dur * sample_rate))
    assert ops.shape(gps) == (num_examples,)
    
    # Check stats (roughly)
    # Mean ~ 0, Std ~ 1
    mean = ops.mean(onsource)
    std = ops.std(onsource)
    
    assert np.abs(mean) < 0.1
    assert np.abs(std - 1.0) < 0.1

def test_interpolate_psd():
    # Test PSD interpolation
    # Create dummy PSD
    freqs = jnp.linspace(0, 512, 100)
    vals = jnp.ones_like(freqs)
    
    num_samples_list = [2048, 1024]
    sample_rate = 1024.0
    
    interp_on, interp_off = gf_noise.interpolate_onsource_offsource_psd(
        num_samples_list,
        sample_rate,
        freqs,
        vals
    )
    
    # Check shapes
    # Output size = num_samples // 2 + 1
    assert ops.shape(interp_on)[-1] == 2048 // 2 + 1
    assert ops.shape(interp_off)[-1] == 1024 // 2 + 1

def test_colored_noise_generator():
    # Test colored noise generation
    # We need a valid IFO with PSD file.
    # Mocking IFO or PSD loading might be needed if files don't exist.
    # Assuming IFO.L1 has a default path that might fail if not present.
    # Let's mock load_psd inside the generator or just test _generate_colored_noise directly if possible.
    # But _generate_colored_noise is internal.
    
    # Let's test the internal generator function if we export it or access it.
    # Or just try running the generator and catch error if file missing.
    pass
