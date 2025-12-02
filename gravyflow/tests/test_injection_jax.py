import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.features import injection as gf_inj
import gravyflow as gf

def test_calculate_hrss():
    # Batch=2, Channels=1, Time=100
    # Signal 1: Amplitude 1.0
    # Signal 2: Amplitude 2.0
    
    t = np.ones((100,), dtype=np.float32)
    s1 = t
    s2 = 2.0 * t
    
    injection = np.stack([s1, s2], axis=0) # (2, 100)
    injection = injection[:, np.newaxis, :] # (2, 1, 100)
    injection = ops.convert_to_tensor(injection)
    
    hrss = gf_inj.calculate_hrss(injection)
    
    # Expected: sqrt(sum(x^2))
    # s1: sqrt(100 * 1^2) = 10.0
    # s2: sqrt(100 * 2^2) = 20.0
    
    assert ops.shape(hrss) == (2,)
    np.testing.assert_allclose(hrss, [10.0, 20.0], atol=1e-5)

def test_scale_to_hrss():
    t = np.ones((100,), dtype=np.float32)
    injection = ops.convert_to_tensor(t.reshape(1, 1, 100))
    
    target_hrss = 5.0
    scaled = gf_inj.scale_to_hrss(injection, target_hrss)
    
    new_hrss = gf_inj.calculate_hrss(scaled)
    np.testing.assert_allclose(new_hrss, [target_hrss], atol=1e-5)

def test_ensure_last_dim_even():
    # Odd length
    x = ops.ones((1, 5))
    y = gf_inj.ensure_last_dim_even(x)
    assert ops.shape(y) == (1, 4)
    
    # Even length
    x = ops.ones((1, 4))
    y = gf_inj.ensure_last_dim_even(x)
    assert ops.shape(y) == (1, 4)

def test_wnb_generator():
    # Test WNBGenerator class
    gen = gf_inj.WNBGenerator(
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        min_frequency_hertz=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        max_frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT)
    )
    
    waveforms, params = gen.generate(
        num_waveforms=2,
        sample_rate_hertz=100.0,
        max_duration_seconds=1.0,
        seed=42
    )
    
    assert ops.shape(waveforms) == (2, 2, 100)
    assert "duration_seconds" in params

def test_ripple_generator():
    # Test RippleGenerator class
    # Needs valid approximant and parameters
    gen = gf_inj.RippleGenerator(
        mass_1_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
        mass_2_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
        distance_mpc=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT)
    )
    
    waveforms, params = gen.generate(
        num_waveforms=2,
        sample_rate_hertz=1024.0,
        duration_seconds=1.0,
        _=0
    )
    
    # Ripple returns (Batch, 2, Time)
    # Time = 1.0 * 1024 = 1024
    assert ops.shape(waveforms) == (2, 2, 1024)
    assert "mass_1_msun" in params
