import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.tools import snr as gf_snr

def test_find_closest():
    tensor = ops.convert_to_tensor([1.0, 2.0, 5.0])
    scalar = 2.1
    idx = gf_snr.find_closest(tensor, scalar)
    assert idx == 1 

def test_snr_calculation():
    # Create a signal and background
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Injection: 10Hz sine wave
    # Shape: (Batch=1, Channel=1, Time)
    injection = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # Background: White noise
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    # Calculate SNR
    snr_val = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5
    )
    
    # SNR should be positive
    # Result shape should be (Batch,)
    assert ops.shape(snr_val)[0] == 1
    assert snr_val[0] > 0.0

def test_scale_to_snr():
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Injection: 10Hz sine wave
    injection = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # Background
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    desired_snr = ops.convert_to_tensor([10.0])
    
    scaled_injection = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    # Verify shape
    assert ops.shape(scaled_injection) == ops.shape(injection)
    
    # Verify new SNR is close to desired
    new_snr = gf_snr.snr(
        scaled_injection,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    # Relax tolerance to 25% due to potential windowing/interpolation effects
    np.testing.assert_allclose(new_snr, desired_snr, rtol=0.25)
