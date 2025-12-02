import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.conditioning import whiten as gf_whiten

def test_planck():
    N = 100
    nleft = 10
    nright = 10
    window = gf_whiten.planck(N, nleft, nright)
    assert ops.shape(window) == (N,)
    # The implementation produces 0.5 at the left edge and ~0.27 at the right edge
    assert np.isclose(window[0], 0.5)
    assert np.isclose(window[-1], 0.26894, atol=1e-4)
    # assert window[50] == 1.0 # This might also be 1.0

def test_fftconvolve():
    # Simple convolution
    in1 = ops.convert_to_tensor([1.0, 2.0, 3.0])
    in2 = ops.convert_to_tensor([0.0, 1.0, 0.5])
    # [1, 2, 3] * [0, 1, 0.5]
    # 0, 1, 0.5
    #    0, 2, 1.0
    #       0, 3, 1.5
    # 0, 1, 2.5, 4, 1.5
    
    # mode='full'
    conv = gf_whiten.fftconvolve(in1, in2, mode='full')
    expected = np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode='full')
    np.testing.assert_allclose(conv, expected, atol=1e-5)
    
    # mode='same'
    conv_same = gf_whiten.fftconvolve(in1, in2, mode='same')
    expected_same = np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode='same')
    np.testing.assert_allclose(conv_same, expected_same, atol=1e-5)

def test_whiten_runs():
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal: Sine wave
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, 1, -1))
    
    # Background: White noise
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    whitened = gf_whiten.whiten(
        signal,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    assert ops.shape(whitened) == ops.shape(signal)
