import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
from gravyflow.src.dataset.tools import psd as gf_psd

def test_fftfreq():
    n = 10
    d = 0.1
    freqs = gf_psd.fftfreq(n, d)
    expected = np.fft.rfftfreq(n, d)
    np.testing.assert_allclose(freqs, expected)

def test_detrend_constant():
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    detrended = gf_psd.detrend(data, type='constant')
    expected = np.array([[-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(detrended, expected)

def test_detrend_linear():
    # y = 2x + 1 -> [1, 3, 5]
    data = ops.convert_to_tensor([[1.0, 3.0, 5.0]], dtype="float32")
    detrended = gf_psd.detrend(data, type='linear')
    # Should be all zeros
    np.testing.assert_allclose(detrended, np.zeros((1, 3)), atol=1e-5)

def test_psd():
    sample_rate = 100.0
    t = np.linspace(0, 1, int(sample_rate), endpoint=False)
    freq = 10.0
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, -1))
    
    freqs, pxx = gf_psd.psd(signal, nperseg=100, sample_rate_hertz=sample_rate)
    
    # Peak should be at 10Hz
    # pxx is (Batch, Freqs)
    peak_idx = np.argmax(pxx[0])
    peak_freq = freqs[peak_idx]
    assert np.isclose(peak_freq, freq, atol=1.0)