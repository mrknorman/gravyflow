import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import gravyflow as gf
import numpy as np
import keras
from keras import ops
from gravyflow.src.utils import tensor as gf_tensor

def test_replace_nan_and_inf_with_zero():
    x = ops.convert_to_tensor([1.0, np.nan, np.inf, -np.inf, 2.0])
    y = gf_tensor.replace_nan_and_inf_with_zero(x)
    expected = np.array([1.0, 0.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(y, expected)

def test_expand_tensor():
    mask = ops.convert_to_tensor([True, False, True])
    signal = ops.convert_to_tensor([1.0, 2.0])
    expanded = gf_tensor.expand_tensor(signal, mask, group_size=1)
    expected = np.array([1.0, 0.0, 2.0])
    np.testing.assert_allclose(expanded, expected)

def test_batch_tensor():
    x = ops.arange(10)
    batched = gf_tensor.batch_tensor(x, batch_size=3)
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    np.testing.assert_allclose(batched, expected)

def test_crop_samples():
    x = ops.reshape(ops.arange(100, dtype="float32"), (1, 100))
    cropped = gf_tensor.crop_samples(x, onsource_duration_seconds=1.0, sample_rate_hertz=10.0)
    expected = np.arange(45, 55, dtype=np.float32).reshape(1, 10)
    np.testing.assert_allclose(cropped, expected)

def test_rfftfreq():
    n = 10
    freqs = gf_tensor.rfftfreq(n, frequency_interval_hertz=1.0)
    expected = np.fft.rfftfreq(n, d=1.0)
    np.testing.assert_allclose(freqs, expected)

def test_pad_if_odd():
    x = ops.convert_to_tensor([1, 2, 3]) 
    padded = gf_tensor.pad_if_odd(x)
    expected = np.array([1, 2, 3, 0])
    np.testing.assert_array_equal(padded, expected)
    
    x_even = ops.convert_to_tensor([1, 2])
    padded_even = gf_tensor.pad_if_odd(x_even)
    np.testing.assert_array_equal(padded_even, x_even)

def test_round_to_even():
    x = ops.convert_to_tensor([1, 2, 3, 4])
    rounded = gf_tensor.round_to_even(x)
    expected = np.array([0, 2, 2, 4])
    np.testing.assert_array_equal(rounded, expected)

def test_pad_to_power_of_two():
    x = ops.convert_to_tensor([1, 2, 3]) 
    padded = gf_tensor.pad_to_power_of_two(x)
    expected = np.array([1, 2, 3, 0])
    np.testing.assert_array_equal(padded, expected)

def test_resample():
    t = np.linspace(0, 1, 100, endpoint=False)
    x = ops.convert_to_tensor(np.sin(2 * np.pi * 5 * t), dtype="float32") 
    resampled = gf_tensor.resample(x, original_size=100, original_sample_rate_hertz=100.0, new_sample_rate_hertz=50.0)
    
    # Check shape
    assert ops.shape(resampled)[0] == 50
