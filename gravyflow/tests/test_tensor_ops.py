import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import gravyflow as gf
import jax
import numpy as np
import keras
from keras import ops
from gravyflow.src.utils import tensor as gf_tensor

def test_replace_nan_and_inf_with_zero_reports_what_it_replaced():
    x = ops.convert_to_tensor([1.0, np.nan, np.inf, -np.inf, 2.0])
    with pytest.warns(gf.NonFiniteWarning, match="onsource: 3 of 5 values were NaN or infinite"):
        y = gf_tensor.replace_nan_and_inf_with_zero(x, "onsource")
    np.testing.assert_allclose(y, np.array([1.0, 0.0, 0.0, 0.0, 2.0]))


def test_replace_nan_and_inf_with_zero_is_silent_on_finite_data():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", gf.NonFiniteWarning)
        y = gf_tensor.replace_nan_and_inf_with_zero(ops.convert_to_tensor([1.0, 2.0]))
    np.testing.assert_allclose(y, [1.0, 2.0])

def test_the_session_fails_on_non_finite_data_past_acquisition(request):
    # The filter is added by conftest.py, not the ini (see the root conftest.py).
    assert any(line.startswith("error::gravyflow.src.utils.numerics.NonFiniteWarning:")
               for line in request.config.getini("filterwarnings"))


@pytest.mark.parametrize("transform", [jax.jit, jax.vmap], ids=["jit", "vmap"])
def test_replace_nan_and_inf_with_zero_refuses_traced_input_by_name(transform):
    # It counts what it replaces, so it runs eagerly; traced code is refused.
    x = ops.convert_to_tensor([[1.0, np.nan], [np.inf, 2.0]])
    with pytest.raises(TypeError, match="replace_nan_and_inf_with_zero runs eagerly"):
        transform(lambda t: gf.replace_nan_and_inf_with_zero(t, "traced"))(x)


def test_replace_nan_and_inf_with_zero_differentiates_eagerly():
    # Under an eager jax.grad the values, and so the count, are concrete.
    x = ops.convert_to_tensor([1.0, np.nan, 2.0])
    with pytest.warns(gf.NonFiniteWarning, match="g: 1 of 3"):
        grad = jax.grad(lambda t: gf.replace_nan_and_inf_with_zero(t, "g").sum())(x)
    np.testing.assert_allclose(grad, [1.0, 0.0, 1.0])

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
