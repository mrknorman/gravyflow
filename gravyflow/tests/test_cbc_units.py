"""
The chirp-time helpers of the CBC generator.

``calc_minimum_frequency`` and ``calc_duration_from_f_min`` are the leading-order
(Newtonian) chirp relation, solved for the frequency and for the time to merger.
They must give finite values in float32 (64-bit types disabled), the precision
every gravyflow array has outside waveform generation, equal to their float64
values to float32's precision.
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gravyflow.src.dataset.features.waveforms.cbc import (
    calc_duration_from_f_min,
    calc_minimum_frequency,
)

# Component masses (solar masses) from light neutron stars to heavy black holes.
MASS_1 = np.array([1.2, 1.4, 2.5, 3.0, 10.0, 30.0, 60.0, 100.0])
MASS_2 = np.array([1.2, 1.3, 1.4, 3.0, 5.0, 25.0, 60.0, 80.0])
DURATIONS = np.array([0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0])
FREQUENCIES = np.array([10.0, 16.0, 20.0, 30.0, 50.0, 100.0, 200.0, 400.0])

# Representation tolerances, c * eps * |value|. Each helper is a chirp mass
# (two powers, a product, a sum, a quotient) and one further power with
# exponent -8/3 or 3/8. A relative error e in the chirp mass becomes 5/3 e in the
# duration and 5/8 e in the frequency, and each operation rounds by up to half an
# ulp: about 4 ulp for the duration and 2 for the frequency, doubled for margin.
DURATION_ULPS = 8
FREQUENCY_ULPS = 4
# The round trip: a frequency error e becomes 8/3 e in the recovered duration,
# on top of that duration's own rounding.
ROUND_TRIP_ULPS = int(np.ceil(8.0 / 3.0 * FREQUENCY_ULPS)) + DURATION_ULPS


def _float32(values):
    return jnp.asarray(np.asarray(values, dtype=np.float32))


def _float64_values():
    with jax.enable_x64(True):
        f_min = np.asarray(calc_minimum_frequency(MASS_1, MASS_2, DURATIONS))
        duration = np.asarray(calc_duration_from_f_min(MASS_1, MASS_2, FREQUENCIES))
    assert f_min.dtype == np.float64 and duration.dtype == np.float64
    return f_min, duration


def test_float32_values_are_the_float64_values_to_float32_precision():
    with jax.enable_x64(False):
        f_min = calc_minimum_frequency(_float32(MASS_1), _float32(MASS_2), _float32(DURATIONS))
        duration = calc_duration_from_f_min(_float32(MASS_1), _float32(MASS_2), _float32(FREQUENCIES))
        assert f_min.dtype == jnp.float32 and duration.dtype == jnp.float32
        f_min, duration = np.asarray(f_min), np.asarray(duration)
    assert np.all(np.isfinite(f_min)) and np.all(np.isfinite(duration)), (f_min, duration)

    eps = np.finfo(np.float32).eps
    reference_f_min, reference_duration = _float64_values()
    np.testing.assert_allclose(f_min, reference_f_min, rtol=FREQUENCY_ULPS * eps)
    np.testing.assert_allclose(duration, reference_duration, rtol=DURATION_ULPS * eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_duration_and_frequency_are_inverse(dtype):
    durations = DURATIONS.astype(dtype)
    with jax.enable_x64(dtype == np.float64):
        f_min = calc_minimum_frequency(MASS_1.astype(dtype), MASS_2.astype(dtype), durations)
        recovered = np.asarray(calc_duration_from_f_min(MASS_1.astype(dtype), MASS_2.astype(dtype), f_min))
    assert recovered.dtype == dtype
    assert np.all(np.asarray(f_min) > 1.0), f_min  # the 1 Hz floor would break the inverse
    np.testing.assert_allclose(recovered, durations, rtol=ROUND_TRIP_ULPS * np.finfo(dtype).eps)


def test_python_floats_are_accepted():
    with jax.enable_x64(False):
        f_min = calc_minimum_frequency(30.0, 25.0, 4.0)
        duration = calc_duration_from_f_min(30.0, 25.0, 20.0)
    with jax.enable_x64(True):
        reference_f_min = float(calc_minimum_frequency(30.0, 25.0, 4.0))
        reference_duration = float(calc_duration_from_f_min(30.0, 25.0, 20.0))
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(float(f_min), reference_f_min, rtol=FREQUENCY_ULPS * eps)
    np.testing.assert_allclose(float(duration), reference_duration, rtol=DURATION_ULPS * eps)
