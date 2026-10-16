"""
The chirp-time helpers of the CBC generator in float32.

``calc_minimum_frequency`` and ``calc_duration_from_f_min`` must give finite,
correct values when JAX computes in float32 (64-bit types disabled), the
precision every gravyflow array has outside waveform generation. The reference
is the leading-order chirp relation evaluated in SI units with NumPy float64.
"""
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gravyflow.src.dataset.features.waveforms.cbc import (
    C_SI,
    G_SI,
    M_SOLAR_SI,
    calc_duration_from_f_min,
    calc_minimum_frequency,
)

# Component masses (solar masses) from light neutron stars to heavy black holes.
MASS_1 = np.array([1.2, 1.4, 2.5, 3.0, 10.0, 30.0, 60.0, 100.0])
MASS_2 = np.array([1.2, 1.3, 1.4, 3.0, 5.0, 25.0, 60.0, 80.0])
DURATIONS = np.array([0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0])
FREQUENCIES = np.array([10.0, 16.0, 20.0, 30.0, 50.0, 100.0, 200.0, 400.0])
RTOL = 1e-5


def _chirp_mass_seconds_si(m1, m2):
    m1_kg = np.asarray(m1, dtype=np.float64) * M_SOLAR_SI
    m2_kg = np.asarray(m2, dtype=np.float64) * M_SOLAR_SI
    chirp_mass_kg = (m1_kg * m2_kg) ** 0.6 / (m1_kg + m2_kg) ** 0.2
    return G_SI * chirp_mass_kg / C_SI**3


def _reference_frequency(m1, m2, duration):
    m = _chirp_mass_seconds_si(m1, m2)
    return np.maximum((duration / 5.0) ** (-3.0 / 8.0) / (8.0 * np.pi) * m ** (-5.0 / 8.0), 1.0)


def _reference_duration(m1, m2, frequency):
    m = _chirp_mass_seconds_si(m1, m2)
    return 5.0 * m * (8.0 * np.pi * m * frequency) ** (-8.0 / 3.0)


def _float32(values):
    return jnp.asarray(np.asarray(values, dtype=np.float32))


def test_minimum_frequency_is_correct_in_float32():
    with jax.enable_x64(False):
        f_min = calc_minimum_frequency(_float32(MASS_1), _float32(MASS_2), _float32(DURATIONS))
        assert f_min.dtype == jnp.float32
        f_min = np.asarray(f_min)
    assert np.all(np.isfinite(f_min)), f_min
    np.testing.assert_allclose(f_min, _reference_frequency(MASS_1, MASS_2, DURATIONS), rtol=RTOL)


def test_duration_from_frequency_is_correct_in_float32():
    with jax.enable_x64(False):
        duration = calc_duration_from_f_min(_float32(MASS_1), _float32(MASS_2), _float32(FREQUENCIES))
        assert duration.dtype == jnp.float32
        duration = np.asarray(duration)
    assert np.all(np.isfinite(duration)) and np.all(duration > 0.0), duration
    np.testing.assert_allclose(duration, _reference_duration(MASS_1, MASS_2, FREQUENCIES), rtol=RTOL)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_duration_and_frequency_are_inverse(dtype):
    durations = DURATIONS.astype(dtype)
    with jax.enable_x64(dtype == np.float64):
        f_min = calc_minimum_frequency(MASS_1.astype(dtype), MASS_2.astype(dtype), durations)
        recovered = np.asarray(calc_duration_from_f_min(MASS_1.astype(dtype), MASS_2.astype(dtype), f_min))
    keep = np.asarray(f_min) > 1.0  # the 1 Hz floor breaks the inverse
    np.testing.assert_allclose(recovered[keep], durations[keep], rtol=10 * np.finfo(dtype).eps ** 0.5)


def test_python_floats_are_accepted():
    with jax.enable_x64(False):
        f_min = float(calc_minimum_frequency(30.0, 25.0, 4.0))
        duration = float(calc_duration_from_f_min(30.0, 25.0, 20.0))
    np.testing.assert_allclose(f_min, _reference_frequency(30.0, 25.0, 4.0), rtol=RTOL)
    np.testing.assert_allclose(duration, _reference_duration(30.0, 25.0, 20.0), rtol=RTOL)
