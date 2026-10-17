"""Tests for gravyflow config module."""
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.config import Defaults, WindowSpec, batch_size
from gravyflow.src.dataset.dataset import GravyflowDataset
from gravyflow.src.dataset.tools.psd import SpectralConfig
from gravyflow.src.utils.numerics import BatchConfig


def test_defaults_set_valid_attribute():
    """Test Defaults.set() with valid attributes."""
    # Store original value
    original_seed = Defaults.seed

    # Set new value
    Defaults.set(seed=12345)
    assert Defaults.seed == 12345

    # Restore original
    Defaults.set(seed=original_seed)


def test_defaults_set_multiple_attributes():
    """Test Defaults.set() with multiple valid attributes."""
    original_seed = Defaults.seed
    original_rate = Defaults.sample_rate_hertz

    Defaults.set(seed=9999, sample_rate_hertz=4096.0)
    assert Defaults.seed == 9999
    assert Defaults.sample_rate_hertz == 4096.0

    # Restore
    Defaults.set(seed=original_seed, sample_rate_hertz=original_rate)


def test_defaults_set_invalid_attribute():
    """Test Defaults.set() raises AttributeError for invalid attribute."""
    with pytest.raises(AttributeError, match="has no attribute named"):
        Defaults.set(nonexistent_attribute=42)


# =============================================================================
# A configuration that can serve nothing is refused by name, as it is built
# =============================================================================

WINDOW = dict(sample_rate_hertz=1024.0, onsource_duration_seconds=1.0,
              offsource_duration_seconds=2.0, crop_duration_seconds=0.5)


def white_dataset(**overrides):
    arguments = dict(
        **WINDOW, seed=42, num_examples_per_batch=4, steps_per_epoch=10, group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        input_variables=[gf.ReturnVariables.ONSOURCE], output_variables=[gf.ReturnVariables.OFFSOURCE],
    )
    arguments.update(overrides)
    return GravyflowDataset(**arguments)


@pytest.mark.parametrize("name, value", [
    ("sample_rate_hertz", -100.0),
    ("onsource_duration_seconds", -1.0),
    ("num_examples_per_batch", 0),
])
def test_a_dataset_that_can_serve_nothing_is_refused_by_name(name, value):
    """A non-positive rate, duration or batch size is refused as the dataset is built, naming it."""
    with pytest.raises(ValueError, match=name):
        white_dataset(**{name: value})


@pytest.mark.parametrize("name, value", [
    ("sample_rate_hertz", 0.0), ("sample_rate_hertz", -2048.0),
    ("sample_rate_hertz", float("nan")), ("sample_rate_hertz", float("inf")),
    ("onsource_duration_seconds", 0.0), ("onsource_duration_seconds", -1.0),
    ("offsource_duration_seconds", -1.0), ("offsource_duration_seconds", float("nan")),
    ("crop_duration_seconds", -0.5), ("crop_duration_seconds", float("inf")),
])
def test_a_window_that_holds_nothing_is_refused_by_name(name, value):
    with pytest.raises(ValueError, match=name):
        WindowSpec(**{**WINDOW, name: value})
    with pytest.raises(ValueError, match=name):
        WindowSpec.from_params(**{name: value})


def test_a_window_without_offsource_or_crop_is_allowed():
    spec = WindowSpec(**{**WINDOW, "offsource_duration_seconds": 0.0, "crop_duration_seconds": 0.0})
    assert spec.sample_counts == (1024, 0)


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "4"])
def test_a_batch_that_holds_no_example_is_refused_by_name(value):
    with pytest.raises(ValueError, match="num_examples_per_batch"):
        batch_size(value)
    with pytest.raises(ValueError, match="num_examples_per_batch"):
        BatchConfig(num_examples_per_batch=value)


def test_the_batch_size_defaults_and_takes_any_integer():
    assert batch_size(None) == Defaults.num_examples_per_batch
    assert batch_size(np.int64(8)) == 8 and type(batch_size(np.int64(8))) is int


@pytest.mark.parametrize("name, value", [
    ("sample_rate_hertz", -1.0), ("onsource_duration_seconds", 0.0),
    ("offsource_duration_seconds", -1.0), ("crop_duration_seconds", -1.0),
    ("num_examples_per_batch", 0), ("num_examples_per_generation_batch", 0), ("memory_bytes", 0),
])
def test_a_default_that_can_serve_nothing_is_refused_and_nothing_changes(name, value):
    before = {key: getattr(Defaults, key) for key in ("seed", name)}
    with pytest.raises(ValueError, match=name):
        Defaults.set(seed=before["seed"] + 1, **{name: value})
    assert {key: getattr(Defaults, key) for key in before} == before


@pytest.mark.parametrize("name, value", [
    ("fft_duration_seconds", 0.0), ("overlap_duration_seconds", -0.1),
    ("filter_duration_seconds", 0.0), ("lower_frequency_hertz", -1.0),
    ("upper_frequency_hertz", 10.0), ("psd_floor", 1.0), ("psd_method", "max"),
])
def test_a_spectral_configuration_out_of_range_is_refused_by_name(name, value):
    with pytest.raises(ValueError, match=name):
        SpectralConfig(**{name: value})


def test_a_band_at_a_rate_with_no_samples_is_refused_by_name():
    with pytest.raises(ValueError, match="sample_rate_hertz"):
        SpectralConfig().validate_band(-100.0)
