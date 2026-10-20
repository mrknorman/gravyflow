"""
Wrapping slides in the validator's stream mode, end to end on synthetic noise of
two detectors: a per-detector model (`PerDetectorModel`) scored once per
detector, its background every slide's coincidences, its injections scored at
zero lag through the same combination; and a model that sees both detectors at
once, run on every slide within the background budget.
"""

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import PerDetectorModel, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.metrics import MIN_BACKGROUND_EVENTS

RATE = 512.0
SAMPLES = int(RATE)
YEAR = 365.25 * 86400.0


def logit_model(detectors: int, seed: int = 1) -> keras.Model:
    inputs = keras.Input(shape=(detectors, SAMPLES), name="WHITENED_ONSOURCE")
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(2, name="output", kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


@pytest.fixture(scope="module")
def dataset_args():
    return validation_dataset_args(config=ValidationDatasetConfig(
        sample_rate_hertz=RATE, offsource_duration_seconds=2.0, ifos=[gf.IFO.H1, gf.IFO.L1]))


def wrapped_config(**extra):
    return ValidationConfig(**{"num_examples": 1500, "batch_size": 64, "snr_range": (0.0, 15.0),
                               "volume_num_injections": 16, "stream_segment_seconds": 64.0,
                               "wrapping_slides": True, **extra})


def test_wrapping_slides_are_a_stream_mode_choice():
    assert ValidationConfig().wrapping_slides is False
    with pytest.raises(ValueError, match="stream mode"):
        ValidationConfig(mode="window", wrapping_slides=True)
    with pytest.raises(ValueError):
        ValidationConfig(max_wrapping_slides=0)


def test_a_per_detector_model_is_its_detectors_scores_combined():
    model = PerDetectorModel(logit_model(1), "max")
    x = {"WHITENED_ONSOURCE": np.random.default_rng(0).standard_normal((5, 2, SAMPLES)).astype(np.float32)}
    each = model.detector_scores(x)
    assert each.shape == (2, 5)
    np.testing.assert_allclose(model.predict_on_batch(x), each.max(axis=0))
    assert model.layers == model.model.layers and model.name == model.model.name
    with pytest.raises(ValueError, match="combine"):
        PerDetectorModel(logit_model(1), "product")


@pytest.fixture(scope="module")
def validated(tmp_path_factory, dataset_args):
    path = tmp_path_factory.mktemp("wired") / "validation.h5"
    model = PerDetectorModel(logit_model(1), "sum")
    return path, Validator.validate(model, "per-detector", dataset_args, wrapped_config(), path)


def test_a_per_detector_model_s_wrapping_slides_end_to_end(validated, dataset_args):
    path, v = validated
    streams = v.noise_streams
    ring = float(np.sum(streams["lengths"]) * streams["stride_seconds"])
    num_slides = streams["ring_lags_seconds"].size
    background = v.background()
    # Every slide the whole ring, less a cluster either side of its join.
    assert background.livetime_seconds == pytest.approx(num_slides * (ring - 2 * v.config.stream_cluster_seconds))
    # Each detector's data scored once: the record holds each detector's scores, not each slide's.
    assert v.far_scores.size == 2 * np.sum(streams["lengths"])
    assert num_slides > 50
    curve = background.far_curve()
    depth = curve.rate_hz[curve.quotable].min()
    print(f"\nwrapping slides: {num_slides} slides of a {ring:g} s ring, background "
          f"{background.livetime_seconds:g} s ({background.livetime_seconds / YEAR * 365.25:.2f} days); "
          f"quotable depth {depth:.3g} Hz (1 per {1 / depth / 86400:.3g} days), "
          f"{curve.independent_events[curve.quotable].min()} independent events there; "
          f"one slide's ring alone would quote no lower than {MIN_BACKGROUND_EVENTS / ring:.3g} Hz")
    assert depth < MIN_BACKGROUND_EVENTS / ring  # deeper than the data reused once could quote
    # The old rule (count / slides) caps the depth near 10 / ring.
    assert depth < 0.5 * MIN_BACKGROUND_EVENTS / ring
    # Injections at zero lag, through the same combination.
    data = v.efficiency_data
    assert data["scores"].size == v.config.num_examples
    assert np.all(np.isfinite(data["scores"]) | (data["scores"] == -np.inf))
    loaded = Validator.load(path)
    reloaded = loaded.background()
    np.testing.assert_array_equal(reloaded.window_scores, background.window_scores)
    assert reloaded.livetime_seconds == background.livetime_seconds
    np.testing.assert_array_equal(reloaded.far_curve().independent_events, curve.independent_events)


def test_the_per_detector_background_combines_what_the_network_scores_at_zero_lag(dataset_args):
    config = wrapped_config(num_examples=300, max_wrapping_slides=3)
    model = PerDetectorModel(logit_model(1), "sum")
    bank = ValidationBank(model, dataset_args, config)
    bank.generate_stream_noise()
    geometry = bank.stream_geometry
    total = int(np.sum(bank.noise_streams["lengths"]))
    detector = bank.far_scores.reshape(2, total)
    strain = bank.stream_source.segment(0)
    index = np.array([0, 7, 50, int(bank.noise_streams["lengths"][0]) - 1])
    on, off = geometry.gather(strain, index * geometry.stride_samples)
    network = bank._score(bank._noise_dataset.model_inputs(on, off))
    np.testing.assert_allclose(network, detector[:, index].sum(axis=0), rtol=1e-3, atol=1e-3)  # float32


def test_a_joint_model_runs_every_slide_within_the_budget(dataset_args):
    config = wrapped_config(num_examples=2000)
    bank = ValidationBank(logit_model(2), dataset_args, config)
    bank.generate_stream_noise()
    total = int(np.sum(bank.noise_streams["lengths"]))
    num_slides = bank.noise_streams["ring_lags_seconds"].size
    assert num_slides == config.num_examples // total and bank.far_scores.size == num_slides * total
    assert "ring_combine" not in bank.noise_streams
    background = bank.background()
    assert background.livetime_seconds == pytest.approx(
        num_slides * (total * float(bank.noise_streams["stride_seconds"]) - 2.0))
    with pytest.raises(ValueError, match="num_examples"):
        ValidationBank(logit_model(2), dataset_args, wrapped_config(num_examples=100)).generate_stream_noise()
