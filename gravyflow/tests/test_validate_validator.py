"""
The validator end to end on synthetic noise: it runs offline, ranks logits,
draws its own seeds, and reuses a checkpoint only for the model and settings
that wrote it.
"""

import h5py
import keras
import numpy as np
import pytest

# Its tests draw their data from global RNGs (see conftest's guard).
pytestmark = pytest.mark.global_rng

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import StaleCheckpointError, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank, per_example
from gravyflow.src.validate.plotting import first_detector
from gravyflow.src.validate.utils import _extract_sample_data

SAMPLES = 2048


def logit_model(seed: int, activation=None, detectors: int = 1) -> keras.Model:
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(detectors, SAMPLES), name="WHITENED_ONSOURCE")
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(2, activation=activation, name="output")(x)
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


@pytest.fixture(scope="module")
def dataset_args():
    return validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=2048.0))


@pytest.fixture(scope="module")
def small_config():
    return ValidationConfig(num_examples=64, batch_size=32, snr_range=(0.0, 15.0))


@pytest.fixture(scope="module")
def validated(tmp_path_factory, dataset_args, small_config):
    path = tmp_path_factory.mktemp("validation") / "validation.h5"
    model = logit_model(1)
    return model, path, Validator.validate(model, "tiny", dataset_args, small_config, path)


def test_validator_runs_offline_on_synthetic_noise(validated, tmp_path):
    # GF-B64: the GPS label of synthetic noise broke the injection stage.
    _, path, v = validated
    assert v.far_scores.shape == (64,) and np.all(np.isfinite(v.far_scores))
    assert v.noise_gps_times is None
    assert set(v.efficiency_data) == {"snrs", "scores", "mass1", "mass2", "central_times", "hpeak", "hrss"}
    assert all(a.shape == (64,) for a in v.efficiency_data.values())
    assert path.exists()
    report = tmp_path / "report.html"
    v.plot(report)
    assert report.stat().st_size > 0


def test_metrics_follow_from_the_stored_scores(validated, small_config):
    _, _, v = validated
    background = v.background()
    assert background.livetime_seconds == 64 * 1.0
    curves = v.efficiency_curves([1.0, 0.1, 1e-3])
    assert curves[1e-3][0].resolved is False and np.all(np.isnan(curves[1e-3][1].efficiency))
    threshold, curve = curves[0.1]
    assert threshold.resolved and threshold.achieved.count <= 6
    assert curve.total.sum() == 64
    assert 0.0 <= v.roc_curves()["SNR≥5"]["roc_auc"] <= 1.0


def test_checkpoint_round_trip(validated):
    _, path, v = validated
    loaded = Validator.load(path)
    np.testing.assert_array_equal(loaded.far_scores, v.far_scores)
    for k in v.efficiency_data:
        np.testing.assert_array_equal(loaded.efficiency_data[k], v.efficiency_data[k])
    assert loaded.model_sha256 == v.model_sha256 and loaded.config_sha256 == v.config_sha256
    assert loaded.config == v.config


def test_checkpoint_resumes_for_the_same_model(validated, dataset_args, small_config, monkeypatch):
    model, path, v = validated

    def fail(self):
        raise AssertionError("a matching checkpoint must not be scored again")

    monkeypatch.setattr(ValidationBank, "generate_noise", fail)
    monkeypatch.setattr(ValidationBank, "generate", fail)
    again = Validator.validate(model, "tiny", dataset_args, small_config, path)
    np.testing.assert_array_equal(again.far_scores, v.far_scores)


def test_checkpoint_refuses_another_model(validated, dataset_args, small_config):
    # GF-B67: checkpoints were reused by path alone.
    _, path, _ = validated
    with pytest.raises(StaleCheckpointError, match="another model"):
        Validator.validate(logit_model(2), "tiny", dataset_args, small_config, path)


def test_checkpoint_refuses_other_settings(validated, dataset_args):
    model, path, _ = validated
    with pytest.raises(StaleCheckpointError, match="validation configuration"):
        Validator.validate(model, "tiny", dataset_args, ValidationConfig(num_examples=32, batch_size=32), path)


def test_checkpoint_of_another_format_is_refused(validated, dataset_args, small_config, tmp_path):
    # A worst performer's injection was (detectors, samples) before format 3.
    model, path, _ = validated
    old = tmp_path / "format.h5"
    old.write_bytes(path.read_bytes())
    with h5py.File(old, "a") as f:
        f.attrs["format_version"] = Validator.FORMAT_VERSION - 1
    assert Validator.load(old).far_scores is None
    # Refused for its format, by name: not as written for another model.
    with pytest.raises(StaleCheckpointError, match=f"format {Validator.FORMAT_VERSION - 1}, not "
                                                   f"{Validator.FORMAT_VERSION}"):
        Validator.validate(model, "tiny", dataset_args, small_config, old)


def test_checkpoint_without_provenance_is_refused(tmp_path, dataset_args, small_config):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.create_group("far_scores").create_dataset("scores", data=np.zeros(64))
    with pytest.raises(StaleCheckpointError):
        Validator.validate(logit_model(1), "tiny", dataset_args, small_config, path)


def test_probability_models_are_refused(dataset_args):
    # GF-B69's root: softmax outputs saturate at 1.0 and tie.
    with pytest.raises(ValueError, match="probabilities"):
        ValidationBank(logit_model(1, activation="softmax"), dataset_args)


def test_the_validator_draws_its_own_streams(dataset_args):
    # B68: the FAR set and the injection set shared noise, and validation
    # reused training's seed. Each stage now draws under validator/{stage}.
    config = ValidationConfig(seed=42, num_examples=32, batch_size=32)
    bank = ValidationBank(logit_model(1), dataset_args, config)
    outputs = [gf.ReturnVariables.WHITENED_ONSOURCE]
    far = bank._dataset("far", [], outputs)
    inj = bank._dataset("inj", [], outputs)
    training = gf.Dataset(**{**dataset_args, "seed": 42, "group": config.group, "waveform_generators": [],
                             "num_examples_per_batch": 32, "steps_per_epoch": 1, "output_variables": outputs})
    batches = [np.asarray(ds[0][0]["WHITENED_ONSOURCE"]) for ds in (far, inj, training)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.array_equal(batches[i], batches[j])
    assert far.seed_metadata["prefix"] == "validator/far"
    with pytest.raises(ValueError):
        ValidationConfig(group="train")


def test_per_example_labels_are_strict():
    np.testing.assert_array_equal(per_example(np.ones((1, 4, 1)), 4, "x"), np.ones(4))
    np.testing.assert_array_equal(per_example(np.arange(8.0).reshape(4, 2), 4, "gps", per_detector=True),
                                  [0.0, 2.0, 4.0, 6.0])
    # A per-generator label of one generator, (B, G=1, I): the first detector's value.
    np.testing.assert_array_equal(per_example(np.arange(8.0).reshape(4, 1, 2), 4, "ct", per_detector=True),
                                  [0.0, 2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="two generators"):
        per_example(np.ones((4, 2, 2)), 4, "two generators", per_detector=True)
    with pytest.raises(ValueError):
        per_example(np.ones((2, 4)), 4, "two generators")
    with pytest.raises(ValueError):
        per_example(np.float64(np.nan), 4, "a scalar for a batch")


def test_a_worst_performer_sample_is_its_own_example():
    # Every output is batch-first: sample i is row i of each array, whatever its rank.
    x = {"WHITENED_ONSOURCE": np.arange(4 * 2 * 8.0).reshape(4, 2, 8)}
    y = {"WHITENED_INJECTIONS": np.arange(4 * 1 * 2 * 8.0).reshape(4, 1, 2, 8),
         "SNR": np.arange(4.0).reshape(4, 1), "INJECTION_PRESENT": np.ones(4)}
    for i in range(4):
        sample = _extract_sample_data(x, y, i, 0.5)
        for key, value in {**x, **y}.items():
            np.testing.assert_array_equal(sample[key], value[i], err_msg=key)
        assert sample["score"] == 0.5


def test_worst_false_negatives_carry_their_own_injection(validated):
    # Each worst false negative's whitened injection has the SNR its label gives.
    _, _, v = validated
    samples = [s for bin_samples in v.worst_performers["false_negatives"].values() for s in bin_samples]
    assert len(samples) > 1
    for sample in samples:
        injection = np.asarray(sample["WHITENED_INJECTIONS"], dtype=np.float64)
        norm = np.sqrt(np.sum(injection ** 2))
        np.testing.assert_allclose(norm, float(np.ravel(sample["SNR"])[0]), rtol=0.05, atol=0.05)
        assert injection.shape == (1, 1, SAMPLES)  # (generators, detectors, samples)


def test_the_waveform_plot_takes_the_first_detector():
    onsource = np.arange(2 * 8.0).reshape(2, 8)
    injection = np.arange(2 * 2 * 8.0).reshape(2, 2, 8)
    plotted_onsource, plotted_injection = first_detector(
        {"WHITENED_ONSOURCE": onsource, "WHITENED_INJECTIONS": injection})
    np.testing.assert_array_equal(plotted_onsource, onsource[0])
    np.testing.assert_array_equal(plotted_injection, injection[0, 0] + injection[1, 0])
    assert first_detector({"WHITENED_ONSOURCE": onsource})[1] is None
    with pytest.raises(ValueError, match="WHITENED_INJECTIONS"):
        first_detector({"WHITENED_ONSOURCE": onsource, "WHITENED_INJECTIONS": injection[0]})


def test_the_validator_scores_injections_on_two_detectors(tmp_path):
    # The injection stage read CENTRAL_TIME, (B, G, I), as one value per example.
    config = ValidationConfig(num_examples=32, batch_size=32, snr_range=(0.0, 15.0))
    args = validation_dataset_args(config=ValidationDatasetConfig(
        sample_rate_hertz=2048.0, ifos=[gf.IFO.H1, gf.IFO.L1]))
    v = Validator.validate(logit_model(1, detectors=2), "two", args, config, tmp_path / "v.h5")
    assert all(a.shape == (32,) and np.all(np.isfinite(a)) for a in v.efficiency_data.values())
    samples = [s for bin_samples in v.worst_performers["false_negatives"].values() for s in bin_samples]
    assert samples and all(np.shape(s["WHITENED_INJECTIONS"]) == (1, 2, SAMPLES) for s in samples)
    v.plot(tmp_path / "report.html")
