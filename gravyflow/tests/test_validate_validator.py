"""
The validator end to end on synthetic noise: it runs offline, ranks logits,
draws its own seeds, and reuses a checkpoint only for the model and settings
that wrote it.
"""

import dataclasses
from copy import deepcopy
from unittest import mock

import h5py
import keras
import numpy as np
import pytest


import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import StaleCheckpointError, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank, per_example
from gravyflow.src.validate.metrics import logit_scores
from gravyflow.src.validate.plotting import first_detector, generate_sensitive_volume_plot
from gravyflow.src.validate.utils import _extract_sample_data

SAMPLES = 2048


def logit_model(seed: int, activation=None, detectors: int = 1) -> keras.Model:
    """A linear model whose weights are drawn from ``seed`` alone, never a global RNG."""
    inputs = keras.Input(shape=(detectors, SAMPLES), name="WHITENED_ONSOURCE")
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(2, activation=activation, name="output",
                                 kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


@pytest.fixture(scope="module")
def dataset_args():
    return validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=2048.0))


@pytest.fixture(scope="module")
def small_config():
    return ValidationConfig(num_examples=64, batch_size=32, snr_range=(0.0, 15.0), volume_num_injections=64)


@pytest.fixture(scope="module")
def validated(tmp_path_factory, dataset_args, small_config):
    path = tmp_path_factory.mktemp("validation") / "validation.h5"
    model = logit_model(1)
    return model, path, Validator.validate(model, "tiny", dataset_args, small_config, path)


def test_validator_runs_offline_on_synthetic_noise(validated, tmp_path):
    # The GPS label of synthetic noise broke the injection stage.
    _, path, v = validated
    assert v.far_scores.shape == (64,) and np.all(np.isfinite(v.far_scores))
    assert v.noise_gps_times is None
    assert set(v.efficiency_data) == {"snrs", "scores", "mass1", "mass2", "central_times", "hpeak", "hrss"}
    assert all(a.shape == (64,) for a in v.efficiency_data.values())
    assert set(v.volume_data) == {"scores", "distances", "snrs", "mass1", "mass2", "reach_mpc"}
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
    # The area under the ROC curve is the Mann-Whitney statistic: the chance that an
    # injection outscores a noise window, ties counting a half.
    injections = v.efficiency_data["scores"][v.efficiency_data["snrs"] >= small_config.default_roc_min_snr]
    noise = v.far_scores
    wins = (injections[:, None] > noise[None, :]).sum() + 0.5 * (injections[:, None] == noise[None, :]).sum()
    # Representation: the trapezoid sums at most 128 float64 terms of one sign, each in [0, 1].
    np.testing.assert_allclose(v.roc_curves()["SNR≥5"]["roc_auc"], wins / (injections.size * noise.size),
                               rtol=0, atol=128 * np.finfo(np.float64).eps)


def test_checkpoint_round_trip(validated):
    _, path, v = validated
    loaded = Validator.load(path)
    np.testing.assert_array_equal(loaded.far_scores, v.far_scores)
    for data in ("efficiency_data", "volume_data"):
        assert set(getattr(loaded, data)) == set(getattr(v, data))
        for k in getattr(v, data):
            np.testing.assert_array_equal(getattr(loaded, data)[k], getattr(v, data)[k])
    assert loaded.model_sha256 == v.model_sha256 and loaded.config_sha256 == v.config_sha256
    assert loaded.config == v.config


def test_the_sensitive_volume_is_recorded_and_plotted(validated, small_config):
    # The volume set: sources at physical distances, uniform in volume out to the
    # population's reach R, recorded; the curve follows from the record alone, with
    # V within R, its three intervals and noise floor, and the estimate beyond R, and
    # the record read back gives the same curve.
    _, path, v = validated
    data = v.volume_data
    reach = float(data["reach_mpc"])
    assert np.all((data["distances"] > 0) & (data["distances"] <= reach))
    # Each source's SNR goes as one over its distance: its SNR at the reach is at most the
    # loudest possible source's, the reach SNR, to the scatter of each window's own PSD
    # estimate (a few per cent), against which its SNR is measured.
    assert np.all(data["snrs"] * data["distances"] / reach <= small_config.volume_reach_snr * 1.1)
    curve = v.sensitive_volume_curve()
    assert curve.far_hz.tolist() == small_config.far_thresholds
    resolved = curve.resolved
    assert resolved.any() and not resolved.all()  # 64 s of background resolves only the higher FARs
    within = curve.volume_within_reach_gpc3
    assert np.all(np.isnan(within[~resolved])) and np.all(np.isnan(curve.beyond_reach_gpc3[~resolved]))
    for value, lower, upper in ((within, curve.injection_lower_gpc3, curve.injection_upper_gpc3),
                                (within, curve.background_lower_gpc3, curve.background_upper_gpc3),
                                (within, curve.joint_lower_gpc3, curve.joint_upper_gpc3),
                                (curve.beyond_reach_gpc3, curve.beyond_reach_lower_gpc3,
                                 curve.beyond_reach_upper_gpc3)):
        assert np.all(lower[resolved] <= value[resolved]) and np.all(value[resolved] <= upper[resolved])
    assert np.all(within[resolved] <= curve.reference_volume_gpc3)
    assert curve.reach_mpc == reach and curve.data_reuse == 1
    assert curve.num_injections == small_config.volume_num_injections
    assert curve.num_bootstrap == small_config.volume_bootstrap_samples
    assert curve.beyond_reach_fraction == small_config.volume_beyond_reach_fraction
    loaded = Validator.load(path).sensitive_volume_curve()
    for field in dataclasses.fields(curve):
        np.testing.assert_array_equal(getattr(loaded, field.name), getattr(curve, field.name), err_msg=field.name)

    plot = generate_sensitive_volume_plot([v])
    # The axes say what the number is: a Euclidean volume within the reach, of detector-frame
    # masses; the legend names the reach.
    assert [child.yaxis.axis_label for child in plot.children] == [
        "Sensitive Volume within reach R (Gpc³, Euclidean, detector-frame masses)",
        "Sensitive Distance within reach R (Mpc, Euclidean, detector-frame masses)"]
    assert all(child.renderers for child in plot.children)
    labels = {item.label.value for child in plot.children for item in child.legend[0].items}
    assert f"Tiny (R = {reach:,.0f} Mpc)" in labels


def volume_set(dataset_args, **settings):
    config = ValidationConfig(**{"num_examples": 32, "batch_size": 32, "volume_num_injections": 32, **settings})
    bank = ValidationBank(logit_model(1), dataset_args, config)
    bank.generate_volume()
    return bank.get_volume_data()


def test_the_sensitive_volume_does_not_depend_on_the_efficiency_snr_range(dataset_args):
    # The volume set draws distances, not SNRs: the efficiency's SNR range plays no part.
    low, high = volume_set(dataset_args, snr_range=(0.0, 15.0)), volume_set(dataset_args, snr_range=(8.0, 30.0))
    assert set(low) == set(high)
    for key in low:
        np.testing.assert_array_equal(low[key], high[key], err_msg=key)


def test_the_reach_is_the_populations_and_can_be_set(dataset_args):
    # The reach reads the population's bounds and the noise, never the injections drawn:
    # another number of injections has the same reach, and a configured one is used as given.
    assert volume_set(dataset_args)["reach_mpc"] == volume_set(
        dataset_args, volume_num_injections=64)["reach_mpc"]
    given = volume_set(dataset_args, volume_reach_mpc=250.0)
    assert given["reach_mpc"] == 250.0 and np.all(given["distances"] <= 250.0)
    with pytest.raises(ValueError, match="volume_reach_mpc"):
        ValidationConfig(volume_reach_mpc=0.0)


def reach_of(dataset_args, masses=None, spins=None, **settings):
    """The validator's reach for the example population, its component masses (low, high) and aligned spins as given."""
    args = deepcopy(dataset_args)
    generator = args["waveform_generators"]
    for attribute, bounds in (("mass_1_msun", masses), ("mass_2_msun", masses), ("spin_1_in", spins),
                              ("spin_2_in", spins)):
        if bounds is not None:
            setattr(generator, attribute, gf.Distribution(value=bounds[0], type_=gf.DistributionType.CONSTANT)
                    if bounds[0] == bounds[1] else
                    gf.Distribution(min_=bounds[0], max_=bounds[1], type_=gf.DistributionType.UNIFORM))
    config = ValidationConfig(**{"num_examples": 32, "batch_size": 32, **settings})
    return ValidationBank(logit_model(1), args, config).reach_distance()


def test_the_reach_is_the_loudest_source_of_the_whole_population(dataset_args):
    # A heavy binary merges below the band, so the heaviest is not the loudest: equal
    # masses of 200 are louder than of 300. A population of masses up to 300 has a reach
    # at least that of its loudest member, never the heaviest's.
    loudest = reach_of(dataset_args, masses=(200.0, 200.0))
    assert reach_of(dataset_args, masses=(300.0, 300.0)) < 0.9 * loudest
    assert reach_of(dataset_args, masses=(5.0, 300.0)) >= loudest * (1 - 1e-4)
    # Aligned spins: the reach is the loudest spin's, at its bounds or between them.
    spinning = reach_of(dataset_args, masses=(5.0, 95.0), spins=(-0.9, 0.9))
    assert spinning >= reach_of(dataset_args, masses=(95.0, 95.0), spins=(0.9, 0.9)) * (1 - 1e-4)
    assert spinning >= reach_of(dataset_args, masses=(5.0, 95.0)) * (1 - 1e-4)


def test_the_reach_is_the_populations_not_the_batchs(dataset_args):
    # The reach reads the population's bounds and the noise's PSD: the batch size plays no
    # part, and another seed moves it only by its PSD estimate's scatter (64 windows).
    reach = reach_of(dataset_args)
    assert reach_of(dataset_args, batch_size=8, num_examples=8) == reach
    assert reach_of(dataset_args, batch_size=512, num_examples=512) == reach
    assert reach_of(dataset_args, seed=2) == pytest.approx(reach, rel=0.01)


def test_the_reach_refuses_a_population_it_cannot_bound(dataset_args):
    from gravyflow.src.validate.reach import population_box

    generator = deepcopy(dataset_args["waveform_generators"])
    box = population_box(generator)
    assert box["mass_1_msun"] == (5.0, 95.0) and box["spin_1_in"] == (0.0,)
    generator.spin_1_in = gf.Distribution(min_=-0.5, max_=0.8, type_=gf.DistributionType.UNIFORM)
    assert population_box(generator)["spin_1_in"] == (-0.5, 0.0, 0.8)
    unbounded = deepcopy(generator)
    unbounded.mass_1_msun = gf.Distribution(mean=30.0, std=5.0, type_=gf.DistributionType.NORMAL)
    with pytest.raises(ValueError, match="mass_1_msun is unbounded"):
        population_box(unbounded)
    precessing = deepcopy(generator)
    precessing.approximant = gf.Approximant.IMRPhenomPv2
    precessing.spin_2_in = (gf.Distribution(min_=0.0, max_=0.5, type_=gf.DistributionType.UNIFORM), 0.0, 0.0)
    with pytest.raises(ValueError, match="in-plane components"):
        population_box(precessing)


def test_a_population_with_no_reach_is_refused_before_any_stage_runs(dataset_args, tmp_path):
    # A precessing or unbounded population has no loudest source, so no reach: the validator
    # refuses it as it is built, before it scores a window or writes a checkpoint. A reach
    # set by hand needs no loudest source.
    precessing = deepcopy(dataset_args)
    precessing["waveform_generators"].approximant = gf.Approximant.IMRPhenomPv2
    precessing["waveform_generators"].spin_1_in = (
        gf.Distribution(min_=0.0, max_=0.5, type_=gf.DistributionType.UNIFORM), 0.0, 0.0)
    unbounded = deepcopy(dataset_args)
    unbounded["waveform_generators"].mass_1_msun = gf.Distribution(mean=30.0, std=5.0,
                                                                   type_=gf.DistributionType.NORMAL)
    config = ValidationConfig(num_examples=32, batch_size=32)
    model = logit_model(1)
    path = tmp_path / "validation.h5"
    with mock.patch.object(model, "predict_on_batch", side_effect=AssertionError("a window was scored")):
        for args, match in ((precessing, "spin_1_in has in-plane components"), (unbounded, "mass_1_msun is unbounded")):
            with pytest.raises(ValueError, match=match):
                Validator.validate(model, "no reach", args, config, path)
            assert not path.exists()
    ValidationBank(model, precessing, dataclasses.replace(config, volume_reach_mpc=500.0))


def test_the_network_bound_is_the_best_response_each_detector_can_have():
    # One detector: a face-on source overhead is within it at every polarisation angle,
    # and reaches it at the best; two detectors of one PSD, root two times it.
    from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window
    from gravyflow.src.dataset.tools.snr import template_sigma
    from gravyflow.src.validate.reach import network_snr_bound

    rate, n = 2048.0, 3
    h = np.asarray(_generate_cbc_window(num_waveforms=n, sample_rate_hertz=rate, window_seconds=2.0,
                                        min_frequency_hertz=np.full(n, 20.0), mass_1_msun=np.array([10.0, 30.0, 60.0]),
                                        mass_2_msun=np.array([10.0, 20.0, 50.0]), distance_mpc=np.full(n, 100.0),
                                        inclination_radians=np.zeros(n), reference_orbital_phase_in=np.zeros(n)))
    psd = np.full(int(rate) // 2 + 1, 1e-46)
    config = gf.SpectralConfig()
    bound = network_snr_bound(h, psd[None], rate, config)
    overhead = np.stack([np.asarray(template_sigma(np.cos(2 * psi) * h[:, 0] + np.sin(2 * psi) * h[:, 1], psd,
                                                   rate, 20.0, None)) for psi in np.linspace(0, np.pi / 2, 91)])
    assert np.all(overhead <= bound * (1 + 1e-5))
    np.testing.assert_allclose(overhead.max(axis=0), bound, rtol=1e-4)
    np.testing.assert_allclose(network_snr_bound(h, np.stack([psd, psd]), rate, config), np.sqrt(2) * bound,
                               rtol=1e-6)


def test_a_placed_source_does_not_depend_on_the_distance_its_generator_drew(dataset_args):
    # The physical placement: the injected strain is the generated source's at the
    # distance its generator drew, times drawn over target, so the same sources at the
    # same targets have the same SNR whether the generator drew them at 100 Mpc, at
    # 1000 Mpc or anywhere between. (The label is the target by construction, and
    # proves nothing of the strain.)
    targets = np.linspace(500.0, 4000.0, 32)

    def placed(distance):
        generator = deepcopy(dataset_args["waveform_generators"])
        generator.distance_mpc = distance
        generator.scaling_method = gf.ScalingMethod(value=targets, type_=gf.WaveformParameters.DISTANCE_MPC)
        generator.injection_chance = 1.0
        args = {**dataset_args, "waveform_generators": [generator], "num_examples_per_batch": 32,
                "steps_per_epoch": 1, "group": "test", "seed": gf.SeedTree(7).sub("validator/vol"),
                "output_variables": [gf.ScalingTypes.SNR, gf.WaveformParameters.MASS_1_MSUN]}
        _, y = gf.Dataset(**args)[0]
        return per_example(y["SNR"], 32, "SNR"), per_example(y["MASS_1_MSUN"], 32, "MASS_1_MSUN")

    (near, masses), (far, far_masses) = (placed(gf.Distribution(value=d, type_=gf.DistributionType.CONSTANT))
                                         for d in (100.0, 1000.0))
    between, _ = placed(gf.Distribution(min_=100.0, max_=1000.0, type_=gf.DistributionType.UNIFORM))
    np.testing.assert_array_equal(masses, far_masses)
    assert np.all(np.isfinite(near) & (near > 0)), near
    np.testing.assert_allclose(near, far, rtol=1e-4)
    np.testing.assert_allclose(near, between, rtol=1e-4)


def test_a_source_placed_at_a_distance_is_the_generated_source_scaled_by_it(dataset_args):
    # ScalingMethod(type_=DISTANCE_MPC) places each generated source at its target distance:
    # labelled there, and the same sources twice as far have half the SNR.
    def placed(distances):
        generator = deepcopy(dataset_args["waveform_generators"])
        generator.scaling_method = gf.ScalingMethod(value=distances, type_=gf.WaveformParameters.DISTANCE_MPC)
        generator.injection_chance = 1.0
        args = {**dataset_args, "waveform_generators": [generator], "num_examples_per_batch": 32,
                "steps_per_epoch": 1, "group": "test", "seed": gf.SeedTree(7).sub("validator/vol"),
                "output_variables": [gf.WaveformParameters.DISTANCE_MPC, gf.ScalingTypes.SNR]}
        _, y = gf.Dataset(**args)[0]
        return (per_example(y["DISTANCE_MPC"], 32, "DISTANCE_MPC"), per_example(y["SNR"], 32, "SNR"))

    d = np.linspace(100.0, 800.0, 32)
    (near, near_snr), (far, far_snr) = placed(d), placed(2 * d)
    np.testing.assert_allclose(near, d, rtol=1e-6)
    np.testing.assert_allclose(far, 2 * d, rtol=1e-6)
    np.testing.assert_allclose(near_snr, 2 * far_snr, rtol=1e-5)
    # A physical distance in dimensionless WHITE noise is refused, as a strain target is.
    generator = deepcopy(dataset_args["waveform_generators"])
    generator.scaling_method = gf.ScalingMethod(value=d, type_=gf.WaveformParameters.DISTANCE_MPC)
    from gravyflow.src.dataset.features.injection import check_scaling_matches_noise
    with pytest.raises(ValueError, match="DISTANCE_MPC, a strain amplitude"):
        check_scaling_matches_noise(gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]), [generator])


def test_injections_without_a_volume_set_have_no_sensitive_volume(validated, dataset_args):
    # A burst set, or one in WHITE noise, has efficiency, but no volume.
    _, _, v = validated
    burst = Validator("burst", v.config, v.dataset_args)
    burst.far_scores, burst.efficiency_data = v.far_scores, v.efficiency_data
    assert burst.sensitive_volume_curve() is None
    assert all(not child.renderers for child in generate_sensitive_volume_plot([burst]).children)
    white = {**dataset_args, "noise_obtainer": gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])}
    assert not ValidationBank(logit_model(1), white).has_volume


def test_checkpoint_resumes_for_the_same_model(validated, dataset_args, small_config, monkeypatch):
    model, path, v = validated

    def fail(self):
        raise AssertionError("a matching checkpoint must not be scored again")

    monkeypatch.setattr(ValidationBank, "generate_noise", fail)
    monkeypatch.setattr(ValidationBank, "generate", fail)
    monkeypatch.setattr(ValidationBank, "generate_volume", fail)
    again = Validator.validate(model, "tiny", dataset_args, small_config, path)
    np.testing.assert_array_equal(again.far_scores, v.far_scores)


def test_checkpoint_refuses_another_model(validated, dataset_args, small_config):
    # Checkpoints were reused by path alone.
    _, path, _ = validated
    with pytest.raises(StaleCheckpointError, match="another model"):
        Validator.validate(logit_model(2), "tiny", dataset_args, small_config, path)


def test_checkpoint_refuses_other_settings(validated, dataset_args):
    model, path, _ = validated
    with pytest.raises(StaleCheckpointError, match="validation configuration"):
        Validator.validate(model, "tiny", dataset_args, ValidationConfig(num_examples=32, batch_size=32), path)


def test_a_record_of_another_format_is_refused(validated, dataset_args, small_config, tmp_path):
    # Format 4 had no volume set: it is refused by name, read or resumed, never converted.
    model, path, _ = validated
    assert Validator.FORMAT_VERSION == 5
    old = tmp_path / "format.h5"
    old.write_bytes(path.read_bytes())
    with h5py.File(old, "a") as f:
        f.attrs["format_version"] = 4
    refusal = "is in validation format 4; this version reads only format 5.*Validate the model again"
    with pytest.raises(StaleCheckpointError, match=refusal):
        Validator.load(old)
    # Refused for its format, by name: not as written for another model.
    with pytest.raises(StaleCheckpointError, match=refusal):
        Validator.validate(model, "tiny", dataset_args, small_config, old)


def test_checkpoint_without_provenance_is_refused(tmp_path, dataset_args, small_config):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.create_group("far_scores").create_dataset("scores", data=np.zeros(64))
    # Written before checkpoints had formats: refused, and said so.
    with pytest.raises(StaleCheckpointError, match="has no format version"):
        Validator.validate(logit_model(1), "tiny", dataset_args, small_config, path)


def test_the_heart_is_anything_that_beats(dataset_args):
    # The validator's annotations name no class gravyflow does not have: they resolve, and the
    # heart it beats as it scores is any object with a beat() method.
    import typing

    from gravyflow.src.validate.bank import Heartbeat

    for function in (ValidationBank.__init__, Validator.validate):
        hints = typing.get_type_hints(function, localns={"keras": keras})
        assert hints["heart"] == typing.Optional[Heartbeat]
    beats = []
    heart = type("Pulse", (), {"beat": lambda self: beats.append(1)})()
    bank = ValidationBank(logit_model(1), dataset_args, ValidationConfig(num_examples=32, batch_size=16), heart)
    bank.generate_noise()
    assert len(beats) == 2


def test_probability_models_are_refused(dataset_args):
    # The root of tied scores: softmax outputs saturate at 1.0 and tie.
    with pytest.raises(ValueError, match="probabilities"):
        ValidationBank(logit_model(1, activation="softmax"), dataset_args)


def test_the_validator_draws_its_own_streams(dataset_args):
    # B68: the FAR set and the injection set shared noise, and validation
    # reused training's seed. Each stage now draws under validator/{stage}.
    config = ValidationConfig(seed=42, num_examples=32, batch_size=32)
    bank = ValidationBank(logit_model(1), dataset_args, config)
    outputs = [gf.ReturnVariables.WHITENED_ONSOURCE]
    far = bank.stage_dataset("far", outputs=outputs)
    inj = bank.stage_dataset("inj", outputs=outputs)
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


#: Measured: the SNR label is set against the PSD of the uncropped onsource, and the
#: injection is whitened by the offsource's and cropped, so the two differ by more than
#: rounding. Worst over this module's 28 worst false negatives, at gravyflow df4acab on
#: CPU: 1.9% (11.03 against 10.82); the bound keeps a margin of 2.
SNR_LABEL_RTOL = 0.04


def test_worst_false_negatives_carry_their_own_injection(validated):
    # Each worst false negative's whitened injection has the SNR its label gives.
    _, _, v = validated
    samples = [s for bin_samples in v.worst_performers["false_negatives"].values() for s in bin_samples]
    assert len(samples) > 1
    for sample in samples:
        injection = np.asarray(sample["WHITENED_INJECTIONS"], dtype=np.float64)
        norm = np.sqrt(np.sum(injection ** 2))
        np.testing.assert_allclose(norm, float(np.ravel(sample["SNR"])[0]), rtol=SNR_LABEL_RTOL)
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


def test_the_validator_scores_real_events(dataset_args, offline_events):
    # The real-event stage scores every confident event, once: real events are test data.
    config = ValidationConfig(num_examples=32, batch_size=32, score_real_events=True)
    bank = ValidationBank(logit_model(1), dataset_args, config)
    bank.far_scores = np.random.default_rng(3).standard_normal(64)
    bank.generate_real_events(observing_runs=[gf.ObservingRun.O3])
    assert sorted(e["name"] for e in bank.real_events) == sorted(e["name"] for e in offline_events)
    assert all(e["status"] == "scored" and np.isfinite(e["score"]) for e in bank.real_events)


def _scoring_args(sample_rate_hertz=2048.0):
    """Validation arguments whose band, PSD estimate and crop are not the defaults."""
    args = validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=sample_rate_hertz))
    args.update(
        crop_duration_seconds=0.75,
        spectral_config=gf.SpectralConfig(lower_frequency_hertz=30.0, fft_duration_seconds=0.5,
                                          overlap_duration_seconds=0.25),
    )
    return args


def test_a_window_scores_as_the_background_scores_it():
    # Real events were whitened by a spectrum of the default settings, not the dataset's,
    # so an event's score was not comparable with the background's.
    args = _scoring_args()
    model = logit_model(1)
    bank = ValidationBank(model, args, ValidationConfig(num_examples=8, batch_size=8))
    dataset = bank.stage_dataset("far")
    raw = dataset.noise(dataset.streams(0))
    inputs, _ = dataset[0]
    background = logit_scores(model.predict_on_batch(inputs))
    windows = bank.score_windows(dataset, raw[gf.ReturnVariables.ONSOURCE], raw[gf.ReturnVariables.OFFSOURCE])
    np.testing.assert_array_equal(windows, background)


def test_real_events_are_read_in_the_backgrounds_window(offline_events):
    # The event windows were read with the default crop, whatever the dataset's.
    read = []
    original = gf.TransientDataObtainer._read_example

    def recording(self, record, reads, window_spec):
        read.append((window_spec.total_onsource_duration_seconds, window_spec.offsource_duration_seconds))
        return original(self, record, reads, window_spec)

    args = _scoring_args()
    bank = ValidationBank(logit_model(1), args, ValidationConfig(num_examples=8, batch_size=8, score_real_events=True))
    bank.far_scores = np.random.default_rng(3).standard_normal(64)
    with mock.patch.object(gf.TransientDataObtainer, "_read_example", recording):
        bank.generate_real_events(observing_runs=[gf.ObservingRun.O3])
    assert read and set(read) == {(1.0 + 2 * 0.75, args["offsource_duration_seconds"])}
    assert all(e["status"] == "scored" and np.isfinite(e["score"]) for e in bank.real_events)


@pytest.mark.parametrize("scaling,value", [(gf.ScalingTypes.HRSS, 1e-22), (gf.ScalingTypes.HPEAK, 1e-21)])
def test_the_injection_set_is_scaled_to_its_snr_range_whatever_the_training_scaling(
        dataset_args, scaling, value):
    # The injection set kept a generator's scaling type, so a model trained with HRSS or
    # HPEAK scaling was validated at an HRSS or HPEAK of 8 to 20, not at SNR 8 to 20.
    args = dict(dataset_args)
    generator = deepcopy(args["waveform_generators"])
    generator.scaling_method = gf.ScalingMethod(
        value=gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT), type_=scaling)
    args["waveform_generators"] = generator
    config = ValidationConfig(num_examples=32, batch_size=32, snr_range=(8.0, 20.0))
    bank = ValidationBank(logit_model(1), args, config)
    bank.generate()
    snrs = bank.get_efficiency_data()["snrs"]
    assert np.all((snrs > 8.0 - 1e-3) & (snrs < 20.0 + 1e-3)), snrs
    assert np.ptp(snrs) > 4.0


def test_a_time_slide_background_is_read_back_as_it_was_written(tmp_path, small_config):
    # validation.h5 keeps a time-slide background's streams beside its scores, but no
    # test read them back: a file whose streams were lost would be counted as
    # independent windows, its adjacent passes each a trigger.
    v = Validator("tiny", small_config, {"onsource_duration_seconds": 1.0})
    v.far_scores = np.array([3.0, 3.0, 0.0, 3.0, 0.0, 3.0, 3.0])
    v.noise_streams = {"lengths": np.array([3, 4], dtype=np.int64), "slides": np.array([1, 2], dtype=np.int64),
                       "blocks": np.array([5, 5], dtype=np.int64)}
    path = tmp_path / "validation.h5"
    v.save(path)
    loaded = Validator.load(path)
    assert set(loaded.noise_streams) == set(v.noise_streams)
    for key, value in v.noise_streams.items():
        np.testing.assert_array_equal(loaded.noise_streams[key], value)
        assert loaded.noise_streams[key].dtype == value.dtype
    background = loaded.background()
    # Two streams, the slides of one block; adjacent passes join: 1 + 2 triggers.
    assert [stream.scores.tolist() for stream in background.streams] == [[3.0, 3.0, 0.0], [3.0, 0.0, 3.0, 3.0]]
    assert {stream.block for stream in background.streams} == {5}
    assert background.trigger_count(1.0) == v.background().trigger_count(1.0) == 3
    assert background.livetime_seconds == 7.0

    # A file without streams (synthetic noise, or written before time slides) is read
    # as windows drawn independently: every pass its own trigger.
    with h5py.File(path, "a") as f:
        del f["far_scores/streams"]
    independent = Validator.load(path)
    assert independent.noise_streams is None
    background = independent.background()
    assert background.streams == () and background.trigger_count(1.0) == 5
    assert background.livetime_seconds == 7.0


def test_the_backgrounds_livetime_is_the_span_of_the_windows_it_scored(dataset_args, monkeypatch):
    # The livetime's window length was read from the dataset arguments, or the default
    # when they named none, apart from the window the background's dataset scored (and
    # its time slides tile): a default moved after scoring moved the livetime, not the
    # windows.
    args = {k: v for k, v in dataset_args.items() if k != "onsource_duration_seconds"}
    window = SAMPLES / args["sample_rate_hertz"]  # the model's input; the default is set to it, whatever it was
    monkeypatch.setattr(gf.Defaults, "onsource_duration_seconds", window)
    bank = ValidationBank(logit_model(1), args, ValidationConfig(num_examples=32, batch_size=32))
    bank.generate_noise()
    assert bank._noise_dataset.window_spec.onsource_duration_seconds == window
    monkeypatch.setattr(gf.Defaults, "onsource_duration_seconds", 2.0 * window)
    assert bank.background().livetime_seconds == 32 * window
