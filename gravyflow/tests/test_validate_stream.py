"""
Stream mode, the validator's default, end to end on synthetic noise: a Keras model
slid along continuous raw strain at its stride, its triggers the background, and
each injection scored locally, which equals scoring the whole stream with the
injection added.
"""

import dataclasses

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank, _INJECTION_LABELS
from gravyflow.src.validate.stream import StreamBackground, cluster, loudest_within

SAMPLES = 2048
RV = gf.ReturnVariables


def logit_model(seed: int = 1) -> keras.Model:
    inputs = keras.Input(shape=(1, SAMPLES), name="WHITENED_ONSOURCE")
    x = keras.layers.Flatten()(inputs)
    outputs = keras.layers.Dense(2, name="output", kernel_initializer=keras.initializers.GlorotUniform(seed=seed))(x)
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


@pytest.fixture(scope="module")
def dataset_args():
    return validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=2048.0))


@pytest.fixture(scope="module")
def stream_config():
    return ValidationConfig(num_examples=64, batch_size=32, snr_range=(0.0, 15.0), volume_num_injections=32,
                            stream_segment_seconds=64.0)


@pytest.fixture(scope="module")
def validated(tmp_path_factory, dataset_args, stream_config):
    path = tmp_path_factory.mktemp("stream") / "validation.h5"
    return path, Validator.validate(logit_model(), "streamed", dataset_args, stream_config, path)


def test_stream_mode_is_the_default():
    config = ValidationConfig()
    assert config.mode == "stream" and config.stream
    assert config.stream_stride_seconds(1.0) == 0.25 and config.stream_cluster_seconds == 1.0
    assert config.found_window_seconds == 0.1
    assert ValidationConfig(stride_seconds=0.5, cluster_seconds=2.0).stream_stride_seconds(1.0) == 0.5
    for bad in ({"mode": "windows"}, {"stride_seconds": 0.0}, {"found_window_seconds": 0.0}):
        with pytest.raises(ValueError):
            ValidationConfig(**bad)


def test_the_validator_streams_by_default(validated, stream_config, tmp_path):
    path, v = validated
    # The background: num_examples windows one stride apart, their triggers its count.
    assert v.far_scores.shape == (64,) and v.noise_streams["stride_seconds"] == 0.25
    background = v.background()
    assert isinstance(background, StreamBackground)
    assert background.livetime_seconds == pytest.approx(63 * 0.25 + 1.0)
    times = np.arange(64) * 0.25
    assert background.num_triggers == cluster(times, v.far_scores, 1.0).size
    # Each injection: its found statistic (a trigger's score, or -inf), its trigger's time from
    # the merger within the found window, its merger's time and its source's records.
    data = v.efficiency_data
    assert {"scores", "trigger_offsets", "merger_times", "snrs", "hpeak", "hrss", "mass1", "mass2",
            "in_band_durations", "whitened_peaks", "bns_ranges"} <= set(data)
    assert data["scores"].shape == (64,) and data["bns_ranges"].shape == (64, 1)
    found = np.isfinite(data["scores"])
    assert found.any()
    assert np.all(np.abs(data["trigger_offsets"][found]) <= 0.1 + 1e-9)
    assert np.all(np.isnan(data["trigger_offsets"][~found]))
    assert np.all(data["in_band_durations"] > 0) and np.all(data["whitened_peaks"] >= 0)
    assert np.all(data["bns_ranges"] > 0)
    assert np.all((data["snrs"] >= -1e-3) & (data["snrs"] <= 15.0 + 1e-3))
    assert {"scores", "chirp_distances", "distances", "snrs", "reach_mpc", "in_band_durations", "whitened_peaks",
            "bns_ranges", "merger_times"} <= set(v.volume_data)
    # The estimators read the found statistic as they read a window's score.
    curves = v.efficiency_curves([1.0, 0.1])
    assert curves[0.1].threshold.resolved and curves[0.1].binned.total.sum() == 64
    assert v.sensitive_volume_curve() is not None and v.roc_curves()
    # The record keeps the mode, and reads back whole.
    loaded = Validator.load(path)
    assert loaded.config.mode == "stream" and loaded.config == v.config
    for key, value in v.noise_streams.items():
        np.testing.assert_array_equal(loaded.noise_streams[key], value)
    for key, value in data.items():
        np.testing.assert_array_equal(loaded.efficiency_data[key], value)
    assert loaded.background().far_count(0.0) == background.far_count(0.0)
    report = tmp_path / "report.html"
    v.plot(report)
    assert report.stat().st_size > 0


def test_a_local_injection_scores_as_the_whole_stream_with_it_added(dataset_args):
    # Every injection scored on its short stretch gives what the whole stream, with that
    # injection added, gives: its found statistic and its trigger's time. Loud signals, so
    # that the windows they reach score differently from the clean stream's.
    config = ValidationConfig(num_examples=64, batch_size=32, snr_range=(40.0, 80.0), stream_segment_seconds=64.0)
    model = logit_model(3)
    bank = ValidationBank(model, dataset_args, config)
    bank.generate_stream_noise()
    num = 6
    labels = {k: v for k, v in _INJECTION_LABELS.items() if k in ("snrs",)}

    def generator():
        g = bank._generator()
        g.scaling_method = gf.ScalingMethod(
            value=gf.Distribution(min_=40.0, max_=80.0, type_=gf.DistributionType.UNIFORM), type_=gf.ScalingTypes.SNR)
        return g

    local = bank._score_locally("inj", generator(), num, labels)
    dataset, placement, half, streams = bank.local_dataset("inj", generator(), num, [RV.ONSOURCE, RV.INJECTIONS])
    _, y = dataset[0]
    injections = np.asarray(y[RV.INJECTIONS.name], dtype=np.float64)[:, 0]  # (B, I, 2 half)
    geometry = bank.stream_geometry
    changed = 0
    for k in range(num):
        stream = streams[int(placement.stream[k])]
        clean, _ = bank.stream_source.stream_strain(stream)
        m = int(placement.sample[k])
        full = clean.copy()
        full[:, m - half:m + half] += injections[k]
        n = geometry.num_windows(full.shape[1])
        index = np.arange(n)
        scores, clean_scores = (
            np.concatenate([bank._score(bank._noise_dataset.model_inputs(*geometry.gather(data, chunk * geometry.stride_samples)))
                            for chunk in np.array_split(index, max(n // 32, 1))]) for data in (full, clean))
        times = geometry.window_time(index)
        keep = cluster(times, scores, config.stream_cluster_seconds)
        merger = m / geometry.sample_rate_hertz
        stat, when = loudest_within(times[keep], scores[keep], [merger], config.found_window_seconds)
        assert local["scores"][k] == pytest.approx(stat[0], rel=1e-5, abs=1e-5), k
        if np.isfinite(stat[0]):
            assert local["trigger_offsets"][k] == pytest.approx(when[0] - merger, abs=1e-9)
        near = np.abs(times - merger) <= 1.0
        changed += not np.allclose(scores[near], clean_scores[near], rtol=0, atol=1e-3)
    assert changed == num, "the injections should change the scores of the windows they reach"


def test_the_stride_is_the_models_own_when_given(dataset_args):
    config = ValidationConfig(num_examples=40, batch_size=20, stride_seconds=0.125, stream_segment_seconds=64.0)
    bank = ValidationBank(logit_model(), dataset_args, config)
    bank.generate_stream_noise()
    assert bank.noise_streams["stride_seconds"] == 0.125
    assert bank.background().livetime_seconds == pytest.approx(39 * 0.125 + 1.0)


def _cluster_earliest_of_ties(time, stat, cluster_seconds):
    """The clustering's earlier tie rule (of equal statistics, the earliest), for comparison."""
    from gravyflow.src.detection.stream import _range_max
    order = np.argsort(time, kind="stable")
    t, s = time[order], stat[order]
    rank = np.empty(len(t), dtype=np.int64)
    rank[np.lexsort((-np.arange(len(t)), s))] = np.arange(len(t))
    lo = np.searchsorted(t, t - cluster_seconds, side="left")
    hi = np.searchsorted(t, t + cluster_seconds, side="right")
    return order[rank == _range_max(rank, lo, hi)]


def test_a_run_of_equal_scores_triggers_at_its_middle():
    # A score that holds while every window holds the peak (a window's largest SNR) is one
    # run: its trigger stands at the run's middle (the earlier of two middles), not its start.
    times = np.arange(12) * 0.25
    for run, middle in (((3, 4, 5), 4), ((3, 4, 5, 6), 4), ((5,), 5)):
        scores = np.zeros(12)
        scores[list(run)] = 7.0
        keep = cluster(times, scores, 1.0)
        assert times[keep][scores[keep] == 7.0].tolist() == [times[middle]]
        stat, when = loudest_within(times[list(run)], scores[list(run)], [times[run[0]] + 0.3], 1.0)
        assert stat[0] == 7.0 and when[0] == times[middle]
    # Distinct statistics: exactly the matched filter's clustering as it was.
    rng = np.random.default_rng(5)
    for _ in range(20):
        t = np.sort(rng.uniform(0, 100, 400))
        s = rng.permutation(400).astype(np.float64)
        np.testing.assert_array_equal(cluster(t, s, 1.0), _cluster_earliest_of_ties(t, s, 1.0))
