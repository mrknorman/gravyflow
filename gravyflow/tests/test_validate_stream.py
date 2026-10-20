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
    for bad in ({"mode": "windows"}, {"stride_seconds": 0.0}, {"found_window_seconds": 0.0},
                {"cluster_seconds": 0.19}, {"found_window_seconds": 0.6}):
        with pytest.raises(ValueError):
            ValidationConfig(**bad)
    # Window mode clusters as it always has.
    assert ValidationConfig(mode="window", cluster_seconds=0.0).cluster_seconds == 0.0
    assert ValidationConfig(cluster_seconds=0.2).stream_cluster_seconds == 0.2


def test_the_stream_background_refuses_a_cluster_shorter_than_two_found_windows():
    from gravyflow.src.validate.metrics import ScoreStream
    streams = [ScoreStream(np.arange(8.0), 0.25, 1.0)]
    assert StreamBackground(streams, 0.2, 0.1).cluster_seconds == 0.2
    with pytest.raises(ValueError, match="apart"):
        StreamBackground(streams, 0.19, 0.1)


def test_the_validator_streams_by_default(validated, stream_config, tmp_path):
    path, v = validated
    # The background: num_examples windows one stride apart, their triggers its count.
    assert v.far_scores.shape == (64,) and v.noise_streams["stride_seconds"] == 0.25
    background = v.background()
    assert isinstance(background, StreamBackground)
    assert background.livetime_seconds == pytest.approx(63 * 0.25 + 1.0 - 2.0)
    times = np.arange(64) * 0.25
    keep = cluster(times, v.far_scores, 1.0)
    assert background.num_triggers == np.sum((times[keep] >= 1.0) & (times[keep] <= 63 * 0.25 - 1.0))
    # Each injection: its found statistic (a trigger's score, or -inf), its trigger's time from
    # the merger within the found window, its merger's time and its source's records.
    data = v.efficiency_data
    assert {"scores", "trigger_offsets", "merger_times", "snrs", "hpeak", "hrss", "mass1", "mass2",
            "in_band_durations", "whitened_peaks", "bns_ranges"} <= set(data)
    assert data["scores"].shape == (64,) and data["bns_ranges"].shape == (64, 1)
    found = np.isfinite(data["scores"])
    assert found.any()
    # The found window: 0.1 s, but at least half the 0.25 s stride, recorded.
    assert v.noise_streams["found_window_seconds"] == 0.125
    assert background.found_window_seconds == 0.125
    assert np.all(np.abs(data["trigger_offsets"][found]) <= 0.125 + 1e-9)
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
    assert bank.noise_streams["found_window_seconds"] == 0.1  # half the stride is less than 0.1 s
    assert bank.background().livetime_seconds == pytest.approx(39 * 0.125 + 1.0 - 2.0)


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


def test_the_default_stride_is_a_whole_number_of_samples_and_no_longer_than_the_window():
    from gravyflow.src.dataset.config import WindowSpec
    from gravyflow.src.validate.stream import StreamGeometry
    # A quarter of a 1.5 s window at 1000 Hz is 375 samples; at 1001 Hz, 375.375, rounded to 375.
    assert StreamGeometry.of(WindowSpec(1000.0, 1.5, 2.0, 0.25), None).stride_samples == 375
    geometry = StreamGeometry.of(WindowSpec(1001.0, 1.5, 2.0, 0.25), None)
    assert geometry.stride_samples == 375 and geometry.stride_seconds == 375 / 1001.0
    with pytest.raises(ValueError, match="whole number"):
        StreamGeometry.of(WindowSpec(1001.0, 1.5, 2.0, 0.25), 0.375)
    assert StreamGeometry.of(WindowSpec(1000.0, 1.5, 2.0, 0.25), 1.5).stride_samples == 1500
    with pytest.raises(ValueError, match="at most"):
        StreamGeometry.of(WindowSpec(1000.0, 1.5, 2.0, 0.25), 1.6)


def test_the_stream_reader_keeps_the_stretches_in_float64():
    from gravyflow.src.dataset.config import WindowSpec
    from gravyflow.src.validate.slides import SlideStream
    from gravyflow.src.validate.stream import Placement, StreamGeometry, StreamReader

    strain = 1e-21 * (1.0 + 1e-9 * np.arange(2 * 4000, dtype=np.float64).reshape(2, 4000))

    class Source:
        def stream_strain(self, stream):
            return strain, 0.0

    geometry = StreamGeometry.of(WindowSpec(100.0, 1.0, 2.0, 0.25), None)
    reader = StreamReader(Source(), [SlideStream((0.0, 40.0), 0, 10, 0, 4000)],
                          Placement(np.array([0]), np.array([2000])), geometry, 500, None)
    batch = next(reader(1))
    on = batch[RV.ONSOURCE]
    assert on.dtype == np.float64 and batch[RV.OFFSOURCE].dtype == np.float64
    np.testing.assert_array_equal(on[0], strain[:, 1500:2500])


def test_the_synthetic_segments_are_as_many_as_the_budget_needs(dataset_args):
    # A budget of exactly two segments' windows draws two segments, not three.
    probe = ValidationBank(logit_model(), dataset_args, ValidationConfig(num_examples=8, stream_segment_seconds=64.0))
    geometry = probe.stream_geometry
    per = geometry.num_windows(int(round(64.0 * geometry.sample_rate_hertz)))
    for num, segments in ((2 * per, 2), (2 * per + 1, 3), (1, 1)):
        bank = ValidationBank(logit_model(), dataset_args,
                              ValidationConfig(num_examples=num, batch_size=8, stream_segment_seconds=64.0))
        assert len(bank.stream_source.dataset) == segments, num


def test_injections_are_placed_only_where_the_background_scored_its_windows():
    # A stream cut short at the budget scores 50 windows of its 100,000 samples: every merger's
    # found test reads only those windows, and its triggers none dropped at the ends.
    from gravyflow.src.dataset.config import WindowSpec
    from gravyflow.src.validate.slides import SlideStream
    from gravyflow.src.validate.stream import StreamGeometry, place_injections

    geometry = StreamGeometry.of(WindowSpec(100.0, 1.0, 2.0, 0.25), None)
    streams = [SlideStream((0.0, 1000.0), 0, 50, 0, 100_000), SlideStream((0.0, 1000.0), 1, 400, 0, 100_000)]
    placement = place_injections(2000, streams, 400, np.random.default_rng(3), geometry, 1.1)
    for k in range(len(placement)):
        stream, m = streams[int(placement.stream[k])], int(placement.sample[k])
        first, last = geometry.windows_near(m, 1.1)
        assert 0 <= first and last <= stream.num_windows - 1, (k, first, last)
        assert 400 <= m <= stream.num_samples - 400
    assert set(placement.stream.tolist()) == {0, 1}
    with pytest.raises(ValueError, match="scored windows"):
        place_injections(1, [SlideStream((0.0, 1000.0), 0, 10, 0, 100_000)], 400, np.random.default_rng(3),
                         geometry, 1.1)


def test_an_injections_labels_do_not_depend_on_the_models_window():
    # Two models of different offsource lengths: the same injection at the same place gets the
    # same SNR (measured, distance-scaled), hrss and hpeak (SNR-scaled): its stretch and spectrum
    # are the same whatever the model's window.
    def bank_of(offsource):
        args = validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=2048.0,
                                                                      offsource_duration_seconds=offsource))
        probe = ValidationBank(logit_model(), args, ValidationConfig(num_examples=8, stream_segment_seconds=64.0))
        per = probe.stream_geometry.num_windows(int(round(64.0 * 2048.0)))  # one whole segment scored
        bank = ValidationBank(logit_model(), args, ValidationConfig(num_examples=per, batch_size=32,
                                                                    stream_segment_seconds=64.0))
        bank.generate_stream_noise()
        return bank

    def labels(bank, type_, value, outputs):
        g = bank._generator()
        g.scaling_method = gf.ScalingMethod(value=value, type_=type_)
        dataset, placement, half, _ = bank.local_dataset("vol", g, 4, [RV.ONSOURCE, *outputs])
        _, y = dataset[0]
        return placement, {o.name: np.asarray(y[o.name], dtype=np.float64)[:4] for o in outputs}

    banks = [bank_of(16.0), bank_of(8.0)]
    snr = [gf.Distribution(min_=8.0, max_=12.0, type_=gf.DistributionType.UNIFORM), gf.ScalingTypes.SNR,
           [gf.ScalingTypes.HRSS, gf.ScalingTypes.HPEAK]]
    distance = [np.array([300.0, 500.0, 700.0, 900.0]), gf.WaveformParameters.CHIRP_DISTANCE_MPC,
                [gf.ScalingTypes.SNR]]
    for value, type_, outputs in (snr, distance):
        (p0, a), (p1, b) = (labels(bank, type_, value, outputs) for bank in banks)
        np.testing.assert_array_equal(p0.sample, p1.sample)
        for name in a:
            assert np.all(np.isfinite(a[name]) & (a[name] > 0)), name
            np.testing.assert_allclose(a[name], b[name], rtol=1e-6, err_msg=name)


def test_the_found_window_is_at_least_half_the_stride():
    from gravyflow.src.validate.stream import found_window
    assert found_window(0.1, 0.25) == 0.125 and found_window(0.1, 0.5) == 0.25 and found_window(0.1, 0.125) == 0.1
    # A perfectly timed, loud signal: its loudest window is the one nearest its merger, within
    # half a stride of it, so it is always found, at any stride.
    rng = np.random.default_rng(9)
    for stride in (0.0625, 0.25, 0.5):
        times = np.arange(2000) * stride
        mergers = rng.uniform(20.0, times[-1] - 20.0, 500)
        found = []
        for merger in mergers:
            scores = -np.abs(times - merger)  # loudest nearest the merger
            keep = cluster(times, scores, 1.0)
            found.append(np.isfinite(loudest_within(times[keep], scores[keep], [merger],
                                                    found_window(0.1, stride))[0][0]))
        assert np.all(found), stride


def test_a_declared_merger_offset_times_every_trigger():
    # A model trained with the merger 0.3 s after its onsource's centre responds loudest to the
    # window whose centre lies 0.3 s before the merger. Timed at the centre, it is never found
    # (0.3 s is beyond the found window, wherever the merger falls on the grid); timed where it
    # declares the merger, it always is, its trigger at the merger.
    from gravyflow.src.dataset.config import WindowSpec
    from gravyflow.src.validate.stream import StreamGeometry, found_in_stretch, found_window, local_windows

    window = WindowSpec(2048.0, 1.0, 2.0, 0.25)
    stride = 0.25
    w = found_window(0.1, stride)
    rng = np.random.default_rng(11)
    for offset, expect in ((None, False), (0.3, True)):
        geometry = StreamGeometry.of(window, stride) if offset is None else StreamGeometry.of(window, stride, offset)
        centre = StreamGeometry.of(window, stride)  # where the response is set: the onsource's centre
        half = geometry.local_half_samples(w + 1.0)
        for sample in rng.integers(200_000, 200_000 + 512, 40):
            index, starts = local_windows(geometry, half, int(sample), w + 1.0)
            assert starts.min() >= 0 and starts.max() + geometry.footprint_samples <= 2 * half
            scores = 10.0 - np.abs(centre.window_time(index) + 0.3 - sample / 2048.0)
            stat, when = found_in_stretch(geometry, index, scores, int(sample), 1.0, w)
            assert np.isfinite(stat) == expect, (offset, sample)
            if expect:
                assert abs(when) <= w + 1e-9


def test_without_a_declaration_a_window_stands_at_its_onsource_centre(dataset_args):
    # The default is unchanged: every window's time, the stretch read around a merger and the
    # latency's wait are the centre's, and the record says so.
    from gravyflow.src.dataset.config import WindowSpec
    from gravyflow.src.validate.stream import StreamGeometry
    from gravyflow.src.validate.timing import algorithmic_wait_seconds

    window = WindowSpec(2048.0, 1.0, 2.0, 0.25)
    geometry = StreamGeometry.of(window, 0.25)
    assert geometry == StreamGeometry.of(window, 0.25, 0.0) and geometry.merger_offset_seconds == 0.0
    index = np.arange(50)
    np.testing.assert_array_equal(geometry.window_time(index),
                                  (index * geometry.stride_samples + geometry.centre_samples) / 2048.0)
    assert geometry.local_half_samples(1.125) == int(window.num_offsource_samples
                                                     + np.ceil(window.num_onsource_samples / 2.0)
                                                     + np.ceil(1.125 * 2048.0) + 1)
    assert algorithmic_wait_seconds(geometry, 1.0) == window.num_onsource_samples / 2.0 / 2048.0 + 1.0
    config = ValidationConfig(num_examples=40, batch_size=20, stream_segment_seconds=64.0)
    bank = ValidationBank(logit_model(), dataset_args, config)
    assert bank.stream_geometry == StreamGeometry.of(bank._noise_dataset.window_spec, None)
    declared = ValidationBank(logit_model(), dataset_args,
                              dataclasses.replace(config, merger_offset_seconds=0.2)).stream_geometry
    assert declared.merger_offset_seconds == 0.2
    np.testing.assert_allclose(declared.window_time(index), bank.stream_geometry.window_time(index) + 0.2)
    bank.generate_stream_noise()
    assert bank.noise_streams["merger_offset_seconds"] == 0.0
