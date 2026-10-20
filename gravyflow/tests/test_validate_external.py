"""
A released-weights model inside the validator: an external model's adapter, run by
``gf.ExternalRunner`` in its own process, validated in stream mode at its own stride
on raw strain, its statistic as published.

The adapter here is a toy in numpy that conditions each segment as a whole (its
normalisation is the segment's), as the released searches do, installed into a
temporary directory as ``pip install --target`` would install it, and run by this
interpreter, isolated. Its weights are served at a ``file://`` URL.
"""

import dataclasses
import hashlib
import sys
import textwrap

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.store.files import RemoteFile
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.stream import StreamBackground, cluster, loudest_within

RATE = 2048.0

TOY = textwrap.dedent('''
    from types import SimpleNamespace

    import numpy as np

    # At 2048 Hz: windows of 1 s every 0.25 s, after and before a 0.5 s edge.
    EDGE, WINDOW, STEP = 1024, 2048, 512


    class Released:
        def prepare(self, fetch):
            self.scale = float(np.load(fetch("toy/scale.npy")))

        def preprocess(self, segment):
            return segment.strain

        def _windows(self, segment):
            x = segment.strain - segment.strain.mean(axis=1, keepdims=True)
            x = x / x.std(axis=1, keepdims=True)  # the whole segment's normalisation
            starts = np.arange(EDGE, x.shape[1] - EDGE - WINDOW + 1, STEP)
            stats = np.array([self.scale * np.log(np.mean(x[:, s:s + WINDOW] ** 2)) for s in starts])
            return segment.t0 + (starts + WINDOW / 2) / segment.sample_rate_hertz, stats

        def scores(self, segment):
            time, stat = self._windows(segment)
            return SimpleNamespace(time=time, stat=stat)

        def search(self, segment):
            time, stat = self._windows(segment)
            keep = stat > 0.5
            return SimpleNamespace(time=time[keep], stat=stat[keep], var=np.full(int(keep.sum()), 0.1))
''')

LOCK = b"toy environment: numpy\n"


class Spy(gf.ExternalRunner):
    """The runner, keeping every segment it scored and what the adapter gave for it."""

    calls = []

    def scores(self, segments):
        found = super().scores(segments)
        type(self).calls.append(list(zip(segments, found)))
        return found


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    """The toy released model's runner: its adapter installed, its weights published."""
    root = tmp_path_factory.mktemp("released")
    target = root / "target"
    info = target / "toy_released-0.1.0.dist-info"
    info.mkdir(parents=True)
    (target / "toy_released.py").write_text(TOY)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-released\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text("[gravyflow.external]\ntoy-released = toy_released:Released\n")
    published = root / "published" / "scale.npy"
    published.parent.mkdir()
    np.save(published, np.float64(3.0))
    data = published.read_bytes()
    weights = RemoteFile("toy/scale.npy", published.as_uri(), hashlib.sha256(data).hexdigest(), len(data),
                         licence=None, redistributable=False)
    (root / "env.lock").write_bytes(LOCK)
    card = gf.ExternalCard(
        name="toy-released", adapter="toy-released", adapter_version="0.1.0",
        environment_lock=hashlib.sha256(LOCK).hexdigest(),
        input=gf.InputContract(RATE, ("L1",), window_seconds=1.0, stride_seconds=0.25,
                               whitening="none: each segment normalised by its own deviation", context_seconds=1.0),
        output=gf.OutputContract("three times the log mean square of the normalised window", signal_column=0,
                                 direction="higher", threshold=0.5, cluster_seconds=0.0),
        weights=(gf.Weights(weights, format="npy", framework="numpy"),),
        training_data="none: the toy is not trained",
    )
    scratch = root / "scratch"
    scratch.mkdir()
    return Spy(card, gf.AdapterEnvironment(sys.executable, root / "env.lock", (target,)), 120, scratch)


@pytest.fixture(scope="module")
def dataset_args():
    return validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=RATE))


@pytest.fixture(scope="module")
def config():
    return ValidationConfig(num_examples=64, batch_size=16, snr_range=(0.0, 15.0), volume_num_injections=8,
                            stream_segment_seconds=64.0)


@pytest.fixture(scope="module")
def validated(tmp_path_factory, released, dataset_args, config):
    path = tmp_path_factory.mktemp("released-validation") / "validation.h5"
    Spy.calls.clear()
    v = Validator.validate(released, "toy-released", dataset_args, config, path)
    return path, v, list(Spy.calls)


def test_a_released_model_streams_at_its_own_stride(validated, released, config):
    path, v, _ = validated
    # Its identity is its card's; the stride its card's.
    assert v.model_sha256 == released.card.fingerprint
    assert float(v.noise_streams["stride_seconds"]) == 0.25
    background = v.background()
    assert isinstance(background, StreamBackground) and background.num_triggers > 0
    data = v.efficiency_data
    assert data["scores"].shape == (64,) and v.volume_data["scores"].shape == (8,)
    found = np.isfinite(data["scores"])
    assert found.any()
    # The found window in force: found_window_seconds, but at least half the stride (recorded).
    assert np.all(np.abs(data["trigger_offsets"][found]) <= float(v.noise_streams["found_window_seconds"]) + 1e-9)
    assert np.all(np.isnan(data["trigger_offsets"][~found]))
    assert v.efficiency_curves([1.0])[1.0].binned.total.sum() == 64
    assert v.sensitive_volume_curve() is not None and v.roc_curves()
    loaded = Validator.load(path)
    np.testing.assert_array_equal(loaded.far_scores, v.far_scores)
    np.testing.assert_array_equal(loaded.efficiency_data["scores"], data["scores"])


def test_the_background_is_the_adapters_own_scores_of_each_stream(validated, released, dataset_args, config):
    # The validator hands each stream's raw strain to the adapter whole, and keeps every
    # window's statistic as the adapter gives it, in stream order.
    _, v, calls = validated
    background = calls[0]
    streams = v.bank._recorded_streams()
    assert len(background) == len(streams) == len(v.noise_streams["lengths"])
    direct = []
    for (segment, scored), stream in zip(background, streams):
        strain, t0 = v.bank.stream_source.stream_strain(stream)
        np.testing.assert_array_equal(segment.strain, strain[:, :segment.strain.shape[1]])
        assert segment.t0 == t0 and segment.detectors == ("L1",)
        (again,) = gf.ExternalRunner.scores(released, [segment])
        np.testing.assert_array_equal(again.stat, scored.stat)
        direct.append(again.stat)
    np.testing.assert_array_equal(v.far_scores, np.concatenate(direct))
    np.testing.assert_array_equal(v.noise_streams["lengths"], [d.size for d in direct])


def test_an_injection_is_found_in_the_adapters_scores_of_its_stretch(validated, config):
    # Each injection's stretch, the signal added, is one segment of the adapter's, starting on
    # the stream's grid: its found statistic is the loudest of the adapter's triggers there
    # within the found window of the merger, which lies half a stretch before its end.
    _, v, calls = validated
    # The background's call, then one per batch of 16 injections, then the volume set's.
    assert [len(call) for call in calls] == [1, 16, 16, 16, 16, 8]
    stretches = [pair for call in calls[1:5] for pair in call]
    found_window = v.bank.stream_found_window_seconds  # at least half the stride
    reach = found_window + config.stream_cluster_seconds
    half = round(config.injection_stretch_seconds * RATE / 2.0) / RATE  # every model's stretch
    offsets = []
    for k, (segment, scored) in enumerate(stretches):
        assert segment.t0 / 0.25 == pytest.approx(round(segment.t0 / 0.25), abs=1e-9), "off the stream's grid"
        merger = segment.t0 + segment.duration_seconds - half
        offsets.append((merger / 0.25) % 1.0)
        near = np.abs(scored.time - merger) <= reach + 1e-9
        keep = cluster(scored.time[near], scored.stat[near], config.stream_cluster_seconds)
        stat, when = loudest_within(scored.time[near][keep], scored.stat[near][keep], [merger],
                                    found_window)
        assert v.efficiency_data["scores"][k] == stat[0], k
        if np.isfinite(stat[0]):
            assert v.efficiency_data["trigger_offsets"][k] == pytest.approx(when[0] - merger, abs=1e-9)
    # The mergers fall anywhere on the stride's grid, as on the stream's.
    assert np.ptp(offsets) > 0.5


def test_a_released_model_resumes_from_its_record(validated, released, dataset_args, config):
    path, v, _ = validated
    Spy.calls.clear()
    again = Validator.validate(released, "toy-released", dataset_args, config, path)
    assert Spy.calls == []
    np.testing.assert_array_equal(again.far_scores, v.far_scores)


def test_what_a_released_model_cannot_be_validated_on_is_refused(released, dataset_args, config):
    with pytest.raises(ValueError, match="stream mode"):
        ValidationBank(released, dataset_args, dataclasses.replace(config, mode="window"))
    with pytest.raises(ValueError, match="its own stride"):
        ValidationBank(released, dataset_args, dataclasses.replace(config, stride_seconds=0.5))
    with pytest.raises(ValueError, match="real events"):
        ValidationBank(released, dataset_args, dataclasses.replace(config, score_real_events=True))
    with pytest.raises(ValueError, match=r"reads \('L1',\) at 2048.0 Hz"):
        ValidationBank(released, validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=1024.0)),
                       config)
    with pytest.raises(ValueError, match=r"reads \('L1',\) at 2048.0 Hz"):
        ValidationBank(released, validation_dataset_args(config=ValidationDatasetConfig(
            sample_rate_hertz=RATE, ifos=[gf.IFO.H1])), config)
    # A probability is ranked as published: the statistic is never refused or rescored.
    ValidationBank(released, dataset_args, config)
    with pytest.raises(ValueError, match="glitch stage"):
        ValidationBank(released, dataset_args, dataclasses.replace(config, score_glitches=True))
    with pytest.raises(ValueError, match="leave merger_offset_seconds unset"):
        ValidationBank(released, dataset_args, dataclasses.replace(config, merger_offset_seconds=0.2))


def _late(released, offset):
    """The toy's runner, its card declaring the merger ``offset`` s after the time its search gives a window."""
    card = dataclasses.replace(released.card, output=dataclasses.replace(
        released.card.output, merger_offset_seconds=offset, merger_range_seconds=(offset - 0.1, offset + 0.1)))
    return gf.ExternalRunner(card, released.environment, 120, released.scratch)


def test_a_released_model_is_timed_where_its_card_places_the_merger(released, dataset_args, config):
    # A released model whose response peaks 0.3 s before the merger, on the time its search
    # gives each window (as a model trained with the merger 0.3 s after that time would):
    # found, its trigger timed at the merger, only when its card declares where its authors
    # place the merger; never without, whatever the merger's place on the stride's grid.
    rng = np.random.default_rng(4)
    for declared, expect_found in ((0.0, False), (0.3, True)):
        runner = released if declared == 0.0 else _late(released, declared)
        bank = ValidationBank(runner, dataset_args, config)
        assert bank.stream_geometry.merger_offset_seconds == declared
        found_window = bank.stream_found_window_seconds
        for merger in 100.0 + rng.uniform(0.0, 0.25, 40):
            time = np.arange(320, 480) * 0.25  # the search's windows, on its grid
            stat = 10.0 - np.abs(time + 0.3 - merger)  # loudest 0.3 s before the merger
            stat_found, when = bank._released_found(gf.WindowScores(time, stat), merger)
            assert np.isfinite(stat_found) == expect_found, (declared, merger)
            if expect_found:
                assert abs(when) <= found_window + 1e-9
    with pytest.raises(ValueError, match="merger range"):
        dataclasses.replace(released.card.output, merger_offset_seconds=0.3, merger_range_seconds=(0.0, 0.2))


def test_the_record_keeps_the_declared_merger_offset(validated):
    _, v, _ = validated
    assert float(v.noise_streams["merger_offset_seconds"]) == 0.0
