"""
Real-noise labels and the segment cache's provenance.

``START_GPS_TIME`` is the GPS time of the first sample of the onsource window
returned after cropping, counted from the first sample the source returned (never
the nominal segment start). The segment file is keyed by everything a stored
segment depends on, and each stored segment carries its own first-sample time,
rate, source and layout, so a cached segment is never served under another
configuration or labelled from a nominal time. Content fingerprints read every
parameter that determines the data.

The strain here is a fake GWOSC fetch: the samples of a segment on the absolute
grid at the requested rate, each valued its index from the segment's first grid
sample, so a window's first sample decodes to its GPS time.
"""

import inspect
import logging
from contextlib import closing, contextmanager
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

import gravyflow as gf
from gravyflow.src.dataset.acquisition import base
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer, segment_key
from gravyflow.src.validate.provenance import UnfingerprintableError, canonical, config_fingerprint

RATE = 256.0
ORIGIN = 1238175000.0  # inside O3; the fake's values count from each segment's start
ONSOURCE, CROP, OFFSOURCE, BATCH = 1.0, 0.5, 4.0, 8
#: One row, a piece per detector, starting off the sample grid and off whole seconds.
ROW = [[ORIGIN + 0.3, ORIGIN + 64.3], [ORIGIN + 100.7, ORIGIN + 164.7]]
IFOS = [gf.IFO.H1, gf.IFO.L1]


class GridFetch:
    """
    A fake GWOSC fetch: the samples of [start, end) on the absolute grid at the
    requested rate, from the first grid sample at or after ``start``
    (``first_grid_gps``), each valued its index from that sample. Records its calls.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, start, end, ifo, frame_type, channel, sample_rate_hertz=None):
        self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        values = np.arange(stop - first, dtype=np.float32)
        return TimeSeries(values, t0=first / rate, sample_rate=rate, channel=f"{ifo.name}:FAKE-GRID")


def first_grid_gps(start, rate=RATE):
    return np.ceil(start * rate - 1e-6) / rate


def refuse_fetch(*args, **kwargs):
    raise AssertionError("fetched although the segment was cached")


@contextmanager
def fetching(fetch):
    with patch.object(BaseDataObtainer, "get_segment_data_static", staticmethod(fetch)):
        yield fetch


def obtainer(**kwargs):
    arguments = dict(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                     observing_runs=gf.ObservingRun.O3, cache_segments=False, prefetch_segments=0)
    arguments.update(kwargs)
    built = gf.IFODataObtainer(**arguments)
    built.augmentations = []
    return built


def batches(source, mode, num_batches=2, seed=3):
    """``num_batches`` batches of the one row ROW, as the obtainer serves them."""
    source.valid_segments = np.asarray([ROW], dtype=np.float64)
    generator = source.get_onsource_offsource_chunks(
        sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE, crop_duration_seconds=CROP,
        offsource_duration_seconds=OFFSOURCE, num_examples_per_batch=BATCH, ifos=IFOS,
        scale_factor=1.0, seed=seed, sampling_mode=mode, crop=True,
    )
    return [next(generator) for _ in range(num_batches)]


def decoded_start_gps(onsource):
    """The GPS time of each (example, detector)'s first returned sample, decoded."""
    first = np.rint(np.asarray(onsource, dtype=np.float64)[..., 0])
    piece_starts = np.array([first_grid_gps(start) for start, _ in ROW])
    return piece_starts[None, :] + first / RATE


class TestStartGpsTime:

    @pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
    def test_is_the_first_returned_onsource_sample(self, mode):
        with fetching(GridFetch()):
            served = batches(obtainer(), mode)
        for batch in served:
            onsource = batch[gf.ReturnVariables.ONSOURCE]
            assert onsource.shape == (BATCH, len(IFOS), int(ONSOURCE * RATE))
            labels = np.asarray(batch[gf.ReturnVariables.START_GPS_TIME])
            assert labels.dtype == np.float64
            # Each window is consecutive samples of its own piece.
            steps = np.diff(np.asarray(onsource, dtype=np.float64), axis=-1)
            assert np.all(np.abs(steps - 1.0) < 0.25)
            error = labels - decoded_start_gps(onsource)
            assert np.max(np.abs(error)) <= 1e-6, f"label minus first sample: {error.min()} to {error.max()} s"

    def test_is_one_definition_with_the_crop(self):
        spec = gf.WindowSpec.from_params(sample_rate_hertz=RATE, onsource_duration_seconds=ONSOURCE,
                                         offsource_duration_seconds=OFFSOURCE, crop_duration_seconds=CROP)
        padded = np.arange(spec.num_onsource_samples, dtype=np.float32)[None, :]
        first_kept = float(np.asarray(gf.crop_samples(padded, ONSOURCE, RATE))[0, 0])
        assert spec.num_crop_samples == first_kept == CROP * RATE
        assert spec.onsource_start_gps(np.array([ORIGIN])) == pytest.approx(ORIGIN + CROP, abs=1e-9)


class TestSegmentCache:

    def test_warm_build_equals_cold(self, tmp_path):
        cold = obtainer(cache_segments=True)
        cold.file_path = tmp_path / "segments.hdf5"
        with fetching(GridFetch()) as fetch:
            cold_batches = batches(cold, gf.SamplingMode.GRID)
        assert len(fetch.calls) == len(IFOS)

        with h5py.File(cold.file_path, "r") as stored:
            for ifo, (start, end) in zip(IFOS, ROW):
                attrs = stored[segment_key(ifo, start, end)].attrs
                assert attrs["t0"] == pytest.approx(first_grid_gps(start), abs=1e-9)
                assert attrs["sample_rate_hertz"] == RATE
                assert attrs["layout"] == base.SEGMENT_LAYOUT_VERSION
                assert "FAKE-GRID" in attrs["source"]

        warm = obtainer(cache_segments=True)
        warm.file_path = cold.file_path
        with fetching(refuse_fetch):
            warm_batches = batches(warm, gf.SamplingMode.GRID)
        for cold_batch, warm_batch in zip(cold_batches, warm_batches):
            for key in (gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE,
                        gf.ReturnVariables.START_GPS_TIME):
                np.testing.assert_array_equal(np.asarray(cold_batch[key]), np.asarray(warm_batch[key]))

    def test_a_segment_stored_without_provenance_is_fetched_again(self, tmp_path, caplog):
        source = obtainer(cache_segments=True)
        source.file_path = tmp_path / "segments.hdf5"
        (start, end), ifo = ROW[0], IFOS[0]
        key = segment_key(ifo, start, end)
        with closing(h5py.File(source.file_path, "a")) as stored:
            stored.create_dataset(key, data=np.full(int(64 * RATE), 7.0, np.float32))  # layout 2: bare samples
        with fetching(GridFetch()) as fetch, caplog.at_level(logging.WARNING):
            segment = source.get_segment(start, end, RATE, ifo, key)
        assert fetch.calls == [(ifo.name, start, end)]
        assert segment.t0 == pytest.approx(first_grid_gps(start), abs=1e-9)
        assert "fetching it again" in caplog.text

    def test_configurations_never_share_a_file(self, tmp_path, monkeypatch):
        def path(source=None, rate=RATE, group="train"):
            return (source or obtainer()).generate_file_path(rate, group, tmp_path)

        paths = {
            "default": path(),
            "rate": path(rate=2 * RATE),
            "group": path(group="test"),
            "flags": path(obtainer(gwosc_flags=gf.GWOSC_DATA_FLAGS)),
            "runs": path(obtainer(observing_runs=gf.ObservingRun.O2)),
            "longest segment": path(obtainer(max_segment_duration_seconds=1024.0)),
            "run bounds": path(obtainer(overrides={"end_gps_times": ORIGIN + 1.0e6})),
        }
        monkeypatch.setattr(base, "SEGMENT_LAYOUT_VERSION", "another-layout")
        paths["layout"] = path()
        assert len(set(paths.values())) == len(paths), paths
        assert path() == paths["layout"]  # deterministic


#: Each data-determining parameter of the real-noise obtainers, and another value.
OBTAINER_PERTURBATIONS = {
    "data_labels": [gf.DataLabel.GLITCHES],
    "observing_runs": gf.ObservingRun.O2,
    "segment_order": gf.SegmentOrder.CHRONOLOGICAL,
    "max_segment_duration_seconds": 1024.0,
    "saturation": 4.0,
    "overrides": {"end_gps_times": ORIGIN + 1.0e6},
    "event_types": [gf.EventConfidence.MARGINAL],
    "augmentations": [gf.SignReversal()],
    "balanced_glitch_types": True,
    "event_names": ["GW150914"],
    "gwosc_flags": gf.GWOSC_DATA_FLAGS,
    "split_spec": gf.SplitSpec(version="fold-1"),
}
#: Data-determining parameters with one buildable value (public runs publish BEST only;
#: training on real events is refused for now).
SINGLE_VALUED = {"data_quality", "train_on_real_events"}
#: Parameters that change how data is fetched or logged, never the data.
NOT_DATA = {"force_acquisition", "cache_segments", "logging_level", "prefetch_segments"}
#: Parameters a NOISE obtainer ignores (they select transients).
TRANSIENT_ONLY = {"event_types", "balanced_glitch_types", "event_names"}


class TestFingerprint:

    def base_arguments(self):
        return dict(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                    observing_runs=gf.ObservingRun.O3)

    def test_every_obtainer_parameter_is_classified(self):
        parameters = set(inspect.signature(gf.IFODataObtainer).parameters)
        assert parameters == set(OBTAINER_PERTURBATIONS) | SINGLE_VALUED | NOT_DATA
        transient = {**self.base_arguments(), "data_labels": [gf.DataLabel.GLITCHES]}
        assert SINGLE_VALUED <= set(gf.IFODataObtainer(**transient).specification())

    @pytest.mark.parametrize("name", sorted(OBTAINER_PERTURBATIONS))
    def test_changes_with_every_data_parameter(self, name):
        arguments = self.base_arguments()
        if name in TRANSIENT_ONLY:
            arguments["data_labels"] = [gf.DataLabel.GLITCHES]
        reference = config_fingerprint(gf.IFODataObtainer(**arguments))
        arguments[name] = OBTAINER_PERTURBATIONS[name]
        assert config_fingerprint(gf.IFODataObtainer(**arguments)) != reference

    @pytest.mark.parametrize("name", sorted(NOT_DATA))
    def test_ignores_how_data_is_fetched(self, name):
        changed = {"force_acquisition": True, "cache_segments": False,
                   "logging_level": logging.DEBUG, "prefetch_segments": 0}
        reference = config_fingerprint(gf.IFODataObtainer(**self.base_arguments()))
        arguments = {**self.base_arguments(), name: changed[name]}
        assert config_fingerprint(gf.IFODataObtainer(**arguments)) == reference

    def test_noise_obtainer_reads_its_data_obtainer(self):
        # O1 with the default flags and O3 with DATA alone once shared a fingerprint:
        # the data obtainer was read as its type's name.
        def noise(**kwargs):
            return gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1],
                                    ifo_data_obtainer=gf.IFODataObtainer(**{**self.base_arguments(), **kwargs}))
        assert config_fingerprint(noise(observing_runs=gf.ObservingRun.O1)) \
            != config_fingerprint(noise(gwosc_flags=gf.GWOSC_DATA_FLAGS))

        fields = set(inspect.signature(gf.NoiseObtainer).parameters)
        reference = config_fingerprint(gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1]))
        changed = {"ifo_data_obtainer": noise(observing_runs=gf.ObservingRun.O1).ifo_data_obtainer,
                   "ifos": [gf.IFO.L1], "noise_type": gf.NoiseType.PSEUDO_REAL,
                   "noise_profile": gf.NoiseProfile.O3_LOW}
        # The split is the data obtainer's (its split_spec), read with it.
        assert fields == set(changed) | {"data_directory_path"}
        for name, value in changed.items():
            other = gf.NoiseObtainer(**{"noise_type": gf.NoiseType.REAL, "ifos": [gf.IFO.H1], name: value})
            assert config_fingerprint(other) != reference, name
        # Where segment files are kept is not what they hold.
        moved = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1], data_directory_path="elsewhere")
        assert config_fingerprint(moved) == reference

    def test_refuses_what_it_cannot_read(self):
        with pytest.raises(UnfingerprintableError, match="builtins.object"):
            canonical(object())
