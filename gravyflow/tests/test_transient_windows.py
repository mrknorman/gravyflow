"""
Transient windows: their geometry, their labels and their data quality.

A transient's window is read from one stretch of strain: the stored offsource
followed by the stored onsource, the transient on its stored onsource's centre
sample. Every request, from the download or from any cache read path, takes its
padded onsource centred on that sample and its offsource ending where the padded
onsource begins, as for noise. ``START_GPS_TIME`` is the GPS time of the first
onsource sample returned after cropping, as for noise, never the transient's time.

The strain is a fake GWOSC fetch: the samples of [start, end) on the absolute grid
at the requested rate, each valued its GPS time minus ``REF``, so a returned sample
decodes to its GPS time.
"""

import dataclasses
import importlib.util
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

import gravyflow as gf
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
from gravyflow.src.dataset.acquisition.cache import DiskCache
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.config import TransientDefaults, WindowSpec
from gravyflow.src.dataset.features import glitch_cache
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.glitch_cache import TransientCache, generate_transient_cache_path
from gravyflow.src.utils.gps import gps_to_key
from gravyflow.tests.conftest import catalogued_events

RV = gf.ReturnVariables
REF = 1240000000.0  # inside O3; every sample is valued its GPS time minus REF
STORED_RATE = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
RATE = 2048.0
WINDOW = dict(
    sample_rate_hertz=RATE,
    onsource_duration_seconds=1.0,
    crop_duration_seconds=0.5,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=3,
    scale_factor=1.0,
)
SPEC = WindowSpec.from_params(
    sample_rate_hertz=RATE, onsource_duration_seconds=1.0,
    crop_duration_seconds=0.5, offsource_duration_seconds=16.0,
)
#: Three glitches, off the sample grid at every rate.
TIMES = REF + np.array([10.3701, 150.01234, 290.90007])
#: The one split holding every record's footprint, which every test reads.
SPLIT = gf.DEFAULT_SPLIT.assign(TIMES[0] - 48.2, TIMES[-1] + 48.2)
assert SPLIT != gf.EXCLUDED
#: Times inside one data-quality segment covering every record.
SCIENCE = np.array([[REF - 1000.0, REF + 1000.0]])


def records(times=TIMES, seen_in=(gf.IFO.L1,)):
    return [
        TransientSegment(
            gps_key=gps_to_key(float(t)),
            transient_gps_time=float(t),
            start_gps_time=float(t) - 48.2,
            end_gps_time=float(t) + 48.2,
            label=gf.DataLabel.GLITCHES,
            kind=GlitchType.BLIP,
            observing_run=gf.ObservingRun.O3,
            seen_in=list(seen_in),
        )
        for t in times
    ]


class GridFetch:
    """
    A fake GWOSC fetch: the samples of [start, end) on the absolute grid at the
    requested rate, from the first grid sample at or after ``start``, each valued
    its GPS time minus ``ref``. ``bad`` maps a detector name to the times whose
    download fails (raises) or, with ``nan``, returns NaN around them.
    """

    def __init__(self, bad=None, nan=False, ref=REF):
        self.calls = []
        self.bad = bad or {}
        self.nan = nan
        self.ref = ref

    def __call__(self, start, end, ifo, frame_type, channel, sample_rate_hertz=None):
        self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        times = np.arange(first, stop, dtype=np.float64) / rate
        values = times - self.ref
        for t in self.bad.get(ifo.name, ()):
            if start <= t <= end:
                if not self.nan:
                    raise ValueError(f"no {ifo.name} data at {t}")
                values[np.abs(times - t) < 2.0] = np.nan
        return TimeSeries(values, t0=first / rate, sample_rate=rate, channel=f"{ifo.name}:FAKE-GRID")


@contextmanager
def offline(fetch=None, catalogue=None, science=SCIENCE, vetoes=None):
    """
    The glitch catalogue, its data quality (usable segments, for every detector or by
    detector), each detector's glitch vetoes (``vetoes``: by detector, default none)
    and GWOSC strain, offline.
    """
    fetch = fetch or GridFetch()
    vetoes = vetoes or {}
    with mock.patch("gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
                    return_value=catalogue if catalogue is not None else records()), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times",
                           lambda self, ifo: science[ifo] if isinstance(science, dict) else science), \
         mock.patch.object(gf.TransientDataObtainer, "get_glitch_veto_segments",
                           lambda self, ifo: np.reshape(vetoes.get(ifo, np.empty((0, 2))), (-1, 2))), \
         mock.patch.object(BaseDataObtainer, "get_segment_data_static", staticmethod(fetch)), \
         catalogued_events():
        yield fetch


def obtainer(directory):
    obtainer = gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3,
        augmentations=[],
    )
    obtainer.data_directory = directory
    return obtainer


def first_batch(directory, ifos, **window):
    return next(obtainer(directory)(ifos=ifos, group=SPLIT, seed=1, **{**WINDOW, **window}))


def transient_times(batch):
    """Each example's window centres, decoded from its padded onsource."""
    onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
    return REF + onsource[..., onsource.shape[-1] // 2]


#: The glitches are seen in L1, the last detector of each set: H1 is read at
#: partner times.
IFO_SETS = [[gf.IFO.L1], [gf.IFO.H1, gf.IFO.L1]]
SEEN = -1


class TestStartGpsTime:
    """START_GPS_TIME is the first returned onsource sample's time (the owner's ruling)."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_obtainer_labels_the_first_cropped_sample(self, tmp_path, ifos):
        with offline():
            batch = first_batch(tmp_path, ifos)
        onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
        first = REF + onsource[..., SPEC.num_crop_samples]
        label = np.asarray(batch[RV.START_GPS_TIME])
        assert label.dtype == np.float64 and label.shape == first.shape
        error = label - first
        assert np.max(np.abs(error)) <= 1.0 / RATE, (
            f"START_GPS_TIME minus the first cropped onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )

    def test_dataset_labels_the_first_cropped_sample(self, tmp_path):
        with offline():
            dataset = gf.Dataset(
                noise_obtainer=obtainer(tmp_path),
                ifos=[gf.IFO.L1],
                sample_rate_hertz=RATE,
                onsource_duration_seconds=1.0,
                crop_duration_seconds=0.5,
                offsource_duration_seconds=16.0,
                num_examples_per_batch=3,
                steps_per_epoch=1,
                scale_factor=1.0,
                group=SPLIT,
                input_variables=[RV.ONSOURCE],
                output_variables=[RV.START_GPS_TIME],
            )
            x, y = dataset[0]
        first = REF + np.asarray(x[RV.ONSOURCE.name], dtype=np.float64)[..., 0]
        error = np.asarray(y[RV.START_GPS_TIME.name]) - first.reshape(np.shape(y[RV.START_GPS_TIME.name]))
        assert np.max(np.abs(error)) <= 1.0 / RATE, (
            f"Dataset START_GPS_TIME minus the first returned onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )


class TestGeometry:
    """One geometry for every transient window (B204)."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_offsource_ends_where_the_padded_onsource_begins(self, tmp_path, ifos):
        with offline():
            batch = first_batch(tmp_path, ifos)
        onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
        offsource = np.asarray(batch[RV.OFFSOURCE], dtype=np.float64)
        assert offsource.shape[-1] == SPEC.num_offsource_samples
        gap = onsource[..., 0] - offsource[..., -1]
        assert np.allclose(gap, 1.0 / RATE, atol=1e-4), f"offsource to onsource: {gap.ravel()} s"
        assert np.allclose(np.diff(offsource, axis=-1), 1.0 / RATE, atol=1e-4), "offsource not contiguous"

    def test_the_transient_is_the_padded_onsource_centre(self, tmp_path):
        with offline():
            batch = first_batch(tmp_path, [gf.IFO.L1])
        served = np.sort(np.unique(np.round(transient_times(batch), 3)))
        error = np.abs(np.sort(transient_times(batch)[:, 0]) - TIMES)
        assert np.max(error) <= 1.0 / RATE, f"centre minus transient time: {error} s ({served})"

    def test_each_detector_is_labelled_its_own_centre(self, tmp_path):
        with offline():
            batch = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        error = transient_times(batch) - np.asarray(batch[RV.TRANSIENT_GPS_TIME])
        assert np.max(np.abs(error)) <= 1.0 / RATE, f"centre minus TRANSIENT_GPS_TIME: {error} s"

    def test_warm_equals_cold(self, tmp_path):
        with offline() as fetch:
            cold = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
            fetched = len(fetch.calls)
            warm = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        assert fetched > 0 and len(fetch.calls) == fetched, "the second read was not a cache hit"
        order = np.argsort(np.asarray(cold[RV.TRANSIENT_GPS_TIME])[:, SEEN])
        again = np.argsort(np.asarray(warm[RV.TRANSIENT_GPS_TIME])[:, SEEN])
        for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME, RV.TRANSIENT_GPS_TIME):
            np.testing.assert_array_equal(
                np.asarray(cold[key])[order], np.asarray(warm[key])[again], err_msg=key.name
            )

    def test_every_cache_read_path_reads_one_geometry(self, tmp_path):
        """
        Each detector's windows, read back by every cache read path: L1's from the
        glitches' rows, H1's from the partners' rows.
        """
        with offline():
            expected = first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        centres = np.asarray(expected[RV.TRANSIENT_GPS_TIME])
        request = dict(sample_rate_hertz=RATE, onsource_duration=SPEC.total_onsource_duration_seconds,
                       offsource_duration=SPEC.offsource_duration_seconds)
        for d, (ifo, prefix) in enumerate((("H1", "partner"), ("L1", "glitch"))):
            on_expected = np.asarray(expected[RV.ONSOURCE])[:, d:d + 1]
            off_expected = np.asarray(expected[RV.OFFSOURCE])[:, d:d + 1]
            cache = TransientCache(generate_transient_cache_path(tmp_path, prefix=prefix), mode="r")
            indices = np.array([cache.index_of(t) for t in centres[:, d]])
            reads = {"get_batch": cache.get_batch(indices, target_ifos=[ifo], **request)[:2]}
            row = list(TransientDefaults.UNIVERSAL_IFOS).index(ifo)
            by_gps = [cache.get_by_gps(t, **request) for t in centres[:, d]]
            reads["get_by_gps"] = (np.stack([r[0][row:row + 1] for r in by_gps]),
                                   np.stack([r[1][row:row + 1] for r in by_gps]))
            cache.enable_chunked_mode(chunk_size=2, **request)
            reads["get_batch_from_chunk"] = cache.get_batch_from_chunk(indices, target_ifos=[ifo])[:2]
            chunked = [cache.get_from_chunk(int(i), target_ifos=[ifo]) for i in indices]
            reads["get_from_chunk"] = (np.stack([r[0] for r in chunked]), np.stack([r[1] for r in chunked]))
            for path, (onsource, offsource) in reads.items():
                np.testing.assert_array_equal(onsource, on_expected, err_msg=f"{ifo} {path} onsource")
                np.testing.assert_array_equal(offsource, off_expected, err_msg=f"{ifo} {path} offsource")


def served(directory, ifos, split=SPLIT, **window):
    """The first batch of a fresh obtainer over every record of ``split``, and the obtainer."""
    reader = obtainer(directory)
    batch = next(reader(ifos=ifos, group=split, seed=1, **{**WINDOW, **window}))
    return batch, reader


def served_times(batch):
    """The transient times served: the centres of the detector the glitches were seen in."""
    return sorted(np.round(np.asarray(batch[RV.TRANSIENT_GPS_TIME])[:, SEEN], 4))


def rounded(times):
    return sorted(np.round(times, 4))


def assert_never_filled(batch):
    for key in (RV.ONSOURCE, RV.OFFSOURCE):
        values = np.asarray(batch[key])
        assert np.all(np.isfinite(values)), f"{key.name} holds non-finite values"
        assert not np.any(np.all(values == 0.0, axis=-1)), f"{key.name} holds a zero-filled row"


class TestDataQuality:
    """Transient windows are checked against data quality, and never zero-filled (B103)."""

    def test_windows_outside_data_quality_are_rejected_and_counted(self, tmp_path):
        # A 5 s gap inside the second record's download, on L1, where it was seen.
        gap = TIMES[1] + 20.0
        science = {
            gf.IFO.H1: SCIENCE,
            gf.IFO.L1: np.array([[REF - 1000.0, gap], [gap + 5.0, REF + 1000.0]]),
        }
        for ifos in IFO_SETS:
            name = "".join(ifo.name for ifo in ifos)
            with offline(science=science) as fetch:
                batch, reader = served(tmp_path / name, ifos, num_examples_per_batch=2)
            assert served_times(batch) == rounded(TIMES[[0, 2]]), name
            assert reader.get_acquisition_stats()["rejections"] == {"data quality": 1}, name
            assert not any(ifo == "L1" and start < TIMES[1] < end for ifo, start, end in fetch.calls), (
                "a rejected record was downloaded"
            )

    def test_a_partner_is_read_in_its_own_data_quality_and_vetoes(self, tmp_path):
        """
        H1, the detector the glitches were not seen in, has gaps and glitches around
        them: every H1 read lies whole in H1's data, away from its glitches and at
        least two record spans from every record, and no record is rejected.
        """
        science = {
            gf.IFO.H1: np.array([[REF - 1000.0, REF - 700.0], [REF - 690.0, REF - 400.0],
                                 [REF + 450.0, REF + 1000.0]]),
            gf.IFO.L1: SCIENCE,
        }
        vetoes = {gf.IFO.H1: np.array([[REF - 560.0, REF - 550.0], [REF + 700.0, REF + 710.0]])}
        with offline(science=science, vetoes=vetoes) as fetch:
            batch, reader = served(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        assert served_times(batch) == rounded(TIMES)
        assert reader.get_acquisition_stats()["rejections"] == {}
        h1 = [(start, end) for ifo, start, end in fetch.calls if ifo == "H1"]
        assert len(h1) == len(TIMES)
        for start, end in h1:
            assert any(a <= start and end <= b for a, b in science[gf.IFO.H1]), f"H1 read {start}-{end} leaves its data"
            assert not any(start < b and a < end for a, b in vetoes[gf.IFO.H1]), f"H1 read {start}-{end} meets a veto"
        centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
        assert np.all(np.abs(centres[:, 0] - centres[:, SEEN]) >= 4 * 48.2), "H1 read near its own glitch"
        assert np.all(np.asarray(batch[RV.DATA_LABEL]) == [[gf.DataLabel.NOISE.value, gf.DataLabel.GLITCHES.value]])
        assert np.all(np.asarray(batch[RV.GLITCH_TYPE])[:, 0] == -1)
        assert np.all(np.asarray(batch[RV.GLITCH_TYPE])[:, SEEN] == GlitchType.BLIP.value)

    @pytest.mark.parametrize("nan", [False, True], ids=["download fails", "NaN strain"])
    def test_a_requested_detector_without_strain_is_rejected_never_zero_filled(self, tmp_path, nan):
        with offline(GridFetch(bad={"L1": [TIMES[1]]}, nan=nan)):
            batch, reader = served(tmp_path, [gf.IFO.H1, gf.IFO.L1], num_examples_per_batch=2)
        assert served_times(batch) == rounded(TIMES[[0, 2]])
        assert_never_filled(batch)
        assert reader.get_acquisition_stats()["rejections"] == {"no data": 1}

    def test_a_detector_a_glitch_was_not_seen_in_is_never_read_at_its_time(self, tmp_path):
        with offline() as fetch:
            served(tmp_path, [gf.IFO.L1])
            assert {ifo for ifo, _, _ in fetch.calls} == {"L1"}, "an unrequested detector was fetched"
            served(tmp_path, [gf.IFO.H1, gf.IFO.L1])
        h1 = [(start, end) for ifo, start, end in fetch.calls if ifo == "H1"]
        assert h1 and not any(start < t < end for start, end in h1 for t in TIMES), (
            "H1 was read at the time of a glitch seen only in L1"
        )
        cache = TransientCache(generate_transient_cache_path(tmp_path, prefix="glitch"), mode="r")
        indices = np.array([cache.index_of(t) for t in TIMES])
        onsource, offsource, _, _ = cache.get_batch(indices, target_ifos=["H1"])
        assert np.all(np.isnan(onsource)) and np.all(np.isnan(offsource)), "a glitch's row holds H1"


def fetches_of(fetch, gps_time):
    """The fetches whose span holds ``gps_time``."""
    return sum(1 for _, start, end in fetch.calls if start < gps_time < end)


def by_time(batch, key):
    """``batch[key]`` in order of the examples' transient times."""
    order = np.argsort(np.asarray(batch[RV.TRANSIENT_GPS_TIME])[:, 0])
    return np.asarray(batch[key])[order]


def weighted(catalogue, weights):
    """``catalogue`` with each record's sampling weight, as class balancing sets it."""
    return [dataclasses.replace(record, weight=w) for record, w in zip(catalogue, weights)]


def within(seconds, call):
    """``call()``, which fails the test if it has not returned within ``seconds``."""
    outcome = {}

    def run():
        try:
            outcome["value"] = call()
        except BaseException as error:  # re-raised in the test's thread
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"no result within {seconds} s: the call spins"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class TestUnavailableRecords:
    """
    A record found without a requested detector's strain is counted once and not
    fetched again in that run; a detector whose fetch failed is never stored as absent.
    """

    def test_a_record_without_strain_is_counted_once_and_fetched_once(self, tmp_path):
        with offline(GridFetch(bad={"L1": [TIMES[1]]})) as fetch:
            reader = obtainer(tmp_path)
            batches = reader(ifos=[gf.IFO.H1, gf.IFO.L1], group=SPLIT, seed=1,
                             **{**WINDOW, "num_examples_per_batch": 2})
            assert served_times(next(batches)) == rounded(TIMES[[0, 2]])
            fetched = fetches_of(fetch, TIMES[1])
            for _ in range(3):  # three more passes over the records
                assert served_times(next(batches)) == rounded(TIMES[[0, 2]])
            stats = reader.get_acquisition_stats()
        assert fetches_of(fetch, TIMES[1]) == fetched, "the unavailable record was fetched every pass"
        assert stats["rejections"] == {"no data": 1} and stats["dropped_segments"] == 1

    def test_a_split_with_no_record_available_is_refused_not_spun(self, tmp_path):
        with offline(GridFetch(bad={"L1": list(TIMES)})) as fetch:
            reader = obtainer(tmp_path)
            batches = reader(ifos=[gf.IFO.L1], group=SPLIT, seed=1, **WINDOW)
            with pytest.raises(ValueError, match=f"split '{SPLIT}'.*no data"):
                next(batches)
        assert all(fetches_of(fetch, t) == 1 for t in TIMES), "each record is fetched once, on L1 alone"
        assert reader.get_acquisition_stats()["rejections"] == {"no data": 3}

    def test_a_weighted_split_is_served_by_the_records_its_first_draws_missed(self, tmp_path):
        # The one servable record is all but never drawn: each pass draws among the
        # records still servable, so it is served once the others are found unavailable.
        catalogue = weighted(records(), [1e-6, 1.0, 1.0])
        with offline(GridFetch(bad={"L1": list(TIMES[1:])}), catalogue=catalogue):
            reader = obtainer(tmp_path)
            batches = reader(ifos=[gf.IFO.L1], group=SPLIT, seed=1, **WINDOW)
            batch = within(60, lambda: next(batches))
        assert set(served_times(batch)) == set(rounded(TIMES[:1]))
        assert reader.get_acquisition_stats()["rejections"] == {"no data": 2}

    def test_a_weighted_split_with_no_record_available_is_refused_not_spun(self, tmp_path):
        catalogue = weighted(records(), [1e-6, 1.0, 1.0])
        with offline(GridFetch(bad={"L1": list(TIMES)}), catalogue=catalogue):
            batches = obtainer(tmp_path)(ifos=[gf.IFO.L1], group=SPLIT, seed=1, **WINDOW)
            with pytest.raises(ValueError, match=f"split '{SPLIT}'.*no data"):
                within(60, lambda: next(batches))

    def test_a_failed_fetch_is_fetched_again_by_a_later_run(self, tmp_path):
        both = [gf.IFO.H1, gf.IFO.L1]
        with offline(GridFetch(bad={"L1": [TIMES[1]]})):
            _, reader = served(tmp_path, both, num_examples_per_batch=2)
        assert reader.get_acquisition_stats()["rejections"] == {"no data": 1}
        with offline() as fetch:
            again, reader = served(tmp_path, both)
            assert served_times(again) == rounded(TIMES), "a failed fetch persisted as absence"
            assert reader.get_acquisition_stats()["rejections"] == {}
            fetched = len(fetch.calls)
            warm, _ = served(tmp_path, both)
            assert len(fetch.calls) == fetched, "the fetched H1 strain was not stored"
            fresh, _ = served(tmp_path / "fresh", both)
        for batch in (again, warm):
            for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME):
                np.testing.assert_array_equal(by_time(batch, key), by_time(fresh, key), err_msg=key.name)


def cache_rows(directory, prefix="glitch"):
    """The rows a transient cache holds: 0 when it does not exist."""
    path = generate_transient_cache_path(directory, prefix=prefix)
    if not path.exists():
        return 0
    with h5py.File(path, "r") as file:
        return int(file["glitches"]["gps_times"].shape[0])


#: A padded onsource of 31.95 s at RATE: within the stored 32 s, but not with the
#: resampler's margin of 132 stored samples past each end, which leaves 31.935546875 s.
BEYOND = dict(onsource_duration_seconds=30.95, crop_duration_seconds=0.5)
LONGEST_PADDED_ONSOURCE = 31.935546875
#: Requests no read can serve, and how each is refused.
UNSERVABLE = {
    "beyond the stored strain": (BEYOND, rf"cannot be read at 2048 Hz.* at most {LONGEST_PADDED_ONSOURCE} s"),
    "not whole hertz": (dict(sample_rate_hertz=2048.5), "2048.5 Hz is not a positive whole number"),
}
#: Enough records in SPLIT, 2.8 s apart, that their caches are read in chunks.
MANY_TIMES = TIMES[0] + 2.8 * np.arange(100)


class TestRequestsNoReadCanServe:
    """
    A request whose windows no read can serve exactly (they reach past the stored
    strain, the resampler's margin counted, or its rate is not whole hertz) is refused
    by name when it is configured: before any cache is opened, read or reset, and
    before any fetch. A read that reaches past the stored strain anyway is a defect,
    which no handler of a cache's or a miss's errors catches.
    """

    @pytest.mark.parametrize("request_name", UNSERVABLE)
    @pytest.mark.parametrize("times", [MANY_TIMES, TIMES], ids=["chunked", "sparse"])
    def test_is_refused_before_any_cache_or_fetch(self, tmp_path, times, request_name):
        window, refusal = UNSERVABLE[request_name]
        with offline(catalogue=records(times)) as fetch:
            _, reader = served(tmp_path, [gf.IFO.L1])
            rows, fetched = cache_rows(tmp_path), len(fetch.calls)
            with pytest.raises(ValueError) as refused:
                served(tmp_path, [gf.IFO.L1], **window)
        # TransientDataObtainer reads the caches of 100 records or more in chunks.
        assert (len(reader.transient_segments) >= 100) == (len(times) >= 100)
        assert rows > 0 and cache_rows(tmp_path) == rows, f"rows before {rows}, rows after {cache_rows(tmp_path)}"
        assert len(fetch.calls) == fetched, f"fetches {len(fetch.calls) - fetched}"
        assert re.search(refusal, str(refused.value)), str(refused.value)

    def test_a_read_past_the_stored_strain_is_a_defect_no_handler_catches(self, tmp_path):
        with offline(catalogue=records(MANY_TIMES)):
            served(tmp_path, [gf.IFO.L1])
        rows = cache_rows(tmp_path)
        spec = WindowSpec.from_params(sample_rate_hertz=RATE, offsource_duration_seconds=16.0, **BEYOND)
        request = (RATE, spec.total_onsource_duration_seconds, spec.offsource_duration_seconds)
        # A chunk's load, inside the handler that resets an incompatible cache:
        with pytest.raises(glitch_cache.WindowReadError):
            obtainer(tmp_path)._initialize_transient_cache([gf.IFO.L1], *request, prefix="glitch")
        # A row's read from disk, inside the handler that fetches a miss again:
        reader = obtainer(tmp_path)
        cache, _ = reader._initialize_transient_cache([gf.IFO.L1], *request, enable_chunked_mode=False,
                                                      prefix="glitch")
        with pytest.raises(glitch_cache.WindowReadError):
            reader._get_sample_from_cache(cache, float(cache.get_all_gps_times()[0]), *request,
                                          target_ifos=["L1"])
        assert cache_rows(tmp_path) == rows, "a read past the stored strain changed the cache"
        assert not issubclass(glitch_cache.WindowReadError, (ValueError, KeyError))

    @pytest.mark.parametrize("rate", [4096, 2048, 1024, 1000, 512, 333])
    def test_the_check_is_the_reads_geometry(self, rate):
        """
        The longest windows ``check_request`` passes are read within the stored strain;
        two samples more of either window is refused by the check, and by the read as a
        defect.
        """
        stored = int(32 * STORED_RATE)
        before, after = TransientCache.window_reach(stored, stored, STORED_RATE, rate)
        longest_offsource = before - 1 - (before - 1) % 2  # beside a two-sample onsource

        def durations(num_onsource, num_offsource):
            return (num_onsource + 0.5) / rate, (num_offsource + 0.5) / rate

        for longest, beyond in [
            (durations(2 * after, 2), durations(2 * after + 2, 2)),
            (durations(2, longest_offsource), durations(2, longest_offsource + 2)),
        ]:
            TransientCache.check_request(STORED_RATE, rate, *longest)
            read = TransientCache.window_read(stored, stored, STORED_RATE, rate, *longest)
            assert 0 <= read.stored.start and read.stored.stop <= 2 * stored
            with pytest.raises(ValueError, match=f"cannot be read at {rate} Hz"):
                TransientCache.check_request(STORED_RATE, rate, *beyond)
            with pytest.raises(glitch_cache.WindowReadError):
                TransientCache.window_read(stored, stored, STORED_RATE, rate, *beyond)

    def test_the_check_refuses_upsampling_and_rates_that_are_not_whole_hertz(self):
        with pytest.raises(ValueError, match="upsample"):
            TransientCache.check_request(STORED_RATE, 2 * STORED_RATE, 1.0, 1.0)
        with pytest.raises(ValueError, match="not a positive whole number"):
            TransientCache.check_request(STORED_RATE, 2048.5, 1.0, 1.0)


#: One of the 89 pairs of confident (ml_confidence of at least 0.9) Gravity Spy
#: records, O1 to O3b, that share a 10 ms gps_key after TransientIndex.dedupe: an H1
#: and an L1 record of one class, 5.85 ms apart (O3b).
SHARED_KEY = [(1264316116.41895, gf.IFO.H1), (1264316116.42480, gf.IFO.L1)]
SHARED_KEY_REF = 1264316000.0
SHARED_KEY_SPLIT = gf.DEFAULT_SPLIT.assign(SHARED_KEY[0][0] - 48.2, SHARED_KEY[1][0] + 48.2)


class TestRecordIdentity:
    """Each record is served its own window, whatever other record shares its gps_key."""

    def test_records_sharing_a_gps_key_are_served_their_own_windows(self, tmp_path):
        catalogue = [
            TransientSegment(
                gps_key=gps_to_key(t), transient_gps_time=t,
                start_gps_time=t - 48.2, end_gps_time=t + 48.2,
                label=gf.DataLabel.GLITCHES, kind=GlitchType.BLIP,
                observing_run=gf.ObservingRun.O3, seen_in=[ifo],
            )
            for t, ifo in SHARED_KEY
        ]
        assert len({record.gps_key for record in catalogue}) == 1
        assert SHARED_KEY_SPLIT != gf.EXCLUDED
        science = np.array([[SHARED_KEY_REF, SHARED_KEY_REF + 1000.0]])
        both = [gf.IFO.H1, gf.IFO.L1]
        with offline(GridFetch(ref=SHARED_KEY_REF), catalogue=catalogue, science=science) as fetch:
            cold, _ = served(tmp_path, both, SHARED_KEY_SPLIT, num_examples_per_batch=2)
            fetched = len(fetch.calls)
            warm, _ = served(tmp_path, both, SHARED_KEY_SPLIT, num_examples_per_batch=2)
        assert fetched > 0 and len(fetch.calls) == fetched, "the second read was not a cache hit"
        for name, batch in (("cold", cold), ("warm", warm)):
            onsource = np.asarray(batch[RV.ONSOURCE], dtype=np.float64)
            centre = SHARED_KEY_REF + onsource[..., onsource.shape[-1] // 2]
            error = centre - np.asarray(batch[RV.TRANSIENT_GPS_TIME])
            assert np.max(np.abs(error)) <= 1.0 / RATE, f"{name}: centre minus transient time {error} s"
        for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME):
            np.testing.assert_array_equal(by_time(warm, key), by_time(cold, key), err_msg=key.name)


def write_format_2_cache(path, times, zero_ifo="H1", version=2):
    """
    A transient cache as gravyflow wrote it before format 3 (``version`` 2): every
    detector's strain in one row, a detector without strain stored as zeros.
    """
    ifos = list(TransientDefaults.UNIVERSAL_IFOS)
    num_onsource = int(TransientDefaults.CACHE_ONSOURCE_DURATION * STORED_RATE)
    num_offsource = int(TransientDefaults.CACHE_OFFSOURCE_DURATION * STORED_RATE)
    onsource = np.ones((len(times), len(ifos), num_onsource), dtype=np.float32)
    offsource = np.ones((len(times), len(ifos), num_offsource), dtype=np.float32)
    onsource[:, ifos.index(zero_ifo)] = 0.0
    offsource[:, ifos.index(zero_ifo)] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        group = f.create_group("glitches")
        for name, data in (("onsource", onsource), ("offsource", offsource)):
            group.create_dataset(name, data=data, maxshape=(None, *data.shape[1:]),
                                 chunks=(1, *data.shape[1:]))
        group.create_dataset("gps_times", data=np.asarray(times, dtype=np.float64), maxshape=(None,))
        group.create_dataset("labels", data=np.full(len(times), GlitchType.BLIP.value, np.int32),
                             maxshape=(None,))
        group.attrs.update(
            sample_rate_hertz=STORED_RATE,
            onsource_duration=TransientDefaults.CACHE_ONSOURCE_DURATION,
            offsource_duration=TransientDefaults.CACHE_OFFSOURCE_DURATION,
            ifo_names=ifos,
            version=version,
        )


class TestCacheFormat:
    """A cache of an older format is never read: its rows are fetched again."""

    @pytest.mark.parametrize("place", ["its own name", "the current name"])
    def test_an_older_cache_with_a_zero_filled_row_is_never_served(self, tmp_path, place):
        path = (tmp_path / "glitch_cache.h5" if place == "its own name"
                else generate_transient_cache_path(tmp_path, prefix="glitch"))
        write_format_2_cache(path, TIMES)
        with offline(GridFetch(bad={"L1": [TIMES[1]]})) as fetch:
            batch, reader = served(tmp_path, [gf.IFO.H1, gf.IFO.L1], num_examples_per_batch=2)
        assert all(fetches_of(fetch, t) for t in TIMES), "a row of the older cache was served"
        assert served_times(batch) == rounded(TIMES[[0, 2]])
        assert_never_filled(batch)
        assert reader.get_acquisition_stats()["rejections"] == {"no data": 1}

    @pytest.mark.parametrize("version", [2, 3.7], ids=["format 2", "version 3.7"])
    def test_every_read_index_and_write_refuses_another_format(self, tmp_path, version):
        path = tmp_path / "old.h5"
        write_format_2_cache(path, TIMES, version=version)
        before = path.read_bytes()
        cache = TransientCache(path, mode="a")
        window = np.ones((len(TransientDefaults.UNIVERSAL_IFOS), 8), dtype=np.float32)
        refused = {
            "validate_request": lambda: cache.validate_request(RATE, 2.0, 16.0),
            "get_batch": lambda: cache.get_batch(np.array([0]), sample_rate_hertz=RATE,
                                                 onsource_duration=2.0, offsource_duration=16.0),
            "index_of": lambda: cache.index_of(TIMES[0]),
            "has_gps": lambda: cache.has_gps(TIMES[0]),
            "get_all_gps_times": cache.get_all_gps_times,
            "DiskCache.has": lambda: DiskCache(path).has(TIMES[0]),
            "DiskCache.get": lambda: DiskCache(path).get(TIMES[0]),
            "append_single (fill)": lambda: cache.append_single(window, window, TIMES[0], 0),
            "append": lambda: cache.append(window[None], window[None], np.array([REF]), np.array([0])),
        }
        for name, call in refused.items():
            with pytest.raises(ValueError, match=f"format {version}"):
                call()
                pytest.fail(f"{name} did not refuse")
        assert path.read_bytes() == before, "a write reached a file of another format"


def check_glitch_cache():
    """``scripts/check_glitch_cache.py``, loaded as a module."""
    script = Path(gf.__file__).resolve().parents[1] / "scripts" / "check_glitch_cache.py"
    spec = importlib.util.spec_from_file_location("check_glitch_cache", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCacheCheck:
    """The cache check reads format 3: a detector not held (NaN) is no issue."""

    def write(self, path, onsource):
        cache = TransientCache(path, mode="a")
        cache.initialize_file(STORED_RATE, 1.0, 1.0, ["H1", "L1"], 2,
                              onsource.shape[-1], onsource.shape[-1])
        cache.append(onsource, onsource.copy(), TIMES[:len(onsource)], np.zeros(len(onsource)))

    def test_a_detector_not_held_is_healthy(self, tmp_path):
        onsource = np.ones((3, 2, 8), dtype=np.float32)
        onsource[1, 0] = np.nan  # H1 not held for the second record
        self.write(tmp_path / "c.h5", onsource)
        assert check_glitch_cache().check_cache(tmp_path / "c.h5")

    def test_a_partly_non_finite_window_is_an_issue(self, tmp_path):
        onsource = np.ones((3, 2, 8), dtype=np.float32)
        onsource[1, 0, :4] = np.nan
        self.write(tmp_path / "c.h5", onsource)
        assert not check_glitch_cache().check_cache(tmp_path / "c.h5")

    def test_another_format_is_refused(self, tmp_path, capsys):
        write_format_2_cache(tmp_path / "old.h5", TIMES)
        assert not check_glitch_cache().check_cache(tmp_path / "old.h5")
        assert "format 2" in capsys.readouterr().out
