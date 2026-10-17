"""
Transient windows: their geometry, their labels, their data quality, and their read
through the store.

A transient's example reads, on each detector, one stretch of strain through the store
(``store.read``): the offsource, then the padded onsource, the transient on the padded
onsource's centre sample, the sample of the absolute grid at the request's rate nearest
the read's centre. ``START_GPS_TIME`` is the GPS time of the first onsource sample
returned after cropping, as for noise, never the transient's time.

The strain is a fake GWOSC fetch on 64 s tiles: the samples of each tile on the absolute
grid at the native rate, each valued its GPS time minus ``REF``, so a returned sample
decodes to its GPS time.
"""

import dataclasses
import itertools
import os
import re
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import gravyflow as gf
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.store import read as store_read
from gravyflow.src.utils.gps import gps_to_key
from gravyflow.tests.conftest import catalogued_events, gwosc_serving

RV = gf.ReturnVariables
REF = 1240000000.0  # inside O3, on the 64 s tile grid; every sample is valued its GPS time minus REF
RATE = 2048.0
TILE_SECONDS = 64
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
#: Three glitches, off the sample grid at every rate, each in tiles of its own.
TIMES = REF + np.array([10.3701, 150.01234, 290.90007])
#: The one split holding every record's footprint, which every test reads.
SPLIT = gf.DEFAULT_SPLIT.assign(TIMES[0] - 48.2, TIMES[-1] + 48.2)
assert SPLIT != gf.EXCLUDED
#: Times inside one data-quality segment covering every record.
SCIENCE = np.array([[REF - 1000.0, REF + 1000.0]])
#: The decoded samples' precision: float32 values of a few hundred seconds.
DECODE = 0.25 / RATE


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
    A fake GWOSC fetch of whole 64 s tiles: the samples of [start, end) on the absolute
    grid at the requested rate, each valued its GPS time minus ``ref``, plus ``offset``.
    ``bad`` maps a detector name to the times whose tile's download fails (raises) or,
    with ``nan``, holds NaN around them.
    """

    source_id = "gwosc-test"
    tile_seconds = TILE_SECONDS

    def __init__(self, bad=None, nan=False, ref=REF, offset=0.0):
        self.calls = []
        self.bad = bad or {}
        self.nan = nan
        self.ref = ref
        self.offset = offset
        self._lock = threading.Lock()

    def __call__(self, ifo, sample_rate_hertz, interval):
        start, end = interval
        with self._lock:
            self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        times = np.arange(first, stop, dtype=np.float64) / rate
        values = times - self.ref + self.offset
        for t in self.bad.get(ifo.name, ()):
            if start <= t <= end:
                if not self.nan:
                    raise gf.StrainUnavailable(gf.GWOSC(), ifo, interval, f"no {ifo.name} data at {t}")
                values[np.abs(times - t) < 2.0] = np.nan
        return SegmentStrain(values, first / rate, rate, self.source_id)


@contextmanager
def offline(fetch=None, catalogue=None, science=SCIENCE, vetoes=None):
    """
    The glitch catalogue, its data quality (usable segments, for every detector or by
    detector), each detector's glitch vetoes (``vetoes``: by detector, default none)
    and GWOSC strain, offline, with no wait between a failed fetch's retries.
    """
    from gravyflow.src.dataset.acquisition.gwosc_data import subtract_segments

    fetch = fetch or GridFetch()
    vetoes = vetoes or {}

    def readable(self, ifo):
        """The detector's usable segments less the obtainer's exclusions, as the library's."""
        usable = science[ifo] if isinstance(science, dict) else science
        excluded = [interval for name, interval, _ in self.exclusions if name == ifo.name]
        return subtract_segments(usable, np.array(excluded)) if excluded else usable

    with mock.patch("gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
                    return_value=catalogue if catalogue is not None else records()), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", readable), \
         mock.patch.object(gf.TransientDataObtainer, "get_glitch_veto_segments",
                           lambda self, ifo: np.reshape(vetoes.get(ifo, np.empty((0, 2))), (-1, 2))), \
         mock.patch.object(store_read, "FETCH_BACKOFF_SECONDS", 0.0), \
         gwosc_serving(fetch), \
         catalogued_events():
        yield fetch


@contextmanager
def cache_root(path):
    """Another cache root for the block, so its store starts empty: a cold host."""
    with mock.patch.dict(os.environ, {"GRAVYFLOW_CACHE_DIR": str(path)}):
        yield Path(path)


def store_records(detector=None):
    """The tiles the current cache root's store holds: {(detector, index)} of each record."""
    root = Path(os.environ["GRAVYFLOW_CACHE_DIR"]) / "store"
    held = {(path.parts[-4], int(path.stem)) for path in root.rglob("*.json")}
    return {key for key in held if detector is None or key[0] == detector}


def store_files():
    """Every file under the current cache root's store, with its bytes' size and mtime."""
    root = Path(os.environ["GRAVYFLOW_CACHE_DIR"]) / "store"
    return {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def tile_of(gps):
    return int(np.floor(gps / TILE_SECONDS))


def obtainer(**kwargs):
    return gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3,
        augmentations=[],
        **kwargs,
    )


def first_batch(ifos, **window):
    return next(obtainer()(ifos=ifos, group=SPLIT, seed=1, **{**WINDOW, **window}))


def decoded(batch, key=RV.ONSOURCE, ref=REF):
    """Each sample's GPS time, decoded from its value."""
    return ref + np.asarray(batch[key], dtype=np.float64)


def transient_times(batch):
    """Each example's window centres, decoded from its padded onsource."""
    onsource = decoded(batch)
    return onsource[..., onsource.shape[-1] // 2]


def nearest_sample(gps, rate=RATE):
    """The sample of the absolute grid at ``rate`` nearest ``gps``."""
    return np.round(np.asarray(gps, dtype=np.float64) * rate) / rate


#: The glitches are seen in L1, the last detector of each set: H1 is read at
#: partner times.
IFO_SETS = [[gf.IFO.L1], [gf.IFO.H1, gf.IFO.L1]]
SEEN = -1


class TestStartGpsTime:
    """START_GPS_TIME is the first returned onsource sample's time (the owner's ruling)."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_obtainer_labels_the_first_cropped_sample(self, ifos):
        with offline():
            batch = first_batch(ifos)
        first = decoded(batch)[..., SPEC.num_crop_samples]
        label = np.asarray(batch[RV.START_GPS_TIME])
        assert label.dtype == np.float64 and label.shape == first.shape
        error = label - first
        assert np.max(np.abs(error)) <= DECODE, (
            f"START_GPS_TIME minus the first cropped onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )

    def test_dataset_labels_the_first_cropped_sample(self):
        with offline():
            dataset = gf.Dataset(
                noise_obtainer=obtainer(),
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
        assert np.max(np.abs(error)) <= DECODE, (
            f"Dataset START_GPS_TIME minus the first returned onsource sample: "
            f"{error.min():.6f} to {error.max():.6f} s"
        )


class TestGeometry:
    """One geometry for every transient window, read exactly from the store."""

    @pytest.mark.parametrize("ifos", IFO_SETS, ids=lambda ifos: "".join(i.name for i in ifos))
    def test_offsource_ends_where_the_padded_onsource_begins(self, ifos):
        with offline():
            batch = first_batch(ifos)
        onsource, offsource = decoded(batch), decoded(batch, RV.OFFSOURCE)
        assert offsource.shape[-1] == SPEC.num_offsource_samples
        gap = onsource[..., 0] - offsource[..., -1]
        assert np.allclose(gap, 1.0 / RATE, atol=DECODE), f"offsource to onsource: {gap.ravel()} s"
        assert np.allclose(np.diff(offsource, axis=-1), 1.0 / RATE, atol=DECODE), "offsource not contiguous"

    def test_the_transient_is_the_padded_onsource_centre(self):
        with offline():
            batch = first_batch([gf.IFO.L1])
        error = np.sort(transient_times(batch)[:, 0]) - nearest_sample(TIMES)
        assert np.max(np.abs(error)) <= DECODE, f"centre minus the transient's sample: {error} s"

    def test_each_detector_reads_its_own_window_exactly(self):
        """Every detector, partners included, is centred on the sample nearest its own centre."""
        with offline():
            batch = first_batch([gf.IFO.H1, gf.IFO.L1])
        error = transient_times(batch) - nearest_sample(batch[RV.TRANSIENT_GPS_TIME])
        assert np.max(np.abs(error)) <= DECODE, f"centre minus TRANSIENT_GPS_TIME's sample: {error} s"

    @pytest.mark.parametrize("rate", [2048, 1000, 512, 333])
    def test_every_rate_serves_the_absolute_grid(self, rate):
        """At any whole rate, every sample served is a sample of the absolute grid at that rate."""
        with offline():
            batch = first_batch([gf.IFO.H1, gf.IFO.L1], sample_rate_hertz=float(rate))
        for key in (RV.ONSOURCE, RV.OFFSOURCE):
            times = decoded(batch, key)
            assert np.max(np.abs(times * rate - np.round(times * rate))) <= 0.25, f"{key.name} off the grid"
            assert np.allclose(np.diff(times, axis=-1), 1.0 / rate, atol=0.25 / rate), f"{key.name} not at {rate} Hz"
        centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
        assert np.max(np.abs(transient_times(batch) - nearest_sample(centres, rate))) <= 0.25 / rate

    def test_warm_equals_cold_and_every_rate_reads_the_same_tiles(self):
        with offline() as fetch:
            cold = first_batch([gf.IFO.H1, gf.IFO.L1])
            fetched = len(fetch.calls)
            first_batch([gf.IFO.H1, gf.IFO.L1], sample_rate_hertz=1024.0)
            warm = first_batch([gf.IFO.H1, gf.IFO.L1])
        assert fetched > 0 and len(fetch.calls) == fetched, "a warm read, or another rate, fetched"
        for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME, RV.TRANSIENT_GPS_TIME):
            np.testing.assert_array_equal(np.asarray(cold[key]), np.asarray(warm[key]), err_msg=key.name)


def served(ifos, split=SPLIT, **window):
    """The first batch of a fresh obtainer over every record of ``split``, and the obtainer."""
    reader = obtainer(**window.pop("obtainer", {}))
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


def read_spans(batch, before=48.2, after=48.2):
    """Each detector's span of each example: its centre less ``before`` to plus ``after``."""
    centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
    return np.stack([centres - before, centres + after], axis=-1)


class TestDataQuality:
    """Transient windows are checked against data quality, and never zero-filled."""

    def test_windows_outside_data_quality_are_rejected_and_counted(self):
        # A 5 s gap inside the second record's read (its offsource), on L1, where it was seen.
        gap = TIMES[1] - 8.0
        science = {
            gf.IFO.H1: SCIENCE,
            gf.IFO.L1: np.array([[REF - 1000.0, gap], [gap + 5.0, REF + 1000.0]]),
        }
        for ifos in IFO_SETS:
            name = "".join(ifo.name for ifo in ifos)
            with offline(science=science) as fetch:
                batch, reader = served(ifos, num_examples_per_batch=2)
            assert served_times(batch) == rounded(TIMES[[0, 2]]), name
            assert reader.get_acquisition_stats()["rejections"] == {"data quality": 1}, name
            assert not any(ifo == "L1" and start <= TIMES[1] < end for ifo, start, end in fetch.calls), (
                "a rejected record was fetched"
            )

    def test_a_partner_is_read_in_its_own_data_quality_and_vetoes(self):
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
        with offline(science=science, vetoes=vetoes):
            batch, reader = served([gf.IFO.H1, gf.IFO.L1])
        assert served_times(batch) == rounded(TIMES)
        assert reader.get_acquisition_stats()["rejections"] == {}
        # Each H1 read, from its first offsource sample to its last onsource sample.
        for start, end in zip(decoded(batch, RV.OFFSOURCE)[:, 0, 0], decoded(batch)[:, 0, -1]):
            assert any(a <= start and end <= b for a, b in science[gf.IFO.H1]), f"H1 read {start}-{end} leaves its data"
            assert not any(start < b and a < end for a, b in vetoes[gf.IFO.H1]), f"H1 read {start}-{end} meets a veto"
        centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
        assert np.all(np.abs(centres[:, 0] - centres[:, SEEN]) >= 4 * 48.2), "H1 read near its own glitch"
        assert np.all(np.asarray(batch[RV.DATA_LABEL]) == [[gf.DataLabel.NOISE.value, gf.DataLabel.GLITCHES.value]])
        assert np.all(np.asarray(batch[RV.GLITCH_TYPE])[:, 0] == -1)
        assert np.all(np.asarray(batch[RV.GLITCH_TYPE])[:, SEEN] == GlitchType.BLIP.value)

    def test_a_detector_a_glitch_was_not_seen_in_is_never_read_at_its_time(self):
        with offline() as fetch:
            served([gf.IFO.L1])
            assert {ifo for ifo, _, _ in fetch.calls} == {"L1"}, "an unrequested detector was fetched"
            batch, _ = served([gf.IFO.H1, gf.IFO.L1])
        h1 = read_spans(batch)[:, 0]
        assert not any(start < t < end for start, end in h1 for t in TIMES), (
            "H1 was read at the time of a glitch seen only in L1"
        )


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


def weighted(catalogue, weights):
    """``catalogue`` with each record's sampling weight, as class balancing sets it."""
    return [dataclasses.replace(record, weight=w) for record, w in zip(catalogue, weights)]


def refusal(call):
    """The transient refusal ``call`` raises."""
    from gravyflow.src.dataset.acquisition.transient import TransientUnavailable

    with pytest.raises(TransientUnavailable) as refused:
        call()
    return refused.value


class TestRefusals:
    """
    A record the timeline holds, whose strain the source does not deliver (its download
    fails, or it is not finite), is refused by name: never skipped, zero-filled or
    stored as absent. The refusal names the record, the detector and the
    ``exclusions`` entry that plans around it.
    """

    def test_a_failed_download_is_refused_by_name_and_stores_nothing(self):
        with offline(GridFetch(bad={"L1": [TIMES[1]]})) as fetch:
            refused = refusal(lambda: served([gf.IFO.H1, gf.IFO.L1]))
            assert isinstance(refused, gf.StrainUnavailable)
            assert refused.record.transient_gps_time == TIMES[1] and refused.detector == "L1"
            message = str(refused)
            assert f"{TIMES[1]:.4f}" in message and "L1" in message and "exclusions=" in message
            failed = tile_of(TIMES[1])
            assert refused.exclusion[:2] == ("L1", (failed * TILE_SECONDS, (failed + 1) * TILE_SECONDS))
            assert sum(ifo == "L1" and start <= TIMES[1] < end for ifo, start, end in fetch.calls) == 3
            # A failed fetch writes nothing, so a timeout never persists as absence.
            assert ("L1", failed) not in store_records()
        with offline() as fetch:
            batch, reader = served([gf.IFO.H1, gf.IFO.L1])
            assert served_times(batch) == rounded(TIMES), "a failed download persisted as absence"
            assert reader.get_acquisition_stats()["rejections"] == {}
            assert any(ifo == "L1" and start <= TIMES[1] < end for ifo, start, end in fetch.calls)
        assert_never_filled(batch)

    def test_non_finite_strain_is_refused_by_name_and_its_exclusion_plans_around_it(self):
        with offline(GridFetch(bad={"L1": [TIMES[1]]}, nan=True)):
            refused = refusal(lambda: served([gf.IFO.H1, gf.IFO.L1]))
            assert "NON_FINITE" in str(refused) and refused.record.transient_gps_time == TIMES[1]
            detector, (start, end), _ = refused.exclusion
            assert detector == "L1" and start < TIMES[1] < end
            batch, reader = served([gf.IFO.H1, gf.IFO.L1], num_examples_per_batch=2,
                                   obtainer={"exclusions": [refused.exclusion]})
        assert served_times(batch) == rounded(TIMES[[0, 2]])
        assert_never_filled(batch)
        # Excluded time is not readable: the record is planned out and counted.
        assert reader.get_acquisition_stats()["rejections"] == {"data quality": 1}

    @pytest.mark.parametrize("weights", [None, [1e-6, 1.0, 1.0]], ids=["equal weights", "weighted"])
    def test_an_epoch_that_cannot_be_served_is_refused_never_spun(self, weights):
        catalogue = records() if weights is None else weighted(records(), weights)
        with offline(GridFetch(bad={"L1": list(TIMES)}), catalogue=catalogue):
            batches = obtainer()(ifos=[gf.IFO.L1], group=SPLIT, seed=1, **WINDOW)
            refused = refusal(lambda: within(60, lambda: next(batches)))
        assert refused.record.transient_gps_time in TIMES


#: A padded onsource whose read reaches past the 48.2 s a record may read before its
#: transient, with the offsource: 16 s + 32.5 s / 2 plus the resampler's margin.
BEYOND = dict(onsource_duration_seconds=31.5, crop_duration_seconds=0.5, offsource_duration_seconds=32.0)
#: Requests no read can serve, and how each is refused.
UNSERVABLE = {
    "beyond the record's span": (BEYOND, r"cannot be read.*48\.2 s"),
    "not whole hertz": (dict(sample_rate_hertz=2048.5), "2048.5 Hz is not a positive whole number"),
}


class TestRequestsNoReadCanServe:
    """
    A request whose windows no read can serve inside a record's span (they reach past
    it, the resampler's margin counted), or whose rate is not whole hertz, is refused by
    name when it is configured, before any fetch; and no request, served or refused,
    ever removes or rewrites a tile another request reads.
    """

    @pytest.mark.parametrize("request_name", UNSERVABLE)
    def test_is_refused_before_any_fetch_and_changes_no_tile(self, request_name):
        window, pattern = UNSERVABLE[request_name]
        with offline() as fetch:
            served([gf.IFO.L1])
            before, fetched = store_files(), len(fetch.calls)
            with pytest.raises(ValueError) as refused:
                served([gf.IFO.L1], **window)
        assert before and store_files() == before
        assert len(fetch.calls) == fetched, f"fetches {len(fetch.calls) - fetched}"
        assert re.search(pattern, str(refused.value)), str(refused.value)

    def test_the_longest_request_inside_the_span_is_served(self):
        longest = dict(onsource_duration_seconds=30.0, crop_duration_seconds=0.5, offsource_duration_seconds=32.0)
        with offline():
            batch, _ = served([gf.IFO.H1, gf.IFO.L1], **longest)
        assert_never_filled(batch)
        assert np.asarray(batch[RV.OFFSOURCE]).shape[-1] == 32 * RATE

    def test_another_request_never_removes_or_rewrites_a_tile(self):
        with offline():
            served([gf.IFO.H1, gf.IFO.L1])
            before = store_files()
            for window in (dict(sample_rate_hertz=1024.0), dict(sample_rate_hertz=4096.0),
                           dict(offsource_duration_seconds=8.0)):
                served([gf.IFO.H1, gf.IFO.L1], **window)
            with pytest.raises(ValueError):
                served([gf.IFO.H1, gf.IFO.L1], **BEYOND)
        after = store_files()
        assert all(after.get(path) == stamp for path, stamp in before.items()), "a tile was removed or rewritten"


#: One of the 89 pairs of confident (ml_confidence of at least 0.9) Gravity Spy
#: records, O1 to O3b, that share a 10 ms gps_key after TransientIndex.dedupe: an H1
#: and an L1 record of one class, 5.85 ms apart (O3b).
SHARED_KEY = [(1264316116.41895, gf.IFO.H1), (1264316116.42480, gf.IFO.L1)]
SHARED_KEY_REF = 1264316000.0
SHARED_KEY_SPLIT = gf.DEFAULT_SPLIT.assign(SHARED_KEY[0][0] - 48.2, SHARED_KEY[1][0] + 48.2)


def by_time(batch, key):
    """``batch[key]`` in order of the examples' transient times."""
    order = np.argsort(np.asarray(batch[RV.TRANSIENT_GPS_TIME])[:, 0])
    return np.asarray(batch[key])[order]


class TestRecordIdentity:
    """Each record is served its own window, whatever other record shares its gps_key."""

    def test_records_sharing_a_gps_key_are_served_their_own_windows(self):
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
            cold, _ = served(both, SHARED_KEY_SPLIT, num_examples_per_batch=2)
            fetched = len(fetch.calls)
            warm, _ = served(both, SHARED_KEY_SPLIT, num_examples_per_batch=2)
        assert fetched > 0 and len(fetch.calls) == fetched, "the second read fetched"
        for name, batch in (("cold", cold), ("warm", warm)):
            error = transient_times_from(batch, SHARED_KEY_REF) - nearest_sample(batch[RV.TRANSIENT_GPS_TIME])
            assert np.max(np.abs(error)) <= DECODE, f"{name}: centre minus transient time {error} s"
        for key in (RV.ONSOURCE, RV.OFFSOURCE, RV.START_GPS_TIME):
            np.testing.assert_array_equal(by_time(warm, key), by_time(cold, key), err_msg=key.name)


def transient_times_from(batch, ref):
    onsource = decoded(batch, ref=ref)
    return onsource[..., onsource.shape[-1] // 2]


class TestOneReadPath:
    """
    Every transient read is a store read: no transient cache, no download cache and no
    strain file read beside it, and an old transient cache is never read, only named.
    """

    SOURCE = Path(gf.__file__).resolve().parent / "src"
    #: Names of the read paths the store replaced; none may remain in the library.
    BYPASSES = ("TransientCache", "DiskCache", "glitch_cache", "set_temp_cache", "download_file",
                "strain_download_cache", "_initialize_transient_cache", "cache=True")
    #: The library's modules that may open HDF5 files, and why: none reads strain.
    HDF5_READERS = {
        "validate/validator.py": "validation results",
        "utils/io.py": "training histories and saved results",
    }

    def sources(self):
        return {path.relative_to(self.SOURCE).as_posix(): path.read_text()
                for path in self.SOURCE.rglob("*.py")}

    def test_no_transient_read_bypasses_the_store(self):
        found = {(name, bypass) for name, text in self.sources().items()
                 for bypass in self.BYPASSES if bypass in text}
        assert not found, f"read paths beside the store remain: {sorted(found)}"

    def test_only_the_store_fetches_strain(self):
        fetches = {name for name, text in self.sources().items()
                   if re.search(r"\bsource\.fetch\(", text)}
        assert fetches == {"dataset/store/read.py"}, fetches
        opens = {name for name, text in self.sources().items() if re.search(r"\bh5py\.File\(", text)}
        assert opens <= set(self.HDF5_READERS), f"modules opening HDF5 files: {sorted(opens - set(self.HDF5_READERS))}"

    def test_an_old_transient_cache_is_never_read_and_is_named(self):
        root = Path(os.environ["GRAVYFLOW_CACHE_DIR"])
        old = root / "glitch_cache_v4_0123456789abcdef.h5"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_bytes(b"not an HDF5 file" * 64)
        before = old.read_bytes()
        with mock.patch.object(store_read, "_legacy_warned", False), offline():
            with pytest.warns(UserWarning, match=r"never read.*glitch_cache_v4_0123456789abcdef\.h5 \(1\.0 KiB\)"):
                batch, _ = served([gf.IFO.L1])
        assert old.read_bytes() == before
        assert served_times(batch) == rounded(TIMES)


# ----------------------------------------------------------------- passes

#: Six glitches of the training split, each in tiles of its own.
PASS_TIMES = REF + 10.3701 + 140.0 * np.arange(6)
#: Five glitches of the test split.
TEST_TIMES = REF + 15000.3701 + 140.0 * np.arange(5)
#: Data quality over both splits' records.
WIDE_SCIENCE = np.array([[REF - 1000.0, REF + 24000.0]])
#: The window, as a batcher takes it.
PASS_WINDOW = {key: value for key, value in WINDOW.items() if key != "scale_factor"}


def times_of(batch):
    """The transient times of a batch's examples, in its order."""
    return [float(t) for t in np.round(np.asarray(batch[RV.TRANSIENT_GPS_TIME])[:, SEEN], 4)]


def direct(count, group=SPLIT, seed=1, reader=None, **window):
    """The first ``count`` batches (fewer, if it ends) of a fresh obtainer's own generator."""
    reader = reader or obtainer()
    return list(itertools.islice(reader(ifos=[gf.IFO.L1], group=group, seed=seed, **{**WINDOW, **window}), count))


def batcher(reader, group=SPLIT, seed=7, steps_per_epoch=2, augmented=False, **window):
    """A dataset's batcher of ``reader`` for split ``group``: a function of a batch's address."""
    from gravyflow.src.dataset.dataset import augmented_batcher
    from gravyflow.src.dataset.noise.noise import obtainer_batcher

    make = augmented_batcher if augmented else obtainer_batcher
    tree = gf.src.rng.SeedTree(seed)
    batch = make(reader, tree, group, ifos=[gf.IFO.L1], steps_per_epoch=steps_per_epoch,
                 **{**PASS_WINDOW, **window})
    return lambda epoch, k: batch(tree.batch(group, epoch, k))


def same_batches(a, b):
    assert len(a) == len(b)
    for one, other in zip(a, b):
        for key in (RV.ONSOURCE, RV.TRANSIENT_GPS_TIME, RV.START_GPS_TIME):
            assert np.array_equal(np.asarray(one[key]), np.asarray(other[key])), key.name


class TestPasses:
    """
    Training concatenates passes over the split's records: pass p is ordered, and
    balanced, by draws of its own, so every pass differs and the same seed gives the
    same passes. A batch is the next B items of the concatenation, so none is partial.
    An evaluation split is one fixed pass, floor(N / B) batches, its remainder never
    served. A batch is a function of its address alone.
    """

    def test_each_pass_has_its_own_order_and_a_seed_gives_one_sequence(self):
        with offline(catalogue=records(PASS_TIMES), science=WIDE_SCIENCE):
            first = [times_of(batch) for batch in direct(8)]
            again = [times_of(batch) for batch in direct(8)]
        assert first == again
        passes = [first[2 * p] + first[2 * p + 1] for p in range(4)]
        for one in passes:
            assert sorted(one) == rounded(PASS_TIMES), "a pass is not every record once"
        assert len({tuple(one) for one in passes}) > 1, "every pass repeats one order"

    def test_balance_is_exact_in_every_pass(self):
        catalogue = records(PASS_TIMES[:4]) + [dataclasses.replace(
            records(PASS_TIMES[4:5])[0], kind=GlitchType.KOI_FISH)]
        with offline(catalogue=catalogue, science=WIDE_SCIENCE):
            batches = direct(6, num_examples_per_batch=4, reader=obtainer(balanced_glitch_types=True))
        kinds = [int(k) for batch in batches for k in np.asarray(batch[RV.GLITCH_TYPE])[:, SEEN]]
        times = sum((times_of(batch) for batch in batches), [])
        assert all(len(times_of(batch)) == 4 for batch in batches)
        rare = round(float(PASS_TIMES[4]), 4)
        for p in range(3):
            assert sorted(kinds[8 * p:8 * p + 8]) == [GlitchType.BLIP.value] * 4 + [GlitchType.KOI_FISH.value] * 4
            assert sorted(t for t in times[8 * p:8 * p + 8] if t != rare) == rounded(PASS_TIMES[:4])

    def test_no_batch_is_partial(self):
        with offline():
            batches = direct(3, num_examples_per_batch=2)
        assert [len(times_of(batch)) for batch in batches] == [2, 2, 2]
        served = sum((times_of(batch) for batch in batches), [])
        assert sorted(served[:3]) == sorted(served[3:]) == rounded(TIMES)

    def test_a_batch_is_a_function_of_its_address(self):
        from concurrent.futures import ThreadPoolExecutor

        addresses = [(0, 0), (0, 1), (1, 0), (1, 1), (3, 1)]
        with offline(catalogue=records(PASS_TIMES), science=WIDE_SCIENCE):
            forward = [batcher(obtainer())(*a) for a in addresses]
            read = batcher(obtainer())
            backward = [read(*a) for a in reversed(addresses)][::-1]
            with ThreadPoolExecutor(4) as pool:
                read = batcher(obtainer())
                threaded = list(pool.map(lambda a: read(*a), addresses))
        same_batches(forward, backward)
        same_batches(forward, threaded)
        # Epoch e of two batches of three is pass e: every record once.
        for e in (0, 1):
            assert sorted(times_of(forward[2 * e]) + times_of(forward[2 * e + 1])) == rounded(PASS_TIMES)

    def test_an_evaluation_split_is_one_fixed_set(self):
        with offline(catalogue=records(TEST_TIMES), science=WIDE_SCIENCE):
            batches = direct(10, group="test", num_examples_per_batch=2)
            again = direct(10, group="test", num_examples_per_batch=2)
            read = batcher(obtainer(), group="test", seed=1, num_examples_per_batch=2)
            by_epoch = [[read(epoch, k) for k in range(2)] for epoch in (0, 3)]
            with pytest.raises(IndexError, match="1 record"):
                read(0, 2)
        assert len(batches) == 2, "an evaluation split is floor(N / B) batches"
        served = sum((times_of(batch) for batch in batches), [])
        assert len(set(served)) == 4 and set(served) <= set(rounded(TEST_TIMES))
        same_batches(batches, again)
        same_batches(by_epoch[0], by_epoch[1])
        same_batches(batches, by_epoch[0])


class TestReadSpan:
    """
    A record is planned where its read, not its whole footprint, is readable; a read
    may reach anywhere in the record's footprint, which for an event is longer before
    its time; RandomShift moves the read, never rolls the strain.
    """

    def test_data_quality_reads_the_windows_span(self):
        science = np.array([[REF - 1000.0, TIMES[2] + 20.0]])
        with offline(science=science):
            batch, reader = served([gf.IFO.L1])
        assert served_times(batch) == rounded(TIMES)
        assert reader.get_acquisition_stats()["rejections"] == {}

    def test_an_events_read_reaches_into_its_footprint(self):
        t = REF + 16000.3701
        event = dataclasses.replace(
            records([t])[0], label=gf.DataLabel.EVENTS, kind=gf.SourceType.BNS, seen_in=[],
            mass1=1.4, mass2=1.4, name="GW-OFFLINE-BNS")
        with offline(catalogue=[event], science=WIDE_SCIENCE):
            batch, _ = served([gf.IFO.L1], split="test", offsource_duration_seconds=96.0, num_examples_per_batch=1)
            assert served_times(batch) == rounded([t])
            assert decoded(batch, RV.OFFSOURCE)[0, 0, 0] < t - 96.0
            with pytest.raises(ValueError, match="cannot be read"):
                served([gf.IFO.L1], split="test", offsource_duration_seconds=200.0, num_examples_per_batch=1)

    def test_random_shift_moves_the_read_and_never_rolls(self):
        reader = gf.TransientDataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
            observing_runs=gf.ObservingRun.O3,
            augmentations=[gf.RandomShift(probability=1.0, shift_fraction=0.25)])
        with offline(catalogue=records(PASS_TIMES), science=WIDE_SCIENCE):
            read = batcher(reader, augmented=True)
            batches = [read(0, k) for k in range(2)]
        shifts = []
        for batch in batches:
            onsource = decoded(batch)[:, 0]
            steps = np.diff(onsource, axis=-1)
            assert np.all(np.abs(steps - 1.0 / RATE) < DECODE), "the strain was rolled"
            shift = onsource[:, onsource.shape[-1] // 2] - np.asarray(batch[RV.TRANSIENT_GPS_TIME])[:, 0]
            assert np.all(np.abs(shift) <= 0.25 * 2.0 + 1.0 / RATE)
            assert np.all(np.asarray(batch[RV.AUGMENTATIONS]) == gf.RandomShift.bit)
            start = np.asarray(batch[RV.START_GPS_TIME])[:, 0]
            assert np.all(np.abs(start - onsource[:, int(0.5 * RATE)]) < DECODE)
            shifts += list(shift)
        assert np.any(np.abs(shifts) > 2.0 / RATE), "no example was shifted"
