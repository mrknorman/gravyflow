"""
Tests for the public real-noise sources: GWOSC run bounds, GWOSC timeline segments,
O1/O2 frame types and channels, the requested sample rate, the Gravity Spy catalogue
and public-data defaults.

The default run is offline. Network access is replaced by small committed fixtures in
``fixtures/public_sources``:

- ``gwosc_timeline.json``: GWOSC timeline segments over one day in each of O1, O2, O3a
  and O3b, from https://gwosc.org: DATA, BURST_CAT1 and BURST_CAT2, and the
  hardware-injection flags NO_CBC_HW_INJ, NO_BURST_HW_INJ, NO_STOCH_HW_INJ and
  NO_DETCHAR_HW_INJ, for H1 and L1.
- ``gravityspy_{IFO}_{RUN}.csv``: a few hundred rows of the Gravity Spy machine-learning
  classifications (Glanzer et al. 2023, Class. Quantum Grav. 40 065004; Zenodo,
  doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0), restricted to the columns gravyflow
  reads.
- ``gwosc_events.csv``: every row of every catalogue GWOSC's event API listed on
  2026-10-16 (https://gwosc.org/eventapi/json/), as gravyflow reads them: 1,062 rows
  of 18 catalogues, with their GPS times and source-frame chirp masses.

Tests marked ``network`` query GWOSC and Zenodo for real; they run only with
``--network``.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer, ObservingRunData
from gravyflow.tests.conftest import refuse_internet

FIXTURES = Path(__file__).parent / "fixtures" / "public_sources"
TIMELINE = json.loads((FIXTURES / "gwosc_timeline.json").read_text())

# GWOSC run API values (https://gwosc.org/api/v2/runs).
GWOSC_RUN_BOUNDS = {
    "O1": (1126051217, 1137254417),
    "O2": (1164556817, 1187733618),
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
}
# Public events inside the fixture windows, used as the event-veto list, with their
# source-frame chirp masses from GWOSC's GWTC table: GW150914, GW170817, GW190521,
# GW200129_065458.
FIXTURE_EVENT_TIMES = np.array([1126259462.4, 1187008882.4, 1242442967.4, 1264316116.4])
FIXTURE_EVENTS = {
    "gps": FIXTURE_EVENT_TIMES,
    "chirp_mass_source": np.array([27.9, 1.186, 63.3, 27.2]),
    "name": np.array(["GW150914", "GW170817", "GW190521", "GW200129_065458"]),
    "catalog": np.array(["GWTC-2.1-confident", "GWTC-1-confident",
                         "GWTC-2.1-confident", "GWTC-3-confident"]),
}


# =============================================================================
# OFFLINE HARNESS
# =============================================================================

class _FakeResponse:
    def __init__(self, url, body: bytes, status: int = 200):
        self.url = url
        self.status_code = status
        self.content = body
        self.headers = {"Content-Length": str(len(body))}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} for {self.url}")

    def json(self):
        return json.loads(self.content)

    def iter_content(self, chunk_size=1024 * 1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        pass


def _fixture_md5s():
    files = {}
    for path in sorted(FIXTURES.glob("gravityspy_*.csv")):
        _, ifo, run = path.stem.split("_")
        body = path.read_bytes()
        files[(ifo, run)] = (f"{ifo}_{run}.csv", hashlib.md5(body).hexdigest(), len(body))
    return files


class FakeNetwork:
    """Serves the fixtures for GWOSC timeline and Zenodo URLs; refuses anything else."""

    def __init__(self):
        self.requests = []
        self.corrupt_zenodo = False

    def get(self, url, *args, **kwargs):
        self.requests.append(url)
        if "/timeline/segments/json/" in url:
            # .../timeline/segments/json/{dataset}/{ifo}_{flag}/{start}/{duration}/
            parts = url.split("/timeline/segments/json/")[1].strip("/").split("/")
            dataset, flag_name = parts[0], parts[1]
            entry = TIMELINE.get(dataset)
            if entry is None or flag_name not in entry:
                return _FakeResponse(url, b"not found", 404)
            body = json.dumps({"dataset": dataset, "id": flag_name,
                               "segments": entry[flag_name]}).encode()
            return _FakeResponse(url, body)
        if "zenodo.org/api/records/5649212/files/" in url:
            key = url.split("/files/")[1].split("/")[0]
            body = (FIXTURES / f"gravityspy_{key}").read_bytes()
            if self.corrupt_zenodo:
                body = body.replace(b"Blip", b"Bl1p", 1)
            return _FakeResponse(url, body)
        raise ConnectionError(f"Offline test: refused request to {url}")


@pytest.fixture
def offline_public_sources(monkeypatch, tmp_path):
    """Offline GWOSC and Zenodo, with every user cache under a temporary HOME."""
    import requests

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path / ".gravyflow" / "cache"))
    refuse_internet(monkeypatch)  # even in runs with --network

    network = FakeNetwork()
    monkeypatch.setattr(requests, "get", network.get)

    # The public catalogue verifies the release's md5; the fixtures have their own.
    try:
        from gravyflow.src.dataset.features import gravity_spy
    except ImportError:
        gravity_spy = None
    if gravity_spy is not None:
        monkeypatch.setattr(gravity_spy, "RELEASE_FILES", _fixture_md5s())
        gravity_spy.clear_memory_cache()

    from gravyflow.src.dataset.features import event as event_module
    monkeypatch.setattr(
        event_module, "get_event_catalogue",
        lambda use_cache=True: {key: value.copy() for key, value in FIXTURE_EVENTS.items()},
        raising=False,
    )
    # Glitch queries retry with sleeps in older code; never sleep in tests.
    import time
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    yield network

    if gravity_spy is not None:
        gravity_spy.clear_memory_cache()


class FakeEventApi:
    """GWOSC's event API: a catalogue list and one table per catalogue."""

    def __init__(self, catalogues):
        self.catalogues = catalogues
        self.fetched = []

    def cataloglist(self, *args, **kwargs):
        return {name: {"url": f"https://gwosc.org/eventapi/json/{name}"}
                for name in self.catalogues}

    def fetch_open_data(self, catalog, *args, **kwargs):
        self.fetched.append(catalog)
        return self.catalogues[catalog]

    def serve(self, monkeypatch):
        """Answer gravyflow's event-API calls."""
        from gwosc import api as gwosc_api
        from gravyflow.src.dataset.features import event as event_module

        monkeypatch.setattr(gwosc_api, "fetch_cataloglist_json", self.cataloglist)
        monkeypatch.setattr(event_module.EventTable, "fetch_open_data", self.fetch_open_data)
        return self


def event_snapshot_tables():
    """The ``gwosc_events.csv`` snapshot, one astropy table per catalogue."""
    from astropy.table import Table

    rows = {}
    with open(FIXTURES / "gwosc_events.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            mass = float(row["chirp_mass_source"]) if row["chirp_mass_source"] else np.nan
            rows.setdefault(row["catalog"], []).append((float(row["gps"]), row["name"], mass))
    return {catalog: Table(rows=entries, names=["GPS", "commonName", "chirp_mass_source"])
            for catalog, entries in rows.items()}


@pytest.fixture
def offline_event_api(monkeypatch, tmp_path):
    """GWOSC's event API served from the snapshot, with an empty event cache."""
    monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path))
    return FakeEventApi(event_snapshot_tables()).serve(monkeypatch)


def _fixture_intersection(dataset, ifo, flags):
    """Expected segments: the fixture flags intersected, by an independent method."""
    entry = TIMELINE[dataset]
    start, end = entry["window"]
    mask = np.ones(end - start, dtype=bool)
    for flag in flags:
        passing = np.zeros(end - start, dtype=bool)
        for a, b in entry[f"{ifo}_{flag}"]:
            passing[max(a, start) - start:min(b, end) - start] = True
        mask &= passing
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return np.stack([edges[0::2] + start, edges[1::2] + start], axis=1).astype(float)


# =============================================================================
# RUN BOUNDS (from the GWOSC run API)
# =============================================================================

class TestRunBounds:

    @pytest.mark.parametrize("run, gwosc_runs", [
        (gf.ObservingRun.O1, ["O1"]),
        (gf.ObservingRun.O2, ["O2"]),
        (gf.ObservingRun.O3, ["O3a", "O3b"]),
    ])
    def test_bounds_match_gwosc_run_api(self, run, gwosc_runs):
        assert run.value.start_gps_time == GWOSC_RUN_BOUNDS[gwosc_runs[0]][0]
        assert run.value.end_gps_time == GWOSC_RUN_BOUNDS[gwosc_runs[-1]][1]

    def test_datetime_runs_count_leap_seconds(self):
        # 2017-01-01 00:00:00 UTC is GPS 1167264018 (18 leap seconds since 1980).
        run = ObservingRunData(
            "custom", datetime(2017, 1, 1), datetime(2017, 1, 2), {}, {}, {}
        )
        assert run.start_gps_time == 1167264018
        assert run.end_gps_time == 1167264018 + 86400
        # 2015-09-12 00:00:00 UTC is GPS 1126051217 (17 leap seconds then).
        run = ObservingRunData(
            "custom", datetime(2015, 9, 12), datetime(2015, 9, 13), {}, {}, {}
        )
        assert run.start_gps_time == 1126051217


# =============================================================================
# SEGMENTS FROM THE GWOSC TIMELINE
# =============================================================================

class TestGwoscSegments:

    def test_segments_come_from_gwosc_not_dqsegdb(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O3,
            cache_segments=False,
        )
        with patch(
            "gwpy.segments.DataQualityDict.query_dqsegdb",
            side_effect=AssertionError("DQSegDB must not be queried"),
        ):
            segments = obtainer.get_all_segment_times(gf.IFO.L1)

        expected = np.concatenate([
            _fixture_intersection("O3a_4KHZ_R1", "L1", ["DATA", "BURST_CAT1", "BURST_CAT2"]),
            _fixture_intersection("O3b_4KHZ_R1", "L1", ["DATA", "BURST_CAT1", "BURST_CAT2"]),
        ])
        np.testing.assert_array_equal(segments, expected)
        assert any("/timeline/segments/json/O3a_4KHZ_R1/L1_BURST_CAT2/" in url
                   for url in offline_public_sources.requests)

    def test_flag_option_selects_data_only(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O1,
            cache_segments=False,
            gwosc_flags=gf.GWOSC_DATA_FLAGS,
        )
        segments = obtainer.get_all_segment_times(gf.IFO.H1)
        np.testing.assert_array_equal(
            segments, _fixture_intersection("O1", "H1", ["DATA"])
        )
        # The default (with the burst vetoes) is a strict subset here.
        default = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O1,
            cache_segments=False,
        ).get_all_segment_times(gf.IFO.H1)
        livetime = lambda s: float(np.sum(s[:, 1] - s[:, 0]))
        assert livetime(default) < livetime(segments)

    def test_dqsegdb_flag_names_are_rejected(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O3,
            cache_segments=False,
        )
        with pytest.raises(ValueError, match="GWOSC"):
            obtainer.get_segment_times(
                1238166018, 1238252418, gf.IFO.L1, "DCS-ANALYSIS_READY_C01:1"
            )

    def test_timeline_is_cached_in_user_cache(self, offline_public_sources, tmp_path):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O1,
            cache_segments=False,
        )
        first = obtainer.get_all_segment_times(gf.IFO.L1)
        n_requests = len(offline_public_sources.requests)
        second = obtainer.get_all_segment_times(gf.IFO.L1)
        np.testing.assert_array_equal(first, second)
        assert len(offline_public_sources.requests) == n_requests
        cached = list((tmp_path / ".gravyflow" / "cache" / "gwosc_timeline").glob("*.json"))
        assert len(cached) == len(gf.DEFAULT_GWOSC_FLAGS)  # one file per flag, L1 O1
        assert not list((gf.PATH / "res").glob("**/*gwosc*"))


# =============================================================================
# THE DEFAULT [O1, O2, O3] CONFIGURATION
# =============================================================================

class TestDefaultConfiguration:

    def test_ragged_runs_concatenate(self):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            cache_segments=False,
        )
        per_run = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0]]),
            np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ]
        with patch.object(obtainer, "get_segment_times", side_effect=per_run):
            segments = obtainer.get_all_segment_times(gf.IFO.H1)
        np.testing.assert_array_equal(segments, np.concatenate(per_run))

    def test_default_runs_build_offline(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            cache_segments=False,
        )
        segments = obtainer.get_valid_segments(
            ifos=[gf.IFO.H1, gf.IFO.L1], seed=1, group_name="train"
        )
        assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
        assert len(segments) > 0
        runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]
        starts = segments[:, :, 0].ravel()
        for run in runs:
            assert np.any((starts >= run.value.start_gps_time) & (starts < run.value.end_gps_time))
        # Every segment lies inside the default flags for its detector.
        for i, ifo in enumerate(["H1", "L1"]):
            allowed = np.concatenate([
                _fixture_intersection(d, ifo, ["DATA", "BURST_CAT1", "BURST_CAT2"])
                for d in TIMELINE
            ])
            for start, end in segments[:, i, :]:
                assert np.any((allowed[:, 0] <= start) & (allowed[:, 1] >= end))


# =============================================================================
# FRAME TYPES AND CHANNELS (O1/O2)
# =============================================================================

class TestFrameTypesAndChannels:

    # Verified against GWOSC's datafind server and the frame files' channel lists.
    @pytest.mark.parametrize("run, frame_type, channel", [
        (gf.ObservingRun.O1, "LOSC_4_V1", "LOSC-STRAIN"),
        (gf.ObservingRun.O2, "GWOSC_O2_4KHZ_R1", "GWOSC-4KHZ_R1_STRAIN"),
    ])
    def test_run_tables(self, run, frame_type, channel):
        assert run.value.frame_types[gf.DataQuality.BEST] == frame_type
        assert run.value.channels[gf.DataQuality.BEST] == channel

    @pytest.mark.parametrize("gps, frame_type, channel", [
        (1126259446, "H1_LOSC_4_V1", "H1:LOSC-STRAIN"),
        (1187008866, "H1_GWOSC_O2_4KHZ_R1", "H1:GWOSC-4KHZ_R1_STRAIN"),
    ])
    def test_local_frame_lookup_uses_matching_pair(self, gps, frame_type, channel):
        calls = {}

        def fake_find_urls(**kwargs):
            calls["frametype"] = kwargs["frametype"]
            return ["file:///frames/fake.gwf"]

        def fake_read(files, channel=None, **kwargs):
            calls["channel"] = channel
            return "data"

        with patch("gravyflow.src.dataset.acquisition.base.find_urls", side_effect=fake_find_urls), \
             patch("gravyflow.src.dataset.acquisition.base.TimeSeries.read", side_effect=fake_read):
            run = [r for r in gf.ObservingRun
                   if r.value.start_gps_time <= gps <= r.value.end_gps_time][0]
            BaseDataObtainer.get_segment_data_static(
                gps, gps + 32, gf.IFO.H1,
                run.value.frame_types[gf.DataQuality.BEST],
                run.value.channels[gf.DataQuality.BEST],
            )
        assert calls == {"frametype": frame_type, "channel": channel}


# =============================================================================
# REQUESTED SAMPLE RATE
# =============================================================================

def _fake_strain(start, rate, duration):
    """Gaussian strain as gwpy returns it: starting at ``start``, at ``rate``."""
    from gwpy.timeseries import TimeSeries
    value = np.random.default_rng(0).normal(size=int(rate * duration)).astype(np.float32)
    return TimeSeries(value, t0=start, sample_rate=rate, channel="H1:FAKE-STRAIN")


class TestSampleRate:

    def _fetch_recorder(self, calls):
        def fake_fetch(ifo, start, end, sample_rate=4096, **kwargs):
            calls.append(sample_rate)
            return _fake_strain(start, sample_rate, end - start)
        return fake_fetch

    @pytest.mark.parametrize("requested, fetched", [
        (2048.0, 4096), (4096.0, 4096), (8192.0, 16384), (16384.0, 16384),
    ])
    def test_noise_fetch_uses_needed_rate(self, requested, fetched):
        calls = []
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O3,
            cache_segments=False,
            prefetch_segments=0,
        )
        with patch("gravyflow.src.dataset.acquisition.base.find_urls",
                   side_effect=ValueError("no local frames")), \
             patch("gravyflow.src.dataset.acquisition.base.TimeSeries.fetch_open_data",
                   side_effect=self._fetch_recorder(calls)):
            segment = obtainer.get_segment(
                1238170000.0, 1238170000.0 + 64.2, requested, gf.IFO.L1, "segments/test"
            )
        assert calls == [fetched]
        assert segment is not None
        assert segment.data.shape[0] == int(64 * requested)

    def test_transient_fetch_uses_needed_rate(self):
        calls = []
        with patch("gravyflow.src.dataset.acquisition.base.find_urls",
                   side_effect=ValueError("no local frames")), \
             patch("gravyflow.src.dataset.acquisition.base.TimeSeries.fetch_open_data",
                   side_effect=self._fetch_recorder(calls)):
            BaseDataObtainer.get_segment_data_static(
                1238170000.0, 1238170032.0, gf.IFO.L1, "HOFT_C01",
                "DCS-CALIB_STRAIN_CLEAN_C01", sample_rate_hertz=16384.0,
            )
        assert calls == [16384]

    def test_rates_above_gwosc_are_refused(self):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O3,
            cache_segments=False,
            prefetch_segments=0,
        )
        obtainer.valid_segments = np.array([[[1238170000.0, 1238172048.0]]])
        with patch("gravyflow.src.dataset.acquisition.base.find_urls",
                   side_effect=ValueError("no local frames")), \
             patch("gravyflow.src.dataset.acquisition.base.TimeSeries.fetch_open_data",
                   side_effect=self._fetch_recorder([])):
            with pytest.raises(ValueError, match="upsample"):
                next(obtainer.acquire(32768.0, ifos=[gf.IFO.L1]))


# =============================================================================
# GRAVITY SPY CATALOGUE (Zenodo 5649212)
# =============================================================================

def _fixture_rows(ifo, run):
    with open(FIXTURES / f"gravityspy_{ifo}_{run}.csv", newline="") as handle:
        return list(csv.DictReader(handle))


class TestGravitySpyCatalogue:

    def test_glitch_times_from_public_release(self, offline_public_sources):
        times = gf.get_glitch_times(gf.IFO.H1, observing_run=gf.ObservingRun.O1)
        rows = _fixture_rows("H1", "O1")
        expected = sorted(
            float(r["event_time"]) for r in rows
            if float(r["ml_confidence"]) >= 0.9 and r["ml_label"] != "No_Glitch"
        )
        np.testing.assert_allclose(np.sort(times), expected)
        assert any("zenodo.org/api/records/5649212/files/H1_O1.csv" in url
                   for url in offline_public_sources.requests)

    def test_confidence_floor_and_all_classes(self, offline_public_sources):
        times, labels = gf.get_glitch_times_with_labels(
            gf.IFO.L1, observing_run=gf.ObservingRun.O3
        )
        rows = _fixture_rows("L1", "O3a") + _fixture_rows("L1", "O3b")
        kept = [r for r in rows
                if float(r["ml_confidence"]) >= 0.9 and r["ml_label"] != "No_Glitch"]
        assert len(times) == len(kept)
        names = {gf.GlitchType(int(v)).name for v in labels}
        assert "CHIRP" in names or not any(r["ml_label"] == "Chirp" for r in kept)
        assert "WANDERING_LINE" in names or not any(r["ml_label"] == "Wandering_Line" for r in kept)
        assert {"BLIP_LOW_FREQUENCY", "FAST_SCATTERING"} & names
        # A lower floor keeps more.
        loose, _ = gf.get_glitch_times_with_labels(
            gf.IFO.L1, observing_run=gf.ObservingRun.O3, min_ml_confidence=0.5
        )
        assert len(loose) > len(times)

    def test_label_sets_map_across_eras(self):
        expected = {
            "1080Lines": "LINES_1080", "1400Ripples": "RIPPLES_1400",
            "Blip_Low_Frequency": "BLIP_LOW_FREQUENCY", "Fast_Scattering": "FAST_SCATTERING",
            "None_of_the_Above": "NONE_OF_THE_ABOVE", "Chirp": "CHIRP",
            "Wandering_Line": "WANDERING_LINE",
        }
        for label, name in expected.items():
            assert gf.GlitchType.from_gravity_spy_label(label) == gf.GlitchType[name]
        # Existing integer labels are unchanged.
        assert gf.GlitchType.BLIP.value == 1 and gf.GlitchType.WHISTLE.value == 19
        # Every label in every fixture era maps.
        for path in FIXTURES.glob("gravityspy_*.csv"):
            for row in csv.DictReader(open(path, newline="")):
                gf.GlitchType.from_gravity_spy_label(row["ml_label"])

    def test_metadata_is_populated(self, offline_public_sources):
        from gravyflow.src.dataset.features.segment_builders import build_glitch_segments
        segments = build_glitch_segments(ifos=[gf.IFO.H1], observing_runs=[gf.ObservingRun.O2])
        assert len(segments) > 0
        rows = {round(float(r["event_time"]), 3): r for r in _fixture_rows("H1", "O2")}
        for segment in segments:
            row = rows[round(segment.transient_gps_time, 3)]
            assert segment.snr == pytest.approx(float(row["snr"]), rel=1e-6)
            assert segment.peak_freq == pytest.approx(float(row["peak_frequency"]), rel=1e-6)
            assert segment.duration == pytest.approx(float(row["duration"]), rel=1e-6)
            assert segment.ml_confidence == pytest.approx(float(row["ml_confidence"]), rel=1e-6)
            assert segment.ml_confidence >= 0.9

    def test_md5_mismatch_is_refused(self, offline_public_sources, tmp_path):
        offline_public_sources.corrupt_zenodo = True
        with pytest.raises(IOError, match="md5"):
            gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O1)
        cache = tmp_path / ".gravyflow" / "cache" / "gravityspy"
        assert not list(cache.glob("*.npz"))
        assert not list(cache.glob("*.csv"))

    def test_compact_cache_is_reused_and_outside_package(self, offline_public_sources, tmp_path):
        from gravyflow.src.dataset.features import gravity_spy
        gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O2)
        n_requests = len(offline_public_sources.requests)
        gravity_spy.clear_memory_cache()
        gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O2)
        assert len(offline_public_sources.requests) == n_requests
        cache = tmp_path / ".gravyflow" / "cache" / "gravityspy"
        assert [p.name for p in cache.glob("*.npz")] == ["L1_O2.npz"]
        assert not list(cache.glob("*.csv"))  # the raw download is not kept

    def test_lvk_database_path_is_gone(self):
        from gravyflow.src.dataset.features import glitch
        assert not hasattr(glitch, "GravitySpyTable")
        assert not hasattr(glitch, "fetch_event_times")
        assert not hasattr(glitch, "_get_cache_path")

    def test_no_public_release_for_o4(self, offline_public_sources):
        with pytest.raises(ValueError, match="Gravity Spy"):
            gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O4)


# =============================================================================
# PUBLIC-DATA DEFAULTS
# =============================================================================

class TestPublicDefaults:

    def test_real_noise_obtainer_defaults_to_public_data(self):
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1])
        assert noise.ifo_data_obtainer is not None
        assert gf.DataLabel.NOISE in noise.ifo_data_obtainer.data_labels
        assert noise.ifo_data_obtainer.state_flags[0] == (
            "DATA", "BURST_CAT1", "BURST_CAT2",
            "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ",
        )

    def test_example_model_real_noise_builds_offline(self, offline_public_sources):
        dataset = gf.Gabbard2017.dataset(
            noise_type=gf.NoiseType.REAL, seed=42, group="train", steps_per_epoch=1
        )
        assert dataset is not None

    def test_default_dataset_builds_offline(self, offline_public_sources):
        dataset = gf.GravyflowDataset(input_variables=[gf.ReturnVariables.ONSOURCE])
        assert dataset is not None


# =============================================================================
# ONLINE (opt-in: --network)
# =============================================================================

@pytest.mark.network
class TestOnline:

    def test_run_bounds_match_live_api(self):
        from gravyflow.src.dataset.acquisition.gwosc_data import fetch_gwosc_runs, GWOSC_RUNS
        live = fetch_gwosc_runs()
        for name, run in GWOSC_RUNS.items():
            assert live[name] == (run["gps_start"], run["gps_end"])

    def test_default_configuration_builds_from_gwosc(self):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],  # events and glitches vetoed
            cache_segments=False,
        )
        segments = obtainer.get_valid_segments(
            ifos=[gf.IFO.H1, gf.IFO.L1], seed=1, group_name="train"
        )
        assert segments.ndim == 3 and len(segments) > 100

    def test_release_md5s_match_zenodo(self):
        import requests
        from gravyflow.src.dataset.features.gravity_spy import RELEASE_FILES, ZENODO_RECORD
        record = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60).json()
        files = {f["key"]: (f["checksum"].split(":")[1], f["size"]) for f in record["files"]}
        assert record["metadata"]["license"]["id"] == "cc-by-4.0"
        assert len(files) == len(RELEASE_FILES) == 8
        for key, md5, size in RELEASE_FILES.values():
            assert files[key] == (md5, size)

    def test_smallest_release_file_downloads(self):
        times = gf.get_glitch_times(gf.IFO.H1, observing_run=gf.ObservingRun.O1)
        assert len(times) > 1000
