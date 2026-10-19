"""
Tests for the public real-noise sources: GWOSC run bounds, GWOSC timeline segments,
O1/O2 frame types and channels, the requested sample rate, the Gravity Spy catalogue
and public-data defaults.

The default run is offline. Network access is replaced by small committed fixtures in
``fixtures/public_sources``:

- ``gwosc_timeline.json``: GWOSC timeline segments over one day in each of O1, O2, O3a,
  O3b, O4a and O4b, each centred on an event, from https://gwosc.org: DATA, BURST_CAT1
  and BURST_CAT2, and the hardware-injection flags NO_CBC_HW_INJ, NO_BURST_HW_INJ,
  NO_STOCH_HW_INJ and NO_DETCHAR_HW_INJ, for H1 and L1, and for V1 in O4b.
- ``gravityspy_{IFO}_{RUN}.csv``: a few hundred rows of the Gravity Spy machine-learning
  classifications (Glanzer et al. 2023, Class. Quantum Grav. 40 065004; Zenodo,
  doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0), restricted to the columns gravyflow
  reads.
- ``gwosc_events.csv``: every row of every catalogue GWOSC's event API listed on
  2026-10-16 (https://gwosc.org/eventapi/json/), as gravyflow reads them: 1,062 rows
  of 18 catalogues, with their GPS times and source-frame chirp masses.

That the live sources still agree with these fixtures is checked apart, in
``tests/online``.
"""
import csv
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.resources import PACKAGE
from gravyflow.src.dataset.acquisition import gwosc_data, sources
from gravyflow.src.dataset.acquisition.base import ObservingRunData
from gravyflow.tests.conftest import GwoscLocator, gwosc_serving, refuse_internet

FIXTURES = Path(__file__).parent / "fixtures" / "public_sources"
TIMELINE = json.loads((FIXTURES / "gwosc_timeline.json").read_text())

# GWOSC run API values (https://gwosc.org/api/v2/runs).
GWOSC_RUN_BOUNDS = {
    "O1": (1126051217, 1137254417),
    "O2": (1164556817, 1187733618),
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
    "O4a": (1368195220, 1389456018),
    "O4b": (1396417050, 1422118818),
}
# Public events inside the fixture windows, used as the event-veto list, with their
# source-frame chirp masses from GWOSC's GWTC table: GW150914, GW170817, GW190521,
# GW200129_065458, GW230529_181500 (O4a) and GW250114_082203 (O4b).
FIXTURE_EVENT_TIMES = np.array([1126259462.4, 1187008882.4, 1242442967.4, 1264316116.4,
                                1369419318.7, 1420878141.2])
FIXTURE_EVENTS = {
    "gps": FIXTURE_EVENT_TIMES,
    "chirp_mass_source": np.array([27.9, 1.186, 63.3, 27.2, 1.94, 28.71]),
    "name": np.array(["GW150914", "GW170817", "GW190521", "GW200129_065458",
                      "GW230529_181500", "GW250114_082203"]),
    "catalog": np.array(["GWTC-2.1-confident", "GWTC-1-confident",
                         "GWTC-2.1-confident", "GWTC-3-confident", "GWTC-4.0", "GWTC-5.0"]),
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
        from gwpy.table import EventTable

        monkeypatch.setattr(gwosc_api, "fetch_cataloglist_json", self.cataloglist)
        monkeypatch.setattr(EventTable, "fetch_open_data", self.fetch_open_data)
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
        (gf.ObservingRun.O4, ["O4a", "O4b"]),
    ])
    def test_bounds_match_gwosc_run_api(self, run, gwosc_runs):
        assert run.value.start_gps_time == GWOSC_RUN_BOUNDS[gwosc_runs[0]][0]
        assert run.value.end_gps_time == GWOSC_RUN_BOUNDS[gwosc_runs[-1]][1]

    def test_datetime_runs_count_leap_seconds(self):
        # 2017-01-01 00:00:00 UTC is GPS 1167264018 (18 leap seconds since 1980).
        run = ObservingRunData("custom", datetime(2017, 1, 1), datetime(2017, 1, 2), {})
        assert run.start_gps_time == 1167264018
        assert run.end_gps_time == 1167264018 + 86400
        # 2015-09-12 00:00:00 UTC is GPS 1126051217 (17 leap seconds then).
        run = ObservingRunData("custom", datetime(2015, 9, 12), datetime(2015, 9, 13), {})
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
        ).get_all_segment_times(gf.IFO.H1)
        livetime = lambda s: float(np.sum(s[:, 1] - s[:, 0]))
        assert livetime(default) < livetime(segments)

    def test_dqsegdb_flag_names_are_rejected(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            observing_runs=gf.ObservingRun.O3,
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
        )
        first = obtainer.get_all_segment_times(gf.IFO.L1)
        n_requests = len(offline_public_sources.requests)
        second = obtainer.get_all_segment_times(gf.IFO.L1)
        np.testing.assert_array_equal(first, second)
        assert len(offline_public_sources.requests) == n_requests
        cached = list((tmp_path / ".gravyflow" / "cache" / "gwosc_timeline").glob("*.json"))
        assert len(cached) == len(gf.DEFAULT_GWOSC_FLAGS)  # one file per flag, L1 O1
        assert not list((PACKAGE / "res").glob("**/*gwosc*"))


# =============================================================================
# THE DEFAULT [O1, O2, O3, O4] CONFIGURATION
# =============================================================================

class TestDefaultConfiguration:

    def test_ragged_runs_concatenate(self):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
        )
        per_run = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0]]),
            np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            np.array([[13.0, 14.0]]),
        ]
        with patch.object(obtainer, "get_segment_times", side_effect=per_run):
            segments = obtainer.get_all_segment_times(gf.IFO.H1)
        np.testing.assert_array_equal(segments, np.concatenate(per_run))

    def test_default_runs_build_offline(self, offline_public_sources):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
        )
        # O4's time is served with no glitch veto, and says so.
        with pytest.warns(gf.NoGlitchVetoWarning, match="No glitch veto for H1"):
            segments = obtainer.get_valid_segments(
                ifos=[gf.IFO.H1, gf.IFO.L1], seed=1, group_name="train"
            )
        assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
        assert len(segments) > 0
        runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3, gf.ObservingRun.O4]
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
# THE DATA SOURCE: GWOSC() BY DEFAULT, LocalFrames(...) BY NAME, NEVER A FALLBACK
# =============================================================================

#: A GPS time inside GWOSC's O3a run, one inside O1, and one in each of O4a and O4b.
O3A_GPS, O1_GPS = 1238170000.0, 1126259446.0
O4A_GPS, O4B_GPS = 1378824192.0, 1410000000.0
#: Sources that must never share a cache entry or a fingerprint: equal but for the source.
SOURCES = {
    "GWOSC": lambda: gf.GWOSC(),
    "LocalFrames": lambda: gf.LocalFrames(),
    "LocalFrames C00": lambda: gf.LocalFrames(
        frametypes={"O3a": "HOFT_C00", "O3b": "HOFT_C00"},
        channels={"O3a": "GDS-CALIB_STRAIN_CLEAN", "O3b": "GDS-CALIB_STRAIN_CLEAN"},
    ),
}


def _noise_obtainer(source=None, **kwargs):
    return gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
        observing_runs=gf.ObservingRun.O3,
        source=source, **kwargs,
    )


def _fake_strain(start, rate, duration, channel="H1:FAKE-STRAIN"):
    """Gaussian strain as gwpy returns it: starting at ``start``, at ``rate``."""
    from gwpy.timeseries import TimeSeries
    value = np.random.default_rng(0).normal(size=int(rate * duration)).astype(np.float32)
    return TimeSeries(value, t0=start, sample_rate=rate, channel=channel)


class GwoscRecorder:
    """
    GWOSC as gwosc 0.8 and gwpy see it, offline: ``locate`` is gwosc's ``get_urls``
    (``GwoscLocator``), and ``__call__`` gwpy's ``fetch_open_data``, which locates the
    same way and serves Gaussian strain. Each records its calls' keywords.
    """

    def __init__(self, releases=None):
        self.locate = GwoscLocator(releases)
        self.calls = []

    @property
    def located(self):
        return self.locate.calls

    def __call__(self, ifo, start, end, sample_rate=4096, **kwargs):
        self.calls.append({"ifo": ifo, "start": start, "end": end, "sample_rate": sample_rate, **kwargs})
        self.locate.release(kwargs.get("dataset"), sample_rate)
        return _fake_strain(start, sample_rate, end - start)


@contextmanager
def serving_gwosc(recorder):
    """Every GWOSC lookup and download is served by ``recorder``."""
    with patch("gwosc.locate.get_urls", side_effect=recorder.locate), \
         patch("gwpy.timeseries.TimeSeries.fetch_open_data", side_effect=recorder):
        yield recorder


class TestDataSource:

    def test_the_default_source_is_gwosc_and_an_obtainer_holds_its_own(self):
        assert gf.Defaults.source == gf.GWOSC()
        assert isinstance(gf.Defaults.source, gf.DataSource)
        assert _noise_obtainer().source == gf.GWOSC()
        local = gf.LocalFrames()
        assert _noise_obtainer(local).source is local
        with patch.object(gf.Defaults, "source", local):
            assert _noise_obtainer().source is local
        with pytest.raises(TypeError, match="gf.DataSource"):
            _noise_obtainer("GWOSC")

    def test_the_source_enters_the_fingerprint(self):
        from gravyflow.src.utils.fingerprint import canonical, config_fingerprint

        obtainer = _noise_obtainer()
        assert obtainer.specification()["source"] == gf.GWOSC()
        assert canonical(obtainer)["source"]["__type__"] == "GWOSC"
        prints = {name: config_fingerprint(_noise_obtainer(make())) for name, make in SOURCES.items()}
        assert len(set(prints.values())) == len(SOURCES), prints
        # Equal sources give equal fingerprints: the fingerprint reads content, not identity.
        assert config_fingerprint(_noise_obtainer(gf.LocalFrames())) == prints["LocalFrames"]

    def test_gwosc_and_local_frames_never_share_a_cache_entry(self):
        # The store's tiles, noise's and transients' alike, are named by the strain they
        # hold (``DataSource.source_id``).
        tiles = {name: make().source_id(O3A_GPS, 16384.0) for name, make in SOURCES.items()}
        assert len(set(tiles.values())) == len(SOURCES), tiles

    def test_an_unreachable_local_frame_refuses_by_name_and_never_falls_back(self, monkeypatch):
        from gravyflow.src.dataset.store import read

        monkeypatch.setattr(read, "FETCH_BACKOFF_SECONDS", 0.0)
        gwosc = GwoscRecorder()

        def unreachable(**kwargs):
            raise ConnectionRefusedError("the datafind server is unreachable")

        with patch("gwdatafind.find_urls", side_effect=unreachable), serving_gwosc(gwosc):
            with pytest.raises(gf.StrainUnavailable, match=(
                    r"LocalFrames\(.*H1 strain over GPS \[1238170000.0, 1238170032.0\).*"
                    r"ConnectionRefusedError: the datafind server is unreachable")) as refused:
                gf.LocalFrames().fetch(gf.IFO.H1, 16384.0, (O3A_GPS, O3A_GPS + 32.0))
            assert refused.value.source == gf.LocalFrames()
            # Through an obtainer too: the read is refused by name, never zero-filled,
            # and nothing is fetched from GWOSC.
            with pytest.raises(gf.StrainUnavailable, match="the datafind server is unreachable"):
                next(_noise_obtainer(gf.LocalFrames()).acquire(
                    2048.0, np.array([[[O3A_GPS, O3A_GPS + 64.0]]]), [gf.IFO.H1]))
        assert gwosc.calls == [], "a local read error fell back to GWOSC"

    def test_unreadable_frames_refuse_by_name(self):
        def read(*args, **kwargs):
            raise OSError("frame file truncated")

        with patch("gwdatafind.find_urls", return_value=["file:///frames/H-H1_HOFT_C01.gwf"]), \
             patch("gwpy.timeseries.TimeSeries.read", side_effect=read):
            with pytest.raises(gf.StrainUnavailable, match="OSError: frame file truncated"):
                gf.LocalFrames().fetch(gf.IFO.H1, 16384.0, (O3A_GPS, O3A_GPS + 32.0))
        with patch("gwdatafind.find_urls", return_value=[]):
            with pytest.raises(gf.StrainUnavailable, match="found no H1_HOFT_C01 frames"):
                gf.LocalFrames().fetch(gf.IFO.H1, 16384.0, (O3A_GPS, O3A_GPS + 32.0))

    @pytest.mark.parametrize("make, gps, frametype, channel", [
        (lambda: gf.LocalFrames(), O3A_GPS, "H1_HOFT_C01", "H1:DCS-CALIB_STRAIN_CLEAN_C01"),
        # GWOSC's own O1 and O2 frames, verified against its datafind server and the
        # frame files' channel lists, named explicitly at their 4096 Hz.
        (lambda: gf.LocalFrames(frametypes={"O1": "LOSC_4_V1"}, channels={"O1": "LOSC-STRAIN"},
                                sample_rate_hertz=4096.0),
         O1_GPS, "H1_LOSC_4_V1", "H1:LOSC-STRAIN"),
        (lambda: gf.LocalFrames(frametypes={"O2": "GWOSC_O2_4KHZ_R1"},
                                channels={"O2": "GWOSC-4KHZ_R1_STRAIN"}, sample_rate_hertz=4096.0),
         1187008866.0, "H1_GWOSC_O2_4KHZ_R1", "H1:GWOSC-4KHZ_R1_STRAIN"),
    ])
    def test_local_frames_read_exactly_the_frame_type_they_name(self, make, gps, frametype, channel):
        source, calls = make(), {}

        def find(**kwargs):
            calls.update(frametype=kwargs["frametype"], on_gaps=kwargs["on_gaps"])
            return ["file:///frames/fake.gwf"]

        def read(files, channel=None, start=None, end=None, **kwargs):
            calls["channel"] = channel
            return _fake_strain(start, source.sample_rate_hertz, end - start, channel)

        with patch("gwdatafind.find_urls", side_effect=find), \
             patch("gwpy.timeseries.TimeSeries.read", side_effect=read):
            strain = source.fetch(gf.IFO.H1, source.sample_rate_hertz, (gps, gps + 32.0))
        assert calls == {"frametype": frametype, "on_gaps": "raise", "channel": channel}
        assert strain.sample_rate_hertz == source.sample_rate_hertz and strain.t0 == gps
        assert strain.source == source.source_id(gps, source.sample_rate_hertz)

    def test_local_frames_never_switch_frame_type_or_rate(self):
        """A run the table does not name, and frames at another rate, refuse by name."""
        o3a_only = gf.LocalFrames(frametypes={"O3a": "HOFT_C00"}, channels={"O3a": "GDS-CALIB_STRAIN"})
        o3b = 1260000000.0
        with pytest.raises(ValueError, match="names no frame type for GWOSC run O3b"):
            o3a_only.fetch(gf.IFO.H1, 16384.0, (o3b, o3b + 32.0))
        with pytest.raises(ValueError, match="names no frame type for GWOSC run O1"):
            gf.LocalFrames().segments(gf.IFO.H1, ("DATA",), (O1_GPS, O1_GPS + 100.0))
        # O4 has no default frames: its GWOSC channels are not LVK's O3 ones, so a reader names them.
        with pytest.raises(ValueError, match="names no frame type for GWOSC run O4a"):
            gf.LocalFrames().fetch(gf.IFO.H1, 16384.0, (O4A_GPS, O4A_GPS + 32.0))
        assert sorted(gf.LocalFrames().frametypes) == ["O3a", "O3b"]

        def read_4096(files, start=None, end=None, **kwargs):
            return _fake_strain(start, 4096.0, end - start)

        with patch("gwdatafind.find_urls", return_value=["file:///frames/fake.gwf"]), \
             patch("gwpy.timeseries.TimeSeries.read", side_effect=read_4096):
            with pytest.raises(gf.StrainUnavailable, match="served at 4096 Hz, not its native 16384 Hz"):
                gf.LocalFrames().fetch(gf.IFO.H1, 16384.0, (O3A_GPS, O3A_GPS + 32.0))
        with pytest.raises(ValueError, match="serves 16384 Hz"):
            gf.LocalFrames().fetch(gf.IFO.H1, 4096.0, (O3A_GPS, O3A_GPS + 32.0))
        with pytest.raises(ValueError, match="frametypes and channels name different runs"):
            gf.LocalFrames(frametypes={"O3a": "HOFT_C00"}, channels={"O3b": "GDS-CALIB_STRAIN"})
        with pytest.raises(ValueError, match="Unknown GWOSC run"):
            gf.LocalFrames(frametypes={"O3": "HOFT_C00"}, channels={"O3": "GDS-CALIB_STRAIN"})

    def test_local_frames_plan_only_where_their_frames_are(self, offline_public_sources):
        """LocalFrames' segments are the GWOSC timeline where its frames are found."""
        o3a = (1238166018.0, 1253977218.0)
        timeline = gf.GWOSC().segments(gf.IFO.H1, ("DATA",), o3a)
        assert len(timeline)
        first, last = float(timeline[0, 0]), float(timeline[0, 1])
        # Frames over the first timeline segment, and over time the timeline excludes.
        found = [(first - 50.0, first - 10.0), (first, last)]
        with patch("gwdatafind.find_times", return_value=found) as find_times:
            local = gf.LocalFrames().segments(gf.IFO.H1, ("DATA",), o3a)
        np.testing.assert_array_equal(local, [[first, last]])
        assert find_times.call_args.args[:2] == ("H", "H1_HOFT_C01")

    @pytest.mark.parametrize("ifo, gps, rate, run, release", [
        (gf.IFO.L1, O3A_GPS, 4096.0, "O3a", "O3a_4KHZ_R1"),
        (gf.IFO.L1, O1_GPS, 16384.0, "O1", "O1_16KHZ"),
        (gf.IFO.H1, O4A_GPS, 4096.0, "O4a", "O4a_4KHZ_R1"),
        (gf.IFO.V1, O4B_GPS, 16384.0, "O4b", "O4b_16KHZ_R1"),
    ])
    def test_gwosc_fetches_the_runs_bulk_release_only(self, ifo, gps, rate, run, release):
        """
        gwosc names a run's bulk data by the run, and an event release never by a run, so
        the run is named on every lookup and download; the files GWOSC lists lie in the
        release ``source_id`` names. Naming the release itself (``^O3a_4KHZ_R1$``) matches
        no GWOSC dataset: every real fetch failed so.
        """
        gwosc = GwoscRecorder()
        with serving_gwosc(gwosc):
            strain = gf.GWOSC().fetch(ifo, rate, (gps, gps + 32.0))
        assert [(call["dataset"], call["sample_rate"]) for call in gwosc.located] == [(f"^{run}$", int(rate))]
        assert [(call["dataset"], call["sample_rate"]) for call in gwosc.calls] == [(f"^{run}$", int(rate))]
        assert gwosc.calls[0]["cache"] is False  # the store keeps the tile
        assert strain.source == f"gwosc-{release}" == gf.GWOSC().source_id(gps, rate)
        assert strain.sample_rate_hertz == rate and strain.t0 == gps and len(strain.data) == 32 * rate

    def test_a_file_outside_the_named_release_is_refused_by_name(self):
        """A run GWOSC serves from another release than the table names is never fetched as the named one."""
        releases = {run: dict(rates) for run, rates in sources.GWOSC_STRAIN_DATASETS.items()}
        releases["O4a"][4096.0] = "O4a_4KHZ_R2"
        gwosc = GwoscRecorder(releases)
        with serving_gwosc(gwosc):
            with pytest.raises(gf.StrainUnavailable, match=(
                    r"H1 strain over GPS .*GWOSC lists \['https://gwosc.org/archive/data/O4a_4KHZ_R2/.*'\] "
                    r"for run O4a at 4096 Hz, outside its release O4a_4KHZ_R1")):
                gf.GWOSC().fetch(gf.IFO.H1, 4096.0, (O4A_GPS, O4A_GPS + 32.0))
        assert gwosc.calls == [], "fetched strain from another release"

    def test_a_gwosc_failure_refuses_by_name(self):
        def refused(*args, **kwargs):
            raise ConnectionRefusedError("gwosc.org unreachable")

        # Whether the lookup or the download fails, the fetch is refused by name.
        with patch("gwosc.locate.get_urls", side_effect=refused):
            with pytest.raises(gf.StrainUnavailable, match=r"GWOSC\(\) could not deliver L1 strain.*unreachable"):
                gf.GWOSC().fetch(gf.IFO.L1, 4096.0, (O3A_GPS, O3A_GPS + 32.0))
        with serving_gwosc(GwoscRecorder()), \
                patch("gwpy.timeseries.TimeSeries.fetch_open_data", side_effect=refused):
            with pytest.raises(gf.StrainUnavailable, match=r"GWOSC\(\) could not deliver L1 strain.*unreachable"):
                gf.GWOSC().fetch(gf.IFO.L1, 4096.0, (O3A_GPS, O3A_GPS + 32.0))

    @pytest.mark.parametrize("gps, rate, expected", [
        (1238163456.0, 4096.0, "gwosc-O3a_4KHZ_R1"),  # the tile begins before O3a does
        (O3A_GPS, 16384.0, "gwosc-O3a_16KHZ_R1"),
        (1126072320.0, 4096.0, "gwosc-O1"),
        (1126072320.0, 16384.0, "gwosc-O1_16KHZ"),
        (1260000000.0, 4096.0, "gwosc-O3b_4KHZ_R1"),
        (1368195072.0, 4096.0, "gwosc-O4a_4KHZ_R1"),  # the tile begins before O4a does
        (O4B_GPS, 16384.0, "gwosc-O4b_16KHZ_R1"),
    ])
    def test_gwosc_names_the_release_of_the_run_a_tile_lies_in(self, gps, rate, expected):
        assert gf.GWOSC().source_id(gps, rate) == expected
        assert gf.GWOSC().tile_seconds(rate) == 4096.0

    def test_gwosc_refuses_time_outside_its_runs_and_rates_it_does_not_publish(self):
        with pytest.raises(ValueError, match="no GWOSC run"):
            gf.GWOSC().source_id(1200000000.0, 4096.0)
        with pytest.raises(ValueError, match="publishes 4096 and 16384 Hz"):
            gf.GWOSC().fetch(gf.IFO.L1, 2048.0, (O3A_GPS, O3A_GPS + 32.0))

    def test_every_gwosc_run_names_its_strain_release_at_every_rate(self):
        assert set(sources.GWOSC_STRAIN_DATASETS) == set(sources.GWOSC_RUNS)
        for run, releases in sources.GWOSC_STRAIN_DATASETS.items():
            assert tuple(sorted(releases)) == sources.GWOSC_SAMPLE_RATES_HERTZ, run
            # The 4096 Hz release is the dataset the run's timeline flags are published under.
            assert releases[4096.0] == sources.GWOSC_RUNS[run]["dataset"], run

    def test_a_run_without_a_named_release_is_refused_by_name(self, monkeypatch):
        releases = {run: dict(rates) for run, rates in sources.GWOSC_STRAIN_DATASETS.items()}
        del releases["O3a"][16384.0]
        monkeypatch.setattr(sources, "GWOSC_STRAIN_DATASETS", releases)
        with pytest.raises(ValueError, match="No GWOSC strain release is named for run O3a at 16384 Hz"):
            gf.GWOSC().source_id(O3A_GPS, 16384.0)


class TestSampleRate:

    @pytest.mark.parametrize("requested, fetched", [
        (2048.0, 4096.0), (4096.0, 4096.0), (8192.0, 16384.0), (16384.0, 16384.0),
    ])
    def test_the_lowest_native_rate_at_or_above_the_request_is_fetched(self, requested, fetched):
        assert sources.fetch_rate(gf.GWOSC(), requested) == fetched
        assert sources.fetch_rate(gf.LocalFrames(), requested) == 16384.0
        gwosc = GwoscRecorder()
        # 64 s tiles keep the test small; the piece crosses one tile boundary.
        with serving_gwosc(gwosc), patch.object(gf.GWOSC, "tile_seconds", lambda source, rate: 64.0):
            [row] = _noise_obtainer().acquire(
                requested, np.array([[[O3A_GPS, O3A_GPS + 64.2]]]), [gf.IFO.L1])
        assert [call["sample_rate"] for call in gwosc.calls] == [int(fetched)] * 2
        # The whole seconds inside the resampler's margin: 64 of the 64.2 s.
        assert row.data[0].shape[0] == int(64 * requested)

    def test_rates_above_the_source_and_not_whole_hertz_are_refused(self):
        with pytest.raises(gf.UpsamplingRefused, match="upsample"):
            sources.fetch_rate(gf.GWOSC(), 32768.0)
        with pytest.raises(ValueError, match="whole number of hertz"):
            sources.fetch_rate(gf.GWOSC(), 2048.5)
        obtainer = _noise_obtainer()
        obtainer.valid_segments = np.array([[[1238170000.0, 1238172048.0]]])
        gwosc = GwoscRecorder()
        with serving_gwosc(gwosc):
            with pytest.raises(ValueError, match="upsample"):
                next(obtainer.acquire(32768.0, ifos=[gf.IFO.L1]))
        assert gwosc.calls == [], "fetched before refusing the rate"


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
        assert names == {gf.GlitchType.from_gravity_spy_label(r["ml_label"]).name for r in kept}
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
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
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
# O4: GWOSC'S O4a (H1, L1) AND O4b (H1, L1, V1)
# =============================================================================

O4_EVENTS = {"GW230529_181500": 1369419318.7, "GW250114_082203": 1420878141.2}


def _o4_noise_obtainer():
    return gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                              observing_runs=gf.ObservingRun.O4)


class TestObservingRunO4:

    @pytest.mark.parametrize("ifo, datasets", [
        ("H1", ["O4a_4KHZ_R1", "O4b_4KHZ_R1"]),
        ("L1", ["O4a_4KHZ_R1", "O4b_4KHZ_R1"]),
        ("V1", ["O4b_4KHZ_R1"]),
    ])
    def test_o4_noise_is_read_where_o4s_default_flags_pass(self, offline_public_sources, ifo, datasets):
        """
        O4's readable time is GWOSC's O4a and O4b timelines under the default flags, each
        detector where GWOSC publishes it (Virgo in O4b only), asked for by those flags.
        """
        readable = _o4_noise_obtainer().get_all_segment_times(gf.IFO[ifo])
        expected = np.concatenate([_fixture_intersection(d, ifo, gf.DEFAULT_GWOSC_FLAGS) for d in datasets])
        np.testing.assert_array_equal(readable, expected)
        asked = {tuple(url.split("/timeline/segments/json/")[1].split("/")[:2])
                 for url in offline_public_sources.requests if "/timeline/segments/json/" in url}
        assert asked == {(d, f"{ifo}_{flag}") for d in datasets for flag in gf.DEFAULT_GWOSC_FLAGS}

    @pytest.mark.parametrize("ifo", ["H1", "L1"])
    def test_o4_events_are_vetoed_from_o4_noise(self, offline_public_sources, ifo):
        """
        Each O4 event in a detector's data is taken from its noise: its in-band signal
        and its footprint widened by the split's guard, as in every run. (The veto list
        is every detector's; Virgo was not observing at either fixture event.)
        """
        obtainer = _o4_noise_obtainer()
        readable = obtainer.get_all_segment_times(gf.IFO[ifo])
        admissible = obtainer.noise_admissible_segments(gf.IFO[ifo])
        vetoes = obtainer.get_event_veto_segments()
        bitten = 0
        for name, gps in O4_EVENTS.items():
            around = np.array([[gps - 300.0, gps + 300.0]])
            if not len(gwosc_data.intersect_segments(readable, around)):
                continue  # this detector had no data at the event
            bitten += 1
            (veto,) = [v for v in gwosc_data.coalesce_segments(vetoes) if v[0] <= gps <= v[1]]
            assert len(gwosc_data.intersect_segments(admissible, [veto])) == 0, name
        assert bitten >= 1, f"no O4 event lies in {ifo}'s fixture data: the test proves nothing"

    def test_o4_is_a_default_run(self):
        """
        Both obtainers default to O1, O2, O3 and O4, from one home
        (``DEFAULT_OBSERVING_RUNS``); the validation bank's real events and the segment
        timeline read it too.
        """
        from gravyflow.src.dataset.acquisition.base import DEFAULT_OBSERVING_RUNS

        assert DEFAULT_OBSERVING_RUNS == (
            gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3, gf.ObservingRun.O4)
        for labels in ([gf.DataLabel.NOISE], [gf.DataLabel.GLITCHES]):
            default = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=labels)
            assert default.start_gps_times[0] == gf.ObservingRun.O1.value.start_gps_time, labels
            assert default.end_gps_times[-1] == GWOSC_RUN_BOUNDS["O4b"][1], labels

    def test_glitches_are_read_in_the_runs_gravity_spy_covers(self, offline_public_sources):
        """
        A detector's glitches are read in the configured runs Gravity Spy's release
        covers, as a detector's strain is read in the runs that publish it: over the
        default runs, O1 to O3's, and none from O4. A detector no configured run covers
        is refused by name (O4 alone; Virgo in any run), never served empty.
        """
        from gravyflow.src.dataset.acquisition.base import DEFAULT_OBSERVING_RUNS
        from gravyflow.src.dataset.features.segment_builders import build_glitch_segments

        segments = build_glitch_segments(ifos=[gf.IFO.H1], observing_runs=list(DEFAULT_OBSERVING_RUNS))
        runs = {segment.observing_run for segment in segments}
        assert runs == {gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3}, runs
        o3 = build_glitch_segments(ifos=[gf.IFO.H1], observing_runs=[gf.ObservingRun.O3])
        assert {s.transient_gps_time for s in o3} == {
            s.transient_gps_time for s in segments if s.observing_run is gf.ObservingRun.O3}
        with pytest.raises(ValueError, match=r"No public Gravity Spy release covers L1 in O4: .*O1, O2, O3a, O3b"):
            build_glitch_segments(ifos=[gf.IFO.L1], observing_runs=[gf.ObservingRun.O4])
        with pytest.raises(ValueError, match=r"No public Gravity Spy release covers V1 in O1, O2, O3, O4"):
            build_glitch_segments(ifos=[gf.IFO.V1], observing_runs=list(DEFAULT_OBSERVING_RUNS))

    def test_a_detector_is_read_only_in_the_runs_gwosc_publishes_it_in(self, offline_public_sources):
        """
        Virgo is in O4b but not O4a (nor O1): its O4 time is O4b's, and no O4a timeline is
        asked for, which GWOSC does not publish (404). A detector no configured run
        publishes is refused by name.
        """
        assert gwosc_data.GWOSC_RUNS["O4a"]["detectors"] == ("H1", "L1")
        assert gwosc_data.GWOSC_RUNS["O4b"]["detectors"] == ("H1", "L1", "V1")
        readable = _o4_noise_obtainer().get_all_segment_times(gf.IFO.V1)
        assert len(readable) and readable[0, 0] >= GWOSC_RUN_BOUNDS["O4b"][0]
        assert not any("/O4a_4KHZ_R1/V1_" in url for url in offline_public_sources.requests)
        o1 = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                                observing_runs=gf.ObservingRun.O1)
        with pytest.raises(ValueError, match=r"GWOSC publishes no V1 data in O1: O1 has H1, L1"):
            o1.get_all_segment_times(gf.IFO.V1)

    def test_o4_glitches_are_refused_by_name(self, offline_public_sources):
        """
        Gravity Spy's public release covers O1 to O3b, H1 and L1. Asking for glitches in
        O4, or Virgo's in any run, is refused by name, never served empty.
        """
        with pytest.raises(ValueError, match=r"No public Gravity Spy release covers GPS .*O1, O2, O3a, O3b"):
            gf.get_glitch_catalogue(gf.IFO.H1, observing_run=gf.ObservingRun.O4)
        with pytest.raises(ValueError, match=r"No public Gravity Spy release for V1 O3a"):
            gf.get_glitch_catalogue(gf.IFO.V1, observing_run=gf.ObservingRun.O3)
        glitches = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
                                      observing_runs=gf.ObservingRun.O4)
        with pytest.raises(ValueError, match="No public Gravity Spy release covers"):
            glitches.get_valid_segments(ifos=[gf.IFO.L1], seed=1, group_name="train")

    @pytest.mark.parametrize("runs, ifo", [
        (gf.ObservingRun.O4, gf.IFO.L1),
        (gf.ObservingRun.O3, gf.IFO.V1),
    ])
    def test_noise_without_a_glitch_veto_is_served_with_a_visible_warning(self, offline_public_sources, runs, ifo):
        """
        Noise in O4, and Virgo's in any run, has no Gravity Spy labels to veto its
        glitches: it is served with no glitch veto, under a ``gf.NoGlitchVetoWarning``
        (a Python warning, shown by default) naming the detector, the time and why.
        """
        obtainer = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                                      observing_runs=runs)
        with pytest.warns(gf.NoGlitchVetoWarning) as caught:
            vetoes = obtainer.get_glitch_veto_segments(ifo)
        assert vetoes.shape == (0, 2)
        (warning,) = [w for w in caught if issubclass(w.category, gf.NoGlitchVetoWarning)]
        message = str(warning.message)
        assert f"No glitch veto for {ifo.name}" in message and "Gravity Spy" in message, message
        assert issubclass(gf.NoGlitchVetoWarning, UserWarning)

    def test_noise_with_a_glitch_veto_gives_no_warning(self, offline_public_sources):
        """O3's LIGO noise is glitch-vetoed from Gravity Spy: no warning."""
        import warnings

        obtainer = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                                      observing_runs=gf.ObservingRun.O3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vetoes = obtainer.get_glitch_veto_segments(gf.IFO.H1)
        assert len(vetoes) > 0
        assert not [w for w in caught if "glitch veto" in str(w.message)], [str(w.message) for w in caught]


# =============================================================================
# PUBLIC-DATA DEFAULTS
# =============================================================================

class TestPublicDefaults:

    def test_real_noise_obtainer_defaults_to_public_data(self):
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1])
        assert noise.ifo_data_obtainer is not None
        assert gf.DataLabel.NOISE in noise.ifo_data_obtainer.data_labels
        assert noise.ifo_data_obtainer.source == gf.GWOSC()
        assert noise.ifo_data_obtainer.state_flags[0] == (
            "DATA", "BURST_CAT1", "BURST_CAT2",
            "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ",
        )

    @pytest.mark.parametrize("build", [
        lambda: gf.Gabbard2017.dataset(noise_type=gf.NoiseType.REAL, seed=42, group="train",
                                       steps_per_epoch=1),
        lambda: gf.GravyflowDataset(input_variables=[gf.ReturnVariables.ONSOURCE], group="train"),
    ], ids=["example_model_real_noise", "default_dataset"])
    def test_builds_and_serves_a_batch_offline(self, offline_public_sources, build):
        """
        A batch is drawn from GWOSC's timeline and strain (both served offline), the
        segments planned from the default flags and the glitch vetoes from Gravity Spy.
        """
        with gwosc_serving(GaussianTiles()) as fetch:
            inputs, _ = build()[0]
        [strain] = [np.asarray(value) for value in inputs.values()]  # one input each
        assert strain.ndim == 3 and np.all(np.isfinite(strain)) and np.any(strain != 0)
        assert fetch.calls, "no strain was read"
        requested = offline_public_sources.requests
        flags = {url.split("/timeline/segments/json/")[1].split("/")[1].split("_", 1)[1]
                 for url in requested if "/timeline/segments/json/" in url}
        assert flags == set(gf.DEFAULT_GWOSC_FLAGS)
        assert any("zenodo.org/api/records/5649212/files/" in url for url in requested)


class GaussianTiles:
    """A GWOSC strain fetch, offline: unit Gaussian samples seeded by the interval, in 64 s tiles."""

    rates = (4096.0, 16384.0)
    tile_seconds = 64.0

    def __init__(self):
        self.calls = []

    def __call__(self, ifo, sample_rate_hertz, interval):
        from gravyflow.src.dataset.acquisition.sources import SegmentStrain

        start, end = interval
        self.calls.append((ifo.name, start, end))
        rng = np.random.default_rng([int(start), sum(map(ord, ifo.name))])
        samples = rng.standard_normal(int(round((end - start) * sample_rate_hertz))).astype(np.float32)
        return SegmentStrain(samples * 1e-21, float(start), float(sample_rate_hertz), "gwosc-test")
