"""
Tests for real-noise vetoes and segment yield.

- Event and glitch vetoes remove every vetoed time, whatever the order of the lists,
  over every configured run, covering each glitch's whole extent and each event's
  in-band signal.
- Catalogued GW events never appear in the glitch catalogue.
- Hardware injections are excluded by default.
- Segments are cut on an absolute 2048 s GPS grid, every piece of each bin is kept,
  and segments are fetched whole: no margin trimmed, no power-of-two truncation.
- Windows are drawn uniformly over each detector's usable pieces in each bin: no
  usable piece is dropped and no short piece is oversampled.

Offline: the GWOSC timeline, the Gravity Spy release and the event catalogue come from
the committed fixtures, through the harness in ``test_public_sources``.
"""
from unittest.mock import patch

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition import gwosc_data
from gravyflow.src.dataset.features import gravity_spy
from gravyflow.tests import test_public_sources as public
from gravyflow.tests.test_public_sources import (  # noqa: F401  (pytest fixture)
    FIXTURE_EVENTS, offline_public_sources,
)


def _inside(times, segments):
    """Boolean mask: which times lie inside any [start, end) segment."""
    segments = np.reshape(np.asarray(segments, dtype=float), (-1, 2))
    times = np.asarray(times, dtype=float)
    return np.array([
        bool(np.any((segments[:, 0] <= t) & (t < segments[:, 1]))) for t in times
    ], dtype=bool)


def _overlap(a, b):
    """Total overlap in seconds between two segment lists."""
    total = 0.0
    for s0, s1 in np.reshape(a, (-1, 2)):
        for v0, v1 in np.reshape(b, (-1, 2)):
            total += max(0.0, min(s1, v1) - max(s0, v0))
    return total


def _noise_obtainer(runs, **kwargs):
    return gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        observing_runs=runs,
        cache_segments=False,
        prefetch_segments=0,
        **kwargs,
    )


# =============================================================================
# SEGMENT SUBTRACTION
# =============================================================================

class TestVetoSubtraction:

    @pytest.fixture
    def obtainer(self):
        return _noise_obtainer(gf.ObservingRun.O3)

    def test_unsorted_overlapping_vetoes_remove_every_vetoed_time(self, obtainer):
        segments = np.array([[0.0, 100.0], [200.0, 300.0]])
        vetoes = np.array([
            [250.0, 254.0], [10.0, 14.0], [50.0, 54.0], [12.0, 20.0],
            [290.0, 310.0], [-5.0, 2.0],
        ])
        result = obtainer.veto_time_segments(segments, vetoes)
        np.testing.assert_array_equal(result, [
            [2.0, 10.0], [20.0, 50.0], [54.0, 100.0], [200.0, 250.0], [254.0, 290.0],
        ])

    def test_unsorted_segments(self, obtainer):
        segments = np.array([[200.0, 300.0], [0.0, 100.0]])
        result = obtainer.veto_time_segments(segments, np.array([[50.0, 60.0]]))
        np.testing.assert_array_equal(result, [[0.0, 50.0], [60.0, 100.0], [200.0, 300.0]])

    def test_random_vetoes_match_a_sample_mask(self, obtainer):
        rng = np.random.default_rng(3)
        starts = np.sort(rng.choice(np.arange(0, 100_000, 10), 300, replace=False))
        segments = np.stack([starts, starts + rng.integers(1, 10, 300)], axis=1).astype(float)
        veto_starts = rng.integers(0, 100_000, 2_000)
        vetoes = np.stack([veto_starts, veto_starts + rng.integers(1, 30, 2_000)], axis=1)
        rng.shuffle(vetoes)
        segments_shuffled = segments.copy()
        rng.shuffle(segments_shuffled)

        result = obtainer.veto_time_segments(segments_shuffled, vetoes.astype(float))

        mask = np.zeros(100_100, dtype=bool)
        for a, b in segments.astype(int):
            mask[a:b] = True
        for a, b in vetoes:
            mask[a:b] = False
        got = np.zeros_like(mask)
        for a, b in result:
            got[int(a):int(b)] = True
        assert np.array_equal(got, mask)


    def test_zero_length_veto_does_not_split_a_segment(self, obtainer):
        result = obtainer.veto_time_segments(
            np.array([[16.0, 48.0], [26.0, 55.0], [25.0, 59.0]]), np.array([[23.0, 23.0]])
        )
        np.testing.assert_array_equal(result, [[16.0, 59.0]])

    def test_no_vetoes_still_sorts_coalesces_and_drops_empty_pieces(self, obtainer):
        segments = np.array([[33.0, 33.0], [40.0, 50.0], [3.0, 11.0], [11.0, 20.0]])
        expected = [[3.0, 20.0], [40.0, 50.0]]
        np.testing.assert_array_equal(obtainer.veto_time_segments(segments, np.empty((0, 2))),
                                      expected)
        np.testing.assert_array_equal(gwosc_data.subtract_segments(segments, np.empty((0, 2))),
                                      expected)
        np.testing.assert_array_equal(gwosc_data.coalesce_segments(segments), expected)


# =============================================================================
# RUN ORDER
# =============================================================================

class TestRunOrder:

    def test_non_chronological_runs_bin_and_veto(self):
        obtainer = _noise_obtainer([gf.ObservingRun.O3, gf.ObservingRun.O1])
        o3 = np.array([[1238170000.0, 1238180000.0]])
        o1 = np.array([[1126060000.0, 1126070000.0]])
        with patch.object(obtainer, "get_segment_times", side_effect=[o3, o1]):
            segments = obtainer.get_all_segment_times(gf.IFO.L1)
        np.testing.assert_array_equal(segments, np.concatenate([o1, o3]))

        with patch.object(obtainer, "get_all_segment_times", return_value=np.concatenate([o3, o1])), \
             patch.object(obtainer, "remove_unwanted_segments",
                          side_effect=lambda ifo, s: s):
            rows = obtainer.get_valid_segments(
                ifos=[gf.IFO.L1], seed=1, group_name="all",
                segment_order=gf.SegmentOrder.CHRONOLOGICAL,
            )
        assert rows.shape[1:] == (1, 2)
        assert np.isclose(np.sum(rows[:, 0, 1] - rows[:, 0, 0]), 20000.0)


# =============================================================================
# VETOES IN THE NOISE PATH
# =============================================================================

def _all_fixture_glitches(ifo_name, runs):
    rows = []
    for run in runs:
        rows.append(gravity_spy.query(
            ifo_name, run.value.start_gps_time, run.value.end_gps_time,
            labels=None, min_ml_confidence=0.0,
        ))
    return {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}


class TestNoiseVetoes:

    @pytest.mark.parametrize("runs", [
        [gf.ObservingRun.O2, gf.ObservingRun.O3],
        [gf.ObservingRun.O3, gf.ObservingRun.O2],
    ])
    def test_glitch_vetoes_cover_every_run(self, offline_public_sources, runs):
        obtainer = _noise_obtainer(runs)
        segments = obtainer.get_all_segment_times(gf.IFO.L1)
        glitches = _all_fixture_glitches("L1", runs)
        in_data = _inside(glitches["event_time"], segments)
        # Every configured run has fixture glitches inside its data, so a veto that
        # skipped any run would leave some.
        for run in runs:
            in_run = ((glitches["event_time"] >= run.value.start_gps_time)
                      & (glitches["event_time"] < run.value.end_gps_time))
            assert np.sum(in_data & in_run) > 5, run

        result = obtainer.remove_unwanted_segments(gf.IFO.L1, segments)

        assert not np.any(_inside(glitches["event_time"], result))
        # The whole Omicron extent is vetoed, not just the peak.
        extents = np.stack([glitches["start_time"],
                            glitches["start_time"] + glitches["duration"]], axis=1)
        assert _overlap(result, extents[in_data]) == 0.0

    def test_glitch_veto_padding(self):
        """Each glitch's extent is widened by 1 s, and by 3 s for Extremely_Loud."""
        from gravyflow.src.dataset.features.glitch import GlitchType

        catalogue = {
            "event_time": np.array([1000.5, 2000.5]),
            "start_time": np.array([1000.0, 2000.0]),
            "duration": np.array([1.0, 1.0]),
            "label": np.array([int(GlitchType.BLIP), int(GlitchType.EXTREMELY_LOUD)]),
        }
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        with patch.object(obtainer, "_glitch_catalogues", return_value=[catalogue]):
            vetoes = obtainer.get_glitch_veto_segments(gf.IFO.L1)
        np.testing.assert_array_equal(vetoes, [[999.0, 1002.0], [1997.0, 2004.0]])

        result = obtainer.veto_time_segments(np.array([[900.0, 2100.0]]), vetoes)
        np.testing.assert_array_equal(result, [[900.0, 999.0], [1002.0, 1997.0],
                                               [2004.0, 2100.0]])

    def test_glitch_veto_covers_pre_peak_extent(self, offline_public_sources):
        catalogue = gf.get_glitch_catalogue(gf.IFO.L1, gf.ObservingRun.O3)
        segments = gf.get_glitch_segments(gf.IFO.L1, gf.ObservingRun.O3)
        assert len(segments) == len(catalogue["event_time"]) > 0
        assert np.all(segments[:, 0] <= catalogue["start_time"])
        assert np.all(segments[:, 1] >= catalogue["start_time"] + catalogue["duration"])
        assert np.all(segments[:, 0] < catalogue["event_time"])
        assert np.all(segments[:, 1] > catalogue["event_time"])

    def test_event_veto_covers_in_band_signal(self, offline_public_sources):
        obtainer = _noise_obtainer(gf.ObservingRun.O2)
        merger = 1187008882.4  # GW170817: about 165 s in band from 20 Hz
        segments = np.array([[merger - 1000.0, merger + 1000.0]])
        result = obtainer.remove_unwanted_segments(gf.IFO.L1, segments)
        assert _overlap(result, [[merger - 166.0, merger + 1.0]]) == 0.0

    def test_event_veto_durations(self):
        vetoes = gf.get_event_veto_segments(catalogue=FIXTURE_EVENTS)
        by_start = {name: None for name in FIXTURE_EVENTS["name"]}
        for name, gps in zip(FIXTURE_EVENTS["name"], FIXTURE_EVENTS["gps"]):
            (row,) = [v for v in vetoes if v[0] <= gps <= v[1]]
            by_start[name] = (gps - row[0], row[1] - gps)
        before, after = by_start["GW170817"]
        assert 166.0 <= before <= 230.0 and after >= 1.0
        before, after = by_start["GW150914"]
        assert 2.0 <= before <= 5.0 and after >= 1.0

    def test_events_without_masses_get_the_longest_duration(self):
        catalogue = {"gps": np.array([1e9]), "chirp_mass_source": np.array([np.nan]),
                     "name": np.array(["x"]), "catalog": np.array(["GWTC-3-marginal"])}
        (veto,) = gf.get_event_veto_segments(catalogue=catalogue)
        assert 1e9 - veto[0] > 250.0


# =============================================================================
# THE EVENT LIST
# =============================================================================

def _table(rows, columns):
    from astropy.table import Table
    return Table(rows=rows, names=columns) if rows else Table(names=columns)


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


@pytest.fixture
def event_api(monkeypatch, tmp_path):
    from gwosc import api as gwosc_api
    from gravyflow.src.dataset.features import event as event_module

    monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path))
    fake = FakeEventApi({
        "GWTC": _table([(1126259462.4, "GW150914", 28.6)],
                       ["GPS", "commonName", "chirp_mass_source"]),
        "IAS-O3a": _table([(1253011836.5, "GW190920_113516", 3.7)],
                          ["GPS", "commonName", "chirp_mass_source"]),
        # Marginal IMBH candidates are published without masses.
        "O3_IMBH_marginal": _table([(1263002916.2, "200114_020818")], ["GPS", "commonName"]),
    })
    monkeypatch.setattr(gwosc_api, "fetch_cataloglist_json", fake.cataloglist)
    monkeypatch.setattr(event_module.EventTable, "fetch_open_data", fake.fetch_open_data)
    return fake


class TestEventList:

    def test_every_catalogue_the_api_lists_is_vetoed(self, event_api):
        catalogue = gf.get_event_catalogue()
        assert sorted(set(catalogue["catalog"])) == ["GWTC", "IAS-O3a", "O3_IMBH_marginal"]
        assert sorted(event_api.fetched) == ["GWTC", "IAS-O3a", "O3_IMBH_marginal"]

        vetoes = gf.get_event_veto_segments(catalogue=catalogue)
        for gps in (1126259462.4, 1253011836.5, 1263002916.2):
            assert _inside([gps], vetoes)[0]
        # No masses: the 1 + 1 solar-mass duration, the longest the veto uses.
        (imbh,) = [v for v in vetoes if v[0] <= 1263002916.2 <= v[1]]
        assert 1263002916.2 - imbh[0] > 250.0

    def test_cache_is_reused_while_the_runs_are_unchanged(self, event_api):
        first = gf.get_event_catalogue()
        second = gf.get_event_catalogue()
        assert len(event_api.fetched) == 3
        np.testing.assert_array_equal(first["gps"], second["gps"])

    def test_cache_is_refetched_when_a_run_is_added(self, event_api, monkeypatch):
        from gravyflow.src.dataset.acquisition import gwosc_data

        gf.get_event_catalogue()
        event_api.catalogues["GWTC"] = _table(
            [(1126259462.4, "GW150914", 28.6), (1430000000.0, "GW_NEW_RUN", 30.0)],
            ["GPS", "commonName", "chirp_mass_source"],
        )
        monkeypatch.setitem(gwosc_data.GWOSC_RUNS, "O4c", {
            "gps_start": 1427000000, "gps_end": 1440000000, "dataset": "O4c_4KHZ_R1",
        })
        catalogue = gf.get_event_catalogue()
        assert len(event_api.fetched) == 6
        assert 1430000000.0 in catalogue["gps"]

    def test_noise_vetoes_read_only_the_event_list(self, event_api):
        obtainer = _noise_obtainer(gf.ObservingRun.O1)
        segments = np.array([[1126259000.0, 1126260000.0]])
        with patch.object(obtainer, "get_glitch_veto_segments", return_value=np.empty((0, 2))):
            result = obtainer.remove_unwanted_segments(gf.IFO.L1, segments)
        assert _overlap(result, [[1126259460.4, 1126259464.4]]) == 0.0  # merger ± 2 s
        assert sorted(event_api.fetched) == ["GWTC", "IAS-O3a", "O3_IMBH_marginal"]
        assert gf.get_all_event_times().size == 3
        assert len(event_api.fetched) == 3  # the same cached list


# =============================================================================
# EVENTS ARE NEVER GLITCHES
# =============================================================================

class TestEventsNotGlitches:

    def test_event_in_glitch_catalogue_is_removed(self, offline_public_sources, monkeypatch):
        times = gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O3)
        assert len(times) > 10
        event_time = float(times[len(times) // 2])

        from gravyflow.src.dataset.features import event as event_module
        events = {key: value.copy() for key, value in FIXTURE_EVENTS.items()}
        events["gps"] = np.append(events["gps"], event_time + 0.04)
        events["chirp_mass_source"] = np.append(events["chirp_mass_source"], 60.0)
        events["name"] = np.append(events["name"], "GW_TEST")
        events["catalog"] = np.append(events["catalog"], "GWTC-3-confident")
        monkeypatch.setattr(event_module, "get_event_catalogue",
                            lambda use_cache=True: events, raising=False)

        after = gf.get_glitch_times(gf.IFO.L1, observing_run=gf.ObservingRun.O3)
        assert event_time not in after
        assert len(after) == len(times) - 1
        times_with_labels, _ = gf.get_glitch_times_with_labels(
            gf.IFO.L1, observing_run=gf.ObservingRun.O3
        )
        assert event_time not in times_with_labels


# =============================================================================
# HARDWARE INJECTIONS
# =============================================================================

class TestHardwareInjections:

    def test_default_flags_exclude_hardware_injections(self):
        # Every kind of hardware injection but the continuous-wave one, which runs
        # through nearly all of each run.
        assert gf.GWOSC_NO_HW_INJECTION_FLAGS == (
            "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ"
        )
        assert "NO_CW_HW_INJ" not in gf.DEFAULT_GWOSC_FLAGS
        assert gf.DEFAULT_GWOSC_FLAGS == gf.GWOSC_BURST_FLAGS + gf.GWOSC_NO_HW_INJECTION_FLAGS
        obtainer = _noise_obtainer(gf.ObservingRun.O1)
        assert obtainer.state_flags[0] == gf.DEFAULT_GWOSC_FLAGS

    @pytest.mark.parametrize("flag", [
        "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ",
    ])
    def test_injection_times_are_removed(self, offline_public_sources, monkeypatch, flag):
        entry = public.TIMELINE["O1"]
        start, end = entry["window"]
        data = public._fixture_intersection("O1", "H1", ["DATA", "BURST_CAT1", "BURST_CAT2"])
        a, b = data[np.argmax(data[:, 1] - data[:, 0])]
        injection = [a + 100.0, a + 110.0]
        monkeypatch.setitem(entry, f"H1_{flag}", [[start, injection[0]], [injection[1], end]])

        segments = _noise_obtainer(gf.ObservingRun.O1).get_all_segment_times(gf.IFO.H1)

        assert _overlap(segments, [injection]) == 0.0
        assert _overlap(segments, [[a, injection[0]]]) == 100.0


# =============================================================================
# SEGMENT YIELD
# =============================================================================

def _block_inside_o3(num_blocks):
    """A GPS time on the absolute 2048 s grid inside O3, with ``num_blocks`` blocks
    after it that do not straddle the relative 8192 s group cut."""
    start = gf.ObservingRun.O3.value.start_gps_time
    k = int(np.ceil((start + 10 * 8192) / 2048))
    while True:
        b = 2048.0 * k
        relative = (b - start) % 8192
        if relative + num_blocks * 2048 <= 8192:
            return b
        k += 1


class TestSegmentYield:

    def _rows(self, obtainer, per_ifo, ifos):
        with patch.object(obtainer, "get_all_segment_times", side_effect=per_ifo), \
             patch.object(obtainer, "remove_unwanted_segments",
                          side_effect=lambda ifo, s: s):
            return obtainer.get_valid_segments(
                ifos=ifos, seed=1, group_name="all",
                segment_order=gf.SegmentOrder.CHRONOLOGICAL,
            )

    def test_absolute_grid_keeps_whole_blocks(self):
        b = _block_inside_o3(2)
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        rows = self._rows(obtainer, [np.array([[b, b + 4096.0]])], [gf.IFO.L1])
        np.testing.assert_array_equal(
            rows[:, 0, :], [[b, b + 2048.0], [b + 2048.0, b + 4096.0]]
        )

    def test_every_piece_of_a_bin_is_kept(self):
        b = _block_inside_o3(1)
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        pieces = np.array([[b, b + 500.0], [b + 600.0, b + 2048.0]])
        rows = self._rows(obtainer, [pieces], [gf.IFO.L1])
        assert sorted(map(tuple, rows[:, 0, :])) == sorted(map(tuple, pieces))

    def test_every_piece_is_kept_for_each_ifo(self):
        b = _block_inside_o3(1)
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        h1 = np.array([[b, b + 2048.0]])
        l1 = np.array([[b, b + 500.0], [b + 600.0, b + 2048.0]])
        rows = self._rows(obtainer, [h1, l1], [gf.IFO.H1, gf.IFO.L1])
        assert rows.shape == (2, 2, 2)
        assert sorted(map(tuple, rows[:, 1, :])) == sorted(map(tuple, l1))
        assert all(tuple(row) == tuple(h1[0]) for row in rows[:, 0, :])
        # The longest pieces are paired.
        longest = rows[np.argmax(rows[:, 1, 1] - rows[:, 1, 0])]
        assert tuple(longest[1]) == (b + 600.0, b + 2048.0)

    def test_the_relative_group_cut_still_splits_a_bin(self):
        # The 8192 s group cut is counted from the first run's start, not GPS 0, so a
        # contiguous stretch is split where it falls inside a 2048 s bin.
        start = gf.ObservingRun.O3.value.start_gps_time
        cut = start + 11 * 8192.0
        b = 2048.0 * np.floor(cut / 2048.0)
        assert b < cut < b + 2048.0
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        rows = self._rows(obtainer, [np.array([[b, b + 2048.0]])], [gf.IFO.L1])
        assert sorted(map(tuple, rows[:, 0, :])) == [(b, cut), (cut, b + 2048.0)]

    def test_merge_bins_with_an_empty_ifo_gives_no_bins(self):
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        assert obtainer.merge_bins([np.array([[0.0, 10.0]]), np.empty((0, 2))], 2048.0) == []


# =============================================================================
# SAMPLING DENSITY
# =============================================================================

from gravyflow.src.dataset.acquisition.base import kept_seconds  # noqa: E402
from gravyflow.src.dataset.acquisition.noise import couple_pieces  # noqa: E402

#: The test window at 64 Hz: 1 s onsource with 0.5 s crops (2 s loaded) and 1.75 s
#: offsource, plus the sampler's 16-sample buffer: 4 s of footprint, so a piece of
#: 4 + x whole seconds has x seconds of window starts. Usable pieces: >= 7.5 s.
RATE, BATCH = 64.0, 16
FOOTPRINT, OFFSOURCE = 4.0, 1.75


def _draw(obtainer, rows, num_batches, mode=gf.SamplingMode.RANDOM, offsource_seconds=OFFSOURCE,
          batch=BATCH, seed=7):
    """START_GPS_TIME of ``num_batches`` batches drawn from ``rows`` with fake strain."""
    def fake_acquire(sample_rate_hertz, valid_segments, ifos, scale_factor, **kwargs):
        # As the loader does: each piece is kept to whole seconds (kept_seconds).
        for row in valid_segments:
            data = [np.zeros(int(kept_seconds(end - start) * sample_rate_hertz), np.float32)
                    for start, end in row]
            yield gf.IFOData(data, sample_rate_hertz, [start for start, _ in row])

    obtainer.valid_segments = np.asarray(rows, dtype=np.float64)
    obtainer.augmentations = []  # saturation exactly: 8 gives 0.5 batches per second of starts
    ifos = [gf.IFO.H1, gf.IFO.L1][:obtainer.valid_segments.shape[1]]
    with patch.object(obtainer, "acquire", side_effect=fake_acquire):
        generator = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=RATE, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
            offsource_duration_seconds=offsource_seconds, num_examples_per_batch=batch, ifos=ifos,
            scale_factor=1.0, seed=seed, sampling_mode=mode,
        )
        starts = [np.asarray(next(generator)[gf.ReturnVariables.START_GPS_TIME])
                  for _ in range(num_batches)]
    return np.concatenate(starts)


def _windows_per_piece(starts, pieces):
    return np.array([np.sum((starts >= a) & (starts < b)) for a, b in pieces])


class TestSamplingDensity:
    """
    Each detector's windows are drawn uniformly over the window starts of its usable
    pieces in each bin: no usable piece is dropped and no short piece is oversampled.
    The test pieces give whole numbers of batches (0.5 batches per second of starts at
    the default saturation of 8), so one pass is exact.
    """

    def test_couple_pieces_uses_each_piece_in_proportion(self):
        index, weights = couple_pieces([np.array([3.0, 1.0]), np.array([2.0, 1.0, 1.0])])
        assert np.isclose(np.sum(weights), 1.0)
        for i, measure in enumerate([np.array([3.0, 1.0]), np.array([2.0, 1.0, 1.0])]):
            # Each detector's pieces come in order, each in one run of rows.
            assert np.all(np.diff(index[:, i]) >= 0)
            for k, m in enumerate(measure):
                in_rows = index[:, i] == k
                assert np.isclose(np.sum(weights[in_rows]), m / np.sum(measure))
        # The longest pieces go together.
        assert tuple(index[0]) == (0, 0)

    def test_short_pieces_are_not_oversampled(self):
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        pieces = np.array([[0.0, FOOTPRINT + 200.0], [300.0, 300.0 + FOOTPRINT + 20.0],
                           [2048.0, 2048.0 + FOOTPRINT + 100.0]])
        starts = _draw(obtainer, pieces[:, None, :], num_batches=100 + 10 + 50)
        density = _windows_per_piece(starts[:, 0], pieces) / np.array([200.0, 20.0, 100.0])
        np.testing.assert_array_equal(density, [8.0, 8.0, 8.0])  # saturation / onsource

    def test_no_usable_piece_is_dropped(self):
        # H1 has two usable pieces; L1's second piece is too short. Pairing by rank put
        # H1's second piece with it, and the row was dropped.
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        h1 = np.array([[0.0, FOOTPRINT + 100.0], [1000.0, 1000.0 + FOOTPRINT + 50.0]])
        l1 = np.array([[0.0, FOOTPRINT + 200.0], [1500.0, 1505.0]])
        rows = np.stack([h1, l1], axis=1)
        starts = _draw(obtainer, rows, num_batches=75)  # 0.5 x min(150, 200) s
        h1_density = _windows_per_piece(starts[:, 0], h1) / np.array([100.0, 50.0])
        l1_windows = _windows_per_piece(starts[:, 1], l1)
        np.testing.assert_array_equal(h1_density, [8.0, 8.0])
        np.testing.assert_array_equal(l1_windows, [75 * BATCH, 0])


    def test_a_pass_with_no_batch_due_does_not_end_the_generator(self):
        # One 40 s piece per detector at saturation 1: about 0.68 batches are due per
        # pass, so a pass can be due none. The generator ended on such a pass.
        piece = np.array([[1238166018.0, 1238166058.0]])
        rows = np.stack([piece, piece], axis=1)
        for seed in range(5):
            obtainer = _noise_obtainer(gf.ObservingRun.O3, saturation=1.0)
            starts = _draw(obtainer, rows, num_batches=20, offsource_seconds=16.0, batch=32, seed=seed)
            assert starts.shape == (20 * 32, 2)

    def test_the_plan_measures_pieces_as_the_loader_keeps_them(self):
        # The loader keeps whole seconds of a piece; the plan measured its exact length,
        # so short fractional pieces were drawn up to 5.6% too densely.
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        plan = obtainer.plan_sampling(np.array([[[0.0, 30.6]]]), 8.0, FOOTPRINT)
        np.testing.assert_allclose(plan.start_seconds, [30.0 - FOOTPRINT])

    def test_grid_walks_each_piece_once_across_its_rows(self):
        # Saturation 0.5: a 2 s stride visits each start once. One H1 piece shares five
        # rows with five L1 pieces, each due about 1.6 batches a pass. GRID wrapped inside
        # each row's share, overlapping that row's own windows.
        obtainer = _noise_obtainer(gf.ObservingRun.O3, saturation=0.5)
        h1 = np.array([[0.0, FOOTPRINT + 256.0]])
        l1 = np.array([[k * 100.0, k * 100.0 + FOOTPRINT + 51.0] for k in range(5)])
        rows = np.stack([np.repeat(h1, 5, axis=0), l1], axis=1)
        # 16 batches of 16 windows walk the piece's 128 two-second positions twice over,
        # across rows and passes.
        starts = _draw(obtainer, rows, num_batches=16, mode=gf.SamplingMode.GRID)[:, 0]
        positions, visits = np.unique(starts, return_counts=True)
        assert len(positions) == 128 and np.all(visits == 2)
        assert np.min(np.diff(positions)) >= 2.0 - 1e-9  # no window overlaps another


class _RecordingFetch:
    """Fake fetch_open_data: records the requested span; returns a ramp."""

    def __init__(self):
        self.calls = []

    def __call__(self, ifo, start, end, sample_rate=4096, **kwargs):
        from types import SimpleNamespace
        self.calls.append((start, end))
        value = np.arange(int(round((end - start) * sample_rate)), dtype=np.float32)
        return SimpleNamespace(sample_rate=SimpleNamespace(value=float(sample_rate)), value=value)


class TestWholeSegments:

    def _get(self, start, end, rate):
        fetch = _RecordingFetch()
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        with patch("gravyflow.src.dataset.acquisition.base.find_urls",
                   side_effect=ValueError("no local frames")), \
             patch("gravyflow.src.dataset.acquisition.base.TimeSeries.fetch_open_data",
                   side_effect=fetch):
            segment = obtainer.get_segment(start, end, rate, gf.IFO.L1, "segments/test")
        return segment, fetch.calls

    def test_no_margin_is_trimmed(self):
        start = 1238171648.0
        segment, calls = self._get(start, start + 2048.0, 2048.0)
        assert calls == [(start, start + 2048.0)]
        assert segment.shape[0] == 2048 * 2048

    @pytest.mark.parametrize("duration, rate", [(1500.0, 2048.0), (700.0, 4096.0), (2048.0, 4096.0)])
    def test_no_power_of_two_truncation(self, duration, rate):
        segment, _ = self._get(1238171648.0, 1238171648.0 + duration, rate)
        assert segment.shape[0] == int(duration * rate)

    def test_segment_starts_at_its_first_sample(self):
        segment, _ = self._get(1238171648.0, 1238171648.0 + 64.5, 4096.0)
        values = np.asarray(segment)
        assert values.shape[0] == 64 * 4096
        assert abs(values[0]) < 1.0 and abs(values[100] - 100.0) < 1.0
