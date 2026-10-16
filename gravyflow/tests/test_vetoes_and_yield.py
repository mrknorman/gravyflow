"""
Tests for real-noise vetoes and segment yield.

- Event and glitch vetoes remove every vetoed time, whatever the order of the lists,
  over every configured run, covering each glitch's whole extent and each event's
  in-band signal.
- Catalogued GW events never appear in the glitch catalogue.
- Hardware injections are excluded by default.
- Segments are cut on an absolute 2048 s GPS grid, every piece of each bin is kept,
  and segments are fetched whole: no margin trimmed, no power-of-two truncation.

Offline: the GWOSC timeline, the Gravity Spy release and the event catalogue come from
the committed fixtures, through the harness in ``test_public_sources``.
"""
from unittest.mock import patch

import numpy as np
import pytest

import gravyflow as gf
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
                          side_effect=lambda ifo, s, get_times=None: (s, {})):
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

    def test_glitch_vetoes_cover_every_run(self, offline_public_sources):
        runs = [gf.ObservingRun.O1, gf.ObservingRun.O3]
        obtainer = _noise_obtainer(runs)
        segments = obtainer.get_all_segment_times(gf.IFO.L1)
        glitches = _all_fixture_glitches("L1", runs)
        in_data = _inside(glitches["event_time"], segments)
        assert np.sum(in_data) > 5  # the fixture has glitches inside the data

        result, _ = obtainer.remove_unwanted_segments(gf.IFO.L1, segments)

        assert not np.any(_inside(glitches["event_time"], result))
        # The whole Omicron extent is vetoed, not just the peak.
        extents = np.stack([glitches["start_time"],
                            glitches["start_time"] + glitches["duration"]], axis=1)
        assert _overlap(result, extents[in_data]) == 0.0

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
        result, _ = obtainer.remove_unwanted_segments(gf.IFO.L1, segments)
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
                          side_effect=lambda ifo, s, get_times=None: (s, {})):
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

    def test_merge_bins_with_an_empty_ifo_gives_no_bins(self):
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        assert obtainer.merge_bins([np.array([[0.0, 10.0]]), np.empty((0, 2))], 2048.0) == []


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
