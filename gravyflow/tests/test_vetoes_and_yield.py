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
from gravyflow.tests.conftest import GwoscLocator
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


#: Slack on a GPS difference near 1.3e9 s: four float64 steps (2.4e-7 s each).
GPS_SLACK = 4 * np.spacing(1.3e9)


def expected_veto(chirp_mass_solar):
    """
    (seconds before, seconds after) an event its veto covers, computed apart from the
    library: the leading-order chirp time from 20 Hz, with LAL's G M_sun / c^3, times the
    library's stated factor, plus its padding; no masses reads as the stated 1 + 1 binary.
    """
    import lal
    from gravyflow.src.dataset.features import event

    if chirp_mass_solar is None or not np.isfinite(chirp_mass_solar):
        chirp_mass_solar = event.UNKNOWN_EVENT_CHIRP_MASS_SOLAR
    tau = (5.0 / 256.0) * (np.pi * 20.0) ** (-8.0 / 3.0) * (chirp_mass_solar * lal.MTSUN_SI) ** (-5.0 / 3.0)
    padding = event.EVENT_VETO_PADDING_SECONDS
    return [event.EVENT_VETO_CHIRP_TIME_FACTOR * tau + padding, padding]


#: Every block in train: the segment tests below read all time as one split.
ALL_TRAIN = gf.SplitSpec(fractions={"train": 1.0, "validate": 0.0, "test": 0.0})


def _noise_obtainer(runs, **kwargs):
    return gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        observing_runs=runs,
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

    def test_non_chronological_runs_concatenate_and_bin(self):
        """Runs given out of order are read in GPS order and binned; vetoes are tested in TestNoiseVetoes."""
        obtainer = _noise_obtainer([gf.ObservingRun.O3, gf.ObservingRun.O1], split_spec=ALL_TRAIN)
        o3 = np.array([[1238170000.0, 1238180000.0]])
        o1 = np.array([[1126060000.0, 1126070000.0]])
        with patch.object(obtainer, "get_segment_times", side_effect=[o3, o1]):
            segments = obtainer.get_all_segment_times(gf.IFO.L1)
        np.testing.assert_array_equal(segments, np.concatenate([o1, o3]))

        with patch.object(obtainer, "get_all_segment_times", return_value=np.concatenate([o3, o1])), \
             patch.object(obtainer, "remove_unwanted_segments",
                          side_effect=lambda ifo, s: s):
            rows = obtainer.get_valid_segments(
                ifos=[gf.IFO.L1], seed=1, group_name="train",
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

    @pytest.mark.parametrize("asked", [[gf.DataLabel.EVENTS], [gf.EventConfidence.CONFIDENT]])
    def test_noise_refuses_to_serve_events(self, asked):
        """Real events are never noise: asking noise for them is refused by name."""
        with pytest.raises(ValueError, match="Real events are not noise"):
            gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE, *asked],
                               observing_runs=gf.ObservingRun.O3)

    @pytest.mark.parametrize("asked, vetoed", [
        ([], {"events", "glitches"}),
        ([gf.DataLabel.GLITCHES], {"events"}),
        # Equal to DataLabel.EVENTS as a number, but glitches.
        ([gf.GlitchType.CHIRP], {"events"}),
    ])
    def test_vetoes_follow_the_kinds_of_data_asked_for(self, asked, vetoed):
        """
        Noise keeps out every event and glitch it does not ask for, by each label's
        kind, never its number (the label enums' values overlap).
        """
        obtainer = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE, *asked],
                                      observing_runs=gf.ObservingRun.O3)
        applied = set()

        def veto(kind):
            def segments(*args):
                applied.add(kind)
                return np.empty((0, 2))
            return segments

        with patch.object(obtainer, "get_event_veto_segments", veto("events")), \
                patch.object(obtainer, "get_glitch_veto_segments", veto("glitches")):
            obtainer.remove_unwanted_segments(gf.IFO.L1, np.array([[0.0, 100.0]]))
        assert applied == vetoed

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
        """Each event's veto is its stated factor times its chirp time from 20 Hz, padded both sides."""
        vetoes = gf.get_event_veto_segments(catalogue=FIXTURE_EVENTS)
        for name, gps, chirp_mass in zip(FIXTURE_EVENTS["name"], FIXTURE_EVENTS["gps"],
                                         FIXTURE_EVENTS["chirp_mass_source"]):
            (row,) = [v for v in vetoes if v[0] <= gps <= v[1]]
            np.testing.assert_allclose([gps - row[0], row[1] - gps], expected_veto(chirp_mass),
                                       rtol=0, atol=GPS_SLACK, err_msg=name)

    def test_events_without_masses_get_the_longest_duration(self):
        catalogue = {"gps": np.array([1e9]), "chirp_mass_source": np.array([np.nan]),
                     "name": np.array(["x"]), "catalog": np.array(["GWTC-3-marginal"])}
        (veto,) = gf.get_event_veto_segments(catalogue=catalogue)
        np.testing.assert_allclose([1e9 - veto[0], veto[1] - 1e9], expected_veto(None), rtol=0, atol=GPS_SLACK)


# =============================================================================
# THE EVENT LIST
# =============================================================================

def _table(rows, columns):
    from astropy.table import Table
    return Table(rows=rows, names=columns) if rows else Table(names=columns)


@pytest.fixture
def event_api(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path))
    return public.FakeEventApi({
        "GWTC": _table([(1126259462.4, "GW150914", 28.6)],
                       ["GPS", "commonName", "chirp_mass_source"]),
        "IAS-O3a": _table([(1253011836.5, "GW190920_113516", 3.7)],
                          ["GPS", "commonName", "chirp_mass_source"]),
        # Marginal IMBH candidates are published without masses.
        "O3_IMBH_marginal": _table([(1263002916.2, "200114_020818")], ["GPS", "commonName"]),
    }).serve(monkeypatch)


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
        np.testing.assert_allclose(1263002916.2 - imbh[0], expected_veto(None)[0], rtol=0, atol=GPS_SLACK)

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

    def test_refresh_rewrites_the_cache_the_vetoes_read(self, event_api):
        # use_cache=False returned the fresh table but left the stale cache, which the
        # noise vetoes went on reading.
        gf.get_event_catalogue()
        event_api.catalogues["GWTC"] = _table(
            [(1126259462.4, "GW150914", 28.6), (1240000000.0, "GW_ADDED", 30.0)],
            ["GPS", "commonName", "chirp_mass_source"],
        )
        assert 1240000000.0 in gf.get_event_catalogue(use_cache=False)["gps"]
        assert 1240000000.0 in gf.get_event_catalogue()["gps"]
        assert _inside([1240000000.0], gf.get_event_veto_segments())[0]
        assert len(event_api.fetched) == 6  # the default calls read the refreshed cache

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
        monkeypatch.setattr(event_module, "get_event_catalogue", lambda use_cache=True: events)

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

    def test_glitches_in_injection_gaps_are_dropped_from_every_class_and_counted(
            self, offline_public_sources, monkeypatch, caplog):
        """
        Gravity Spy entries inside any NO_CBC, NO_BURST, NO_STOCH
        or NO_DETCHAR hardware-injection gap are dropped from every class, and
        counted. The noise vetoes still veto them.
        """
        from gravyflow.src.dataset.features.glitch import glitch_extents

        # Every entry, before any timeline is read (and cached).
        every = gf.get_glitch_catalogue(gf.IFO.H1, gf.ObservingRun.O3, exclude_hardware_injections=False)
        data = public._fixture_intersection("O3a_4KHZ_R1", "H1", ["DATA"])
        inside = [i for i, t in enumerate(every["event_time"]) if _inside([t], data)[0]]
        first = inside[0]
        second = next(i for i in inside if every["label"][i] != every["label"][first])
        entry = public.TIMELINE["O3a_4KHZ_R1"]
        for index, flag in ((first, "H1_NO_CBC_HW_INJ"), (second, "H1_NO_STOCH_HW_INJ")):
            (extent,) = glitch_extents({key: value[[index]] for key, value in every.items()})
            gap = [[extent[0] - 0.01, extent[1] + 0.01]]
            monkeypatch.setitem(entry, flag, gwosc_data.subtract_segments(entry[flag], gap).tolist())

        with caplog.at_level("INFO"):
            kept = gf.get_glitch_catalogue(gf.IFO.H1, gf.ObservingRun.O3)
        dropped = sorted(set(every["event_time"]) - set(kept["event_time"]))
        assert {every["event_time"][first], every["event_time"][second]} <= set(dropped)
        assert f"{len(dropped)} of {len(every['event_time'])} Gravity Spy entries" in caplog.text

        # From every class: asked for by class, each is dropped too.
        for index in (first, second):
            glitch_type = gf.GlitchType(int(every["label"][index]))
            by_class = gf.get_glitch_catalogue(gf.IFO.H1, gf.ObservingRun.O3, glitch_types=[glitch_type])
            assert every["event_time"][index] not in by_class["event_time"]

        # Noise is vetoed around every entry, injection gap or not: the noise path's own vetoes.
        vetoes = _noise_obtainer(gf.ObservingRun.O3).get_glitch_veto_segments(gf.IFO.H1)
        assert np.all(_inside(dropped, vetoes))


# =============================================================================
# SEGMENT YIELD
# =============================================================================

def _block_inside_o3(num_blocks):
    """A GPS time on the absolute 2048 s grid inside O3, with ``num_blocks`` blocks
    after it inside one absolute 8192 s split block."""
    start = gf.ObservingRun.O3.value.start_gps_time
    k = int(np.ceil((start + 10 * 8192) / 2048))
    while True:
        b = 2048.0 * k
        if b % 8192 + num_blocks * 2048 <= 8192:
            return b
        k += 1


class TestSegmentYield:

    def _rows(self, obtainer, per_ifo, ifos, split="train"):
        with patch.object(obtainer, "get_all_segment_times", side_effect=per_ifo), \
             patch.object(obtainer, "remove_unwanted_segments",
                          side_effect=lambda ifo, s: s):
            return obtainer.get_valid_segments(
                ifos=ifos, seed=1, group_name=split,
                segment_order=gf.SegmentOrder.CHRONOLOGICAL,
            )

    def test_absolute_grid_keeps_whole_blocks(self):
        b = _block_inside_o3(2)
        obtainer = _noise_obtainer(gf.ObservingRun.O3, split_spec=ALL_TRAIN)
        rows = self._rows(obtainer, [np.array([[b, b + 4096.0]])], [gf.IFO.L1])
        np.testing.assert_array_equal(
            rows[:, 0, :], [[b, b + 2048.0], [b + 2048.0, b + 4096.0]]
        )

    def test_every_piece_of_a_bin_is_kept(self):
        b = _block_inside_o3(1)
        obtainer = _noise_obtainer(gf.ObservingRun.O3, split_spec=ALL_TRAIN)
        pieces = np.array([[b, b + 500.0], [b + 600.0, b + 2048.0]])
        rows = self._rows(obtainer, [pieces], [gf.IFO.L1])
        assert sorted(map(tuple, rows[:, 0, :])) == sorted(map(tuple, pieces))

    def test_every_piece_is_kept_for_each_ifo(self):
        b = _block_inside_o3(1)
        obtainer = _noise_obtainer(gf.ObservingRun.O3, split_spec=ALL_TRAIN)
        h1 = np.array([[b, b + 2048.0]])
        l1 = np.array([[b, b + 500.0], [b + 600.0, b + 2048.0]])
        rows = self._rows(obtainer, [h1, l1], [gf.IFO.H1, gf.IFO.L1])
        assert rows.shape == (2, 2, 2)
        assert sorted(map(tuple, rows[:, 1, :])) == sorted(map(tuple, l1))
        assert all(tuple(row) == tuple(h1[0]) for row in rows[:, 0, :])
        # The longest pieces are paired.
        longest = rows[np.argmax(rows[:, 1, 1] - rows[:, 1, 0])]
        assert tuple(longest[1]) == (b + 600.0, b + 2048.0)

    def test_the_split_never_cuts_a_bin(self):
        # The split's blocks are absolute and whole bins: at an edge between splits,
        # each side's bin loses only its half of the guard. The 8192 s cut was
        # counted from the first run's start, so it split bins.
        spec = gf.DEFAULT_SPLIT
        k = spec.block_of(gf.ObservingRun.O3.value.start_gps_time) + 10
        while spec.block_split(k) != "train" or spec.block_split(k + 1) != "validate":
            k += 1
        edge = (k + 1) * spec.block_seconds
        half = spec.guard_seconds / 2.0
        stretch = [np.array([[edge - 2048.0, edge + 2048.0]])]
        train = self._rows(_noise_obtainer(gf.ObservingRun.O3), stretch, [gf.IFO.L1])
        validate = self._rows(_noise_obtainer(gf.ObservingRun.O3), stretch, [gf.IFO.L1], "validate")
        assert list(map(tuple, train[:, 0, :])) == [(edge - 2048.0, edge - half)]
        assert list(map(tuple, validate[:, 0, :])) == [(edge + half, edge + 2048.0)]

    def test_merge_bins_with_an_empty_ifo_gives_no_bins(self):
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        assert obtainer.merge_bins([np.array([[0.0, 10.0]]), np.empty((0, 2))], 2048.0) == []


# =============================================================================
# SAMPLING DENSITY
# =============================================================================

from gravyflow.src.dataset.acquisition.noise import couple_pieces  # noqa: E402
from gravyflow.src.dataset.store.read import interiors  # noqa: E402

#: The test window at 64 Hz: 1 s onsource with 0.5 s crops (2 s loaded) and 1.75 s
#: offsource, plus the sampler's 16-sample buffer: 4 s of footprint, so a piece of
#: 4 + x whole seconds has x seconds of window starts. Usable pieces: >= 7.5 s.
RATE, BATCH = 64.0, 16
FOOTPRINT, OFFSOURCE = 4.0, 1.75


class _AtTheRate(gf.GWOSC):
    """GWOSC publishing RATE: reads need no resampling, so a piece's interior is its whole seconds."""

    def rates(self):
        return (RATE,)

    def specification(self):
        return {"rates": [RATE]}


def _density_obtainer(**kwargs):
    return _noise_obtainer(gf.ObservingRun.O3, source=_AtTheRate(), **kwargs)


def _draw(obtainer, rows, num_batches, mode=gf.SamplingMode.RANDOM, offsource_seconds=OFFSOURCE,
          batch=BATCH, seed=7):
    """
    START_GPS_TIME of ``num_batches`` batches drawn from ``rows`` with fake strain, or of
    every batch the generator serves when ``num_batches`` is None.
    """
    def fake_acquire(sample_rate_hertz, valid_segments, ifos, **kwargs):
        # As the loader does: each piece is read as its interior (store.read.interiors).
        for row in valid_segments:
            firsts, counts = interiors(obtainer.source, sample_rate_hertz, row)
            data = [np.zeros(int(count), np.float32) for count in counts]
            yield gf.IFOData(data, sample_rate_hertz, list(firsts / sample_rate_hertz))

    def fake_windows(ifos, sample_rate_hertz, starts, num_onsource_samples, num_offsource_samples):
        # RANDOM reads each window through the store (read_windows).
        shape = (len(starts), len(ifos))
        return np.zeros(shape + (num_onsource_samples,), np.float32), \
            np.zeros(shape + (num_offsource_samples,), np.float32)

    obtainer.valid_segments = np.asarray(rows, dtype=np.float64)
    obtainer.augmentations = []  # saturation exactly: 8 gives 0.5 batches per second of starts
    ifos = [gf.IFO.H1, gf.IFO.L1][:obtainer.valid_segments.shape[1]]
    with patch.object(obtainer, "acquire", side_effect=fake_acquire), \
            patch.object(obtainer, "read_windows", side_effect=fake_windows):
        generator = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=RATE, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
            offsource_duration_seconds=offsource_seconds, num_examples_per_batch=batch, ifos=ifos,
            seed=seed, sampling_mode=mode,
        )
        batches = generator if num_batches is None else (next(generator) for _ in range(num_batches))
        starts = [np.asarray(batch[gf.ReturnVariables.START_GPS_TIME]) for batch in batches]
    return np.concatenate(starts)


def _windows_per_piece(starts, pieces):
    return np.array([np.sum((starts >= a) & (starts < b)) for a, b in pieces])


#: Passes drawn where a density is measured: RANDOM serves each row exactly its batches
#: a pass, and draws each example's row from the rows of eight batches of the pass, so a
#: piece's examples match its seconds of starts exactly in expectation, here drawn to
#: within DENSITY_RTOL.
DENSITY_PASSES, DENSITY_RTOL = 5, 0.15


class TestSamplingDensity:
    """
    Each detector's windows are drawn uniformly over the window starts of its usable
    pieces in each bin: no usable piece is dropped and no short piece is oversampled.
    The test pieces give whole numbers of batches a pass (0.5 batches per second of
    starts at the default saturation of 8).
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
        obtainer = _density_obtainer()
        pieces = np.array([[0.0, FOOTPRINT + 200.0], [300.0, 300.0 + FOOTPRINT + 20.0],
                           [2048.0, 2048.0 + FOOTPRINT + 100.0]])
        starts = _draw(obtainer, pieces[:, None, :], num_batches=DENSITY_PASSES * (100 + 10 + 50))
        density = _windows_per_piece(starts[:, 0], pieces) / np.array([200.0, 20.0, 100.0]) / DENSITY_PASSES
        np.testing.assert_allclose(density, [8.0, 8.0, 8.0], rtol=DENSITY_RTOL)  # saturation / onsource

    def test_no_usable_piece_is_dropped(self):
        # H1 has two usable pieces; L1's second piece is too short. Pairing by rank put
        # H1's second piece with it, and the row was dropped.
        obtainer = _density_obtainer()
        h1 = np.array([[0.0, FOOTPRINT + 100.0], [1000.0, 1000.0 + FOOTPRINT + 50.0]])
        l1 = np.array([[0.0, FOOTPRINT + 200.0], [1500.0, 1505.0]])
        rows = np.stack([h1, l1], axis=1)
        starts = _draw(obtainer, rows, num_batches=DENSITY_PASSES * 75)  # 0.5 x min(150, 200) s a pass
        h1_density = _windows_per_piece(starts[:, 0], h1) / np.array([100.0, 50.0]) / DENSITY_PASSES
        l1_windows = _windows_per_piece(starts[:, 1], l1)
        np.testing.assert_allclose(h1_density, [8.0, 8.0], rtol=DENSITY_RTOL)
        np.testing.assert_array_equal(l1_windows, [DENSITY_PASSES * 75 * BATCH, 0])


    def test_a_pass_with_no_batch_due_does_not_end_the_generator(self):
        # One 40 s piece per detector at saturation 1: about 0.68 batches are due per
        # pass, so a pass can be due none. The generator ended on such a pass. (The
        # pieces are a minute apart, so their windows are never coincident.)
        h1 = np.array([[1238166018.0, 1238166058.0]])
        rows = np.stack([h1, h1 + 60.0], axis=1)
        for seed in range(5):
            obtainer = _density_obtainer(saturation=1.0)
            starts = _draw(obtainer, rows, num_batches=20, offsource_seconds=16.0, batch=32, seed=seed)
            assert starts.shape == (20 * 32, 2)

    def test_the_plan_measures_pieces_as_the_loader_keeps_them(self):
        # The loader keeps whole seconds of a piece; the plan measured its exact length,
        # so short fractional pieces were drawn up to 5.6% too densely.
        obtainer = _density_obtainer()
        plan = obtainer.plan_sampling(np.array([[[0.0, 30.6]]]), 8.0, FOOTPRINT, sample_rate_hertz=RATE)
        np.testing.assert_allclose(plan.start_seconds, [30.0 - FOOTPRINT])

    def test_grid_serves_each_window_once_a_pass_then_ends(self):
        # One H1 piece shares five rows with five L1 pieces. GRID wrapped inside each
        # row's share and never ended; one pass now serves each window at most once
        # across the rows, keeps the detectors two footprints apart, and ends.
        obtainer = _density_obtainer(saturation=0.5)
        h1 = np.array([[0.0, FOOTPRINT + 256.0]])
        l1 = np.array([[k * 100.0, k * 100.0 + FOOTPRINT + 51.0] for k in range(5)])
        rows = np.stack([np.repeat(h1, 5, axis=0), l1], axis=1)
        starts = _draw(obtainer, rows, num_batches=None, mode=gf.SamplingMode.GRID)
        assert len(starts) > 0
        for detector in range(2):
            assert len(np.unique(starts[:, detector])) == len(starts)
        assert np.min(np.diff(np.sort(starts[:, 0]))) >= 2.0 - 1e-9  # no window overlaps another
        assert np.min(np.abs(starts[:, 0] - starts[:, 1])) >= 2 * (2.0 + OFFSOURCE) - 1e-9

    def test_a_row_whose_detectors_cannot_be_separated_is_left_out(self):
        # Both detectors on one 40 s piece: window starts span 36 s, so they can be 36 s
        # apart but never 37 s. Drawing from such a row spun without end.
        piece = np.array([[1238166018.0, 1238166058.0]])
        rows = np.stack([piece, piece], axis=1)
        obtainer = _density_obtainer()
        kept = obtainer.plan_sampling(rows, 8.0, FOOTPRINT, separation_seconds=36.0, sample_rate_hertz=RATE)
        assert len(kept.rows) == 1 and kept.inseparable == 0
        left_out = obtainer.plan_sampling(rows, 8.0, FOOTPRINT, separation_seconds=37.0, sample_rate_hertz=RATE)
        assert len(left_out.rows) == 0 and left_out.inseparable == 1

    def test_separable_tries_every_order(self):
        from gravyflow.src.dataset.acquisition.base import separable
        # Earliest-deadline order fails here (L1 at 15, H1 at 33 > 25); H1 first fits.
        assert separable([(15.0, 20.0), (0.0, 25.0)], 18.0)
        assert separable([(0.0, 6.0)] * 3, 3.0)
        assert not separable([(0.0, 5.0)] * 3, 3.0)


class _RecordingFetch:
    """Fake fetch_open_data: records the requested span; returns a ramp."""

    def __init__(self):
        self.calls = []

    def __call__(self, ifo, start, end, sample_rate=4096, **kwargs):
        from gwpy.timeseries import TimeSeries
        self.calls.append((start, end))
        value = np.arange(int(round((end - start) * sample_rate)), dtype=np.float32)
        return TimeSeries(value, t0=start, sample_rate=sample_rate, channel="L1:FAKE-RAMP")


class TestWholeSegments:

    TILE = 1238171648.0  # a GWOSC file's first sample: tiles are whole 4096 s files

    def _get(self, start, end, rate):
        fetch = _RecordingFetch()
        obtainer = _noise_obtainer(gf.ObservingRun.O3)
        with patch("gwosc.locate.get_urls", side_effect=GwoscLocator()), \
                patch("gwpy.timeseries.TimeSeries.fetch_open_data", side_effect=fetch):
            [row] = obtainer.acquire(rate, np.array([[[start, end]]]), [gf.IFO.L1])
        return row.data[0], fetch.calls

    def test_the_whole_file_is_fetched_and_the_piece_read_inside_it(self):
        segment, calls = self._get(self.TILE, self.TILE + 2048.0, 2048.0)
        assert calls == [(self.TILE, self.TILE + 4096.0)]
        # Read at 2048 Hz from 4096 Hz: the whole seconds inside the resampler's margin.
        assert segment.shape[0] == 2047 * 2048

    @pytest.mark.parametrize("duration, rate", [(1500.0, 2048.0), (700.0, 4096.0), (2048.0, 4096.0)])
    def test_no_power_of_two_truncation(self, duration, rate):
        segment, _ = self._get(self.TILE, self.TILE + duration, rate)
        # At the native rate nothing is resampled, so the piece is read whole.
        assert segment.shape[0] == int((duration - (rate != 4096.0)) * rate)

    def test_segment_starts_at_its_first_sample(self):
        segment, _ = self._get(self.TILE, self.TILE + 64.5, 4096.0)
        values = np.asarray(segment)
        assert values.shape[0] == 64 * 4096
        assert abs(values[0]) < 1.0 and abs(values[100] - 100.0) < 1.0
