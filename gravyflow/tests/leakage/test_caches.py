"""
Catalogue caches serve only their own parameters: a warm cache answers a
query exactly as a cold one would, whatever was asked of it before.
"""

import numpy as np
import pytest
from gwpy.segments import Segment, SegmentList

import gravyflow as gf
from gravyflow.src.dataset.acquisition import gwosc_data
from gravyflow.src.dataset.features import gravity_spy
from gravyflow.tests.test_public_sources import _fixture_rows, offline_public_sources  # noqa: F401

ERAS = ("O1", "O2", "O3a", "O3b")
RUNS = {"O1": gf.ObservingRun.O1, "O2": gf.ObservingRun.O2, "O3": gf.ObservingRun.O3}


def _expected_times(ifo, era, floor):
    bounds = gwosc_data.GWOSC_RUNS[era]
    times = [float(r["event_time"]) for r in _fixture_rows(ifo, era)
             if float(r["ml_confidence"]) >= floor
             and bounds["gps_start"] <= float(r["event_time"]) <= bounds["gps_end"]]
    return np.sort(np.asarray(times))


def _segments(array):
    return SegmentList(Segment(float(a), float(b)) for a, b in np.asarray(array)).coalesce()


def _livetime(segments):
    return float(sum(abs(s) for s in segments))


def test_catalogue_caches(offline_public_sources, tmp_path):
    """
    A Gravity Spy query at a confidence floor returns exactly the catalogue's
    rows at or above it, before and after a query at another floor. The GWOSC
    timeline cache, asked for each detector and flag set in turn, gives each
    the answer a cold cache gives it alone, and a superset of flags gives a
    subset of time.
    """
    served = {}
    for ifo in ("H1", "L1"):
        for era in ERAS:
            bounds = gwosc_data.GWOSC_RUNS[era]
            for floor in (0.9, 0.0, 0.9):
                rows = gravity_spy.query(ifo, bounds["gps_start"], bounds["gps_end"], min_ml_confidence=floor)
                assert np.all(rows["ml_confidence"] >= floor)
                np.testing.assert_array_equal(np.sort(rows["event_time"]), _expected_times(ifo, era, floor),
                                              err_msg=f"{ifo} {era} at floor {floor}")
                served[(ifo, era, floor)] = rows["event_time"].size
    # The floor must matter somewhere, or the check is empty.
    assert any(served[(ifo, era, 0.0)] > served[(ifo, era, 0.9)] for ifo in ("H1", "L1") for era in ERAS)

    def segments(run, ifo, flags):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
            observing_runs=RUNS[run], cache_segments=False, gwosc_flags=flags,
        )
        return _segments(obtainer.get_all_segment_times(getattr(gf.IFO, ifo)))

    def timeline_requests():
        return sum("/timeline/" in url for url in offline_public_sources.requests)

    def cold(run, ifo, flags):
        """The answer of a cache that has seen nothing else: a fresh root, filled by this query alone."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path / "cold" / f"{run}-{ifo}-{len(flags)}"))
            before = timeline_requests()
            answer = segments(run, ifo, flags)
            assert timeline_requests() > before, f"{run} {ifo} {flags}: the cold query fetched nothing"
            return answer

    # One warm cache answers every query in turn: each detector and flag set,
    # in both orders, then each again. Every answer must be the cold one.
    stricter, detectors_differ = [], []
    for run in RUNS:
        cold_answers = {}
        for ifo, order in (("H1", (gf.GWOSC_BURST_FLAGS, gf.GWOSC_DATA_FLAGS)),
                           ("L1", (gf.GWOSC_DATA_FLAGS, gf.GWOSC_BURST_FLAGS))):
            got = {flags: segments(run, ifo, flags) for flags in order}
            for flags in order:
                cold_answers[(ifo, flags)] = cold(run, ifo, flags)
                assert got[flags] == cold_answers[(ifo, flags)], \
                    f"{run} {ifo} {flags}: the warm cache answers otherwise than a cold one"
            data, burst = got[gf.GWOSC_DATA_FLAGS], got[gf.GWOSC_BURST_FLAGS]
            assert _livetime(burst) > 0, f"{run} {ifo}: no time with the burst flags"
            outside = burst - data
            assert _livetime(outside) == 0, f"{run} {ifo}: more flags added {outside}"
            stricter.append(_livetime(burst) < _livetime(data))
            # Asked again, each answer comes from the warm cache, unchanged.
            before = timeline_requests()
            for flags in order:
                assert segments(run, ifo, flags) == got[flags], f"{run} {ifo} {flags} changed when warm"
            assert timeline_requests() == before, f"{run} {ifo}: the repeat was not served from the cache"
        detectors_differ.append(any(cold_answers[("H1", f)] != cold_answers[("L1", f)] for f in order))
    # The flags and the detector must matter somewhere, or the checks are empty.
    assert any(stricter), "no run where the extra flags remove time; the property is untested"
    assert any(detectors_differ), "no run where the detectors' time differs; the cache key is untested"
