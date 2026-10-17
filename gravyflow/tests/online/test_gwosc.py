"""
GWOSC, live: its run bounds, timeline, event list and strain still agree with
gravyflow's committed tables and with the fixtures the offline suites serve
(``tests/fixtures/public_sources``).
"""

import csv
import json
from pathlib import Path

import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.acquisition.gwosc_data import (
    GWOSC_RUNS, fetch_gwosc_runs, fetch_timeline_segments,
)
from gravyflow.src.dataset.features.event import get_event_catalogue

FIXTURES = Path(__file__).parents[1] / "fixtures" / "public_sources"


def test_run_bounds_match_the_live_run_api():
    live = fetch_gwosc_runs()
    assert {name: live[name] for name in GWOSC_RUNS} == {
        name: (run["gps_start"], run["gps_end"]) for name, run in GWOSC_RUNS.items()}


def test_the_timeline_fixture_is_the_live_timeline():
    """Every flag of every window the offline timeline fake serves, as GWOSC serves it now."""
    timeline = json.loads((FIXTURES / "gwosc_timeline.json").read_text())
    for dataset, entry in timeline.items():
        start, end = entry["window"]
        for flag, segments in entry.items():
            if flag in ("run", "window"):
                continue
            live = fetch_timeline_segments(dataset, flag, start, end, use_cache=False)
            np.testing.assert_array_equal(
                live, np.reshape(np.asarray(segments, dtype=float), (-1, 2)),
                err_msg=f"{dataset} {flag}")


def test_every_event_in_the_snapshot_is_still_listed():
    """GWOSC may add events; the offline snapshot's rows are never dropped or moved."""
    with open(FIXTURES / "gwosc_events.csv", newline="") as handle:
        snapshot = {(row["catalog"], row["name"], round(float(row["gps"]), 3))
                    for row in csv.DictReader(handle)}
    table = get_event_catalogue(use_cache=False)
    live = {(str(catalog), str(name), round(float(gps), 3))
            for catalog, name, gps in zip(table["catalog"], table["name"], table["gps"])}
    assert snapshot - live == set()


def test_the_default_configuration_builds_from_gwosc():
    """Noise segments for H1 and L1 over every default run, events and glitches vetoed."""
    obtainer = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE])
    segments = obtainer.get_valid_segments(ifos=[gf.IFO.H1, gf.IFO.L1], seed=1, group_name="train")
    assert segments.ndim == 3 and segments.shape[1:] == (2, 2)
    starts = segments[:, :, 0].ravel()
    for run in (gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3):
        assert np.any((starts >= run.value.start_gps_time) & (starts < run.value.end_gps_time)), run.name


def test_the_source_fetches_strain_at_the_rate_and_time_asked():
    start = 1238166100.0  # inside O3a
    strain = gf.GWOSC().fetch(gf.IFO.L1, 4096.0, (start, start + 10.0))
    assert (strain.source, strain.sample_rate_hertz, strain.t0) == ("gwosc-O3a_4KHZ_R1", 4096.0, start)
    assert len(strain.data) == 10 * 4096
    assert np.all(np.isfinite(strain.data)) and np.any(strain.data != 0)
