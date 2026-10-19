"""
gravyflow's standard datasets (``gf.standard_dataset``): how their segments and events
are computed (on synthetic inputs), the pinned files they ship and the refusal of a
changed one, their injections, and what their fingerprints cover.

The segment and event procedures are checked against intervals worked out here by
hand from the split's blocks; the real lists' place in the test split is the leakage
suite's (``leakage/test_standard_datasets.py``).
"""

import dataclasses
import shutil

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.features.event import noise_event_vetoes
from gravyflow.src.dataset.standard import build, presets
from gravyflow.src.dataset.standard.population import (
    BBHPopulation, InjectionSchedule, injection_table, table_sha256,
)
from gravyflow.src.rng import SeedTree

SPLIT = gf.DEFAULT_SPLIT
B, HALF = SPLIT.block_seconds, SPLIT.guard_seconds / 2


def _isolated_test_block(near: int = 167_000) -> int:
    """A test block whose neighbours are both of other splits, at or after block ``near``."""
    k = near
    while not (SPLIT.block_split(k) == "test" and SPLIT.block_split(k - 1) != "test"
               and SPLIT.block_split(k + 1) != "test"):
        k += 1
    return k


K = _isolated_test_block()
T0, T1 = K * B, (K + 1) * B


# ----------------------------------------------------------------- segments


def test_segments_are_the_test_time_every_detector_holds_less_the_vetoes():
    readable = {
        "H1": np.array([[T0 - 1000.0, T1 + 1000.0]]),
        # L1 is off for 100 s inside the block.
        "L1": np.array([[T0 - 1000.0, T0 + 2000.0], [T0 + 2100.0, T1 + 1000.0]]),
    }
    vetoes = np.array([[T0 + 5000.25, T0 + 5010.75]])
    segments = build.split_segments(readable, vetoes, SPLIT, "test", 0, gf.GWOSC(), 2048.0)
    # A read at 2048 Hz reaches 132 GWOSC samples (32.2 ms) past each end, so every edge
    # moves inward to the next whole second that keeps it inside: a whole-second edge by
    # one second, the veto's edges at 5000.25 and 5010.75 to 5000 and 5011.
    expected = [[T0 + HALF + 1, T0 + 1999], [T0 + 2101, T0 + 5000], [T0 + 5011, T1 - HALF - 1]]
    np.testing.assert_array_equal(segments, np.array(expected, dtype=np.int64))
    assert segments.dtype == np.int64


def test_segments_shorter_than_the_minimum_are_dropped():
    readable = {"H1": np.array([[T0 + 100.0, T0 + 150.0], [T0 + 200.0, T0 + 400.0]])}
    segments = build.split_segments(readable, np.empty((0, 2)), SPLIT, "test", 64, gf.GWOSC(), 2048.0)
    np.testing.assert_array_equal(segments, [[T0 + 201, T0 + 399]])


def test_no_time_of_another_split_is_a_segment():
    readable = {"H1": np.array([[T0 - B, T0]]), "L1": np.array([[T0 - B, T0]])}
    assert build.split_segments(readable, np.empty((0, 2)), SPLIT, "test", 0, gf.GWOSC(), 2048.0).shape == (0, 2)


def test_a_read_of_a_segment_stays_inside_it():
    from gravyflow.src.dataset.store.read import geometry, read_span

    readable = {"H1": np.array([[T0 + 100.5, T0 + 900.25]])}
    (start, end), = build.split_segments(readable, np.empty((0, 2)), SPLIT, "test", 0, gf.GWOSC(), 2048.0)
    g = geometry(gf.GWOSC(), 2048.0)
    first, stop = read_span(g, start * 2048, (end - start) * 2048)
    assert first / g.native >= T0 + 100.5 and stop / g.native <= T0 + 900.25


def test_the_noise_obtainers_event_vetoes_are_the_presets():
    catalogue = {"gps": np.array([T0 + 3000.0]), "chirp_mass_source": np.array([20.0])}
    vetoes = noise_event_vetoes(SPLIT.guard_seconds, catalogue)
    # The footprint widened by the guard reaches past the in-band signal on both sides.
    assert vetoes.min() < T0 + 3000.0 - 48.2 - SPLIT.guard_seconds + 1e-6
    assert vetoes.max() >= T0 + 3000.0 + 48.2 + SPLIT.guard_seconds - 1e-6


# ------------------------------------------------------------------- events


def _catalogue(rows):
    keys = ("gps", "chirp_mass_source", "name", "catalog", "p_astro")
    return {key: np.array([row[i] for row in rows]) for i, key in enumerate(keys)}


def test_events_are_those_the_test_split_holds_one_row_per_name():
    catalogue = _catalogue([
        (T0 + 3000.0, 25.0, "GW_A", "GWTC-2.1-marginal", 0.4),
        (T0 + 3000.0, 25.0, "GW_A", "GWTC-3-confident", 0.9),  # the confident entry is kept
        (T0 + 6000.0, 30.0, "GW_B", "IAS-O3a", 0.6),
        (T0 + 20.0, 30.0, "GW_EDGE", "GWTC", 0.99),  # its footprint meets the guard
        (T0 - 3000.0, 30.0, "GW_OTHER", "GWTC", 0.99),  # in another split
    ])
    readable = {"H1": np.array([[T0, T1]]), "L1": np.array([[T0 + 5000.0, T1]])}
    rows = build.split_events(catalogue, readable, SPLIT, "test", (T0 - B, T1 + B))
    assert [row["name"] for row in rows] == ["GW_A", "GW_B"]
    assert rows[0]["catalog"] == "GWTC-3-confident" and rows[0]["confident"]
    assert rows[0]["detectors"] == "H1" and rows[1]["detectors"] == "H1+L1"
    assert not rows[1]["confident"]


def test_the_event_file_has_its_columns_and_spells_a_missing_value_nan():
    rows = [{"name": "GW_A", "gps": 1.5, "catalog": "GWTC", "confident": True, "chirp_mass_source": float("nan"),
             "p_astro": 0.9, "detectors": "H1+L1"}]
    data = build.events_csv(rows)
    assert data.decode().splitlines()[0] == ",".join(build.EVENT_COLUMNS)
    assert b"nan" in data


# -------------------------------------------------------------- injections


POPULATION = BBHPopulation(mass_msun=(5.0, 100.0), spin_z=(-0.99, 0.99), reach_mpc=1000.0,
                           approximant="IMRPhenomXAS", lower_frequency_hertz=20.0)


def test_the_schedule_keeps_mergers_apart_and_inside_the_segments():
    segments = np.array([[0, 1000], [2000, 2100], [3000, 3200]], dtype=np.int64)
    schedule = InjectionSchedule(count=40, spacing_seconds=64.0, padding_seconds=32.0)
    # 1000 s holds 14 slots, 100 s none, 200 s two: 16 slots, so two full passes and 8 more.
    assert len(schedule.slots(segments)) == 16
    table = injection_table(POPULATION, schedule, SeedTree(3), segments)
    assert np.bincount(table["pass"]).tolist() == [16, 16, 8]
    for p in range(3):
        tc = table["tc"][table["pass"] == p]
        assert np.all(np.diff(tc) >= 32.0)
    inside = np.zeros(len(table), bool)
    for start, end in segments:
        inside |= (table["tc"] >= start + 32.0) & (table["tc"] <= end - 32.0)
    assert inside.all()


def test_the_population_is_in_its_ranges():
    table = injection_table(POPULATION, InjectionSchedule(2000, 64.0, 32.0), SeedTree(5),
                            np.array([[0, 64 * 3000]], dtype=np.int64))
    assert np.all(table["mass1"] >= table["mass2"])
    assert table["mass2"].min() >= 5.0 and table["mass1"].max() <= 100.0
    assert np.abs(table[["spin1z", "spin2z"]].tolist()).max() <= 0.99
    assert 0.0 <= table["distance"].min() and table["distance"].max() <= 1000.0
    # Uniform in volume: the median distance is R / 2^(1/3).
    assert abs(np.median(table["distance"]) / 1000.0 - 2 ** (-1 / 3)) < 0.03


def test_a_schedule_with_nowhere_to_inject_is_refused():
    with pytest.raises(ValueError, match="nowhere to inject"):
        InjectionSchedule(5, 64.0, 32.0).times(SeedTree(1), np.array([[0, 100]]))


def test_the_table_is_a_function_of_the_seed():
    segments = np.array([[0, 64 * 100]], dtype=np.int64)
    schedule = InjectionSchedule(150, 64.0, 32.0)
    one = injection_table(POPULATION, schedule, SeedTree(9), segments)
    assert table_sha256(one) == table_sha256(injection_table(POPULATION, schedule, SeedTree(9), segments))
    assert table_sha256(one) != table_sha256(injection_table(POPULATION, schedule, SeedTree(10), segments))


# ------------------------------------------------------------------ presets


@pytest.mark.parametrize("name", sorted(gf.STANDARD_DATASETS))
def test_each_preset_ships_its_pinned_files_and_makes_its_pinned_injections(name):
    preset = gf.standard_dataset(name)
    assert preset.segments.shape[1] == 2 and len(preset.segments) > 0
    assert np.all(np.diff(preset.segments, axis=1) >= preset.minimum_segment_seconds)
    assert np.all(preset.segments[1:, 0] > preset.segments[:-1, 1])
    assert len(preset.injections) == preset.schedule.count == 100_000
    assert preset.sample_rate_hertz == 2048.0 and preset.detectors == ("H1", "L1")
    assert preset.population.reach_mpc > 0 and preset.events


def test_the_twin_has_the_same_segments_and_injections_and_another_fingerprint():
    real = gf.standard_dataset("O4 standard")
    twin = gf.standard_dataset("O4 standard", gaussian_twin=True)
    assert twin.twin == gf.GaussianTwin() and real.twin is None
    np.testing.assert_array_equal(real.segments, twin.segments)
    assert table_sha256(real.injections) == table_sha256(twin.injections)
    assert real.fingerprint() != twin.fingerprint()
    assert twin.label == "O4 standard (Gaussian twin)"


def test_an_unknown_preset_is_refused_by_name():
    with pytest.raises(KeyError, match="O4 standard"):
        gf.standard_dataset("O4")


@pytest.mark.parametrize("change", [
    dict(seed=1), dict(minimum_segment_seconds=128.0), dict(sample_rate_hertz=4096.0),
    dict(flags=gf.GWOSC_DATA_FLAGS), dict(injections_sha256="0" * 64),
    dict(split=gf.SplitSpec(guard_seconds=256.0)), dict(gwosc_runs=("O4a",)),
    dict(twin=gf.GaussianTwin(psd_seconds=8.0)),
])
def test_the_fingerprint_covers_what_fixes_the_data(change):
    preset = gf.standard_dataset("O4 standard")
    assert dataclasses.replace(preset, **change).fingerprint() != preset.fingerprint()


@pytest.mark.parametrize("file", presets.PINNED_FILES)
def test_the_fingerprint_covers_each_pinned_file(file):
    preset = gf.standard_dataset("O4 standard")
    changed = dataclasses.replace(preset, pins={**preset.pins, file: "0" * 64})
    assert changed.fingerprint() != preset.fingerprint()


@pytest.mark.parametrize("file", presets.PINNED_FILES)
def test_a_changed_pinned_file_is_refused_by_name(file, tmp_path, monkeypatch):
    from gravyflow.src import resources

    preset = gf.standard_dataset("O3 open-weights")
    copy = tmp_path / "res"
    shutil.copytree(resources.RESOURCES / "standard", copy / "standard")
    with open(copy / "standard" / preset.slug / file, "ab") as handle:
        handle.write(b"\n")
    monkeypatch.setattr(resources, "RESOURCES", copy)
    fresh = dataclasses.replace(preset)  # no cached reads
    with pytest.raises(gf.PinnedFileChanged, match=file):
        {"segments.csv": lambda: fresh.segments, "events.csv": lambda: fresh.events,
         "population.json": lambda: fresh.population}[file]()


def test_injections_drawn_otherwise_are_refused():
    preset = dataclasses.replace(gf.standard_dataset("O3 open-weights"), injections_sha256="0" * 64)
    with pytest.raises(gf.PinnedFileChanged, match="injection table"):
        preset.injections
