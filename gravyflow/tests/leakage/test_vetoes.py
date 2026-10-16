"""
Forbidden data never reaches noise: glitches, events and hardware injections.

Each test checks two layers. The first is exhaustive: the library's served-time set
for every split (the segments the sampler draws from, after vetoes) avoids the
forbidden time. The second ties it to the data: decoded windows from RANDOM and GRID
batches of every split lie inside that served time and avoid the forbidden time
themselves. Sampled windows alone would rarely land on a short veto; the first layer
has no miss probability.
"""

import numpy as np
import pytest

import gravyflow as gf

from . import audit
from . import fixtures as fx
from .fakes import HARDWARE_INJECTION_FLAGS

IFOS = ["H1", "L1"]
SPLITS = ("train", "validate", "test")
MODES = [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID]


def _check_noise_avoids(world, tmp_path, fixture, forbidden):
    """Both layers, for every split and both sampling modes."""
    with world(fixture) as harness:
        served = {split: fx.served_segments(fixture, IFOS, split) for split in SPLITS}
        sets = {
            f"{split} {mode.name}": (split, fx.noise_batches(fixture, tmp_path / f"{split}-{mode.name}",
                                                             split, mode=mode))
            for split in SPLITS for mode in MODES
        }
        records = audit.collect(harness, sets, 2, IFOS, fx.window(), onsource_cropped=False)
    for split in SPLITS:
        audit.assert_avoids(served[split], forbidden)
    for record in records.values():
        audit.assert_inside([record], served[record.split])
        audit.assert_avoids([record], forbidden)


def test_glitches_vetoed_from_noise(world, tmp_path):
    """
    No noise window of any split or mode touches a Gravity Spy entry's whole extent
    plus its class padding, for entries of every class (No_Glitch included), every
    confidence and every configured run, from deliberately unsorted catalogues.
    """
    fixture, edges = fx.two_splits_per_run(("O1", "O2", "O3"))
    rows = fx.glitches(edges)
    fixture = fixture.with_catalogues(glitches=rows)
    assert {r.label for r in rows} >= {"No_Glitch", "Extremely_Loud"}
    assert min(r.confidence for r in rows) < 0.5
    _check_noise_avoids(world, tmp_path, fixture, fx.glitch_forbidden(rows))


def test_veto_subtraction_removes_unsorted_overlapping_vetoes():
    """Unit check: vetoes given unsorted and overlapping are all removed."""
    obtainer = fx.ifo_data_obtainer(fx.straddle("train", "test", half_width=64.0))
    segments = np.array([[0.0, 1000.0]])
    vetoes = np.array([[500.0, 520.0], [100.0, 110.0], [105.0, 130.0], [900.0, 905.0], [100.0, 101.0]])
    kept = obtainer.veto_time_segments(segments, vetoes)
    audit.assert_avoids({"H1": kept}, [tuple(v) for v in vetoes])
    assert np.isclose(np.sum(kept[:, 1] - kept[:, 0]), 1000.0 - 30.0 - 20.0 - 5.0)


def test_events_vetoed(world, tmp_path):
    """
    No noise window of any split or mode touches a catalogued event's in-band signal
    (from 20 Hz to merger), for confident and marginal events of 2 s, 60 s and 165 s;
    and no glitch the catalogue serves overlaps one.
    """
    fixture, edges = fx.two_splits_per_run(("O3",), half_width=900.0, origin="O3")
    found, companions = fx.events(edges)
    fixture = fixture.with_catalogues(glitches=companions, events=found)
    extents = [e.extent for e in found]
    assert {e.in_band_seconds for e in found} == {2.0, 60.0, 165.0}
    _check_noise_avoids(world, tmp_path, fixture, extents)
    with world(fixture):
        served_glitches = {
            name: gf.get_glitch_catalogue(gf.IFO[name], observing_run=gf.ObservingRun.O3)
            for name in IFOS
        }
    assert sum(len(c["event_time"]) for c in served_glitches.values()) > 0
    for name, catalogue in served_glitches.items():
        audit.assert_avoids(
            {name: np.stack([catalogue["start_time"], catalogue["start_time"] + catalogue["duration"]], 1)},
            extents,
        )


def test_no_hardware_injections(world, tmp_path):
    """
    The default flags exclude all four hardware-injection kinds, and no noise window
    of any split or mode touches a CBC, burst, stochastic (coherent in H1 and L1) or
    detchar injection gap.
    """
    from gravyflow.src.dataset.acquisition.gwosc_data import DEFAULT_GWOSC_FLAGS

    assert set(HARDWARE_INJECTION_FLAGS) <= set(DEFAULT_GWOSC_FLAGS)
    fixture = fx.hw_injection_gaps()
    forbidden = {ifo: [] for ifo in IFOS}
    for (ifo, flag), gaps in fixture.flag_gaps.items():
        forbidden[ifo] += [tuple(g) for g in gaps]
    _check_noise_avoids(world, tmp_path, fixture, forbidden)
