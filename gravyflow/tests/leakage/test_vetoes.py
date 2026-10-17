"""
Forbidden data never reaches noise: glitches, events and hardware injections.

Each test checks two layers. The first is exhaustive: the library's served-time set
for every split (the segments the sampler draws from, after vetoes) avoids the
forbidden time. The second ties it to the data: decoded windows from RANDOM and GRID
batches of every split lie inside that served time and avoid the forbidden time
themselves. Sampled windows alone would rarely land on a short veto; the first layer
has no miss probability.
"""

from unittest import mock

import numpy as np
import pytest

import gravyflow as gf

from . import audit
from . import fixtures as fx
from .fakes import GlitchRecord, HARDWARE_INJECTION_FLAGS

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
    fixture, found = _event_world()
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


# =============================================================================
# EVENTS: THEIR FOOTPRINT AND THE GUARD, AND EVALUATION ONLY
# =============================================================================

#: Each fixture event's footprint before its merger, by its in-band seconds, blessed
#: by hand: max(48.2, tau + 1) s, tau the library's time above 20 Hz (1.25 times the
#: Newtonian chirp time). After the merger it is 48.2 s.
FOOTPRINT_BEFORE = {2.0: 48.2, 60.0: 76.0, 165.0: 207.25}
FOOTPRINT_AFTER = 48.2


def _footprints(found, guard=0.0):
    return [(e.gps - FOOTPRINT_BEFORE[e.in_band_seconds] - guard, e.gps + FOOTPRINT_AFTER + guard)
            for e in found]


def _event_world():
    """
    Events and their companion glitches around each edge (``fx.events``), noise beyond
    them (each event's footprint and guard reach 985 s from the edge), and Blips there,
    1150 s and 1250 s from each edge in each detector.
    """
    fixture, edges = fx.two_splits_per_run(("O3",), half_width=1400.0, origin="O3")
    found, companions = fx.events(edges)
    far = [GlitchRecord(ifo, edge + side * distance, edge + side * distance - 0.5, 1.0, "Blip", 0.95)
           for edge in edges for side in (-1.0, 1.0) for distance in (1150.0, 1250.0) for ifo in IFOS]
    return fixture.with_catalogues(glitches=companions + tuple(far), events=found), found


def _glitch_footprints(fixture, split):
    """{ifo: footprints} of the glitches a transient obtainer serves for ``split``."""
    obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
    obtainer.get_valid_segments([gf.IFO[name] for name in IFOS], 7, split)
    out = {name: [] for name in IFOS}
    for record in obtainer.transient_segments:
        for ifo in record.seen_in:
            out[ifo.name].append((record.transient_gps_time - 48.2, record.transient_gps_time + 48.2))
    return {name: np.array(rows).reshape(-1, 2) for name, rows in out.items()}


def test_events_vetoed_with_their_guard(world, tmp_path):
    """
    No noise window of any split or mode comes within the guard of an event's footprint;
    no glitch served outside test comes within it, and none served in test meets it.
    """
    fixture, found = _event_world()
    _check_noise_avoids(world, tmp_path, fixture, _footprints(found, fx.GUARD_SECONDS))
    with world(fixture):
        glitches = {split: _glitch_footprints(fixture, split) for split in SPLITS}
    assert sum(len(rows) for served in glitches.values() for rows in served.values()) > 0
    for split, served in glitches.items():
        guard = 0.0 if split == "test" else fx.GUARD_SECONDS
        audit.assert_avoids(served, _footprints(found, guard))


def test_event_guard_check_sees_the_in_band_veto(world, monkeypatch):
    """The planted leak: noise vetoed over the signal in band alone is caught."""
    from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
    from gravyflow.src.dataset.features.event import get_event_veto_segments

    monkeypatch.setattr(BaseDataObtainer, "get_event_veto_segments", lambda self: get_event_veto_segments())
    fixture, found = _event_world()
    with world(fixture):
        served = [fx.served_segments(fixture, IFOS, split) for split in SPLITS]
    with pytest.raises(AssertionError):
        for segments in served:
            audit.assert_avoids(segments, _footprints(found, fx.GUARD_SECONDS))


def _event_obtainer(fixture, **overrides):
    return fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.EVENTS], **overrides)


def test_events_evaluation_only(world):
    """
    Real events are test data: an events obtainer refuses train and validate (named
    events too); test serves every confident event once and no marginal one; and the
    validator scores exactly those, and none from another split.
    """
    fixture, found = _event_world()
    confident = sorted(e.name for e in found if e.confident)
    assert len(confident) < len(found)
    with world(fixture):
        for split in ("train", "validate"):
            for overrides in ({}, {"event_names": confident}):
                with pytest.raises(ValueError, match="evaluation data"):
                    _event_obtainer(fixture, **overrides).get_valid_segments(
                        [gf.IFO.H1, gf.IFO.L1], 7, split)
        obtainer = _event_obtainer(fixture)
        obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, "test")
        served = sorted(record.name for record in obtainer.transient_segments)
        assert served == confident

        scored = {split: _validator_event_names(split) for split in ("test", "validate")}
    assert scored == {"test": confident, "validate": []}


class _Stop(Exception):
    pass


def _validator_event_names(split):
    """The events the validator's real-event stage asks for, in split ``split``."""
    from gravyflow.src.validate.bank import ValidationBank

    bank = object.__new__(ValidationBank)
    bank.far_scores = np.zeros(4)
    bank.real_events = None
    bank.config = gf.ValidationConfig(group=split)
    bank.dataset_args = {"ifos": [gf.IFO.H1, gf.IFO.L1], "onsource_duration_seconds": 1.0}
    asked = {}

    class Recording(gf.TransientDataObtainer):
        def __init__(self, *args, event_names=None, **kwargs):
            asked["names"] = sorted(event_names)
            raise _Stop

    with mock.patch.object(gf, "TransientDataObtainer", Recording):
        try:
            bank.generate_real_events([gf.ObservingRun.O3])
        except _Stop:
            pass
    return asked.get("names", sorted(e["name"] for e in bank.real_events or []))
