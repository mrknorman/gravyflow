"""
A transient record's split is its footprint's (L14): ``spec.assign`` of the time the
library may read for it, whatever the seed, the runs, the detectors, the window or
the route. Records whose footprint is in no split are left out and counted.
"""

import hashlib
from unittest import mock

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.transient_index import TransientIndex, transient_footprint
from gravyflow.src.utils.gps import gps_to_key
from gravyflow.tests.conftest import catalogued_events, no_glitch_vetoes, usable_over_its_runs

SPLITS = ("train", "validate", "test")
O3_START = gf.ObservingRun.O3.value.start_gps_time


def records(count=2000):
    """L1 glitches over O3, many of them near a block edge."""
    rng = np.random.default_rng(11)
    blocks = gf.DEFAULT_SPLIT.block_of(O3_START) + 2 + rng.integers(0, 3000, count)
    times = blocks * 8192.0 + rng.uniform(-150.0, 150.0, count)
    return [
        TransientSegment(gps_key=gps_to_key(float(t)), transient_gps_time=float(t),
                         start_gps_time=float(t) - 48.2, end_gps_time=float(t) + 48.2,
                         label=gf.DataLabel.GLITCHES, kind=GlitchType.BLIP,
                         observing_run=gf.ObservingRun.O3, seen_in=[gf.IFO.L1])
        for t in np.unique(times)
    ]


@pytest.fixture
def catalogue():
    rows = records()
    with mock.patch("gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
                    return_value=rows), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", usable_over_its_runs), \
         catalogued_events(), no_glitch_vetoes():
        yield rows


def expected(rows, spec):
    """{split or EXCLUDED: keys} from the spec alone."""
    out = {}
    for record in rows:
        out.setdefault(spec.assign(*transient_footprint(record)), set()).add(record.gps_key)
    return out


def assert_membership_is_assign(served, rows, spec, where):
    """Each split served exactly the records whose footprint ``spec.assign`` puts in it."""
    truth = expected(rows, spec)
    for split in SPLITS:
        wrong = served[split] ^ truth.get(split, set())
        assert not wrong, f"{where}: {len(wrong)} records in or out of {split} against spec.assign"


class Stop(Exception):
    pass


def served_by_dataset(obtainer, split, onsource_seconds, seed):
    """The records a dataset serves for ``split``, stopped where it would read its cache."""
    served = {}

    def stop(self, *args, **kwargs):
        served["keys"] = {record.gps_key for record in self.transient_segments}
        raise Stop

    with mock.patch.object(gf.TransientDataObtainer, "_initialize_transient_cache", stop), \
            pytest.raises(Exception):
        gf.Dataset(noise_obtainer=obtainer, group=split, seed=seed, sample_rate_hertz=256.0,
                   onsource_duration_seconds=onsource_seconds, offsource_duration_seconds=16.0,
                   input_variables=[gf.ReturnVariables.ONSOURCE], steps_per_epoch=1)[0]
    return served["keys"]


CONFIGURATIONS = {
    "seed 0, [O3], L1, 1 s": dict(seed=0, runs=[gf.ObservingRun.O3], ifos=[gf.IFO.L1], onsource=1.0),
    "seed 7, [O1, O2, O3], H1 L1, 8 s": dict(
        seed=7, runs=[gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3],
        ifos=[gf.IFO.H1, gf.IFO.L1], onsource=8.0),
}
SPECS = {"default": gf.SplitSpec(),
         "0.6/0.2/0.2, another salt": gf.SplitSpec(
             fractions={"train": 0.6, "validate": 0.2, "test": 0.2}, version="fold-2")}


@pytest.mark.parametrize("spec_name", list(SPECS))
@pytest.mark.parametrize("configuration", list(CONFIGURATIONS))
def test_record_split_is_spec_assign(catalogue, configuration, spec_name):
    settings, spec = CONFIGURATIONS[configuration], SPECS[spec_name]

    def obtainer():
        return gf.TransientDataObtainer(data_quality=gf.DataQuality.BEST,
                                        data_labels=[gf.DataLabel.GLITCHES],
                                        observing_runs=settings["runs"], split_spec=spec)

    by_segments, by_dataset = {}, {}
    for split in SPLITS:
        reader = obtainer()
        reader.get_valid_segments(settings["ifos"], settings["seed"], split)
        by_segments[split] = {record.gps_key for record in reader.transient_segments}
        by_dataset[split] = served_by_dataset(obtainer(), split, settings["onsource"], settings["seed"])
    assert_membership_is_assign(by_segments, catalogue, spec, f"get_valid_segments, {configuration}")
    assert_membership_is_assign(by_dataset, catalogue, spec, f"a dataset, {configuration}")

    index = TransientIndex(catalogue)
    excluded = index.assign_splits(spec)
    left_out = expected(catalogue, spec).get(gf.EXCLUDED, set())
    assert left_out and sum(excluded.values()) == len(left_out)


def test_membership_check_sees_a_seeded_split(catalogue):
    """The planted leak: membership by a hash of the key and the seed is caught."""
    def seeded(seed):
        served = {split: set() for split in SPLITS}
        for record in catalogue:
            u = int(hashlib.md5(f"{record.gps_key}:{seed}".encode()).hexdigest(), 16) / 16 ** 32
            served["test" if u < 0.1 else "validate" if u < 0.2 else "train"].add(record.gps_key)
        return served

    with pytest.raises(AssertionError, match="against spec.assign"):
        assert_membership_is_assign(seeded(7), catalogue, gf.SplitSpec(), "a seeded split")
