"""
Split membership: which split an example belongs to never depends on a seed.

A dataset's split streams (``train``, ``validate``, ``test``) differ by design,
so they may order and shuffle a split, but never decide its members: a record
or a segment is in one split whatever the seed, and no state an obtainer built
while serving one split reaches another. These tests run offline, on
synthetic catalogue records.
"""

import copy
from unittest import mock

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.cache import DiskCache
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.transient_index import TransientIndex
from gravyflow.src.dataset.noise.noise import sequential_seed
from gravyflow.src.utils.gps import gps_to_key

SPLITS = ("train", "validate", "test")
O3_START = 1238166018.0


def glitch_records(count=3000):
    """Synthetic L1 glitches spread over O3, one per distinct GPS key."""
    times = O3_START + np.sort(np.random.default_rng(7).uniform(0.0, 3.0e7, count))
    return [
        TransientSegment(
            gps_key=gps_to_key(float(t)),
            transient_gps_time=float(t),
            start_gps_time=float(t) - 48.2,
            end_gps_time=float(t) + 48.2,
            label=gf.DataLabel.GLITCHES,
            kind=GlitchType.BLIP,
            observing_run=gf.ObservingRun.O3,
            seen_in=[gf.IFO.L1],
            weight=1.0,
        )
        for t in times
    ]


@pytest.fixture
def catalogue():
    """The glitch catalogue, offline: every index is built from these records."""
    records = glitch_records()
    with mock.patch(
        "gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
        return_value=records,
    ):
        yield {record.gps_key for record in records}


def glitch_obtainer():
    return gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3,
    )


class OfflineNoiseObtainer(gf.NoiseDataObtainer):
    """Science segments of 1000 s every 1500 s, with nothing vetoed."""

    def get_segment_times(self, start, end, ifo, state_flags):
        starts = np.arange(start, end - 1000.0, 1500.0)
        return np.stack([starts, starts + 1000.0], axis=-1)

    def remove_unwanted_segments(self, ifo, valid_segments):
        return valid_segments


def offline_noise_obtainer():
    return OfflineNoiseObtainer(
        observing_runs=gf.ObservingRun.O3,
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        prefetch_segments=0,
    )


def members(obtainer, split, seed):
    obtainer.get_valid_segments([gf.IFO.L1], seed=seed, group_name=split)
    return {segment.gps_key for segment in obtainer.transient_segments}


class TestTransientMembership:

    def test_one_root_gives_disjoint_complete_splits(self, catalogue):
        """
        Each split of one root reads its own copy of the obtainer, seeded from
        its own stream, as a dataset does: the memberships must still be a
        partition of the catalogue.
        """
        tree = gf.SeedTree(gf.Defaults.seed)
        obtainer = glitch_obtainer()
        split_members = {
            split: members(
                copy.deepcopy(obtainer), split, sequential_seed(tree, split, "transient/reader")
            )
            for split in SPLITS
        }
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                assert not split_members[a] & split_members[b], f"{a} and {b} share records"
        assert set().union(*split_members.values()) == catalogue
        assert all(split_members[split] for split in SPLITS)

    @pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
    def test_membership_does_not_depend_on_the_seed(self, catalogue, seed):
        reference = members(glitch_obtainer(), "test", 1000)
        assert members(glitch_obtainer(), "test", seed) == reference

    def test_the_seed_still_orders_a_split(self, catalogue):
        def order(seed):
            obtainer = glitch_obtainer()
            obtainer.get_valid_segments([gf.IFO.L1], seed=seed, group_name="train")
            return [segment.gps_key for segment in obtainer.transient_segments]

        assert order(1) != order(2)
        assert sorted(order(1)) == sorted(order(2))

    def test_index_assignment_is_a_partition(self):
        index = TransientIndex(glitch_records(5000))
        index.assign_groups({"train": 0.8, "validate": 0.1, "test": 0.1})
        counts = index.get_group_counts()
        assert sum(counts.values()) == len(index)
        assert set(counts) == {"train", "validate", "test"}


class TestObtainerCopies:
    """
    An obtainer is a specification: a copy, which every split reads from,
    never carries what serving data built (segments, index, caches, rng).
    """

    def used_glitch_obtainer(self, tmp_path):
        obtainer = glitch_obtainer()
        obtainer.get_valid_segments([gf.IFO.L1], seed=1000, group_name="train")
        obtainer.rng = np.random.default_rng(1)
        # What a served batch leaves behind: an open disk cache, with its lock.
        obtainer._glitch_cache = DiskCache(tmp_path / "glitch_cache.h5", mode="a")
        return obtainer

    def test_a_copy_of_a_used_transient_obtainer_is_fresh(self, catalogue, tmp_path):
        used = self.used_glitch_obtainer(tmp_path)
        train = {segment.gps_key for segment in used.transient_segments}
        fresh = copy.deepcopy(used)
        assert fresh.transient_segments == []
        assert fresh._feature_index is None
        assert fresh.rng is None
        assert fresh._glitch_cache is None
        assert not members(fresh, "test", 1000) & train

    def test_a_dataset_builds_from_a_used_transient_obtainer(self, catalogue, tmp_path):
        used = self.used_glitch_obtainer(tmp_path)
        segments_before = list(used.transient_segments)
        gf.Dataset(
            noise_obtainer=used,
            group="validate",
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.DATA_LABEL],
            steps_per_epoch=1,
        )
        assert used.transient_segments == segments_before

    def test_real_noise_reads_its_own_split_from_a_used_obtainer(self):
        """A caller's obtainer that already holds one split's segments never serves them to another."""
        used = offline_noise_obtainer()
        noise = gf.NoiseObtainer(ifo_data_obtainer=used, noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1])
        train = used.get_valid_segments([gf.IFO.L1], seed=1000, groups=noise.groups, group_name="train")

        used.rng = np.random.default_rng(1)
        fresh = copy.deepcopy(used)
        assert fresh.valid_segments is None and fresh.rng is None

        reader = noise._reader([gf.IFO.L1], gf.SeedTree(gf.Defaults.seed), "validate", 2048.0)
        validate = reader.valid_segments
        assert len(validate) and len(train)
        assert not set(map(tuple, validate[:, 0])) & set(map(tuple, train[:, 0]))


class Served(Exception):
    """A reader reached its data: the split it serves is fixed by then."""


def served_members(route, split):
    """
    The records a route serves for ``split``: the reader's segments when it
    first reaches its cache, where reading data would begin (stopped there,
    offline).
    """
    served = {}

    def stop(self, *args, **kwargs):
        served["members"] = {segment.gps_key for segment in self.transient_segments}
        raise Served

    with mock.patch.object(gf.TransientDataObtainer, "_initialize_transient_cache", stop):
        # A dataset reraises the stop as its own noise-generation error.
        with pytest.raises(Exception) as stopped:
            route(glitch_obtainer(), split)
    assert "members" in served, f"the route stopped before its data: {stopped.value!r}"
    return served["members"]


DATASET_ARGS = dict(
    input_variables=[gf.ReturnVariables.ONSOURCE],
    output_variables=[gf.ReturnVariables.DATA_LABEL],
    steps_per_epoch=1,
)


def through_a_direct_call(obtainer, split):
    next(obtainer(ifos=[gf.IFO.L1], group=split))


def through_a_dataset(obtainer, split):
    gf.Dataset(noise_obtainer=obtainer, group=split, **DATASET_ARGS)[0]


def through_a_noise_obtainer(obtainer, split):
    noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifo_data_obtainer=obtainer, ifos=[gf.IFO.L1])
    gf.Dataset(noise_obtainer=noise, group=split, **DATASET_ARGS)[0]


def through_a_composed_dataset(obtainer, split):
    pool = gf.FeaturePool(name="glitches", label=0, probability=1.0, transient_obtainer=obtainer)
    gf.ComposedDataset(pools=[pool], group=split, **DATASET_ARGS)[0]


ROUTES = {
    "direct-call": through_a_direct_call,
    "dataset": through_a_dataset,
    "noise-obtainer": through_a_noise_obtainer,
    "composed-dataset": through_a_composed_dataset,
}


class TestEveryRouteSplitsAlike:
    """
    Every entry point splits one catalogue by one set of fractions, so a
    record is in the same split whichever route trains or evaluates on it.
    """

    @pytest.fixture
    def by_route(self, catalogue):
        index = TransientIndex(glitch_records())
        index.assign_groups()
        by_route = {
            route: {split: served_members(through, split) for split in SPLITS}
            for route, through in ROUTES.items()
        }
        by_route["index"] = {
            split: {key for key, group in index._groups.items() if group == split} for split in SPLITS
        }
        by_route["valid-segments"] = {split: members(glitch_obtainer(), split, 1000) for split in SPLITS}
        return by_route

    def test_each_split_has_one_membership(self, by_route):
        for split in SPLITS:
            reference = by_route["index"][split]
            assert reference, split
            for route in by_route:
                assert by_route[route][split] == reference, f"{route} serves another {split} split"

    def test_splits_are_disjoint_across_routes(self, by_route):
        for a in by_route:
            for b in by_route:
                for i, first in enumerate(SPLITS):
                    for second in SPLITS[i + 1:]:
                        shared = by_route[a][first] & by_route[b][second]
                        assert not shared, f"{a} {first} and {b} {second} share {len(shared)} records"

    def test_the_default_split_is_the_ruled_one(self):
        assert dict(gf.DEFAULT_SPLIT_FRACTIONS) == {"train": 0.8, "validate": 0.1, "test": 0.1}
        assert gf.NoiseObtainer().groups == dict(gf.DEFAULT_SPLIT_FRACTIONS)


class TestOneObtainerCalledForTwoSplits:
    """
    An obtainer's served state belongs to the split that built it: called
    again for another split, it rebuilds rather than serving the first.
    """

    def test_a_transient_obtainer_called_for_another_split(self, catalogue):
        def train_then(obtainer, split):
            with pytest.raises(Served):
                through_a_direct_call(obtainer, "train")
            through_a_direct_call(obtainer, split)

        assert served_members(train_then, "validate") == served_members(through_a_direct_call, "validate")

    def test_a_noise_obtainer_called_for_another_split(self):
        obtainer = offline_noise_obtainer()
        train = obtainer.get_valid_segments([gf.IFO.L1], seed=1000, group_name="train")
        validate = obtainer.get_valid_segments([gf.IFO.L1], seed=1000, group_name="validate")
        fresh = offline_noise_obtainer().get_valid_segments([gf.IFO.L1], seed=1000, group_name="validate")
        np.testing.assert_array_equal(validate, fresh)
        assert not set(map(tuple, validate[:, 0])) & set(map(tuple, train[:, 0]))
