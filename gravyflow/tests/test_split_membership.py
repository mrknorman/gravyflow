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

        class OfflineNoiseObtainer(gf.NoiseDataObtainer):
            """Science segments of 1000 s every 1500 s, with nothing vetoed."""

            def get_segment_times(self, start, end, ifo, state_flags):
                starts = np.arange(start, end - 1000.0, 1500.0)
                return np.stack([starts, starts + 1000.0], axis=-1)

            def remove_unwanted_segments(self, ifo, valid_segments, get_times=None):
                return valid_segments, None

        used = OfflineNoiseObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE],
            prefetch_segments=0,
        )
        noise = gf.NoiseObtainer(ifo_data_obtainer=used, noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1])
        train = used.get_valid_segments([gf.IFO.L1], seed=1000, groups=noise.groups, group_name="train")

        used.rng = np.random.default_rng(1)
        fresh = copy.deepcopy(used)
        assert fresh.valid_segments is None and fresh.rng is None

        reader = noise._reader([gf.IFO.L1], gf.SeedTree(gf.Defaults.seed), "validate", 2048.0)
        validate = reader.valid_segments
        assert len(validate) and len(train)
        assert not set(map(tuple, validate[:, 0])) & set(map(tuple, train[:, 0]))
