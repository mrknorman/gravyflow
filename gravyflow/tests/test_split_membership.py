"""
Split membership: which split an example belongs to never depends on a seed.

A dataset's split streams (``train``, ``validate``, ``test``) differ by design,
so they may order and shuffle a split, but never decide its members: a record
or a segment is in one split whatever the seed, and no state an obtainer built
while serving one split reaches another. These tests run offline, on
synthetic catalogue records.
"""

import copy
from dataclasses import fields, replace
from unittest import mock

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.transient_index import TransientIndex
from gravyflow.src.dataset.splits import EXCLUDED
from gravyflow.src.utils.gps import gps_to_key
from gravyflow.tests.conftest import catalogued_events, no_glitch_vetoes, usable_over_its_runs

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
    """
    The glitch catalogue and its data quality, offline: every index is built from
    these records, and every record passes data quality.
    """
    records = glitch_records()
    with mock.patch(
        "gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
        return_value=records,
    ), mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", usable_over_its_runs), \
            catalogued_events():
        yield {record.gps_key for record in records}


def glitch_obtainer():
    return gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3,
    )


#: Where each detector's science segments start in each 1500 s, so each has its own.
SCIENCE_OFFSET_SECONDS = {gf.IFO.L1: 0.0, gf.IFO.H1: 300.0}


class OfflineNoiseObtainer(gf.NoiseDataObtainer):
    """Science segments of 1000 s every 1500 s, offset by detector, with nothing vetoed."""

    def get_segment_times(self, start, end, ifo, state_flags):
        starts = np.arange(start, end - 1000.0, 1500.0) + SCIENCE_OFFSET_SECONDS[ifo]
        return np.stack([starts, starts + 1000.0], axis=-1)

    def remove_unwanted_segments(self, ifo, valid_segments):
        return valid_segments


def offline_noise_obtainer():
    return OfflineNoiseObtainer(
        observing_runs=gf.ObservingRun.O3,
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
    )


def members(obtainer, split, seed):
    obtainer.get_valid_segments([gf.IFO.L1], seed=seed, group_name=split)
    return {segment.gps_key for segment in obtainer.transient_segments}


class TestTransientMembership:

    def test_one_root_gives_disjoint_complete_splits(self, catalogue):
        """
        Each split reads its own copy of the obtainer, as a dataset does: the
        memberships must be a partition of the catalogue.
        """
        obtainer = glitch_obtainer()
        split_members = {split: members(copy.deepcopy(obtainer), split, gf.Defaults.seed) for split in SPLITS}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                assert not split_members[a] & split_members[b], f"{a} and {b} share records"
        assigned = TransientIndex(glitch_records())
        excluded = assigned.assign_splits(gf.SplitSpec())
        assert set().union(*split_members.values()) == set(assigned._groups) <= catalogue
        assert len(catalogue) - len(assigned._groups) == sum(excluded.values())
        assert all(split_members[split] for split in SPLITS)

    @pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
    def test_membership_does_not_depend_on_the_seed(self, catalogue, seed):
        reference = members(glitch_obtainer(), "test", 1000)
        assert members(glitch_obtainer(), "test", seed) == reference

    def test_the_seed_orders_the_passes_never_the_plan(self, catalogue):
        """A split's plan is its records in time order, whatever the seed: each pass's
        order is drawn when batches are served (test_transient_windows)."""
        def order(seed):
            obtainer = glitch_obtainer()
            obtainer.get_valid_segments([gf.IFO.L1], seed=seed, group_name="train")
            return [segment.gps_key for segment in obtainer.transient_segments]

        assert order(1) == order(2) == sorted(order(1))

    def test_a_saved_index_loads_every_record_whole(self, tmp_path):
        """
        Save and load keep every field of every record: a glitch's SNR, peak frequency,
        duration and confidence, and an event's masses (its footprint), distance and
        astrophysical probability, NaN where the catalogue has none.
        """
        from gravyflow.src.dataset.features.event import EventConfidence, SourceType
        from gravyflow.src.dataset.features.transient_index import transient_footprint

        glitch = replace(glitch_records(1)[0], snr=12.5, peak_freq=211.0, duration=0.75, ml_confidence=0.97)
        t = 1238170000.4
        event = TransientSegment(
            gps_key=gps_to_key(t), transient_gps_time=t, start_gps_time=t - 48.2, end_gps_time=t + 48.2,
            label=gf.DataLabel.EVENTS, kind=SourceType.BBH, observing_run=gf.ObservingRun.O3,
            seen_in=[gf.IFO.H1, gf.IFO.L1], confidence=EventConfidence.CONFIDENT, name="GW-ROUND-TRIP",
            mass1=1.46, mass2=1.27, distance=40.0, p_astro=1.0,
        )
        index = TransientIndex([glitch, event, replace(glitch_records(2)[1], snr=float("nan"))])
        index.assign_splits(gf.SplitSpec())
        index.save(tmp_path / "index.npz")
        loaded = TransientIndex.load(tmp_path / "index.npz")
        names = [field.name for field in fields(TransientSegment)]
        for before, after in zip(index.records, loaded.records):
            for field in names:
                a, b = getattr(before, field), getattr(after, field)
                same = (a == b) or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
                assert same, f"{field}: saved {a!r}, loaded {b!r}"
            assert transient_footprint(after) == transient_footprint(before)

    def test_index_assignment_is_a_partition(self):
        """Every record is in one split or counted out of all of them, by its footprint's split."""
        spec = gf.SplitSpec()
        index = TransientIndex(glitch_records(5000))
        excluded = index.assign_splits(spec)
        counts = index.get_group_counts()
        assert sum(counts.values()) + sum(excluded.values()) == len(index)
        assert set(counts) == {"train", "validate", "test"}
        for record in index:
            split = spec.assign(record.transient_gps_time - 48.2, record.transient_gps_time + 48.2)
            assert index._groups.get(record.gps_key, EXCLUDED) == split


class TestObtainerCopies:
    """
    An obtainer is a specification: a copy, which every split reads from,
    never carries what serving data built (segments, index, rng).
    """

    def used_glitch_obtainer(self):
        obtainer = glitch_obtainer()
        obtainer.get_valid_segments([gf.IFO.L1], seed=1000, group_name="train")
        obtainer.rng = np.random.default_rng(1)
        return obtainer

    def test_a_copy_of_a_used_transient_obtainer_is_fresh(self, catalogue):
        used = self.used_glitch_obtainer()
        train = {segment.gps_key for segment in used.transient_segments}
        fresh = copy.deepcopy(used)
        assert fresh.transient_segments == []
        assert fresh._feature_index is None
        assert fresh.rng is None
        assert not members(fresh, "test", 1000) & train

    def test_a_dataset_builds_from_a_used_transient_obtainer(self, catalogue):
        used = self.used_glitch_obtainer()
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
        train = used.get_valid_segments([gf.IFO.L1], seed=1000, group_name="train")

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
    first reads an example, where reading data begins (stopped there, offline).
    """
    served = {}

    def stop(self, *args, **kwargs):
        served["members"] = {segment.gps_key for segment in self.transient_segments}
        raise Served

    with mock.patch.object(gf.TransientDataObtainer, "_read_example", stop):
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
        index.assign_splits(gf.SplitSpec())
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
        spec = gf.SplitSpec()
        assert spec == gf.DEFAULT_SPLIT
        assert spec.fractions == (("test", 0.1), ("validate", 0.1), ("train", 0.8))
        assert (spec.block_seconds, spec.guard_seconds) == (8192.0, 128.0)
        assert glitch_obtainer().split_spec == offline_noise_obtainer().split_spec == spec


#: One obtainer's calls, in order: each must serve exactly its own split.
CALLS = ("train", "validate", "train", "test")

#: A small window at 64 Hz, for batches drawn from fake strain.
WINDOW = dict(
    sample_rate_hertz=64.0,
    onsource_duration_seconds=1.0,
    crop_duration_seconds=0.5,
    offsource_duration_seconds=1.75,
    num_examples_per_batch=4,
)


def noise_split(split, ifos=(gf.IFO.L1,)):
    """
    The segments of ``split`` on ``ifos``, from a fresh obtainer (the first
    detector's): the reference membership.
    """
    segments = offline_noise_obtainer().get_valid_segments(list(ifos), seed=1000, group_name=split)
    return set(map(tuple, segments[:, 0]))


def served_noise(call, split):
    """
    The segments a real-noise call for ``split`` serves: those of the reader
    when it first reads strain, where reading data would begin (stopped there,
    offline). RANDOM reads windows (``read_windows``); the other modes acquire rows.
    """
    served = {}

    def stop(self, *args, **kwargs):
        served["members"] = set(map(tuple, self.valid_segments[:, 0]))
        raise Served

    with mock.patch.object(OfflineNoiseObtainer, "acquire", stop), \
            mock.patch.object(OfflineNoiseObtainer, "read_windows", stop):
        with pytest.raises(Served):
            call(split)
    return served["members"]


def windows_of(values):
    """A ``read_windows`` serving each detector's window the constant ``values[ifo]``."""
    def read_windows(self, ifos, sample_rate_hertz, starts, num_onsource_samples, num_offsource_samples):
        level = np.array([values(ifo) for ifo in ifos], np.float32)[None, :, None]
        shape = (len(starts), len(ifos))
        return (np.broadcast_to(level, shape + (num_onsource_samples,)).copy(),
                np.broadcast_to(level, shape + (num_offsource_samples,)).copy())
    return read_windows


#: Zero strain in every window, as the store's reads serve it.
fake_windows = windows_of(lambda ifo: 0.0)


def read_glitch(self, record, reads, window_spec):
    """An example of zeros for every record, on each detector it reads."""
    return (
        np.zeros((len(reads), window_spec.num_onsource_samples), np.float32),
        np.zeros((len(reads), window_spec.num_offsource_samples), np.float32),
    )


#: A strain value for each detector, so a batch shows whose strain it holds.
DETECTOR_STRAIN = {gf.IFO.L1: 1.0, gf.IFO.H1: 2.0}


#: Constant strain in each detector's windows: its DETECTOR_STRAIN.
detector_windows = windows_of(DETECTOR_STRAIN.__getitem__)


class TestOneObtainerCalledForEachSplit:
    """
    An obtainer's served state belongs to the split that built it: one
    obtainer called for train, validate, train and test serves exactly each
    split in turn, on every route, and a generator it gave out for one split
    refuses once the obtainer serves another.
    """

    def test_a_transient_obtainer_called_directly(self, catalogue):
        obtainer = glitch_obtainer()
        for split in CALLS:
            reference = served_members(through_a_direct_call, split)
            assert served_members(lambda _, split: through_a_direct_call(obtainer, split), split) \
                == reference, f"the {split} call served another split"

    def test_a_noise_obtainer_called_directly(self):
        obtainer = offline_noise_obtainer()
        for split in CALLS:
            served = served_noise(lambda split: next(obtainer(ifos=[gf.IFO.L1], group=split)), split)
            assert served == noise_split(split), f"the {split} call served another split"

    @pytest.mark.parametrize("noise_type", [gf.NoiseType.REAL, gf.NoiseType.PSEUDO_REAL])
    def test_one_noise_obtainer_for_each_split(self, noise_type):
        obtainer = offline_noise_obtainer()
        noise = gf.NoiseObtainer(noise_type=noise_type, ifo_data_obtainer=obtainer, ifos=[gf.IFO.L1])
        for split in CALLS:
            served = served_noise(lambda split: next(noise(group=split, **WINDOW)), split)
            assert served == noise_split(split), f"the {split} call served another split"
        assert obtainer._served_split is None, "the caller's obtainer was changed"

    def test_hand_set_segments_are_served_when_no_split_is_named(self):
        obtainer = offline_noise_obtainer()
        rows = np.array([[[1238170000.0, 1238171000.0]]])
        obtainer.valid_segments = rows.copy()
        served = served_noise(lambda _: next(obtainer(ifos=[gf.IFO.L1])), None)
        assert served == {(1238170000.0, 1238171000.0)}

    def test_a_noise_generator_refuses_once_its_obtainer_serves_another_split(self):
        obtainer = offline_noise_obtainer()
        obtainer.augmentations = []
        with mock.patch.object(OfflineNoiseObtainer, "read_windows", fake_windows):
            train = obtainer(ifos=[gf.IFO.L1], group="train", seed=7, **WINDOW)
            next(train)
            validate = obtainer(ifos=[gf.IFO.L1], group="validate", seed=7, **WINDOW)
            next(validate)
            with pytest.raises(RuntimeError, match="another split"):
                next(train)
            next(validate)

    def test_a_transient_generator_refuses_once_its_obtainer_serves_another_split(self, catalogue):
        obtainer = glitch_obtainer()
        obtainer.augmentations = []
        with mock.patch.object(gf.TransientDataObtainer, "_read_example", read_glitch):
            train = obtainer(ifos=[gf.IFO.L1], group="train", seed=7, **WINDOW)
            next(train)
            validate = obtainer(ifos=[gf.IFO.L1], group="validate", seed=7, **WINDOW)
            next(validate)
            with pytest.raises(RuntimeError, match="another split"):
                next(train)
            next(validate)


class TestNoAllSplit:
    """
    There is no split of every record: each name is test, validate or train, and
    real events, named or not, are served in test.
    """

    @pytest.mark.parametrize("name", ["all", "val", "Train", "direct"])
    def test_other_names_are_refused(self, catalogue, name):
        with pytest.raises(KeyError, match="a split is one of"):
            members(glitch_obtainer(), name, 1000)
        with pytest.raises(KeyError, match="a split is one of"):
            offline_noise_obtainer().get_valid_segments([gf.IFO.L1], seed=1000, group_name=name)

    def test_the_splits_cover_the_catalogue_less_what_is_excluded(self, catalogue):
        served = set().union(*(members(glitch_obtainer(), split, 1000) for split in SPLITS))
        index = TransientIndex(glitch_records())
        excluded = index.assign_splits(gf.SplitSpec())
        assert served <= catalogue
        assert len(catalogue - served) == sum(excluded.values())

    def test_named_events_are_served_in_test(self, offline_events):
        names = [event["name"] for event in offline_events]
        obtainer = gf.TransientDataObtainer(
            observing_runs=gf.ObservingRun.O3,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            event_names=names,
        )
        obtainer.augmentations = []
        batch = next(obtainer(ifos=[gf.IFO.H1, gf.IFO.L1], group="test",
                              **{**WINDOW, "num_examples_per_batch": 3}))
        assert np.shape(batch[gf.ReturnVariables.ONSOURCE])[:2] == (3, 2)
        assert {segment.name for segment in obtainer.transient_segments} == set(names)
        with pytest.raises(ValueError, match="evaluation data") as refusal:
            next(obtainer(ifos=[gf.IFO.H1, gf.IFO.L1], group="train", **WINDOW))
        # It names the whole opt-in that would train on them.
        assert "train_on_real_events=True" in str(refusal.value)
        assert "coincidence=Coincidence.COINCIDENT" in str(refusal.value)

    @pytest.mark.parametrize("labels", [[gf.DataLabel.EVENTS], [gf.DataLabel.GLITCHES]])
    def test_training_on_real_events_needs_the_opt_in(self, labels):
        """Real events are trained on only under the coincident opt-in, whose flag marks them."""
        for build in (gf.IFODataObtainer, gf.TransientDataObtainer):
            with pytest.raises(ValueError, match="coincidence=Coincidence.COINCIDENT"):
                build(data_quality=gf.DataQuality.BEST, data_labels=labels, train_on_real_events=True)
            build(data_quality=gf.DataQuality.BEST, data_labels=labels, train_on_real_events=True,
                  coincidence=gf.Coincidence.COINCIDENT)

    def test_a_noise_obtainer_refuses_events_by_kind(self):
        """
        A noise obtainer refuses real events however they are named, and never a
        chirp glitch, which equals DataLabel.EVENTS as a number.
        """
        for events in (gf.DataLabel.EVENTS, gf.EventConfidence.CONFIDENT):
            with pytest.raises(ValueError, match="Real events are not noise"):
                gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE, events])
        chirps = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST,
                                    data_labels=[gf.DataLabel.NOISE, gf.GlitchType.CHIRP])
        assert isinstance(chirps, gf.NoiseDataObtainer)

    def test_a_chirp_glitch_obtainer_serves_no_real_events(self, offline_events):
        """
        GlitchType.CHIRP equals DataLabel.EVENTS as a number; it asks for chirp
        glitches alone, so its index holds no event and every split is served.
        """
        with mock.patch("gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
                        return_value=[]) as glitches:
            obtainer = gf.TransientDataObtainer(observing_runs=gf.ObservingRun.O3,
                                                data_quality=gf.DataQuality.BEST,
                                                data_labels=[gf.GlitchType.CHIRP])
            for split in SPLITS:
                obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], seed=1000, group_name=split)
                assert obtainer.transient_segments == [], split
        assert len(obtainer._feature_index) == 0
        assert glitches.call_args.kwargs["glitch_types"] == [gf.GlitchType.CHIRP]


@pytest.fixture
def two_detector_catalogue():
    """
    Glitches offline, alternately seen in L1 and in H1, every one passing data
    quality. Yields their keys by detector.
    """
    records = [
        replace(record, seen_in=[(gf.IFO.L1, gf.IFO.H1)[i % 2]])
        for i, record in enumerate(glitch_records())
    ]
    with mock.patch(
        "gravyflow.src.dataset.features.segment_builders.build_glitch_segments",
        return_value=records,
    ), mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", usable_over_its_runs), \
            catalogued_events(), no_glitch_vetoes():
        yield {ifo: {r.gps_key for r in records if ifo in r.seen_in} for ifo in (gf.IFO.L1, gf.IFO.H1)}


#: One obtainer's calls for one split, on other detectors in turn.
DETECTOR_CALLS = ([gf.IFO.L1], [gf.IFO.H1], [gf.IFO.L1, gf.IFO.H1], [gf.IFO.L1])


class TestOneObtainerCalledForOtherDetectors:
    """
    The served state belongs to the detectors that built it too: one obtainer
    called for one split on L1, then H1, then both, serves each call its own
    detectors' records, never the first call's.
    """

    def test_a_noise_obtainer_serves_each_call_its_detectors(self):
        obtainer = offline_noise_obtainer()
        obtainer.augmentations = []
        for ifos in DETECTOR_CALLS:
            served = served_noise(lambda split: next(obtainer(ifos=ifos, group=split, **WINDOW)), "train")
            names = [ifo.name for ifo in ifos]
            assert served == noise_split("train", ifos), f"the {names} call served other detectors"
            with mock.patch.object(OfflineNoiseObtainer, "read_windows", detector_windows):
                onsource = np.asarray(next(obtainer(ifos=ifos, group="train", seed=7, **WINDOW))[
                    gf.ReturnVariables.ONSOURCE])
            assert onsource.shape[:2] == (4, len(ifos))
            for channel, ifo in enumerate(ifos):
                np.testing.assert_array_equal(onsource[:, channel], DETECTOR_STRAIN[ifo])

    def test_a_call_keeps_the_order_its_split_was_given(self):
        # NoiseObtainer orders a split by one seed and draws its windows by
        # another: the drawing call keeps the order, on the same detectors.
        obtainer = offline_noise_obtainer()
        ordered = obtainer.get_valid_segments([gf.IFO.L1], seed=1000, group_name="train").copy()
        reordered = offline_noise_obtainer().get_valid_segments([gf.IFO.L1], seed=7, group_name="train")
        assert not np.array_equal(reordered, ordered), "the seed does not order the split"
        served = {}

        def stop(self, *args, **kwargs):
            served["segments"] = self.valid_segments.copy()
            raise Served

        with mock.patch.object(OfflineNoiseObtainer, "read_windows", stop), pytest.raises(Served):
            next(obtainer(ifos=[gf.IFO.L1], group="train", seed=7, **WINDOW))
        np.testing.assert_array_equal(served["segments"], ordered)

    def test_a_transient_obtainer_serves_each_call_its_detectors(self, two_detector_catalogue):
        obtainer = glitch_obtainer()
        obtainer.augmentations = []
        for ifos in DETECTOR_CALLS:
            reference = served_members(lambda fresh, split: next(fresh(ifos=ifos, group=split)), "train")
            assert reference and reference <= set().union(*(two_detector_catalogue[i] for i in ifos))
            served = served_members(lambda _, split: next(obtainer(ifos=ifos, group=split)), "train")
            assert served == reference, f"the {[ifo.name for ifo in ifos]} call served other detectors"
            with mock.patch.object(gf.TransientDataObtainer, "_read_example", read_glitch):
                onsource = next(obtainer(ifos=ifos, group="train", seed=7, **WINDOW))[
                    gf.ReturnVariables.ONSOURCE]
            assert np.shape(onsource)[:2] == (4, len(ifos))


#: Specs other than the default: other fractions, another salt.
OTHER_SPECS = [
    gf.SplitSpec(fractions={"train": 0.6, "validate": 0.2, "test": 0.2}),
    gf.SplitSpec(version="fold-1"),
]


class TestOneSpecPerObtainer:
    """
    The obtainer that partitions time holds the split: every split it serves is
    its spec's, and a route has no second set of fractions to give.
    """

    @pytest.mark.parametrize("spec", OTHER_SPECS, ids=["fractions", "salt"])
    def test_an_obtainer_serves_its_specs_split(self, catalogue, spec):
        obtainer = gf.TransientDataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
            observing_runs=gf.ObservingRun.O3, split_spec=spec,
        )
        served = members(obtainer, "validate", 1000)
        assert served
        for record in glitch_records():
            t = record.transient_gps_time
            assert (record.gps_key in served) == (spec.assign(t - 48.2, t + 48.2) == "validate")
        assert served != members(glitch_obtainer(), "validate", 1000)

    def test_no_route_takes_fractions(self):
        with pytest.raises(TypeError):
            gf.NoiseObtainer(groups={"train": 1.0})
        with pytest.raises(TypeError):
            offline_noise_obtainer().get_valid_segments([gf.IFO.L1], seed=1, groups={}, group_name="train")
        assert not hasattr(TransientIndex, "assign_groups")
        assert not hasattr(gf, "DEFAULT_SPLIT_FRACTIONS")
