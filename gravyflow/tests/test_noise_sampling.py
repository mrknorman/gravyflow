"""
Noise sampling: the default sampling mode, and RANDOM addressed by index (``RandomPlan``,
``draw_examples``): each row's density over passes, the arithmetic that locates a batch,
the reordering of passes, the mixing of rows within a batch, and the draw of examples.
"""
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.noise import (
    MIXING_WIDTH, RandomPlan, SamplingPlan, draw_examples,
)
from gravyflow.tests._helpers import representation_tolerance


def white_dataset(**kwargs):
    """A dataset over white noise: nothing is read."""
    return gf.GravyflowDataset(
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS],
        steps_per_epoch=1,
        **kwargs,
    )


class TestSamplingMode:

    @pytest.mark.parametrize("group", ["train", "validate", "test"])
    def test_every_group_defaults_to_random(self, group):
        assert white_dataset(group=group).sampling_mode == gf.SamplingMode.RANDOM

    def test_an_explicit_mode_is_kept(self):
        assert white_dataset(group="train", sampling_mode=gf.SamplingMode.GRID).sampling_mode == gf.SamplingMode.GRID


# ---------------------------------------------------------------------------
# RANDOM addressed by index: the plan's arithmetic (``RandomPlan``)
# ---------------------------------------------------------------------------

#: Rows' seconds of window starts: a long row, a short one, fractional ones.
SECONDS = np.array([200.0, 20.0, 100.0, 3.3, 41.9, 7.1])


def random_plan(seconds=SECONDS, batches_per_second=0.5, seed=7, fixed_order=None, detectors=1):
    """A RANDOM plan over rows of ``seconds`` of starts, each piece 10,000 samples, a million apart."""
    rows = len(seconds)
    sampling = SamplingPlan(np.zeros((rows, detectors, 2)), np.asarray(seconds, float))
    firsts = np.repeat(np.arange(rows, dtype=np.int64)[:, None] * 10 ** 6, detectors, axis=1)
    return RandomPlan.build(gf.SeedTree(seed), "train", sampling, firsts,
                            np.full((rows, detectors), 10_000, np.int64), batches_per_second, fixed_order)


class TestRandomPlan:
    """
    Row j serves floor((p + 1) e_j + u_j) - floor(p e_j + u_j) batches in pass p, so
    its density over any number of passes is exact to within one batch; batch g is
    located by arithmetic alone; every pass reorders the rows; and each batch mixes
    its examples over its pass so every row serves exactly its batches' examples in
    expectation.
    """

    @pytest.mark.parametrize("passes", [1, 2, 7, 50])
    def test_each_row_serves_its_density_to_within_one_batch(self, passes):
        plan = random_plan()  # 0.5 batches a second of starts: the test's own nominal density
        served = sum(plan.due(p) for p in range(passes))
        # Exact: floor() telescopes, so the count differs from the density by less than one batch.
        assert np.all(np.abs(served - passes * 0.5 * SECONDS) < 1.0)
        assert plan.batches_before(passes) == int(np.sum(served))

    def test_a_batch_is_located_by_arithmetic(self):
        plan = random_plan()
        g = 0
        for p in range(6):
            for position in range(int(np.sum(plan.due(p)))):
                assert plan.locate(g) == (p, position)
                g += 1

    def test_every_pass_reorders_the_rows(self):
        plan = random_plan()
        orders = {tuple(plan.order(p)) for p in range(8)}
        assert len(orders) == 8, "a pass repeated another's order"
        assert all(sorted(order) == list(range(len(SECONDS))) for order in orders)
        shortest_first = np.argsort(SECONDS)
        fixed = random_plan(fixed_order=shortest_first)
        assert {tuple(fixed.order(p)) for p in range(8)} == {tuple(shortest_first)}

    def test_a_pass_visits_each_row_for_its_batches_in_its_order(self):
        plan = random_plan()
        for p in range(3):
            own = [plan.mixed(g)[0] for g in range(plan.batches_before(p), plan.batches_before(p + 1))]
            runs = [row for k, row in enumerate(own) if k == 0 or own[k - 1] != row]
            assert runs == [row for row in plan.order(p) if plan.due(p)[row] > 0]
            assert np.array_equal(np.bincount(own, minlength=len(SECONDS)), plan.due(p))

    @pytest.mark.parametrize("seconds", [SECONDS, [200.0, 20.0, 100.0], [100.0, 50.0], [3.0]])
    def test_mixing_serves_each_row_exactly_its_batches_in_expectation(self, seconds):
        plan = random_plan(np.asarray(seconds))
        for p in range(4):
            expected = np.zeros(len(seconds))
            for g in range(plan.batches_before(p), plan.batches_before(p + 1)):
                mixed = plan.mixed(g)
                assert len(mixed) == MIXING_WIDTH
                np.add.at(expected, mixed, 1.0 / MIXING_WIDTH)
            # Representation: sum of 1/MIXING_WIDTH over batches * MIXING_WIDTH terms, in float64.
            terms = (plan.batches_before(p + 1) - plan.batches_before(p)) * MIXING_WIDTH
            tolerance = representation_tolerance(np.float64, np.max(plan.due(p)), c=np.sqrt(terms))
            np.testing.assert_allclose(expected, plan.due(p), rtol=0, atol=tolerance)

    @pytest.mark.parametrize("segment_order", list(gf.SegmentOrder))
    def test_the_obtainers_segment_order_orders_every_pass(self, segment_order):
        from gravyflow.src.dataset.config import WindowSpec

        obtainer = gf.NoiseDataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                                        observing_runs=[gf.ObservingRun.O3], augmentations=[],
                                        segment_order=segment_order)
        start = 1238170000.0
        lengths = [900.0, 200.0, 500.0, 300.0]  # rows in time order, each 1000 s after the last
        obtainer.valid_segments = np.array([[[start + 1000.0 * i, start + 1000.0 * i + n]]
                                            for i, n in enumerate(lengths)])
        window = WindowSpec.from_params(sample_rate_hertz=2048.0, onsource_duration_seconds=1.0,
                                        offsource_duration_seconds=1.75, crop_duration_seconds=0.5)
        plan = obtainer.random_plan(gf.SeedTree(1), window, 16)
        orders = [list(plan.order(p)) for p in range(4)]
        if segment_order is gf.SegmentOrder.RANDOM:
            assert len({tuple(order) for order in orders}) > 1
        else:
            fixed = {gf.SegmentOrder.CHRONOLOGICAL: [0, 1, 2, 3], gf.SegmentOrder.SHORTEST_FIRST: [1, 3, 2, 0]}
            assert orders == [fixed[segment_order]] * 4

    def test_no_row_feeds_a_run_of_batches_alone(self):
        plan = random_plan()
        alone = [len(set(plan.mixed(g).tolist())) == 1 for g in range(plan.batches_before(3))]
        assert not any(alone)


class TestDrawExamples:
    """Exactly B examples, each from its batch's mixed rows, on the read's grid, a pure function of the address."""

    def test_exactly_the_batch_from_its_mixed_rows_on_the_grid(self):
        plan, tree = random_plan(), gf.SeedTree(1)
        for g in range(30):
            rows, starts, rounds = draw_examples(plan, tree.batch("train", 0, g), g, [gf.IFO.H1], 16, 100, 300,
                                                 0.0, False, quantum=4)
            assert rows.shape == (16,) and starts.shape == (16, 1) and np.all(rounds == 1)
            assert set(rows.tolist()) <= set(plan.mixed(g).tolist())
            offsets = starts[:, 0] - plan.firsts[rows, 0]
            assert np.all((offsets >= 300) & (offsets + 100 + 16 <= plan.counts[rows, 0]))
            assert np.all(starts % 4 == 300 % 4)
            again = draw_examples(plan, tree.batch("train", 0, g), g, [gf.IFO.H1], 16, 100, 300, 0.0, False, 4)
            assert all(np.array_equal(a, b) for a, b in zip((rows, starts, rounds), again))

    def test_coincident_detectors_start_at_one_time(self):
        plan = random_plan(detectors=2)
        _, starts, _ = draw_examples(plan, gf.SeedTree(1).batch("train", 0, 0), 0, [gf.IFO.H1, gf.IFO.L1],
                                     16, 100, 300, 0.0, True)
        assert np.array_equal(starts[:, 0], starts[:, 1])

    def test_independent_detectors_start_the_separation_apart(self):
        plan, separation = random_plan(detectors=2), 3000.0  # of a piece's 10,000 samples: chance would break it
        ifos = [gf.IFO.H1, gf.IFO.L1]
        for g in range(10):
            _, starts, rounds = draw_examples(plan, gf.SeedTree(1).batch("train", 0, g), g, ifos,
                                              16, 100, 300, separation, False)
            assert np.all(np.abs(starts[:, 0] - starts[:, 1]) >= separation)
            assert np.any(starts[:, 0] != starts[:, 1])
