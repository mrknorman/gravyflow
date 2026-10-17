"""
The acquisition module's pieces: IFOData, the obtainer's construction and the label
kinds it serves, segment arithmetic (vetoes, cuts, bins, ordering, splits), reads
through the store, and the segment plots.

Every test runs offline: strain comes from a fake GWOSC fetch (``conftest.gwosc_serving``)
whose samples count from a known GPS time, so each read decodes to its GPS label.
"""

import numpy as np
import pytest
from bokeh.models import BoxAnnotation, GlyphRenderer, Span
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset import acquisition as gf_acq
from gravyflow.src.dataset.acquisition.base import label_kinds
from gravyflow.src.dataset.acquisition.gwosc_data import DEFAULT_GWOSC_FLAGS, GWOSC_RUNS
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.tests.conftest import gwosc_serving

RV = gf.ReturnVariables

#: A whole GPS second inside O3a; the fake's samples count from it.
ORIGIN = 1238175000
#: The fake source's one native rate: reads at it are never resampled.
FAKE_RATE = 256.0


class CountingFetch:
    """A GWOSC fetch, offline: each sample of [start, end) on the absolute grid valued its index from ORIGIN."""

    rates = (FAKE_RATE,)
    tile_seconds = 64.0
    source_id = "gwosc-counting"

    def __init__(self):
        self.calls = []

    def __call__(self, ifo, sample_rate_hertz, interval):
        start, end = interval
        self.calls.append((ifo.name, float(start), float(end)))
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate)), int(np.ceil(end * rate))
        values = np.arange(first, stop, dtype=np.int64) - int(ORIGIN * rate)
        return SegmentStrain(values.astype(np.float32), first / rate, rate, self.source_id)


def noise_obtainer(**kwargs):
    arguments = dict(data_quality=gf.DataQuality.BEST, data_labels=gf.DataLabel.NOISE,
                     observing_runs=gf.ObservingRun.O3)
    arguments.update(kwargs)
    return gf.IFODataObtainer(**arguments)


@pytest.fixture
def obtainer():
    return noise_obtainer()


# =============================================================================
# CORE UTILITIES AND IFOData
# =============================================================================

def test_concatenate_batches_stacks_detectors():
    """Each detector's (batch, 1, samples) block is stacked on axis 1, its start times beside it."""
    first = np.arange(20, dtype=np.float32).reshape(2, 1, 10)
    second = first + 100.0
    onsource, offsource, starts = gf_acq.concatenate_batches(
        [first, second], [first + 1000.0, second + 1000.0], [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])])
    np.testing.assert_array_equal(np.asarray(onsource), np.concatenate([first, second], axis=1))
    np.testing.assert_array_equal(np.asarray(offsource), np.concatenate([first, second], axis=1) + 1000.0)
    np.testing.assert_array_equal(starts, [[1.0, 3.0], [2.0, 4.0]])


class TestIFOData:

    def test_a_bare_array_is_one_piece(self):
        data = np.arange(1000, dtype=np.float32)
        ifo_data = gf_acq.IFOData(data=data, sample_rate_hertz=100.0, start_gps_time=[0.0])
        assert isinstance(ifo_data.data, list) and len(ifo_data.data) == 1
        np.testing.assert_array_equal(ifo_data.numpy()[0], data)

    def test_duration_and_sample_interval(self):
        ifo_data = gf_acq.IFOData(data=[np.zeros(1000, np.float32), np.zeros(250, np.float32)],
                                  sample_rate_hertz=100.0, start_gps_time=[0.0, 0.0])
        np.testing.assert_array_equal([float(d) for d in ifo_data.duration_seconds], [10.0, 2.5])
        assert ifo_data.time_interval_seconds == 0.01

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_strain_is_refused(self, bad):
        """Held strain is finite: a row holding NaN or Inf is refused, never zeroed."""
        data = np.array([1.0, bad, 3.0, 5.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite strain"):
            gf_acq.IFOData(data=[data], sample_rate_hertz=1.0, start_gps_time=[0.0])


# =============================================================================
# CONSTRUCTION AND LABELS
# =============================================================================

class TestIFODataObtainerInit:

    def test_a_noise_obtainer_holds_its_runs_bounds_and_flags(self):
        built = noise_obtainer(observing_runs=[gf.ObservingRun.O2, gf.ObservingRun.O3])
        assert isinstance(built, gf.NoiseDataObtainer) and built.data_quality == gf.DataQuality.BEST
        assert built.start_gps_times == [GWOSC_RUNS["O2"]["gps_start"], GWOSC_RUNS["O3a"]["gps_start"]]
        assert built.end_gps_times == [GWOSC_RUNS["O2"]["gps_end"], GWOSC_RUNS["O3b"]["gps_end"]]
        assert built.state_flags == [DEFAULT_GWOSC_FLAGS, DEFAULT_GWOSC_FLAGS]

    def test_override_attributes_valid(self):
        """Pins today's wrapping of an override in a list (see the open question on scalar overrides)."""
        assert noise_obtainer(overrides={"saturation": 0.5}).saturation == [0.5]

    def test_override_attributes_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalide override value"):
            noise_obtainer(overrides={"nonexistent_attribute": 123})

    def test_built_in_runs_are_gwoscs_bounds_and_default_flags(self):
        """O3 is GWOSC's O3a and O3b together; leap seconds are tested in test_public_sources."""
        o3 = gf.ObservingRun.O3.value
        assert (o3.name, o3.start_gps_time, o3.end_gps_time) == (
            "O3", GWOSC_RUNS["O3a"]["gps_start"], GWOSC_RUNS["O3b"]["gps_end"])
        for run in (gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3):
            assert run.value.state_flags == {gf.DataQuality.BEST: DEFAULT_GWOSC_FLAGS}


@pytest.mark.parametrize("labels, kind, kinds", [
    ([gf.DataLabel.NOISE], gf.NoiseDataObtainer, {gf.DataLabel.NOISE}),
    ([gf.DataLabel.NOISE, gf.DataLabel.GLITCHES], gf.NoiseDataObtainer, {gf.DataLabel.NOISE, gf.DataLabel.GLITCHES}),
    ([gf.DataLabel.NOISE, gf.GlitchType.BLIP, gf.GlitchType.TOMTE], gf.NoiseDataObtainer,
     {gf.DataLabel.NOISE, gf.DataLabel.GLITCHES}),
    ([gf.DataLabel.EVENTS], gf.TransientDataObtainer, {gf.DataLabel.EVENTS}),
    ([gf.EventConfidence.CONFIDENT], gf.TransientDataObtainer, {gf.DataLabel.EVENTS}),
    ([gf.DataLabel.EVENTS, gf.EventConfidence.MARGINAL], gf.TransientDataObtainer, {gf.DataLabel.EVENTS}),
    ([gf.DataLabel.GLITCHES], gf.TransientDataObtainer, {gf.DataLabel.GLITCHES}),
    ([gf.GlitchType.BLIP, gf.GlitchType.TOMTE], gf.TransientDataObtainer, {gf.DataLabel.GLITCHES}),
    ([gf.EventConfidence.CONFIDENT, gf.GlitchType.BLIP, gf.GlitchType.SCATTERED_LIGHT],
     gf.TransientDataObtainer, {gf.DataLabel.EVENTS, gf.DataLabel.GLITCHES}),
], ids=lambda value: "-".join(label.name for label in value) if isinstance(value, list) else None)
def test_labels_choose_the_obtainer_and_the_kinds_it_serves(labels, kind, kinds):
    """
    NOISE among the labels makes a noise obtainer, else a transient one; the kinds are
    read from each label's type, never its number (BLIP, CONFIDENT and GLITCHES are all 1).
    """
    built = gf.IFODataObtainer(observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
                               data_labels=labels)
    assert type(built) is kind
    assert built.acquisition_mode == (gf_acq.AcquisitionMode.NOISE if kind is gf.NoiseDataObtainer
                                      else gf_acq.AcquisitionMode.TRANSIENT)
    assert [(type(label), label) for label in built.data_labels] == [(type(label), label) for label in labels]
    assert label_kinds(built.data_labels) == kinds


# =============================================================================
# SEGMENT ARITHMETIC
# =============================================================================

class TestVetoes:

    @pytest.mark.parametrize("segments, vetoes, expected", [
        ([[0, 100], [200, 300]], [[50, 75], [250, 275]], [[0, 50], [75, 100], [200, 250], [275, 300]]),
        ([[0, 100], [200, 300]], np.empty((0, 2)), [[0, 100], [200, 300]]),
        ([[50, 100], [200, 300]], [[0, 150]], [[200, 300]]),
        ([[0, 1000]], [[100, 200], [400, 500], [700, 800]], [[0, 100], [200, 400], [500, 700], [800, 1000]]),
    ], ids=["inside", "no_veto", "covers_a_segment", "several"])
    def test_vetoed_time_is_removed(self, obtainer, segments, vetoes, expected):
        np.testing.assert_array_equal(
            obtainer.veto_time_segments(np.array(segments, float), np.array(vetoes, float)), expected)


class TestCompressSegments:

    def test_overlapping_segments_merge(self, obtainer):
        compressed = obtainer.compress_segments(np.array([[0, 10], [5, 15], [20, 30]]))
        np.testing.assert_array_equal(compressed, [[0, 15], [20, 30]])

    def test_empty_input(self, obtainer):
        assert obtainer.compress_segments(np.empty((0, 2))).size == 0


def test_calculate_bin_indices(obtainer):
    """A segment's first bin is the floor of its start, its end bin the ceiling of its end."""
    start_bins, end_bins = obtainer.calculate_bin_indices(
        np.array([[100.0, 200.0], [300.0, 400.0], [550.0, 601.0]]), 100.0, 0.0)
    np.testing.assert_array_equal(start_bins, [1, 3, 5])
    np.testing.assert_array_equal(end_bins, [2, 4, 7])


class TestPlanSampling:

    def test_short_pieces_are_dropped(self, obtainer):
        """The sampling plan draws only from pieces of at least the minimum length."""
        segments = np.array([[[0, 10]], [[2048, 2053]], [[4096, 4116]]], dtype=float)  # each in its own bin
        plan = obtainer.plan_sampling(segments, 8.0, 2.0, sample_rate_hertz=4096.0)
        np.testing.assert_array_equal(plan.rows, segments[[0, 2]])
        np.testing.assert_allclose(plan.start_seconds, [8.0, 18.0])

    def test_a_bin_needs_every_detector(self, obtainer):
        """A bin is drawn from only where every IFO has a usable piece."""
        segments = np.array([
            [[0, 10], [0, 15]],
            [[2048, 2053], [2048, 2058]],  # L1's 5 s is too short: dropped
            [[4096, 4116], [4096, 4121]],
        ], dtype=float)
        plan = obtainer.plan_sampling(segments, 8.0, 2.0, sample_rate_hertz=4096.0)
        np.testing.assert_array_equal(plan.rows, segments[[0, 2]])


class TestSegmentOrdering:

    def test_random_order_is_the_seeds_permutation(self, obtainer):
        segments = np.array([[[10.0 * i, 10.0 * i + 5]] for i in range(10)])
        ordered = obtainer.order_segments(segments.copy(), gf.SegmentOrder.RANDOM, seed=42)
        np.testing.assert_array_equal(np.sort(ordered, axis=0), segments)  # the same rows
        assert not np.array_equal(ordered, segments)
        np.testing.assert_array_equal(obtainer.order_segments(segments.copy(), gf.SegmentOrder.RANDOM, seed=42),
                                      ordered)
        assert not np.array_equal(obtainer.order_segments(segments.copy(), gf.SegmentOrder.RANDOM, seed=7),
                                  ordered)

    def test_shortest_first_orders_rows_by_the_first_detector(self, obtainer):
        segments = np.array([
            [[0, 10], [0, 40]],
            [[20, 50], [20, 30]],
            [[60, 80], [60, 61]],
        ])
        ordered = obtainer.order_segments(segments.copy(), gf.SegmentOrder.SHORTEST_FIRST, seed=42)
        np.testing.assert_array_equal(ordered, segments[[0, 2, 1]])  # 10, 20, 30 s in the first

    def test_order_segments_chronological(self, obtainer):
        """CHRONOLOGICAL keeps the order given (see the open question on whether it should sort)."""
        segments = np.array([[[100, 200]], [[0, 50]], [[300, 400]]])
        result = obtainer.order_segments(segments.copy(), gf.SegmentOrder.CHRONOLOGICAL, seed=42)
        np.testing.assert_array_equal(result, segments)


class TestCutSegments:

    @pytest.mark.parametrize("segments, expected", [
        ([[0, 250], [300, 400]], [[0, 100], [100, 200], [200, 250], [300, 400]]),
        ([[0, 100], [100, 200]], [[0, 100], [100, 200]]),
        ([[0, 50]], [[0, 50]]),
    ], ids=["across_boundaries", "on_boundaries", "short"])
    def test_segments_are_cut_at_the_grid(self, obtainer, segments, expected):
        np.testing.assert_array_equal(obtainer.cut_segments(np.array(segments, float), 100.0, 0.0), expected)


class TestMergeBins:

    @pytest.mark.parametrize("lists, expected", [
        ([[[0, 50], [100, 150]], [[10, 60], [110, 160]]],
         [[[[0, 50]], [[10, 60]]], [[[100, 150]], [[110, 160]]]]),
        # The bin [100, 200) holds neither detector's data: it is dropped.
        ([[[0, 50], [200, 250]], [[10, 60], [210, 260]]],
         [[[[0, 50]], [[10, 60]]], [[[200, 250]], [[210, 260]]]]),
        ([[[0, 50], [100, 150]]], [[[[0, 50]]], [[[100, 150]]]]),
        # A segment across a bin edge is clipped to each bin.
        ([[[50, 150]]], [[[[50, 100]]], [[[100, 150]]]]),
    ], ids=["two_detectors", "empty_bin_dropped", "one_detector", "clipped"])
    def test_each_bin_holds_every_detectors_pieces_in_it(self, obtainer, lists, expected):
        bins = obtainer.merge_bins([np.array(s, float) for s in lists], interval=100.0)
        assert len(bins) == len(expected)
        for got, want in zip(bins, expected):
            assert len(got) == len(want)
            for pieces, pieces_expected in zip(got, want):
                np.testing.assert_array_equal(pieces, pieces_expected)


def test_clear_valid_segments(obtainer):
    obtainer.valid_segments = np.array([[0, 100]])
    obtainer.sampling_plan = obtainer.plan_sampling(np.array([[[0.0, 100.0]]]), 8.0, 2.0, sample_rate_hertz=4096.0)
    obtainer.ifos = [gf.IFO.L1]
    obtainer.clear_valid_segments()
    assert (obtainer.valid_segments, obtainer.sampling_plan, obtainer.ifos) == (None, None, None)


class TestSplits:

    def test_a_split_reads_its_admissible_time(self):
        """A split's segments lie in that split's blocks (``SplitSpec.admissible``)."""
        spec = gf.SplitSpec(block_seconds=100.0, guard_seconds=0.0)
        segments = np.array([[100.0 * k, 100.0 * (k + 1)] for k in range(40)])
        for name in ("train", "validate", "test"):
            served = spec.admissible(segments, name)
            assert len(served), name
            for start, end in served:
                assert spec.assign(start, end) == name
        blocks = {spec.block_split(k) for k in range(40)}
        assert blocks == {"train", "validate", "test"}

    def test_the_splits_partition_time_less_their_guards(self):
        """
        Without a guard the splits share out all time; a guard removes half its width on
        each side of every edge between blocks of different splits, and nothing more.
        """
        segments = np.array([[0.0, 100.0], [100.0, 450.0], [500.0, 800.0], [800.0, 3000.0]])
        fractions = {"train": 0.6, "validate": 0.2, "test": 0.2}
        names = ("train", "validate", "test")
        seconds = lambda s: float(np.sum(s[:, 1] - s[:, 0]))
        total = seconds(segments)

        unguarded = gf.SplitSpec(fractions=fractions, block_seconds=100.0, guard_seconds=0.0)
        split = {name: unguarded.admissible(segments, name) for name in names}
        assert all(len(split[name]) for name in names)
        assert sum(seconds(s) for s in split.values()) == total

        guard = 10.0
        guarded = gf.SplitSpec(fractions=fractions, block_seconds=100.0, guard_seconds=guard)
        split = {name: guarded.admissible(segments, name) for name in names}
        edges = [100.0 * k for k in range(1, 30) if guarded.block_split(k - 1) != guarded.block_split(k)]
        assert edges

        def covered(a, b):
            return float(np.sum(np.clip(np.minimum(segments[:, 1], b) - np.maximum(segments[:, 0], a), 0, None)))

        lost = sum(covered(edge - guard / 2, edge + guard / 2) for edge in edges)
        assert sum(seconds(s) for s in split.values()) == pytest.approx(total - lost, abs=1e-9)
        for a in names:
            for b in names:
                if a != b:
                    for start, end in split[a]:
                        gaps = np.maximum(split[b][:, 0] - end, start - split[b][:, 1])
                        assert np.all(gaps >= guard - 1e-9), (a, b)


class TestSegmentPipeline:
    """Compress, cut at the grid, and share out: the segment arithmetic in sequence."""

    def test_compressed_segments_are_cut_at_the_grid(self, obtainer):
        original = np.array([[0.0, 100.0], [50.0, 150.0], [200.0, 300.0], [250.0, 350.0], [400.0, 450.0]])
        compressed = obtainer.compress_segments(original)
        np.testing.assert_array_equal(compressed, [[0, 150], [200, 350], [400, 450]])
        np.testing.assert_array_equal(obtainer.cut_segments(compressed, 100.0, 0.0),
                                      [[0, 100], [100, 150], [200, 300], [300, 350], [400, 450]])

    def test_the_splits_share_out_all_the_cut_time(self, obtainer):
        cut = obtainer.cut_segments(np.array([[0.0, 150.0], [250.0, 400.0], [500.0, 520.0], [600.0, 800.0]]),
                                    100.0, 0.0)
        spec = gf.SplitSpec(fractions={"train": 0.7, "validate": 0.15, "test": 0.15},
                            block_seconds=100.0, guard_seconds=0.0)
        seconds = lambda s: float(np.sum(s[:, 1] - s[:, 0]))
        assert sum(seconds(spec.admissible(cut, name)) for name in ("train", "validate", "test")) == seconds(cut)


# =============================================================================
# READS
# =============================================================================

class TestGetOnsourceOffsourceChunks:

    def test_get_chunks_defaults_applied(self):
        """The batch size and seed default when not given; RANDOM reads its windows by index."""
        obtainer = noise_obtainer()
        # Hand-set rows inside O3a; the windows are zeros, read without the network.
        obtainer.valid_segments = np.array([
            [[1238166100.0, 1238167000.0]],
            [[1238168000.0, 1238169000.0]],
        ])

        def zero_windows(ifos, sample_rate_hertz, starts, num_onsource_samples, num_offsource_samples):
            shape = (len(starts), len(ifos))
            return np.zeros(shape + (num_onsource_samples,), np.float32), \
                np.zeros(shape + (num_offsource_samples,), np.float32)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(obtainer, "read_windows", zero_windows)
            output = next(obtainer.get_onsource_offsource_chunks(
                sample_rate_hertz=1024.0, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
                offsource_duration_seconds=1.0,
                num_examples_per_batch=None,  # the default
                ifos=gf.IFO.L1,
                seed=None,  # the default
            ))
        onsource = output[RV.ONSOURCE]
        assert ops.shape(onsource)[:2] == (gf.Defaults.num_examples_per_batch, 1)
        gps = np.asarray(output[RV.START_GPS_TIME])[:, 0]
        assert np.all((gps >= 1238166100.0) & (gps < 1238169000.0))

    def test_windows_of_a_served_row_decode_to_their_labels(self):
        """Each window is consecutive samples of its row; its label is its first returned sample's GPS."""
        obtainer = noise_obtainer()
        obtainer.augmentations = []
        obtainer.valid_segments = np.array([[[ORIGIN + 10.0, ORIGIN + 40.0]]])
        with gwosc_serving(CountingFetch()):
            output = next(obtainer.get_onsource_offsource_chunks(
                sample_rate_hertz=FAKE_RATE, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
                offsource_duration_seconds=1.0, num_examples_per_batch=2, ifos=gf.IFO.L1, seed=42, crop=True,
            ))
        onsource = np.asarray(output[RV.ONSOURCE], dtype=np.float64)
        assert onsource.shape == (2, 1, int(FAKE_RATE))
        np.testing.assert_array_equal(np.diff(onsource, axis=-1), 1.0)
        labels = np.asarray(output[RV.START_GPS_TIME], dtype=np.float64)
        np.testing.assert_array_equal(labels, ORIGIN + onsource[..., 0] / FAKE_RATE)


class TestAcquire:

    def test_a_row_is_read_as_its_whole_seconds_with_its_first_samples_gps(self):
        """At the source's own rate a piece's read is its whole seconds from the first grid sample inside it."""
        obtainer = noise_obtainer()
        start = ORIGIN + 0.3
        with gwosc_serving(CountingFetch()):
            [row] = list(obtainer.acquire(FAKE_RATE, np.array([[[start, start + 20.0]]]), [gf.IFO.L1]))
        first = int(np.ceil(start * FAKE_RATE))  # 0.3 s is 76.8 samples: the first is the 77th
        assert row.start_gps_time == [first / FAKE_RATE]
        np.testing.assert_array_equal(row.numpy()[0], np.arange(20 * int(FAKE_RATE)) + (first - ORIGIN * FAKE_RATE))


class TestGetValidSegmentsErrors:

    def test_get_valid_segments_invalid_group(self, obtainer):
        with pytest.raises(KeyError, match="a split is one of"):
            obtainer.get_valid_segments(ifos=[gf.IFO.L1], seed=42, group_name="nonexistent_group")


# =============================================================================
# SEGMENT PLOTS
# =============================================================================

def glyphs(figure):
    return [r for r in figure.renderers if isinstance(r, GlyphRenderer)]


class TestGenerateSegmentTimelinePlot:

    SEGMENTS = {
        gf.IFO.L1: np.array([[1238166018.0, 1238170000.0], [1238175000.0, 1238180000.0]]),
        gf.IFO.H1: np.array([[1238166018.0, 1238172000.0]]),
        gf.IFO.V1: np.array([[1238168000.0, 1238175000.0]]),
    }

    def test_each_detector_is_one_glyph_of_its_segments(self):
        p = gf.generate_segment_timeline_plot(self.SEGMENTS, observing_runs=[gf.ObservingRun.O3], title="Timeline")
        assert p.title.text == "Timeline"
        drawn = {tuple(map(tuple, np.stack([r.data_source.data["left"], r.data_source.data["right"]], axis=1)))
                 for r in glyphs(p)}
        assert drawn == {tuple(map(tuple, s)) for s in self.SEGMENTS.values()}

    @pytest.mark.parametrize("show", [True, False])
    def test_run_bands_are_shown_on_request(self, show):
        runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]
        p = gf.generate_segment_timeline_plot(self.SEGMENTS, observing_runs=runs, show_observing_runs=show)
        bands = [(b.left, b.right) for b in p.center if isinstance(b, BoxAnnotation)]
        assert bands == ([(r.value.start_gps_time, r.value.end_gps_time) for r in runs] if show else [])

    def test_an_empty_detector_draws_nothing(self):
        p = gf.generate_segment_timeline_plot({gf.IFO.L1: self.SEGMENTS[gf.IFO.L1], gf.IFO.H1: np.empty((0, 2))})
        assert [r.data_source.data["ifo"][0] for r in glyphs(p)] == [gf.IFO.L1.value.name]


class TestGenerateExampleExtractionPlot:

    SEGMENT = np.array([1238166100.0, 1238166200.0])

    @pytest.mark.parametrize("ifos", [[gf.IFO.L1], [gf.IFO.L1, gf.IFO.H1, gf.IFO.V1]], ids=["one", "three"])
    def test_one_subplot_per_detector_marking_each_point(self, ifos):
        points = np.array([1238166130.0, 1238166150.0])
        layout = gf.generate_example_extraction_plot({ifo: self.SEGMENT for ifo in ifos}, points,
                                                     onsource_duration_seconds=2.0)
        assert [p.title.text for p in layout.children] == [ifo.value.name for ifo in ifos]
        for p in layout.children:
            assert [s.location for s in p.center if isinstance(s, Span)] == list(points)
            assert (glyphs(p)[0].glyph.left, glyphs(p)[0].glyph.right) == tuple(self.SEGMENT)

    def test_the_onsource_is_padded_by_the_crop_and_the_offsource_drawn(self):
        point, onsource, offsource, crop = 1238166150.0, 2.0, 1.0, 0.5
        layout = gf.generate_example_extraction_plot({gf.IFO.L1: self.SEGMENT}, np.array([point]),
                                                     onsource_duration_seconds=onsource,
                                                     offsource_duration_seconds=offsource,
                                                     crop_duration_seconds=crop)
        _, onsource_quad, offsource_quad = (r.glyph for r in glyphs(layout.children[0]))
        assert onsource_quad.right - onsource_quad.left == onsource + 2 * crop
        assert offsource_quad.right - offsource_quad.left == pytest.approx(offsource)
        assert offsource_quad.right < onsource_quad.left


# =============================================================================
# POST-PROCESSING
# =============================================================================

class TestPostProcessWhitening:
    """The obtainer's whiten=True post-processing."""

    def test_whitening_never_writes_the_buffers_it_hands_to_jax(self):
        """
        On CPU, JAX may read a numpy array without copying it, and it whitens
        asynchronously, so a buffer written after the call can change what the
        whitening reads. The caller's arrays are never written, and every call on
        identical input matches a whitening of untouched copies. The match allows
        1e-5 of the peak: XLA on CPU rarely differs in the last few bits between
        identical calls (about 6e-7 of a 3.7 peak, about once in a thousand
        calls), while a whitening that read a rewritten buffer is off by far more.
        """
        rng = np.random.default_rng(0)
        onsource = (rng.standard_normal((8, 2, 512)) * 1e-21).astype(np.float32)
        offsource = (rng.standard_normal((8, 2, 4096)) * 1e-21).astype(np.float32)
        expected = np.asarray(gf.replace_nan_and_inf_with_zero(
            gf.whiten(onsource, offsource, 256.0)
        ))
        obtainer = noise_obtainer()
        tolerance = 1e-5 * np.abs(expected).max()
        calls, wrong, written = 300, 0, 0
        for _ in range(calls):
            given = {RV.ONSOURCE: onsource.copy(), RV.OFFSOURCE: offsource.copy()}
            served = obtainer._post_process_batch(dict(given), 256.0, 1.0, whiten=True)
            wrong += not np.allclose(np.asarray(served[RV.ONSOURCE]), expected, rtol=0, atol=tolerance)
            written += not (np.array_equal(given[RV.ONSOURCE], onsource)
                            and np.array_equal(given[RV.OFFSOURCE], offsource)
                            and np.array_equal(np.asarray(served[RV.OFFSOURCE]), offsource))
        assert (wrong, written) == (0, 0), (
            f"of {calls} calls, {wrong} whitened onsources differ from a whitening of "
            f"untouched copies and {written} wrote the buffers given or served another offsource")
