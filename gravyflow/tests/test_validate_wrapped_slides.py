"""
Wrapping time slides and the per-detector path (`validate.slides.WrappedSlides`,
`validate.per_detector`).

Known answers: the background time of a given segment set and lag list, with
wrapping inside a segment and across segments; where a slid time lands; that no
slide reuses zero lag or another slide's pairing; and, on a small synthetic case,
that a per-detector model's background and found injections are the same by the
per-detector path (scored once per detector) as by the joint path (a full run per
slide).
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.detection.stream import DetectorTriggers, bank_identity, coincide
from gravyflow.src.validate.per_detector import (
    _gather_paired, found_statistic, joint_slide_streams, paired_windows, score_detectors, slide_coincidences,
    slide_streams,
)
from gravyflow.src.validate.slides import WrappedSlides, footprint_seconds, minimum_shift_seconds
from gravyflow.src.validate.stream import StreamBackground, StreamGeometry

#: Three segments, 100 + 50 + 30 = 180 s of ring.
SEGMENTS = [(100.0, 200.0), (300.0, 350.0), (1000.0, 1030.0)]
YEAR = 365.25 * 86400.0


# ------------------------------------------------------------ known answers

def test_the_background_time_is_every_slide_the_whole_ring():
    slides = WrappedSlides(SEGMENTS, [10.0, 50.0, 170.0], num_detectors=2, minimum_seconds=5.0)
    assert slides.ring_seconds == 180.0
    np.testing.assert_array_equal(slides.ring_starts, [0.0, 100.0, 150.0])
    assert slides.background_seconds == 3 * 180.0
    # Three detectors slide by the same lags (detector 2 by twice each); the ring is still whole.
    assert WrappedSlides(SEGMENTS, [10.0, 50.0, 70.0], 3, 5.0).background_seconds == 540.0


def test_slid_times_wrap_inside_a_segment_and_across_segments():
    slides = WrappedSlides(SEGMENTS, [10.0, 50.0], num_detectors=2, minimum_seconds=5.0)
    # Detector 1 read 50 s further round the ring: its data at ring q stands at q - 50.
    # 120 (ring 20) wraps back past the first segment's start to the last segment's (ring 150).
    # 310 (ring 110) crosses back into the first segment (ring 60); 1020 (ring 170) into the second (ring 120).
    np.testing.assert_allclose(slides.slid_times([120.0, 310.0, 1020.0, 180.0], 1, 1), [1000.0, 160.0, 320.0, 130.0])
    np.testing.assert_allclose(slides.slid_times([120.0], 0, 1), [110.0])
    # Detector 0 is never moved.
    np.testing.assert_allclose(slides.slid_times([120.0, 310.0], 1, 0), [120.0, 310.0])
    # One segment: the slide wraps round its own end.
    one = WrappedSlides([(0.0, 100.0)], [30.0], 2, 5.0)
    np.testing.assert_allclose(one.slid_times([10.0, 40.0], 0, 1), [80.0, 10.0])
    # Ring positions and GPS times are each other's inverse.
    gps = np.array([100.0, 150.5, 300.0, 349.9, 1000.0, 1029.0])
    np.testing.assert_allclose(slides.from_ring(slides.to_ring(gps)), gps)
    with pytest.raises(ValueError, match="outside every segment"):
        slides.to_ring([250.0])


@pytest.mark.parametrize("lags, detectors, match", [
    ([180.0], 2, "zero lag"),          # the whole ring: zero lag again
    ([360.0], 2, "zero lag"),
    ([3.0], 2, "zero lag"),            # nearer zero than the minimum
    ([177.0], 2, "zero lag"),          # 3 s short of the ring: 3 s from zero round it
    ([5.0], 2, "zero lag"),            # the minimum itself: a shift must exceed it
    ([90.0], 3, "zero lag"),           # detectors 0 and 2 slid by 180 s: zero lag
    ([50.0, 52.0], 2, "reuse"),        # two slides 2 s apart: one pairing reused
    ([10.0, 190.0], 2, "reuse"),       # equal round the ring
    ([40.0, 132.0], 3, "reuse"),       # detectors 0 and 2: 80 and 264 = 84 round the ring
])
def test_no_slide_reuses_zero_lag_or_another_slide_s_pairing(lags, detectors, match):
    with pytest.raises(ValueError, match=match):
        WrappedSlides(SEGMENTS, lags, detectors, minimum_seconds=5.0)


def test_spaced_slides_fill_the_ring_and_keep_clear_of_zero_lag():
    two = WrappedSlides.spaced(SEGMENTS, 2, minimum_seconds=5.0, step_seconds=6.0)
    np.testing.assert_allclose(two.lags, 6.0 * np.arange(1, 30))  # 29 * 6 = 174: 6 s short of the ring
    assert two.background_seconds == 29 * 180.0
    three = WrappedSlides.spaced(SEGMENTS, 3, minimum_seconds=5.0, step_seconds=6.0)
    assert len(three) == 14  # detector 2's largest lag 168 s, 12 s short of the ring
    assert len(WrappedSlides.spaced(SEGMENTS, 2, 5.0, 6.0, max_slides=4)) == 4
    with pytest.raises(ValueError, match="exceed"):
        WrappedSlides.spaced(SEGMENTS, 2, minimum_seconds=5.0, step_seconds=5.0)


def test_a_thirty_day_test_set_yields_thousands_of_years_of_background():
    window = WindowSpec(sample_rate_hertz=2048.0, onsource_duration_seconds=1.0, offsource_duration_seconds=1.0,
                        crop_duration_seconds=0.0)
    minimum = minimum_shift_seconds([gf.IFO.H1, gf.IFO.L1], window)
    light = gf.Network([gf.IFO.H1, gf.IFO.L1]).max_arrival_time_difference_seconds
    assert minimum == pytest.approx(light + footprint_seconds(window))
    # Thirty days in 300 segments of 8640 s, with gaps between them.
    starts = 1e9 + np.arange(300) * 10_000.0
    slides = WrappedSlides.spaced(np.stack([starts, starts + 8640.0], axis=1), 2, minimum)
    assert slides.ring_seconds == 30 * 86400.0
    assert slides.background_seconds / YEAR > 1000.0
    assert np.all(slides.lags > minimum)


# ------------------------------------------------- the per-detector path

RATE = 32.0
WINDOW = WindowSpec(sample_rate_hertz=RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=0.5,
                    crop_duration_seconds=0.0)
GEOMETRY = StreamGeometry.of(WINDOW, 0.25)


def detector_model(onsource, offsource, detector):
    """A per-detector model: its window's loudness against its offsource's, weighted by detector."""
    on = np.log(np.mean(onsource[:, 0] ** 2, axis=-1))
    off = np.log(np.mean(offsource[:, 0] ** 2, axis=-1))
    return (1.0 + 0.1 * detector) * (on - off)


def joint_model(onsource, offsource):
    """The same model looking at every detector at once: the sum of its per-detector scores."""
    return sum(detector_model(onsource[:, j:j + 1], offsource[:, j:j + 1], j) for j in range(onsource.shape[1]))


def synthetic_strain(rng):
    """Two detectors' noise over three segments of unequal length, with loud bursts in it."""
    strains, first = [], []
    for start, seconds in ((0.0, 37.0), (100.0, 23.5), (200.0, 18.0)):
        strain = rng.standard_normal((2, int(seconds * RATE)))
        for _ in range(4):
            j, at = rng.integers(2), rng.integers(strain.shape[1] - 16)
            strain[j, at:at + 16] *= 6.0
        strains.append(strain)
        first.append(start)
    return strains, first


def test_the_per_detector_path_gives_the_joint_path_s_background():
    rng = np.random.default_rng(3)
    strains, first = synthetic_strain(rng)
    scores = score_detectors(detector_model, strains, first, GEOMETRY, batch_size=17)
    slides = scores.slides(minimum_seconds=1.6)
    assert len(slides) == 41  # (74.75 - 1.75) / 1.75: the ring is 299 windows of 0.25 s
    np.testing.assert_allclose(slides.lags[:3], [1.75, 3.5, 5.25])  # the fewest whole strides past 1.6 s
    per_detector = slide_streams(scores, slides)
    joint = joint_slide_streams(joint_model, strains, first, GEOMETRY, slides, batch_size=29)
    assert len(per_detector) == len(joint) == len(slides)
    for a, b in zip(per_detector, joint):
        np.testing.assert_allclose(a.scores, b.scores, rtol=0, atol=1e-12)
        assert (a.slide, a.block) == (b.slide, b.block)
    pd_background, joint_background = (StreamBackground(s, 1.0, 0.1) for s in (per_detector, joint))
    assert pd_background.livetime_seconds == pytest.approx(slides.background_seconds)
    assert joint_background.livetime_seconds == pytest.approx(slides.background_seconds)
    np.testing.assert_allclose(np.sort(pd_background.window_scores), np.sort(joint_background.window_scores),
                               rtol=0, atol=1e-12)
    thresholds = np.quantile(pd_background.window_scores, [0.5, 0.9, 0.99])
    np.testing.assert_array_equal(pd_background.far_count(thresholds), joint_background.far_count(thresholds))
    # A slide pairs data across segments: some window's partner lies in another segment.
    pairs = paired_windows(GEOMETRY, scores.segment_windows, slides, 5, np.arange(scores.scores.shape[1]), 2)
    assert np.any(pairs[0][0] != pairs[1][0])


def test_the_per_detector_path_finds_what_the_joint_path_finds():
    rng = np.random.default_rng(4)
    strains, first = synthetic_strain(rng)
    scores = score_detectors(detector_model, strains, first, GEOMETRY)
    slides = scores.slides(minimum_seconds=1.6)
    total = scores.scores.shape[1]
    reach = 0.1 + 1.0
    half = int(np.floor(reach / GEOMETRY.stride_seconds)) + 1
    centre = GEOMETRY.centre_samples / RATE
    found = {"per_detector": [], "joint": []}
    for _ in range(40):
        slide, c = int(rng.integers(len(slides))), int(rng.integers(total))
        amplitude = rng.uniform(0.0, 4.0)
        ring = np.arange(c - half, c + half + 1)
        merger = c * GEOMETRY.stride_seconds + centre
        onsource, offsource = _gather_paired(strains, GEOMETRY, paired_windows(
            GEOMETRY, scores.segment_windows, slides, slide, ring, 2))
        # The signal: a burst at the merger, in every detector, placed in each local window.
        t = (np.arange(WINDOW.num_onsource_samples) + WINDOW.num_offsource_samples) / RATE
        offset = merger - (ring * GEOMETRY.stride_seconds + WINDOW.num_offsource_samples / RATE)
        burst = amplitude * np.exp(-((t[None, :] - WINDOW.num_offsource_samples / RATE - offset[:, None]) / 0.05) ** 2)
        injected = onsource + burst[:, None, :]
        local_pd = sum(detector_model(injected[:, j:j + 1], offsource[:, j:j + 1], j) for j in range(2))
        local_joint = joint_model(injected, offsource)
        found["per_detector"].append(found_statistic(local_pd, ring, merger, GEOMETRY, 1.0, 0.1))
        found["joint"].append(found_statistic(local_joint, ring, merger, GEOMETRY, 1.0, 0.1))
    pd, joint = np.array(found["per_detector"]), np.array(found["joint"])
    np.testing.assert_allclose(pd, joint, rtol=0, atol=1e-12)
    threshold = np.median(pd[np.isfinite(pd)])
    np.testing.assert_array_equal(pd >= threshold, joint >= threshold)
    assert 0 < np.sum(pd >= threshold) < pd.size


def test_slides_planned_on_other_segments_are_refused():
    strains, first = synthetic_strain(np.random.default_rng(5))
    scores = score_detectors(detector_model, strains, first, GEOMETRY)
    with pytest.raises(ValueError, match="other segments"):
        slide_streams(scores, WrappedSlides(SEGMENTS, [10.0], 2, 5.0))
    with pytest.raises(ValueError, match="whole number"):
        slide_streams(scores, WrappedSlides(scores.segments, [2.1], 2, 1.6))


# ------------------------------------------------ stored per-detector triggers

BANK = "sha256:" + "0" * 64


def stored(detector, time, template, snr):
    n = len(time)
    return DetectorTriggers(detector, np.asarray(time, float), np.asarray(template), np.asarray(snr, float),
                            np.ones(n), span=(SEGMENTS[0][0], SEGMENTS[-1][1]), bank=BANK)


def test_stored_triggers_coincide_in_the_slide_that_wraps_them_together():
    slides = WrappedSlides(SEGMENTS, [10.0, 50.0], num_detectors=2, minimum_seconds=5.0)
    h1 = stored("H1", [320.002, 150.0], [3, 3], [8.0, 6.0])
    # L1's 1020 lands on 320 in the 50 s slide (across segments); its 160 lands on 110.
    l1 = stored("L1", [1020.0, 160.0], [3, 4], [7.0, 9.0])
    unslid = coincide({"H1": h1, "L1": l1})
    assert len(unslid) == 0
    first, second = slide_coincidences({"H1": h1, "L1": l1}, slides)
    assert len(first) == 0  # 1020 lands on 310; 160 on 150, but of another template
    assert len(second) == 1
    assert second.time[0] == pytest.approx(320.002)
    np.testing.assert_allclose(second.reweighted_snr["L1"], l1.reweighted_snr[:1])
    np.testing.assert_allclose(second.stat, np.hypot(h1.reweighted_snr[0], l1.reweighted_snr[0]), rtol=1e-6)
    with pytest.raises(ValueError, match="outside every segment"):
        slide_coincidences({"H1": h1, "L1": stored("L1", [250.0], [3], [7.0])}, slides)


def test_one_segment_s_wrapping_slides_are_the_triggers_shifted_round_their_span():
    rng = np.random.default_rng(6)
    span = (0.0, 500.0)
    make = lambda name: DetectorTriggers(name, np.sort(rng.uniform(*span, 400)), rng.integers(0, 3, 400),
                                         rng.uniform(5, 9, 400), rng.uniform(0.5, 2, 400), span=span, bank=BANK)
    h1, l1 = make("H1"), make("L1")
    slides = WrappedSlides([span], [7.0, 120.0, 333.0], 2, 1.0)
    for k, wrapped in enumerate(slide_coincidences({"H1": h1, "L1": l1}, slides)):
        expected = coincide({"H1": h1, "L1": l1.shifted(-slides.lags[k], wrap=True)})
        np.testing.assert_allclose(wrapped.time, expected.time)
        np.testing.assert_allclose(wrapped.stat, expected.stat)
