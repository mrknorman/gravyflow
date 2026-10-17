"""
The time-slide background: a network's detectors slid against each other, every
slid window inside the scored split, and the livetime of every slide counted.

Every window is decoded from the fake strain itself (`provenance`), so where it
came from is read from its samples, never from what the library says of it.
"""

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.dataset.dataset import split_spec_of
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.slides import TimeSlideBackground, TimeSlides
from gravyflow.tests.leakage import fixtures as fx
from gravyflow.tests.leakage.provenance import Window, as_seconds, footprints

RATE = 256.0
IFOS = ("H1", "L1")
WINDOW = WindowSpec(RATE, fx.ONSOURCE_SECONDS, fx.OFFSOURCE_SECONDS, fx.CROP_SECONDS)
GEOMETRY = Window.from_seconds(RATE, fx.ONSOURCE_SECONDS, fx.CROP_SECONDS, fx.OFFSOURCE_SECONDS)


def slid_windows(obtainer, split, slides, batch, budget, seed=3, split_spec=None):
    """
    Every batch the background serves for ``split``, and the background: split by
    ``split_spec``, by default the obtainer's (the one computation, as the validator's).
    """
    spec = split_spec_of([obtainer]) if split_spec is None else split_spec
    background = TimeSlideBackground(obtainer, [gf.IFO[name] for name in IFOS], spec, split, WINDOW, slides,
                                     np.random.default_rng(seed))
    return list(background.batches(batch, budget)), background


def expected_windows(stretch, slide, step):
    """Windows a stretch holds at a slide: the last detector's slid footprint must fit."""
    length = int(round((stretch[1] - stretch[0]) * RATE))
    offset = int(round((len(IFOS) - 1) * slide * step * RATE))
    footprint = WINDOW.num_offsource_samples + WINDOW.num_onsource_samples
    stride = int(round(WINDOW.onsource_duration_seconds * RATE))
    return max((length - footprint - offset) // stride + 1, 0)


@pytest.mark.parametrize("split,other", [("test", "train"), ("validate", "train")])
def test_slid_windows_stay_inside_the_split(world, split, other):
    # A slide pairs one detector's data with another's from elsewhere in time; every
    # sample of every window must still be the scored split's, guard included.
    fixture = fx.straddle(split, other, half_width=600.0)
    slides = TimeSlides(num_slides=4, step_seconds=40.0)
    with world(fixture):
        batches, background = slid_windows(fx.ifo_data_obtainer(fixture), split, slides, 64, 10_000)
        assert batches, "the background served no windows"
        seen_slides = set()
        for batch in batches:
            spans = as_seconds(footprints(batch.onsource, batch.offsource, IFOS, fixture.encoding,
                                          GEOMETRY, onsource_cropped=False), fixture.encoding)
            for example in spans:
                for start, end in example:
                    assert fx.golden_split(fixture.origin, start, end) == split
            # Detector j is slid by j times the slide's step, never by zero.
            lag = spans[:, 1, 0] - spans[:, 0, 0]
            np.testing.assert_array_equal(lag, batch.stream.slide * slides.step_seconds)
            assert batch.stream.slide >= 1
            seen_slides.add(batch.stream.slide)
            # The scored windows tile the time: one onsource apart.
            np.testing.assert_array_equal(np.diff(spans[:, 0, 0]), WINDOW.onsource_duration_seconds)
            np.testing.assert_allclose(batch.start_gps, spans[:, :, 0] + WINDOW.offsource_duration_seconds
                                       + WINDOW.crop_duration_seconds)
        assert seen_slides == {1, 2, 3, 4}
        # The split's whole admissible time, less the guard, is the one stretch.
        (stretch,) = background.stretches
        assert fx.golden_split(fixture.origin, *stretch) == split


def test_slides_refuse_time_outside_the_split_the_data_is_certified_by(world):
    # The stretches come from the obtainer's admissible time; they are checked against
    # the split the validator certified (its one computation), never one read again,
    # so time another split owns is refused before any of it is read.
    fixture = fx.straddle("test", "train", half_width=600.0)
    other = gf.SplitSpec(version="fold-2")
    (stretch,) = gf.DEFAULT_SPLIT.admissible(fixture.segments["H1"], "test")
    assert other.assign(*stretch) != "test"
    with world(fixture), pytest.raises(ValueError, match="is not inside split 'test'"):
        slid_windows(fx.ifo_data_obtainer(fixture), "test", TimeSlides(num_slides=2, step_seconds=50.0),
                     32, 100, split_spec=other)


def test_each_slide_counts_its_own_livetime(world):
    # A slide of k steps loses k steps of each stretch: its livetime is its own windows'.
    fixture = fx.straddle("test", "train", half_width=600.0)
    slides = TimeSlides(num_slides=3, step_seconds=50.0)
    with world(fixture):
        batches, background = slid_windows(fx.ifo_data_obtainer(fixture), "test", slides, 32, 10_000)
    streams = {}
    for batch in batches:
        streams.setdefault(batch.stream, 0)
        streams[batch.stream] += len(batch.onsource)
    (stretch,) = background.stretches
    for stream, served in streams.items():
        assert served == stream.num_windows == expected_windows(stretch, stream.slide, 50.0) > 0
    assert sorted(stream.slide for stream in streams) == [1, 2, 3]


def test_the_window_budget_ends_a_stream_short_and_counts_it(world):
    fixture = fx.straddle("test", "train", half_width=600.0)
    slides = TimeSlides(num_slides=3, step_seconds=50.0)
    with world(fixture):
        batches, _ = slid_windows(fx.ifo_data_obtainer(fixture), "test", slides, 32, 700)
    served = {}
    for batch in batches:
        served[batch.stream] = served.get(batch.stream, 0) + len(batch.onsource)
    assert sum(served.values()) == 700
    assert all(count == stream.num_windows for stream, count in served.items())


def logit_model(detectors: int, samples: int) -> keras.Model:
    keras.utils.set_random_seed(1)
    inputs = keras.Input(shape=(detectors, samples), name="WHITENED_ONSOURCE")
    outputs = keras.layers.Dense(2, name="output")(keras.layers.Flatten()(inputs))
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


@pytest.mark.global_rng
def test_the_validator_backgrounds_real_network_noise_by_time_slides(world):
    # The validator scored real noise as independent windows of each detector, so its
    # FAR had no slides, no clustering and no livetime per slide.
    fixture = fx.straddle("test", "train", half_width=600.0)
    (stretch,) = gf.DEFAULT_SPLIT.admissible(fixture.segments["H1"], "test")
    first = expected_windows(stretch, 1, 50.0)
    budget = first + 100  # all of slide 1, and the start of slide 2
    config = gf.ValidationConfig(num_examples=budget, batch_size=64, num_time_slides=3,
                                 time_slide_step_seconds=50.0)
    with world(fixture):
        args = dict(
            noise_obtainer=fx.noise_obtainer(fixture),
            ifos=[gf.IFO[name] for name in IFOS], sample_rate_hertz=RATE,
            onsource_duration_seconds=fx.ONSOURCE_SECONDS, crop_duration_seconds=fx.CROP_SECONDS,
            offsource_duration_seconds=fx.OFFSOURCE_SECONDS,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        )
        bank = ValidationBank(logit_model(2, int(RATE)), args, config)
        bank.generate_noise()
    background = bank.background()
    # Slides 1 and 2 of the one stretch, tiled: the livetime is the windows' span.
    assert [stream.scores.size for stream in background.streams] == [first, 100]
    assert background.num_windows == budget
    assert background.livetime_seconds == budget * fx.ONSOURCE_SECONDS
    # Slides of one stretch share its data: one correlated block.
    assert len({stream.block for stream in background.streams}) == 1
    # Adjacent windows above a threshold are one trigger.
    assert background.cluster_seconds == fx.ONSOURCE_SECONDS
    assert np.all(np.isfinite(bank.far_scores)) and bank.far_scores.shape == (budget,)
