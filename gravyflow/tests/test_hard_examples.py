"""
Hard examples (gf.HardExampleGenerator): each violation is what it claims, and
the examples are the generator's in everything else.
"""

import dataclasses
import hashlib
import itertools
import pickle

import numpy as np
import pytest
import scipy.optimize

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C, ExampleNetworks
from gravyflow.src.dataset.features.hard_examples import _MismatchedStreams
from gravyflow.src.dataset.features.injection import SIGNAL_EXTENT
from gravyflow.src.rng.consumers import check_consumer
from gravyflow.tests import test_injection
from gravyflow.tests._helpers import representation_tolerance

RV = gf.ReturnVariables
SAMPLE_RATE = 512.0
BATCH = 8
TARGET_SNR = 12.0


def constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def uniform(low, high, dtype=float):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM, dtype=dtype)


def wnb(**kwargs):
    return gf.WNBGenerator(**{
        "scaling_method": gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR),
        "duration_seconds": constant(0.1), "min_frequency_hertz": constant(40.0),
        "max_frequency_hertz": constant(200.0), **kwargs})


def ringdown(**kwargs):
    return gf.RingdownGenerator(**{
        "scaling_method": gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR),
        "frequency_hertz": constant(100.0), "damping_time_seconds": constant(0.02), **kwargs})


def binary(**kwargs):
    return gf.CBCGenerator(**{
        "scaling_method": gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR),
        "mass_1_msun": uniform(20.0, 40.0), "mass_2_msun": uniform(20.0, 40.0), **kwargs})


def dataset(generators, ifos=(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), seed=3, outputs=(), group="train"):
    return gf.Dataset(
        seed=seed, group=group,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED),
        ifos=list(ifos) if isinstance(ifos, tuple) else ifos,
        waveform_generators=generators,
        sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
        num_examples_per_batch=BATCH, input_variables=[RV.ONSOURCE, RV.DETECTOR_MASK],
        output_variables=[RV.INJECTIONS, RV.DETECTOR_SNR, RV.CENTRAL_TIME, RV.VIOLATION, gf.ScalingTypes.SNR,
                          *outputs],
    )


#: Per-example networks: two bundled detectors and two drawn slots, at least three of them present.
DETECTORS = gf.Detectors(
    [gf.IFO.H1, gf.IFO.L1, gf.DetectorSlot("D0", gf.IFO.H1), gf.DetectorSlot("D1", gf.IFO.L1)],
    num_present=uniform(3, 4, int))

#: Representation: float32 rounding of the SNR's sum of N squares, twice (scale, measure).
SNR_TOLERANCE = representation_tolerance(np.float32, 1.0, 2 * np.sqrt(SAMPLE_RATE))


def test_the_violations_draw_from_registered_streams():
    for name in ("inject/0:HardExampleGenerator/violation/detector", "inject/hard/violation/partner",
                 "inject/hard/violation/sign", "inject/hard/violation/margin",
                 "wave/hard/violation/correlation", "wave/hard/comp/2/param/mass_1_msun", "wave/hard/comp/0/noise"):
        assert check_consumer(f"dataset/train/{name}")
    with pytest.raises(ValueError, match="not a registered"):
        check_consumer("inject/hard/violation/other")


def test_a_real_signal_is_labelled_none_and_each_hard_example_its_violation():
    violations = [gf.WrongDelay(0.01), gf.MismatchedWaveforms(("noise",)), gf.TimeReversed()]
    generators = [wnb(name="real")] + [
        gf.HardExampleGenerator(binary() if isinstance(violation, gf.TimeReversed) else wnb(), violation,
                                name=f"hard{j}") for j, violation in enumerate(violations)]
    _, y = dataset(generators)[0]
    expected = [gf.ViolationCode.NONE] + [violation.code for violation in violations]
    np.testing.assert_array_equal(np.asarray(y["VIOLATION"]), np.broadcast_to(np.float32(expected), (BATCH, 4)))
    assert {code.name: int(code) for code in gf.ViolationCode} == {
        "NONE": 0, "WRONG_DELAY": 1, "MISMATCHED_WAVEFORMS": 2, "TIME_REVERSED": 3}
    # Every one is scaled to the network SNR target, over the detectors present.
    snr, each = np.asarray(y["SNR"], np.float64), np.asarray(y["DETECTOR_SNR"], np.float64)
    np.testing.assert_allclose(snr, TARGET_SNR, rtol=SNR_TOLERANCE)
    np.testing.assert_allclose(np.sqrt(np.sum(each ** 2, axis=-1)), snr, rtol=SNR_TOLERANCE)


def test_a_hard_example_draws_as_its_generator_under_its_own_name():
    """Reversed in time about its reference sample, a hard example's waveform is the generator's of its name."""
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/hard")
    base = binary(name="hard")
    reversed_ = gf.HardExampleGenerator(binary(), gf.TimeReversed(), name="hard")
    waveforms, parameters = base.generate(BATCH, SAMPLE_RATE, 2.0, streams)
    reflected, reflected_parameters = reversed_.generate(BATCH, SAMPLE_RATE, 2.0, streams)
    waveforms, reflected = np.asarray(waveforms), np.asarray(reflected)
    num_samples = waveforms.shape[-1]
    centre = num_samples // 2
    index = np.arange(1, num_samples)
    np.testing.assert_array_equal(reflected[..., index], waveforms[..., 2 * centre - index])
    np.testing.assert_array_equal(reflected[..., 0], 0.0)
    for key, value in parameters.items():
        np.testing.assert_array_equal(np.asarray(reflected_parameters[key]), np.asarray(value), err_msg=str(key))
    # The signal's extent is reversed with it.
    before, after = reversed_.get_signal_extent_seconds(reflected_parameters)
    expected = np.broadcast_to(np.float64(base.get_signal_extent_seconds(parameters)[::-1]).T, (BATCH, 2)).T
    np.testing.assert_array_equal(np.stack([before, after]), expected)


def test_a_time_reversed_chirp_rises_from_its_merger():
    """A CBC's frequency rises into its merger; reversed, it falls from it, with the merger at the same sample."""
    cbc = dict(mass_1_msun=constant(30.0), mass_2_msun=constant(30.0),
               scaling_method=gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR))
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/chirp")
    forward, _ = gf.CBCGenerator(**cbc, name="chirp").generate(1, 2048.0, 2.0, streams)
    backward, _ = gf.HardExampleGenerator(gf.CBCGenerator(**cbc), gf.TimeReversed(), name="chirp").generate(
        1, 2048.0, 2.0, streams)
    forward, backward = (np.asarray(w)[0, 0] for w in (forward, backward))
    centre = forward.shape[-1] // 2

    def zero_crossings(strain):
        return np.sum(np.abs(np.diff(np.sign(strain))) > 0)

    # Within 0.2 s of the merger, the inspiral side crosses zero more slowly than the side nearest the merger.
    width = int(0.1 * 2048)
    assert zero_crossings(forward[centre - 2 * width:centre - width]) < zero_crossings(forward[centre - width:centre])
    assert zero_crossings(backward[centre + width:centre + 2 * width]) < zero_crossings(backward[centre:centre + width])
    assert np.argmax(np.abs(backward)) == 2 * centre - np.argmax(np.abs(forward))


@pytest.mark.parametrize("ifos", [(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), DETECTORS], ids=["fixed", "per example"])
def test_a_wrong_delay_exceeds_its_pairs_light_travel_time_by_its_margin(ifos):
    margin = uniform(0.005, 0.02)
    # Plus polarisation alone: each detector's copy is the one waveform, scaled and delayed.
    generator = gf.HardExampleGenerator(wnb(polarization_eccentricity=constant(1.0)), gf.WrongDelay(margin), name="late")
    ds = dataset([generator], ifos=ifos)
    for k in range(3):
        x, y = ds[k]
        present = np.asarray(x["DETECTOR_MASK"]) > 0
        times = np.asarray(y["CENTRAL_TIME"], np.float64)[:, 0] * 1.0  # The window is 1 s.
        networks = ds._example_networks(gf.SeedTree(3).batch("train", 0, k)) if isinstance(ifos, gf.Detectors) \
            else gf.ExampleNetworks.of_network(gf.Network(list(ifos)), BATCH)
        location = np.asarray(networks.location, np.float64)
        np.testing.assert_array_equal(np.asarray(networks.present) > 0, present)
        for b in range(BATCH):
            indices = np.flatnonzero(present[b])
            excess = {(i, j): abs(times[b, i] - times[b, j]) - np.linalg.norm(location[b, i] - location[b, j]) / C
                      for i in indices for j in indices if i < j}
            # Every pair beyond light travel has the one shifted detector, and one of them is beyond it by
            # the margin; the others are within it. Representation: float32 times of up to 1 s.
            beyond = {pair: value for pair, value in excess.items() if value > 1e-6}
            assert len(set.intersection(*(set(pair) for pair in beyond))) >= 1, (k, b, excess)
            assert any(0.005 - 1e-6 <= value <= 0.02 + 1e-6 for value in beyond.values()), (k, b, excess)
        # Each copy is the signal itself, where its label says: the lag at which two detectors' copies
        # correlate best is their central times' difference, to a sample.
        injections = np.asarray(y["INJECTIONS"], np.float64)[:, 0]
        num_samples = injections.shape[-1]
        spectra = np.fft.rfft(injections, n=2 * num_samples)
        for b in range(BATCH):
            indices = np.flatnonzero(present[b])
            for i in indices:
                for j in indices[indices > i]:
                    # The correlation's envelope (its analytic signal's magnitude), whose peak no
                    # carrier cycle moves.
                    analytic = np.zeros(2 * num_samples, complex)
                    analytic[:num_samples + 1] = spectra[b, i] * np.conj(spectra[b, j])
                    analytic[1:num_samples] *= 2.0
                    lag = np.argmax(np.abs(np.fft.ifft(analytic)))
                    lag = lag - 2 * num_samples if lag >= num_samples else lag
                    assert abs(lag / SAMPLE_RATE - (times[b, i] - times[b, j])) <= 1.0 / SAMPLE_RATE, (k, b, i, j)
        np.testing.assert_array_less(0.0, np.where(present, np.asarray(y["DETECTOR_SNR"])[:, 0], 1.0))


def test_a_wrong_delay_is_placed_as_its_generators_signals():
    """
    A wrong delay's placement and sky are its generator's exactly: under one name, every detector's copy but the
    shifted one arrives where the generator's own signal does, to the bit, so where a signal sits in the window says
    nothing of its kind.
    """
    def central_times(generator):
        return np.asarray(dataset([generator])[0][1]["CENTRAL_TIME"])[:, 0]

    real = central_times(wnb(name="w"))
    hard = central_times(gf.HardExampleGenerator(wnb(), gf.WrongDelay(uniform(0.005, 0.02)), name="w"))
    np.testing.assert_array_equal(np.sum(real != hard, axis=-1), 1)


@pytest.mark.parametrize("padding, margin", [(None, uniform(0.3, 0.4)), (0.1, uniform(0.01, 0.05))],
                         ids=["whole window", "padded"])
def test_a_wrong_delays_copy_is_kept_where_its_generator_places_signals(padding, margin):
    """
    The copy is shifted in the drawn direction, or the other where only that keeps it where the generator's signals
    arrive in a detector: its placement bounds (its whole extent at least the detector margin from the window's
    edges, and inside the padding) widened by the detector margin; with margins that leave little room, every copy
    lies there, and holds its share of the SNR.
    """
    padded = {} if padding is None else dict(front_padding_duration_seconds=padding,
                                             back_padding_duration_seconds=padding)
    generator = gf.HardExampleGenerator(wnb(**padded), gf.WrongDelay(margin), name="far")
    ds, real = dataset([generator]), dataset([wnb(**padded, name="far")])
    detector_margin = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]).max_geocentre_delay_seconds()
    # The window is 1 s; the burst extends 0.05 s either side of its reference time; placement rounds to a sample.
    low, high = (0.05, 0.95) if padding is None else \
        (0.5 - padding - detector_margin, 0.5 + padding + detector_margin)
    for k in range(4):
        _, y = ds[k]
        times = np.asarray(y["CENTRAL_TIME"], np.float64)[:, 0]
        # The copy is the one arrival that is not the real signal's of its name.
        copies = times[times != np.asarray(real[k][1]["CENTRAL_TIME"], np.float64)[:, 0]]
        assert copies.size == BATCH
        assert np.all(copies >= low - 1.0 / SAMPLE_RATE) and np.all(copies <= high + 1.0 / SAMPLE_RATE), (k, copies)
        np.testing.assert_array_less(0.0, np.asarray(y["DETECTOR_SNR"])[:, 0])


@pytest.mark.parametrize("margin, padding", [(0.002, None), (0.01, None), (0.002, 0.1)],
                         ids=["short reach", "longer reach", "padded"])
def test_a_wrong_delays_copy_falls_where_real_arrivals_do(margin, padding):
    """
    A copy's reference time is kept where a real signal's arrival in a detector falls: the generator's placement
    bounds widened by the detector margin, which keeps its whole signal in the window. So it reaches the margin's
    band beyond the bounds as real arrivals do, and a reach shorter than the detector margin, which the bounds alone
    cannot hold shifted either way from an arrival in that band, never leaves it.
    """
    num_examples, window = 50000, 1.0
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    padded = {} if padding is None else dict(front_padding_duration_seconds=padding,
                                             back_padding_duration_seconds=padding)
    hard = gf.HardExampleGenerator(wnb(network=network, **padded), gf.WrongDelay(margin))
    hard.check_window(window)
    rng = np.random.default_rng(1)
    # The burst extends 0.05 s either side of its reference time.
    extent = np.tile([0.05, 0.05], (num_examples, 1))
    low, high, _ = hard.placement_bounds_seconds({SIGNAL_EXTENT: extent}, window, num_examples)
    offsets = low + (high - low) * rng.random(num_examples)
    delays = np.asarray(network.get_time_delay(
        np.float32(2.0 * np.pi * rng.random(num_examples)),
        np.float32(np.arcsin(2.0 * rng.random(num_examples) - 1.0))), np.float64)
    shifts = hard.draw_projection(
        network, delays, gf.SeedTree(5).batch("train", 0, 0), "hard", num_examples,
        offsets_seconds=offsets, extent_seconds=extent, window_seconds=window)
    real = (offsets[:, None] + delays).ravel()
    copies = np.sum((offsets[:, None] + delays + shifts) * (shifts != 0.0), axis=-1)
    margin_seconds = hard.detector_margin_seconds()
    reach_low, reach_high = low[0] - margin_seconds, high[0] + margin_seconds
    assert reach_low <= real.min() and real.max() <= reach_high
    # Float32 delays: the copy lies its partner's float32 arrival plus a float64 shift.
    resolution = 1.0e-6
    assert np.all(copies >= reach_low - resolution) and np.all(copies <= reach_high + resolution), \
        (np.min(copies) - reach_low, reach_high - np.max(copies))
    assert np.any(copies < low[0]) and np.any(copies > high[0])


def test_a_window_that_cannot_hold_a_wrong_delays_copy_is_refused():
    """
    A margin whose copy, shifted either way from its partner, could leave where its generator's signals arrive is
    refused by name when the dataset is built: twice the longest shift (the network's longest light travel plus the
    largest margin) must fit the arrival bounds of the generator's longest signal (its placement bounds, the window
    less that signal and the detector margin either side, inside the padding, or the padding alone if it allows
    partial signals, widened by the detector margin). With H1, L1 and V1 (light travel up to 27 ms, a detector
    margin of 21 ms) and a 0.1 s burst in a 1 s window, that is 0.9 s, 0.24 s with 0.1 s of padding and 1.04 s
    for partial signals.
    """
    padded = dict(front_padding_duration_seconds=0.1, back_padding_duration_seconds=0.1)
    for margin, settings in ((2.0, {}), (uniform(0.3, 0.5), {}), (0.45, {}), (0.43, {}), (0.1, padded),
                             (uniform(0.3, 0.5), dict(allow_partially_out_of_window=True))):
        with pytest.raises(ValueError, match="WrongDelay hard example shifts its copy"):
            dataset([gf.HardExampleGenerator(wnb(**settings), gf.WrongDelay(margin))])
    dataset([gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.41))])
    dataset([gf.HardExampleGenerator(wnb(allow_partially_out_of_window=True), gf.WrongDelay(0.45))])
    # A partner's arrival can lie the detector margin beyond the padding, and its copy too.
    dataset([gf.HardExampleGenerator(wnb(**padded), gf.WrongDelay(0.08))])


def test_a_hard_examples_reference_times_fall_where_its_generators_do():
    """A wrong delay's copy is kept where its generator places signals, so no reference time falls elsewhere."""
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    padded = dict(front_padding_duration_seconds=0.1, back_padding_duration_seconds=0.1)
    for violation in (gf.WrongDelay(uniform(0.01, 0.03)), gf.MismatchedWaveforms("noise")):
        real = wnb(network=network, **padded)
        hard = gf.HardExampleGenerator(wnb(network=network, **padded), violation)
        assert hard.reference_time_window_seconds(1.0) == real.reference_time_window_seconds(1.0)
        assert hard.detector_margin_seconds() == real.detector_margin_seconds()


def test_mismatched_waveforms_differ_in_the_named_draws_alone():
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/odd")
    content = gf.HardExampleGenerator(
        wnb(network=network, duration_seconds=uniform(0.05, 0.2)), gf.MismatchedWaveforms(("noise",)), name="odd")
    waveforms, parameters = content.generate(BATCH, SAMPLE_RATE, 2.0, streams)
    waveforms = np.asarray(waveforms)
    assert waveforms.shape[:3] == (BATCH, 3, 2)
    assert not np.allclose(waveforms[:, 0], waveforms[:, 1])
    # Shared: the duration, and so each detector's burst lies under one envelope.
    assert np.all(np.isfinite(np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS])))
    support = np.abs(waveforms).max(axis=(0, 2)) > 0
    assert np.all(support == support[0])
    half = np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS], np.float64) / 2.0
    np.testing.assert_array_equal(np.asarray(parameters[SIGNAL_EXTENT]), np.stack([half, half], axis=-1))


def test_a_mismatch_is_refused_where_it_moves_its_signals_extent():
    """
    A mismatched example is placed once for all its detectors, so a draw that moves its signal's extent (a burst's
    length) would place a detector's shorter burst as the longest is placed: never as near the window's edge as a
    real burst of its length. A mismatch in such a draw is refused by name when it is built (a draw of one value
    moves nothing); one in the shape draws that leave the extent alone is taken.
    """
    for generator, differ in (
            (wnb(duration_seconds=uniform(0.05, 0.2)), ("duration_seconds", "noise")),
            (wnb(min_frequency_hertz=uniform(40.0, 60.0)), ("min_frequency_hertz", "noise")),
            (wnb(max_frequency_hertz=uniform(150.0, 200.0)), ("max_frequency_hertz",)),
            (gf.SineGaussianGenerator(frequency_hertz=uniform(80.0, 120.0)), ("frequency_hertz",)),
            (gf.SineGaussianGenerator(quality_factor=uniform(5.0, 10.0)), ("quality_factor",)),
            (gf.ChirpletGenerator(duration_seconds=uniform(0.2, 0.5)), ("duration_seconds",)),
            (gf.RingdownGenerator(damping_time_seconds=uniform(0.01, 0.05)), ("damping_time_seconds",)),
            (gf.PeriodicWaveGenerator(duration_seconds=uniform(0.2, 0.5)), ("frequency_hertz", "duration_seconds"))):
        with pytest.raises(ValueError, match="moves its signal's extent"):
            gf.HardExampleGenerator(generator, gf.MismatchedWaveforms(differ))
    for generator, differ in (
            (wnb(duration_seconds=uniform(0.05, 0.2)), ("noise",)),
            (wnb(), ("duration_seconds", "noise")),
            (gf.ChirpletGenerator(start_frequency_hertz=uniform(40.0, 60.0), duration_seconds=uniform(0.2, 0.5)),
             ("start_frequency_hertz",)),
            (gf.RingdownGenerator(frequency_hertz=uniform(100.0, 200.0), damping_time_seconds=uniform(0.01, 0.05)),
             ("frequency_hertz",)),
            (gf.PeriodicWaveGenerator(frequency_hertz=uniform(80.0, 120.0)), ("frequency_hertz",)),
            (gf.CBCGenerator(mass_1_msun=uniform(15.0, 80.0)), ("mass_1_msun",))):
        gf.HardExampleGenerator(generator, gf.MismatchedWaveforms(differ))
    assert gf.WaveformGenerator.extent_attributes == ()


#: Each generator with every draw that changes its shape drawn from a range; a white-noise burst's band edges overlap,
#: so its floor on duration x bandwidth draws some bursts again.
EXTENTS = {
    gf.SineGaussianGenerator: dict(frequency_hertz=uniform(80.0, 120.0), quality_factor=uniform(5.0, 10.0)),
    gf.ChirpletGenerator: dict(start_frequency_hertz=uniform(40.0, 60.0), end_frequency_hertz=uniform(100.0, 150.0),
                               duration_seconds=uniform(0.2, 0.5)),
    gf.RingdownGenerator: dict(frequency_hertz=uniform(100.0, 200.0), damping_time_seconds=uniform(0.01, 0.05)),
    gf.PeriodicWaveGenerator: dict(frequency_hertz=uniform(80.0, 120.0), duration_seconds=uniform(0.2, 0.5)),
    gf.WNBGenerator: dict(duration_seconds=uniform(0.02, 0.3), min_frequency_hertz=uniform(40.0, 120.0),
                          max_frequency_hertz=uniform(100.0, 200.0)),
    gf.CBCGenerator: dict(mass_1_msun=uniform(15.0, 80.0), mass_2_msun=uniform(15.0, 80.0),
                          spin_1_in=(constant(0.0), constant(0.0), uniform(-0.5, 0.5)),
                          spin_2_in=(constant(0.0), constant(0.0), uniform(-0.5, 0.5))),
}


@pytest.mark.parametrize("generator", EXTENTS, ids=lambda generator: generator.__name__)
def test_a_generators_extent_attributes_are_the_draws_that_move_its_extent(generator):
    """
    Measured on the mismatch itself, its refusal when built passed by: a detector's own draw of one of a generator's
    ``extent_attributes`` gives its signal an extent of its own (a white-noise burst's own band edges, its duration
    drawn again with them), which the mismatch refuses by name as it draws; every other shape draw leaves every
    detector's extent one, the one it records.
    """
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/odd")
    made = generator(**EXTENTS[generator])
    assert set(generator.extent_attributes) <= set(generator.shape_attributes)
    for draw in generator.shape_attributes:
        violation = gf.MismatchedWaveforms((draw,))
        if draw in generator.extent_attributes:
            with pytest.raises(ValueError, match="extend differently"):
                violation.generate(made, network, 16, SAMPLE_RATE, 2.0, streams)
            continue
        _, parameters = violation.generate(made, network, 16, SAMPLE_RATE, 2.0, streams)
        _, real = made.generate(16, SAMPLE_RATE, 2.0, streams)
        extent = np.broadcast_to(np.stack(
            [np.asarray(side, np.float64) for side in made.get_signal_extent_seconds(real)], axis=-1), (16, 2))
        np.testing.assert_array_equal(np.asarray(parameters[SIGNAL_EXTENT]), extent, err_msg=draw)


def test_a_mismatched_binarys_detectors_carry_one_sources_loudness():
    """
    Two co-located, co-aligned detectors receive one real signal at one SNR. Each detector's chirp of a mismatched
    binary (masses of its own) is balanced to one source's noise-weighted loudness, so the two still receive it at
    one SNR: its SNR in each detector differs from a real signal's only through the antenna pattern, never through
    its masses' loudness. H1 beside them gives the network the second independent response a mismatch needs.
    """
    twins = [*(gf.DetectorSlot.at(gf.Site.I1, gf.IFO.H1, name=name) for name in ("A", "B")), gf.IFO.H1]
    binary = dict(mass_1_msun=uniform(15.0, 80.0), mass_2_msun=uniform(15.0, 80.0),
                  scaling_method=gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR))
    spreads = {}
    for kind, generator in (
            ("real", gf.CBCGenerator(**binary, name="b")),
            ("mismatched", gf.HardExampleGenerator(
                gf.CBCGenerator(**binary), gf.MismatchedWaveforms(("mass_1_msun", "mass_2_msun")), name="b"))):
        each = np.asarray(dataset([generator], ifos=twins)[0][1]["DETECTOR_SNR"], np.float64)[:, 0]
        spreads[kind] = np.abs(np.log(each[:, 0] / each[:, 1]))
    # Each detector's spectrum is estimated from its own 4 s of noise, so even a real signal's two SNRs differ, by
    # up to about 10 %; a binary's masses alone change its loudness by up to a factor of four here.
    for spread in spreads.values():
        assert np.median(spread) < 0.05 and np.max(spread) < 0.15, spreads


def test_a_mismatch_is_scaled_to_a_network_snr():
    for type_ in (gf.ScalingTypes.HRSS, gf.ScalingTypes.HPEAK):
        with pytest.raises(ValueError, match="network SNR"):
            gf.HardExampleGenerator(wnb(scaling_method=gf.ScalingMethod(constant(1e-21), type_)),
                                    gf.MismatchedWaveforms("noise"))
    distance = gf.ScalingMethod(constant(500.0), gf.WaveformParameters.DISTANCE_MPC)
    with pytest.raises(ValueError, match="network SNR"):
        gf.HardExampleGenerator(gf.CBCGenerator(mass_1_msun=uniform(20.0, 40.0), scaling_method=distance),
                                gf.MismatchedWaveforms("mass_1_msun"))
    gf.HardExampleGenerator(wnb(scaling_method=gf.ScalingMethod(constant(1e-21), gf.ScalingTypes.HRSS)),
                            gf.WrongDelay(0.01))


def test_mismatched_waveforms_correlate_as_asked():
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/odd")

    def generated(correlation):
        generator = gf.HardExampleGenerator(
            wnb(network=network), gf.MismatchedWaveforms("noise", correlation=correlation), name="odd")
        return np.asarray(generator.generate(64, SAMPLE_RATE, 2.0, streams)[0], np.float64)

    half = generated(0.5)
    rho = np.sum(half[:, 0] * half[:, 1]) / np.sqrt(np.sum(half[:, 0] ** 2) * np.sum(half[:, 1] ** 2))
    # Statistical: 64 bursts x 2 polarisations of about 0.1 s x 160 Hz, so ~2000 degrees of freedom.
    assert abs(rho - 0.5) < 0.1


def test_a_hard_examples_fingerprint_is_its_generator_and_violation():
    from gravyflow.src.utils.fingerprint import canonical
    prints = {repr(canonical(gf.HardExampleGenerator(base, violation))) for base, violation in (
        (wnb(), gf.WrongDelay(0.01)), (wnb(), gf.WrongDelay(0.02)), (wnb(), gf.MismatchedWaveforms("noise")),
        (wnb(duration_seconds=constant(0.2)), gf.MismatchedWaveforms("noise")))}
    assert len(prints) == 4
    assert canonical(gf.HardExampleGenerator(binary(), gf.TimeReversed())) == \
        canonical(gf.HardExampleGenerator(binary(), gf.TimeReversed()))


@pytest.mark.parametrize("setting", sorted(test_injection.CHANGED_SETTINGS))
def test_a_hard_examples_fingerprint_reads_its_own_settings(setting):
    """
    A hard example's settings, its generator's when it is built, are its own, and the injection layer reads them
    from it: one changed after it is built moves its fingerprint, as it moves its batches.
    """
    from gravyflow.src.utils.fingerprint import config_fingerprint
    changed = gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.01))
    setattr(changed, setting, test_injection.CHANGED_SETTINGS[setting]())
    assert config_fingerprint(changed) != config_fingerprint(gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.01)))


def test_a_hard_examples_injection_chance_is_its_own():
    """
    Its injection chance decides which examples hold a signal, as for any generator, whatever its generator's
    is: set to 1 after it is built (as the validator sets it), every example holds one.
    """
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/hard")
    hard = gf.HardExampleGenerator(wnb(injection_chance=0.5), gf.WrongDelay(0.01), name="hard")
    halved = np.asarray(hard.generate(64, SAMPLE_RATE, 1.0, streams)[0])
    assert 0 < np.sum(np.all(halved == 0.0, axis=(1, 2))) < 64
    # The presence is drawn as its generator's would be, from the hard example's own chance stream.
    np.testing.assert_array_equal(halved, np.asarray(
        wnb(injection_chance=0.5, name="hard").generate(64, SAMPLE_RATE, 1.0, streams)[0]))
    hard.injection_chance = 1.0
    every = np.asarray(hard.generate(64, SAMPLE_RATE, 1.0, streams)[0])
    assert not np.any(np.all(every == 0.0, axis=(1, 2)))


def test_a_wrong_delays_margin_lies_above_the_delays_resolution():
    """
    A margin that can be zero, or below what float32 delays resolve, gives delays a real source makes: refused,
    so every wrong delay exceeds its pair's light travel.
    """
    from gravyflow.src.dataset.features.hard_examples import MIN_MARGIN_SECONDS
    for margin in (0.0, uniform(0.0, 0.03), -0.01, 0.5 * MIN_MARGIN_SECONDS):
        with pytest.raises(ValueError, match="bounded and above"):
            gf.WrongDelay(margin)
    gf.WrongDelay(uniform(2.0 * MIN_MARGIN_SECONDS, 0.03))


def test_a_mismatch_correlates_only_its_content():
    """
    A correlation mixes in the shared waveform, drawn with nothing differing: a burst of the same parameters where
    only its content (its noise) differs, so the mix lies inside the detectors' recorded extent. Where a parameter
    differs, the shared waveform's own draw gives it another shape (two waves' or two binaries' chirps beat), so a
    correlation is refused by name.
    """
    for generator, differ in ((wnb(duration_seconds=uniform(0.05, 0.4)), ("duration_seconds",)),
                              (wnb(duration_seconds=uniform(0.05, 0.4)), ("noise", "duration_seconds")),
                              (gf.CBCGenerator(mass_1_msun=uniform(20.0, 40.0)), ("mass_1_msun",))):
        with pytest.raises(ValueError, match="correlation mixes in the shared waveform"):
            gf.HardExampleGenerator(generator, gf.MismatchedWaveforms(differ, correlation=0.5))
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    hard = gf.HardExampleGenerator(wnb(network=network, duration_seconds=uniform(0.05, 0.4)),
                                   gf.MismatchedWaveforms("noise", correlation=0.5), name="mixed")
    waveforms, parameters = hard.generate(64, SAMPLE_RATE, 2.0, gf.SeedTree(2).batch("train", 0, 0).sub("wave/mixed"))
    waveforms = np.asarray(waveforms)
    before, after = hard.get_signal_extent_seconds(parameters)
    time = (np.arange(waveforms.shape[-1]) - waveforms.shape[-1] // 2) / SAMPLE_RATE
    support = np.any(waveforms != 0.0, axis=(1, 2))
    # Two samples for the burst's taper and its reference sample's rounding.
    outside = (time[None] < -before[:, None] - 2.0 / SAMPLE_RATE) | (time[None] > after[:, None] + 2.0 / SAMPLE_RATE)
    assert not np.any(support & outside)


def test_a_mismatched_correlation_is_finite_and_below_one():
    """A NaN correlation is refused by name, never served as a batch of NaN; rho = 1, the coherent signal, is a real one."""
    for correlation in (1.5, 1.0, uniform(0.5, 1.0), float("nan"), constant(float("nan")), uniform(0.0, float("nan"))):
        with pytest.raises(ValueError, match="correlation lies in"):
            gf.MismatchedWaveforms("noise", correlation=correlation)


def test_a_binarys_mismatch_changes_its_chirps_shape():
    """
    A binary's chirps alike but for the amplitude and the phase each detector receives one chirp with are what other
    skies and polarisations give: a binary's mismatch differs in at least one draw that changes its shape, its masses
    or spins.
    """
    cbc = gf.CBCGenerator(mass_1_msun=uniform(15.0, 40.0), mass_2_msun=uniform(15.0, 40.0),
                          reference_orbital_phase_in=uniform(0.0, 2.0 * np.pi), inclination_radians=uniform(0.0, 3.0),
                          distance_mpc=uniform(100.0, 500.0))
    for differ in (("reference_orbital_phase_in",), ("distance_mpc", "inclination_radians"),
                   ("ascending_node_longitude",)):
        with pytest.raises(ValueError, match="shape"):
            gf.HardExampleGenerator(cbc, gf.MismatchedWaveforms(differ))
    gf.HardExampleGenerator(cbc, gf.MismatchedWaveforms(("mass_1_msun", "reference_orbital_phase_in")))
    spinning = gf.CBCGenerator(spin_1_in=(constant(0.0), constant(0.0), uniform(-0.5, 0.5)))
    gf.HardExampleGenerator(spinning, gf.MismatchedWaveforms(("spin_1_in",)))


def test_a_bursts_mismatch_changes_its_shape():
    """
    A burst's amplitude, phase and polarisation eccentricity change only the amplitude and the phase each detector
    receives one burst with, as other skies and polarisations do: a mismatch in them alone is refused. A
    sine-Gaussian's every shape draw moves its envelope's length, so it takes no mismatch at all.
    """
    drawn = {"amplitude": uniform(0.5, 2.0), "phase_radians": uniform(0.0, np.pi),
             "polarization_eccentricity": uniform(0.0, 1.0)}
    for generator, shape, taken in (
            (gf.SineGaussianGenerator, {"frequency_hertz": uniform(80.0, 120.0)}, False),
            (gf.ChirpletGenerator, {"start_frequency_hertz": uniform(40.0, 60.0)}, True),
            (gf.RingdownGenerator, {"frequency_hertz": uniform(100.0, 200.0)}, True),
            (gf.PeriodicWaveGenerator, {"frequency_hertz": uniform(80.0, 120.0)}, True)):
        own = {name: value for name, value in drawn.items() if name in generator.distributed_attributes}
        burst = generator(**own, **shape)
        for name in (*own, tuple(own)):
            with pytest.raises(ValueError, match="none of them a draw that changes"):
                gf.HardExampleGenerator(burst, gf.MismatchedWaveforms(name))
        if taken:
            gf.HardExampleGenerator(burst, gf.MismatchedWaveforms((*shape, *own)))
        else:
            with pytest.raises(ValueError, match="moves its signal's extent"):
                gf.HardExampleGenerator(burst, gf.MismatchedWaveforms((*shape, *own)))
    with pytest.raises(ValueError, match="none of them a draw that changes"):
        gf.HardExampleGenerator(
            wnb(polarization_eccentricity=uniform(0.0, 1.0)), gf.MismatchedWaveforms("polarization_eccentricity"))


def test_a_mismatch_draws_more_than_one_value():
    """
    A draw that changes the shape but takes one value (a binary's masses, by default) is the same in every detector,
    which then receives the coherent, real signal: a mismatch whose shape draws each take one value is refused.
    """
    single = gf.Distribution(possible_values=[30.0], type_=gf.DistributionType.CHOICE)
    for generator, differ in (
            (gf.CBCGenerator(), ("mass_1_msun",)),
            (gf.CBCGenerator(), ("mass_1_msun", "mass_2_msun")),
            (gf.CBCGenerator(distance_mpc=uniform(100.0, 500.0)), ("mass_1_msun", "distance_mpc")),
            (gf.CBCGenerator(), ("spin_1_in",)),
            (gf.CBCGenerator(mass_1_msun=uniform(30.0, 30.0)), ("mass_1_msun",)),
            (gf.CBCGenerator(mass_1_msun=single), ("mass_1_msun",)),
            (wnb(), ("duration_seconds", "min_frequency_hertz"))):
        with pytest.raises(ValueError, match="has one value only"):
            gf.HardExampleGenerator(generator, gf.MismatchedWaveforms(differ))
    gf.HardExampleGenerator(gf.CBCGenerator(mass_2_msun=uniform(20.0, 30.0)),
                            gf.MismatchedWaveforms(("mass_1_msun", "mass_2_msun")))
    gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms(("duration_seconds", "noise")))


#: Each generator's draws: a base value of every one, then the draws that change its waveform's shape and those that
#: change only the amplitude and the phase each detector receives it with (each at another value), then those kept out
#: of the shape untested (a binary's that it refuses to vary, that its default approximant ignores, or that end its
#: chirp in band). The base eccentricity is elliptical, so the polarisations span both quadratures.
SHAPES = {
    gf.SineGaussianGenerator: (
        dict(frequency_hertz=100.0, quality_factor=10.0, amplitude=1.0, phase_radians=0.0, polarization_eccentricity=0.5),
        dict(frequency_hertz=120.0, quality_factor=5.0),
        dict(amplitude=1.7, phase_radians=0.9, polarization_eccentricity=0.2), ()),
    gf.ChirpletGenerator: (
        dict(start_frequency_hertz=50.0, end_frequency_hertz=150.0, duration_seconds=0.3, amplitude=1.0,
             phase_radians=0.0, polarization_eccentricity=0.5),
        dict(start_frequency_hertz=70.0, end_frequency_hertz=200.0, duration_seconds=0.5),
        dict(amplitude=1.7, phase_radians=0.9, polarization_eccentricity=0.2), ()),
    gf.RingdownGenerator: (
        dict(frequency_hertz=150.0, damping_time_seconds=0.02, amplitude=1.0, phase_radians=0.0,
             polarization_eccentricity=0.5),
        dict(frequency_hertz=180.0, damping_time_seconds=0.05),
        dict(amplitude=1.7, phase_radians=0.9, polarization_eccentricity=0.2), ()),
    gf.PeriodicWaveGenerator: (
        dict(frequency_hertz=100.0, duration_seconds=0.5, amplitude=1.0, phase_radians=0.0),
        dict(frequency_hertz=120.0, duration_seconds=0.3),
        dict(amplitude=1.7, phase_radians=0.9), ()),
    gf.WNBGenerator: (
        dict(duration_seconds=0.2, min_frequency_hertz=40.0, max_frequency_hertz=200.0, polarization_eccentricity=0.5),
        dict(duration_seconds=0.3, min_frequency_hertz=60.0, max_frequency_hertz=150.0, noise=None),
        dict(polarization_eccentricity=0.2), ()),
    gf.CBCGenerator: (
        dict(mass_1_msun=30.0, mass_2_msun=25.0, inclination_radians=1.0, distance_mpc=100.0,
             reference_orbital_phase_in=0.0, ascending_node_longitude=0.0),
        dict(mass_1_msun=35.0, mass_2_msun=20.0, spin_1_in=(0.0, 0.0, 0.6), spin_2_in=(0.0, 0.0, -0.6)),
        dict(inclination_radians=2.0, distance_mpc=300.0, reference_orbital_phase_in=0.9),
        ("ascending_node_longitude", "eccentricity", "mean_periastron_anomaly", "lambda_1", "lambda_2",
         "min_frequency_hertz")),
}


def _outside_span(waveforms, basis):
    """The largest share of a polarisation of ``waveforms`` (2, Time) outside the span of ``basis``'s and their quadratures."""
    from scipy.signal import hilbert
    vectors = np.concatenate([basis, np.imag(hilbert(basis, axis=-1))]).T
    shares = []
    for polarisation in waveforms:
        fit, *_ = np.linalg.lstsq(vectors, polarisation, rcond=None)
        shares.append(np.linalg.norm(polarisation - vectors @ fit) / np.linalg.norm(polarisation))
    return max(shares)


@pytest.mark.parametrize("generator", SHAPES, ids=lambda generator: generator.__name__)
def test_a_generators_shape_attributes_change_its_shape_and_no_other_draw_does(generator):
    """
    A generator's ``shape_attributes`` change its waveform beyond the span of the base waveform's polarisations and
    their quadratures, what other skies and polarisations give a detector; its other draws keep the waveform in it.
    """
    base, shape, other, untested = SHAPES[generator]
    assert set(generator.shape_attributes) == set(shape)
    assert set(shape) | set(other) | set(untested) == set(generator.distributed_attributes) | (set(shape) & {"noise"})

    def waveforms(seed=2, **changed):
        streams = gf.SeedTree(seed).batch("train", 0, 0).sub("wave/shape")
        made, _ = generator(**{**base, **changed}).generate(1, SAMPLE_RATE, 2.0, streams)
        return np.asarray(made, np.float64)[0]

    reference = waveforms()
    for name, value in shape.items():
        changed = waveforms(seed=3) if name == "noise" else waveforms(**{name: value})
        assert _outside_span(changed, reference) > 0.1, name
    for name, value in other.items():
        assert _outside_span(waveforms(**{name: value}), reference) < 1e-3, name


def test_hard_examples_refuse_what_they_cannot_be():
    with pytest.raises(ValueError, match="bounded and above"):
        gf.WrongDelay(gf.Distribution(mean=0.0, std=0.01, type_=gf.DistributionType.NORMAL))
    with pytest.raises(ValueError, match="does not draw"):
        gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms("mass_1_msun"))
    with pytest.raises(TypeError, match="wraps a coherent"):
        gf.HardExampleGenerator(gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.01)), gf.WrongDelay(0.01))
    with pytest.raises(TypeError, match="violation is a gf.Violation"):
        gf.HardExampleGenerator(wnb(), "late")
    for violation in (gf.WrongDelay(0.01), gf.MismatchedWaveforms("noise")):
        with pytest.raises(ValueError, match="needs 3 detectors present"):
            gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1, gf.IFO.L1])), violation)
        with pytest.raises(ValueError, match="needs 3 detectors present"):
            dataset([gf.HardExampleGenerator(wnb(), violation)],
                    ifos=gf.Detectors([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1], num_present=uniform(2, 3, int)))
    # Reversal needs no second detector.
    gf.HardExampleGenerator(binary(network=gf.Network([gf.IFO.H1])), gf.TimeReversed())


def _forward_and_reversed(generator, seed=2, num_waveforms=1, **parameters):
    """A generator's waveforms, (Batch, 2, Time) float64, and each reversed in time about its reference sample."""
    from gravyflow.src.dataset.features.hard_examples import _reflect
    streams = gf.SeedTree(seed).batch("train", 0, 0).sub("wave/reversed")
    made, _ = generator(**parameters).generate(num_waveforms, SAMPLE_RATE, 2.0, streams)
    return np.asarray(made, np.float64), np.asarray(_reflect(made), np.float64)


def test_a_reversed_sine_gaussian_is_a_sine_gaussian():
    """Its envelope is symmetric, so reversed it is the same wavelet at another phase and handedness."""
    forward, backward = _forward_and_reversed(
        gf.SineGaussianGenerator, frequency_hertz=100.0, quality_factor=10.0, phase_radians=0.7,
        polarization_eccentricity=0.5)
    assert _outside_span(backward[0], forward[0]) < 1e-5


def test_a_reversed_chirplet_is_a_chirplet_from_its_end_to_its_start():
    chirplet = dict(duration_seconds=0.3, phase_radians=0.7, polarization_eccentricity=0.5)
    _, backward = _forward_and_reversed(
        gf.ChirpletGenerator, start_frequency_hertz=50.0, end_frequency_hertz=150.0, **chirplet)
    swapped, _ = _forward_and_reversed(
        gf.ChirpletGenerator, start_frequency_hertz=150.0, end_frequency_hertz=50.0, **chirplet)
    assert _outside_span(backward[0], swapped[0]) < 1e-5


@pytest.mark.parametrize("shape", list(gf.WaveShape), ids=lambda shape: shape.name)
def test_a_reversed_periodic_wave_is_a_periodic_wave(shape):
    """Reversed, each wave shape is the shape at the opposite phase, or its negative (the polarisation turned by 90 degrees)."""
    wave = dict(wave_shape=shape, frequency_hertz=100.0, duration_seconds=0.3)
    _, backward = _forward_and_reversed(gf.PeriodicWaveGenerator, phase_radians=0.7, **wave)
    opposite, _ = _forward_and_reversed(gf.PeriodicWaveGenerator, phase_radians=-0.7, **wave)
    residual = min(np.linalg.norm(backward[0, 0] - sign * opposite[0, 0]) for sign in (1.0, -1.0))
    # A square or sawtooth wave's jumps land a sample apart, so a few samples differ.
    assert residual / np.linalg.norm(backward[0, 0]) < 0.1


def test_a_reversed_white_noise_burst_is_another_such_burst():
    """Reversal keeps each polarisation's amplitude spectrum, its band and power; Gaussian noise has no direction."""
    forward, backward = _forward_and_reversed(gf.WNBGenerator, duration_seconds=0.2, min_frequency_hertz=40.0,
                                              max_frequency_hertz=200.0)
    spectrum = np.abs(np.fft.rfft(forward[0], axis=-1))
    np.testing.assert_allclose(np.abs(np.fft.rfft(backward[0], axis=-1)), spectrum, atol=1e-5 * spectrum.max())


def test_time_reversal_is_refused_where_it_stays_in_its_generators_family():
    """
    A hard example is never a real signal of its own generator: time reversal is accepted only by a generator
    that declares its reversal leaves its family (a binary's chirp), and refused by name elsewhere: a ringdown too,
    whose reversed signal lies before its reference time, so it would not be placed as a ringdown's signals are.
    """
    for generator in (wnb(), gf.SineGaussianGenerator(), gf.ChirpletGenerator(), gf.RingdownGenerator(),
                      *(gf.PeriodicWaveGenerator(wave_shape=shape) for shape in gf.WaveShape)):
        with pytest.raises(ValueError, match=f"leaves its family .*not a {type(generator).__name__}"):
            gf.HardExampleGenerator(generator, gf.TimeReversed())
    gf.HardExampleGenerator(gf.CBCGenerator(), gf.TimeReversed())
    declared = {generator for generator in SHAPES if generator.time_reversal_leaves_family}
    assert declared == {gf.CBCGenerator}
    assert not gf.WaveformGenerator.time_reversal_leaves_family


def test_mismatched_waveforms_of_nothing_are_refused_by_name():
    for differ in (None, (), (1,), ("noise", None)):
        with pytest.raises(ValueError, match="at least one named draw"):
            gf.MismatchedWaveforms(differ)


def test_an_incoherent_generators_violation_is_refused():
    """A different waveform in each detector is no real signal: its violation is refused by name, never labelled NONE."""
    with pytest.raises(ValueError, match="IncoherentGenerator"):
        dataset([gf.IncoherentGenerator([wnb(), wnb(), wnb()])])[0]


def test_a_violation_acts_through_its_own_hooks():
    """
    What a violation does is its own: a wrong delay that moves the first present detector's copy by its margin, by its
    own projection alone, moves it there, with nothing in the hard example naming it.
    """
    @dataclasses.dataclass(frozen=True)
    class FirstDetectorLate(gf.WrongDelay):
        def projection(self, hard_example, network, time_delays, present, stream, num_examples, *placement):
            shifts = np.zeros(present.shape)
            shifts[np.arange(num_examples), np.argmax(present > 0, axis=1)] = self.max_margin_seconds
            return shifts

    x, real = dataset([wnb(name="w")], ifos=DETECTORS)[0]
    _, y = dataset([gf.HardExampleGenerator(wnb(), FirstDetectorLate(0.01), name="w")], ifos=DETECTORS)[0]
    present = np.asarray(x["DETECTOR_MASK"]) > 0
    first = np.arange(present.shape[-1]) == np.argmax(present, axis=1)[:, None]
    moved = np.asarray(y["CENTRAL_TIME"], np.float64)[:, 0] - np.asarray(real["CENTRAL_TIME"], np.float64)[:, 0]
    # CENTRAL_TIME is a fraction of the 1 s window, in float32.
    np.testing.assert_allclose(moved[present], np.where(first, 0.01, 0.0)[present], atol=1e-6)


#: A sky held fixed, so each example's own geometry is known: every detector's delay and antenna pattern.
SKY = dict(right_ascension_radians=1.0, declination_radians=0.3, polarization_radians=0.4)
NETWORK = (gf.IFO.H1, gf.IFO.L1, gf.IFO.V1)
SNR_SCALED = dict(scaling_method=gf.ScalingMethod(constant(TARGET_SNR), gf.ScalingTypes.SNR))

#: Each generator, the independent waveforms its polarisations carry, and the draws a mismatch of it differs in
#: (None: it takes no mismatch).
COHERENCE = {
    "binary": (lambda **kwargs: gf.CBCGenerator(
        mass_1_msun=uniform(30.0, 40.0), mass_2_msun=uniform(30.0, 40.0), min_frequency_hertz=40.0,
        inclination_radians=1.0, front_padding_duration_seconds=0.2, back_padding_duration_seconds=0.2,
        **SNR_SCALED, **kwargs), 1, ("mass_1_msun",)),
    "white-noise burst": (wnb, 2, ("noise",)),
    "linear white-noise burst": (lambda **kwargs: wnb(polarization_eccentricity=constant(1.0), **kwargs), 1, ("noise",)),
    "sine-Gaussian": (lambda **kwargs: gf.SineGaussianGenerator(
        frequency_hertz=100.0, quality_factor=5.0, polarization_eccentricity=0.5, **SNR_SCALED, **kwargs), 1, None),
    "chirplet": (lambda **kwargs: gf.ChirpletGenerator(
        start_frequency_hertz=uniform(50.0, 80.0), end_frequency_hertz=150.0, duration_seconds=0.2,
        polarization_eccentricity=0.5, **SNR_SCALED, **kwargs), 1, ("start_frequency_hertz",)),
    "ringdown": (lambda **kwargs: ringdown(frequency_hertz=uniform(100.0, 200.0), damping_time_seconds=constant(0.05),
                                           polarization_eccentricity=0.5, **kwargs), 1, ("frequency_hertz",)),
    "periodic wave": (lambda **kwargs: gf.PeriodicWaveGenerator(
        frequency_hertz=uniform(80.0, 120.0), duration_seconds=0.2, **SNR_SCALED, **kwargs), 1, ("frequency_hertz",)),
}


def _fibonacci_sky(num_points):
    """Right ascensions and declinations spread evenly over the sphere."""
    index = np.arange(num_points) + 0.5
    return np.mod(np.pi * (1.0 + np.sqrt(5.0)) * index, 2.0 * np.pi), np.arcsin(1.0 - 2.0 * index / num_points)


def _draw_spread(generator, violation, num_detectors):
    """
    Each example's widest difference between its detectors' draws of a mismatch's parameters, as a share of each
    one's range, in the first batch of the reachability test's datasets (seed 3, generator "w"): infinite for a
    content draw, a burst's noise, which no two detectors draw alike, and for any other violation, which draws none.
    """
    if not isinstance(violation, gf.MismatchedWaveforms):
        return np.full(BATCH, np.inf)
    streams = gf.SeedTree(3).batch("train", 0, 0).sub("wave/w")
    spread = np.zeros(BATCH)
    for draw in violation.differ:
        distribution = getattr(generator, draw, None)
        if not isinstance(distribution, gf.Distribution):
            return np.full(BATCH, np.inf)
        values = [distribution.sample(BATCH, _MismatchedStreams(streams, violation.differ, j).stream(f"param/{draw}"))
                  for j in range(num_detectors)]
        spread = np.maximum(spread, np.ptp(values, axis=0) / (distribution.max_ - distribution.min_))
    return spread


def _unfit_share(strains, delays, independent, antenna=None):
    """
    The share of the detectors' signal power (strains, (Detectors, Time)) that no source whose polarisations carry
    ``independent`` waveforms gives them, for each set of arrival delays (delays, (Skies, Detectors)): each detector's
    spectrum aligned by its delay, its best fit by that many free waveforms, each with a complex gain per detector
    (any polarisation angle and state), within the span of the detectors' antenna patterns (antenna, (Detectors, 2):
    F+ and Fx at one polarisation angle) where given, any gains where not.
    """
    num_samples = strains.shape[-1]
    spectra = np.fft.rfft(np.asarray(strains, np.float64), n=2 * num_samples, axis=-1)
    frequencies = np.fft.rfftfreq(2 * num_samples, 1.0 / SAMPLE_RATE)
    aligned = spectra[None] * np.exp(2j * np.pi * frequencies * np.asarray(delays, np.float64)[..., None])
    total = np.sum(np.abs(spectra) ** 2)
    if antenna is not None:
        basis, _ = np.linalg.qr(np.asarray(antenna, np.float64))
        aligned = np.einsum("ij,gjf->gif", basis @ basis.T, aligned)
    power = np.linalg.eigvalsh(aligned @ np.conj(np.swapaxes(aligned, -1, -2)))[..., ::-1]
    return np.clip(1.0 - np.sum(power[..., :independent], axis=-1) / total, 0.0, 1.0)


@pytest.mark.parametrize("name", COHERENCE)
def test_a_coherence_violation_needs_more_detectors_than_its_generators_waveforms(name):
    """
    Every detector receives a combination of the polarisations, delayed, so a network of no more detectors than the
    independent waveforms the polarisations carry (one for a binary's dominant mode or a burst whose cross
    polarisation is its plus a quarter cycle on, or none; two for a white-noise burst's independent pair) receives
    any waveforms in them: there a wrong delay or a mismatch is a real signal. Each is refused there by name; on the
    smallest network taken, no sky gives it.

    Measured by a least-squares fit of the detectors' noise-free injections by free waveforms: on the refused
    networks (the taken network's examples, fewer detectors kept), at the example's own sky, it leaves nothing; on
    the taken network it leaves a share of the power that no sky removes, where the generator's real signal leaves
    next to none at its sky. A mismatch is as far from a real signal as its detectors' draws differ, so it is measured where they
    differ by at least a twentieth of their range. A binary's merger is kept from the window's edges, which would
    cut its inspiral differently in each detector, and the ringdown rings for long enough to fit the model.
    """
    make, independent, differ = COHERENCE[name]
    generator = make(**SKY)
    assert generator.independent_waveforms == independent
    violations = [gf.WrongDelay(uniform(0.01, 0.02))] + ([] if differ is None else [gf.MismatchedWaveforms(differ)])
    for violation in violations:
        needed = violation.fewest_detectors(generator)
        assert needed == independent + 1
        with pytest.raises(ValueError, match=f"needs {needed} detectors present"):
            dataset([gf.HardExampleGenerator(make(**SKY), violation)], ifos=NETWORK[:needed - 1])
        ifos = NETWORK[:needed]
        network = gf.Network(list(ifos))
        delays = np.asarray(network.get_time_delay(
            np.float32([SKY["right_ascension_radians"]]), np.float32([SKY["declination_radians"]])), np.float64)
        antenna = np.asarray(network.get_antenna_pattern(
            np.float32([SKY["right_ascension_radians"]]), np.float32([SKY["declination_radians"]]),
            np.float32([SKY["polarization_radians"]])), np.float64)[0]
        skies = np.asarray(network.get_time_delay(*(np.float32(angle) for angle in _fibonacci_sky(4000))), np.float64)
        real = np.asarray(dataset([make(**SKY, name="w")], ifos=ifos)[0][1]["INJECTIONS"], np.float64)[:, 0]
        hard = np.asarray(dataset([gf.HardExampleGenerator(make(**SKY), violation, name="w")], ifos=ifos)[0][1][
            "INJECTIONS"], np.float64)[:, 0]
        kept = [list(rows) for rows in itertools.combinations(range(needed), needed - 1)]
        refused, taken, coherent = [], [], []
        for b in range(BATCH):
            coherent.append(_unfit_share(real[b], delays, independent, antenna)[0])
            refused.extend(_unfit_share(hard[b][rows], delays[:, rows], independent, antenna[rows])[0] for rows in kept)
            taken.append(np.min(_unfit_share(hard[b], skies, independent)))
        spread = _draw_spread(generator, violation, needed)
        apart = spread >= 0.05
        taken = np.asarray(taken)
        print(f"{name}, {type(violation).__name__} on {needed} detectors: real {max(coherent):.2e}, "
              f"refused {max(refused):.2e}, taken {min(taken[apart]):.2e} to {max(taken[apart]):.2e} "
              f"({np.sum(apart)} of {BATCH}; draws closer: {np.round(taken[~apart], 6).tolist()})")
        # A real signal is fit to within the model's quarter cycle: a binary's cross polarisation is its plus a
        # quarter cycle on to within its taper, a ringdown's to within its sudden start.
        assert max(coherent) < 1e-2 and max(refused) < 1e-9, (name, violation, coherent, refused)
        assert np.sum(apart) >= BATCH - 1, (name, violation)
        # A binary's masses drawn 0.2 % of their range apart leave 3e-5, 7 % apart 3e-4, 15 % apart 2e-3; a
        # chirplet's start frequencies 4 % apart leave 2e-4.
        assert min(taken[apart]) > 2.0 * max(coherent) and min(taken[apart]) > 1e-4, (name, violation, taken, coherent)


def _local_axes(latitude, longitude):
    """The north, east and up unit vectors at a vertex of geodetic ``latitude`` and ``longitude``, Earth-fixed."""
    north = np.array([-np.sin(latitude) * np.cos(longitude), -np.sin(latitude) * np.sin(longitude), np.cos(latitude)])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0])
    up = np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])
    return north, east, up


def antipodal_copy(ifo, name):
    """
    A fixed slot near the antipode of ``ifo``'s vertex whose arms lie along ``ifo``'s, so its response tensor is
    ``ifo``'s up to its sign: seen from above, the antipode turns the arms' order, so its x arm is ``ifo``'s y arm.
    """
    network = gf.Network([ifo])
    location = np.asarray(network.location, np.float64)[0]
    latitude = -np.arcsin(location[2] / np.linalg.norm(location))
    longitude = float(np.arctan2(location[1], location[0]) + np.pi)
    north, east, up = _local_axes(latitude, longitude)

    def pointing(arm):
        arm = np.asarray(arm, np.float64)[0]
        return float(np.arctan2(arm @ east, arm @ north)), float(np.arcsin(np.clip(arm @ up, -1.0, 1.0)))

    (x_angle, x_altitude), (y_angle, y_altitude) = pointing(network.y_vector), pointing(network.x_vector)
    return gf.DetectorSlot(name, ifo, latitude_radians=float(latitude), longitude_radians=longitude,
                           x_angle_radians=x_angle, opening_angle_radians=float(np.mod(x_angle - y_angle, 2 * np.pi)),
                           x_altitude_radians=x_altitude, y_altitude_radians=y_altitude)


#: Networks of three detectors whose responses span two dimensions: H1 with a copy of its response on the far side
#: of the Earth and L1; H1 with two slots at one site; the Einstein Telescope's triangle, whose tensors sum to zero.
DEPENDENT = {
    "antipodal H1": [gf.IFO.H1, antipodal_copy(gf.IFO.H1, "A1"), gf.IFO.L1],
    "twin slots": [gf.IFO.H1, gf.DetectorSlot.at(gf.Site.I1, gf.IFO.H1, "Ia"),
                   gf.DetectorSlot.at(gf.Site.I1, gf.IFO.H1, "Ib")],
    "Einstein Telescope": [gf.DetectorSlot.at(site, gf.IFO.V1) for site in (gf.Site.E1, gf.Site.E2, gf.Site.E3)],
}


@pytest.mark.parametrize("violation", [gf.WrongDelay(0.005), gf.MismatchedWaveforms("noise")],
                         ids=["wrong delay", "mismatch"])
@pytest.mark.parametrize("network", DEPENDENT)
def test_a_network_counts_the_rank_of_its_responses(network, violation):
    """
    A network's detectors count against its generator's independent waveforms as the rank of the span of their
    response tensors: detectors of equal, opposite or dependent responses count once, however far apart. Each of
    these networks counts two, so a white-noise burst's two independent waveforms are refused on it by name when it
    is set, and one plus polarisation, which needs two, is taken.
    """
    with pytest.raises(ValueError, match="needs 3 detectors present in every example, but its network can have 2"):
        dataset([gf.HardExampleGenerator(wnb(), violation)], ifos=DEPENDENT[network])
    dataset([gf.HardExampleGenerator(wnb(polarization_eccentricity=constant(1.0)), violation)],
            ifos=DEPENDENT[network])[0]


def test_generic_networks_count_every_detector():
    """Detectors at distinct sites count one each: H1, L1 and V1 three, two of them two."""
    for violation in (gf.WrongDelay(0.005), gf.MismatchedWaveforms("noise")):
        gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])), violation)
        with pytest.raises(ValueError, match="but its network can have 2"):
            gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1, gf.IFO.V1])), violation)


def test_a_fixed_networks_fewest_present_detectors_are_counted_by_their_worst_subset():
    """Of H1, its antipodal copy and L1, two present can be H1 and its copy, of one response: one counts."""
    ifos = gf.Detectors(DEPENDENT["antipodal H1"], num_present=uniform(2, 3, int))
    with pytest.raises(ValueError, match="needs 2 detectors present in every example, but its network can have 1"):
        dataset([gf.HardExampleGenerator(binary(), gf.WrongDelay(0.005))], ifos=ifos)
    dataset([gf.HardExampleGenerator(binary(), gf.WrongDelay(0.005))], ifos=DEPENDENT["antipodal H1"])


def test_an_example_is_counted_as_it_is_projected():
    """Where a network is drawn per example, each example's own detectors are counted as it is projected."""
    examples = ExampleNetworks.of_network(gf.Network([gf.IFO.H1, gf.IFO.L1]), BATCH)
    delays = np.zeros((BATCH, 2), np.float32)
    with pytest.raises(ValueError, match="but an example of this batch has 2"):
        gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.005)).draw_projection(
            examples, delays, gf.SeedTree(5).batch("train", 0, 0), "hard", BATCH, offsets_seconds=np.zeros(BATCH),
            extent_seconds=np.full((BATCH, 2), 0.05), window_seconds=1.0)


def test_a_precessing_binary_carries_two_independent_waveforms():
    """
    A binary whose spin can lie in its orbit's plane precesses: its cross polarisation is no quadrature of its plus,
    so it carries two independent waveforms, and a wrong delay of it on two detectors, which receive any two, is
    refused. Measured: the share of the cross polarisation's power outside the span of the plus and its quadrature.
    """
    def binary(approximant, spin):
        return gf.CBCGenerator(mass_1_msun=30.0, mass_2_msun=8.0, inclination_radians=1.0, approximant=approximant,
                               spin_1_in=spin, **SNR_SCALED)

    def unfit_cross(generator):
        waveforms, _ = generator.generate(1, 2048.0, 4.0, gf.SeedTree(2).batch("train", 0, 0).sub("wave/w"))
        plus, cross = np.asarray(waveforms, np.float64)[0]
        return _outside_span(cross[None], plus[None]) ** 2

    aligned = binary(gf.Approximant.IMRPhenomD, (0.0, 0.0, 0.3))
    precessing = binary(gf.Approximant.IMRPhenomPv2, (0.7, 0.0, 0.1))
    assert (aligned.independent_waveforms, precessing.independent_waveforms) == (1, 2)
    assert binary(gf.Approximant.IMRPhenomPv2, (0.0, 0.0, 0.3)).independent_waveforms == 1
    shares = unfit_cross(aligned), unfit_cross(precessing)
    print(f"cross polarisation unfit by the plus and its quadrature: aligned {shares[0]:.1e}, precessing {shares[1]:.2f}")
    assert shares[0] < 1e-2 < 1e-1 < shares[1]
    with pytest.raises(ValueError, match="needs 3 detectors present"):
        gf.HardExampleGenerator(binary(gf.Approximant.IMRPhenomPv2, (0.7, 0.0, 0.1)), gf.WrongDelay(0.01)).network = \
            gf.Network([gf.IFO.H1, gf.IFO.L1])


def _direct(generator, outputs, window=1.0):
    """One batch of ``generator`` through a gf.InjectionGenerator of its own, outside any dataset: its labels."""
    injector = gf.InjectionGenerator([generator])
    streams = injector.streams(0)
    injections, mask, parameters = injector.generate(
        streams, sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=window, crop_duration_seconds=0.5,
        num_examples_per_batch=BATCH)
    noise = np.random.default_rng(0)
    num_detectors = int(generator.network.num_detectors)
    onsource = noise.normal(size=(BATCH, num_detectors, int((window + 1.0) * SAMPLE_RATE))).astype(np.float32)
    offsource = noise.normal(size=(BATCH, num_detectors, int(4.0 * SAMPLE_RATE))).astype(np.float32)
    _, _, labels = injector.add_injections_to_onsource(
        injections, mask, onsource, offsource, parameters_to_return=list(outputs), onsource_duration_seconds=window,
        injection_parameters=parameters, streams=streams)
    return labels


def test_what_cannot_label_or_hold_a_violation_is_refused_outside_a_dataset():
    """
    Where the label is made and where the copy is projected, not only when a dataset is built: an
    IncoherentGenerator's VIOLATION, and a window that cannot hold a wrong delay's copy, through a
    gf.InjectionGenerator alone.
    """
    two = gf.Network([gf.IFO.H1, gf.IFO.L1])
    incoherent = gf.IncoherentGenerator([wnb(network=two), wnb(network=two)])
    _direct(incoherent, [gf.ScalingTypes.SNR])
    with pytest.raises(ValueError, match="IncoherentGenerator"):
        _direct(incoherent, [RV.VIOLATION])
    with pytest.raises(ValueError, match="WrongDelay hard example shifts its copy"):
        _direct(gf.HardExampleGenerator(ringdown(network=two), gf.WrongDelay(0.9)), [RV.VIOLATION])
    labels = _direct(gf.HardExampleGenerator(ringdown(network=two), gf.WrongDelay(0.01)), [RV.VIOLATION])
    np.testing.assert_array_equal(np.asarray(labels[RV.VIOLATION]), float(gf.ViolationCode.WRONG_DELAY))


def test_a_violation_is_labelled_by_kind_alone():
    """
    VIOLATION labels what an example is; which detector's copy of a two-detector wrong delay was moved is no label,
    since the data cannot say (one copy shifted from the other is the other shifted back), so none is given.
    """
    assert not hasattr(gf.ReturnVariables, "VIOLATION_DETECTORS")


def test_a_mismatch_differs_in_content_only_where_its_generator_draws_one():
    """'noise', a burst's random content, is a draw only of a generator that names it among its shape draws."""
    binary = gf.CBCGenerator(mass_1_msun=uniform(20.0, 40.0), **SNR_SCALED)
    with pytest.raises(ValueError, match="does not draw"):
        gf.HardExampleGenerator(binary, gf.MismatchedWaveforms(("mass_1_msun", "noise")))
    with pytest.raises(ValueError, match="correlation mixes in the shared waveform"):
        gf.HardExampleGenerator(binary, gf.MismatchedWaveforms(("mass_1_msun",), correlation=0.5))
    gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms("noise", correlation=0.5))


def test_what_cannot_label_or_hold_a_violation_is_refused_when_built():
    """
    An IncoherentGenerator's VIOLATION is refused when the dataset is built, not at its first batch; a hard example
    is no component of an IncoherentGenerator, which would drop its violation; and a wrong delay's window is refused
    when the dataset is built, its placement range the generator's.
    """
    with pytest.raises(ValueError, match="IncoherentGenerator"):
        dataset([gf.IncoherentGenerator([wnb(), wnb(), wnb()])])
    with pytest.raises(TypeError, match="hard example"):
        gf.IncoherentGenerator([gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.01)), wnb(), wnb()])
    network = gf.Network(list(NETWORK))
    hard = gf.HardExampleGenerator(wnb(network=network), gf.WrongDelay(2.0))
    assert hard.placement_range_seconds(1.0) == wnb(network=network).placement_range_seconds(1.0)
    with pytest.raises(ValueError, match="WrongDelay hard example shifts its copy"):
        dataset([hard])


#: A hard example of each violation, from the generator it suits.
STREAM_CASES = {
    "wrong delay": lambda: gf.HardExampleGenerator(wnb(), gf.WrongDelay(uniform(0.01, 0.02)), name="hard"),
    "mismatched": lambda: gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms("noise", correlation=uniform(0.2, 0.6)),
                                                  name="hard"),
    "time reversed": lambda: gf.HardExampleGenerator(binary(), gf.TimeReversed(), name="hard"),
}


def _digest(batch):
    h = hashlib.sha256()
    for part in batch:
        for key in sorted(part):
            h.update(key.encode())
            h.update(np.ascontiguousarray(np.asarray(part[key])).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("ifos", [NETWORK, DETECTORS], ids=["fixed network", "per-example network"])
@pytest.mark.parametrize("case", STREAM_CASES)
def test_a_hard_example_leaves_the_draws_beside_it_as_they_were(case, ifos):
    """
    A hard example draws from streams of its own name, so the generator beside it, the noise and the network draw what
    they draw without it, to the byte.
    """
    outputs = (RV.OFFSOURCE,)
    x_alone, y_alone = dataset([wnb(name="real")], ifos=ifos, outputs=outputs)[1]
    x_beside, y_beside = dataset([wnb(name="real"), STREAM_CASES[case]()], ifos=ifos, outputs=outputs)[1]
    np.testing.assert_array_equal(np.asarray(x_beside["DETECTOR_MASK"]), np.asarray(x_alone["DETECTOR_MASK"]))
    for key, alone in y_alone.items():
        alone, beside = np.asarray(alone), np.asarray(y_beside[key])
        if key != "OFFSOURCE":
            # The generator's own: its column of the generator axis.
            assert beside.shape[1] == 2 and alone.shape[1] == 1, key
            beside = beside[:, :1]
        assert alone.tobytes() == beside.tobytes(), key


@pytest.mark.parametrize("case", STREAM_CASES)
def test_a_hard_examples_batches_are_a_function_of_their_address(case):
    """
    A hard example's batch is the same rebuilt and pickled, and another at another index, seed or split.
    """
    reference = _digest(dataset([STREAM_CASES[case]()])[0])
    assert _digest(dataset([STREAM_CASES[case]()])[0]) == reference
    assert _digest(pickle.loads(pickle.dumps(dataset([STREAM_CASES[case]()])))[0]) == reference
    assert _digest(dataset([pickle.loads(pickle.dumps(STREAM_CASES[case]()))])[0]) == reference
    others = [dataset([STREAM_CASES[case]()])[1], dataset([STREAM_CASES[case]()], seed=4)[0],
              dataset([STREAM_CASES[case]()], group="validate")[0]]
    assert len({reference, *map(_digest, others)}) == 4
