"""
Hard examples (gf.HardExampleGenerator): each violation is what it claims, and
the examples are the generator's in everything else.
"""

import dataclasses

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C
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


def dataset(generators, ifos=(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), seed=3, outputs=()):
    return gf.Dataset(
        seed=seed, group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED),
        ifos=list(ifos) if isinstance(ifos, tuple) else ifos,
        waveform_generators=generators,
        sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
        num_examples_per_batch=BATCH, input_variables=[RV.ONSOURCE, RV.DETECTOR_MASK],
        output_variables=[RV.INJECTIONS, RV.DETECTOR_SNR, RV.CENTRAL_TIME, RV.VIOLATION, gf.ScalingTypes.SNR,
                          *outputs],
    )


#: Per-example networks: two bundled detectors and two drawn slots, two to four of them present.
DETECTORS = gf.Detectors(
    [gf.IFO.H1, gf.IFO.L1, gf.DetectorSlot("D0", gf.IFO.H1), gf.DetectorSlot("D1", gf.IFO.L1)],
    num_present=uniform(2, 4, int))

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
    violations = [gf.WrongDelay(0.01), gf.OneDetector(), gf.MismatchedWaveforms(("noise",)), gf.TimeReversed()]
    generators = [wnb(name="real")] + [
        gf.HardExampleGenerator(ringdown() if isinstance(violation, gf.TimeReversed) else wnb(), violation,
                                name=f"hard{j}") for j, violation in enumerate(violations)]
    _, y = dataset(generators)[0]
    expected = [gf.ViolationCode.NONE] + [violation.code for violation in violations]
    np.testing.assert_array_equal(np.asarray(y["VIOLATION"]), np.broadcast_to(np.float32(expected), (BATCH, 5)))
    assert [int(code) for code in expected] == [0, 1, 2, 3, 4]
    # Every one is scaled to the network SNR target, over the detectors present.
    np.testing.assert_allclose(np.asarray(y["SNR"]), TARGET_SNR, rtol=SNR_TOLERANCE)
    each = np.asarray(y["DETECTOR_SNR"], np.float64)
    np.testing.assert_allclose(np.sqrt(np.sum(each ** 2, axis=-1)), TARGET_SNR, rtol=SNR_TOLERANCE)


def test_a_hard_example_draws_as_its_generator_under_its_own_name():
    """Reversed in time about its reference sample, a hard example's waveform is the generator's of its name."""
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/hard")
    base = ringdown(name="hard")
    reversed_ = gf.HardExampleGenerator(ringdown(), gf.TimeReversed(), name="hard")
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
    # The signal's extent is reversed with it: a ringdown's, all after its peak, falls all before it.
    before, after = reversed_.get_signal_extent_seconds(reflected_parameters)
    np.testing.assert_array_equal((before, after), base.get_signal_extent_seconds(parameters)[::-1])
    assert np.all(before > 0.0) and np.all(after == 0.0)


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
    The copy is shifted in the drawn direction, or the other where only that keeps it where the generator places its
    signals' reference times: its whole extent at least the detector margin from the window's edges, as a real
    signal's reference time is, and inside the padding; with margins that leave little room, every copy lies there,
    and holds its share of the SNR.
    """
    padded = {} if padding is None else dict(front_padding_duration_seconds=padding,
                                             back_padding_duration_seconds=padding)
    generator = gf.HardExampleGenerator(wnb(**padded), gf.WrongDelay(margin), name="far")
    ds, real = dataset([generator]), dataset([wnb(**padded, name="far")])
    detector_margin = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]).max_geocentre_delay_seconds()
    # The window is 1 s; the burst extends 0.05 s either side of its reference time; placement rounds to a sample.
    low, high = (0.05 + detector_margin, 0.95 - detector_margin) if padding is None else (0.5 - padding, 0.5 + padding)
    for k in range(4):
        _, y = ds[k]
        times = np.asarray(y["CENTRAL_TIME"], np.float64)[:, 0]
        # The copy is the one arrival that is not the real signal's of its name.
        copies = times[times != np.asarray(real[k][1]["CENTRAL_TIME"], np.float64)[:, 0]]
        assert copies.size == BATCH
        assert np.all(copies >= low - 1.0 / SAMPLE_RATE) and np.all(copies <= high + 1.0 / SAMPLE_RATE), (k, copies)
        np.testing.assert_array_less(0.0, np.asarray(y["DETECTOR_SNR"])[:, 0])


def test_a_window_that_cannot_hold_a_wrong_delays_copy_is_refused():
    """
    A margin whose copy, shifted either way from its partner, could leave where its generator places signals is
    refused by name when the dataset is built: twice the longest shift (the network's longest light travel plus the
    largest margin) must fit the placement bounds of the generator's longest signal (the window less that signal
    and the detector margin either side, inside the padding; the padding alone if it allows partial signals).
    """
    padded = dict(front_padding_duration_seconds=0.1, back_padding_duration_seconds=0.1)
    for margin, settings in ((2.0, {}), (uniform(0.3, 0.5), {}), (0.45, {}), (0.41, {}), (0.08, padded),
                             (uniform(0.3, 0.5), dict(allow_partially_out_of_window=True))):
        with pytest.raises(ValueError, match="WrongDelay hard example shifts its copy"):
            dataset([gf.HardExampleGenerator(wnb(**settings), gf.WrongDelay(margin))])
    dataset([gf.HardExampleGenerator(wnb(allow_partially_out_of_window=True), gf.WrongDelay(0.45))])
    dataset([gf.HardExampleGenerator(wnb(**padded), gf.WrongDelay(0.04))])


def test_a_hard_examples_reference_times_fall_where_its_generators_do():
    """A wrong delay's copy is kept where its generator places signals, so no reference time falls elsewhere."""
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    padded = dict(front_padding_duration_seconds=0.1, back_padding_duration_seconds=0.1)
    real = wnb(network=network, **padded)
    for violation in (gf.WrongDelay(uniform(0.01, 0.03)), gf.OneDetector(), gf.MismatchedWaveforms("noise")):
        hard = gf.HardExampleGenerator(wnb(network=network, **padded), violation)
        assert hard.reference_time_window_seconds(1.0) == real.reference_time_window_seconds(1.0)
        assert hard.detector_margin_seconds() == real.detector_margin_seconds()


@pytest.mark.parametrize("ifos", [(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), DETECTORS], ids=["fixed", "per example"])
def test_one_detector_holds_the_whole_network_snr(ifos):
    x, y = dataset([gf.HardExampleGenerator(wnb(), gf.OneDetector())], ifos=ifos)[0]
    present = np.asarray(x["DETECTOR_MASK"]) > 0
    each = np.asarray(y["DETECTOR_SNR"], np.float64)[:, 0]
    holding = each > 0
    assert np.all(np.sum(holding, axis=-1) == 1) and np.all(present[holding])
    np.testing.assert_allclose(np.max(each, axis=-1), TARGET_SNR, rtol=SNR_TOLERANCE)
    injections = np.asarray(y["INJECTIONS"])[:, 0]
    assert np.all(injections[~holding] == 0.0)
    times = np.asarray(y["CENTRAL_TIME"])[:, 0]
    assert np.all(np.isfinite(times[holding])) and np.all(np.isnan(times[~holding]))


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

    durations = gf.HardExampleGenerator(
        wnb(network=network, duration_seconds=uniform(0.05, 0.2)),
        gf.MismatchedWaveforms(("duration_seconds",)), name="odd")
    _, parameters = durations.generate(BATCH, SAMPLE_RATE, 2.0, streams)
    assert np.all(np.isnan(np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS])))
    extent = np.asarray(parameters[SIGNAL_EXTENT])
    assert extent.shape == (BATCH, 2) and np.all(extent <= 0.1 + 1e-9)


def test_a_mismatched_binarys_detectors_carry_one_sources_loudness():
    """
    Two co-located, co-aligned detectors receive one real signal at one SNR. Each detector's chirp of a mismatched
    binary (masses of its own) is balanced to one source's noise-weighted loudness, so the two still receive it at
    one SNR: its SNR in each detector differs from a real signal's only through the antenna pattern, never through
    its masses' loudness.
    """
    twins = [gf.DetectorSlot.at(gf.Site.I1, gf.IFO.H1, name=name) for name in ("A", "B")]
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
                            gf.OneDetector())


def test_mismatched_waveforms_correlate_as_asked():
    network = gf.Network([gf.IFO.H1, gf.IFO.L1])
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
        (wnb(), gf.WrongDelay(0.01)), (wnb(), gf.WrongDelay(0.02)), (wnb(), gf.OneDetector()),
        (wnb(duration_seconds=constant(0.2)), gf.OneDetector()))}
    assert len(prints) == 4
    assert canonical(gf.HardExampleGenerator(ringdown(), gf.TimeReversed())) == \
        canonical(gf.HardExampleGenerator(ringdown(), gf.TimeReversed()))


@pytest.mark.parametrize("setting", sorted(test_injection.CHANGED_SETTINGS))
def test_a_hard_examples_fingerprint_reads_its_own_settings(setting):
    """
    A hard example's settings, its generator's when it is built, are its own, and the injection layer reads them
    from it: one changed after it is built moves its fingerprint, as it moves its batches.
    """
    from gravyflow.src.utils.fingerprint import config_fingerprint
    changed = gf.HardExampleGenerator(wnb(), gf.OneDetector())
    setattr(changed, setting, test_injection.CHANGED_SETTINGS[setting]())
    assert config_fingerprint(changed) != config_fingerprint(gf.HardExampleGenerator(wnb(), gf.OneDetector()))


def test_a_hard_examples_injection_chance_is_its_own():
    """
    Its injection chance decides which examples hold a signal, as for any generator, whatever its generator's
    is: set to 1 after it is built (as the validator sets it), every example holds one.
    """
    streams = gf.SeedTree(2).batch("train", 0, 0).sub("wave/hard")
    hard = gf.HardExampleGenerator(wnb(injection_chance=0.5), gf.OneDetector(), name="hard")
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
    receives one burst with, as other skies and polarisations do: a mismatch in them alone is refused.
    """
    drawn = {"amplitude": uniform(0.5, 2.0), "phase_radians": uniform(0.0, np.pi),
             "polarization_eccentricity": uniform(0.0, 1.0)}
    for generator, shape in (
            (gf.SineGaussianGenerator, {"frequency_hertz": uniform(80.0, 120.0)}),
            (gf.ChirpletGenerator, {"duration_seconds": uniform(0.2, 0.5)}),
            (gf.RingdownGenerator, {"damping_time_seconds": uniform(0.01, 0.05)}),
            (gf.PeriodicWaveGenerator, {"frequency_hertz": uniform(80.0, 120.0)})):
        own = {name: value for name, value in drawn.items() if name in generator.distributed_attributes}
        burst = generator(**own, **shape)
        for name in (*own, tuple(own)):
            with pytest.raises(ValueError, match="none of them a draw that changes"):
                gf.HardExampleGenerator(burst, gf.MismatchedWaveforms(name))
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
        gf.HardExampleGenerator(gf.HardExampleGenerator(wnb(), gf.OneDetector()), gf.OneDetector())
    with pytest.raises(TypeError, match="violation is a gf.Violation"):
        gf.HardExampleGenerator(wnb(), "late")
    for violation in (gf.WrongDelay(0.01), gf.OneDetector(), gf.MismatchedWaveforms("noise")):
        with pytest.raises(ValueError, match="needs two detectors present"):
            gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1])), violation)
        with pytest.raises(ValueError, match="needs two detectors present"):
            dataset([gf.HardExampleGenerator(wnb(), violation)],
                    ifos=gf.Detectors([gf.IFO.H1, gf.IFO.L1], num_present=uniform(1, 2, int)))
    # Reversal needs no second detector.
    gf.HardExampleGenerator(ringdown(network=gf.Network([gf.IFO.H1])), gf.TimeReversed())


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


def _bin_energies(strain, width):
    """The energy of ``strain`` (Batch, Time) in successive bins of ``width`` samples."""
    usable = strain.shape[-1] // width * width
    return np.sum(strain[:, :usable].reshape(strain.shape[0], -1, width) ** 2, axis=-1)


def test_a_reversed_ringdown_rises_where_every_ringdown_decays():
    """
    Every ringdown, whatever its frequency, damping time, phase or polarisation and wherever it is placed, holds
    less energy in each stretch of its support than in the one before; reversed, each holds more: no ringdown is
    a reversed one, so a reversed ringdown is a waveform its generator never makes.
    """
    forward, backward = _forward_and_reversed(
        gf.RingdownGenerator, num_waveforms=32, frequency_hertz=uniform(60.0, 120.0),
        damping_time_seconds=uniform(0.02, 0.08), phase_radians=uniform(0.0, 2.0 * np.pi),
        polarization_eccentricity=uniform(0.0, 1.0))
    angles = np.random.default_rng(1).uniform(0.0, np.pi, (32, 1))
    width = int(0.05 * SAMPLE_RATE)  # Three cycles or more of every ringdown here, well below the Nyquist frequency.
    for waveforms, decays in ((forward, True), (backward, False)):
        strain = np.cos(angles) * waveforms[:, 0] + np.sin(angles) * waveforms[:, 1]
        for energies in _bin_energies(strain, width):
            held = energies[energies > 1e-6 * energies.max()]
            steps = np.diff(held)
            assert len(held) >= 2 and np.all(steps < 0.0 if decays else steps > 0.0)


def test_time_reversal_is_refused_where_it_stays_in_its_generators_family():
    """
    A hard example is never a real signal of its own generator: time reversal is accepted only by a generator
    that declares its reversal leaves its family (a binary's chirp, a ringdown), and refused by name elsewhere.
    """
    for generator in (wnb(), gf.SineGaussianGenerator(), gf.ChirpletGenerator(),
                      *(gf.PeriodicWaveGenerator(wave_shape=shape) for shape in gf.WaveShape)):
        with pytest.raises(ValueError, match="reversed in time"):
            gf.HardExampleGenerator(generator, gf.TimeReversed())
    for generator in (gf.CBCGenerator(), gf.RingdownGenerator()):
        gf.HardExampleGenerator(generator, gf.TimeReversed())
    declared = {generator for generator in SHAPES if generator.time_reversal_leaves_family}
    assert declared == {gf.CBCGenerator, gf.RingdownGenerator}
    assert not gf.WaveformGenerator.time_reversal_leaves_family


def test_mismatched_waveforms_of_nothing_are_refused_by_name():
    for differ in (None, (), (1,), ("noise", None)):
        with pytest.raises(ValueError, match="at least one named draw"):
            gf.MismatchedWaveforms(differ)


def test_an_incoherent_generators_violation_is_refused():
    """A different waveform in each detector is no real signal: its violation is refused by name, never labelled NONE."""
    with pytest.raises(ValueError, match="IncoherentGenerator"):
        dataset([gf.IncoherentGenerator([wnb(), wnb(), wnb()])])[0]


@pytest.mark.parametrize("ifos", [(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), DETECTORS], ids=["fixed", "per example"])
def test_violation_detectors_name_each_copy_that_carries_the_violation(ifos):
    """
    Per detector: none for a real signal; the shifted copy of a wrong delay (the one detector whose arrival is not
    the real signal's of its name); the one detector that holds a one-detector signal; every present detector for
    a mismatched or reversed waveform; never an absent detector; NaN where nothing was injected.
    """
    def labels(generators):
        x, y = dataset(generators, ifos=ifos, outputs=[RV.VIOLATION_DETECTORS])[0]
        return np.asarray(x["DETECTOR_MASK"]) > 0, y

    _, real = labels([wnb(name="delayed")])
    present, y = labels([wnb(name="real"), gf.HardExampleGenerator(wnb(), gf.WrongDelay(0.01), name="delayed"),
                         gf.HardExampleGenerator(wnb(), gf.OneDetector(), name="one"),
                         gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms("noise"), name="mismatched"),
                         gf.HardExampleGenerator(ringdown(), gf.TimeReversed(), name="reversed"),
                         wnb(name="sometimes", injection_chance=0.5)])
    carriers = np.asarray(y["VIOLATION_DETECTORS"])
    assert carriers.shape == (BATCH, 6, present.shape[-1]) and carriers.dtype == np.float32
    times, each = np.asarray(y["CENTRAL_TIME"]), np.asarray(y["DETECTOR_SNR"])
    np.testing.assert_array_equal(carriers[:, 0], 0.0)
    shifted = np.where(present, times[:, 1] != np.asarray(real["CENTRAL_TIME"])[:, 0], False)
    assert np.all(np.sum(shifted, axis=-1) == 1)
    np.testing.assert_array_equal(carriers[:, 1], shifted.astype(np.float32))
    np.testing.assert_array_equal(carriers[:, 2], (each[:, 2] > 0).astype(np.float32))
    assert np.all(np.sum(carriers[:, 2], axis=-1) == 1)
    for g in (3, 4):
        np.testing.assert_array_equal(carriers[:, g], present.astype(np.float32))
    injected = ~np.isnan(np.asarray(y["VIOLATION"])[:, 5])
    assert 0 < np.sum(injected) < BATCH
    assert np.all(np.isnan(carriers[~injected, 5])) and np.all(carriers[injected, 5] == 0.0)


def test_a_violation_acts_through_its_own_hooks():
    """
    What a violation does is its own: a one-detector signal that keeps the first present detector, by its own
    projection alone, is held there, with nothing in the hard example naming it.
    """
    @dataclasses.dataclass(frozen=True)
    class FirstDetector(gf.OneDetector):
        def projection(self, hard_example, network, time_delays, present, stream, num_examples, *placement):
            kept = np.zeros(present.shape, dtype=np.float32)
            kept[np.arange(num_examples), np.argmax(present > 0, axis=1)] = 1.0
            return None, kept, kept

    x, y = dataset([gf.HardExampleGenerator(wnb(), FirstDetector())], ifos=DETECTORS,
                   outputs=[RV.VIOLATION_DETECTORS])[0]
    present = np.asarray(x["DETECTOR_MASK"]) > 0
    first = np.zeros(present.shape, dtype=np.float32)
    first[np.arange(BATCH), np.argmax(present, axis=1)] = 1.0
    np.testing.assert_array_equal(np.asarray(y["VIOLATION_DETECTORS"])[:, 0], first)
    np.testing.assert_array_equal(np.asarray(y["DETECTOR_SNR"])[:, 0] > 0, first > 0)
