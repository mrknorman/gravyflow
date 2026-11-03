"""
Hard examples (gf.HardExampleGenerator): each violation is what it claims, and
the examples are the generator's in everything else.
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C
from gravyflow.src.dataset.features.injection import SIGNAL_EXTENT
from gravyflow.src.rng.consumers import check_consumer
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


def dataset(generators, ifos=(gf.IFO.H1, gf.IFO.L1, gf.IFO.V1), seed=3):
    return gf.Dataset(
        seed=seed, group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED),
        ifos=list(ifos) if isinstance(ifos, tuple) else ifos,
        waveform_generators=generators,
        sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
        num_examples_per_batch=BATCH, input_variables=[RV.ONSOURCE, RV.DETECTOR_MASK],
        output_variables=[RV.INJECTIONS, RV.DETECTOR_SNR, RV.CENTRAL_TIME, RV.VIOLATION, gf.ScalingTypes.SNR],
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
        gf.HardExampleGenerator(wnb(), violation, name=f"hard{j}") for j, violation in enumerate(violations)]
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
    base = wnb(name="hard")
    reversed_ = gf.HardExampleGenerator(wnb(), gf.TimeReversed(), name="hard")
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
    # The signal's extent is reversed with it: a burst's, half its duration either side.
    before, after = reversed_.get_signal_extent_seconds(reflected_parameters)
    np.testing.assert_allclose(before, 0.05)
    np.testing.assert_allclose(after, 0.05)


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


def test_a_wrong_delays_placement_allows_for_its_shifted_copy():
    network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    base = wnb(network=network)
    hard = gf.HardExampleGenerator(wnb(network=network), gf.WrongDelay(uniform(0.001, 0.03)))
    assert hard.detector_margin_seconds() == pytest.approx(
        base.detector_margin_seconds() + network.max_arrival_time_difference_seconds + 0.03)
    for violation in (gf.OneDetector(), gf.TimeReversed(), gf.MismatchedWaveforms("noise")):
        assert gf.HardExampleGenerator(wnb(network=network), violation).detector_margin_seconds() \
            == base.detector_margin_seconds()


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
    assert canonical(gf.HardExampleGenerator(wnb(), gf.TimeReversed())) == \
        canonical(gf.HardExampleGenerator(wnb(), gf.TimeReversed()))


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


def test_hard_examples_refuse_what_they_cannot_be():
    with pytest.raises(ValueError, match="bounded and above"):
        gf.WrongDelay(gf.Distribution(mean=0.0, std=0.01, type_=gf.DistributionType.NORMAL))
    with pytest.raises(ValueError, match="does not draw"):
        gf.HardExampleGenerator(wnb(), gf.MismatchedWaveforms("mass_1_msun"))
    with pytest.raises(TypeError, match="wraps a coherent"):
        gf.HardExampleGenerator(gf.HardExampleGenerator(wnb(), gf.TimeReversed()), gf.TimeReversed())
    with pytest.raises(TypeError, match="violation is one of"):
        gf.HardExampleGenerator(wnb(), "late")
    for violation in (gf.WrongDelay(0.01), gf.OneDetector(), gf.MismatchedWaveforms("noise")):
        with pytest.raises(ValueError, match="needs two detectors present"):
            gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1])), violation)
        with pytest.raises(ValueError, match="needs two detectors present"):
            dataset([gf.HardExampleGenerator(wnb(), violation)],
                    ifos=gf.Detectors([gf.IFO.H1, gf.IFO.L1], num_present=uniform(1, 2, int)))
    # Reversal needs no second detector.
    gf.HardExampleGenerator(wnb(network=gf.Network([gf.IFO.H1])), gf.TimeReversed())
