"""
Waveform physics against references: lal and lalsimulation, and analytic forms.

- Burst conventions: the sine-Gaussian against lalsimulation's
  SimBurstSineGaussian (Q, eccentricity, phase); the chirplet's instantaneous
  frequency; the ringdown's start and shape; the WNB's band and Gaussianity.
- Polarisation, sky and placement: isotropic sky draws, configured sky and
  polarisation labels, uniform placement and the peak-time label against the
  measured peak.
- CBC parameters: spins, the mass ordering, unsupported parameters.
- Detector geometry: sidereal time, delays and antenna patterns against lal.

lal is a required input of these tests: if it cannot be imported, they fail.
"""
import itertools
import warnings

import numpy as np
import pytest
import scipy.signal
import scipy.stats

import lal
import lalsimulation

import gravyflow as gf
from gravyflow.src.dataset.features.waveforms import wnb as wnb_module
from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window

FS = 2048.0
ONSOURCE = 1.0
CROP = 0.5
#: The root of the direct streams the injection tests draw from.
TREE = gf.SeedTree(7)


def uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def hrss(h):
    return float(np.sqrt(np.sum(np.asarray(h, np.float64) ** 2) / FS))


def first_batch(generator, num_examples=64):
    """An injector of ``generator`` and its batches k = 0, 1, ... of ``TREE``'s direct streams."""
    injector = gf.InjectionGenerator([generator])
    batches = (injector.generate(
        TREE.direct(k), sample_rate_hertz=FS, onsource_duration_seconds=ONSOURCE,
        crop_duration_seconds=CROP, num_examples_per_batch=num_examples,
    ) for k in itertools.count())
    return injector, batches


# --------------------------------------------------------------------------
# Sine-Gaussian against lalsimulation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("q, frequency, eccentricity, phase", [
    (9.0, 200.0, 1.0, 0.0),
    (3.0, 70.0, 0.0, 0.7),
    (30.0, 500.0, 0.5, 2.1),
    (100.0, 1000.0, 0.9, 4.0),
])
def test_sine_gaussian_matches_lalsimulation(q, frequency, eccentricity, phase):
    generator = gf.SineGaussianGenerator(
        frequency_hertz=frequency, quality_factor=q, amplitude=1.0,
        phase_radians=phase, polarization_eccentricity=eccentricity,
    )
    waveforms, parameters = generator.generate(1, FS, 4.0, streams=generator.direct_streams(1))
    waveforms = np.asarray(waveforms[0], np.float64)

    plus, cross = lalsimulation.SimBurstSineGaussian(q, frequency, 1.0, eccentricity, phase, 1.0 / FS)
    num_samples = waveforms.shape[-1]
    reference = np.zeros((2, num_samples))
    length = plus.data.length
    start = num_samples // 2 - (length - 1) // 2  # LAL's t = 0 at its middle sample
    reference[0, start:start + length] = plus.data.data
    reference[1, start:start + length] = cross.data.data

    scale = np.max(np.abs(reference))
    assert np.max(np.abs(waveforms - reference)) < 2e-4 * scale
    # hrss 1 as LAL samples it (near Nyquist the sampled sum departs from the
    # integral, for both).
    assert hrss(waveforms) == pytest.approx(hrss(reference), rel=1e-4)
    tau = float(np.asarray(parameters[gf.WaveformParameters.TAU_SECONDS])[0])
    assert tau == pytest.approx(q / (np.sqrt(2.0) * np.pi * frequency), rel=1e-6)


def test_sine_gaussian_eccentricity_sets_the_polarisation_content():
    rss = {}
    for eccentricity in (0.0, 1.0):
        generator = gf.SineGaussianGenerator(
            frequency_hertz=200.0, quality_factor=9.0, polarization_eccentricity=eccentricity
        )
        waveforms, _ = generator.generate(1, FS, 2.0, streams=generator.direct_streams(1))
        rss[eccentricity] = (hrss(waveforms[0, 0]), hrss(waveforms[0, 1]))
    # Circular: equal power in both; linear: plus only.
    assert rss[0.0][0] == pytest.approx(rss[0.0][1], rel=1e-4)
    assert rss[1.0][1] == 0.0


# --------------------------------------------------------------------------
# Chirplet and ringdown against their analytic forms
# --------------------------------------------------------------------------

def test_chirplet_instantaneous_frequency_runs_from_start_to_end():
    generator = gf.ChirpletGenerator(
        start_frequency_hertz=50.0, end_frequency_hertz=200.0, duration_seconds=0.5,
        phase_radians=0.4, polarization_eccentricity=0.0,
    )
    waveforms, parameters = generator.generate(1, FS, 2.0, streams=generator.direct_streams(1))
    plus = np.asarray(waveforms[0, 0], np.float64)
    cross = np.asarray(waveforms[0, 1], np.float64)

    # Circular: h+ + i hx = A(t) exp(i Phi(t)), so the phase is exact.
    phase = np.unwrap(np.angle(plus + 1j * cross))
    frequency = np.gradient(phase) * FS / (2.0 * np.pi)
    centre = plus.size // 2
    for offset, expected in ((-0.125, 87.5), (0.0, 125.0), (0.125, 162.5)):
        assert frequency[centre + int(offset * FS)] == pytest.approx(expected, abs=0.5)
    # Phase at the envelope's centre: Phi(0) = -phase.
    assert np.angle(plus[centre] + 1j * cross[centre]) == pytest.approx(-0.4, abs=1e-3)
    assert float(np.asarray(parameters[gf.WaveformParameters.PHASE_RADIANS])[0]) == pytest.approx(0.4)
    assert hrss(waveforms) == pytest.approx(1.0, rel=1e-4)


def test_ringdown_starts_at_its_reference_time_as_a_damped_sinusoid():
    frequency, tau, phase = 250.0, 0.02, 0.3
    generator = gf.RingdownGenerator(
        frequency_hertz=frequency, damping_time_seconds=tau, phase_radians=phase,
        polarization_eccentricity=0.0,
    )
    waveforms, _ = generator.generate(1, FS, 2.0, streams=generator.direct_streams(1))
    waveforms = np.asarray(waveforms[0], np.float64)
    num_samples = waveforms.shape[-1]
    t = (np.arange(num_samples) - num_samples / 2) / FS
    envelope = np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau), 0.0)
    expected_plus = envelope * np.cos(2 * np.pi * frequency * t - phase)
    expected_cross = envelope * np.sin(2 * np.pi * frequency * t - phase)

    for actual, expected in ((waveforms[0], expected_plus), (waveforms[1], expected_cross)):
        expected = expected / np.sqrt(np.sum(expected ** 2) / FS) / np.sqrt(2.0)
        assert np.max(np.abs(actual - expected)) < 1e-4 * np.max(np.abs(expected))
    assert np.all(waveforms[:, : num_samples // 2] == 0.0)


def test_ringdown_injection_keeps_its_energy():
    generator = gf.RingdownGenerator(
        frequency_hertz=200.0, damping_time_seconds=0.01, network=[gf.IFO.L1],
    )
    _, batches = first_batch(generator, num_examples=16)
    injections, mask, _ = next(batches)
    kept = np.sum(np.asarray(injections[0], np.float64) ** 2, axis=(1, 2)) / FS
    # amplitude 1 is hrss 1, so the generated energy is 1.
    np.testing.assert_allclose(kept, 1.0, rtol=1e-3)
    assert np.all(np.asarray(mask) == 1.0)


# --------------------------------------------------------------------------
# White noise bursts
# --------------------------------------------------------------------------

def test_wnb_band_edges_are_the_rfft_bins_in_the_band():
    mask = np.asarray(wnb_module.band_mask(512, FS, np.array([500.0]), np.array([1000.0])))[0]
    bins = np.nonzero(mask)[0]
    # Bin k is k * 2048 / 512 = 4k Hz.
    assert (bins.min(), bins.max()) == (125, 250)


def test_wnb_swapped_band_is_ordered_not_collapsed():
    generator = gf.WNBGenerator(
        duration_seconds=0.5, min_frequency_hertz=uniform(300.0, 400.0),
        max_frequency_hertz=uniform(100.0, 200.0),
    )
    waveforms, parameters = generator.generate(16, FS, 2.0, streams=generator.direct_streams(3))
    low = np.asarray(parameters[gf.WaveformParameters.MIN_FREQUENCY_HERTZ])
    high = np.asarray(parameters[gf.WaveformParameters.MAX_FREQUENCY_HERTZ])
    assert np.all((low >= 100.0) & (low <= 200.0))
    assert np.all((high >= 300.0) & (high <= 400.0))

    spectra = np.abs(np.fft.rfft(np.asarray(waveforms, np.float64), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(waveforms.shape[-1], 1.0 / FS)
    for spectrum, a, b in zip(spectra, low, high):
        inside = (freqs >= a) & (freqs <= b)
        assert spectrum[:, inside].sum() / spectrum.sum() > 0.97


def test_wnb_time_bandwidth_floor():
    generator = gf.WNBGenerator(
        duration_seconds=uniform(0.02, 0.5), min_frequency_hertz=uniform(20.0, 500.0),
        max_frequency_hertz=uniform(20.0, 500.0),
    )
    _, parameters = generator.generate(512, FS, 2.0, streams=generator.direct_streams(1))
    low = np.asarray(parameters[gf.WaveformParameters.MIN_FREQUENCY_HERTZ])
    high = np.asarray(parameters[gf.WaveformParameters.MAX_FREQUENCY_HERTZ])
    duration = np.asarray(parameters[gf.WaveformParameters.DURATION_SECONDS])
    assert np.min(duration * (high - low)) >= generator.min_time_bandwidth * (1 - 1e-6)

    with pytest.raises(ValueError, match="duration \\* bandwidth"):
        gf.WNBGenerator(duration_seconds=0.1, min_frequency_hertz=200.0, max_frequency_hertz=205.0)


def test_wnb_band_at_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        gf.WNBGenerator(
            duration_seconds=0.5, min_frequency_hertz=1100.0, max_frequency_hertz=1200.0
        ).generate(2, FS, 2.0)

    generator = gf.WNBGenerator(
        duration_seconds=0.5, min_frequency_hertz=900.0, max_frequency_hertz=1200.0
    )
    with pytest.warns(UserWarning, match="clipped"):
        _, parameters = generator.generate(2, FS, 2.0, streams=generator.direct_streams(1))
    np.testing.assert_allclose(parameters[gf.WaveformParameters.MAX_FREQUENCY_HERTZ], FS / 2)


def test_wnb_is_gaussian_and_unpolarised_by_default():
    n = 64
    waveforms = np.asarray(gf.wnb(
        n, FS, 2.0, np.full(n, 0.5, "f4"), np.full(n, 100.0, "f4"), np.full(n, 400.0, "f4"), key=gf.SeedTree(5).direct().key("wave/g/noise")
    ), np.float64)
    t = np.asarray(wnb_module.time_axis(waveforms.shape[-1], FS))
    envelope = np.asarray(wnb_module.hann_envelope(wnb_module.time_axis(waveforms.shape[-1], FS),
                                                   np.full(1, 0.5, "f4")))[0]
    middle = np.abs(t) < 0.1
    samples = (waveforms[:, :, middle] / envelope[middle]).ravel()[::8]
    assert scipy.stats.kstest(samples / samples.std(), "norm").pvalue > 0.01
    plus, cross = waveforms[:, 0, middle].ravel(), waveforms[:, 1, middle].ravel()
    assert np.var(plus) / np.var(cross) == pytest.approx(1.0, abs=0.05)
    assert abs(np.corrcoef(plus, cross)[0, 1]) < 0.05

    linear = np.asarray(gf.wnb(
        2, FS, 2.0, np.full(2, 0.5, "f4"), np.full(2, 100.0, "f4"), np.full(2, 400.0, "f4"),
        key=gf.SeedTree(5).direct().key("wave/g/noise"), polarization_eccentricity=np.ones(2, "f4"),
    ))
    assert np.all(linear[:, 1] == 0.0)


def test_wnb_generate_leaves_its_distributions_alone():
    generator = gf.WNBGenerator(duration_seconds=uniform(0.1, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.generate(2, FS, 0.5, streams=generator.direct_streams(1))
    assert (generator.duration_seconds.min_, generator.duration_seconds.max_) == (0.1, 1.0)


@pytest.mark.parametrize("make", [
    lambda d: gf.WNBGenerator(duration_seconds=d),
    lambda d: gf.ChirpletGenerator(duration_seconds=d),
    lambda d: gf.PeriodicWaveGenerator(duration_seconds=d),
    lambda d: gf.SineGaussianGenerator(quality_factor=d),
    lambda d: gf.RingdownGenerator(damping_time_seconds=d),
])
def test_burst_distribution_types(make):
    unbounded = gf.Distribution(mean=0.3, std=0.1, type_=gf.DistributionType.NORMAL)
    with pytest.raises(ValueError, match="unbounded support"):
        make(unbounded)
    bounded = gf.Distribution(mean=0.3, std=0.1, min_=0.1, max_=0.5, type_=gf.DistributionType.NORMAL)
    choice = gf.Distribution(possible_values=[0.1, 0.3], type_=gf.DistributionType.CHOICE)
    for distribution in (bounded, choice):
        generator = make(distribution)
        assert np.isfinite(generator.get_max_generated_duration())
        waveforms, _ = generator.generate(2, FS, 4.0, streams=generator.direct_streams(1))
        assert np.all(np.isfinite(np.asarray(waveforms)))


@pytest.mark.parametrize("make", [
    lambda: gf.SineGaussianGenerator(frequency_hertz=1100.0),
    lambda: gf.ChirpletGenerator(end_frequency_hertz=1500.0),
    lambda: gf.RingdownGenerator(frequency_hertz=1024.0),
])
def test_burst_frequencies_must_be_below_nyquist(make):
    with pytest.raises(ValueError, match="Nyquist"):
        make().generate(1, FS, 1.0)


def test_generators_without_scaling_are_refused_by_the_dataset():
    for generator in (gf.WNBGenerator(network=[gf.IFO.L1]), gf.CBCGenerator(network=[gf.IFO.L1])):
        with pytest.raises(ValueError, match="no scaling_method"):
            gf.GravyflowDataset(
                noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
                sample_rate_hertz=FS, onsource_duration_seconds=ONSOURCE,
                offsource_duration_seconds=4.0, num_examples_per_batch=2,
                waveform_generators=[generator],
                input_variables=[gf.ReturnVariables.ONSOURCE], seed=7,
                group="train",
            )


# --------------------------------------------------------------------------
# Placement, sky and the peak-time label
# --------------------------------------------------------------------------

def _offsets_seconds(generator, batches=4, num_examples=512):
    _, iterator = first_batch(generator, num_examples=num_examples)
    offsets = []
    for _ in range(batches):
        _, _, parameters = next(iterator)
        offsets.append(np.asarray(parameters["shift"]).ravel() / FS)
    return np.concatenate(offsets)


def test_default_placement_is_uniform_and_keeps_the_signal_in_the_window():
    generator = gf.SineGaussianGenerator(frequency_hertz=200.0, quality_factor=9.0, network=[gf.IFO.L1])
    offsets = _offsets_seconds(generator)
    tau = 9.0 / (np.sqrt(2.0) * np.pi * 200.0)
    geocentre_delay = np.max(np.linalg.norm(np.asarray(generator.network.location), axis=-1)) / lal.C_SI
    reach = ONSOURCE / 2 - np.sqrt(np.log(1000.0)) * tau - geocentre_delay

    assert np.all(np.abs(offsets) <= reach + 0.5 / FS)
    counts, _ = np.histogram(offsets, bins=16, range=(-reach, reach))
    assert scipy.stats.chisquare(counts).pvalue > 1e-3
    assert abs(np.mean(offsets)) < 0.03


def test_configured_placement_range_and_the_partial_option():
    generator = gf.SineGaussianGenerator(
        frequency_hertz=200.0, quality_factor=9.0, network=[gf.IFO.L1],
        front_padding_duration_seconds=0.1, back_padding_duration_seconds=0.3,
    )
    offsets = _offsets_seconds(generator, batches=2)
    assert offsets.min() >= -0.1 - 0.5 / FS and offsets.max() <= 0.3 + 0.5 / FS
    counts, _ = np.histogram(offsets, bins=8, range=(-0.1, 0.3))
    assert scipy.stats.chisquare(counts).pvalue > 1e-3

    # A 0.8 s burst in a 1 s window can only move a little without leaving it...
    long_burst = gf.WNBGenerator(duration_seconds=0.8, min_frequency_hertz=50.0,
                                 max_frequency_hertz=300.0, network=[gf.IFO.L1])
    offsets = _offsets_seconds(long_burst, batches=1, num_examples=64)
    assert np.max(np.abs(offsets)) <= 0.1 - 0.021 + 0.5 / FS
    # ...unless partial signals are allowed.
    partial = gf.WNBGenerator(duration_seconds=0.8, min_frequency_hertz=50.0,
                              max_frequency_hertz=300.0, network=[gf.IFO.L1],
                              allow_partially_out_of_window=True)
    offsets = _offsets_seconds(partial, batches=1, num_examples=256)
    assert np.max(np.abs(offsets)) > 0.4


def _add_to_zero_onsource(injector, injections, parameters, requested):
    num_examples = injections.shape[1]
    num_samples = injections.shape[-1]
    zeros = np.zeros((num_examples, 1, num_samples), "float32")
    offsource = np.random.default_rng(0).normal(size=(num_examples, 1, 8 * int(FS))).astype("float32")
    return injector.add_injections_to_onsource(
        injections, None, zeros, offsource, parameters_to_return=requested,
        onsource_duration_seconds=ONSOURCE, injection_parameters=parameters,
        streams=TREE.direct(0),
    )


def test_peak_time_label_matches_the_measured_peak():
    hrss_scaling = gf.ScalingMethod(value=constant(1.0), type_=gf.ScalingTypes.HRSS)
    generator = gf.SineGaussianGenerator(
        frequency_hertz=150.0, quality_factor=20.0, scaling_method=hrss_scaling,
        network=[gf.IFO.H1, gf.IFO.L1],
    )
    injector, iterator = first_batch(generator, num_examples=32)
    injections, _, parameters = next(iterator)
    _, projected, labels = _add_to_zero_onsource(
        injector, injections, parameters, [gf.ReturnVariables.CENTRAL_TIME]
    )
    projected = np.asarray(projected[0], np.float64)  # (batch, detector, time)
    envelope = np.abs(scipy.signal.hilbert(projected, axis=-1))
    measured = np.argmax(envelope, axis=-1) / FS - CROP  # seconds into the onsource window
    label = np.asarray(labels[gf.ReturnVariables.CENTRAL_TIME][0]) * ONSOURCE
    assert np.max(np.abs(measured - label)) <= 1.0 / FS


def test_isotropic_sky_and_polarisation():
    hrss_scaling = gf.ScalingMethod(value=constant(1.0), type_=gf.ScalingTypes.HRSS)
    generator = gf.SineGaussianGenerator(scaling_method=hrss_scaling, network=[gf.IFO.L1])
    injector, iterator = first_batch(generator, num_examples=1024)
    injections, _, parameters = next(iterator)
    _, _, labels = _add_to_zero_onsource(injector, injections, parameters, [])
    ra = np.asarray(labels[gf.WaveformParameters.RIGHT_ASCENSION_RADIANS][0], np.float64)
    dec = np.asarray(labels[gf.WaveformParameters.DECLINATION_RADIANS][0], np.float64)
    psi = np.asarray(labels[gf.WaveformParameters.POLARIZATION_RADIANS][0], np.float64)
    assert scipy.stats.kstest(ra / (2 * np.pi), "uniform").pvalue > 1e-3
    assert scipy.stats.kstest((np.sin(dec) + 1) / 2, "uniform").pvalue > 1e-3
    assert scipy.stats.kstest(psi / np.pi, "uniform").pvalue > 1e-3


def test_configured_sky_is_labelled_and_projected_as_lal():
    ra, dec, psi, gps = 1.2, -0.4, 0.7, 1187008882.4
    hrss_scaling = gf.ScalingMethod(value=constant(1.0), type_=gf.ScalingTypes.HRSS)
    generator = gf.SineGaussianGenerator(
        frequency_hertz=150.0, quality_factor=20.0, polarization_eccentricity=0.3,
        scaling_method=hrss_scaling, network=[gf.IFO.H1],
        right_ascension_radians=ra, declination_radians=dec, polarization_radians=psi,
        reference_gps_time_seconds=gps,
        front_padding_duration_seconds=0.0, back_padding_duration_seconds=0.0,
    )
    injector, iterator = first_batch(generator, num_examples=2)
    injections, _, parameters = next(iterator)
    _, projected, labels = _add_to_zero_onsource(injector, injections, parameters, [])
    for key, value in ((gf.WaveformParameters.RIGHT_ASCENSION_RADIANS, ra),
                       (gf.WaveformParameters.DECLINATION_RADIANS, dec),
                       (gf.WaveformParameters.POLARIZATION_RADIANS, psi)):
        np.testing.assert_allclose(np.asarray(labels[key]), value, rtol=1e-6)

    # The same polarisations projected with lal's antenna pattern and delay;
    # the amplitude scaling is a common factor, fitted out.
    raw = np.asarray(injections[0, 0], np.float64)
    detector = lal.CachedDetectors[lal.LHO_4K_DETECTOR]
    f_plus, f_cross = lal.ComputeDetAMResponse(
        detector.response, ra, dec, psi, lal.GreenwichMeanSiderealTime(gps)
    )
    delay = lal.TimeDelayFromEarthCenter(detector.location, ra, dec, gps)
    strain = f_plus * raw[0] + f_cross * raw[1]
    freqs = np.fft.rfftfreq(strain.size, 1.0 / FS)
    expected = np.fft.irfft(np.fft.rfft(strain) * np.exp(-2j * np.pi * freqs * delay), n=strain.size)
    actual = np.asarray(projected[0, 0, 0], np.float64)
    scale = np.dot(actual, expected) / np.dot(expected, expected)
    assert scale > 0
    assert np.max(np.abs(actual / scale - expected)) < 1e-3 * np.max(np.abs(expected))


# --------------------------------------------------------------------------
# CBC parameters
# --------------------------------------------------------------------------

def test_single_spin_distribution_is_the_aligned_component():
    generator = gf.CBCGenerator(spin_1_in=uniform(-0.9, 0.9), spin_2_in=uniform(-0.5, 0.5))
    _, parameters = generator.generate(64, FS, 2.0, streams=generator.direct_streams(5))
    for key in (gf.WaveformParameters.SPIN_1_IN, gf.WaveformParameters.SPIN_2_IN):
        spins = np.asarray(parameters[key])
        assert spins.shape == (64, 3)
        assert np.all(spins[:, :2] == 0.0)
        assert np.all(np.abs(spins[:, 2]) <= 0.9)
    assert np.std(np.asarray(parameters[gf.WaveformParameters.SPIN_1_IN])[:, 2]) > 0.3


def test_spins_that_the_model_cannot_represent_are_refused():
    with pytest.raises(ValueError, match="Kerr bound"):
        gf.CBCGenerator(spin_1_in=uniform(-1.2, 1.2))
    with pytest.raises(ValueError, match="Kerr bound"):
        gf.CBCGenerator(spin_1_in=(uniform(-0.9, 0.9), uniform(-0.9, 0.9), 0.0),
                        approximant=gf.Approximant.IMRPhenomPv2)
    with pytest.raises(ValueError, match="in-plane"):
        gf.CBCGenerator(spin_1_in=(0.3, 0.0, 0.2), approximant=gf.Approximant.IMRPhenomD)


def test_spin_tuples_stay_per_source_and_match_the_waveform():
    generator = gf.CBCGenerator(
        mass_1_msun=uniform(30.0, 40.0), mass_2_msun=uniform(20.0, 29.0),
        spin_1_in=(uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)),
        spin_2_in=(0.1, 0.2, 0.3), approximant=gf.Approximant.IMRPhenomPv2,
    )
    waveforms, parameters = generator.generate(4, FS, 2.0, streams=generator.direct_streams(5))
    spin_1 = np.asarray(parameters[gf.WaveformParameters.SPIN_1_IN])
    spin_2 = np.asarray(parameters[gf.WaveformParameters.SPIN_2_IN])
    assert len({tuple(row) for row in spin_1.round(6)}) == 4
    assert len(set(spin_1[:, 0].round(6)) & set(spin_1[:, 1].round(6))) == 0

    reference = _generate_cbc_window(
        num_waveforms=4, sample_rate_hertz=FS, window_seconds=2.0,
        min_frequency_hertz=np.asarray(parameters[gf.WaveformParameters.get("min_frequency_hertz")]),
        approximant=gf.Approximant.IMRPhenomPv2,
        mass_1_msun=np.asarray(parameters[gf.WaveformParameters.MASS_1_MSUN]),
        mass_2_msun=np.asarray(parameters[gf.WaveformParameters.MASS_2_MSUN]),
        inclination_radians=np.asarray(parameters[gf.WaveformParameters.INCLINATION_RADIANS]),
        distance_mpc=np.asarray(parameters[gf.WaveformParameters.DISTANCE_MPC]),
        reference_orbital_phase_in=np.asarray(parameters[gf.WaveformParameters.REFERENCE_ORBITAL_PHASE_IN]),
        ascending_node_longitude=np.zeros(4), eccentricity=np.zeros(4),
        mean_periastron_anomaly=np.zeros(4), spin_1_in=spin_1, spin_2_in=spin_2,
        lambda_1=np.zeros(4), lambda_2=np.zeros(4),
    )
    # Generators return physical strain: no internal scale to apply.
    reference = np.asarray(reference)[..., : waveforms.shape[-1]]
    np.testing.assert_allclose(np.asarray(waveforms), reference, rtol=1e-4, atol=1e-6 * np.max(np.abs(reference)))


def test_the_mass_ordering_carries_each_bodys_spin_and_tides():
    generator = gf.CBCGenerator(
        mass_1_msun=uniform(1.2, 3.0), mass_2_msun=uniform(2.5, 10.0),
        lambda_1=400.0, lambda_2=0.0, spin_1_in=constant(0.05), spin_2_in=constant(0.6),
        approximant=gf.Approximant.IMRPhenomD_NRTidalv2,
    )
    _, parameters = generator.generate(64, FS, 2.0, streams=generator.direct_streams(2))
    m1 = np.asarray(parameters[gf.WaveformParameters.MASS_1_MSUN])
    m2 = np.asarray(parameters[gf.WaveformParameters.MASS_2_MSUN])
    l1 = np.asarray(parameters[gf.WaveformParameters.LAMBDA_1])
    l2 = np.asarray(parameters[gf.WaveformParameters.LAMBDA_2])
    s1 = np.asarray(parameters[gf.WaveformParameters.SPIN_1_IN])[:, 2]
    s2 = np.asarray(parameters[gf.WaveformParameters.SPIN_2_IN])[:, 2]
    assert np.all(m1 >= m2)
    neutron_star_is_1 = l1 == 400.0
    assert np.all(neutron_star_is_1 != (l2 == 400.0))
    assert np.all(np.where(neutron_star_is_1, m1, m2) <= 3.0)
    np.testing.assert_allclose(np.where(neutron_star_is_1, s1, s2), 0.05, rtol=1e-6)
    assert np.any(~neutron_star_is_1)


def test_unmodelled_cbc_parameters_are_refused():
    for attribute in ("eccentricity", "ascending_node_longitude", "mean_periastron_anomaly"):
        with pytest.raises(ValueError, match=f"{attribute} must be 0"):
            gf.CBCGenerator(**{attribute: uniform(0.0, 0.1)})
    with pytest.raises(ValueError, match="no tidal deformability"):
        gf.CBCGenerator(lambda_1=300.0, approximant=gf.Approximant.IMRPhenomD)
    gf.CBCGenerator(lambda_1=300.0, approximant=gf.Approximant.TaylorF2)


@pytest.mark.parametrize("distribution", [
    gf.Distribution(mean=30.0, std=5.0, type_=gf.DistributionType.NORMAL),
    gf.Distribution(possible_values=[20.0, 30.0, 40.0], type_=gf.DistributionType.CHOICE),
])
def test_cbc_accepts_normal_and_choice_masses(distribution):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator = gf.CBCGenerator(mass_1_msun=distribution)
    waveforms, _ = generator.generate(2, FS, 2.0, streams=generator.direct_streams(1))
    assert np.all(np.isfinite(np.asarray(waveforms)))


# --------------------------------------------------------------------------
# Detector geometry against lal
# --------------------------------------------------------------------------

def test_sidereal_time_and_leap_seconds_against_lal():
    gps = np.random.default_rng(0).uniform(6.5e8, 1.45e9, 500)
    mine = gf.greenwich_mean_sidereal_time(gps)
    reference = np.array([lal.GreenwichMeanSiderealTime(float(g)) for g in gps]) % (2 * np.pi)
    assert np.max(np.abs((mine - reference + np.pi) % (2 * np.pi) - np.pi)) < 1e-8


def test_geometry_against_lal_for_every_detector():
    network = gf.Network([gf.IFO.L1, gf.IFO.H1, gf.IFO.V1])
    rng = np.random.default_rng(1)
    ra = rng.uniform(0, 2 * np.pi, 200).astype("f4")
    dec = np.arcsin(rng.uniform(-1, 1, 200)).astype("f4")
    psi = rng.uniform(0, np.pi, 200).astype("f4")
    gps = 1239082262.2
    delays = np.asarray(network.get_time_delay(ra, dec, gps_time_seconds=gps))
    patterns = np.asarray(network.get_antenna_pattern(ra, dec, psi, gps_time_seconds=gps))
    gmst = lal.GreenwichMeanSiderealTime(gps)
    for index, name in enumerate((lal.LLO_4K_DETECTOR, lal.LHO_4K_DETECTOR, lal.VIRGO_DETECTOR)):
        detector = lal.CachedDetectors[name]
        np.testing.assert_allclose(np.asarray(network.response)[index], detector.response, atol=1e-6)
        expected_delay = [lal.TimeDelayFromEarthCenter(detector.location, float(a), float(d), gps)
                          for a, d in zip(ra, dec)]
        expected_pattern = [lal.ComputeDetAMResponse(detector.response, float(a), float(d), float(p), gmst)
                            for a, d, p in zip(ra, dec, psi)]
        np.testing.assert_allclose(delays[:, index], expected_delay, atol=1e-7)
        np.testing.assert_allclose(patterns[:, index], expected_pattern, atol=1e-5)


@pytest.mark.parametrize("gps", [630763213.0, 1126259462.4, 1187008882.4, 1268903496.0])
def test_antenna_response_peaks_at_each_detectors_zenith_and_nadir(gps):
    """
    |F|^2 = F+^2 + Fx^2 is 1 overhead, RA = longitude + GMST(t) and dec = latitude,
    and underfoot, the antipode: RA + pi and dec = -latitude.
    """
    ifos = (gf.IFO.L1, gf.IFO.H1, gf.IFO.V1)
    lal_detectors = (lal.LLO_4K_DETECTOR, lal.LHO_4K_DETECTOR, lal.VIRGO_DETECTOR)
    network = gf.Network(list(ifos))
    gmst = lal.GreenwichMeanSiderealTime(gps)

    rng = np.random.default_rng(3)
    sky_ra = rng.uniform(0, 2 * np.pi, 2000)
    sky_dec = np.arcsin(rng.uniform(-1, 1, 2000))
    for index, (ifo, name) in enumerate(zip(ifos, lal_detectors)):
        longitude, latitude = ifo.value.longitude_radians, ifo.value.latitude_radians
        ra = np.mod(np.array([longitude + gmst, longitude + gmst + np.pi]), 2 * np.pi)
        dec = np.array([latitude, -latitude])
        ra = np.concatenate([ra, sky_ra]).astype("f4")
        dec = np.concatenate([dec, sky_dec]).astype("f4")
        psi = np.full(ra.shape, 0.3, "f4")
        pattern = np.asarray(network.get_antenna_pattern(ra, dec, psi, gps_time_seconds=gps))[:, index]
        power = np.sum(pattern.astype(np.float64) ** 2, axis=-1)

        # Overhead and underfoot the response is (to the arms' tilt) total,
        # and nowhere on the sky is it larger.
        assert np.all(power[:2] > 0.999)
        assert np.max(power[2:]) <= np.min(power[:2]) + 1e-6

        response = lal.CachedDetectors[name].response
        expected = np.array([
            lal.ComputeDetAMResponse(response, float(a), float(d), float(p), gmst)
            for a, d, p in zip(ra[:2], dec[:2], psi[:2])
        ])
        np.testing.assert_allclose(pattern[:2], expected, atol=1e-5)


def test_projection_shift_only_zeroes_wrapped_samples():
    network = gf.Network([gf.IFO.L1])
    strain = np.ones((1, 2, 4096), "float32")
    from gravyflow.src.dataset.conditioning.detector import shift_waveform
    unshifted = np.asarray(shift_waveform(strain[:, :1], FS, np.zeros((1, 1), "float32")))
    assert np.all(unshifted == pytest.approx(1.0, abs=1e-5))
    delayed = np.asarray(shift_waveform(strain[:, :1], FS, np.full((1, 1), 10.3 / FS, "float32")))
    assert np.all(delayed[..., :11] == 0.0) and np.all(np.abs(delayed[..., 11:] - 1.0) < 1e-4)
