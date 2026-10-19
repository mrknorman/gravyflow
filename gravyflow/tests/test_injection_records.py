"""
The records each injection carries for later analysis (gf.INJECTION_RECORDS)
and each example's BNS range, each against a calculation independent of
gravyflow:

- IN_BAND_DURATION_SECONDS against LALSimulation: its IMRPhenomD chirp time,
  and the time from the band's lower frequency to the envelope's peak in its
  time-domain waveforms (SimInspiralTD);
- WHITENED_PEAK against the injection whitened in NumPy by the noise's true
  PSD, and the noise level against whitened noise's measured deviation;
- BNS_RANGE_MPC against LALSimulation's SenseMonitor range of the same curve.
"""
import jax.numpy as jnp
import lal
import lalsimulation as lalsim
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.features.waveforms import cbc

FS = 2048.0
LOWER_HERTZ = 20.0
IFOS = (gf.IFO.H1, gf.IFO.L1)


def _constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def _uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def _batch(generator, outputs, num_examples=4, seed=1234):
    dataset = gf.GravyflowDataset(
        seed=seed,
        sample_rate_hertz=FS,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=list(IFOS)),
        waveform_generators=[generator],
        num_examples_per_batch=num_examples,
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=list(outputs),
        group="train",
    )
    x, y = dataset[0]
    return {key: np.asarray(value, np.float64) for key, value in {**x, **y}.items()}


def _profile_psd(ifo, frequencies):
    """The coloured noise's true one-sided PSD (its curve), on ``frequencies``."""
    f, asd = np.loadtxt(ifo.value.optimal_psd_path, delimiter=",").T
    return np.interp(frequencies, f, asd) ** 2


def _lal_psd(psd, delta_f):
    series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, delta_f, lal.Unit("strain^2 s"), len(psd)
    )
    series.data.data = np.asarray(psd, np.float64)
    return series


def _exact_spectrum(psd, delta_f, sample_rate, lower_hertz):
    """A PowerSpectrum holding ``psd`` exactly, with no estimator correction."""
    nperseg = int(round(sample_rate / delta_f))
    scale = np.sqrt(np.max(psd) * sample_rate)
    config = gf.SpectralConfig(
        lower_frequency_hertz=lower_hertz, fft_duration_seconds=nperseg / sample_rate,
        overlap_duration_seconds=0.0, psd_floor=0.0, correct_inverse_bias=False,
    )
    return gf.PowerSpectrum(
        jnp.asarray((psd * sample_rate / scale ** 2)[None, :], jnp.float32),
        jnp.asarray([[scale]], jnp.float32), float(sample_rate), config, 1.0, 1,
    )


# ---------------------------------------------------------------------------
# In-band duration
# ---------------------------------------------------------------------------

def _lal_time_from(frequency_hertz, approximant, m1, m2, chi1=0.0, chi2=0.0, sample_rate=8192.0):
    """
    LAL's time-domain waveform's time from ``frequency_hertz`` to its merger,
    face-on: from where the frequency of h+ - i hx last rises through it to the
    peak of |h+ - i hx|.
    """
    hp, hc = lalsim.SimInspiralTD(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
        100e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / sample_rate,
        0.5 * frequency_hertz, LOWER_HERTZ, lal.CreateDict(),
        lalsim.GetApproximantFromString(approximant),
    )
    h = hp.data.data - 1j * hc.data.data
    peak = int(np.argmax(np.abs(h)))
    frequency = np.abs(np.gradient(np.unwrap(np.angle(h)))) * sample_rate / (2.0 * np.pi)
    below = np.nonzero(frequency[:peak] < frequency_hertz)[0][-1]
    crossing = below + (frequency_hertz - frequency[below]) / (frequency[below + 1] - frequency[below])
    return (peak - crossing) / sample_rate


@pytest.mark.parametrize("m1, m2, chi1, chi2", [(1.4, 1.4, 0.0, 0.0), (10.0, 8.0, 0.3, -0.2)])
def test_in_band_duration_is_lal_chirp_time_however_long(m1, m2, chi1, chi2):
    """A CBC's in-band duration is its model's own, not cut by the window: a
    1.4 + 1.4 binary has 160 s from 20 Hz in a 1 s window. It matches
    LALSimulation's IMRPhenomD chirp time within 1e-3 where the merger is a
    negligible part of it."""
    generator = gf.CBCGenerator(
        mass_1_msun=m1, mass_2_msun=m2, spin_1_in=(0.0, 0.0, chi1), spin_2_in=(0.0, 0.0, chi2),
        inclination_radians=_uniform(0.0, np.pi), reference_orbital_phase_in=_uniform(0.0, 2 * np.pi),
        approximant=gf.Approximant.IMRPhenomD, network=list(IFOS),
        scaling_method=gf.ScalingMethod(_constant(10.0), gf.ScalingTypes.SNR),
    )
    _, parameters = generator.generate(3, FS, 1.0, generator.direct_streams(7))
    duration = generator.in_band_duration_seconds(parameters, LOWER_HERTZ)
    expected = lalsim.SimIMRPhenomDChirpTime(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, chi1, chi2, LOWER_HERTZ)
    np.testing.assert_allclose(duration, expected, rtol=1e-3)


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXAS"])
def test_in_band_duration_record_is_lal_time_to_merger(approximant):
    """IN_BAND_DURATION_SECONDS of a dataset's CBCs is the time from the band's
    lower frequency (20 Hz) to the merger in LAL's own time-domain waveforms,
    within 2 ms, from 8 to 95 solar masses."""
    generator = gf.CBCGenerator(
        mass_1_msun=_uniform(8.0, 95.0), mass_2_msun=_uniform(8.0, 95.0),
        approximant=gf.Approximant(approximant), network=list(IFOS),
        scaling_method=gf.ScalingMethod(_constant(10.0), gf.ScalingTypes.SNR),
    )
    batch = _batch(generator, [
        gf.ReturnVariables.IN_BAND_DURATION_SECONDS,
        gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN,
    ])
    duration = batch["IN_BAND_DURATION_SECONDS"]
    assert duration.shape == (4, 1)  # (B, G)
    for b in range(4):
        expected = _lal_time_from(
            LOWER_HERTZ, approximant, batch["MASS_1_MSUN"][b, 0], batch["MASS_2_MSUN"][b, 0]
        )
        assert abs(duration[b, 0] - expected) < 2e-3, (b, duration[b, 0], expected)


# ---------------------------------------------------------------------------
# Peak whitened amplitude
# ---------------------------------------------------------------------------

def test_whitened_noise_std_is_whitened_noise_deviation():
    """Coloured Gaussian noise whitened by gf.whiten has the deviation
    gf.whitened_noise_std gives for its band, within 2 %."""
    config = gf.SpectralConfig(lower_frequency_hertz=LOWER_HERTZ)
    batch = _batch(
        gf.SineGaussianGenerator(
            frequency_hertz=200.0, quality_factor=9.0, network=list(IFOS), injection_chance=0.0,
            scaling_method=gf.ScalingMethod(_constant(10.0), gf.ScalingTypes.SNR),
        ),
        [gf.ReturnVariables.WHITENED_ONSOURCE], num_examples=8,
    )
    measured = np.std(batch["WHITENED_ONSOURCE"])
    expected = gf.whitened_noise_std(config, FS)
    assert abs(measured / expected - 1.0) < 0.02, (measured, expected)


def test_whitened_peak_is_the_injection_whitened_by_the_true_psd():
    """WHITENED_PEAK is max |whitened injection| / sigma per detector: within
    5 % of the injection (INJECTIONS, its strain) whitened in NumPy by the
    noise's true PSD over the band; NETWORK_WHITENED_PEAK is their
    root-sum-square. A sine-Gaussian has no in-band duration (NaN)."""
    generator = gf.SineGaussianGenerator(
        frequency_hertz=200.0, quality_factor=9.0, network=list(IFOS),
        scaling_method=gf.ScalingMethod(_constant(20.0), gf.ScalingTypes.SNR),
        front_padding_duration_seconds=0.0, back_padding_duration_seconds=0.0,
    )
    batch = _batch(generator, [
        gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.WHITENED_PEAK,
        gf.ReturnVariables.NETWORK_WHITENED_PEAK, gf.ReturnVariables.IN_BAND_DURATION_SECONDS,
    ])
    injections = batch["INJECTIONS"][:, 0]  # (B, I, T)
    num_samples = injections.shape[-1]
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / FS)
    in_band = frequencies >= LOWER_HERTZ
    sigma = np.sqrt((FS / 2.0 - LOWER_HERTZ) / (FS / 2.0))
    expected = np.empty(injections.shape[:2])
    for i, ifo in enumerate(IFOS):
        psd = _profile_psd(ifo, frequencies)
        # White noise of unit variance has the one-sided PSD 2 / fs.
        transfer = np.where(in_band, np.sqrt(2.0 / FS / np.where(in_band, psd, 1.0)), 0.0)
        whitened = np.fft.irfft(np.fft.rfft(injections[:, i]) * transfer, n=num_samples)
        expected[:, i] = np.max(np.abs(whitened), axis=-1) / sigma

    peak = batch["WHITENED_PEAK"][:, 0]
    assert peak.shape == (4, len(IFOS))
    np.testing.assert_allclose(peak, expected, rtol=0.05)
    np.testing.assert_allclose(
        batch["NETWORK_WHITENED_PEAK"][:, 0], np.sqrt(np.sum(peak ** 2, axis=-1)), rtol=1e-6
    )
    assert np.all(np.isnan(batch["IN_BAND_DURATION_SECONDS"]))


# ---------------------------------------------------------------------------
# BNS range
# ---------------------------------------------------------------------------

def _curve_psd(profile, ifo, sample_rate, delta_f):
    """The one-sided PSD of a bundled noise curve on a grid from 0 Hz to Nyquist."""
    f, asd = np.loadtxt(profile.path(ifo), delimiter=",").T
    frequencies = np.arange(int(sample_rate / 2 / delta_f) + 1) * delta_f
    return np.interp(frequencies, f, asd, left=0.0) ** 2


def test_bns_range_of_a_known_curve_is_lal_sensemonitor_range():
    """bns_range_mpc of the aLIGO zero-detuned high-power design curve
    (T0900288) from 10 Hz is LALSimulation's SenseMonitor range of it,
    197.7 Mpc, within 0.2 %."""
    sample_rate, delta_f = 8192.0, 0.25
    psd = _curve_psd(gf.NoiseProfile.ZERO_DET_HIGH_POWER, gf.IFO.L1, sample_rate, delta_f)
    expected = lalsim.MeasureStandardSirenSenseMonitorRange(_lal_psd(psd, delta_f), 10.0, 2048.0)
    expected /= 1e6 * lal.PC_SI
    measured = gf.bns_range_mpc(_exact_spectrum(psd, delta_f, sample_rate, 10.0))
    assert measured.shape == (1,)
    assert abs(expected - 197.7) < 0.5
    assert abs(float(measured[0]) / expected - 1.0) < 2e-3, (measured, expected)


def test_bns_range_of_the_o4_projection_is_its_published_range():
    """The O4 high projection's aLIGO curve (LIGO-T2000012-v2) has the BNS range
    the observing scenarios publish for it, 190 Mpc (Living Rev. Relativ. 23, 3),
    within 1 %."""
    sample_rate, delta_f = 8192.0, 0.25
    psd = _curve_psd(gf.NoiseProfile.O4_HIGH, gf.IFO.H1, sample_rate, delta_f)
    measured = gf.bns_range_mpc(_exact_spectrum(psd, delta_f, sample_rate, 10.0))
    assert abs(float(measured[0]) / 190.0 - 1.0) < 0.01, measured


def test_bns_range_record_is_the_noise_curves_range():
    """BNS_RANGE_MPC, (B, I), is each detector's range from the batch's PSD
    estimate: within 3 % of LALSimulation's SenseMonitor range of the
    coloured noise's true curve over the band [20, 1024) Hz."""
    batch = _batch(
        gf.SineGaussianGenerator(
            frequency_hertz=200.0, quality_factor=9.0, network=list(IFOS),
            scaling_method=gf.ScalingMethod(_constant(10.0), gf.ScalingTypes.SNR),
        ),
        [gf.ReturnVariables.BNS_RANGE_MPC],
    )
    ranges = batch["BNS_RANGE_MPC"]
    assert ranges.shape == (4, len(IFOS))
    delta_f = 0.125
    frequencies = np.arange(int(FS / 2 / delta_f) + 1) * delta_f
    for i, ifo in enumerate(IFOS):
        psd = _profile_psd(ifo, frequencies)
        expected = lalsim.MeasureStandardSirenSenseMonitorRange(
            _lal_psd(psd, delta_f), LOWER_HERTZ, FS / 2.0
        ) / (1e6 * lal.PC_SI)
        np.testing.assert_allclose(ranges[:, i], expected, rtol=0.03, err_msg=ifo.name)


def test_bns_range_refuses_white_noise_and_a_band_without_a_lower_edge():
    """WHITE noise has no range (it is not strain), and a band without a lower
    edge has none either: both are refused by name."""
    with pytest.raises(ValueError, match="BNS_RANGE_MPC"):
        gf.GravyflowDataset(
            seed=1, sample_rate_hertz=FS, onsource_duration_seconds=1.0,
            offsource_duration_seconds=16.0, crop_duration_seconds=0.5,
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
            num_examples_per_batch=2, input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.BNS_RANGE_MPC], group="train",
        )
    frequencies = np.arange(129) * 4.0
    with pytest.raises(ValueError, match="lower edge"):
        gf.bns_range_mpc(_exact_spectrum(np.ones_like(frequencies), 4.0, 1024.0, None))
