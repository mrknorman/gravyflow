"""
The optimal SNR (``snr``) and the scaling to a target SNR (``scale_to_snr``),
against closed forms. Their agreement with pycbc, scale freedom, network sums,
per-example targets with cropping, and whitening are tested in test_spectral.py.
"""
import numpy as np
import pytest
from keras import ops
from scipy.signal import chirp

import gravyflow as gf
from gravyflow.src.dataset.tools import snr as gf_snr
from gravyflow.tests._helpers import representation_tolerance, statistical_bound
from gravyflow.tests.test_psd import welch_relative_sigma


def white_spectrum(sample_rate, lower_frequency_hertz):
    """The exact PSD of unit-variance white noise, 2 / fs, as a PowerSpectrum (1 s segments)."""
    config = gf.SpectralConfig(lower_frequency_hertz=lower_frequency_hertz)
    return gf.PowerSpectrum(np.full((1, 1, int(sample_rate) // 2 + 1), 2.0, dtype=np.float32),
                            np.ones((1, 1, 1), dtype=np.float32), sample_rate, config)


def closed_form_snr(h, sample_rate, lower_frequency_hertz):
    """
    sqrt(4 df sum |h~(f)|^2 / S) over the band, S = 2 / fs, h~ = rfft(h) / fs, in
    float64, with pycbc's band convention (from int(f_low / df), DC and an even
    length's Nyquist bin left out).
    """
    h = np.asarray(h, np.float64)
    n = h.shape[-1]
    df = sample_rate / n
    kmin, kmax = max(int(lower_frequency_hertz / df), 1), int((n + 1) / 2.0)
    power = np.abs(np.fft.rfft(h, axis=-1)[..., kmin:kmax] / sample_rate) ** 2
    return np.sqrt(4.0 * df * np.sum(power, axis=-1) / (2.0 / sample_rate))


def test_find_closest():
    tensor = ops.convert_to_tensor([1.0, 2.0, 5.0])
    assert gf_snr.find_closest(tensor, 2.1) == 1


def test_snr_theoretical_sine():
    """A sine of amplitude A in unit white noise has SNR A sqrt(T fs / 2)."""
    sample_rate, duration, freq = 1024.0, 64.0, 100.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    injection = np.sin(2 * np.pi * freq * t).astype(np.float32).reshape(1, 1, -1)
    expected_snr = np.sqrt(duration * sample_rate / 2.0)

    # With the exact PSD of the background (2 / fs) the formula is exact, to
    # float32 rounding of a sum of N squares.
    calculated = gf_snr.snr(injection, white_spectrum(sample_rate, 5.0), sample_rate_hertz=sample_rate)
    np.testing.assert_allclose(calculated[0], expected_snr, rtol=representation_tolerance(np.float32, 1.0, np.sqrt(t.size)))

    # With a PSD estimated from noise, a bin-centred sine reads one bin of the
    # estimate (no interpolation: 100 Hz is a bin of the 1 s segments), so its SNR
    # carries that bin's estimation noise. Statistical: SNR ~ P_hat^(-1/2), so its
    # relative sigma is half the mean Welch estimate's.
    background = np.random.default_rng(42).standard_normal(t.size).astype(np.float32).reshape(1, 1, -1)
    estimated = gf_snr.snr(injection, background, sample_rate_hertz=sample_rate, fft_duration_seconds=1.0,
                           overlap_duration_seconds=0.5, lower_frequency_cutoff=5.0, psd_method="mean")
    num_segments = (t.size - 1024) // 512 + 1
    sigma = welch_relative_sigma(1024, 512, num_segments) / 2.0
    assert abs(float(estimated[0]) / expected_snr - 1.0) < statistical_bound(sigma)


def _sine_gaussian(rate, seconds):
    gen = gf.SineGaussianGenerator(
        frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        quality_factor=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1])
    return gen.generate(num_waveforms=1, sample_rate_hertz=rate, duration_seconds=seconds)[0][:, 0:1]


def _ringdowns(rate, seconds):
    gen = gf.RingdownGenerator(
        frequency_hertz=gf.Distribution(value=150.0, type_=gf.DistributionType.CONSTANT),
        damping_time_seconds=gf.Distribution(value=0.05, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1])
    return gen.generate(num_waveforms=2, sample_rate_hertz=rate, duration_seconds=seconds)[0][:, 0:1]


def _periodic(shape):
    def make(rate, seconds):
        gen = gf.PeriodicWaveGenerator(
            wave_shape=shape,
            frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT),
            duration_seconds=gf.Distribution(value=2.0, type_=gf.DistributionType.CONSTANT),
            amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
            network=[gf.IFO.L1])
        return gen.generate(num_waveforms=1, sample_rate_hertz=rate, duration_seconds=seconds)[0][:, 0:1]
    return make


@pytest.mark.parametrize("make, rate, seconds, lower", [
    (_sine_gaussian, 1024.0, 4.0, 10.0),
    (_ringdowns, 1024.0, 2.0, 50.0),
    (_periodic(gf.WaveShape.SINE), 512.0, 4.0, 10.0),
    (_periodic(gf.WaveShape.SQUARE), 512.0, 4.0, 10.0),
    (_periodic(gf.WaveShape.TRIANGLE), 512.0, 4.0, 10.0),
], ids=["sine_gaussian", "ringdowns", "sine", "square", "triangle"])
def test_snr_matches_the_closed_form(make, rate, seconds, lower):
    """Each waveform's SNR against the exact white PSD is the closed-form integral, per example."""
    injection = np.asarray(make(rate, seconds), np.float32)
    calculated = np.asarray(gf_snr.snr(injection, white_spectrum(rate, lower), sample_rate_hertz=rate))
    expected = closed_form_snr(injection[:, 0], rate, lower)
    assert calculated.shape == (injection.shape[0],) and np.all(expected > 1.0)
    # Representation: float32 rounding of an FFT and a sum of N squares, relative.
    np.testing.assert_allclose(calculated, expected,
                               rtol=representation_tolerance(np.float32, 1.0, np.sqrt(injection.shape[-1])))


def test_snr_of_a_1d_injection_is_the_one_row_value():
    """A 1D injection and 1D background give the scalar SNR of the same one-row batch."""
    rate = 256.0
    rng = np.random.default_rng(42)
    t = np.arange(int(4 * rate)) / rate
    injection = np.sin(2 * np.pi * 30.0 * t).astype(np.float32)
    background = rng.standard_normal(int(16 * rate)).astype(np.float32)
    settings = dict(sample_rate_hertz=rate, fft_duration_seconds=1.0, overlap_duration_seconds=0.5,
                    lower_frequency_cutoff=10.0)
    scalar = gf_snr.snr(injection, background, **settings)
    assert np.shape(scalar) == ()
    np.testing.assert_array_equal(scalar, np.asarray(gf_snr.snr(injection[None], background[None], **settings))[0])


def _sine(rate, seconds, freq):
    return np.sin(2 * np.pi * freq * np.arange(int(rate * seconds)) / rate).astype(np.float32)


def _chirplet(rate, seconds):
    gen = gf.ChirpletGenerator(
        start_frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT),
        end_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        duration_seconds=gf.Distribution(value=2.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1])
    waveforms, _ = gen.generate(num_waveforms=1, sample_rate_hertz=rate, duration_seconds=seconds)
    return np.asarray(waveforms[:, 0:1], np.float32)


#: (injection, background shape, targets, sample rate, lower cutoff), each a
#: shape scale_to_snr takes.
SCALING_CASES = {
    "sine (B, I, T)": (lambda: _sine(100.0, 4.0, 10.0).reshape(1, 1, -1), (1, 1, 1600), [10.0], 100.0, 5.0),
    "sine against a 1D background": (lambda: _sine(100.0, 4.0, 10.0).reshape(1, 1, -1), (1600,), [10.0], 100.0,
                                     5.0),
    "chirp": (lambda: chirp(np.arange(4096) / 1024.0, f0=20, t1=4.0, f1=400,
                            method="linear").astype(np.float32).reshape(1, 1, -1), (1, 1, 16384), 15.0, 1024.0, 15.0),
    "chirplet": (lambda: _chirplet(1024.0, 4.0), (1, 1, 16384), 20.0, 1024.0, 15.0),
    "two rows (B, T)": (lambda: np.stack([_sine(256.0, 4.0, 50.0), 0.5 * _sine(256.0, 4.0, 50.0)]), (2, 4096),
                        [10.0, 15.0], 256.0, 10.0),
}


@pytest.mark.parametrize("case", list(SCALING_CASES))
def test_scale_to_snr_gives_the_target_when_measured_again(case):
    """
    Scaled to a target and measured again against the same background, each
    example has the target SNR, to float32 rounding (the same PSD estimate
    serves both).
    """
    make, background_shape, targets, rate, lower = SCALING_CASES[case]
    injection = make()
    background = np.random.default_rng(42).standard_normal(background_shape).astype(np.float32)
    settings = dict(sample_rate_hertz=rate, fft_duration_seconds=1.0, overlap_duration_seconds=0.5,
                    lower_frequency_cutoff=lower)
    scaled = gf_snr.scale_to_snr(injection, background, targets, **settings)
    assert ops.shape(scaled) == injection.shape
    measured = np.asarray(gf_snr.snr(scaled, background, **settings))
    expected = np.broadcast_to(np.asarray(targets, np.float32), measured.shape)
    # Representation: float32 rounding of the SNR's sum of N squares, twice (scale, measure).
    np.testing.assert_allclose(measured, expected,
                               rtol=representation_tolerance(np.float32, 1.0, 2 * np.sqrt(injection.shape[-1])))


def test_find_closest_edge_cases():
    """Test find_closest with various edge cases."""
    tensor = ops.convert_to_tensor([1.0, 5.0, 10.0])
    assert int(gf_snr.find_closest(tensor, 5.0)) == 1
    assert int(gf_snr.find_closest(tensor, 0.5)) == 0
    assert int(gf_snr.find_closest(tensor, 12.0)) == 2
    # A tie goes to the first of the closest (argmin).
    assert int(gf_snr.find_closest(tensor, 3.0)) == 0
