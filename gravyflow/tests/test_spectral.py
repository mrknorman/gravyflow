"""
The one PSD, whitening and SNR definition, against independent references:
scipy.signal.welch and pycbc.psd.welch (PSD), gwpy's TimeSeries.whiten
(whitening) and pycbc.filter.sigma (SNR).
"""
import numpy as np
import pytest
import scipy.signal

import gravyflow as gf
from gravyflow.src.dataset.tools.psd import estimate_psd
from gravyflow.src.dataset.tools.psd import PowerSpectrum, SpectralConfig, inverse_bias
from gravyflow.src.dataset.tools import snr as gf_snr
from gravyflow.src.dataset.conditioning import whiten as gf_whiten

FS = 2048.0
SCALE = 1.0e21


def design_psd(freqs):
    """aLIGO design curve (T1800044), held constant below 9 Hz."""
    import pycbc.psd
    df = freqs[1] - freqs[0]
    p = np.asarray(pycbc.psd.from_string("aLIGODesignSensitivityT1800044", len(freqs), df, 9.0), dtype=np.float64)
    p[:np.argmax(p > 0)] = p[np.argmax(p > 0)]
    return p


def coloured_noise(rng, shape, duration, fs=FS):
    """Gaussian noise with the design PSD, in physical strain (float64)."""
    n = int(duration * fs)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    amp = np.sqrt(design_psd(freqs) * n * fs / 4.0)
    spec = amp * (rng.standard_normal(shape + (len(freqs),)) + 1j * rng.standard_normal(shape + (len(freqs),)))
    spec[..., 0] = 0.0
    spec[..., -1] = spec[..., -1].real * np.sqrt(2)
    return np.fft.irfft(spec, n=n)


def chirp(duration, fs=FS):
    """An IMRPhenomD 30+30 Msun chirp, merger at the centre, physical strain."""
    from pycbc.waveform import get_td_waveform
    hp, _ = get_td_waveform(approximant="IMRPhenomD", mass1=30, mass2=30, delta_t=1 / fs, f_lower=20.0,
                            distance=400.0)
    h = np.asarray(hp)
    n = int(duration * fs)
    out = np.zeros(n)
    start = n // 2 - int(np.argmax(np.abs(h)))
    lo, hi = max(start, 0), min(start + len(h), n)
    out[lo:hi] = h[lo - start:hi - start]
    return out


def sine_gaussian(f0, q, duration, fs=FS, phase=0.0):
    t = np.arange(int(duration * fs)) / fs - duration / 2
    tau = q / (np.sqrt(2) * np.pi * f0)
    return np.exp(-t ** 2 / tau ** 2) * np.sin(2 * np.pi * f0 * t + phase)


# --- PSD --------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["mean", "median"])
def test_psd_matches_scipy_welch(mode):
    rng = np.random.default_rng(1)
    x = (coloured_noise(rng, (3,), 16.0) * SCALE).astype(np.float32)
    mine = gf.psd(x, nperseg=2048, sample_rate_hertz=FS, noverlap=1024, mode=mode).value()
    _, ref = scipy.signal.welch(x.astype(np.float64), fs=FS, nperseg=2048, noverlap=1024, average=mode)
    band = slice(20, 1000)
    np.testing.assert_allclose(np.asarray(mine)[:, band], ref[:, band], rtol=1e-3)


@pytest.mark.parametrize("mode", ["mean", "median"])
def test_psd_matches_pycbc_welch(mode):
    import pycbc.psd
    import pycbc.types
    rng = np.random.default_rng(2)
    x = coloured_noise(rng, (1,), 16.0)[0] * SCALE
    mine = gf.psd(x[None].astype(np.float32), nperseg=2048, sample_rate_hertz=FS, noverlap=1024, mode=mode).value()
    ref = np.asarray(pycbc.psd.welch(pycbc.types.TimeSeries(x, delta_t=1 / FS), seg_len=2048,
                                     seg_stride=1024, avg_method=mode))
    band = slice(20, 1000)
    # pycbc's window is the symmetric Hann, scipy's (and gravyflow's) the periodic one.
    ratio = np.asarray(mine)[0, band] / ref[band]
    assert np.median(np.abs(ratio - 1)) < 2e-3
    assert np.max(np.abs(ratio - 1)) < 2e-2


def test_psd_median_has_no_bias_on_white_noise():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, int(16 * FS))).astype(np.float32)
    pxx = gf.psd(x, nperseg=2048, sample_rate_hertz=FS, noverlap=1024, mode="median").value()
    level = np.mean(np.asarray(pxx)[:, 10:-10]) / (2.0 / FS)
    assert abs(level - 1.0) < 0.02, level


def test_psd_refuses_a_signal_shorter_than_one_segment():
    x = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(ValueError, match="at least one segment"):
        gf.psd(x, nperseg=2048, sample_rate_hertz=FS)


def test_estimate_psd_refuses_too_few_segments():
    with pytest.raises(ValueError, match="Welch segment"):
        estimate_psd(np.zeros((1, 1, int(1.5 * FS)), dtype=np.float32), FS)


def test_estimate_psd_is_scale_free():
    rng = np.random.default_rng(4)
    x = coloured_noise(rng, (2, 1), 16.0)
    physical = estimate_psd(x.astype(np.float32), FS)
    internal = estimate_psd((x * SCALE).astype(np.float32), FS)
    np.testing.assert_allclose(np.asarray(physical.normalised), np.asarray(internal.normalised), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(internal.scale) / np.asarray(physical.scale), SCALE, rtol=1e-5)


def test_inverse_bias_matches_the_chi_squared_formula_for_disjoint_segments():
    # Without overlap the mean of K periodograms is chi^2 with 2K degrees of
    # freedom over 2K, so E[P / P_hat] = 2K / (2K - 2).
    k = 8
    assert abs(inverse_bias(256, 0, k, "mean") - (2 * k) / (2 * k - 2)) < 3e-3


# --- whitening ----------------------------------------------------------------

def test_planck_window_is_zero_at_both_ends():
    window = np.asarray(gf_whiten.planck(64, 8, 8))
    assert window[0] == 0.0 and window[-1] == 0.0
    np.testing.assert_allclose(window[8:56], 1.0)


@pytest.mark.parametrize("with_chirp", [False, True])
def test_whiten_matches_gwpy(with_chirp):
    from gwpy.timeseries import TimeSeries
    rng = np.random.default_rng(5)
    off = coloured_noise(rng, (1,), 16.0)[0] * SCALE
    x = coloured_noise(rng, (1,), 4.0)[0] * SCALE
    if with_chirp:
        x = x + chirp(4.0) * SCALE
    asd = TimeSeries(off, sample_rate=FS).asd(fftlength=1.0, overlap=0.5, method="median")
    # gwpy ramps its transfer up from the highpass frequency over 5 bins of
    # 1 / duration; gravyflow ramps up to the band edge over the same 5 bins.
    ref = TimeSeries(x, sample_rate=FS).whiten(asd=asd, fduration=1.0, highpass=20.0 - 5 / 4.0).value
    config = SpectralConfig(lower_frequency_hertz=20.0, psd_floor=0.0, correct_inverse_bias=False)
    spectrum = estimate_psd(off[None].astype(np.float32), FS, config)
    mine = np.asarray(gf.whiten(x[None].astype(np.float32), spectrum, FS))[0]
    keep = slice(int(0.5 * FS), -int(0.5 * FS))
    assert np.max(np.abs(mine[keep] - ref[keep])) < 1e-4 * np.std(ref[keep])


@pytest.mark.parametrize("scale", [1.0, SCALE])
def test_whitened_gaussian_noise_has_unit_variance_at_any_scale(scale):
    rng = np.random.default_rng(6)
    on = coloured_noise(rng, (48, 1), 2.0) * scale
    off = coloured_noise(rng, (48, 1), 16.0) * scale
    spectrum = estimate_psd(off.astype(np.float32), FS, SpectralConfig(lower_frequency_hertz=None))
    whitened = np.asarray(gf.whiten(on.astype(np.float32), spectrum, FS))[..., 1024:-1024]
    assert whitened.dtype == np.float32
    # The Nyquist taper removes about 0.3% of the variance.
    assert abs(np.var(whitened) - 1.0) < 0.02, np.var(whitened)


def test_whitening_without_the_correction_is_biased_high():
    # The control for the test above: the inverse of a noisy PSD estimate whitens
    # too strongly on average.
    rng = np.random.default_rng(6)
    on = coloured_noise(rng, (48, 1), 2.0)
    off = coloured_noise(rng, (48, 1), 16.0)
    config = SpectralConfig(lower_frequency_hertz=None, correct_inverse_bias=False)
    whitened = np.asarray(gf.whiten(on.astype(np.float32), estimate_psd(off.astype(np.float32), FS, config),
                                    FS))[..., 1024:-1024]
    assert np.var(whitened) > 1.02


def test_whiten_refuses_a_window_shorter_than_its_filter():
    x = np.zeros((2, 1, 1638), dtype=np.float32)
    off = np.random.default_rng(7).standard_normal((2, 1, int(16 * FS))).astype(np.float32)
    with pytest.raises(ValueError, match="shorter than"):
        gf.whiten(x, off, FS)


def test_whiten_keeps_the_length_and_broadcasts_leading_axes():
    rng = np.random.default_rng(8)
    off = rng.standard_normal((3, 2, int(16 * FS))).astype(np.float32)
    x = rng.standard_normal((4, 3, 2, int(2 * FS))).astype(np.float32)
    spectrum = estimate_psd(off, FS)
    out = np.asarray(gf.whiten(x, spectrum, FS))
    assert out.shape == x.shape
    np.testing.assert_allclose(out[1], np.asarray(gf.whiten(x[1], spectrum, FS)), atol=1e-5)


def test_whiten_band_removes_content_outside_it():
    rng = np.random.default_rng(9)
    off = rng.standard_normal((1, 1, int(16 * FS))).astype(np.float32)
    t = np.arange(int(4 * FS)) / FS
    low = np.sin(2 * np.pi * 8.0 * t).astype(np.float32)[None, None]
    inside = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)[None, None]
    spectrum = estimate_psd(off, FS, SpectralConfig(lower_frequency_hertz=20.0, upper_frequency_hertz=500.0))
    keep = slice(1024, -1024)
    out_low = np.asarray(gf.whiten(low, spectrum, FS))[..., keep]
    out_in = np.asarray(gf.whiten(inside, spectrum, FS))[..., keep]
    assert np.std(out_low) < 1e-3 * np.std(out_in)


def test_whiten_floor_bounds_the_gain_of_an_empty_band():
    # Noise with nothing above 700 Hz: the floor, relative to the in-band median,
    # caps the filter's gain there at 1/sqrt(psd_floor) times the typical level.
    rng = np.random.default_rng(10)
    off = scipy.signal.sosfiltfilt(scipy.signal.butter(16, 700.0, fs=FS, output="sos"),
                                   rng.standard_normal((1, 1, int(16 * FS)))).astype(np.float32)
    spectrum = estimate_psd(off, FS)
    psd = np.asarray(spectrum.normalised)[0, 0]
    band = psd[20:1024]
    floor = spectrum.config.psd_floor * np.median(band)
    assert band.min() >= floor * (1 - 1e-6)
    assert np.all(psd[800:1000] <= floor * (1 + 1e-6))


def test_whiten_glitch_in_the_offsource_barely_moves_the_median_psd():
    rng = np.random.default_rng(11)
    off = rng.standard_normal((16, 1, int(16 * FS)))
    on = rng.standard_normal((16, 1, int(2 * FS)))
    glitch = 40.0 * sine_gaussian(150.0, 9.0, 1.0)
    off[..., int(7 * FS):int(8 * FS)] += glitch
    out = {}
    for method in ("mean", "median"):
        spectrum = estimate_psd(off.astype(np.float32), FS, SpectralConfig(psd_method=method))
        w = np.asarray(gf.whiten(on.astype(np.float32), spectrum, FS))[..., 1024:-1024]
        f, p = scipy.signal.welch(w, fs=FS, nperseg=256, axis=-1)
        out[method] = np.mean(p[..., (f >= 130) & (f <= 170)]) / (2 / FS)
    assert out["median"] > 0.9, out
    assert out["mean"] < 0.8, out


def test_whiten_layer_honours_its_arguments():
    rng = np.random.default_rng(12)
    on = rng.standard_normal((2, 1, int(2 * FS))).astype(np.float32)
    off = rng.standard_normal((2, 1, int(16 * FS))).astype(np.float32)
    layer = gf.Whiten(sample_rate_hertz=FS, onsource_duration_seconds=1.0, highpass_hertz=200.0,
                      fft_duration_seconds=2.0, overlap_duration_seconds=1.0, filter_duration_seconds=1.0)
    expected = gf.whiten(on, off, FS, highpass_hertz=200.0, fft_duration_seconds=2.0,
                         overlap_duration_seconds=1.0, filter_duration_seconds=1.0)
    expected = gf.crop_samples(expected, 1.0, FS)
    np.testing.assert_allclose(np.asarray(layer([on, off])), np.asarray(expected), atol=1e-6)
    config = layer.get_config()
    assert config["highpass_hertz"] == 200.0 and config["fft_duration_seconds"] == 2.0


def test_fftconvolve_pads_to_a_fast_length_and_matches_numpy():
    rng = np.random.default_rng(13)
    a = rng.standard_normal(4096).astype(np.float32)
    b = rng.standard_normal(2048).astype(np.float32)
    np.testing.assert_allclose(np.asarray(gf_whiten.fftconvolve(a, b, mode="same")),
                               scipy.signal.convolve(a, b, mode="same"), atol=2e-3)


# --- SNR ----------------------------------------------------------------------

@pytest.mark.parametrize("band", [(20.0, None), (30.0, 512.0)])
def test_snr_matches_pycbc_sigma(band):
    import pycbc.filter
    import pycbc.types
    f_low, f_high = band
    n = int(4 * FS)
    p = design_psd(np.fft.rfftfreq(n, 1 / FS)) * SCALE ** 2
    h = (chirp(4.0) * SCALE).astype(np.float32)
    config = SpectralConfig(lower_frequency_hertz=f_low, upper_frequency_hertz=f_high)
    spectrum = PowerSpectrum(np.asarray(p * FS, np.float32)[None, None], np.ones((1, 1, 1), np.float32), FS, config)
    mine = float(np.asarray(gf.snr(h[None, None], spectrum, FS))[0])
    hf = pycbc.types.TimeSeries(h.astype(np.float64), delta_t=1 / FS).to_frequencyseries()
    ref = float(pycbc.filter.sigma(hf, pycbc.types.FrequencySeries(p, delta_f=hf.delta_f),
                                   low_frequency_cutoff=f_low, high_frequency_cutoff=f_high))
    assert abs(mine / ref - 1) < 1e-4
    optimal = float(np.asarray(gf_snr.optimal_snr(h, psd=p.astype(np.float32), sample_rate_hertz=FS,
                                                  lower_frequency_cutoff=f_low, upper_frequency_cutoff=f_high)))
    assert abs(optimal / ref - 1) < 1e-4


def test_snr_is_scale_free():
    rng = np.random.default_rng(14)
    off = coloured_noise(rng, (2, 1), 16.0)
    h = np.repeat(sine_gaussian(100.0, 9.0, 1.0)[None, None] * 2e-22, 2, axis=0)
    internal = np.asarray(gf.snr((h * SCALE).astype(np.float32), (off * SCALE).astype(np.float32), FS))
    physical = np.asarray(gf.snr(h.astype(np.float32), off.astype(np.float32), FS))
    np.testing.assert_allclose(physical, internal, rtol=1e-4)
    assert np.all(internal > 1.0)


def test_snr_of_batch_time_input_is_one_value_per_row():
    rng = np.random.default_rng(15)
    off = rng.standard_normal((3, int(16 * FS))).astype(np.float32)
    h = sine_gaussian(100.0, 9.0, 1.0).astype(np.float32)
    rows = np.stack([h, 2 * h, 3 * h])
    two_d = np.asarray(gf.snr(rows, off, FS))
    three_d = np.asarray(gf.snr(rows[:, None], off[:, None], FS))
    assert two_d.shape == (3,)
    np.testing.assert_allclose(two_d, three_d, rtol=1e-5)


def test_network_snr_is_the_quadrature_sum_over_detectors():
    rng = np.random.default_rng(16)
    off = rng.standard_normal((2, 2, int(16 * FS))).astype(np.float32)
    h = sine_gaussian(100.0, 9.0, 1.0).astype(np.float32)
    x = np.stack([np.stack([h, 2 * h]), np.stack([3 * h, h])])
    spectrum = estimate_psd(off, FS)
    network = np.asarray(gf.snr(x, spectrum, FS))
    per_detector = np.stack([np.asarray(gf.snr(x[:, i], gf.PowerSpectrum(
        spectrum.normalised[:, i], spectrum.scale[:, i], FS, spectrum.config, spectrum.inverse_bias,
        spectrum.num_segments), FS)) for i in range(2)], axis=-1)
    np.testing.assert_allclose(network, np.sqrt(np.sum(per_detector ** 2, axis=-1)), rtol=1e-5)


def test_scale_to_snr_scales_each_example_to_its_own_target():
    rng = np.random.default_rng(17)
    off = rng.standard_normal((3, 1, int(16 * FS))).astype(np.float32)
    h = np.repeat(sine_gaussian(100.0, 9.0, 2.0)[None, None], 3, axis=0).astype(np.float32)
    targets = np.array([5.0, 10.0, 20.0], np.float32)
    spectrum = estimate_psd(off, FS)
    scaled = gf.scale_to_snr(h, spectrum, targets, FS, onsource_duration_seconds=1.0)
    measured = np.asarray(gf.snr(gf.crop_samples(scaled, 1.0, FS), spectrum, FS))
    np.testing.assert_allclose(measured, targets, rtol=1e-4)
    # Huge or tiny amplitudes do not overflow the power sums.
    for factor in (1e20, 1e-20):
        again = gf.scale_to_snr(h * factor, spectrum, targets, FS, onsource_duration_seconds=1.0)
        np.testing.assert_allclose(np.asarray(again), np.asarray(scaled), rtol=1e-3, atol=1e-6)


def test_scale_to_snr_gives_zero_for_no_power_in_the_band():
    off = np.random.default_rng(18).standard_normal((1, 1, int(16 * FS))).astype(np.float32)
    scaled = gf.scale_to_snr(np.zeros((1, 1, int(FS)), np.float32), off, np.array([10.0], np.float32), FS)
    assert np.all(np.isfinite(np.asarray(scaled))) and np.all(np.asarray(scaled) == 0)


def test_scaled_injection_carries_its_target_in_the_whitened_data():
    rng = np.random.default_rng(19)
    off = coloured_noise(rng, (8, 1), 16.0) * SCALE
    h = np.repeat(sine_gaussian(150.0, 9.0, 2.0)[None, None], 8, axis=0) * SCALE * 1e-22
    spectrum = estimate_psd(off.astype(np.float32), FS)
    scaled = gf.scale_to_snr(h.astype(np.float32), spectrum, np.full(8, 10.0, np.float32), FS,
                             onsource_duration_seconds=1.0)
    whitened = np.asarray(gf.crop_samples(gf.whiten(scaled, spectrum, FS), 1.0, FS), dtype=np.float64)
    norm = np.sqrt(np.sum(whitened ** 2, axis=(-1, -2)))
    np.testing.assert_allclose(norm, 10.0, rtol=0.02)


def test_target_snr_is_unbiased_against_the_true_psd():
    # The SNR a target-10 injection really has, with the true PSD, averaged over
    # noise draws: the estimator's inverse bias would leave it about 3% low.
    import pycbc.filter
    import pycbc.types
    rng = np.random.default_rng(20)
    draws = 48
    off = coloured_noise(rng, (draws, 1), 16.0) * SCALE
    h = np.repeat(sine_gaussian(100.0, 9.0, 1.0)[None, None], draws, axis=0) * SCALE * 1e-22
    scaled = np.asarray(gf.scale_to_snr(h.astype(np.float32), off.astype(np.float32),
                                        np.full(draws, 10.0, np.float32), FS), dtype=np.float64) / SCALE
    p = pycbc.types.FrequencySeries(design_psd(np.fft.rfftfreq(int(FS), 1 / FS)), delta_f=1.0)
    true = [float(pycbc.filter.sigma(pycbc.types.TimeSeries(s[0], delta_t=1 / FS).to_frequencyseries(), p,
                                     low_frequency_cutoff=20.0)) for s in scaled]
    assert abs(np.mean(true) / 10.0 - 1.0) < 0.015, np.mean(true)


# --- matched filter -------------------------------------------------------------

@pytest.mark.parametrize("phase", [0.0, np.pi / 2])
def test_matched_filter_recovers_the_true_snr_of_whitened_data(phase):
    fs = 4096.0
    template = sine_gaussian(200.0, 3.0, 1.0, fs=fs)
    data = sine_gaussian(200.0, 3.0, 1.0, fs=fs, phase=phase)
    data = data / float(np.asarray(gf_snr.optimal_snr(data.astype(np.float32), None, fs))) * 10.0
    series = np.asarray(gf_snr.matched_filter_fft(data.astype(np.float32), template[None].astype(np.float32),
                                                  psd=None, sample_rate_hertz=fs))
    assert abs(np.max(series) - 10.0) < 1e-2
    assert int(np.argmax(series[0])) == 0
