import numpy as np
import pytest
import scipy.signal
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.conditioning import whiten as gf_whiten
from gravyflow.tests._helpers import representation_tolerance, statistical_bound
from gravyflow.tests.test_psd import welch_relative_sigma


def test_whitening_removes_spectral_lines(tmp_path):
    """
    Two sines on unit white noise, whitened with their own PSD: the raw data's
    line bins stand far above the noise, the whitened data's are level with it.
    """
    rate, seconds, nperseg = 4096.0, 16.0, 4096
    lines = {210.0: 5.0, 543.0: 3.0}  # Hz: amplitude
    t = np.arange(int(rate * seconds)) / rate
    rng = np.random.default_rng(42)
    data = (sum(a * np.sin(2 * np.pi * f * t) for f, a in lines.items())
            + rng.standard_normal(t.size)).astype(np.float32)
    whitened = np.asarray(gf.whiten(data, data, rate, fft_duration_seconds=1.0, overlap_duration_seconds=0.5))
    # The filter's first and last half-lengths (0.5 s) are corrupted.
    whitened = whitened[int(0.5 * rate):-int(0.5 * rate)]

    def line_levels(series):
        """Each line bin's mean Welch estimate over the median of the bins around it."""
        f, p = scipy.signal.welch(series, fs=rate, nperseg=nperseg, noverlap=nperseg // 2)
        levels = []
        for line in lines:
            k = int(round(line * nperseg / rate))
            levels.append(p[k] / np.median(p[k - 64:k + 65]))
        return np.array(levels), f, p

    raw, f, raw_psd = line_levels(data)
    flat, _, flat_psd = line_levels(whitened)
    # Statistical: a mean Welch estimate of Gaussian noise has relative sigma
    # welch_relative_sigma at every bin, so a whitened line bin sits within k
    # sigma of the level; the raw lines (A^2 / 2 of power in about one bin, against
    # 2 / fs per hertz) stand far above it, the control.
    num_segments = (whitened.size - nperseg) // (nperseg // 2) + 1
    bound = 1.0 + statistical_bound(welch_relative_sigma(nperseg, nperseg // 2, num_segments))
    assert np.all(raw > 100.0 * bound), f"the control's lines are not visible: {raw}"
    assert np.all(flat < bound), f"whitened line levels {flat}, above {bound:.3f}"

    plot_path = tmp_path / "whitening_lines.html"
    output_file(plot_path)
    save(gridplot([[gf.generate_psd_plot(psd={"raw": raw_psd}, frequencies=f, title="Raw")],
                   [gf.generate_psd_plot(psd={"whitened": flat_psd}, frequencies=f, title="Whitened")]]))
    assert plot_path.stat().st_size > 0


def test_planck():
    N = 100
    nleft = 10
    nright = 10
    window = gf_whiten.planck(N, nleft, nright)
    assert ops.shape(window) == (N,)
    window = np.asarray(window)
    # A Planck taper reaches exactly 0 at both ends and is flat in the middle.
    assert window[0] == 0.0 and window[-1] == 0.0
    np.testing.assert_allclose(window[nleft:N - nright], 1.0)
    assert np.all(np.diff(window[:nleft]) > 0) and np.all(np.diff(window[N - nright:]) < 0)

def test_fftconvolve():
    in1, in2 = [1.0, 2.0, 3.0], [0.0, 1.0, 0.5]
    for mode in ("full", "same"):
        conv = gf_whiten.fftconvolve(ops.convert_to_tensor(in1), ops.convert_to_tensor(in2), mode=mode)
        expected = np.convolve(in1, in2, mode=mode)
        # Representation: an FFT convolution of a few float32 values up to the output's peak.
        np.testing.assert_allclose(conv, expected, rtol=0.0,
                                   atol=representation_tolerance(np.float32, np.abs(expected).max(), 8))


def test_whiten_colored_noise():
    """
    Brownian noise (1 / f^2), whitened with its own PSD, has a flat spectrum:
    the scatter of its Welch estimate over independent bins is no more than
    white noise's (whitening with its own estimate follows some of the data's
    own fluctuations, so it is lower). The raw noise, the control, is far from flat.
    """
    from gravyflow.tests.test_psd import cv_sigma, spectral_cv, welch_relative_sigma

    sample_rate, nperseg = 1024.0, 1024
    rng = np.random.default_rng(42)
    colored = scipy.signal.detrend(np.cumsum(rng.standard_normal(int(16 * sample_rate))))
    colored = (colored / np.std(colored)).astype(np.float32)
    whitened = np.asarray(gf_whiten.whiten(colored[None, None], colored[None, None], sample_rate_hertz=sample_rate,
                                           fft_duration_seconds=1.0, overlap_duration_seconds=0.5))[0, 0]

    f, pxx = scipy.signal.welch(whitened, fs=sample_rate, nperseg=nperseg)
    keep = (f > 20) & (f < sample_rate / 2 - 20)
    cv, num_bins = spectral_cv(pxx, keep)
    # Statistical: white noise's CV is the estimator's relative sigma s, with the
    # sample CV's own sigma; the bound is one-sided (see above).
    s = welch_relative_sigma(nperseg, nperseg // 2, (whitened.size - nperseg) // (nperseg // 2) + 1)
    assert cv < s + statistical_bound(cv_sigma(s, num_bins)), f"whitened CV {cv:.3f}, white noise gives {s:.3f}"

    _, raw = scipy.signal.welch(colored, fs=sample_rate, nperseg=nperseg)
    assert spectral_cv(raw, keep)[0] > 10 * s, "the control is not coloured"




def test_truncate_transfer_ncorner_validation():
    """Test that truncate_transfer raises ValueError for invalid ncorner."""
    transfer = ops.ones((10,), dtype="float32")
    with pytest.raises(ValueError, match="ncorner must be less than"):
        gf_whiten.truncate_transfer(transfer, ncorner=15)  # ncorner >= nsamp


def test_truncate_impulse_odd_ntaps():
    """Test that truncate_impulse raises ValueError for odd ntaps."""
    impulse = ops.ones((100,), dtype="float32")
    with pytest.raises(ValueError, match="ntaps must be an even number"):
        gf_whiten.truncate_impulse(impulse, ntaps=51)


def test_truncate_impulse_unsupported_window():
    """Test that truncate_impulse raises ValueError for unsupported window."""
    impulse = ops.ones((100,), dtype="float32")
    with pytest.raises(ValueError, match="Window function .* not supported"):
        gf_whiten.truncate_impulse(impulse, ntaps=50, window="hamming")


def test_fir_from_transfer_odd_ntaps():
    """Test that fir_from_transfer raises ValueError for odd ntaps."""
    transfer = ops.ones((100,), dtype="float32")
    with pytest.raises(ValueError, match="ntaps must be an even number"):
        gf_whiten.fir_from_transfer(transfer, ntaps=51)


def test_fftconvolve_valid_mode():
    """Test fftconvolve with mode='valid'."""
    in1 = ops.convert_to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    in2 = ops.convert_to_tensor([0.5, 0.5])
    
    conv = gf_whiten.fftconvolve(in1, in2, mode='valid')
    expected = np.convolve([1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5], mode='valid')
    # Representation: an FFT convolution of a few float32 values up to the output's peak.
    np.testing.assert_allclose(conv, expected, rtol=0.0, atol=representation_tolerance(np.float32, 4.5, 8))


def test_fftconvolve_invalid_mode():
    """Test fftconvolve with invalid mode raises ValueError."""
    in1 = ops.convert_to_tensor([1.0, 2.0, 3.0])
    in2 = ops.convert_to_tensor([0.5, 0.5])
    
    with pytest.raises(ValueError, match="Acceptable mode flags"):
        gf_whiten.fftconvolve(in1, in2, mode='invalid')


def test_convolve_unsupported_window():
    """Test that convolve raises ValueError for unsupported window."""
    timeseries = ops.ones((100,), dtype="float32")
    fir = ops.ones((10,), dtype="float32")
    
    with pytest.raises(ValueError, match="Window function .* not supported"):
        gf_whiten.convolve(timeseries, fir, window="hamming")


def test_whiten_invalid_highpass():
    """Test that whiten raises ValueError for invalid highpass frequency."""
    sample_rate = 100.0
    timeseries = ops.ones((100,), dtype="float32")
    background = ops.ones((100,), dtype="float32")
    
    # Negative highpass
    with pytest.raises(ValueError, match="Invalid highpass frequency"):
        gf_whiten.whiten(timeseries, background, sample_rate, highpass_hertz=-10.0)
    
    # Highpass >= Nyquist
    with pytest.raises(ValueError, match="Invalid highpass frequency"):
        gf_whiten.whiten(timeseries, background, sample_rate, highpass_hertz=60.0)


def test_whiten_invalid_filter_duration():
    """Test that whiten raises ValueError for invalid filter duration."""
    sample_rate = 100.0
    timeseries = ops.ones((100,), dtype="float32")
    background = ops.ones((100,), dtype="float32")
    
    with pytest.raises(ValueError, match="Filter duration should be greater than zero"):
        gf_whiten.whiten(timeseries, background, sample_rate, filter_duration_seconds=0.0)


def test_whiten_invalid_window():
    """Test that whiten raises ValueError for invalid window type."""
    sample_rate = 100.0
    timeseries = ops.ones((100,), dtype="float32")
    background = ops.ones((100,), dtype="float32")
    
    with pytest.raises(ValueError, match="Invalid window type"):
        gf_whiten.whiten(timeseries, background, sample_rate, window="hamming")


def test_whiten_1d_input():
    """A 1D series is whitened as a one-row batch and comes back 1D."""
    sample_rate = 100.0
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(int(4 * sample_rate)).astype(np.float32)
    background = rng.standard_normal(int(8 * sample_rate)).astype(np.float32)
    settings = dict(sample_rate_hertz=sample_rate, fft_duration_seconds=1.0, overlap_duration_seconds=0.5)
    whitened = np.asarray(gf_whiten.whiten(signal, background, **settings))
    batched = np.asarray(gf_whiten.whiten(signal[None], background[None], **settings))
    assert whitened.shape == signal.shape
    np.testing.assert_array_equal(whitened, batched[0])

# ============================================================================
# KERAS LAYER TESTS
# ============================================================================

def test_whiten_pass_layer():
    """WhitenPass crops the onsource and scales each example to unit RMS, without whitening."""
    rate, seconds = 2048.0, 4.0
    layer = gf_whiten.WhitenPass(sample_rate_hertz=rate, onsource_duration_seconds=seconds,
                                 fft_duration_seconds=1.0, overlap_duration_seconds=0.5, highpass_hertz=20.0)
    rng = np.random.default_rng(1)
    inputs = (rng.standard_normal((4, 2, int(seconds * rate) + 1024))
              * np.array([1e-21, 1.0, 3e2, 5e-3])[:, None, None]).astype(np.float32)
    output = np.asarray(layer(ops.convert_to_tensor(inputs)), np.float64)
    assert output.shape == (4, 2, int(seconds * rate))
    cropped = inputs[..., 512:-512].astype(np.float64)
    rms = np.sqrt(np.mean(cropped ** 2, axis=(1, 2), keepdims=True))
    # Representation: a float32 mean of squares over 2 x 8192 samples, scaled to unit RMS.
    np.testing.assert_allclose(output, cropped / rms, rtol=0.0,
                               atol=representation_tolerance(np.float32, np.abs(cropped / rms).max(),
                                                             np.sqrt(output[0].size)))


def test_whiten_pass_layer_get_config():
    """Test WhitenPass get_config method."""
    layer = gf_whiten.WhitenPass(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=4.0,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        highpass_hertz=20.0,
        detrend='constant',
        filter_duration_seconds=2.0,
        window='hann'
    )
    
    config = layer.get_config()
    
    assert config['sample_rate_hertz'] == 2048.0
    assert config['fft_duration_seconds'] == 1.0
    assert config['overlap_duration_seconds'] == 0.5
    assert config['highpass_hertz'] == 20.0
    assert config['detrend'] == 'constant'
    assert config['filter_duration_seconds'] == 2.0
    assert config['window'] == 'hann'


def test_whiten_pass_layer_compute_output_shape():
    """Test WhitenPass compute_output_shape method."""
    layer = gf_whiten.WhitenPass(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=4.0
    )
    
    input_shape = (4, 1, 10000)
    output_shape = layer.compute_output_shape(input_shape)
    
    expected_samples = int(4.0 * 2048.0)
    assert output_shape == (4, 1, expected_samples)


def test_whiten_layer_get_config():
    """Test Whiten get_config method."""
    layer = gf_whiten.Whiten(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=4.0,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        highpass_hertz=20.0,
        detrend='constant',
        filter_duration_seconds=2.0,
        window='hann'
    )
    
    config = layer.get_config()
    
    assert config['sample_rate_hertz'] == 2048.0
    assert config['fft_duration_seconds'] == 1.0
    assert config['overlap_duration_seconds'] == 0.5
    assert config['highpass_hertz'] == 20.0
    assert config['detrend'] == 'constant'
    assert config['filter_duration_seconds'] == 2.0
    assert config['window'] == 'hann'


def test_whiten_layer_compute_output_shape():
    """Test Whiten compute_output_shape method."""
    layer = gf_whiten.Whiten(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=4.0
    )
    
    timeseries_shape = (4, 1, 10000)
    background_shape = (4, 1, 10000)
    output_shape = layer.compute_output_shape((timeseries_shape, background_shape))
    
    expected_samples = int(4.0 * 2048.0)
    assert output_shape == (4, 1, expected_samples)


def test_whiten_layer_defaults():
    """Test Whiten layer with default parameters."""
    layer = gf_whiten.Whiten()
    
    # Should use Defaults values
    assert layer.sample_rate_hertz == gf.Defaults.sample_rate_hertz
    assert layer.onsource_duration_seconds == gf.Defaults.onsource_duration_seconds


def test_whiten_pass_layer_defaults():
    """Test WhitenPass layer with default parameters."""
    layer = gf_whiten.WhitenPass()
    
    # Should use Defaults values
    assert layer.sample_rate_hertz == gf.Defaults.sample_rate_hertz
    assert layer.onsource_duration_seconds == gf.Defaults.onsource_duration_seconds


def test_truncate_transfer_basic():
    """Zero below ncorner, a rising ramp to full gain, and the mirror-image ramp down at the top."""
    result = np.asarray(gf_whiten.truncate_transfer(ops.ones((100,), dtype="float32"), ncorner=10))
    ramp = gf_whiten.EDGE_TAPER_BINS
    np.testing.assert_array_equal(result[:11], np.zeros(11))
    assert np.all(np.diff(result[10:10 + ramp + 1]) > 0)
    np.testing.assert_array_equal(result[10 + ramp:100 - ramp], np.ones(80))
    np.testing.assert_array_equal(result[10:10 + ramp], result[100 - ramp:][::-1])


def test_truncate_impulse_basic():
    """The first and last ntaps / 2 samples keep a Hann window's halves; the middle is zeroed."""
    result = np.asarray(gf_whiten.truncate_impulse(ops.ones((100,), dtype="float32"), ntaps=20, window='hann'))
    window = scipy.signal.get_window("hann", 20).astype(np.float32)  # periodic, as the module's
    assert result.shape == (100,)
    np.testing.assert_array_equal(result[10:90], np.zeros(80))
    np.testing.assert_allclose(result[:10], window[10:], rtol=0.0, atol=representation_tolerance(np.float32, 1.0, 2))
    np.testing.assert_allclose(result[90:], window[:10], rtol=0.0, atol=representation_tolerance(np.float32, 1.0, 2))


def test_fir_from_transfer_basic():
    """
    The FIR is linear phase about tap ntaps / 2 - 1 (symmetric), and that tap is
    the zero-lag sample of the truncated transfer's impulse response (the Hann
    window is 1 at its centre).
    """
    transfer = ops.ones((100,), dtype="float32")
    result = np.asarray(gf_whiten.fir_from_transfer(transfer, ntaps=20))
    assert result.shape == (20,)
    centre = 9
    np.testing.assert_allclose(result[centre - 9:centre], result[centre + 1:centre + 10][::-1], rtol=0.0,
                               atol=representation_tolerance(np.float32, 1.0, np.log2(198)))
    zero_lag = np.fft.irfft(np.asarray(gf_whiten.truncate_transfer(transfer), np.float64))[0]
    np.testing.assert_allclose(result[centre], zero_lag, rtol=0.0,
                               atol=representation_tolerance(np.float32, 1.0, np.log2(198)))

