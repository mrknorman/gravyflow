import numpy as np
import pytest
import scipy.signal
from keras import ops

from gravyflow.src.dataset.tools import psd as gf_psd
from gravyflow.tests._helpers import representation_tolerance, statistical_bound


def welch_relative_sigma(nperseg: int, noverlap: int, num_segments: int) -> float:
    """
    The relative standard deviation of a mean-averaged Welch estimate of
    Gaussian noise, at a bin away from DC and Nyquist, with the periodic Hann
    window (Percival and Walden, eq. 292b): sqrt((1 + 2 sum_j (1 - j / K) rho_j) / K),
    rho_j the squared overlap correlation of segments j steps apart.
    """
    window = gf_psd.hann(nperseg).astype(np.float64)
    step, norm = nperseg - noverlap, np.sum(window ** 2)
    total = 1.0
    for j in range(1, num_segments):
        lag = j * step
        if lag >= nperseg:
            break
        total += 2.0 * (1.0 - j / num_segments) * (np.sum(window[:nperseg - lag] * window[lag:]) / norm) ** 2
    return float(np.sqrt(total / num_segments))


#: Bins of a Hann-windowed periodogram this many apart are uncorrelated: the
#: squared Hann window's DFT has no term beyond two bins.
INDEPENDENT_BIN_STEP = 3


def spectral_cv(psd, keep) -> tuple:
    """
    The coefficient of variation of a mean Welch estimate over the bins ``keep``
    (taken every INDEPENDENT_BIN_STEP-th), and the number of bins it used.
    """
    values = np.asarray(psd)[keep][::INDEPENDENT_BIN_STEP]
    return float(np.std(values) / np.mean(values)), len(values)


def cv_sigma(relative_sigma: float, num_bins: int) -> float:
    """
    The standard deviation of the sample coefficient of variation of
    ``num_bins`` independent estimates with relative sigma s, each Gamma
    distributed (excess kurtosis 6 s^2): s * sqrt((2 + 6 s^2) / num_bins) / 2.
    """
    s = relative_sigma
    return float(s * np.sqrt((2.0 + 6.0 * s * s) / num_bins) / 2.0)


def test_detrend_constant():
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    detrended = gf_psd.detrend(data, type='constant')
    np.testing.assert_allclose(detrended, np.array([[-1.0, 0.0, 1.0]]))


def test_detrend_linear():
    # y = 2x + 1 -> [1, 3, 5]
    data = ops.convert_to_tensor([[1.0, 3.0, 5.0]], dtype="float32")
    detrended = gf_psd.detrend(data, type='linear')
    # Representation: a least-squares fit of three float32 values up to 5.
    np.testing.assert_allclose(detrended, np.zeros((1, 3)), rtol=0.0,
                               atol=representation_tolerance(np.float32, 5.0, 8))


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_detrend_invalid_type():
    """Verify detrend raises error for invalid type parameter."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="Trend type must be"):
        gf_psd.detrend(data, type='invalid')


def test_detrend_breakpoints_not_supported():
    """Verify detrend raises NotImplementedError for breakpoints."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype="float32")

    with pytest.raises(NotImplementedError, match="Breakpoints not yet supported"):
        gf_psd.detrend(data, type='linear', bp=1)


def test_psd_invalid_mode():
    """Verify psd raises error for unsupported mode."""
    signal = ops.convert_to_tensor([[1.0, 2.0, 3.0, 4.0] * 100], dtype="float32")

    with pytest.raises(ValueError, match="Mode not supported"):
        gf_psd.psd(signal, nperseg=100, sample_rate_hertz=100.0, mode='invalid')


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_psd_1d_input():
    """A 1D signal has the PSD of the same signal as a one-row batch."""
    sample_rate = 100.0
    t = np.arange(int(sample_rate)) / sample_rate
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)

    spectrum = gf_psd.psd(ops.convert_to_tensor(signal), nperseg=100, sample_rate_hertz=sample_rate)
    batched = gf_psd.psd(ops.convert_to_tensor(signal[None]), nperseg=100, sample_rate_hertz=sample_rate)
    assert spectrum.value().shape == (51,)
    np.testing.assert_array_equal(spectrum.value(), batched.value()[0])
    # Exact: the 10 Hz sine is bin-centred (1 Hz bins).
    assert int(np.argmax(spectrum.value())) == 10


def test_detrend_type_aliases():
    """Test detrend accepts 'c' and 'l' type aliases."""
    data = ops.convert_to_tensor([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(gf_psd.detrend(data, type='c'), gf_psd.detrend(data, type='constant'))

    data_linear = ops.convert_to_tensor([[1.0, 3.0, 5.0]], dtype="float32")
    np.testing.assert_array_equal(gf_psd.detrend(data_linear, type='l'),
                                  gf_psd.detrend(data_linear, type='linear'))


# ============================================================================
# NUMERICAL CORRECTNESS TESTS
# ============================================================================

@pytest.mark.parametrize("freq", [50.0, 100.0, 200.0])
def test_psd_pure_sine_peak_location(freq):
    """A bin-centred sine peaks in its own bin, exactly."""
    sample_rate, nperseg = 1024.0, 1024
    t = np.arange(int(4 * sample_rate)) / sample_rate
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    spectrum = gf_psd.psd(ops.convert_to_tensor(signal[None]), nperseg=nperseg, sample_rate_hertz=sample_rate)
    assert int(np.argmax(spectrum.value()[0])) == int(freq * nperseg / sample_rate)


def test_psd_white_noise_flatness():
    """
    The PSD of white noise is flat at 2 sigma^2 / fs, with the scatter its
    segment count gives: the coefficient of variation over independent bins is
    the estimator's relative sigma (welch_relative_sigma), within k sigma of
    the sample CV.
    """
    sample_rate, nperseg = 1024.0, 1024
    noise = np.random.default_rng(42).standard_normal(int(8 * sample_rate)).astype(np.float32)
    spectrum = gf_psd.psd(ops.convert_to_tensor(noise[None]), nperseg=nperseg, sample_rate_hertz=sample_rate)
    num_segments = spectrum.num_segments
    s = welch_relative_sigma(nperseg, nperseg // 2, num_segments)
    cv, num_bins = spectral_cv(spectrum.value()[0], slice(5, -5))
    assert abs(cv - s) < statistical_bound(cv_sigma(s, num_bins)), \
        f"CV {cv:.3f} over {num_bins} bins; {num_segments} segments give {s:.3f}"
    # Statistical: the mean over independent bins of the level 2 / fs.
    level = np.mean(spectrum.value()[0][5:-5:INDEPENDENT_BIN_STEP]) * sample_rate / 2.0
    assert abs(level - 1.0) < statistical_bound(s / np.sqrt(num_bins))


def test_detrend_linear_non_default_axis():
    """Linear detrend along axis 1 of (batch, time, channel) data matches scipy's, each column its own trend."""
    rng = np.random.default_rng(0)
    slopes = rng.uniform(-3.0, 3.0, size=(2, 1, 3))
    offsets = rng.uniform(-5.0, 5.0, size=(2, 1, 3))
    data = (slopes * np.arange(5)[None, :, None] + offsets + rng.standard_normal((2, 5, 3))).astype(np.float32)
    detrended = np.asarray(gf_psd.detrend(ops.convert_to_tensor(data), type='linear', axis=1))
    expected = scipy.signal.detrend(data.astype(np.float64), axis=1, type='linear')
    # Representation: a float32 least-squares fit of five values of the data's scale.
    np.testing.assert_allclose(detrended, expected, rtol=0.0,
                               atol=representation_tolerance(np.float32, np.abs(data).max(), 16))
