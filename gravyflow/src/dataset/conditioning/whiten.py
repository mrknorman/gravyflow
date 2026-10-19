"""Whitening by inverse spectrum truncation, as gwpy's TimeSeries.whiten does it.

The filter is the inverse ASD of the background, band-limited to the
SpectralConfig band with 5-bin Hann tapers just outside each edge, truncated to a
`filter_duration_seconds` FIR with a Hann window and applied by FFT
convolution. The output is scaled so that stationary Gaussian noise with the
background's PSD comes out with unit variance.

Scale free: the data are divided by the background's amplitude scale before any
spectrum is formed, so physical strain and scaled strain whiten identically.
"""
from functools import lru_cache
from math import ceil
from typing import Optional, Union

import keras
from keras import ops, Layer
import jax.numpy as jnp
import numpy as np
import jax
from scipy.fft import next_fast_len

from gravyflow.src.dataset.tools.psd import (
    PowerSpectrum, SpectralConfig, estimate_psd, hann, median_bias, detrend as _detrend
)
from gravyflow.src.dataset.config import Defaults
from gravyflow.src.utils.tensor import crop_samples

# Width, in frequency bins, of the Hann taper at each edge of the band (gwpy's).
EDGE_TAPER_BINS = 5


@jax.jit(static_argnames=["N", "nleft", "nright"])
def planck(N: int, nleft: int, nright: int):
    """
    Planck-taper window of length N, rising over the first `nleft` samples and
    falling over the last `nright`; exactly 0 at both ends and 1 in the middle.
    """
    window = np.ones(N, dtype=np.float64)
    if nleft:
        window[0] = 0.0
        k = np.arange(1, nleft, dtype=np.float64)
        window[1:nleft] = 1.0 / (np.exp(nleft * (1.0 / k + 1.0 / (k - nleft))) + 1.0)
    if nright:
        window[N - 1] = 0.0
        k = np.arange(1, nright, dtype=np.float64)
        window[N - nright:N - 1] = 1.0 / (np.exp(-nright * (1.0 / (k - nright) + 1.0 / k)) + 1.0)
    return ops.convert_to_tensor(window.astype(np.float32))


def _edge_taper(n: int) -> np.ndarray:
    """The rising half of a 2n-sample Hann (Tukey alpha=1) window: 0 at the edge."""
    full = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(2 * n) / (2 * n - 1))
    return full[:n].astype(np.float32)


@jax.jit(static_argnames=["ncorner", "nupper"])
def truncate_transfer(
    transfer,
    ncorner: int = 0,
    nupper: Optional[int] = None,
    ):
    """
    Zero the transfer function outside [ncorner, nupper) and taper
    EDGE_TAPER_BINS bins inside each edge with a Hann ramp that reaches 0 at the
    edge (gwpy.signal.filter_design.truncate_transfer, plus an upper edge).
    """
    nsamp = ops.shape(transfer)[-1]
    ncorner = ncorner if ncorner else 0
    nupper = nsamp if nupper is None else nupper

    if ncorner >= nsamp:
        raise ValueError(
            "ncorner must be less than the size of the transfer array"
        )
    if not ncorner < nupper <= nsamp:
        raise ValueError("nupper must be in (ncorner, size of the transfer array].")

    window = np.zeros(nsamp, dtype=np.float32)
    window[ncorner:nupper] = 1.0
    ntaper = min(EDGE_TAPER_BINS, (nupper - ncorner) // 2)
    if ntaper > 0:
        ramp = _edge_taper(ntaper)
        window[ncorner:ncorner + ntaper] *= ramp
        window[nupper - ntaper:nupper] *= ramp[::-1]
    return transfer * ops.convert_to_tensor(window)


@jax.jit(static_argnames=["ntaps", "window"])
def truncate_impulse(
    impulse,
    ntaps: int,
    window: str = 'hann'
    ):
    """
    Smoothly truncate a time domain impulse response to ntaps samples.
    """
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")

    trunc_start = int(ntaps / 2)
    trunc_stop = ops.shape(impulse)[-1] - trunc_start

    if window == 'hann':
        win = ops.convert_to_tensor(hann(ntaps))
    else:
        raise ValueError(f"Window function {window} not supported")

    impulse_start = impulse[..., :trunc_start] * win[trunc_start:ntaps]
    impulse_stop = impulse[..., trunc_stop:] * win[:trunc_start]
    impulse_middle = ops.zeros_like(impulse[..., trunc_start:trunc_stop])

    return ops.concatenate([impulse_start, impulse_middle, impulse_stop], axis=-1)


@jax.jit(static_argnames=["ntaps", "window", "ncorner", "nupper"])
def fir_from_transfer(
    transfer,
    ntaps: int,
    window: str = 'hann',
    ncorner: int = 0,
    nupper: Optional[int] = None,
    ):
    """
    Design a Type II FIR filter given an arbitrary transfer function.
    """
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")

    transfer = truncate_transfer(transfer, ncorner=ncorner, nupper=nupper)
    impulse = jnp.fft.irfft(ops.cast(transfer, "complex64"))
    if ops.shape(impulse)[-1] < ntaps:
        raise ValueError(
            f"The filter needs {ntaps} taps but the transfer function only supports "
            f"{ops.shape(impulse)[-1]}."
        )
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)

    shift = int(ntaps / 2 - 1)
    return ops.roll(impulse, shift=shift, axis=-1)[..., :ntaps]


def _centered(arr, newsize):
    start_ind = (ops.shape(arr)[-1] - newsize) // 2
    return arr[..., start_ind:start_ind + newsize]


@jax.jit(static_argnames=["mode"])
def fftconvolve(in1, in2, mode="full"):
    """
    Linear convolution along the last axis by FFT, with leading axes broadcast.
    The FFT length is the next fast length of s1 + s2 - 1, not s1 + s2 - 1
    itself, which is often a prime (6143 at 2048 Hz) with a slow FFT.
    """
    s1 = ops.shape(in1)[-1]
    s2 = ops.shape(in2)[-1]
    shape = s1 + s2 - 1
    nfft = next_fast_len(shape, real=True)

    sp1 = jnp.fft.rfft(in1, n=nfft)
    sp2 = jnp.fft.rfft(in2, n=nfft)
    ret = jnp.fft.irfft(sp1 * sp2, n=nfft)[..., :shape]

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    raise ValueError("Acceptable mode flags are 'valid', 'same', or 'full'.")


@jax.jit(static_argnames=["window"])
def convolve(
    timeseries,
    fir,
    window: str = 'hann'
    ):
    """
    Convolve the timeseries with the FIR filter (mode 'same'), tapering the
    first and last half-filter of the input, as gwpy's TimeSeries.convolve does.
    Those half-filter lengths of the output are corrupted. Raises ValueError when
    the timeseries is shorter than the filter.
    """
    ntaps = ops.shape(fir)[-1]
    num_samples = ops.shape(timeseries)[-1]
    if num_samples < ntaps:
        raise ValueError(
            f"The timeseries ({num_samples} samples) is shorter than the whitening "
            f"filter ({ntaps} taps); its output would be corrupted throughout."
        )
    pad = int(ceil(ntaps / 2))

    if window == 'hann':
        win = ops.convert_to_tensor(hann(ntaps))
    else:
        raise ValueError(f"Window function {window} not supported")

    timeseries = ops.concatenate([
        timeseries[..., :pad] * win[:pad],
        timeseries[..., pad:num_samples - pad],
        timeseries[..., num_samples - pad:] * win[ntaps - pad:],
    ], axis=-1)

    out = fftconvolve(timeseries, fir, mode='same')
    assert out.shape[-1] == num_samples
    return out


@lru_cache(maxsize=None)
def whitening_gain(
        nperseg: int,
        noverlap: int,
        num_frames: int,
        method: str,
        num_samples: int,
        ntaps: int,
    ) -> float:
    """
    The mean in-band power gain of this module's whitening filter on Gaussian
    noise, relative to whitening with the true PSD: E[|FIR(f)|^2 * P(f) * 2 dt].

    A noisy PSD estimate makes the inverse-ASD filter too strong on average
    (Jensen), and truncating the filter to ntaps smooths part of that back out,
    so the gain depends on the estimator, the segment count, the window length
    and the filter length together. It is measured here by a fixed-seed Monte
    Carlo that runs this module's own pipeline (numpy port) on white noise, so it
    is deterministic. Its standard error is about 1e-3.
    """
    rng = np.random.default_rng(20261016)
    step = nperseg - noverlap
    length = nperseg + (num_frames - 1) * step
    window = hann(nperseg).astype(np.float64)
    norm = np.sum(window ** 2)
    one_sided = np.full(nperseg // 2 + 1, 2.0)
    one_sided[0] = 1.0
    if nperseg % 2 == 0:
        one_sided[-1] = 1.0
    source = np.arange(nperseg // 2 + 1) / nperseg
    target = np.arange(num_samples // 2 + 1) / num_samples
    taper = np.ones(num_samples // 2 + 1)
    ntaper = min(EDGE_TAPER_BINS, (num_samples // 2 + 1) // 2)
    ramp = _edge_taper(ntaper).astype(np.float64)
    taper[:ntaper] *= ramp
    taper[len(taper) - ntaper:] *= ramp[::-1]
    fir_window = hann(ntaps).astype(np.float64)
    half = ntaps // 2
    interior = slice(4 * EDGE_TAPER_BINS, num_samples // 2 + 1 - 4 * EDGE_TAPER_BINS)
    gains = []
    for _ in range(4):
        x = rng.standard_normal((64, length))
        idx = np.arange(nperseg)[None, :] + np.arange(num_frames)[:, None] * step
        frames = x[:, idx]
        frames = frames - frames.mean(axis=-1, keepdims=True)
        periodograms = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2 / norm
        if method == "median":
            p_hat = np.median(periodograms, axis=1) / median_bias(num_frames)
        else:
            p_hat = np.mean(periodograms, axis=1)
        p_hat = p_hat * one_sided  # estimates 2 (white noise, unit variance, fs = 1)
        asd = np.array([np.interp(target, source, np.sqrt(row)) for row in p_hat])
        impulse = np.fft.irfft(taper / asd, n=num_samples, axis=-1)
        impulse[:, :half] *= fir_window[half:ntaps]
        impulse[:, num_samples - half:] *= fir_window[:half]
        impulse[:, half:num_samples - half] = 0.0
        fir = np.roll(impulse, half - 1, axis=-1)[:, :ntaps]
        response = np.abs(np.fft.rfft(fir, n=num_samples, axis=-1)) ** 2
        # True PSD 2 times |FIR|^2 times 2 dt (dt = 1): the output PSD relative to 2.
        gains.append(np.mean(2.0 * response[:, interior], axis=-1))
    return float(np.mean(np.concatenate(gains)))


def whitened_noise_std(config: SpectralConfig, sample_rate_hertz: float) -> float:
    """
    The standard deviation of stationary Gaussian noise whitened with `config`'s
    band: its PSD is white noise's of unit variance on [lower, upper) and zero
    outside, so its variance is the band's share of [0, Nyquist), (upper - lower)
    / (sample_rate_hertz / 2), with no lower edge at 0 Hz and no upper edge at the
    Nyquist frequency. The 5-bin edge tapers outside the band are neglected.
    """
    nyquist = sample_rate_hertz / 2.0
    lower = config.lower_frequency_hertz or 0.0
    upper = nyquist if config.upper_frequency_hertz is None else config.upper_frequency_hertz
    return float(np.sqrt((upper - lower) / nyquist))


def _resolve_spectrum(
        background,
        sample_rate_hertz: float,
        fft_duration_seconds,
        overlap_duration_seconds,
        highpass_hertz,
        lowpass_hertz,
        filter_duration_seconds,
        psd_method,
    ) -> PowerSpectrum:
    """
    The PowerSpectrum to whiten with: the given one (whose config must agree with
    every explicit argument), or one estimated from a background timeseries.
    """
    if isinstance(background, PowerSpectrum):
        config = background.config
        explicit = {
            "fft_duration_seconds": fft_duration_seconds,
            "overlap_duration_seconds": overlap_duration_seconds,
            "psd_method": psd_method,
            "filter_duration_seconds": filter_duration_seconds,
        }
        for name, value in explicit.items():
            if value is not None and value != getattr(config, name):
                raise ValueError(
                    f"{name}={value} conflicts with the given PowerSpectrum's "
                    f"{name}={getattr(config, name)}."
                )
        if highpass_hertz is not None and highpass_hertz != config.lower_frequency_hertz:
            raise ValueError("highpass_hertz conflicts with the PowerSpectrum's lower_frequency_hertz.")
        if lowpass_hertz is not None and lowpass_hertz != config.upper_frequency_hertz:
            raise ValueError("lowpass_hertz conflicts with the PowerSpectrum's upper_frequency_hertz.")
        if background.sample_rate_hertz != float(sample_rate_hertz):
            raise ValueError("sample_rate_hertz differs from the PowerSpectrum's.")
        return background

    defaults = SpectralConfig()
    config = SpectralConfig(
        lower_frequency_hertz=highpass_hertz,
        upper_frequency_hertz=lowpass_hertz,
        fft_duration_seconds=defaults.fft_duration_seconds if fft_duration_seconds is None else fft_duration_seconds,
        overlap_duration_seconds=defaults.overlap_duration_seconds if overlap_duration_seconds is None else overlap_duration_seconds,
        psd_method=defaults.psd_method if psd_method is None else psd_method,
        filter_duration_seconds=defaults.filter_duration_seconds if filter_duration_seconds is None else filter_duration_seconds,
    )
    return estimate_psd(background, sample_rate_hertz, config)


def _filter_settings(spectrum: PowerSpectrum, num_samples: int, sample_rate_hertz: float):
    """
    (ntaps, ncorner, nupper, gain) of the whitening filter for a num_samples
    series: the FIR length, the band's transfer-function edges and the
    inverse-bias power gain (1.0 when the correction is off).
    """
    config = spectrum.config
    ntaps = config.num_taps(sample_rate_hertz)
    if ntaps % 2:
        ntaps += 1
    if num_samples < ntaps:
        raise ValueError(
            f"The timeseries has {num_samples} samples ({num_samples / sample_rate_hertz:g} s), "
            f"shorter than the {ntaps}-tap ({config.filter_duration_seconds:g} s) whitening "
            "filter; lengthen the window or shorten the filter."
        )
    # The edge tapers sit outside the band, so the filter has full gain on every
    # bin the SNR counts ([f_low, f_high), see SpectralConfig.band_indices); only
    # at the Nyquist frequency does the taper fall inside.
    df = sample_rate_hertz / num_samples
    ncorner = 0
    if config.lower_frequency_hertz:
        ncorner = max(int(config.lower_frequency_hertz / df) - EDGE_TAPER_BINS, 0)
    nupper = None
    if config.upper_frequency_hertz is not None:
        nupper = min(int(config.upper_frequency_hertz / df) + EDGE_TAPER_BINS, num_samples // 2 + 1)

    gain = 1.0
    if config.correct_inverse_bias:
        gain = whitening_gain(
            config.nperseg(sample_rate_hertz), config.noverlap(sample_rate_hertz),
            spectrum.num_segments, config.psd_method, num_samples, ntaps,
        )
    return ntaps, ncorner, nupper, gain


def whitening_filter(spectrum: PowerSpectrum, num_samples: int, window: str = "hann"):
    """
    The FIR filter (..., ntaps) that `whiten` applies, with this spectrum, to a
    num_samples series. One filter per row of the spectrum. `whiten` first
    removes the series' mean (its `detrend`) and Hann-tapers its first and last
    ceil(ntaps / 2) samples (`convolve`); the whitened series is then the
    'same'-mode convolution of that series, divided by the spectrum's scale,
    with this filter, times sqrt(2) / sqrt(gain). A matched filter whitens its
    templates with this filter alone: away from the tapered edges, they see
    what the data saw.
    """
    sample_rate_hertz = spectrum.sample_rate_hertz
    ntaps, ncorner, nupper, _ = _filter_settings(spectrum, int(num_samples), sample_rate_hertz)
    return _fir(spectrum, int(num_samples), ntaps, ncorner, nupper, window)


@jax.jit(static_argnames=["num_samples", "ntaps", "ncorner", "nupper", "window"])
def _fir(spectrum: PowerSpectrum, num_samples, ntaps, ncorner, nupper, window):
    # Work in the background's normalised units: the transfer 1/ASD then has the
    # same magnitude at any input scale.
    asd = spectrum.interpolate(num_samples, amplitude=True)
    return fir_from_transfer(1.0 / asd, ntaps, window=window, ncorner=ncorner, nupper=nupper)


def whiten(
    timeseries,
    background: Union[jnp.ndarray, PowerSpectrum],
    sample_rate_hertz: float,
    fft_duration_seconds: Optional[float] = None,
    overlap_duration_seconds: Optional[float] = None,
    highpass_hertz: Optional[float] = None,
    detrend: str = 'constant',
    filter_duration_seconds: Optional[float] = None,
    window: str = "hann",
    num_samples: Optional[int] = None,
    lowpass_hertz: Optional[float] = None,
    psd_method: Optional[str] = None,
    ):
    """
    Whiten `timeseries` (..., T) with the PSD of `background`.

    `background` is either a timeseries (..., T_off) whose PSD is estimated here
    (Welch, `fft_duration_seconds` / `overlap_duration_seconds`, default 1 s /
    0.5 s, `psd_method`, default median), or a PowerSpectrum from `estimate_psd`,
    whose band and filter settings then apply. The band is
    [highpass_hertz, lowpass_hertz): the filter has full gain on it, ramps to 0
    over the 5 frequency bins just outside each edge, and removes everything
    beyond. (gwpy starts its ramp at the highpass frequency instead.)

    Leading axes broadcast: a (G, B, I, T) timeseries can be whitened with a
    (B, I) spectrum. The first and last filter_duration / 2 of the output are
    corrupted; crop them. Returns float32 of the input's shape.
    """
    if highpass_hertz:
        if highpass_hertz < 0 or highpass_hertz >= (sample_rate_hertz / 2):
            raise ValueError("Invalid highpass frequency.")
    if filter_duration_seconds is not None and filter_duration_seconds <= 0:
        raise ValueError("Filter duration should be greater than zero.")
    if window not in ["hann"]:
        raise ValueError("Invalid window type.")
    if detrend not in ("constant", "linear", "c", "l", None):
        raise ValueError("detrend must be 'constant', 'linear' or None.")

    spectrum = _resolve_spectrum(
        background, sample_rate_hertz, fft_duration_seconds, overlap_duration_seconds,
        highpass_hertz or None, lowpass_hertz, filter_duration_seconds, psd_method,
    )

    timeseries = ops.cast(ops.convert_to_tensor(timeseries), "float32")
    is_1d = len(ops.shape(timeseries)) == 1
    if is_1d:
        timeseries = timeseries[None, :]
    if num_samples is not None and num_samples != timeseries.shape[-1]:
        raise ValueError("num_samples differs from the timeseries length.")
    num_samples = int(timeseries.shape[-1])

    ntaps, ncorner, nupper, gain = _filter_settings(spectrum, num_samples, sample_rate_hertz)
    out = _whiten(timeseries, spectrum, ntaps, ncorner, nupper, detrend, window, float(1.0 / np.sqrt(gain)))
    return out[0] if is_1d else out


@jax.jit(static_argnames=["ntaps", "ncorner", "nupper", "detrend", "window", "correction"])
def _whiten(timeseries, spectrum: PowerSpectrum, ntaps, ncorner, nupper, detrend, window, correction):
    num_samples = timeseries.shape[-1]
    if detrend is not None:
        timeseries = _detrend(timeseries, axis=-1, type=detrend)
    # Work in the background's normalised units: the transfer 1/ASD then has the
    # same magnitude at any input scale.
    timeseries = timeseries / spectrum.scale
    fir = _fir(spectrum, num_samples, ntaps, ncorner, nupper, window)
    out = convolve(timeseries, fir, window=window)
    # sqrt(2 dt) in physical units; in normalised units dt cancels with the
    # sample rate folded into the normalised PSD.
    return ops.cast(out * np.float32(np.sqrt(2.0) * correction), "float32")


def _whitening_kwargs(layer) -> dict:
    return dict(
        fft_duration_seconds=layer.fft_duration_seconds,
        overlap_duration_seconds=layer.overlap_duration_seconds,
        highpass_hertz=layer.highpass_hertz,
        lowpass_hertz=layer.lowpass_hertz,
        detrend=layer.detrend,
        filter_duration_seconds=layer.filter_duration_seconds,
        window=layer.window,
        psd_method=layer.psd_method,
    )


def unit_rms(examples):
    """
    Each example of `examples` (Batch, ...) divided by its root mean square over
    all its other axes (detectors and samples together, so their ratios stay).
    Taken over the example's peak first: the square of strain (~1e-42) is below
    float32's range. An all-zero example stays zero.
    """
    axes = tuple(range(1, len(examples.shape)))
    peak = ops.max(ops.abs(examples), axis=axes, keepdims=True)
    examples = examples / ops.where(peak > 0, peak, ops.ones_like(peak))
    rms = ops.sqrt(ops.mean(ops.square(examples), axis=axes, keepdims=True))
    return examples / ops.where(rms > 0, rms, ops.ones_like(rms))


@keras.saving.register_keras_serializable(package="gravyflow")
class WhitenPass(Layer):
    """
    Crops the onsource and scales each example to unit RMS, without whitening;
    the ablation against Whiten, whose output also has unit variance. The scaling
    is what lets a network learn from raw strain: inputs of ~1e-21 give
    gradients and normalisation statistics below float32's range.
    """
    def __init__(
            self,
            sample_rate_hertz = None,
            onsource_duration_seconds = None,
            fft_duration_seconds=1.0,
            overlap_duration_seconds=0.5,
            highpass_hertz=None,
            detrend='constant',
            filter_duration_seconds=1.0,
            window="hann",
            lowpass_hertz=None,
            psd_method="median",
            **kwargs
        ):
        super().__init__(**kwargs)

        if sample_rate_hertz is None:
            sample_rate_hertz = Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = Defaults.onsource_duration_seconds

        self.onsource_duration_seconds = onsource_duration_seconds
        self.sample_rate_hertz = sample_rate_hertz
        self.fft_duration_seconds = fft_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds
        self.highpass_hertz = highpass_hertz
        self.lowpass_hertz = lowpass_hertz
        self.detrend = detrend
        self.filter_duration_seconds = filter_duration_seconds
        self.window = window
        self.psd_method = psd_method
        self.num_output_samples = int(self.onsource_duration_seconds*self.sample_rate_hertz)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        timeseries = inputs

        cropped = crop_samples(timeseries, self.onsource_duration_seconds, self.sample_rate_hertz)

        dynamic_shape = ops.shape(timeseries)
        return unit_rms(ops.reshape(cropped, (dynamic_shape[0], dynamic_shape[1], self.num_output_samples)))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            **_whitening_kwargs(self),
        })
        return config

    def compute_output_shape(self, input_shape):
        timeseries_shape = input_shape
        return (timeseries_shape[0], timeseries_shape[1], self.num_output_samples)


@keras.saving.register_keras_serializable(package="gravyflow")
class Whiten(Layer):
    """
    Whitens [onsource, offsource] inside a model, with gf.whiten and every
    argument given here, then crops to the onsource.
    """
    def __init__(
            self,
            sample_rate_hertz = None,
            onsource_duration_seconds = None,
            fft_duration_seconds=1.0,
            overlap_duration_seconds=0.5,
            highpass_hertz=None,
            detrend='constant',
            filter_duration_seconds=1.0,
            window="hann",
            lowpass_hertz=None,
            psd_method="median",
            **kwargs
        ):
        super().__init__(**kwargs)

        if sample_rate_hertz is None:
            sample_rate_hertz = Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = Defaults.onsource_duration_seconds

        self.onsource_duration_seconds = onsource_duration_seconds
        self.sample_rate_hertz = sample_rate_hertz
        self.fft_duration_seconds = fft_duration_seconds
        self.overlap_duration_seconds = overlap_duration_seconds
        self.highpass_hertz = highpass_hertz
        self.lowpass_hertz = lowpass_hertz
        self.detrend = detrend
        self.filter_duration_seconds = filter_duration_seconds
        self.window = window
        self.psd_method = psd_method
        self.num_output_samples = int(self.onsource_duration_seconds*self.sample_rate_hertz)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        timeseries, background = inputs

        whitened = whiten(timeseries, background, self.sample_rate_hertz, **_whitening_kwargs(self))
        cropped = crop_samples(whitened, self.onsource_duration_seconds, self.sample_rate_hertz)

        dynamic_shape = ops.shape(timeseries)
        return ops.reshape(cropped, (dynamic_shape[0], dynamic_shape[1], self.num_output_samples))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            **_whitening_kwargs(self),
        })
        return config

    def compute_output_shape(self, input_shape):
        timeseries_shape, _ = input_shape
        return (timeseries_shape[0], timeseries_shape[1], int(self.onsource_duration_seconds*self.sample_rate_hertz))
