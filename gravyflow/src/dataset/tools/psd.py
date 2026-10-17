"""Power spectral density estimation.

One estimator serves the whole data path: the dataset estimates one
`PowerSpectrum` per batch from the offsource, and SNR scaling, SNR labels and
whitening all read that one object (see `estimate_psd`).

Conventions:

- PSDs are one-sided, in units of (input units)^2 / Hz: unit-variance white
  noise sampled at `fs` has a PSD of 2 / fs.
- Welch's method: periodic Hann segments, each segment detrended to zero mean,
  as `scipy.signal.welch` does. `mode="median"` divides by the median bias of
  the number of segments, as scipy and pycbc do.
- Scale free: the estimate is computed on data divided by its own peak
  amplitude, so physical strain (~1e-21) and the same data at any other
  amplitude give the same normalised spectrum. A PSD of physical strain (~1e-46)
  is not representable in float32 (the smallest normal float32 is 1.2e-38), so
  every estimate, `psd()` and `estimate_psd()` alike, is a `PowerSpectrum`: the
  normalised spectrum and the amplitude scale kept apart. Its consumers never
  form the physical PSD in float32; `PowerSpectrum.value()` forms it in float64.
"""
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Optional

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import jax

from gravyflow.src.dataset.config import require_positive

# Smallest number of Welch segments `estimate_psd` accepts. Fewer segments give
# an estimate whose inverse (used by whitening and the SNR) has a huge or
# unbounded variance; one segment has no finite inverse mean at all.
MIN_PSD_SEGMENTS = 3


@partial(jax.jit, static_argnames=["axis", "type", "bp", "overwrite_data"])
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove a constant or linear trend along `axis`.
    """
    data = ops.convert_to_tensor(data)
    dtype = data.dtype

    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")

    if type in ['constant', 'c']:
        mean = ops.mean(data, axis=axis, keepdims=True)
        return data - mean

    if bp != 0:
        raise NotImplementedError("Breakpoints not yet supported in JAX port of detrend.")

    N = ops.shape(data)[axis]
    t = ops.arange(N, dtype=dtype)
    ones = ops.ones((N,), dtype=dtype)
    A = ops.stack([ones, t], axis=-1)  # (N, 2)

    if axis != -1 and axis != len(ops.shape(data)) - 1:
        data_transposed = ops.moveaxis(data, axis, -1)
    else:
        data_transposed = data

    original_shape = ops.shape(data_transposed)
    data_reshaped = ops.reshape(data_transposed, (-1, N))
    b = ops.transpose(data_reshaped, (1, 0))  # (N, Batch)
    coeffs, _, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    trend = ops.transpose(ops.matmul(A, coeffs), (1, 0))
    detrended = ops.reshape(data_reshaped - trend, original_shape)

    if axis != -1 and axis != len(ops.shape(data)) - 1:
        detrended = ops.moveaxis(detrended, -1, axis)

    return detrended


def hann(n: int) -> np.ndarray:
    """Periodic Hann window of length n (scipy.signal.get_window('hann', n))."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def median_bias(n: int) -> float:
    """
    Ratio of the median to the mean of n chi-squared(2) periodogram values, as
    scipy.signal.welch(average='median') and pycbc.psd.median_bias compute it.
    """
    ii_2 = 2 * np.arange(1., (n - 1) // 2 + 1)
    return float(1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2))


# Up to this many segments the median uses a sorting network; beyond it, jnp.median.
_NETWORK_MEDIAN_MAX_SEGMENTS = 64


def _median_over_segments(periodograms):
    """
    The median over axis -2 (segments), exactly as jnp.median. XLA's CPU sort is
    slow (60 ms for a 32 x 2 batch of 31 one-second segments); an odd-even
    transposition network of min/max over the segment rows takes 0.2 ms.
    """
    k = periodograms.shape[-2]
    if k > _NETWORK_MEDIAN_MAX_SEGMENTS:
        return jnp.median(periodograms, axis=-2)
    rows = [periodograms[..., i, :] for i in range(k)]
    for round_ in range(k):
        for i in range(round_ % 2, k - 1, 2):
            rows[i], rows[i + 1] = ops.minimum(rows[i], rows[i + 1]), ops.maximum(rows[i], rows[i + 1])
    if k % 2:
        return rows[k // 2]
    return 0.5 * (rows[k // 2 - 1] + rows[k // 2])


def _num_frames(signal_len: int, nperseg: int, noverlap: int) -> int:
    if nperseg < 2:
        raise ValueError(f"nperseg must be at least 2, got {nperseg}.")
    if not 0 <= noverlap < nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg), got {noverlap} for nperseg {nperseg}.")
    step = nperseg - noverlap
    num_frames = (signal_len - nperseg) // step + 1
    if num_frames < 1:
        raise ValueError(
            f"The PSD needs at least one segment of {nperseg} samples, but the "
            f"signal has only {signal_len} samples."
        )
    return num_frames


def _normalised_welch(signal, nperseg: int, noverlap: int, mode: str):
    """
    Welch PSD of `signal` / scale, and the scale (the peak absolute value of each
    row after removing its mean). Returns (pxx_normalised (..., F), scale (..., 1)).
    The PSD of `signal` is pxx_normalised * scale**2 per unit sample rate (the
    caller divides by the sample rate).
    """
    if mode not in ("mean", "median"):
        raise ValueError("Mode not supported")

    signal = ops.cast(ops.convert_to_tensor(signal), "float32")
    original_shape = tuple(ops.shape(signal))
    signal_len = original_shape[-1]
    num_frames = _num_frames(signal_len, nperseg, noverlap)
    step = nperseg - noverlap

    flat = ops.reshape(signal, (-1, signal_len))
    flat = flat - ops.mean(flat, axis=-1, keepdims=True)
    # Divide by the peak amplitude before any square is formed: squares of
    # physical strain underflow in float32.
    scale = ops.max(ops.abs(flat), axis=-1, keepdims=True)
    scale = ops.where(scale > 0, scale, ops.ones_like(scale))
    flat = flat / scale

    indices = np.arange(nperseg)[None, :] + np.arange(num_frames)[:, None] * step
    frames = ops.take(flat, indices, axis=-1)  # (Batch, Frames, nperseg)
    frames = frames - ops.mean(frames, axis=-1, keepdims=True)

    window = ops.convert_to_tensor(hann(nperseg))
    fft_vals = jnp.fft.rfft(frames * window)
    periodograms = (ops.real(fft_vals) ** 2 + ops.imag(fft_vals) ** 2) / float(np.sum(hann(nperseg).astype(np.float64) ** 2))

    if mode == "median":
        pxx = _median_over_segments(periodograms) / median_bias(num_frames)
    else:
        pxx = ops.mean(periodograms, axis=-2)

    num_freqs = nperseg // 2 + 1
    one_sided = np.full(num_freqs, 2.0, dtype=np.float32)
    one_sided[0] = 1.0
    if nperseg % 2 == 0:
        one_sided[-1] = 1.0
    pxx = pxx * ops.convert_to_tensor(one_sided)

    batch_shape = original_shape[:-1]
    pxx = ops.reshape(pxx, batch_shape + (num_freqs,))
    scale = ops.reshape(scale, batch_shape + (1,))
    return pxx, scale


# ---------------------------------------------------------------------------
# One spectral definition for the data path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralConfig:
    """
    The one definition of the noise spectrum, the analysis band and the
    whitening filter. SNR scaling, SNR labels and whitening all read it, so the
    SNR a label reports is the SNR the whitened data carry.

    Attributes:
        lower_frequency_hertz: lower edge of the band; None for no lower edge.
            Whitening zeroes everything below it, and the SNR integral starts
            there.
        upper_frequency_hertz: upper edge of the band; None for the Nyquist
            frequency.
        fft_duration_seconds, overlap_duration_seconds: Welch segment length
            and overlap of the PSD estimate.
        psd_method: "median" (robust to a glitch in the offsource, bias
            corrected) or "mean".
        filter_duration_seconds: length of the whitening FIR. Its first and
            last half-lengths are corrupted, so the crop on each side of the
            onsource must be at least half of it.
        psd_floor: PSD floor relative to the median PSD in the band; it bounds
            the gain in bands where the noise has (almost) no power, and is the
            same in whitening and in the SNR.
        correct_inverse_bias: remove the bias the estimate's noise puts into its
            inverse (Jensen: E[1 / P_hat] > 1 / P): the SNR weighs by the
            inverse PSD divided by E[P / P_hat] (PowerSpectrum.inverse), and
            whitening divides by its own measured gain, so that the SNR of
            Gaussian noise has no estimator bias at any onsource length and
            whitened Gaussian noise has unit in-band PSD.
    """
    lower_frequency_hertz: Optional[float] = 20.0
    upper_frequency_hertz: Optional[float] = None
    fft_duration_seconds: float = 1.0
    overlap_duration_seconds: float = 0.5
    psd_method: str = "median"
    filter_duration_seconds: float = 1.0
    psd_floor: float = 1.0e-4
    correct_inverse_bias: bool = True

    def __post_init__(self):
        if self.psd_method not in ("mean", "median"):
            raise ValueError(f"psd_method must be 'mean' or 'median', got {self.psd_method!r}.")
        if self.fft_duration_seconds <= 0:
            raise ValueError("fft_duration_seconds must be positive.")
        if not 0 <= self.overlap_duration_seconds < self.fft_duration_seconds:
            raise ValueError("overlap_duration_seconds must be in [0, fft_duration_seconds).")
        if self.filter_duration_seconds <= 0:
            raise ValueError("filter_duration_seconds must be positive.")
        if self.lower_frequency_hertz is not None and self.lower_frequency_hertz < 0:
            raise ValueError("lower_frequency_hertz must be non-negative.")
        if (self.lower_frequency_hertz is not None and self.upper_frequency_hertz is not None
                and self.upper_frequency_hertz <= self.lower_frequency_hertz):
            raise ValueError("upper_frequency_hertz must be above lower_frequency_hertz.")
        if not 0 <= self.psd_floor < 1:
            raise ValueError("psd_floor must be in [0, 1).")

    def nperseg(self, sample_rate_hertz: float) -> int:
        return int(round(self.fft_duration_seconds * sample_rate_hertz))

    def noverlap(self, sample_rate_hertz: float) -> int:
        return int(round(self.overlap_duration_seconds * sample_rate_hertz))

    def num_taps(self, sample_rate_hertz: float) -> int:
        return int(round(self.filter_duration_seconds * sample_rate_hertz))

    def band_indices(self, num_samples: int, sample_rate_hertz: float):
        """
        [kmin, kmax) of the rfft bins of a num_samples series inside the band,
        with pycbc's convention (pycbc.filter.get_cutoff_indices): DC is never
        included, and the Nyquist bin of an even length is not either.
        """
        df = sample_rate_hertz / num_samples
        kmax_all = int((num_samples + 1) / 2.0)
        kmin = int(self.lower_frequency_hertz / df) if self.lower_frequency_hertz else 1
        kmin = max(kmin, 1)
        if self.upper_frequency_hertz is not None:
            kmax = min(int(self.upper_frequency_hertz / df), kmax_all)
        else:
            kmax = kmax_all
        if kmax <= kmin:
            raise ValueError(
                f"The band [{self.lower_frequency_hertz}, {self.upper_frequency_hertz}) Hz holds "
                f"no frequency bin of a {num_samples}-sample series at {sample_rate_hertz} Hz."
            )
        return kmin, kmax

    def validate_band(self, sample_rate_hertz: float):
        require_positive("sample_rate_hertz", sample_rate_hertz)
        nyquist = sample_rate_hertz / 2.0
        if self.lower_frequency_hertz is not None and self.lower_frequency_hertz >= nyquist:
            raise ValueError(
                f"lower_frequency_hertz ({self.lower_frequency_hertz}) must be below the "
                f"Nyquist frequency ({nyquist})."
            )
        if self.upper_frequency_hertz is not None and self.upper_frequency_hertz > nyquist:
            raise ValueError(
                f"upper_frequency_hertz ({self.upper_frequency_hertz}) must not exceed the "
                f"Nyquist frequency ({nyquist})."
            )

    def num_segments(self, num_samples: int, sample_rate_hertz: float) -> int:
        nperseg = self.nperseg(sample_rate_hertz)
        return _num_frames(num_samples, nperseg, self.noverlap(sample_rate_hertz))


@lru_cache(maxsize=None)
def inverse_bias(nperseg: int, noverlap: int, num_frames: int, method: str) -> float:
    """
    E[P / P_hat] for the Welch estimate P_hat of a Gaussian process with PSD P,
    at a frequency bin away from DC and Nyquist.

    Whitening multiplies by P_hat^(-1/2) and the SNR integrates |h|^2 / P_hat,
    so both inherit E[1 / P_hat] > 1 / P (Jensen). This is computed exactly for
    Gaussian noise by a fixed-seed Monte Carlo over the joint distribution of the
    overlapping segments' Fourier coefficients (correlated through the window's
    overlap), so it is deterministic. Its relative standard error, measured
    for the median with half overlap, is 1.6e-3 at 7 segments and 3e-4 at 127.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    w = hann(nperseg).astype(np.float64)
    step = nperseg - noverlap
    norm = np.sum(w * w)
    r = np.zeros(num_frames)
    for j in range(num_frames):
        lag = j * step
        r[j] = np.sum(w[: nperseg - lag] * w[lag:]) / norm if lag < nperseg else 0.0
    R = r[np.abs(np.arange(num_frames)[:, None] - np.arange(num_frames)[None, :])]
    L = np.linalg.cholesky(R + 1e-12 * np.eye(num_frames))
    rng = np.random.default_rng(20261016)
    draws = 200_000
    z = (rng.standard_normal((draws, num_frames)) + 1j * rng.standard_normal((draws, num_frames))) / np.sqrt(2.0)
    z = z @ L.T
    p = np.abs(z) ** 2  # unit-mean periodogram values
    if method == "median":
        est = np.median(p, axis=1) / median_bias(num_frames)
    elif method == "mean":
        est = np.mean(p, axis=1)
    else:
        raise ValueError("Mode not supported")
    return float(np.mean(1.0 / est))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PowerSpectrum:
    """
    A per-row PSD estimate, kept scale free.

    normalised: (..., F) float32, the Welch PSD of background / scale times the
        sample rate, floored relative to its in-band median.
    scale: (..., 1) float32, the amplitude scale of each row.
    inverse_bias: E[P / P_hat] for this estimator (1.0 when the config turns the
        correction off); `inverse` divides by it.
    The PSD is normalised * scale**2 / sample_rate_hertz; it is never formed in
    float32 by the data path.
    """
    normalised: jnp.ndarray
    scale: jnp.ndarray
    sample_rate_hertz: float = field(metadata=dict(static=True))
    config: SpectralConfig = field(metadata=dict(static=True))
    inverse_bias: float = field(default=1.0, metadata=dict(static=True))
    num_segments: int = field(default=0, metadata=dict(static=True))

    def tree_flatten(self):
        return (self.normalised, self.scale), (self.sample_rate_hertz, self.config, self.inverse_bias, self.num_segments)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1], *aux)

    @property
    def nperseg(self) -> int:
        return (int(self.normalised.shape[-1]) - 1) * 2

    @property
    def frequencies(self) -> np.ndarray:
        return np.arange(self.normalised.shape[-1], dtype=np.float32) * (self.sample_rate_hertz / self.nperseg)

    def value(self) -> np.ndarray:
        """
        The PSD in input units^2/Hz, as float64 NumPy: the PSD of physical strain
        (~1e-46) is below float32's range. For plots and comparisons; it is not
        traceable, and the data path never forms it.
        """
        normalised = np.asarray(self.normalised, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return normalised * scale ** 2 / self.sample_rate_hertz

    def interpolate(self, num_samples: int, amplitude: bool = False):
        """
        The normalised PSD (or its square root) linearly interpolated onto the rfft
        grid of a num_samples series at the same sample rate.
        """
        return self._on_grid(ops.sqrt(self.normalised) if amplitude else self.normalised, num_samples)

    def inverse(self, num_samples: int):
        """
        An estimate of 1 / normalised on the rfft grid of a num_samples series,
        without the estimator's inverse bias at any bin: 1 / normalised linearly
        interpolated, divided by inverse_bias. The SNR and the matched filter
        weigh by it.

        The inverse is interpolated, not the PSD, because only then is one factor
        exact between the PSD's bins too: E[1 / P_hat] = inverse_bias / P at each
        bin, so by linearity at every point between two bins. The inverse of an
        interpolated PSD is biased less between bins (the mean of two estimates is
        less noisy than either), so dividing it by inverse_bias over-corrects
        wherever the onsource is longer than a Welch segment. A bin with no
        power (an unfloored PSD's DC bin) has inverse zero rather than infinity,
        so that it carries no weight and spreads none to its neighbours.
        """
        powered = self.normalised > 0
        inverse = ops.where(powered, 1.0 / ops.where(powered, self.normalised, 1.0), 0.0)
        return self._on_grid(inverse, num_samples) / self.inverse_bias

    def _on_grid(self, values, num_samples: int):
        """values (..., F) on the PSD's grid, linearly interpolated onto the rfft grid of num_samples."""
        target = np.arange(num_samples // 2 + 1, dtype=np.float32) * (self.sample_rate_hertz / num_samples)
        flat = ops.reshape(values, (-1, values.shape[-1]))
        out = jax.vmap(lambda v: jnp.interp(target, self.frequencies, v))(flat)
        return ops.reshape(out, tuple(values.shape[:-1]) + (len(target),))


def estimate_psd(
        background,
        sample_rate_hertz: float,
        config: Optional[SpectralConfig] = None,
    ) -> PowerSpectrum:
    """
    Estimate one PowerSpectrum per row of `background` (..., T), with `config`.

    Raises ValueError when the background holds fewer than MIN_PSD_SEGMENTS
    Welch segments.
    """
    config = SpectralConfig() if config is None else config
    config.validate_band(sample_rate_hertz)
    background = ops.convert_to_tensor(background)
    num_samples = int(background.shape[-1])
    nperseg = config.nperseg(sample_rate_hertz)
    noverlap = config.noverlap(sample_rate_hertz)
    num_frames = _num_frames(num_samples, nperseg, noverlap)
    if num_frames < MIN_PSD_SEGMENTS:
        raise ValueError(
            f"The offsource has {num_samples / sample_rate_hertz:g} s, which gives "
            f"{num_frames} Welch segment(s) of {config.fft_duration_seconds:g} s; at least "
            f"{MIN_PSD_SEGMENTS} are needed for a usable PSD."
        )
    factor = inverse_bias(nperseg, noverlap, num_frames, config.psd_method) if config.correct_inverse_bias else 1.0
    normalised, scale = _estimate(background, nperseg, noverlap, config.psd_method,
                                  config.psd_floor, config.band_indices(nperseg, sample_rate_hertz))
    return PowerSpectrum(normalised, scale, float(sample_rate_hertz), config, factor, num_frames)


_welch = jax.jit(_normalised_welch, static_argnames=["nperseg", "noverlap", "mode"])


def psd(
        signal,
        nperseg : int,
        sample_rate_hertz : float,
        noverlap : Optional[int] = None,
        mode : str = "mean"
    ) -> PowerSpectrum:
    """
    The one-sided Welch PSD of each row of `signal` (..., T), as a PowerSpectrum.

    The estimate is scipy.signal.welch(x, fs, window='hann', nperseg, noverlap,
    detrend='constant', average=mode): `value()` gives it in float64 (at any
    amplitude, physical strain included), and `frequencies` its grid. Unlike
    `estimate_psd`, it has no band, no floor and no inverse-bias correction, and
    takes a single segment. Raises ValueError when the signal is shorter than
    one segment.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    signal = ops.convert_to_tensor(signal)
    num_frames = _num_frames(int(signal.shape[-1]), nperseg, noverlap)
    normalised, scale = _welch(signal, nperseg, noverlap, mode)
    config = SpectralConfig(
        lower_frequency_hertz=None,
        fft_duration_seconds=nperseg / sample_rate_hertz,
        overlap_duration_seconds=noverlap / sample_rate_hertz,
        psd_method=mode,
        psd_floor=0.0,
        correct_inverse_bias=False,
    )
    return PowerSpectrum(normalised, scale, float(sample_rate_hertz), config, 1.0, num_frames)


@partial(jax.jit, static_argnames=["nperseg", "noverlap", "method", "floor", "band"])
def _estimate(background, nperseg, noverlap, method, floor, band):
    pxx_n, scale = _normalised_welch(background, nperseg, noverlap, method)
    kmin, kmax = band
    reference = jnp.median(pxx_n[..., kmin:kmax], axis=-1, keepdims=True)
    pxx_n = ops.maximum(pxx_n, floor * reference)
    return pxx_n, scale
