"""The one SNR definition, for the data path and the matched filter.

The optimal SNR of a signal h in noise with one-sided PSD S, over the band
[f_low, f_high), is

    rho^2 = 4 * df * sum_{k in band} |h~_k|^2 / S_k,    h~ = rfft(h) / fs,

with pycbc's band convention (pycbc.filter.sigma): DC is never included, and
neither is the Nyquist bin of an even-length series. A signal on several
detectors has the network SNR sqrt(sum over detectors of rho^2).

The PSD comes from a PowerSpectrum (see gravyflow.src.dataset.tools.psd), so
the data path uses one estimate per batch, with its band and its relative
floor, for scaling, labels and whitening alike. `psd=None` in the matched-filter
functions means unit-variance white noise (S = 2 / fs): the PSD of whitened
data.
"""
from typing import Optional, Union

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import jax

from gravyflow.src.dataset.tools.psd import (
    PowerSpectrum, SpectralConfig, estimate_psd
)
from gravyflow.src.utils.tensor import crop_samples, replace_nan_and_inf_with_zero
from gravyflow.src.rng.legacy import warn_deprecated


@jax.jit(static_argnames=["scalar"])
def find_closest(
        tensor,
        scalar : float
    ):
    tensor = ops.convert_to_tensor(tensor)
    return ops.argmin(ops.abs(tensor - scalar))


def _spectrum_for(
        background,
        sample_rate_hertz: float,
        fft_duration_seconds,
        overlap_duration_seconds,
        lower_frequency_cutoff,
        upper_frequency_cutoff,
        psd_method,
    ) -> PowerSpectrum:
    """The given PowerSpectrum (checked against explicit arguments), or one estimated now."""
    if isinstance(background, PowerSpectrum):
        config = background.config
        explicit = {
            "fft_duration_seconds": fft_duration_seconds,
            "overlap_duration_seconds": overlap_duration_seconds,
            "lower_frequency_hertz": lower_frequency_cutoff,
            "upper_frequency_hertz": upper_frequency_cutoff,
            "psd_method": psd_method,
        }
        for name, value in explicit.items():
            if value is not None and value != getattr(config, name):
                raise ValueError(
                    f"{name}={value} conflicts with the given PowerSpectrum's "
                    f"{name}={getattr(config, name)}."
                )
        if background.sample_rate_hertz != float(sample_rate_hertz):
            raise ValueError("sample_rate_hertz differs from the PowerSpectrum's.")
        return background

    defaults = SpectralConfig()
    config = SpectralConfig(
        lower_frequency_hertz=defaults.lower_frequency_hertz if lower_frequency_cutoff is None else (lower_frequency_cutoff or None),
        upper_frequency_hertz=upper_frequency_cutoff,
        fft_duration_seconds=defaults.fft_duration_seconds if fft_duration_seconds is None else fft_duration_seconds,
        overlap_duration_seconds=defaults.overlap_duration_seconds if overlap_duration_seconds is None else overlap_duration_seconds,
        psd_method=defaults.psd_method if psd_method is None else psd_method,
    )
    background = ops.cast(ops.convert_to_tensor(background), "float32")
    return estimate_psd(background, sample_rate_hertz, config)


@jax.jit(static_argnames=["band", "network"])
def _snr_from_spectrum(injection, spectrum: PowerSpectrum, band, network: bool):
    """Per-channel optimal SNR over `band`, combined over the detector axis if `network`."""
    num_samples = injection.shape[-1]
    kmin, kmax = band
    # Normalised units: h / scale against the normalised PSD (see psd.py); the
    # scale cancels, so physical and scaled strain give the same SNR.
    h = injection / spectrum.scale
    # Unit peak per row before squaring, so the power sums cannot overflow or
    # underflow float32 whatever the injection's amplitude; rho scales back exactly.
    peak = ops.max(ops.abs(h), axis=-1, keepdims=True)
    peak = ops.where(peak > 0, peak, ops.ones_like(peak))
    h_fft = jnp.fft.rfft(h / peak)[..., kmin:kmax]
    power = ops.real(h_fft) ** 2 + ops.imag(h_fft) ** 2
    psd_n = spectrum.interpolate(num_samples)[..., kmin:kmax] * spectrum.inverse_bias
    rho = peak[..., 0] * ops.sqrt((4.0 / num_samples) * ops.sum(power / psd_n, axis=-1))
    if network:
        largest = ops.max(rho, axis=-1, keepdims=True)
        largest = ops.where(largest > 0, largest, ops.ones_like(largest))
        rho = largest[..., 0] * ops.sqrt(ops.sum((rho / largest) ** 2, axis=-1))
    return rho


def snr(
        injection,
        background: Union[jnp.ndarray, PowerSpectrum],
        sample_rate_hertz: float,
        fft_duration_seconds: Optional[float] = None,
        overlap_duration_seconds: Optional[float] = None,
        lower_frequency_cutoff: Optional[float] = None,
        upper_frequency_cutoff: Optional[float] = None,
        psd_method: Optional[str] = None,
    ):
    """
    Optimal SNR of `injection` in the noise of `background`.

    Shapes: (B, I, T) gives the network SNR per example, (B,); (B, T) gives one
    SNR per row, (B,); (T,) gives a scalar. Leading axes beyond three are kept:
    (G, B, I, T) gives (G, B).

    `background` is a PowerSpectrum (its band and estimator apply; explicit
    arguments must agree with it) or a timeseries (..., T_off) whose PSD is
    estimated with the given settings, defaulting to SpectralConfig's (1 s / 0.5 s
    median Welch, band from 20 Hz to Nyquist). `lower_frequency_cutoff=0` means
    no lower edge.
    """
    spectrum = _spectrum_for(
        background, sample_rate_hertz, fft_duration_seconds, overlap_duration_seconds,
        lower_frequency_cutoff, upper_frequency_cutoff, psd_method,
    )
    injection = ops.cast(ops.convert_to_tensor(injection), "float32")
    rank = len(injection.shape)
    if rank == 1:
        injection = injection[None, :]
    num_samples = int(injection.shape[-1])
    band = spectrum.config.band_indices(num_samples, sample_rate_hertz)
    result = _snr_from_spectrum(injection, spectrum, band, rank >= 3)
    return result[0] if rank == 1 else result


def _expand_like(values, rank: int):
    """(…batch) values to broadcast against a rank-`rank` injection."""
    trailing = 2 if rank >= 3 else 1
    return ops.reshape(values, tuple(values.shape) + (1,) * trailing)


def scale_to_snr(
        injection,
        background: Union[jnp.ndarray, PowerSpectrum],
        desired_snr,
        sample_rate_hertz: float,
        fft_duration_seconds: Optional[float] = None,
        overlap_duration_seconds: Optional[float] = None,
        lower_frequency_cutoff: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        upper_frequency_cutoff: Optional[float] = None,
        psd_method: Optional[str] = None,
        return_factor: bool = False,
        return_scale_factor: Optional[bool] = None,
    ):
    """
    Scale each example of `injection` to `desired_snr`, measured with `snr()` on
    the central `onsource_duration_seconds` window (the whole injection if None):
    the SNR of what the onsource window shows.

    Each example is scaled by its own factor: (B, I, T) and (B, T) inputs take
    (B,) targets. An example with no power in the band gets the factor 0, never
    a division by zero. With return_factor, returns (scaled, factor) with
    the factor k applied to each example, shaped like the SNR (for example (B,)).
    ``return_scale_factor`` is its deprecated name.
    """
    if return_scale_factor is not None:
        warn_deprecated("scale_to_snr(return_scale_factor=)", "pass return_factor=")
        return_factor = return_scale_factor
    spectrum = _spectrum_for(
        background, sample_rate_hertz, fft_duration_seconds, overlap_duration_seconds,
        lower_frequency_cutoff, upper_frequency_cutoff, psd_method,
    )
    injection = ops.cast(ops.convert_to_tensor(injection), "float32")
    injection = replace_nan_and_inf_with_zero(injection, "scale_to_snr's injection")
    rank = len(injection.shape)

    measured = injection
    if onsource_duration_seconds is not None:
        measured = crop_samples(injection, onsource_duration_seconds, sample_rate_hertz)

    current_snr = snr(measured, spectrum, sample_rate_hertz)
    desired_snr = ops.cast(ops.convert_to_tensor(desired_snr), "float32")
    if desired_snr.size == current_snr.size:
        desired_snr = ops.reshape(desired_snr, current_snr.shape)
    else:
        desired_snr = ops.broadcast_to(desired_snr, current_snr.shape)
    factor = ops.where(current_snr > 0, desired_snr / ops.where(current_snr > 0, current_snr, 1.0), 0.0)

    scaled = injection * factor if rank == 1 else injection * _expand_like(factor, rank)
    if return_factor:
        return scaled, factor
    return scaled


# ---------------------------------------------------------------------------
# Matched filtering
# ---------------------------------------------------------------------------

def _weight_on_grid(psd, num_samples: int, sample_rate_hertz: float):
    """
    The PSD on the rfft grid of a num_samples series as a weight relative to
    unit-variance white noise (S = 2 / fs * weight), and the amplitude scale to
    divide the data (and templates) by first, so that no square of strain is
    formed in float32. `psd` is None (white, unit variance: the PSD of whitened
    data), a PowerSpectrum (scale free), or a one-sided PSD array (F,) or (B, F)
    on a uniform grid from DC to Nyquist, in the data's units squared per hertz
    (interpolated, each row floored at 1e-4 of its median). An array is
    normalised in float64 by its median: the PSD of physical strain (~1e-46) is
    below float32's range.
    Returns (weight (..., F), scale or None).
    """
    num_freqs = num_samples // 2 + 1
    if psd is None:
        return jnp.ones((num_freqs,), dtype="float32"), None
    if isinstance(psd, PowerSpectrum):
        if psd.sample_rate_hertz != float(sample_rate_hertz):
            raise ValueError("sample_rate_hertz differs from the PowerSpectrum's.")
        # S = normalised * scale^2 / fs: with the data divided by scale, the
        # weight is normalised / 2.
        return psd.interpolate(num_samples) * (psd.inverse_bias / 2.0), psd.scale
    given = np.asarray(psd)
    weight = given.astype(np.float64) * (sample_rate_hertz / 2.0)
    if weight.shape[-1] != num_freqs:
        source = np.linspace(0.0, sample_rate_hertz / 2.0, weight.shape[-1])
        target = np.arange(num_freqs) * (sample_rate_hertz / num_samples)
        flat = weight.reshape(-1, weight.shape[-1])
        weight = np.stack([np.interp(target, source, row) for row in flat]).reshape(
            weight.shape[:-1] + (num_freqs,))
    # One reference for every row, so per-row weights keep their ratios and the
    # data and templates share one scale.
    reference = float(np.median(weight))
    tiny = float(np.finfo(np.float32).tiny)
    if not np.isfinite(reference) or reference <= 0 or (
            given.dtype.itemsize < 8 and np.median(np.abs(given)) < tiny):
        raise ValueError(
            "The PSD array's median is zero, below float32's range or not finite. The PSD "
            "of physical strain (~1e-46) underflows float32: pass the array as float64, or "
            "pass a PowerSpectrum (gf.estimate_psd or gf.psd)."
        )
    weight = weight / reference
    weight = np.maximum(weight, 1.0e-4 * np.median(weight, axis=-1, keepdims=True))
    return (ops.convert_to_tensor(weight.astype(np.float32)),
            ops.convert_to_tensor(np.float32(np.sqrt(reference))))


def _band(num_samples: int, sample_rate_hertz: float, lower, upper):
    config = SpectralConfig(lower_frequency_hertz=lower, upper_frequency_hertz=upper)
    return config.band_indices(num_samples, sample_rate_hertz)


@jax.jit(static_argnames=["num_samples", "kmin", "kmax"])
def _sigma_squared(h_fft, weight, num_samples, kmin, kmax):
    """
    <h|h> = 4 df sum |h~|^2 / S with h~ = rfft/fs, S = 2/fs * weight, df = fs/N:
    (2 / N) * sum |rfft(h)|^2 / weight. h_fft (..., F) against weight (F,) or
    (B, F) (then the result is (B, ...)).
    """
    power = ops.real(h_fft[..., kmin:kmax]) ** 2 + ops.imag(h_fft[..., kmin:kmax]) ** 2
    w = weight[..., kmin:kmax]
    if w.ndim > 1:
        power = power[None]
        w = w.reshape(w.shape[:1] + (1,) * (power.ndim - 2) + w.shape[-1:])
    return (2.0 / num_samples) * ops.sum(power / w, axis=-1)


def template_sigma(
    templates: jnp.ndarray,
    psd=None,
    sample_rate_hertz: float = 8192.0,
    lower_frequency_cutoff: Optional[float] = None,
    upper_frequency_cutoff: Optional[float] = None,
) -> jnp.ndarray:
    """
    sigma = sqrt(<h|h>) of each template with the module's SNR definition.
    templates: (NumTemplates, Samples) -> (NumTemplates,); with a per-row PSD
    (B, F) -> (B, NumTemplates).
    """
    templates = ops.cast(ops.convert_to_tensor(templates), "float32")
    num_samples = int(templates.shape[-1])
    kmin, kmax = _band(num_samples, sample_rate_hertz, lower_frequency_cutoff, upper_frequency_cutoff)
    weight, scale = _weight_on_grid(psd, num_samples, sample_rate_hertz)
    if scale is not None:
        if scale.size != 1:
            raise ValueError("template_sigma takes a PowerSpectrum of one row.")
        templates = templates / ops.reshape(scale, ())
        weight = ops.reshape(weight, (-1, weight.shape[-1]))
        if weight.shape[0] == 1:
            weight = weight[0]
    return ops.sqrt(_sigma_squared(jnp.fft.rfft(templates), weight, num_samples, kmin, kmax))


def optimal_snr(
    signal: jnp.ndarray,
    psd=None,
    sample_rate_hertz: float = 8192.0,
    lower_frequency_cutoff: Optional[float] = None,
    upper_frequency_cutoff: Optional[float] = None,
) -> float:
    """Optimal SNR of one signal (Samples,): the SNR an exactly matching template recovers."""
    signal = ops.convert_to_tensor(signal)
    return template_sigma(signal[None, :], psd, sample_rate_hertz,
                          lower_frequency_cutoff, upper_frequency_cutoff)[..., 0]


def matched_filter_fft(
    data: jnp.ndarray,
    templates: jnp.ndarray,
    psd=None,
    sample_rate_hertz: float = 8192.0,
    lower_frequency_cutoff: Optional[float] = None,
    upper_frequency_cutoff: Optional[float] = None,
) -> jnp.ndarray:
    """
    Matched-filter SNR timeseries |<d|h>(t)| / sigma_h, maximised over the
    template's phase (the complex SNR), for every template.

    data: (Samples,) or (Batch, Samples); templates: (NumTemplates, Samples).
    psd: None for whitened data (unit-variance white noise), otherwise the
    one-sided PSD of the data: an array (shared, or one row per batch entry) or
    a PowerSpectrum with one row per batch entry.
    The correlation is circular and the data are not windowed: give conditioned
    data (whitened, or tapered and high-passed). Raw strain leaks its
    low-frequency power into every bin. The template-bank search on whitened
    windows is gravyflow.src.detection.MatchedFilter.
    Returns (Batch, NumTemplates, Samples), or (NumTemplates, Samples) for 1D data.
    """
    data = ops.cast(ops.convert_to_tensor(data), "float32")
    templates = ops.cast(ops.convert_to_tensor(templates), "float32")
    data_1d = data.ndim == 1
    if data_1d:
        data = data[None, :]
    num_samples = int(data.shape[-1])
    kmin, kmax = _band(num_samples, sample_rate_hertz, lower_frequency_cutoff, upper_frequency_cutoff)
    weight, scale = _weight_on_grid(psd, num_samples, sample_rate_hertz)
    if scale is not None:
        data = data / ops.reshape(scale, (-1, 1))
        weight = ops.reshape(weight, (-1, weight.shape[-1]))
    # The template's amplitude cancels in the SNR; unit peak keeps its power sums
    # representable in float32 whatever its units.
    peak = ops.max(ops.abs(templates), axis=-1, keepdims=True)
    templates = templates / ops.where(peak > 0, peak, 1.0)
    snr_series = _matched_filter(data, templates, weight, kmin, kmax)
    return snr_series[0] if data_1d else snr_series


@jax.jit(static_argnames=["kmin", "kmax"])
def _matched_filter(data, templates, weight, kmin, kmax):
    num_samples = data.shape[-1]
    d_fft = jnp.fft.rfft(data, axis=-1)            # (Batch, F)
    h_fft = jnp.fft.rfft(templates, axis=-1)       # (NumTemplates, F)
    num_freqs = d_fft.shape[-1]
    band = np.zeros(num_freqs, dtype=np.float32)
    band[kmin:kmax] = 1.0
    w = weight[:, None, :] if weight.ndim > 1 else weight
    corr = d_fft[:, None, :] * jnp.conj(h_fft)[None, :, :] * (band / w)
    # Positive frequencies only: the inverse FFT then gives the complex inner
    # product, whose modulus is the SNR maximised over the template's phase.
    # <d|h>(t) = 4 df sum d~ h~* / S e^{2 pi i f t} = 2 * ifft(D H* / weight).
    full = jnp.zeros(corr.shape[:-1] + (num_samples,), dtype=corr.dtype).at[..., :num_freqs].set(corr)
    inner = 2.0 * jnp.fft.ifft(full, axis=-1)
    sigma = ops.sqrt(_sigma_squared(h_fft, weight, num_samples, kmin, kmax))  # (T,) or (B, T)
    sigma = jnp.where(sigma > 0, sigma, 1.0)
    sigma = sigma[None, :, None] if sigma.ndim == 1 else sigma[:, :, None]
    return jnp.abs(inner) / sigma
