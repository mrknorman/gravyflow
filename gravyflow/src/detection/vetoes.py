"""
The matched filter's noise-robustness methods, each as PyCBC publishes and
runs it: gating of loud transients, the PSD-variation statistic, and the
reweighted SNR that combines them with the chi-squared and sine-Gaussian
vetoes.

**Gating** (PyCBC's autogating: Usman et al. 2016, Class. Quantum Grav. 33,
215004, section 2.4; ``pycbc.strain.detect_loud_glitches`` and
``gate_data``). The strain is whitened and its magnitude, in units of the
whitened noise's standard deviation, thresholded; samples above threshold
are clustered with FindChirp's rule over ``cluster_seconds``, and each
cluster's peak is a gate: the raw strain is zeroed over +-``width_seconds``
of it and tapered to zero over ``taper_seconds`` on each side by an inverted
Tukey window. Gating needs no model: `autogate` serves any search's input.

**PSD variation** (Mozzon et al. 2020, Class. Quantum Grav. 37, 215014;
``pycbc.psd.variation.calc_filt_psd_variation``). The strain is filtered with
a band-pass (20-480 Hz), a weighting of a CBC template's spectrum (f^-7/6)
and the whitening of its long-term PSD (median Welch, 8 s segments, 4 s
stride), normalised so that stationary noise with that PSD gives unit
variance. Its mean square over 0.25 s stretches, with each stretch larger than
twice the mean of its neighbours replaced by that mean, is averaged over 8 s
windows once a second: v(t), about 1 in stationary noise, larger where the
noise is louder than the long-term PSD. As in PyCBC, the 8 s window that
starts at t0 is labelled with time t0 + 8 s, and a trigger takes v linearly
interpolated at its time (`value_at`).

**The ranking** (``pycbc.events.ranking.newsnr_sgveto_psdvar``). With the SNR
rho, the reduced chi-squared chi_r^2, the sine-Gaussian chi-squared chi_sg^2
and v (taken as 1 where it is below 0.65, PyCBC's minimum expected value):
rho' = rho v^-1/2, chi_r'^2 = chi_r^2 / v, rho_hat = newsnr(rho', chi_r'^2)
(Babak et al. 2013), then rho_hat (chi_sg^2 / 4)^-1/2 where chi_sg^2 > 4
(Nitz 2018, Class. Quantum Grav. 35, 035016, eq. 7).

**The sine-Gaussian veto** (Nitz 2018, eqs. 5-7; ``pycbc.vetoes.sgchisq``).
Seven sine-Gaussian tiles of quality factor Q = 20, at 30 to 120 Hz in 15 Hz
steps above the template's peak frequency, each over +-50 Hz of its centre,
centred at the trigger's time: in Gaussian noise each tile's |SNR|^2 is
chi-squared with 2 degrees of freedom, so chi_sg^2 = sum |rho_i|^2 / (2N) is a
reduced chi-squared with 2N = 14. A signal has no power above its peak
frequency; a short glitch that rings a high-mass template does. The peak
frequency is PyCBC's estimate from the chi-squared bins: the start of the last
bin plus the width of the one before it. Where a tile would reach 90% of the
Nyquist frequency the test is not applied (chi_sg^2 = 1), as in PyCBC. The
tiles are computed in the streaming filter's chi-squared kernel.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sig

#: The sine-Gaussian tiles (Nitz 2018): their quality factor, their centres'
#: offsets above the template's peak frequency (Hz), and their half-width (Hz).
SG_QUALITY = 20.0
SG_OFFSETS_HERTZ = (30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0)
SG_HALF_WIDTH_HERTZ = 50.0
SG_NYQUIST_FRACTION = 0.9
SG_THRESHOLD = 4.0

#: PyCBC's minimum expected PSD variation, below which v is taken as 1.
MIN_EXPECTED_PSD_VARIATION = 0.65

#: newsnr's exponents (Babak et al. 2013, eq. 6), as pycbc.events.ranking uses them.
NEWSNR_Q = 6
NEWSNR_N = 2

#: The ranking's name: PyCBC's function of the same name.
RANKING_NAME = "newsnr_sgveto_psdvar"


# ---------------------------------------------------------------------------
# The ranking
# ---------------------------------------------------------------------------

def newsnr(snr, reduced_chisq):
    """PyCBC's newsnr (Babak et al. 2013, eq. 6; q = 6, n = 2): the SNR lowered where chi_r^2 exceeds 1."""
    snr = np.asarray(snr, dtype=np.float64)
    reduced_chisq = np.asarray(reduced_chisq, dtype=np.float64)
    return snr / ((1.0 + np.maximum(reduced_chisq, 1.0) ** (NEWSNR_Q // NEWSNR_N)) / 2.0) ** (1.0 / NEWSNR_Q)


def ranking_definition(sine_gaussian: bool = True, psd_variation: bool = True) -> dict:
    """
    The identity of the ranking `ranking_statistic` computes: its name and
    every parameter (newsnr's exponents, the PSD variation's floor, the
    sine-Gaussian threshold, and which corrections are applied). Stored
    triggers carry it, so a file ranked by another definition is refused.
    """
    return {"name": RANKING_NAME,
            "parameters": {"newsnr_q": float(NEWSNR_Q), "newsnr_n": float(NEWSNR_N),
                           "min_expected_psd_variation": float(MIN_EXPECTED_PSD_VARIATION),
                           "sg_threshold": float(SG_THRESHOLD),
                           "sine_gaussian": bool(sine_gaussian), "psd_variation": bool(psd_variation)}}


def ranking_corrections(definition) -> Tuple[bool, bool]:
    """
    (sine_gaussian, psd_variation): the corrections of a ranking definition,
    which must be one `ranking_definition` gives (this gravyflow's ranking),
    or it is refused by name.
    """
    try:
        corrections = (definition["parameters"]["sine_gaussian"], definition["parameters"]["psd_variation"])
    except (TypeError, KeyError):
        corrections = None
    if (corrections is None or not all(isinstance(value, bool) for value in corrections)
            or definition != ranking_definition(*corrections)):
        raise ValueError(f"The ranking {definition!r} is not this gravyflow's ranking, "
                         f"{ranking_definition()!r} (with either correction on or off).")
    return corrections


def ranking_statistic(snr, reduced_chisq, sg_chisq=None, psd_variation=None) -> np.ndarray:
    """
    PyCBC's newsnr_sgveto_psdvar (see the module docstring), float32; with
    sg_chisq or psd_variation None, that correction is not applied.
    """
    snr = np.asarray(snr, dtype=np.float64)
    reduced_chisq = np.asarray(reduced_chisq, dtype=np.float64)
    if psd_variation is not None:
        v = np.array(psd_variation, dtype=np.float64, copy=True)
        v[v < MIN_EXPECTED_PSD_VARIATION] = 1.0
        snr, reduced_chisq = snr * v ** -0.5, reduced_chisq / v
    rank = newsnr(snr, reduced_chisq)
    if sg_chisq is not None:
        sg = np.asarray(sg_chisq, dtype=np.float64)
        rank = np.where(sg > SG_THRESHOLD, rank / np.sqrt(np.maximum(sg, SG_THRESHOLD) / SG_THRESHOLD), rank)
    return rank.astype(np.float32)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Autogating:
    """
    PyCBC's autogating settings (pycbc_inspiral's --autogating-* options):
    `threshold` in whitened standard deviations (detect_loud_glitches'
    default, 50), `cluster_seconds` (FindChirp clustering, default 5 s),
    `width_seconds` (the half-width zeroed, default 0.25 s) and
    `taper_seconds` (default 0.25 s).
    """
    threshold: float = 50.0
    cluster_seconds: float = 5.0
    width_seconds: float = 0.25
    taper_seconds: float = 0.25


def findchirp_cluster(indices: np.ndarray, values: np.ndarray, window: int) -> np.ndarray:
    """
    FindChirp's clustering (``pycbc.events.findchirp_cluster_over_window``):
    in time order, an event within `window` samples of the event kept last
    replaces it when louder, and otherwise is kept as a new one. Returns the
    positions (in `indices`) of the events kept.
    """
    kept = []
    for i in range(len(indices)):
        if kept and indices[i] - indices[kept[-1]] <= window:
            if abs(values[i]) > abs(values[kept[-1]]):
                kept[-1] = i
        else:
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def loud_glitches(whitened: np.ndarray, sample_rate_hertz: float, settings: Autogating,
                  margin_samples: int = 0) -> np.ndarray:
    """
    The samples of the loud transients in `whitened` (strain whitened to unit
    variance): where its magnitude exceeds the threshold, FindChirp-clustered,
    `margin_samples` at each end (where the whitening filter is corrupted)
    ignored.
    """
    magnitude = np.abs(np.asarray(whitened, dtype=np.float64).reshape(-1))
    if margin_samples:
        magnitude[:margin_samples] = 0.0
        magnitude[-margin_samples:] = 0.0
    above = np.nonzero(magnitude > settings.threshold)[0]
    if not len(above):
        return above
    keep = findchirp_cluster(above, magnitude[above], int(settings.cluster_seconds * sample_rate_hertz))
    return above[keep]


def gate(strain: np.ndarray, samples: Sequence[int], sample_rate_hertz: float,
         width_seconds: float = 0.25, taper_seconds: float = 0.25) -> np.ndarray:
    """
    `strain` gated at each of `samples` (``pycbc.strain.gate_data``): zero over
    +-width_seconds, tapered by a Hann half-window over taper_seconds on each
    side (an inverted Tukey window), clipped to the series.

    The gate is defined on samples, not times: with h = floor(width_seconds
    fs) and p = floor(taper_seconds fs), the gate at sample c zeroes samples
    [c - h, c + h) and tapers the p samples before and after. This equals
    ``gate_data`` exactly when both widths are whole samples and the gate's
    time, c / fs from the series' start, is a whole number of nanoseconds (at
    2048 Hz, c a multiple of 4). ``gate_data`` takes the gate's time in
    seconds, which PyCBC holds to whole nanoseconds, and truncates the start
    sample it computes from it, so for other gate times its window starts
    one sample earlier; for widths that are not whole samples its zeroed
    stretch can also be a sample longer (it floors 2 fs (width + taper)
    whole, not each half). The search gates at samples (`loud_glitches`), so
    this convention is the exact one; one sample of a 0.25 s taper does not
    change what is gated.
    """
    out = np.array(strain, dtype=np.float64, copy=True)
    pad = int(taper_seconds * sample_rate_hertz)
    half = int(width_seconds * sample_rate_hertz)
    taper = 0.5 * (1.0 + np.cos(np.pi * np.arange(pad) / pad)) if pad else np.zeros(0)
    window = np.concatenate([taper, np.zeros(2 * half), taper[::-1]])
    for centre in samples:
        start = int(centre) - half - pad
        lo, hi = max(start, 0), min(start + len(window), len(out))
        if lo < hi:
            out[lo:hi] *= window[lo - start:hi - start]
    return out


def autogate(strain: np.ndarray, sample_rate_hertz: float, settings: Autogating = Autogating(),
             spectral_config=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (gated strain, gate samples): PyCBC's autogating of one detector's raw
    strain, whitened with the data path's `whiten` and its median-Welch
    spectrum (`spectral_config`, default the streaming filter's), its
    corrupted half-filter at each end ignored. Any search can gate its input
    with it.
    """
    import jax
    from gravyflow.src.dataset.conditioning.whiten import whiten
    from gravyflow.src.dataset.tools.psd import estimate_psd
    from gravyflow.src.rng.platform import cbc_device
    if spectral_config is None:
        from gravyflow.src.detection.stream import STREAM_SPECTRAL_CONFIG
        spectral_config = STREAM_SPECTRAL_CONFIG
    block = np.asarray(strain, dtype=np.float32)
    with jax.default_device(cbc_device()):
        spectrum = estimate_psd(block[None], sample_rate_hertz, spectral_config)
        white = np.asarray(whiten(block, spectrum, sample_rate_hertz), dtype=np.float64).reshape(-1)
    margin = spectral_config.num_taps(sample_rate_hertz) // 2
    samples = loud_glitches(white, sample_rate_hertz, settings, margin)
    if not len(samples):
        return np.asarray(strain, dtype=np.float64), samples
    return gate(strain, samples, sample_rate_hertz, settings.width_seconds, settings.taper_seconds), samples


# ---------------------------------------------------------------------------
# PSD variation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PSDVariation:
    """
    The PSD-variation settings, pycbc_inspiral's --psdvar-* defaults: the
    averaging window (`segment_seconds`, 8 s) and the outlier stretch
    (`short_segment_seconds`, 0.25 s), the long-term PSD's Welch segments
    (8 s, stride 4 s, median), the band (20-480 Hz), and the series' crop at
    each end (8 s, PyCBC's fixed strain_crop). The long segment is the
    stretch given (in the streaming filter, one block).
    """
    segment_seconds: float = 8.0
    short_segment_seconds: float = 0.25
    psd_seconds: float = 8.0
    psd_stride_seconds: float = 4.0
    low_frequency_hertz: float = 20.0
    high_frequency_hertz: float = 480.0
    crop_seconds: float = 8.0


def _median_welch(strain: np.ndarray, fs: int, seconds: float, stride: float):
    """The one-sided median-Welch PSD with Hann windows (scipy's, with its median bias correction)."""
    return sig.welch(strain, fs=fs, window="hann", nperseg=int(seconds * fs), noverlap=int((seconds - stride) * fs),
                     average="median", detrend=False)


def psd_variation(strain: np.ndarray, sample_rate_hertz: float,
                  settings: PSDVariation = PSDVariation()) -> Tuple[np.ndarray, np.ndarray]:
    """
    (times, v): the PSD variation of one long segment of raw strain, times in
    seconds from its first sample (each the end of its averaging window, as
    PyCBC labels them), one a second. See the module docstring.

    v compares the strain with its own long-term spectrum, which noise
    defines. Strain without noise (a lone signal, or zeros) has a long-term
    median spectrum that is zero, or at rounding level, somewhere: there v is
    0 / 0 or rounding, not a measurement. So the long-term spectrum must be
    positive and finite at every frequency above 0 and span at most
    1 / (float64's epsilon) over the band (real detector noise spans far
    less), and v must be finite; otherwise the strain is refused by name.
    """
    fs = int(sample_rate_hertz)
    strain = np.asarray(strain, dtype=np.float64)
    band = sig.firwin(4 * fs, [settings.low_frequency_hertz, settings.high_frequency_hertz], pass_zero=False,
                      window="hann", fs=fs)
    band = np.concatenate([band, np.zeros(int(settings.psd_seconds * fs) - len(band))])  # PyCBC's resize
    band = np.abs(np.fft.rfft(band))
    freqs, plong = _median_welch(strain, fs, settings.psd_seconds, settings.psd_stride_seconds)
    in_band = (freqs >= settings.low_frequency_hertz) & (freqs <= settings.high_frequency_hertz)
    if (not np.all(np.isfinite(plong[1:])) or not np.all(plong[1:] > 0.0)
            or np.max(plong[in_band]) > np.min(plong[in_band]) / np.finfo(np.float64).eps):
        raise _no_psd_variation("its long-term (median Welch) spectrum is zero, not finite or at rounding level "
                                "at some frequency")
    with np.errstate(divide="ignore"):
        weight = freqs ** (-7.0 / 6.0) * band / np.sqrt(plong)
    weight[0] = 0.0
    weight *= (np.sum(np.abs(weight) ** 2) / (len(weight) - 1.0)) ** -0.5
    whitening = np.sqrt(2.0 / fs) / np.sqrt(plong)
    whitening[0] = 0.0
    length = int(settings.psd_seconds * fs)
    full = sig.windows.hann(length) * np.roll(np.fft.irfft(whitening * weight), int(settings.psd_seconds / 2) * fs)
    filtered = sig.fftconvolve(strain, full, mode="same")
    crop = int(settings.crop_seconds * fs)
    filtered = filtered[crop:-crop]
    # Mean squares over short stretches, outliers replaced by their neighbours' mean.
    short = int(settings.short_segment_seconds * fs)
    usable = len(filtered) // short * short
    short_ms = np.mean(filtered[:usable].reshape(-1, short) ** 2, axis=1)
    mean = 0.5 * (short_ms[2:] + short_ms[:-2])
    outliers = short_ms[1:-1] > 2.0 * mean
    short_ms[1:-1][outliers] = mean[outliers]
    per = int(round(1.0 / settings.short_segment_seconds))
    seconds = len(filtered) / fs
    count = int(seconds - settings.segment_seconds + 1)
    segment = int(settings.segment_seconds)
    values = np.array([np.mean(short_ms[per * j:per * (j + segment)]) for j in range(max(count, 0))])
    times = settings.crop_seconds + settings.segment_seconds + np.arange(len(values), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise _no_psd_variation("its PSD variation is not finite")
    return times, values


def _no_psd_variation(reason: str) -> ValueError:
    return ValueError(f"The strain has no PSD variation: {reason}, as for strain without noise (a lone signal, or "
                      "zeros). PSD variation measures noise against its own long-term spectrum; search such strain "
                      "with psd_variation=None.")


def value_at(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """
    v at the times `at`, linearly interpolated (PyCBC's find_trigger_value).
    Outside the series the nearest value is taken: PyCBC fills 1 there, where
    its search never looks (its segments' padding exceeds the series' crop),
    but a short block's searched span can begin before its first value.
    """
    if not len(values):
        return np.ones(np.shape(at))
    return np.interp(at, times, values)


__all__ = [
    "Autogating",
    "PSDVariation",
    "autogate",
    "gate",
    "loud_glitches",
    "newsnr",
    "psd_variation",
    "ranking_definition",
    "ranking_statistic",
    "value_at",
]
