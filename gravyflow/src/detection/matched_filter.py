"""
The matched-filter baseline: a CBC template-bank search on whitened data.

The data path whitens each example with the PSD of its offsource
(`gf.whiten`), and models take that whitened onsource window. The matched
filter takes the same window and the same spectrum, and whitens its templates
with the exact FIR filter the data were whitened with
(`whitening_filter`), so a template sees what an injected signal of its
parameters looks like in the data. On whitened data the noise is white with
unit variance, so the inner product is a plain sum over samples.

The statistic, for every template and every merger time in the window:

- The signal in a window is the part of the whitened waveform inside it, which
  depends on where the merger falls: a long inspiral starts before the window.
  So the filter correlates the window with the whitened template linearly (no
  wrap-around), and normalises each merger time by the norm of the part of the
  template that falls inside the window.
- The SNR is maximised over the template's phase exactly: with the in-window
  quadratures g_c, g_s (the template and its Hilbert transform), z = (<d|g_c>,
  <d|g_s>) and their Gram matrix M, rho^2 = z^T M^-1 z. In Gaussian noise rho^2
  is chi-squared with 2 degrees of freedom at every merger time, and a template
  that matches the signal recovers the signal's optimal SNR in the window (the
  SNR the dataset's labels report).
- Detectors are combined in coincidence: rho_net^2(t) = sum over detectors of
  the largest rho_i^2 within the light travel time (plus a margin) of t in the
  first detector, for the same template.

The reported time is the merger time in the window (the template's merger is
its reference point), in seconds from the window's start.

Memory is bounded by chunking over examples and templates; the cost per
(example, detector, template) is two complex FFTs of about the template buffer
plus a filter length.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import keras
import numpy as np
from keras import ops
from scipy.fft import next_fast_len

from gravyflow.src.dataset.config import Defaults
from gravyflow.src.dataset.conditioning.whiten import whitening_filter
from gravyflow.src.dataset.tools.psd import PowerSpectrum, SpectralConfig, estimate_psd
from gravyflow.src.detection.template_grid import TemplateBank, TemplateGrid, cbc_templates

SPEED_OF_LIGHT = 299792458.0
# Added to the light travel time between detectors in the coincidence window,
# for the timing error of each detector's peak.
COINCIDENCE_MARGIN_SECONDS = 0.002


def light_travel_seconds(ifos: Sequence) -> float:
    """The largest light travel time between any two of `ifos` (gf.IFO members), in seconds."""
    from astropy import coordinates, units
    positions = []
    for ifo in ifos:
        v = ifo.value
        loc = coordinates.EarthLocation.from_geodetic(
            v.longitude_radians * units.rad, v.latitude_radians * units.rad, v.height_meters * units.meter)
        positions.append([loc.x.value, loc.y.value, loc.z.value])
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return float(distances.max() / SPEED_OF_LIGHT)


def _as_spectral_config(value) -> SpectralConfig:
    if value is None:
        return SpectralConfig()
    if isinstance(value, SpectralConfig):
        return value
    if isinstance(value, dict):
        return SpectralConfig(**value)
    raise TypeError("spectral_config must be a gf.SpectralConfig, a dict of its fields, or None.")


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["n_fft", "merger_index", "num_samples", "lags", "max_shift",
                                   "template_chunk", "reduce"])
def _search_chunks(data, fir, template_fft, **kwargs):
    """_search over batch chunks: data (C, Bc, I, N), fir (C, Bc, I, K) or None."""
    if fir is None:
        return jax.lax.map(lambda d: _search(d, None, template_fft, **kwargs), data)
    return jax.lax.map(lambda args: _search(args[0], args[1], template_fft, **kwargs), (data, fir))


def _search(data, fir, template_fft, *, n_fft, merger_index, num_samples, lags, max_shift,
            template_chunk, reduce):
    """
    data (B, I, N) whitened; fir (B, I, K) whitening filters, or None for data that
    are white already (then template_fft must already be band-limited);
    template_fft (T, F), T a multiple of template_chunk.

    merger_index: the merger's index in the linear convolution of a raw template
    with the filter. lags: (first, stop) merger times searched, in window samples.
    max_shift: the coincidence window in samples.

    reduce=True returns per example (rho_net (B,), template (B,), lag (B,));
    reduce=False returns rho_net (B, T, num_lags).
    """
    num_freqs = n_fft // 2 + 1
    lag_lo, lag_hi = lags
    num_lags = lag_hi - lag_lo
    d_fft = jnp.conj(jnp.fft.rfft(data, n=n_fft, axis=-1))                  # (B, I, F)
    if fir is not None:
        f_fft = jnp.fft.rfft(fir, n=n_fft, axis=-1)                         # (B, I, F)
    # One-sided weights of the analytic signal: 1 at DC and Nyquist, 2 between.
    analytic = np.full(num_freqs, 2.0, dtype=np.float32)
    analytic[0] = 1.0
    analytic[-1] = 1.0
    # Shift s = merger_index - lag; window sums over [s, s + N) of the template.
    shifts = merger_index - np.arange(lag_lo, lag_hi)                       # (L,)

    def one_chunk(h_chunk):
        # h_chunk (Tc, F)
        g = h_chunk[None, None] if fir is None else f_fft[:, :, None, :] * h_chunk[None, None]  # (B, I, Tc, F)
        g = g * analytic
        pad = [(0, 0)] * (g.ndim - 1) + [(0, n_fft - num_freqs)]
        g_a = jnp.fft.ifft(jnp.pad(g * d_fft[:, :, None, :], pad), axis=-1)  # z(s) = sum_t d(t) g_a(t + s)
        z = g_a[..., shifts]                                                # (B, I, Tc, L)
        t_a = jnp.fft.ifft(jnp.pad(g, pad), axis=-1)                        # analytic whitened template
        gc, gs = jnp.real(t_a), jnp.imag(t_a)

        def window_sums(x):
            c = jnp.concatenate([jnp.zeros(x.shape[:-1] + (1,), x.dtype), jnp.cumsum(x, axis=-1)], axis=-1)
            return c[..., shifts + num_samples] - c[..., shifts]

        a = window_sums(gc * gc)
        c = window_sums(gs * gs)
        b = window_sums(gc * gs)
        zc, zs = jnp.real(z), jnp.imag(z)
        det = a * c - b * b
        scale = jnp.maximum(a * c, 1e-30)
        ok = (det > 1e-6 * scale) & (a > 0) & (c > 0)
        rho2 = jnp.where(ok, (c * zc * zc - 2.0 * b * zc * zs + a * zs * zs) / jnp.where(ok, det, 1.0), 0.0)
        rho2 = jnp.maximum(rho2, 0.0)                                       # (B, I, Tc, L)
        if rho2.shape[1] > 1:
            others = rho2[:, 1:]
            if max_shift > 0:
                others = jax.lax.reduce_window(
                    others, 0.0, jax.lax.max, (1, 1, 1, 2 * max_shift + 1), (1, 1, 1, 1),
                    [(0, 0), (0, 0), (0, 0), (max_shift, max_shift)])
            net = rho2[:, 0] + jnp.sum(others, axis=1)
        else:
            net = rho2[:, 0]
        return net                                                          # (B, Tc, L)

    chunks = template_fft.reshape((-1, template_chunk, num_freqs))
    if not reduce:
        out = jax.lax.map(one_chunk, chunks)                                # (C, B, Tc, L)
        out = jnp.moveaxis(out, 0, 1).reshape((data.shape[0], -1, num_lags))
        return jnp.sqrt(out)

    batch = data.shape[0]

    def step(carry, inputs):
        best, best_t, best_l = carry
        index, h_chunk = inputs
        net = one_chunk(h_chunk).reshape((batch, -1))
        arg = jnp.argmax(net, axis=-1)
        value = jnp.take_along_axis(net, arg[:, None], axis=-1)[:, 0]
        better = value > best
        t = index * template_chunk + arg // num_lags
        return (jnp.where(better, value, best), jnp.where(better, t, best_t),
                jnp.where(better, arg % num_lags, best_l)), None

    init = (jnp.full((batch,), -1.0, jnp.float32), jnp.zeros((batch,), jnp.int32), jnp.zeros((batch,), jnp.int32))
    (best, best_t, best_l), _ = jax.lax.scan(step, init, (jnp.arange(chunks.shape[0]), chunks))
    return jnp.sqrt(jnp.maximum(best, 0.0)), best_t, best_l + lag_lo


# ---------------------------------------------------------------------------
# The matched filter
# ---------------------------------------------------------------------------

@dataclass
class MatchedFilter:
    """
    A CBC template-bank matched filter on whitened data (see the module docstring).

    Args:
        bank: the template bank; default `TemplateBank.stochastic` over
            `mass_range` at `minimal_match`, placed for this window and band.
        mass_range, minimal_match: the default bank's mass range (solar masses,
            both components) and minimal match.
        sample_rate_hertz, onsource_duration_seconds, crop_duration_seconds:
            the dataset's window: the whitened window the filter takes, and the
            crop on each side of it (the whitening filter is designed on the
            onsource plus both crops, as the dataset designs it).
        spectral_config: the dataset's gf.SpectralConfig (band, PSD estimator,
            whitening filter). The SNR's band starts at its lower frequency, and
            so do the templates.
        search_window_seconds: (start, end) of the merger times searched, in
            seconds from the window's start; default the whole window.
        ifos: the detectors of the data's channels, for the coincidence window
            (their largest light travel time plus 2 ms). Required for more than
            one channel unless `max_time_offset_seconds` is given.
        max_time_offset_seconds: the coincidence window, overriding `ifos`.
        batch_chunk, template_chunk: examples and templates per step; memory
            grows with their product (see `chunk_bytes`).
    """

    bank: Optional[TemplateBank] = None
    mass_range: Tuple[float, float] = (5.0, 95.0)
    minimal_match: float = 0.97
    sample_rate_hertz: float = Defaults.sample_rate_hertz
    onsource_duration_seconds: float = Defaults.onsource_duration_seconds
    crop_duration_seconds: float = Defaults.crop_duration_seconds
    spectral_config: Optional[SpectralConfig] = None
    search_window_seconds: Optional[Tuple[float, float]] = None
    ifos: Optional[Sequence] = None
    max_time_offset_seconds: Optional[float] = None
    batch_chunk: int = 16
    template_chunk: int = 8

    _templates: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.spectral_config = _as_spectral_config(self.spectral_config)
        fs = float(self.sample_rate_hertz)
        self.sample_rate_hertz = fs
        self.num_samples = int(round(self.onsource_duration_seconds * fs))
        self.num_whitening_samples = self.num_samples + 2 * int(round(self.crop_duration_seconds * fs))
        ntaps = self.spectral_config.num_taps(fs)
        self.num_taps = ntaps + (ntaps % 2)
        if 2 * int(round(self.crop_duration_seconds * fs)) < self.num_taps:
            raise ValueError("crop_duration_seconds must be at least half the whitening filter, as in the dataset.")
        if self.spectral_config.lower_frequency_hertz is None:
            raise ValueError("The matched filter needs a lower band edge (spectral_config.lower_frequency_hertz).")
        self.spectral_config.validate_band(fs)
        if self.bank is None:
            import gravyflow as gf
            # The bank is placed with the first detector's noise curve; H1 and L1
            # share theirs, so both get the same (bundled) bank.
            ifo = self.ifos[0] if self.ifos else gf.IFO.L1
            if ifo.value.optimal_psd_path == gf.IFO.L1.value.optimal_psd_path:
                ifo = gf.IFO.L1
            self.bank = TemplateBank.stochastic(
                mass_range=tuple(self.mass_range), minimal_match=self.minimal_match, sample_rate_hertz=fs,
                visible_seconds=self.onsource_duration_seconds + self.num_taps / (2.0 * fs),
                lower_frequency_hertz=self.spectral_config.lower_frequency_hertz,
                ifo=ifo)
        elif isinstance(self.bank, TemplateGrid):
            self.bank = TemplateBank.from_grid(self.bank)
        # The raw template buffer: the merger at the centre, and enough on each
        # side that the whitened template is exact over the window at any lag.
        half = self.num_samples + self.num_taps // 2
        self.template_samples = 2 * half
        self.template_merger_index = half
        # The merger's index in the linear convolution with the filter: the
        # dataset's 'same'-mode convolution starts (ntaps - 1) // 2 samples in.
        self.merger_index = half + (self.num_taps - 1) // 2
        self.n_fft = int(next_fast_len(self.template_samples + self.num_taps - 1, real=True))
        if self.search_window_seconds is None:
            self.lags = (0, self.num_samples)
        else:
            lo, hi = self.search_window_seconds
            lags = (int(round(lo * fs)), int(round(hi * fs)))
            if not 0 <= lags[0] < lags[1] <= self.num_samples:
                raise ValueError("search_window_seconds must lie inside the window, start before end.")
            self.lags = lags

    # ------------------------------------------------------------ templates

    @property
    def num_templates(self) -> int:
        return self.bank.num_templates

    @property
    def template_masses(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bank.mass_1, self.bank.mass_2

    def generate_templates(self) -> np.ndarray:
        """
        The raw templates (T, template_samples), float32, unit peak: the plus
        polarisation of each bank point from the dataset's CBC generator, merger
        at index template_merger_index, no content below the band's lower edge.
        """
        if self._templates is None:
            self._templates = cbc_templates(
                self.bank.mass_1, self.bank.mass_2, self.sample_rate_hertz,
                self.template_samples / self.sample_rate_hertz, self.spectral_config.lower_frequency_hertz)
        return self._templates

    def template_spectra(self, templates=None) -> jnp.ndarray:
        """rfft of the templates on the search's FFT grid, padded to a multiple of template_chunk."""
        templates = self.generate_templates() if templates is None else templates
        templates = jnp.asarray(templates, dtype=jnp.float32)
        pad = (-templates.shape[0]) % self.template_chunk
        if pad:
            templates = jnp.pad(templates, ((0, pad), (0, 0)))
        return jnp.fft.rfft(templates, n=self.n_fft, axis=-1)

    # ------------------------------------------------------------- inputs

    def spectrum(self, background) -> Optional[PowerSpectrum]:
        """The PowerSpectrum the data were whitened with: given, or estimated from an offsource."""
        if background is None or isinstance(background, PowerSpectrum):
            if background is not None and background.sample_rate_hertz != self.sample_rate_hertz:
                raise ValueError("The PowerSpectrum's sample rate differs from the matched filter's.")
            return background
        return estimate_psd(ops.cast(ops.convert_to_tensor(background), "float32"),
                            self.sample_rate_hertz, self.spectral_config)

    def filters(self, background, shape) -> Optional[jnp.ndarray]:
        """The whitening FIR of each (example, detector), (B, I, K), or None for white data."""
        spectrum = self.spectrum(background)
        if spectrum is None:
            return None
        fir = whitening_filter(spectrum, self.num_whitening_samples)
        rows = tuple(fir.shape[:-1])
        if rows != tuple(shape[:-1]) and int(np.prod(rows)) == int(np.prod(shape[:-1])):
            fir = fir.reshape(tuple(shape[:-1]) + (fir.shape[-1],))  # (B,) rows for (B, 1) data
        elif rows != tuple(shape[:-1]):
            if int(np.prod(rows)) != 1:
                raise ValueError(
                    f"The spectrum has rows {rows}, the data {tuple(shape[:-1])}: one spectrum row per "
                    "(example, detector) is needed, or a single row for all.")
            fir = jnp.broadcast_to(fir.reshape((1, 1, -1)), tuple(shape[:-1]) + (fir.shape[-1],))
        return fir

    def _white_band_spectra(self, h_fft):
        kmin, kmax = self.spectral_config.band_indices(self.n_fft, self.sample_rate_hertz)
        mask = np.zeros(self.n_fft // 2 + 1, dtype=np.float32)
        mask[kmin:kmax] = 1.0
        return h_fft * mask

    def _prepare(self, data, background):
        data = ops.cast(ops.convert_to_tensor(data), "float32")
        rank = len(data.shape)
        if rank == 1:
            data = data[None, None]
        elif rank == 2:
            data = data[:, None]
        elif rank != 3:
            raise ValueError("data must be (samples,), (batch, samples) or (batch, detectors, samples).")
        if data.shape[-1] != self.num_samples:
            raise ValueError(f"The window has {data.shape[-1]} samples; this filter takes {self.num_samples}.")
        if background is not None and not isinstance(background, PowerSpectrum):
            background = ops.cast(ops.convert_to_tensor(background), "float32")
            if len(background.shape) == rank:
                background = background.reshape(data.shape[:-1] + (background.shape[-1],))
        fir = self.filters(background, data.shape)
        return data, fir, rank

    def max_time_offset_samples(self, num_detectors: int) -> int:
        if num_detectors < 2:
            return 0
        if self.max_time_offset_seconds is not None:
            seconds = float(self.max_time_offset_seconds)
        elif self.ifos is not None:
            if len(self.ifos) != num_detectors:
                raise ValueError(f"{len(self.ifos)} ifos given for {num_detectors} data channels.")
            seconds = light_travel_seconds(self.ifos) + COINCIDENCE_MARGIN_SECONDS
        else:
            raise ValueError("Multi-detector data need `ifos` (or max_time_offset_seconds) for the coincidence window.")
        return int(np.ceil(seconds * self.sample_rate_hertz))

    def chunk_bytes(self, num_detectors: int = 1) -> int:
        """An estimate of the working memory of one step (complex64 and float32 arrays)."""
        per = self.batch_chunk * num_detectors * self.template_chunk
        return int(per * (self.n_fft * (8 * 4 + 4 * 2) + (self.lags[1] - self.lags[0]) * 4 * 8))

    # ------------------------------------------------------------- search

    def _run(self, data, background, reduce: bool, h_fft=None):
        data, fir, rank = self._prepare(data, background)
        if h_fft is None:
            h_fft = self.template_spectra()
        if fir is None:
            h_fft = self._white_band_spectra(h_fft)
        batch = data.shape[0]
        step = min(self.batch_chunk, batch)
        pad = (-batch) % step
        if pad:
            data = jnp.pad(data, ((0, pad), (0, 0), (0, 0)))
            fir = None if fir is None else jnp.pad(fir, ((0, pad), (0, 0), (0, 0)))
        chunks = data.shape[0] // step
        data = data.reshape((chunks, step) + tuple(data.shape[1:]))
        fir = None if fir is None else fir.reshape((chunks, step) + tuple(fir.shape[1:]))
        out = _search_chunks(
            data, fir, h_fft, n_fft=self.n_fft, num_samples=self.num_samples,
            merger_index=self.template_merger_index if fir is None else self.merger_index,
            lags=self.lags, max_shift=self.max_time_offset_samples(data.shape[2]),
            template_chunk=self.template_chunk, reduce=reduce)
        if reduce:
            out = tuple(o.reshape((-1,))[:batch] for o in out)
        else:
            out = out.reshape((-1,) + tuple(out.shape[2:]))[:batch, :self.num_templates]
        return out, rank

    def snr_series(self, data, background=None) -> jnp.ndarray:
        """
        The network SNR of every template at every searched merger time:
        (batch, templates, lags), or (templates, lags) for unbatched data. Its
        size is the product of those; use `max_snr` for large inputs.

        data: whitened windows, (samples,), (batch, samples) or (batch,
            detectors, samples). background: the spectrum they were whitened
            with (a PowerSpectrum, or the offsource it was estimated from), or
            None for data that are white already (unit-variance noise).
        """
        out, rank = self._run(data, background, reduce=False)
        return out[0] if rank == 1 else out

    def max_snr(self, data, background=None) -> Dict[str, np.ndarray]:
        """
        The loudest template and merger time of each example: a dict of (batch,)
        arrays 'snr' (network SNR), 'template_index', 'time_seconds' (merger time
        from the window's start), 'mass_1', 'mass_2'. Unbatched data give scalars.
        """
        (snr, template, lag), rank = self._run(data, background, reduce=True)
        template = np.asarray(template)
        result = {
            "snr": np.asarray(snr, dtype=np.float32),
            "template_index": template,
            "time_seconds": np.asarray(lag) / self.sample_rate_hertz,
            "mass_1": self.bank.mass_1[template],
            "mass_2": self.bank.mass_2[template],
        }
        if rank == 1:
            result = {k: v[0] for k, v in result.items()}
        return result

    def statistic(self, data, background=None) -> jnp.ndarray:
        """The loudest network SNR of each example, (batch,) float32, traceable under jit."""
        (snr, _, _), _ = self._run(data, background, reduce=True)
        return snr

    def detect(self, data, background=None, threshold: float = 8.0,
               cluster_seconds: float = 0.1) -> List[Dict[str, Any]]:
        """
        Triggers above `threshold`, clustered in time per example (see
        `find_triggers`), each with 'batch_index', 'snr', 'time_seconds' (merger
        time), 'template_index', 'mass_1' and 'mass_2'.
        """
        series = self.snr_series(data, background)
        series = np.asarray(series)
        if series.ndim == 2:
            series = series[None]
        triggers = find_triggers(series, threshold, int(round(cluster_seconds * self.sample_rate_hertz)))
        for trig in triggers:
            trig["time_seconds"] = (trig.pop("time_index") + self.lags[0]) / self.sample_rate_hertz
            trig["mass_1"] = float(self.bank.mass_1[trig["template_index"]])
            trig["mass_2"] = float(self.bank.mass_2[trig["template_index"]])
        return triggers


def find_triggers(snr, threshold: float = 8.0, cluster_window: int = 100) -> List[Dict[str, Any]]:
    """
    Triggers in SNR series above `threshold`, clustered in time.

    snr: (templates, samples) or (batch, templates, samples). The series is first
    maximised over templates. Above-threshold samples of one example whose gap to
    the previous above-threshold sample is at most `cluster_window` samples form
    one trigger (a gap rule, so an excursion of any length is one trigger), at
    its loudest sample. Returns dicts with 'snr', 'time_index' and
    'template_index', plus 'batch_index' for batched input.
    """
    snr = np.asarray(snr)
    batched = snr.ndim == 3
    if snr.ndim == 2:
        snr = snr[None]
    if snr.ndim != 3:
        raise ValueError("snr must be (templates, samples) or (batch, templates, samples).")
    best = snr.max(axis=1)
    template = snr.argmax(axis=1)
    triggers = []
    for b in range(best.shape[0]):
        above = np.flatnonzero(best[b] > threshold)
        if not len(above):
            continue
        breaks = np.flatnonzero(np.diff(above) > cluster_window) + 1
        for cluster in np.split(above, breaks):
            peak = int(cluster[np.argmax(best[b, cluster])])
            trig = {"snr": float(best[b, peak]), "time_index": peak, "template_index": int(template[b, peak])}
            if batched:
                trig["batch_index"] = b
            triggers.append(trig)
    return triggers


# =============================================================================
# Keras layer
# =============================================================================

@keras.saving.register_keras_serializable(package="gravyflow")
class MatchedFilterLayer(keras.Layer):
    """
    The matched filter as a Keras layer: [whitened onsource (B, I, N), offsource
    (B, I, M)] in, logits (B, 2) out, with z1 - z0 = (rho_net - snr_threshold) /
    temperature, so a softmax gives P(signal) = sigmoid((rho - threshold) / T)
    and the validator ranks examples by their network SNR.

    The offsource is the one the dataset estimated the whitening PSD from; the
    layer estimates the same PowerSpectrum from it (with `spectral_config`) and
    whitens its templates with the dataset's filter.

    The bank's masses are a non-trainable weight, so a model's fingerprint
    includes its bank.
    """

    def __init__(
        self,
        mass_range: Tuple[float, float] = (5.0, 95.0),
        minimal_match: float = 0.97,
        sample_rate_hertz: float = Defaults.sample_rate_hertz,
        onsource_duration_seconds: float = Defaults.onsource_duration_seconds,
        crop_duration_seconds: float = Defaults.crop_duration_seconds,
        spectral_config=None,
        search_window_seconds=None,
        ifos=None,
        max_time_offset_seconds=None,
        snr_threshold: float = 8.0,
        temperature: float = 2.0,
        batch_chunk: int = 16,
        template_chunk: int = 8,
        bank_masses=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import gravyflow as gf
        self.mass_range = tuple(float(m) for m in mass_range)
        self.minimal_match = float(minimal_match)
        self.sample_rate_hertz = float(sample_rate_hertz)
        self.onsource_duration_seconds = float(onsource_duration_seconds)
        self.crop_duration_seconds = float(crop_duration_seconds)
        self.spectral_config = _as_spectral_config(spectral_config)
        self.search_window_seconds = None if search_window_seconds is None else tuple(search_window_seconds)
        self.ifos = None if ifos is None else [gf.IFO[i] if isinstance(i, str) else i for i in ifos]
        self.max_time_offset_seconds = max_time_offset_seconds
        self.snr_threshold = float(snr_threshold)
        self.temperature = float(temperature)
        self.batch_chunk = int(batch_chunk)
        self.template_chunk = int(template_chunk)
        bank = None
        if bank_masses is not None:
            bank = TemplateBank.from_masses(*np.asarray(bank_masses, dtype=np.float64))
        self._matched_filter = MatchedFilter(
            bank=bank, mass_range=self.mass_range, minimal_match=self.minimal_match,
            sample_rate_hertz=self.sample_rate_hertz, onsource_duration_seconds=self.onsource_duration_seconds,
            crop_duration_seconds=self.crop_duration_seconds, spectral_config=self.spectral_config,
            search_window_seconds=self.search_window_seconds, ifos=self.ifos,
            max_time_offset_seconds=self.max_time_offset_seconds,
            batch_chunk=self.batch_chunk, template_chunk=self.template_chunk)

    def build(self, input_shape):
        mf = self._matched_filter
        templates = mf.generate_templates()
        self.bank_masses = self.add_weight(
            name="bank_masses", shape=(2, mf.num_templates), trainable=False, dtype="float32",
            initializer=keras.initializers.Constant(np.stack([mf.bank.mass_1, mf.bank.mass_2]).astype(np.float32)))
        self.templates = self.add_weight(
            name="templates", shape=templates.shape, trainable=False, dtype="float32",
            initializer=keras.initializers.Constant(templates))
        super().build(input_shape)

    def _split(self, inputs):
        import gravyflow as gf
        if isinstance(inputs, dict):
            return (inputs[gf.ReturnVariables.WHITENED_ONSOURCE.name], inputs[gf.ReturnVariables.OFFSOURCE.name])
        whitened, offsource = inputs
        return whitened, offsource

    def call(self, inputs, training=None):
        whitened, offsource = self._split(inputs)
        mf = self._matched_filter
        h_fft = mf.template_spectra(self.templates)
        (snr, _, _), _ = mf._run(whitened, offsource, reduce=True, h_fft=h_fft)
        logit = (snr - self.snr_threshold) / self.temperature
        return ops.stack([ops.zeros_like(logit), logit], axis=-1)

    def compute_output_spec(self, inputs, training=None):
        whitened, _ = self._split(inputs)
        return keras.KerasTensor((whitened.shape[0], 2), dtype="float32")

    def compute_output_shape(self, input_shape):
        first = input_shape[0] if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)) \
            else next(iter(input_shape.values())) if isinstance(input_shape, dict) else input_shape
        return (first[0], 2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "mass_range": list(self.mass_range),
            "minimal_match": self.minimal_match,
            "sample_rate_hertz": self.sample_rate_hertz,
            "onsource_duration_seconds": self.onsource_duration_seconds,
            "crop_duration_seconds": self.crop_duration_seconds,
            "spectral_config": dict(self.spectral_config.__dict__),
            "search_window_seconds": None if self.search_window_seconds is None else list(self.search_window_seconds),
            "ifos": None if self.ifos is None else [i.name for i in self.ifos],
            "max_time_offset_seconds": self.max_time_offset_seconds,
            "snr_threshold": self.snr_threshold,
            "temperature": self.temperature,
            "batch_chunk": self.batch_chunk,
            "template_chunk": self.template_chunk,
            "bank_masses": np.stack([self._matched_filter.bank.mass_1, self._matched_filter.bank.mass_2]).tolist(),
        })
        return config

    @property
    def matched_filter(self) -> MatchedFilter:
        return self._matched_filter

    @property
    def num_templates(self) -> int:
        return self._matched_filter.num_templates
