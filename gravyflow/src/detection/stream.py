"""
The matched filter over continuous strain, scored per detector.

Each detector's strain is filtered once, on its own, into single-detector
triggers; a network trigger is a coincidence of those triggers. So a time
slide (one detector's triggers shifted against the other's) costs only the
coincidence step, never the filter, and a search reaches background many
times longer than its data.

**The filter** (`StreamingMatchedFilter.filter`). The strain is cut into
overlapping blocks. Each block is whitened by the data path's own `whiten`,
with the PSD of that block (median Welch) or a given one, and every template
is whitened with the same filter (`whitening_filter`), as the window-mode
`MatchedFilter` does. For each template the complex SNR series is

    z(t) = sum_n w(n) g_a(n - t),  rho(t) = |z(t)| / sigma,

with w the whitened block, g_a the analytic whitened template (merger at
index 0) and sigma^2 = sum g^2: in Gaussian noise rho^2 is chi-squared with 2
degrees of freedom. Templates are whole (the bank's span, from the band's
lower edge), so the full template norm normalises every merger time. A
merger time is searched only where the template lies inside the block's
uncorrupted part, and the blocks overlap by that margin, so every merger time
of the strain away from its ends is searched exactly once.

**Single-detector triggers.** Per template, a sample is a trigger when its
SNR is at least `snr_threshold` and the largest within +-`cluster_seconds`
(pycbc_inspiral's window clustering). At each trigger the signal-consistency
test is computed, and the SNR is reweighted by it.

**The chi-squared veto** (Allen 2005, Phys. Rev. D 71, 062001). The template's
whitened band is split into p contiguous frequency bins of equal template
power (sigma_j^2 = sigma^2 / p, up to the frequency grid), and z_j(t) is the
SNR's contribution from bin j, so z = sum_j z_j. Then

    chi^2 = sum_j |z_j - (sigma_j^2 / sigma^2) z|^2 / sigma_j^2,

which in Gaussian noise is chi-squared with 2p - 2 degrees of freedom (the
complex z_j / sigma_j projected orthogonally to the signal direction), and
for a signal that matches the template has the same distribution. A glitch
puts its power into a few bins and makes it large. The reduced chi-squared is
chi_r^2 = chi^2 / (2p - 2).

**The reweighted SNR** (PyCBC's ``newsnr``: Babak et al. 2013, Phys. Rev. D
87, 024033, eq. 6, with its exponents q = 6, n = 2, as pycbc.events.ranking
uses them):

    rho_hat = rho                                   if chi_r^2 <= 1,
    rho_hat = rho / ((1 + (chi_r^2)^3) / 2)^(1/6)   otherwise.

**Coincidence** (`coincide`). Triggers of the same template in every
detector, each within the light travel time between it and the first
detector plus `COINCIDENCE_MARGIN_SECONDS` of the first detector's trigger
(the loudest such in each detector), form a coincidence. Its network
statistic is the quadrature sum of the detectors' reweighted SNRs, its time
the first detector's merger time. Coincidences are clustered in time (the
loudest within +-cluster_seconds), and become `gf.Triggers`.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

from gravyflow.src.dataset.conditioning.whiten import whiten, whitening_filter
from gravyflow.src.dataset.tools.psd import PowerSpectrum, SpectralConfig, estimate_psd
from gravyflow.src.detection.matched_filter import COINCIDENCE_MARGIN_SECONDS
from gravyflow.src.detection.template_grid import BankPlacement, MatchSpace, TemplateBank
from gravyflow.src.detection.triggers import Triggers
from gravyflow.src.rng.platform import cbc_device

# ripplegw builds float64 tables when it is imported (the CBC module imports
# it), on JAX's default device: on an accelerator without float64 (an Apple
# GPU) the import fails. The templates are made on the host, so the module is
# imported there, whatever the default device; imported already, this is a
# no-op.
with jax.default_device(cbc_device()):
    from gravyflow.src.dataset.features.waveforms import cbc as _cbc  # noqa: F401

# The spectrum of each block: 16 s Welch segments overlapping by 8 s, and a
# 16 s whitening filter, as PyCBC's searches estimate theirs
# (--psd-segment-length 16 --psd-segment-stride 8 --psd-inverse-length 16).
STREAM_SPECTRAL_CONFIG = SpectralConfig(
    lower_frequency_hertz=20.0, fft_duration_seconds=16.0, overlap_duration_seconds=8.0,
    filter_duration_seconds=16.0)


def stream_bank_placement(
    mass_range: Tuple[float, float] = (5.0, 95.0),
    spin_range: float = 0.99,
    minimal_match: float = 0.97,
    sample_rate_hertz: float = 2048.0,
    lower_frequency_hertz: float = 20.0,
    visible_seconds: float = 24.0,
    noise_profile_name: str = "O4_HIGH",
) -> BankPlacement:
    """
    A placement of the streaming filter's bank: aligned-spin
    IMRPhenomD binaries with both component masses in `mass_range` and each
    aligned spin in [-spin_range, spin_range], at `minimal_match`, on L1's
    curve in `noise_profile_name` (aLIGO's O4 high projection, H1's and L1's
    alike) from `lower_frequency_hertz`. `visible_seconds` holds every
    template whole in the band (5 + 5 solar masses at spins 0.99 is in band
    for 20.1 s from 20 Hz). Each candidate is matched with its 32 nearest
    templates in the inspiral metric (see `MatchSpace.metric_coordinates`).
    """
    space = MatchSpace(float(sample_rate_hertz), float(visible_seconds), float(lower_frequency_hertz), "L1",
                       "IMRPhenomD", noise_profile_name)
    return BankPlacement(space, (float(mass_range[0]), float(mass_range[1])), float(minimal_match),
                         spin_range=float(spin_range), neighbours=32)


#: The fraction of its peak below which a template's ends are cut. Measured at
#: the default bank's corners (5 to 95 solar masses, spins +-0.99): the energy
#: cut is at most 5e-8 of the template's (its match with the uncut template
#: at least 0.9999999), and the inspiral at 20 Hz is at least 9% of the peak.
TEMPLATE_SUPPORT = 1e-4


def reweighted_snr(snr, reduced_chisq):
    """PyCBC's newsnr (see the module docstring): the SNR, lowered where the reduced chi-squared exceeds 1."""
    snr = np.asarray(snr, dtype=np.float64)
    reduced_chisq = np.asarray(reduced_chisq, dtype=np.float64)
    factor = ((1.0 + np.maximum(reduced_chisq, 1.0) ** 3) / 2.0) ** (1.0 / 6.0)
    return (snr / factor).astype(np.float32)


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DetectorTriggers:
    """
    One detector's triggers over a stretch of strain.

    Args:
        detector: the detector's name (a gf.IFO name).
        time: each trigger's merger time, GPS seconds (float64).
        template: its template's index in the bank.
        snr: its SNR.
        reduced_chisq: its chi-squared over its 2p - 2 degrees of freedom.
        reweighted_snr: its SNR reweighted by the chi-squared.
        span: (start, end), GPS seconds: the merger times searched.
    """
    detector: str
    time: np.ndarray
    template: np.ndarray
    snr: np.ndarray
    reduced_chisq: np.ndarray
    reweighted_snr: np.ndarray
    span: Tuple[float, float]

    def __post_init__(self):
        dtypes = {"time": np.float64, "template": np.int32, "snr": np.float32, "reduced_chisq": np.float32,
                  "reweighted_snr": np.float32}
        for name, dtype in dtypes.items():
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype).reshape(-1))
        if len({len(getattr(self, name)) for name in dtypes}) != 1:
            raise ValueError("DetectorTriggers' arrays must have one length.")
        object.__setattr__(self, "span", (float(self.span[0]), float(self.span[1])))

    def __len__(self) -> int:
        return len(self.time)

    @property
    def live_seconds(self) -> float:
        return self.span[1] - self.span[0]

    def shifted(self, seconds: float, wrap: bool = False) -> "DetectorTriggers":
        """
        The triggers moved by `seconds` (a time slide). With `wrap`, times are
        wrapped around the span, so the slid triggers cover the same span.
        """
        time = self.time + float(seconds)
        span = self.span
        if wrap:
            time = span[0] + np.mod(time - span[0], span[1] - span[0])
        else:
            span = (span[0] + seconds, span[1] + seconds)
        return DetectorTriggers(self.detector, time, self.template, self.snr, self.reduced_chisq,
                                self.reweighted_snr, span)

    @classmethod
    def concatenate(cls, parts: Sequence["DetectorTriggers"]) -> "DetectorTriggers":
        """Triggers of consecutive stretches of one detector's strain, as one."""
        if len({part.detector for part in parts}) != 1:
            raise ValueError("Only one detector's triggers can be concatenated.")
        joined = {name: np.concatenate([getattr(part, name) for part in parts])
                  for name in ("time", "template", "snr", "reduced_chisq", "reweighted_snr")}
        span = (min(part.span[0] for part in parts), max(part.span[1] for part in parts))
        return cls(parts[0].detector, span=span, **joined)


@dataclass(frozen=True, eq=False)
class Coincidences:
    """
    Network triggers: clustered coincidences of single-detector triggers.

    time: the first detector's merger time (GPS); stat: the network statistic
    (the quadrature sum of the reweighted SNRs); template: the template index;
    reweighted_snr: each detector's reweighted SNR, by detector name.
    """
    time: np.ndarray
    stat: np.ndarray
    template: np.ndarray
    reweighted_snr: Dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.time)

    def triggers(self, timing_tolerance_seconds: float) -> Triggers:
        """These coincidences as `gf.Triggers`, each with `timing_tolerance_seconds` as its var."""
        return Triggers(self.time, self.stat, np.full(len(self.time), float(timing_tolerance_seconds)))


def coincide(
    triggers: Mapping[str, DetectorTriggers],
    cluster_seconds: float = 1.0,
    margin_seconds: float = COINCIDENCE_MARGIN_SECONDS,
) -> Coincidences:
    """
    The coincidences of `triggers` (one DetectorTriggers per detector, the
    first the reference), clustered: see the module docstring. A time slide
    is a call with one detector's triggers `shifted`.
    """
    import gravyflow as gf
    names = list(triggers)
    if len(names) < 2:
        raise ValueError("A coincidence needs triggers from at least two detectors.")
    reference = triggers[names[0]]
    template, time = reference.template, reference.time
    rows = np.arange(len(reference))
    snrs = {names[0]: reference.reweighted_snr}
    for name in names[1:]:
        other = triggers[name]
        window = gf.Network([gf.IFO[names[0]], gf.IFO[name]]).max_arrival_time_difference_seconds + margin_seconds
        found, loudest = _loudest_within(template[rows], time[rows], other, window)
        rows = rows[found]
        snrs = {key: value[found] for key, value in snrs.items()}
        snrs[name] = loudest[found]
    stat = np.sqrt(sum(value.astype(np.float64) ** 2 for value in snrs.values())).astype(np.float32)
    keep = _cluster_in_time(time[rows], stat, cluster_seconds)
    return Coincidences(time[rows][keep], stat[keep], template[rows][keep],
                        {key: value[keep] for key, value in snrs.items()})


def _loudest_within(template, time, other: DetectorTriggers, window: float):
    """
    For each (template, time), whether `other` has a trigger of that template
    within `window` seconds, and the largest such reweighted SNR.
    """
    found = np.zeros(len(time), dtype=bool)
    loudest = np.zeros(len(time), dtype=np.float32)
    if not len(time) or not len(other):
        return found, loudest
    # One sorted key, template-major: each template's times sit in their own
    # stretch, further apart than any window.
    origin = min(time.min(), other.time.min()) - window
    stride = max(time.max(), other.time.max()) - origin + 2.0 * window + 1.0
    key = other.template.astype(np.float64) * stride + (other.time - origin)
    order = np.argsort(key, kind="stable")
    key, snr = key[order], other.reweighted_snr[order]
    centre = template.astype(np.float64) * stride + (time - origin)
    lo = np.searchsorted(key, centre - window, side="left")
    hi = np.searchsorted(key, centre + window, side="right")
    found = hi > lo
    loudest[found] = _range_max(snr, lo[found], hi[found])
    return found, loudest


def _range_max(values, lo, hi):
    """max of values[lo[k]:hi[k]] for each k (every range non-empty)."""
    if not len(lo):
        return np.zeros(0, dtype=values.dtype)
    padded = np.append(values, values.dtype.type(0))  # reduceat needs each end index to exist
    bounds = np.stack([lo, hi], axis=1).ravel()
    return np.maximum.reduceat(padded, bounds)[::2]


def _cluster_in_time(time, stat, cluster_seconds: float) -> np.ndarray:
    """
    The indices of the events that are the loudest within +-cluster_seconds
    of themselves (of equal statistics, the earliest), in time order.
    """
    if not len(time):
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(time, kind="stable")
    t, s = time[order], stat[order]
    # A strict order: louder first, then earlier.
    rank = np.empty(len(t), dtype=np.int64)
    rank[np.lexsort((-np.arange(len(t)), s))] = np.arange(len(t))
    lo = np.searchsorted(t, t - cluster_seconds, side="left")
    hi = np.searchsorted(t, t + cluster_seconds, side="right")
    return order[rank == _range_max(rank, lo, hi)]


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

@contextmanager
def float32_on(device):
    """
    Run JAX on `device` with 64-bit mode off, whatever the process's setting.

    The filter's correlation, peak finding and chi-squared are float32 and
    complex64 kernels. Importing some of ripplegw's models (IMRPhenomXP)
    switches 64-bit mode on for the whole process, which would make parts of
    them float64 (an index divided by a count, a cumulative sum), and an
    Apple GPU has no float64. So the kernels are traced and run with it off,
    whatever was imported before, and on the device the caller chose.
    """
    with jax.enable_x64(False), jax.default_device(device):
        yield


def _one_sided_weights(num_samples: int) -> np.ndarray:
    """The analytic signal's weights on the rfft grid: 1 at DC and an even length's Nyquist, 2 between."""
    weights = np.full(num_samples // 2 + 1, 2.0, dtype=np.float32)
    weights[0] = 1.0
    if num_samples % 2 == 0:
        weights[-1] = 1.0
    return weights


@partial(jax.jit, static_argnames=("n", "first", "windows", "width"))
def _snr_peaks(white_fft, response, templates, *, n, first, windows, width):
    """
    The window-clustered SNR peaks of each template over one block.

    white_fft (F,): rfft of the whitened block. response (F,): the whitening
    filter's response, aligned as the data path applies it, times the shift
    that puts a template's merger at index 0. templates (C, Tc, L): raw
    templates, merger at index L - post (folded into `response`).

    The searched span is `windows` windows of `width` samples from sample
    `first`; the first and last windows only serve the clustering of their
    neighbours. Returns, for windows 1 .. windows - 2 of each template, the
    peak's SNR and offset in its window, and whether it is the largest
    within +-width of itself: (C, Tc, windows - 2) each.
    """
    weights = jnp.asarray(_one_sided_weights(n))
    span = windows * width

    def one_chunk(chunk):
        g = jnp.fft.rfft(chunk, n=n, axis=-1) * response                       # (Tc, F)
        sigma = jnp.sqrt(jnp.sum(weights * jnp.abs(g) ** 2, axis=-1) / n)       # (Tc,)
        product = jnp.pad(weights * white_fft * jnp.conj(g), ((0, 0), (0, n - g.shape[-1])))
        z = jnp.fft.ifft(product, axis=-1)[:, first:first + span]               # (Tc, span)
        rho = jnp.abs(z) / jnp.where(sigma > 0, sigma, 1.0)[:, None]
        rho = jnp.where(sigma[:, None] > 0, rho, 0.0).reshape(chunk.shape[0], windows, width)
        offset = jnp.argmax(rho, axis=-1)                                      # (Tc, W)
        peak = jnp.max(rho, axis=-1)
        prefix = jax.lax.cummax(rho, axis=2)
        suffix = jax.lax.cummax(rho, axis=2, reverse=True)
        o = offset[:, 1:-1, None]
        left = jnp.take_along_axis(suffix[:, :-2], o, axis=-1)[..., 0]          # window j-1 from offset o on
        right = jnp.take_along_axis(prefix[:, 2:], o, axis=-1)[..., 0]          # window j+1 up to offset o
        middle = peak[:, 1:-1]
        return middle, offset[:, 1:-1].astype(jnp.int32), (middle >= left) & (middle > right)

    return jax.lax.map(one_chunk, templates)


@partial(jax.jit, static_argnames=("n", "bins", "band"))
def _chisq(white_fft, response, templates, phase, *, n, bins, band):
    """
    Allen's chi-squared of each (template, time) pair.

    templates (P, L) raw; phase (P, F): 2 pi k t / n at each pair's merger
    sample t, computed exactly by the caller; band: [kmin, kmax), the band's
    frequency bins (SpectralConfig.band_indices). Returns chi^2 (P,).
    """
    weights = jnp.asarray(_one_sided_weights(n))
    g = jnp.fft.rfft(templates, n=n, axis=-1) * response                       # (P, F)
    power = weights * jnp.abs(g) ** 2 / n                                       # sigma^2 per frequency
    frequency = jnp.arange(power.shape[-1])
    # Contiguous bins of equal template power over the band, as PyCBC places
    # them (pycbc.vetoes.power_chisq_bins): bin j is [k_j, k_(j+1)), k_j the
    # first frequency whose cumulative in-band power exceeds j / p of the
    # total. The template's sliver below the band (the whitening filter's
    # ramp) joins the first bin and the Nyquist frequency the last, so the
    # bins sum to the SNR. In noise one frequency carries about 1 / sqrt(N_j)
    # of a bin's sum, so the edge convention moves chi-squared by a few per
    # cent; this one is PyCBC's.
    in_band = (frequency >= band[0]) & (frequency < band[1])
    cumulative = jnp.cumsum(jnp.where(in_band, power, 0.0), axis=-1)
    targets = cumulative[:, -1:] * (jnp.arange(bins) / bins)
    starts = jax.vmap(lambda c, t: jnp.searchsorted(c, t, side="right"))(cumulative, targets)
    frequency_bin = jax.vmap(lambda s: jnp.searchsorted(s, frequency, side="right") - 1)(starts)
    frequency_bin = jnp.maximum(frequency_bin, 0)
    contribution = weights * white_fft * jnp.conj(g) * jnp.exp(1j * phase) / n

    def per_bin(values):
        return jnp.stack([jnp.sum(jnp.where(frequency_bin == j, values, 0.0), axis=-1) for j in range(bins)],
                         axis=-1)

    z_bins = per_bin(contribution)                                              # (P, p)
    sigma2_bins = per_bin(power)
    sigma2 = jnp.sum(sigma2_bins, axis=-1)
    z = jnp.sum(z_bins, axis=-1)
    share = sigma2_bins / jnp.where(sigma2 > 0, sigma2, 1.0)[:, None]
    residual = jnp.abs(z_bins - share * z[:, None]) ** 2 / jnp.where(sigma2_bins > 0, sigma2_bins, 1.0)
    return jnp.sum(jnp.where(sigma2_bins > 0, residual, 0.0), axis=-1)


@dataclass
class StreamingMatchedFilter:
    """
    The matched filter over continuous strain, per detector (see the module
    docstring).

    Args:
        bank: the template bank, for example `TemplateBank.place` of
            `stream_bank_placement`. Its templates are whole over its match
            space's span (`MatchSpace.templates`; a bank with no placement
            takes `stream_bank_placement`'s), which must have this filter's
            sample rate and lower band edge. There is no default yet: the
            aligned-spin bank over 5-95 solar masses and spins to 0.99 holds
            more than 20,000 templates, which this machine cannot filter
            over a month of strain.
        sample_rate_hertz: the strain's sample rate.
        spectral_config: the band, PSD estimate and whitening filter of each
            block (default STREAM_SPECTRAL_CONFIG).
        block_seconds: the length of the blocks the strain is filtered in.
        snr_threshold: the smallest SNR kept as a single-detector trigger.
        chisq_bins: p, the chi-squared's frequency bins.
        cluster_seconds: a trigger is the loudest within +-this of itself.
        template_chunk, pair_chunk: templates per filter step, and triggers
            per chi-squared step; memory grows with each times the block.
        device: the JAX device the float32 kernels (correlation, peak
            finding, chi-squared) run on, for example an accelerator; the
            host's CPU when None, whatever JAX's default device. Templates
            (ripplegw, float64), spectra and whitening filters are always
            computed on the host's CPU (`cbc_device`), since an accelerator may
            have no float64; the kernels run with 64-bit mode off
            (`float32_on`).
    """

    bank: TemplateBank
    sample_rate_hertz: float = 2048.0
    spectral_config: SpectralConfig = STREAM_SPECTRAL_CONFIG
    block_seconds: float = 256.0
    snr_threshold: float = 5.5
    chisq_bins: int = 16
    cluster_seconds: float = 1.0
    template_chunk: int = 16
    pair_chunk: int = 32
    device: Optional[object] = None

    _templates: Optional[jnp.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._host = cbc_device()
        self._device = self._host if self.device is None else self.device
        fs = self.sample_rate_hertz = float(self.sample_rate_hertz)
        config = self.spectral_config
        if config.lower_frequency_hertz is None:
            raise ValueError("The streaming filter needs a lower band edge (spectral_config.lower_frequency_hertz).")
        config.validate_band(fs)
        space = self.bank.space
        if space is None:
            space = stream_bank_placement(sample_rate_hertz=fs, lower_frequency_hertz=config.lower_frequency_hertz).space
        if space.sample_rate_hertz != fs or space.lower_frequency_hertz != config.lower_frequency_hertz:
            raise ValueError(
                f"The bank's templates are at {space.sample_rate_hertz} Hz from {space.lower_frequency_hertz} Hz; "
                f"this filter runs at {fs} Hz from {config.lower_frequency_hertz} Hz.")
        self.space = space
        if self.chisq_bins < 2:
            raise ValueError("chisq_bins must be at least 2.")
        self.chisq_dof = 2 * int(self.chisq_bins) - 2
        self.cluster_samples = int(round(self.cluster_seconds * fs))
        if self.cluster_samples < 1:
            raise ValueError("cluster_seconds must be at least one sample.")
        ntaps = config.num_taps(fs)
        self.num_taps = ntaps + ntaps % 2
        self.block_samples = int(round(self.block_seconds * fs))

    # ------------------------------------------------------------ templates

    def templates(self) -> jnp.ndarray:
        """
        The raw templates (T, pre + post), float32: each the part of its match
        space template from `pre` samples before its merger to `post` after,
        the merger at index `pre`, the span every template's support fits.
        Generated once.

        A template's support is where it exceeds TEMPLATE_SUPPORT of its peak:
        the generator leaves numerical residue far below that before its taper
        starts, which would otherwise lengthen every template to the whole span.
        """
        if self._templates is None:
            full = []
            for start in range(0, self.bank.num_templates, 256):
                rows = slice(start, start + 256)
                full.append(self.space.templates(*(values[rows] for values in self.bank.parameters)))
            full = np.concatenate(full)
            centre = self.space.num_samples // 2
            support = np.any(np.abs(full) > TEMPLATE_SUPPORT, axis=0)  # templates are unit peak
            first = int(np.argmax(support))
            last = full.shape[-1] - int(np.argmax(support[::-1]))
            self.pre, self.post = centre - first, max(last - centre, 1)
            with float32_on(self._device):
                self._templates = jax.device_put(np.ascontiguousarray(full[:, first:centre + self.post]),
                                                 self._device)
        return self._templates

    # --------------------------------------------------------------- blocks

    def _layout(self, num_samples: int):
        """
        (n, first, stride): the block length, the first searched merger sample
        of a block (after the corrupted half-filter, one template span and one
        clustering window) and the samples each block searches (whole windows).
        """
        self.templates()
        n = min(self.block_samples, num_samples)
        margin = self.num_taps // 2
        first = margin + self.pre + self.cluster_samples
        stride = (n - first - self.post - margin - self.cluster_samples) // self.cluster_samples * self.cluster_samples
        if stride < self.cluster_samples:
            raise ValueError(
                f"{num_samples / self.sample_rate_hertz:g} s of strain (blocks of {n / self.sample_rate_hertz:g} s) "
                "hold no searchable merger time: they need the whitening filter, the longest template and two "
                "clustering windows; give more strain or longer blocks.")
        return n, first, stride

    def _spectrum(self, block, psd) -> PowerSpectrum:
        """
        The block's spectrum: estimated from it, or the given one. A given
        spectrum must have this filter's sample rate, whitening filter length
        (the blocks' corrupted margins) and lower band edge (the templates'),
        or it is refused by name; the rest of its config (how it was
        estimated) is its own.
        """
        if psd is None:
            return estimate_psd(block[None], self.sample_rate_hertz, self.spectral_config)
        given = psd.config
        if (psd.sample_rate_hertz != self.sample_rate_hertz
                or given.num_taps(self.sample_rate_hertz) + given.num_taps(self.sample_rate_hertz) % 2 != self.num_taps
                or given.lower_frequency_hertz != self.spectral_config.lower_frequency_hertz):
            raise ValueError(
                "The given PowerSpectrum's sample rate, filter_duration_seconds or lower_frequency_hertz differs "
                "from the filter's spectral_config: build the filter with the spectrum's config.")
        return psd

    def _response(self, fir: np.ndarray, n: int) -> jnp.ndarray:
        """The whitening filter's response on an n-sample grid, aligned as `whiten` applies it, times the merger shift."""
        k = np.arange(n // 2 + 1)
        # whiten's 'same'-mode convolution keeps the full convolution from
        # (ntaps - 1) // 2 on; the templates' merger is at index pre.
        shift = (len(fir) - 1) // 2 + self.pre
        response = np.fft.rfft(fir, n=n) * np.exp(2j * np.pi * k * (shift % n) / n)
        return jax.device_put(response.astype(np.complex64), self._device)

    def filter(self, strain, start_gps: float, detector, psd: Optional[PowerSpectrum] = None) -> DetectorTriggers:
        """
        One detector's triggers over a continuous stretch of raw strain.

        strain: (samples,) at the filter's sample rate, finite; start_gps: its
        first sample's GPS time; detector: a gf.IFO or its name; psd: a
        PowerSpectrum to whiten every block with (this filter's config), or
        None to estimate each block's from the block itself.
        """
        name = detector if isinstance(detector, str) else detector.name
        strain = np.asarray(strain, dtype=np.float64)
        if strain.ndim != 1:
            raise ValueError("strain must be one detector's (samples,) series.")
        if not np.isfinite(strain).all():
            raise ValueError("The strain holds a NaN or an infinity: split it into finite stretches.")
        n, first, blocks = self._blocks(len(strain))
        chunked = self._chunked(self.templates())
        parts = []
        for start, lo, hi in blocks:
            found = self._search_block(strain[start:start + n], chunked, psd, first, lo - start, hi - start)
            parts.append((found[0], found[1] + start, found[2], found[3]))
        span = (start_gps + blocks[0][1] / self.sample_rate_hertz, start_gps + blocks[-1][2] / self.sample_rate_hertz)
        return self._triggers(name, start_gps, parts, span)

    def _triggers(self, name, start_gps, parts, span) -> DetectorTriggers:
        template, sample, snr, chisq = (np.concatenate(values) for values in zip(*parts))
        reduced = chisq / self.chisq_dof
        return DetectorTriggers(name, start_gps + sample / self.sample_rate_hertz, template, snr, reduced,
                                reweighted_snr(snr, reduced), span)

    def _blocks(self, num_samples: int):
        """
        (n, first, blocks): the block length, a block's first searchable merger
        sample, and each block's (start, lo, hi) in the strain's samples: it
        searches the merger samples [lo, hi), [start + first, start + first +
        stride) less what the block before it searched. The last block ends
        with the strain.
        """
        n, first, stride = self._layout(num_samples)
        starts = list(range(0, num_samples - n + 1, stride))
        if starts[-1] + first + stride < num_samples - (n - first - stride):
            starts.append(num_samples - n)
        blocks, until = [], None
        for start in starts:
            lo = start + first if until is None else max(start + first, until)
            until = start + first + stride
            blocks.append((start, lo, until))
        return n, first, blocks

    def _chunked(self, templates):
        with float32_on(self._device):
            pad = (-templates.shape[0]) % self.template_chunk
            padded = jnp.pad(templates, ((0, pad), (0, 0))) if pad else templates
            return padded.reshape((-1, self.template_chunk, templates.shape[-1]))

    def stretch_samples(self, lo: int, hi: int, first: int) -> Tuple[int, int]:
        """
        (a, length): the stretch of a block's whitened data the SNR at merger
        samples [lo, hi) depends on, clustering included. The clustering
        windows (cluster_seconds wide, on the block's grid from `first`) that
        hold [lo, hi), one more on each side, then the template span before
        (pre) and after (post) and half the whitening filter beyond each:
        length = windows * width + pre + post + num_taps.
        """
        width, half = self.cluster_samples, self.num_taps // 2
        j0, j1 = (lo - first) // width, -(-(hi - first) // width)
        windows = j1 - j0 + 2
        return first + (j0 - 1) * width - self.pre - half, windows * width + self.pre + self.post + 2 * half

    def _search_block(self, block, chunked, psd, first, lo, hi, length=None):
        """
        (template, sample in block, snr, chi^2) of one block's triggers with
        merger samples in [lo, hi). The block is whitened whole, with its own
        (or the given) spectrum, as the full pass whitens it; the SNR is then
        correlated over `stretch_samples` of it only (or its first `length`
        samples, to test that rule), and chi-squared, a statistic of the
        block, on the block's grid at the triggers.
        """
        n, fs, width = len(block), self.sample_rate_hertz, self.cluster_samples
        block = np.asarray(block, dtype=np.float32)
        with jax.default_device(self._host):  # the spectrum and its filter: the host's precision
            spectrum = self._spectrum(block, psd)
            white = np.asarray(whiten(block, spectrum, fs), dtype=np.float32).reshape(-1)
            fir = np.asarray(whitening_filter(spectrum, n), dtype=np.float64).reshape(-1)
        a, need = self.stretch_samples(lo, hi, first)
        windows = (need - self.pre - self.post - self.num_taps) // width
        length = need if length is None else min(length, need)
        size = int(next_fast_len(need))  # zero padding past the stretch wraps nothing into it
        with float32_on(self._device):
            stretch = jax.device_put(np.ascontiguousarray(white[a:a + length]), self._device)
            peak, offset, local = (np.asarray(v).reshape(-1, windows - 2) for v in _snr_peaks(
                jnp.fft.rfft(stretch, n=size), self._response(fir, size), chunked,
                n=size, first=self.pre + self.num_taps // 2, windows=windows, width=width))
        rows, window = np.nonzero(local & (peak >= self.snr_threshold))
        real = rows < self.bank.num_templates  # not the zero templates padding the last chunk
        rows, window = rows[real], window[real]
        sample = a + self.pre + self.num_taps // 2 + (window + 1) * width + offset[rows, window]
        keep = (sample >= lo) & (sample < hi)
        rows, sample, snr = rows[keep], sample[keep], peak[rows[keep], window[keep]]
        with float32_on(self._device):
            white_fft = jnp.fft.rfft(jax.device_put(white, self._device), n=n)
            chisq = self._chisq(white_fft, self._response(fir, n), rows, sample, n)
        return rows, sample, snr, chisq

    def _chisq(self, white_fft, response, rows, sample, n) -> np.ndarray:
        """Allen's chi-squared at each (template row, merger sample) of a block."""
        out = np.zeros(len(rows), dtype=np.float32)
        templates = self.templates()
        k = np.arange(n // 2 + 1, dtype=np.int64)
        for start in range(0, len(rows), self.pair_chunk):
            r = rows[start:start + self.pair_chunk]
            t = sample[start:start + self.pair_chunk].astype(np.int64)
            pad = self.pair_chunk - len(r)
            # 2 pi k t / n, exactly: the product reduced modulo n in integers.
            phase = (2.0 * np.pi / n) * np.mod(np.outer(t, k), n).astype(np.float64)
            phase = np.pad(phase.astype(np.float32), ((0, pad), (0, 0)))
            pair_templates = jnp.pad(templates[r], ((0, pad), (0, 0)))
            chisq = _chisq(white_fft, response, pair_templates, jax.device_put(phase, self._device), n=n,
                           bins=self.chisq_bins,
                           band=self.spectral_config.band_indices(n, self.sample_rate_hertz))
            out[start:start + len(r)] = np.asarray(chisq)[:len(r)]
        return out

    # -------------------------------------------------------------- network

    def search(self, strain: Mapping, start_gps: float, psds: Optional[Mapping] = None,
               coincidence_cluster_seconds: Optional[float] = None) -> "StreamResult":
        """
        Filter each detector's strain (a mapping from gf.IFO or name to raw
        strain, all starting at `start_gps`; the first is the coincidence's
        reference) and form the network's coincidences at zero lag.
        """
        detectors = {}
        for detector, series in strain.items():
            name = detector if isinstance(detector, str) else detector.name
            psd = None if psds is None else psds.get(detector, psds.get(name))
            detectors[name] = self.filter(series, start_gps, name, psd)
        cluster = self.cluster_seconds if coincidence_cluster_seconds is None else coincidence_cluster_seconds
        return StreamResult(detectors, coincide(detectors, cluster_seconds=cluster))

    # ----------------------------------------------------------- injections

    def score_injections(self, strain: Mapping, start_gps: float, injections: Sequence["Injection"],
                         window_seconds: float = 0.1, psds: Optional[Mapping] = None,
                         coincidence_cluster_seconds: Optional[float] = None) -> list:
        """
        Each injection's triggers, scored locally: what `search` over the
        strain with that injection added would give within `window_seconds`
        of its time, without filtering the strain again.

        strain: the clean data, a mapping from gf.IFO or name to raw strain
        (an array or a memory map; all start at `start_gps`), the data the
        full pass filtered. For each injection and detector, each block of
        the full pass whose searched merger times meet [time - window, time +
        window] is read, the injection added, and the block whitened whole
        (its spectrum, from the block, is the full pass's); the templates are
        then correlated over `stretch_samples` of it only, chi-squared is
        taken on the block's grid at those triggers, and the window's
        coincidences are formed. Returns one StreamResult per injection; each
        detector's triggers span the window.
        """
        fs = self.sample_rate_hertz
        chunked = self._chunked(self.templates())
        half = int(round(window_seconds * fs))
        cluster = self.cluster_seconds if coincidence_cluster_seconds is None else coincidence_cluster_seconds
        series = {(d if isinstance(d, str) else d.name): s for d, s in strain.items()}
        layouts = {name: self._blocks(len(s)) for name, s in series.items()}
        results = []
        for injection in injections:
            centre = int(round((injection.time - start_gps) * fs))
            detectors = {}
            for name, data in series.items():
                psd = None if psds is None else psds.get(name)
                n, first, blocks = layouts[name]
                parts = []
                for start, lo, hi in blocks:
                    lo, hi = max(lo, centre - half), min(hi, centre + half + 1)
                    if lo >= hi:
                        continue
                    block = injection.add_to(np.array(data[start:start + n], dtype=np.float64), name,
                                             start_gps + start / fs, fs)
                    found = self._search_block(block, chunked, psd, first, lo - start, hi - start)
                    parts.append((found[0], found[1] + start, found[2], found[3]))
                if not parts:
                    raise ValueError(f"The injection at {injection.time} is outside {name}'s searched span.")
                span = (start_gps + (centre - half) / fs, start_gps + (centre + half + 1) / fs)
                detectors[name] = self._triggers(name, start_gps, parts, span)
            network = coincide(detectors, cluster_seconds=cluster) if len(detectors) > 1 else None
            results.append(StreamResult(detectors, network))
        return results


@dataclass(frozen=True, eq=False)
class Injection:
    """
    A signal to score: its time (GPS; the triggers within the scoring window
    of it are its), and each detector's projected strain, `waveform[name]`,
    whose first sample is at GPS `start[name]` on the strain's sample grid.
    """
    time: float
    start: Mapping[str, float]
    waveform: Mapping[str, np.ndarray]

    def add_to(self, series: np.ndarray, detector: str, series_start_gps: float, sample_rate_hertz: float):
        """`series` (starting at `series_start_gps`) with this injection's strain in `detector` added where they overlap."""
        if detector not in self.waveform:
            return series
        offset = (self.start[detector] - series_start_gps) * sample_rate_hertz
        first = int(round(offset))
        if abs(offset - first) > 1e-3:
            raise ValueError("The injection's waveform is not on the strain's sample grid.")
        wave = np.asarray(self.waveform[detector], dtype=np.float64)
        lo, hi = max(first, 0), min(first + len(wave), len(series))
        if lo < hi:
            series[lo:hi] += wave[lo - first:hi - first]
        return series


@dataclass(frozen=True, eq=False)
class StreamResult:
    """A streaming search's per-detector triggers and its zero-lag coincidences."""
    detectors: Dict[str, DetectorTriggers]
    coincidences: Coincidences

    def triggers(self, timing_tolerance_seconds: float) -> Triggers:
        return self.coincidences.triggers(timing_tolerance_seconds)


__all__ = [
    "Coincidences",
    "DetectorTriggers",
    "Injection",
    "STREAM_SPECTRAL_CONFIG",
    "StreamResult",
    "StreamingMatchedFilter",
    "coincide",
    "reweighted_snr",
    "stream_bank_placement",
]
