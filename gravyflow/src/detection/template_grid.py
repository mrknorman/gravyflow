"""
Template banks for matched filtering.

`TemplateBank.stochastic` places CBC templates, non-spinning or with spins
aligned with the orbital angular momentum, so that every point of the mass
and spin range has a template whose match with it is at least the minimal
match, measured with a named noise curve on the analysis band: the bank's
coverage is a measured property, not an assumption. `TemplateGrid` is a
plain grid in component masses, with no coverage guarantee; a linear grid
leaves light binaries badly covered (a fitting factor near 0.5 below 30 solar
masses for a 16 x 16 grid over 5-95), so it is kept only as a simple utility.
"""

import hashlib
import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from gravyflow.src.dataset.conditioning.detector import DEFAULT_NOISE_PROFILE

#: G M_sun / c^3: a solar mass in seconds.
SOLAR_MASS_SECONDS = 4.925490947e-6

# A placement's progress (templates kept, candidates offered, seconds), at
# INFO every 64 rounds: a long placement is sized from it while it runs.
_log = logging.getLogger(__name__)


@dataclass
class TemplateGrid:
    """
    A uniform grid in component masses, m1 >= m2. It is not a template bank
    with a coverage guarantee: use `TemplateBank.stochastic` for searches.

    Args:
        mass_1_range: (min, max) for primary mass in solar masses
        mass_2_range: (min, max) for secondary mass in solar masses
        num_mass_1_points: Grid resolution for mass_1
        num_mass_2_points: Grid resolution for mass_2 (defaults to mass_1)
    """

    mass_1_range: Tuple[float, float] = (5.0, 75.0)
    mass_2_range: Tuple[float, float] = (5.0, 75.0)
    num_mass_1_points: int = 32
    num_mass_2_points: int = None  # Defaults to num_mass_1_points

    def __post_init__(self):
        if self.num_mass_2_points is None:
            self.num_mass_2_points = self.num_mass_1_points

    def get_parameters(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(mass_1, mass_2) arrays of the grid points with m1 >= m2, each (num_templates,)."""
        m1_vals = jnp.linspace(self.mass_1_range[0], self.mass_1_range[1], self.num_mass_1_points)
        m2_vals = jnp.linspace(self.mass_2_range[0], self.mass_2_range[1], self.num_mass_2_points)
        m1_grid, m2_grid = jnp.meshgrid(m1_vals, m2_vals, indexing='ij')
        m1_flat = m1_grid.ravel()
        m2_flat = m2_grid.ravel()
        mask = m1_flat >= m2_flat
        return m1_flat[mask], m2_flat[mask]

    @property
    def num_templates(self) -> int:
        """Total number of templates in grid."""
        m1, _ = self.get_parameters()
        return len(m1)

    def get_chirp_masses(self) -> jnp.ndarray:
        """Chirp mass (m1 m2)^(3/5) / (m1 + m2)^(1/5) of each grid point."""
        m1, m2 = self.get_parameters()
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


# ---------------------------------------------------------------------------
# The match
# ---------------------------------------------------------------------------

def noise_curve_psd(ifo, frequencies: np.ndarray, profile) -> np.ndarray:
    """
    The one-sided PSD (strain^2/Hz) of the IFO's curve in `profile` (a
    gf.NoiseProfile) on `frequencies`, log-log interpolated. Frequencies beyond
    the curve's band are refused: a curve is never extrapolated.
    """
    from gravyflow.src.dataset.conditioning.detector import read_noise_curve
    f = np.asarray(frequencies, dtype=np.float64)
    curve_frequencies, asd = read_noise_curve(profile.path(ifo), from_hertz=f.min(), up_to_hertz=f.max())
    return np.exp(2.0 * np.interp(np.log(f), np.log(curve_frequencies), np.log(asd)))


def waveform_model(approximant_name: str) -> dict:
    """
    The waveform model `cbc_templates` computes: the approximant and the
    release series (major.minor) of ripplegw, which computes it. A release
    series is the unit the dependency is pinned to; a change within one is
    caught by the bank's coverage tests and by a fresh placement
    (`bundle_banks --check`), not by the bank's key.
    """
    from importlib.metadata import version
    major, minor = version("ripplegw").split(".")[:2]
    return {"approximant": approximant_name, "package": "ripplegw", "series": f"{major}.{minor}"}


def cbc_templates(
    mass_1,
    mass_2,
    sample_rate_hertz: float,
    window_seconds: float,
    lower_frequency_hertz: float,
    approximant=None,
    spin_1z=None,
    spin_2z=None,
) -> np.ndarray:
    """
    Plus polarisations of face-on binaries at reference phase 0, non-spinning
    or with the aligned spins `spin_1z`, `spin_2z` (dimensionless, along the
    orbital angular momentum), from the CBC generator the dataset injects with
    (`_generate_cbc_window`), with the merger at the centre of a
    `window_seconds` buffer, whole from `lower_frequency_hertz` and tapered
    below it. Unit-peak rows, float32, shape (n, samples).

    The waveforms are computed on the host's CPU (`cbc_device`), in float64,
    whatever JAX's default device: an accelerator may have no float64 (Apple
    GPUs have none), as a dataset's CBC stage runs on the host.
    """
    from gravyflow.src.dataset.features.waveforms.cbc import Approximant, _generate_cbc_window
    from gravyflow.src.rng.platform import cbc_device
    approximant = Approximant.IMRPhenomD if approximant is None else approximant
    mass_1 = np.atleast_1d(np.asarray(mass_1, dtype=np.float64))
    mass_2 = np.atleast_1d(np.asarray(mass_2, dtype=np.float64))
    n = len(mass_1)
    spins = {}
    for name, value in (("spin_1_in", spin_1z), ("spin_2_in", spin_2z)):
        spins[name] = np.zeros((n, 3))
        if value is not None:
            spins[name][:, 2] = value
    with jax.default_device(cbc_device()):
        waveforms = _generate_cbc_window(
            num_waveforms=n,
            sample_rate_hertz=float(sample_rate_hertz),
            window_seconds=float(window_seconds),
            min_frequency_hertz=np.full(n, float(lower_frequency_hertz)),
            approximant=approximant,
            f_ref=float(lower_frequency_hertz),
            mass_1_msun=mass_1,
            mass_2_msun=mass_2,
            distance_mpc=np.full(n, 100.0),
            reference_orbital_phase_in=np.zeros(n),
            inclination_radians=np.zeros(n),
            **spins,
        )
        plus = np.asarray(waveforms[:, 0, :], dtype=np.float64)
    peak = np.max(np.abs(plus), axis=-1, keepdims=True)
    return (plus / np.where(peak > 0, peak, 1.0)).astype(np.float32)


@dataclass(frozen=True)
class MatchSpace:
    """
    The inner product the bank is placed with: the detector's curve in the
    noise profile `noise_profile_name` on the band [lower_frequency_hertz,
    Nyquist), on the visible part of each template, the `visible_seconds`
    before its merger (with a Hann ramp over the first tenth) and everything
    after it.

    The window-mode matched filter sees at most the onsource window plus half
    the whitening filter of a signal's inspiral, so a bank placed on that span
    is at least as dense as any analysis with a shorter window needs. The
    streaming filter sees whole templates: its bank's span holds every
    template of its range whole in the band, and its templates are this
    space's (`templates`).

    The profile is the default noise's (DEFAULT_NOISE_PROFILE, each detector's
    measured O3 curve) unless named otherwise. A bank is the search's, one set
    of templates for every detector of the network, placed on one detector's
    curve, which its key names. The search itself whitens with the data's own
    spectrum.
    """
    sample_rate_hertz: float = 2048.0
    visible_seconds: float = 1.5
    lower_frequency_hertz: float = 20.0
    ifo_name: str = "L1"
    approximant_name: str = "IMRPhenomD"
    noise_profile_name: str = DEFAULT_NOISE_PROFILE.name

    @property
    def window_seconds(self) -> float:
        return 2.0 * self.visible_seconds

    @property
    def noise_curve(self) -> Path:
        """The file of the detector's curve in the space's noise profile."""
        import gravyflow as gf
        return gf.NoiseProfile[self.noise_profile_name].path(gf.IFO[self.ifo_name])

    def identity(self) -> dict:
        """
        Everything the match depends on: these settings, the noise curve (its
        file and a hash of its contents) and the waveform model (the
        approximant, and the release series of the package that computes it).
        """
        curve = self.noise_curve
        return {
            **asdict(self),
            "noise_curve": {"file": curve.name, "sha256": hashlib.sha256(curve.read_bytes()).hexdigest()},
            "waveform_model": waveform_model(self.approximant_name),
        }

    @property
    def num_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate_hertz))

    def _ramp(self) -> np.ndarray:
        n = self.num_samples
        start = n // 2 - int(round(self.visible_seconds * self.sample_rate_hertz))
        width = max(int(round(0.1 * self.visible_seconds * self.sample_rate_hertz)), 1)
        ramp = np.zeros(n, dtype=np.float32)
        ramp[start:start + width] = 0.5 - 0.5 * np.cos(np.pi * np.arange(width) / width)
        ramp[start + width:] = 1.0
        return ramp

    def templates(self, mass_1, mass_2, spin_1z=None, spin_2z=None) -> np.ndarray:
        """
        The visible part of each template (n, num_samples), float32: the
        `visible_seconds` before its merger (the merger at the centre), Hann-ramped
        over their first tenth, and everything after.
        """
        from gravyflow.src.dataset.features.waveforms.cbc import Approximant
        h = cbc_templates(mass_1, mass_2, self.sample_rate_hertz, self.window_seconds,
                          self.lower_frequency_hertz, Approximant[self.approximant_name], spin_1z, spin_2z)
        return h * self._ramp()

    def _band_weight(self, num_samples: int) -> np.ndarray:
        """1 / ASD of the space's curve on the rfft grid of num_samples, zero outside [f_low, Nyquist)."""
        import gravyflow as gf
        freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate_hertz)
        weight = np.zeros_like(freqs)
        band = (freqs >= self.lower_frequency_hertz) & (freqs < self.sample_rate_hertz / 2.0)
        weight[band] = 1.0 / np.sqrt(noise_curve_psd(gf.IFO[self.ifo_name], freqs[band],
                                                     gf.NoiseProfile[self.noise_profile_name]))
        return weight

    def whitened_spectra(self, mass_1, mass_2, spin_1z=None, spin_2z=None) -> np.ndarray:
        """Unit-norm noise-weighted spectra (n, F) complex64, zero outside the band."""
        h = self.templates(mass_1, mass_2, spin_1z, spin_2z)
        spectra = np.fft.rfft(h.astype(np.float64), axis=-1) * self._band_weight(self.num_samples)
        norm = np.sqrt(np.sum(np.abs(spectra) ** 2, axis=-1, keepdims=True))
        return (spectra / np.where(norm > 0, norm, 1.0)).astype(np.complex64)

    def metric_coordinates(self, mass_1, mass_2, spin_1z=None, spin_2z=None) -> np.ndarray:
        """
        Coordinates (n, 3) in which the inspiral's mismatch (1 - match) is about
        the squared Euclidean distance: TaylorF2's Newtonian, 1PN and 1.5PN phase
        coefficients (the last with the spin-orbit term), transformed by the
        Cholesky factor of their Fisher metric on this space's curve, with the
        coalescence time and phase projected out. The phase is linear in these
        coefficients, so the metric is constant in them.

        Placement uses them only to choose which templates to compute the match
        with (the nearest few); the match itself is always computed. Near
        merger the inspiral metric is a poor guide, so a shortlist can only
        miss a template that would have covered a candidate: the candidate is
        then kept, and the bank is denser, never thinner.
        """
        m1 = np.atleast_1d(np.asarray(mass_1, dtype=np.float64))
        m2 = np.atleast_1d(np.asarray(mass_2, dtype=np.float64))
        s1 = np.zeros_like(m1) if spin_1z is None else np.asarray(spin_1z, dtype=np.float64)
        s2 = np.zeros_like(m1) if spin_2z is None else np.asarray(spin_2z, dtype=np.float64)
        f0 = self.lower_frequency_hertz
        total = m1 + m2
        eta = m1 * m2 / total ** 2
        v = (np.pi * total * SOLAR_MASS_SECONDS * f0) ** (1.0 / 3.0)
        beta = (113.0 * ((m1 / total) ** 2 * s1 + (m2 / total) ** 2 * s2) + 75.0 * eta * (s1 + s2)) / 12.0
        psi0 = 3.0 / (128.0 * eta * v ** 5)
        psi = np.stack([psi0, psi0 * (3715.0 / 756.0 + 55.0 * eta / 9.0) * v ** 2,
                        psi0 * (4.0 * beta - 16.0 * np.pi) * v ** 3], axis=-1)
        return psi @ self._metric_factor()

    @lru_cache(maxsize=None)
    def _metric_factor(self) -> np.ndarray:
        """L with L L^T the Fisher metric of the phase coefficients at f0 = f_low (see metric_coordinates)."""
        num_samples = 1 << 16
        freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate_hertz)
        weight = self._band_weight(num_samples) ** 2 * np.where(freqs > 0, freqs, 1.0) ** (-7.0 / 3.0)
        weight = weight / weight.sum()
        x = freqs / self.lower_frequency_hertz
        basis = np.stack([np.where(x > 0, x, 1.0) ** p for p in (-5.0 / 3.0, -1.0, -2.0 / 3.0)])
        nuisance = np.stack([np.ones_like(x), x])  # the coalescence phase and time
        inner = lambda a, b: (a * weight) @ b.T
        projection = inner(basis, nuisance) @ np.linalg.solve(inner(nuisance, nuisance), inner(nuisance, basis))
        return np.linalg.cholesky(0.5 * (inner(basis, basis) - projection))


def _peak_overlap(product):
    """max over time shifts of |<a|b>| from the products a * conj(b) of unit-norm one-sided spectra (..., F)."""
    num_freqs = product.shape[-1]
    n = 2 * (num_freqs - 1)
    full = jnp.zeros(product.shape[:-1] + (n,), dtype=product.dtype).at[..., :num_freqs].set(product)
    return jnp.max(jnp.abs(jnp.fft.ifft(full, axis=-1)), axis=-1) * n


@jax.jit
def _max_overlap(a, b):
    """The overlap of every row of a (P, F) with every row of b (Q, F): (P, Q)."""
    return _peak_overlap(a[:, None, :] * jnp.conj(b)[None, :, :])


@jax.jit
def _pair_overlap(a, b):
    """The overlap of each row of a (P, F) with the same row of b (P, F): (P,)."""
    return _peak_overlap(a * jnp.conj(b))


def max_matches(a: np.ndarray, b: np.ndarray, chunk: int = 64) -> np.ndarray:
    """The match (phase and time maximised) of every row of `a` with every row of `b`: (P, Q)."""
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    for i in range(0, a.shape[0], chunk):
        for j in range(0, b.shape[0], chunk):
            ai, bj = a[i:i + chunk], b[j:j + chunk]
            pad_a, pad_b = chunk - len(ai), chunk - len(bj)
            ai = np.pad(ai, ((0, pad_a), (0, 0)))
            bj = np.pad(bj, ((0, pad_b), (0, 0)))
            block = np.asarray(_max_overlap(jnp.asarray(ai), jnp.asarray(bj)))
            out[i:i + chunk, j:j + chunk] = block[:chunk - pad_a, :chunk - pad_b]
    return out


def pair_matches(a: np.ndarray, b: np.ndarray, rows_a, rows_b, chunk: int = 256) -> np.ndarray:
    """The match of each pair (a[rows_a[k]], b[rows_b[k]]): (K,)."""
    rows_a, rows_b = np.asarray(rows_a, dtype=np.int64), np.asarray(rows_b, dtype=np.int64)
    out = np.zeros(len(rows_a), dtype=np.float32)
    for k in range(0, len(rows_a), chunk):
        ra, rb = rows_a[k:k + chunk], rows_b[k:k + chunk]
        pad = chunk - len(ra)
        ai = np.pad(a[ra], ((0, pad), (0, 0)))
        bi = np.pad(b[rb], ((0, pad), (0, 0)))
        out[k:k + len(ra)] = np.asarray(_pair_overlap(jnp.asarray(ai), jnp.asarray(bi)))[:len(ra)]
    return out


class _RowStore:
    """
    Rows of one width, appended in blocks and read by index like an array.

    A placement's template spectra (each row 393 kB at the streaming bank's
    span) grow to gigabytes: appended to one array, every round would copy the
    whole bank and hold it twice. Here a block is filled in place and a full
    one is never copied again.
    """

    def __init__(self, width: int, block_rows: int = 512, dtype=np.complex64):
        self.width, self.block_rows, self.dtype = int(width), int(block_rows), dtype
        self.blocks, self.size = [], 0

    def __len__(self) -> int:
        return self.size

    def extend(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=self.dtype)
        done = 0
        while done < len(rows):
            if self.size == len(self.blocks) * self.block_rows:
                self.blocks.append(np.empty((self.block_rows, self.width), dtype=self.dtype))
            block, at = divmod(self.size, self.block_rows)
            count = min(self.block_rows - at, len(rows) - done)
            self.blocks[block][at:at + count] = rows[done:done + count]
            done += count
            self.size += count

    def __getitem__(self, index) -> np.ndarray:
        """The rows at `index` (an integer array), or every row (a slice `[:]`)."""
        if isinstance(index, slice):
            index = np.arange(self.size)[index]
        index = np.asarray(index, dtype=np.int64)
        if index.size and (index.min() < 0 or index.max() >= self.size):
            raise IndexError("row index out of range")
        out = np.empty(index.shape + (self.width,), dtype=self.dtype)
        block, at = np.divmod(index, self.block_rows)
        for b in np.unique(block):
            where = block == b
            out[where] = self.blocks[b][at[where]]
        return out


def nearest(coordinates: np.ndarray, to: np.ndarray, count: int) -> np.ndarray:
    """The indices (len(to), min(count, len(coordinates))) of the `count` rows of `coordinates` nearest each row of `to`."""
    count = min(int(count), len(coordinates))
    distance = np.sum((to[:, None, :] - coordinates[None, :, :]) ** 2, axis=-1)
    return np.argpartition(distance, count - 1, axis=1)[:, :count]


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------

# The gravyflow source a placement runs: the CBC generator its templates come
# from, and this module (the match space, the overlap, the proposal and the
# acceptance rule).
PLACEMENT_SOURCES = (
    Path(__file__).resolve().parents[1] / "dataset" / "features" / "waveforms" / "cbc.py",
    Path(__file__).resolve(),
)


def placement_code() -> dict:
    """
    The version of the gravyflow code a placement runs: the SHA-256 of the
    contents of PLACEMENT_SOURCES, with line endings normalised so that a
    checkout's conversion does not change it. Any edit to those files changes
    it, so a bank placed by other code is never reused for this code.
    """
    digest = hashlib.sha256()
    for path in PLACEMENT_SOURCES:
        digest.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return {"files": [path.name for path in PLACEMENT_SOURCES], "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class BankPlacement:
    """
    The arguments of a stochastic placement (see `TemplateBank.stochastic`):
    the match space, the component-mass range, the minimal match, the proposal
    and stopping rule, `spin_range` (each component's aligned spin is placed
    over [-spin_range, spin_range]; 0 for a non-spinning bank) and
    `neighbours` (each candidate's match is computed with that many nearest
    templates in `MatchSpace.metric_coordinates`, or with every template when
    None). Its `key` names what the placed bank depends on: these settings, the match
    space with its noise curve and waveform model, and the version of the
    gravyflow code that places it (`placement_code`). A stored bank is reused
    only for exactly the placement it came from. What the key cannot name (a
    dependency's change within its release series) is caught by placing the
    bundled banks again (`bundle_banks --check`).
    """
    space: MatchSpace
    mass_range: Tuple[float, float]
    minimal_match: float
    seed: int = 2026
    candidates_per_round: int = 128
    target_coverage: float = 0.99
    convergence_candidates: int = 2048
    spin_range: float = 0.0
    neighbours: Optional[int] = None

    def __post_init__(self):
        if not 0.0 < self.minimal_match < 1.0:
            raise ValueError("minimal_match must be in (0, 1).")
        lo, hi = self.mass_range
        if not 0.0 < lo <= hi:
            raise ValueError("mass_range must be (min, max) with 0 < min <= max.")
        if not 0.0 <= self.spin_range < 1.0:
            raise ValueError("spin_range (the largest aligned spin magnitude) must be in [0, 1).")
        if self.neighbours is not None and self.neighbours < 1:
            raise ValueError("neighbours must be None (every template) or at least 1.")

    def key(self) -> dict:
        return {
            "space": self.space.identity(),
            "mass_range": [float(m) for m in self.mass_range],
            "minimal_match": float(self.minimal_match),
            "seed": int(self.seed),
            "candidates_per_round": int(self.candidates_per_round),
            "target_coverage": float(self.target_coverage),
            "convergence_candidates": int(self.convergence_candidates),
            "spin_range": float(self.spin_range),
            "neighbours": None if self.neighbours is None else int(self.neighbours),
            "code": placement_code(),
        }

    def proposal(self, rng: np.random.Generator, count: int):
        """`count` draws (m1, m2, s1, s2) from the interior's proposal: log-uniform masses, uniform aligned spins."""
        return _draw_interior(rng, count, *self.mass_range, self.spin_range)

    @classmethod
    def from_key(cls, key: dict) -> "BankPlacement":
        fields = {name: key["space"][name] for name in MatchSpace.__dataclass_fields__}
        return cls(MatchSpace(**fields), tuple(key["mass_range"]), key["minimal_match"], key["seed"],
                   key["candidates_per_round"], key["target_coverage"], key["convergence_candidates"],
                   key["spin_range"], key["neighbours"])


@dataclass(frozen=True)
class TemplateBank:
    """
    Component masses (m1 >= m2) and aligned spins of a CBC template bank, with
    the placement that made it. Build one with `TemplateBank.stochastic`, or
    wrap fixed parameters with `TemplateBank.from_masses` (no coverage
    guarantee). Spins default to zero: a non-spinning bank.
    """
    mass_1: np.ndarray
    mass_2: np.ndarray
    spin_1z: Optional[np.ndarray] = None
    spin_2z: Optional[np.ndarray] = None
    placement: Optional[BankPlacement] = None
    candidates_tried: int = 0
    final_acceptance: float = float("nan")

    def __post_init__(self):
        for name in ("spin_1z", "spin_2z"):
            value = getattr(self, name)
            value = np.zeros(len(self.mass_1)) if value is None else np.asarray(value, dtype=np.float64)
            if value.shape != np.shape(self.mass_1):
                raise ValueError(f"{name} must have one value per template.")
            object.__setattr__(self, name, value)

    @property
    def num_templates(self) -> int:
        return int(len(self.mass_1))

    @property
    def parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(mass_1, mass_2, spin_1z, spin_2z), each (num_templates,)."""
        return self.mass_1, self.mass_2, self.spin_1z, self.spin_2z

    @property
    def minimal_match(self) -> Optional[float]:
        return None if self.placement is None else self.placement.minimal_match

    @property
    def space(self) -> Optional[MatchSpace]:
        return None if self.placement is None else self.placement.space

    @classmethod
    def from_masses(cls, mass_1, mass_2, spin_1z=None, spin_2z=None) -> "TemplateBank":
        """A bank of the given parameters, each component's spin kept with its mass when m1 < m2 is swapped."""
        m1 = np.atleast_1d(np.asarray(mass_1, dtype=np.float64))
        m2 = np.atleast_1d(np.asarray(mass_2, dtype=np.float64))
        if m1.shape != m2.shape or m1.ndim != 1 or len(m1) == 0:
            raise ValueError("mass_1 and mass_2 must be equal-length, non-empty 1D arrays.")
        s1 = np.zeros_like(m1) if spin_1z is None else np.broadcast_to(np.asarray(spin_1z, np.float64), m1.shape)
        s2 = np.zeros_like(m1) if spin_2z is None else np.broadcast_to(np.asarray(spin_2z, np.float64), m1.shape)
        swap = m1 < m2
        return cls(np.where(swap, m2, m1), np.where(swap, m1, m2), np.where(swap, s2, s1), np.where(swap, s1, s2))

    @classmethod
    def from_grid(cls, grid: TemplateGrid) -> "TemplateBank":
        m1, m2 = grid.get_parameters()
        return cls.from_masses(np.asarray(m1), np.asarray(m2))

    @classmethod
    def stochastic(
        cls,
        mass_range: Tuple[float, float] = (5.0, 95.0),
        minimal_match: float = 0.97,
        sample_rate_hertz: float = 2048.0,
        visible_seconds: float = 1.5,
        lower_frequency_hertz: float = 20.0,
        ifo=None,
        seed: int = 2026,
        candidates_per_round: int = 128,
        target_coverage: float = 0.99,
        convergence_candidates: int = 2048,
        max_templates: int = 4096,
        use_bundled: bool = True,
        spin_range: float = 0.0,
        neighbours: Optional[int] = None,
    ) -> "TemplateBank":
        """
        Stochastic placement (Harry, Allen & Sathyaprakash 2009; Babak 2008):
        candidates are kept when their match with every template already kept
        is below `minimal_match`. The boundary of `mass_range` (one component
        at a bound) is placed first, with candidates in order along it, so
        every point of it is covered; with spins, along it at each corner of
        the spin range, and then the faces of the range (one mass or spin at a
        bound) stochastically. Then candidates drawn log-uniform in each
        component mass, and uniform in each aligned spin over
        [-spin_range, spin_range], fill the interior, until, over the last
        `convergence_candidates` of them, fewer than 1 - `target_coverage` were
        kept: the fraction of the space (in the proposal's measure) left with a
        fitting factor below the minimal match is then about that small.

        The match is `MatchSpace`'s: the IFO's (default L1) curve in the
        default noise profile over [lower_frequency_hertz, Nyquist), on the
        `visible_seconds` before merger. With `neighbours`, each candidate is
        matched with that many nearest templates only (see BankPlacement).
        See `place` for caching and the banks bundled with gravyflow.
        """
        import gravyflow as gf
        ifo = gf.IFO.L1 if ifo is None else ifo
        space = MatchSpace(float(sample_rate_hertz), float(visible_seconds), float(lower_frequency_hertz),
                           ifo.name, "IMRPhenomD")
        placement = BankPlacement(space, (float(mass_range[0]), float(mass_range[1])), float(minimal_match),
                                  int(seed), int(candidates_per_round), float(target_coverage),
                                  int(convergence_candidates), float(spin_range), neighbours)
        return cls.place(placement, max_templates=max_templates, use_bundled=use_bundled)

    @classmethod
    def place(cls, placement: BankPlacement, max_templates: int = 4096,
              use_bundled: bool = True) -> "TemplateBank":
        """
        The bank of `placement`. Deterministic for a given seed; banks are cached
        per placement, and a bank bundled with gravyflow (res/template_banks)
        whose stored key equals the placement's is loaded instead of placed
        again. A bundled bank for the same settings but another noise curve,
        waveform model or code is not loaded: a warning names what differs and
        the bank is placed. A bank of more than `max_templates` is refused,
        whether it is placed or loaded.
        """
        if use_bundled:
            key = _normalised(placement.key())
            for record in _bundled_banks():
                if record["arguments"] == key:
                    bank = cls.from_dict(record)
                    if bank.num_templates > max_templates:
                        raise _too_many(max_templates)
                    return bank
            for record in _bundled_banks():
                stored = _provenance(record["arguments"])
                stale = [name for name, value in _provenance(key).items() if stored[name] != value]
                if stale and _settings(record["arguments"]) == _settings(key):
                    warnings.warn(f"The bundled bank for these arguments was placed with another {' and '.join(stale)} "
                                  "(regenerate it with gravyflow.src.detection.bundle_banks); placing a new bank.")
        from gravyflow.src.rng.platform import cbc_device
        # On the host's CPU whatever the default device, so a bank's bits do
        # not depend on it (an accelerator was no faster: the matches move
        # more data than they compute).
        with jax.default_device(cbc_device()):
            m1, m2, s1, s2, tried, acceptance = _stochastic_placement(placement, int(max_templates))
        return cls(m1, m2, s1, s2, placement=placement, candidates_tried=tried, final_acceptance=acceptance)

    def fitting_factors(self, mass_1, mass_2, space: Optional[MatchSpace] = None, spin_1z=None, spin_2z=None,
                        neighbours: Optional[int] = None, chunk: int = 256) -> np.ndarray:
        """
        The fitting factor (best match with any template) of each signal
        (mass_1[i], mass_2[i], spin_1z[i], spin_2z[i]) in the bank's match space
        (or `space`). This is the metric the bank is placed with; it measures
        placement, and agrees with independent overlaps only as far as the
        coverage tests show. With `neighbours`, each signal is matched with
        that many templates nearest it in the space's metric coordinates only,
        which can only lower its fitting factor: a lower bound.
        """
        space = space or self.space or MatchSpace()
        mass_1, mass_2 = np.atleast_1d(mass_1), np.atleast_1d(mass_2)
        out = np.zeros(len(mass_1), dtype=np.float32)
        coordinates = space.metric_coordinates(*self.parameters) if neighbours else None
        spin = lambda value, rows: None if value is None else np.atleast_1d(value)[rows]
        templates = lambda rows: space.whitened_spectra(*(values[rows] for values in self.parameters))
        for start in range(0, len(mass_1), chunk):
            rows = slice(start, start + chunk)
            signal = (mass_1[rows], mass_2[rows], spin(spin_1z, rows), spin(spin_2z, rows))
            spectra = space.whitened_spectra(*signal)
            if neighbours:
                shortlist = nearest(coordinates, space.metric_coordinates(*signal), neighbours)
                # Only the shortlisted templates' spectra are made.
                needed, index = np.unique(shortlist.ravel(), return_inverse=True)
                bank = np.concatenate([templates(needed[k:k + chunk]) for k in range(0, len(needed), chunk)])
                matches = pair_matches(spectra, bank, np.repeat(np.arange(len(spectra)), shortlist.shape[1]),
                                       index.ravel()).reshape(shortlist.shape)
                out[rows] = np.max(matches, axis=1)
            else:
                # The bank's spectra a chunk at a time: a large bank's whole
                # set does not fit in memory (30,000 templates of the
                # streaming span are 12 GB).
                best = np.zeros(len(spectra), dtype=np.float32)
                for first in range(0, self.num_templates, 4 * chunk):
                    bank = templates(slice(first, first + 4 * chunk))
                    best = np.maximum(best, np.max(max_matches(spectra, bank), axis=1))
                out[rows] = best
        return out

    def to_dict(self) -> dict:
        """The bank as JSON-ready data: its placement's key, parameters and placement statistics."""
        return {
            "arguments": None if self.placement is None else _normalised(self.placement.key()),
            "mass_1": [float(v) for v in self.mass_1],
            "mass_2": [float(v) for v in self.mass_2],
            "spin_1z": [float(v) for v in self.spin_1z],
            "spin_2z": [float(v) for v in self.spin_2z],
            "candidates_tried": int(self.candidates_tried),
            "final_acceptance": float(self.final_acceptance),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "TemplateBank":
        arguments = record.get("arguments")
        array = lambda name: np.asarray(record[name], dtype=np.float64)
        return cls(array("mass_1"), array("mass_2"), array("spin_1z"), array("spin_2z"),
                   placement=None if arguments is None else BankPlacement.from_key(arguments),
                   candidates_tried=int(record["candidates_tried"]),
                   final_acceptance=float(record["final_acceptance"]))


# The spacing, in log mass, of the candidates placed along the boundary of the
# mass range: a small fraction of the smallest template's extent there (about
# 0.02 at 5 solar masses), so that between two neighbouring candidates the
# fitting factor cannot dip measurably below the minimal match.
BOUNDARY_SPACING = 0.002


def _boundary_candidates(lo: float, hi: float):
    """
    (m1, m2) along the boundary of the mass range, in order: the edge m2 = lo
    (m1 rising from lo to hi), then the edge m1 = hi (m2 rising). Their ends
    are the range's corners. The diagonal m1 = m2 is not a boundary: the space
    folds there, so a template on either side covers both.
    """
    edge = np.geomspace(lo, hi, max(int(np.ceil(np.log(hi / lo) / BOUNDARY_SPACING)), 1) + 1)  # exact at both ends
    return (np.concatenate([edge, np.full(len(edge), hi)]), np.concatenate([np.full(len(edge), lo), edge]))


def _spin_corners(spin_range: float):
    """The corners (s1, s2) of the aligned-spin range; the single point (0, 0) without spins."""
    if spin_range == 0.0:
        return [(0.0, 0.0)]
    return [(a, b) for a in (spin_range, -spin_range) for b in (spin_range, -spin_range)]


def _draw_interior(rng, count: int, lo: float, hi: float, spin_range: float):
    """Candidates from the proposal: log-uniform component masses (m1 >= m2), uniform aligned spins."""
    draws = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(count, 2)))
    spins = rng.uniform(-spin_range, spin_range, size=(count, 2)) if spin_range else np.zeros((count, 2))
    return np.max(draws, axis=1), np.min(draws, axis=1), spins[:, 0], spins[:, 1]


def _draw_faces(rng, count: int, lo: float, hi: float, spin_range: float):
    """
    Candidates on the faces of the range: each one mass or spin at a bound
    (m2 = lo, m1 = hi, s1 = +-spin_range or s2 = +-spin_range, chosen
    uniformly), its other parameters from the proposal.
    """
    m1, m2, s1, s2 = _draw_interior(rng, count, lo, hi, spin_range)
    face = rng.integers(0, 6, size=count)
    free = np.exp(rng.uniform(np.log(lo), np.log(hi), size=count))
    m1, m2 = np.where(face == 0, free, m1), np.where(face == 0, lo, m2)
    m1, m2 = np.where(face == 1, hi, m1), np.where(face == 1, free, m2)
    sign = np.where(face % 2 == 0, 1.0, -1.0)
    s1 = np.where((face == 2) | (face == 3), sign * spin_range, s1)
    s2 = np.where((face == 4) | (face == 5), sign * spin_range, s2)
    return m1, m2, s1, s2


@lru_cache(maxsize=16)
def _stochastic_placement(placement: BankPlacement, max_templates: int):
    """
    The parameters (m1, m2, s1, s2) of `placement`'s bank, the candidates
    tried and the interior's final acceptance: its boundary first, then its
    interior.

    The boundary (one component mass at a bound of the mass range) is placed
    in order along it, every `BOUNDARY_SPACING` in log mass, at each corner of
    the spin range, so each of those points ends with a template within the
    minimal match. A stochastic placement alone leaves it badly covered: a
    template's neighbourhood there lies partly outside the range, and where
    templates are small (low masses) a strip along an edge is so small a share
    of the proposal's measure that the stopping rule does not see it (the
    corner m1 = m2 = 5 had a fitting factor of 0.43). With spins, the faces of
    the range (one parameter at a bound) are then placed stochastically, by
    the interior's stopping rule, for the same reason one dimension up. The
    interior is then placed stochastically until the stopping rule holds; the
    acceptance returned is the interior's.
    """
    space, minimal_match, per_round = placement.space, placement.minimal_match, placement.candidates_per_round
    lo, hi = placement.mass_range
    spin_range, neighbours = placement.spin_range, placement.neighbours
    convergence_candidates, target_coverage = placement.convergence_candidates, placement.target_coverage
    rng = np.random.default_rng(placement.seed)
    bank_spectra = _RowStore(space.num_samples // 2 + 1)
    bank_coordinates = np.zeros((0, 3))
    kept = [[], [], [], []]
    clock = {"start": time.perf_counter(), "rounds": 0}

    def matches_with(spectra, coordinates, others, other_coordinates):
        """(n, len(others)) matches of each candidate with `others`: every one, or its nearest `neighbours` (-1 elsewhere)."""
        if neighbours is None:
            return max_matches(spectra, others[:])
        out = np.full((len(spectra), len(others)), -1.0, dtype=np.float32)
        shortlist = nearest(other_coordinates, coordinates, neighbours)
        rows = np.repeat(np.arange(len(spectra)), shortlist.shape[1])
        out[rows, shortlist.ravel()] = pair_matches(spectra, others, rows, shortlist.ravel())
        return out

    def offer(c1, c2, cs1, cs2):
        """Keep each candidate, in order, unless a template already kept covers it: 1 for kept, 0 for not."""
        nonlocal bank_coordinates
        spectra = space.whitened_spectra(c1, c2, cs1, cs2)
        coordinates = space.metric_coordinates(c1, c2, cs1, cs2) if neighbours is not None else None
        against_bank = (np.max(matches_with(spectra, coordinates, bank_spectra, bank_coordinates), axis=1)
                        if len(bank_spectra) else np.zeros(len(c1), dtype=np.float32))
        among = matches_with(spectra, coordinates, spectra, coordinates)
        kept_now = []
        for i in range(len(c1)):
            if against_bank[i] < minimal_match and all(among[i, j] < minimal_match for j in kept_now):
                kept_now.append(i)
        clock["rounds"] += 1
        clock["offered"] = clock.get("offered", 0) + len(c1)
        if clock["rounds"] % 64 == 0:
            _log.info("placement (%s): %d templates kept, %d candidates offered, %.0f s", clock.get("phase"),
                      len(kept[0]) + len(kept_now), clock["offered"], time.perf_counter() - clock["start"])
        if kept_now:
            bank_spectra.extend(spectra[kept_now])
            if neighbours is not None:
                bank_coordinates = np.concatenate([bank_coordinates, coordinates[kept_now]])
            for values, candidates in zip(kept, (c1, c2, cs1, cs2)):
                values.extend(candidates[kept_now])
        if len(kept[0]) > max_templates:
            raise _too_many(max_templates)
        return [int(i in kept_now) for i in range(len(c1))]

    def converge(draw):
        """Offer rounds of `draw`'s candidates until the stopping rule holds: (candidates tried, final acceptance)."""
        history = []  # 1 for each kept candidate, 0 for each rejected one
        while True:
            history.extend(offer(*draw(rng, per_round, lo, hi, spin_range)))
            recent = history[-convergence_candidates:]
            if len(history) >= convergence_candidates and np.mean(recent) < 1.0 - target_coverage:
                return len(history), float(np.mean(recent))

    b1, b2 = _boundary_candidates(lo, hi)
    tried = 0
    clock["phase"] = "boundary"
    for corner in _spin_corners(spin_range):
        bs1, bs2 = np.full(len(b1), corner[0]), np.full(len(b1), corner[1])
        for start in range(0, len(b1), per_round):
            rows = slice(start, start + per_round)
            offer(b1[rows], b2[rows], bs1[rows], bs2[rows])
        tried += len(b1)
    if spin_range:
        clock["phase"] = "faces"
        tried += converge(_draw_faces)[0]
    clock["phase"] = "interior"
    interior, acceptance = converge(_draw_interior)
    _log.info("placement done: %d templates, %d candidates, %.0f s", len(kept[0]), tried + interior,
              time.perf_counter() - clock["start"])
    return (*(np.array(values) for values in kept), tried + interior, acceptance)


BUNDLED_BANK_DIRECTORY = "res/template_banks"


def _normalised(data: dict) -> dict:
    """`data` as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(data))


def _too_many(max_templates: int) -> ValueError:
    return ValueError(f"The bank exceeds max_templates ({max_templates}); raise it or lower the minimal match.")


# The parts of a bank's key that say what placed it, rather than what was asked for.
_SPACE_PROVENANCE = ("noise_curve", "waveform_model")


def _provenance(arguments: dict) -> dict:
    space = arguments["space"]
    return {**{name: space.get(name) for name in _SPACE_PROVENANCE}, "code": arguments.get("code")}


def _settings(arguments: dict) -> dict:
    space = {k: v for k, v in arguments["space"].items() if k not in _SPACE_PROVENANCE}
    return {**{k: v for k, v in arguments.items() if k != "code"}, "space": space}


@lru_cache(maxsize=1)
def _bundled_banks() -> tuple:
    """The bank records shipped in res/template_banks."""
    from gravyflow.src.resources import PACKAGE
    directory = PACKAGE / BUNDLED_BANK_DIRECTORY
    if not directory.is_dir():
        return ()
    return tuple(json.loads(path.read_text()) for path in sorted(directory.glob("*.json")))
