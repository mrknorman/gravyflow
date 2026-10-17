"""
Template banks for matched filtering.

`TemplateBank.stochastic` places non-spinning CBC templates so that every point
of the mass range has a template whose match with it is at least the minimal
match, measured with the detector's noise curve on the analysis band: the
bank's coverage is a measured property, not an assumption. `TemplateGrid` is a
plain grid in component masses, with no coverage guarantee; a linear grid
leaves light binaries badly covered (a fitting factor near 0.5 below 30 solar
masses for a 16 x 16 grid over 5-95), so it is kept only as a simple utility.
"""

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp


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

def noise_curve_psd(ifo, frequencies: np.ndarray) -> np.ndarray:
    """
    The one-sided PSD (strain^2/Hz) of an IFO's bundled noise curve (the curve
    its coloured noise is drawn from) on `frequencies`, log-log interpolated and
    held constant beyond the tabulated range.
    """
    curve_frequencies, asd = np.loadtxt(ifo.value.optimal_psd_path, delimiter=",").T
    f = np.clip(np.asarray(frequencies, dtype=np.float64), curve_frequencies[0], curve_frequencies[-1])
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
) -> np.ndarray:
    """
    Plus polarisations of face-on, non-spinning binaries at reference phase 0,
    from the CBC generator the dataset injects with (`_generate_cbc_window`), with
    the merger at the centre of a `window_seconds` buffer and no content below
    `lower_frequency_hertz`. Unit-peak rows, float32, shape (n, samples).
    """
    from gravyflow.src.dataset.features.waveforms.cbc import Approximant, _generate_cbc_window
    approximant = Approximant.IMRPhenomD if approximant is None else approximant
    mass_1 = np.atleast_1d(np.asarray(mass_1, dtype=np.float64))
    mass_2 = np.atleast_1d(np.asarray(mass_2, dtype=np.float64))
    n = len(mass_1)
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
    )
    plus = np.asarray(waveforms[:, 0, :], dtype=np.float64)
    peak = np.max(np.abs(plus), axis=-1, keepdims=True)
    return (plus / np.where(peak > 0, peak, 1.0)).astype(np.float32)


@dataclass(frozen=True)
class MatchSpace:
    """
    The inner product the bank is placed with: the detector's noise curve on the
    band [lower_frequency_hertz, Nyquist), on the visible part of each template,
    the `visible_seconds` before its merger (with a Hann ramp over the first
    tenth) and everything after it.

    The matched filter sees at most the onsource window plus half the whitening
    filter of a signal's inspiral, so a bank placed on that span is at least as
    dense as any analysis with a shorter window needs.
    """
    sample_rate_hertz: float = 2048.0
    visible_seconds: float = 1.5
    lower_frequency_hertz: float = 20.0
    ifo_name: str = "L1"
    approximant_name: str = "IMRPhenomD"

    @property
    def window_seconds(self) -> float:
        return 2.0 * self.visible_seconds

    def identity(self) -> dict:
        """
        Everything the match depends on: these settings, the noise curve (its
        file and a hash of its contents) and the waveform model (the
        approximant, and the release series of the package that computes it).
        """
        import gravyflow as gf
        curve = Path(gf.IFO[self.ifo_name].value.optimal_psd_path)
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

    def whitened_spectra(self, mass_1, mass_2) -> np.ndarray:
        """Unit-norm noise-weighted spectra (n, F) complex64, zero outside the band."""
        import gravyflow as gf
        from gravyflow.src.dataset.features.waveforms.cbc import Approximant
        h = cbc_templates(mass_1, mass_2, self.sample_rate_hertz, self.window_seconds,
                          self.lower_frequency_hertz, Approximant[self.approximant_name])
        h = h * self._ramp()
        freqs = np.fft.rfftfreq(self.num_samples, 1.0 / self.sample_rate_hertz)
        weight = np.zeros_like(freqs)
        band = (freqs >= self.lower_frequency_hertz) & (freqs < self.sample_rate_hertz / 2.0)
        weight[band] = 1.0 / np.sqrt(noise_curve_psd(gf.IFO[self.ifo_name], freqs[band]))
        spectra = np.fft.rfft(h.astype(np.float64), axis=-1) * weight
        norm = np.sqrt(np.sum(np.abs(spectra) ** 2, axis=-1, keepdims=True))
        return (spectra / np.where(norm > 0, norm, 1.0)).astype(np.complex64)


@jax.jit
def _max_overlap(a, b):
    """max over time shifts of |<a|b>| for unit-norm one-sided spectra a (P, F), b (Q, F): (P, Q)."""
    num_freqs = a.shape[-1]
    n = 2 * (num_freqs - 1)
    product = a[:, None, :] * jnp.conj(b)[None, :, :]
    full = jnp.zeros(product.shape[:-1] + (n,), dtype=product.dtype).at[..., :num_freqs].set(product)
    return jnp.max(jnp.abs(jnp.fft.ifft(full, axis=-1)), axis=-1) * n


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
    The arguments of a stochastic placement (see `TemplateBank.stochastic`).
    Its `key` names what the placed bank depends on: these settings, the match
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

    def __post_init__(self):
        if not 0.0 < self.minimal_match < 1.0:
            raise ValueError("minimal_match must be in (0, 1).")
        lo, hi = self.mass_range
        if not 0.0 < lo <= hi:
            raise ValueError("mass_range must be (min, max) with 0 < min <= max.")

    def key(self) -> dict:
        return {
            "space": self.space.identity(),
            "mass_range": [float(m) for m in self.mass_range],
            "minimal_match": float(self.minimal_match),
            "seed": int(self.seed),
            "candidates_per_round": int(self.candidates_per_round),
            "target_coverage": float(self.target_coverage),
            "convergence_candidates": int(self.convergence_candidates),
            "code": placement_code(),
        }

    @classmethod
    def from_key(cls, key: dict) -> "BankPlacement":
        fields = {name: key["space"][name] for name in MatchSpace.__dataclass_fields__}
        return cls(MatchSpace(**fields), tuple(key["mass_range"]), key["minimal_match"], key["seed"],
                   key["candidates_per_round"], key["target_coverage"], key["convergence_candidates"])


@dataclass(frozen=True)
class TemplateBank:
    """
    Component masses of a non-spinning CBC template bank (m1 >= m2), with the
    placement that made it. Build one with `TemplateBank.stochastic`, or wrap
    fixed masses with `TemplateBank.from_masses` (no coverage guarantee).
    """
    mass_1: np.ndarray
    mass_2: np.ndarray
    placement: Optional[BankPlacement] = None
    candidates_tried: int = 0
    final_acceptance: float = float("nan")

    @property
    def num_templates(self) -> int:
        return int(len(self.mass_1))

    @property
    def minimal_match(self) -> Optional[float]:
        return None if self.placement is None else self.placement.minimal_match

    @property
    def space(self) -> Optional[MatchSpace]:
        return None if self.placement is None else self.placement.space

    @classmethod
    def from_masses(cls, mass_1, mass_2) -> "TemplateBank":
        m1 = np.atleast_1d(np.asarray(mass_1, dtype=np.float64))
        m2 = np.atleast_1d(np.asarray(mass_2, dtype=np.float64))
        if m1.shape != m2.shape or m1.ndim != 1 or len(m1) == 0:
            raise ValueError("mass_1 and mass_2 must be equal-length, non-empty 1D arrays.")
        return cls(np.maximum(m1, m2), np.minimum(m1, m2))

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
    ) -> "TemplateBank":
        """
        Stochastic placement (Harry, Allen & Sathyaprakash 2009; Babak 2008):
        candidates drawn log-uniform in each component mass over `mass_range` are
        kept when their match with every template already kept is below
        `minimal_match`. Placement stops when, over the last
        `convergence_candidates` candidates, fewer than 1 - `target_coverage` were
        kept: the fraction of the space (in the proposal's measure) left with a
        fitting factor below the minimal match is then about that small.

        The match is `MatchSpace`'s: the IFO's noise curve (default L1) over
        [lower_frequency_hertz, Nyquist), on the `visible_seconds` before merger.
        See `place` for caching and the banks bundled with gravyflow.
        """
        import gravyflow as gf
        ifo = gf.IFO.L1 if ifo is None else ifo
        space = MatchSpace(float(sample_rate_hertz), float(visible_seconds), float(lower_frequency_hertz),
                           ifo.name, "IMRPhenomD")
        placement = BankPlacement(space, (float(mass_range[0]), float(mass_range[1])), float(minimal_match),
                                  int(seed), int(candidates_per_round), float(target_coverage),
                                  int(convergence_candidates))
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
        m1, m2, tried, acceptance = _stochastic_masses(placement, int(max_templates))
        return cls(m1, m2, placement, tried, acceptance)

    def fitting_factors(self, mass_1, mass_2, space: Optional[MatchSpace] = None) -> np.ndarray:
        """
        The fitting factor (best match with any template) of each signal
        (mass_1[i], mass_2[i]) in the bank's match space (or `space`). This is
        the metric the bank is placed with; it measures placement, and agrees
        with independent overlaps only as far as the coverage tests show.
        """
        space = space or self.space or MatchSpace()
        signals = space.whitened_spectra(mass_1, mass_2)
        bank = space.whitened_spectra(self.mass_1, self.mass_2)
        return np.max(max_matches(signals, bank), axis=1)

    def to_dict(self) -> dict:
        """The bank as JSON-ready data: its placement's key, masses and placement statistics."""
        return {
            "arguments": None if self.placement is None else _normalised(self.placement.key()),
            "mass_1": [float(v) for v in self.mass_1],
            "mass_2": [float(v) for v in self.mass_2],
            "candidates_tried": int(self.candidates_tried),
            "final_acceptance": float(self.final_acceptance),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "TemplateBank":
        arguments = record.get("arguments")
        return cls(np.asarray(record["mass_1"], dtype=np.float64), np.asarray(record["mass_2"], dtype=np.float64),
                   None if arguments is None else BankPlacement.from_key(arguments),
                   int(record["candidates_tried"]), float(record["final_acceptance"]))


@lru_cache(maxsize=16)
def _stochastic_masses(placement: BankPlacement, max_templates: int):
    space, minimal_match, per_round = placement.space, placement.minimal_match, placement.candidates_per_round
    lo, hi = placement.mass_range
    convergence_candidates, target_coverage = placement.convergence_candidates, placement.target_coverage
    rng = np.random.default_rng(placement.seed)
    bank_spectra = np.zeros((0, space.num_samples // 2 + 1), dtype=np.complex64)
    m1_kept, m2_kept = [], []
    history = []  # 1 for each kept candidate, 0 for each rejected one
    tried = 0
    while True:
        draws = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(per_round, 2)))
        c1, c2 = np.max(draws, axis=1), np.min(draws, axis=1)
        spectra = space.whitened_spectra(c1, c2)
        against_bank = (np.max(max_matches(spectra, bank_spectra), axis=1)
                        if len(bank_spectra) else np.zeros(per_round, dtype=np.float32))
        among = max_matches(spectra, spectra)
        kept_now = []
        for i in range(per_round):
            covered = against_bank[i] >= minimal_match or any(among[i, j] >= minimal_match for j in kept_now)
            history.append(0 if covered else 1)
            if not covered:
                kept_now.append(i)
        tried += per_round
        if kept_now:
            bank_spectra = np.concatenate([bank_spectra, spectra[kept_now]])
            m1_kept.extend(c1[kept_now])
            m2_kept.extend(c2[kept_now])
        if len(m1_kept) > max_templates:
            raise _too_many(max_templates)
        recent = history[-convergence_candidates:]
        if len(history) >= convergence_candidates and np.mean(recent) < 1.0 - target_coverage:
            return np.array(m1_kept), np.array(m2_kept), tried, float(np.mean(recent))


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
    import gravyflow as gf
    directory = gf.PATH / BUNDLED_BANK_DIRECTORY
    if not directory.is_dir():
        return ()
    return tuple(json.loads(path.read_text()) for path in sorted(directory.glob("*.json")))
