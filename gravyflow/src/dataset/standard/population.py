"""
A standard dataset's injections: which sources (``BBHPopulation``), when
(``InjectionSchedule``), and how loud: two sets drawn from the one population on the
same segments.

- The **volume set** (``VolumeSet``) places each source at a luminosity distance drawn
  uniform on [0, R(m1, m2)], where R is its masses' reach (``Reach``): the distance at
  which the best-case source of those masses has a network SNR of ``Reach.network_snr``.
  Each injection carries the importance weight that makes the sensitive volume exact
  for a population uniform in Euclidean volume (``VolumeSet.place``).
- The **SNR set** (``SNRSet``) gives each source a network optimal SNR drawn uniform over
  a range; the distance follows at run time, from each segment's measured spectrum.

Each table is a function of the dataset's seed, its population file and its segments
alone. Every parameter is drawn from its own stream (``standard/<set>/<name>`` of the
test split), so adding a parameter moves no other, and the two sets share none.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from gravyflow.src.rng import SeedTree

#: The split a standard dataset's streams are drawn under: it is test data.
SPLIT = "test"

#: Every injection's fields, in order: the pass it is injected in, its merger time
#: (GPS), then the source. Spins are Cartesian, dimensionless, in the frame of the
#: orbital angular momentum at the population's reference frequency (z along it).
SOURCE_FIELDS = [
    ("pass", "<i8"), ("tc", "<f8"), ("mass1", "<f8"), ("mass2", "<f8"),
    ("spin1x", "<f8"), ("spin1y", "<f8"), ("spin1z", "<f8"), ("spin2x", "<f8"), ("spin2y", "<f8"), ("spin2z", "<f8"),
    ("inclination", "<f8"), ("ra", "<f8"), ("dec", "<f8"), ("coa_phase", "<f8"), ("polarization", "<f8"),
]

Stream = Callable[[str], np.random.Generator]


def _streams(tree: SeedTree, set_name: str) -> Stream:
    """The streams of one injection set: ``name`` -> its generator."""
    tree = tree.sub(f"standard/{set_name}")
    return lambda name: tree.stream(SPLIT, name)


@dataclass(frozen=True)
class BBHPopulation:
    """
    A broad binary-black-hole population with precession and higher modes.

    Args:
        mass_msun: Each component's mass, detector frame, uniform in this range; the
            heavier is ``mass1``.
        spin_magnitude: Each component's dimensionless spin magnitude, uniform in this
            range, its direction isotropic (so the spins precess).
        approximant: The waveform model the sources are defined for (ripple's name).
        lower_frequency_hertz: Where each waveform starts.
        reference_frequency_hertz: Where the spins and the phase are defined.

    The sky position is uniform on the sphere, the inclination uniform in its cosine, the
    coalescence phase uniform in [0, 2 pi) and the polarisation in [0, pi).
    """

    mass_msun: Tuple[float, float]
    spin_magnitude: Tuple[float, float]
    approximant: str
    lower_frequency_hertz: float
    reference_frequency_hertz: float

    def __post_init__(self):
        object.__setattr__(self, "mass_msun", tuple(float(m) for m in self.mass_msun))
        object.__setattr__(self, "spin_magnitude", tuple(float(s) for s in self.spin_magnitude))
        low, high = self.mass_msun
        if not 0.0 < low <= high:
            raise ValueError(f"The mass range must be positive and ordered, got {self.mass_msun}.")
        if not 0.0 <= self.spin_magnitude[0] <= self.spin_magnitude[1] < 1.0:
            raise ValueError(f"The spin magnitudes must be ordered inside [0, 1), got {self.spin_magnitude}.")

    def _spins(self, stream: Stream, component: str, count: int) -> np.ndarray:
        magnitude = stream(f"{component}_magnitude").uniform(*self.spin_magnitude, count)
        cos_tilt = stream(f"{component}_cos_tilt").uniform(-1.0, 1.0, count)
        azimuth = stream(f"{component}_azimuth").uniform(0.0, 2.0 * np.pi, count)
        sin_tilt = np.sqrt(1.0 - cos_tilt ** 2)
        return magnitude * np.stack([sin_tilt * np.cos(azimuth), sin_tilt * np.sin(azimuth), cos_tilt])

    def draw(self, stream: Stream, count: int) -> Dict[str, np.ndarray]:
        """``count`` sources: every field of ``SOURCE_FIELDS`` but the pass and ``tc``."""
        masses = np.sort(stream("mass").uniform(*self.mass_msun, size=(2, count)), axis=0)
        spin1, spin2 = self._spins(stream, "spin1", count), self._spins(stream, "spin2", count)
        return {
            "mass1": masses[1], "mass2": masses[0],
            "spin1x": spin1[0], "spin1y": spin1[1], "spin1z": spin1[2],
            "spin2x": spin2[0], "spin2y": spin2[1], "spin2z": spin2[2],
            "inclination": np.arccos(stream("inclination").uniform(-1.0, 1.0, count)),
            "ra": stream("ra").uniform(0.0, 2.0 * np.pi, count),
            "dec": np.arcsin(stream("dec").uniform(-1.0, 1.0, count)),
            "coa_phase": stream("coa_phase").uniform(0.0, 2.0 * np.pi, count),
            "polarization": stream("polarization").uniform(0.0, np.pi, count),
        }


@dataclass(frozen=True)
class InjectionSchedule:
    """
    When the injections happen. Each segment, less ``padding_seconds`` at each end, is cut
    into slots of ``spacing_seconds`` from its start; each slot holds one injection per
    pass, its merger uniform in the slot's middle half, so the mergers of one pass are at
    least half a spacing apart and never within the padding of a segment's edge. A pass
    fills every slot; the last pass fills ``count`` less the earlier passes' slots,
    chosen at random.

    Args:
        count: The injections.
        spacing_seconds: One slot's length.
        padding_seconds: No merger within this of a segment's edge.
    """

    count: int
    spacing_seconds: float
    padding_seconds: float

    def __post_init__(self):
        if not (isinstance(self.count, int) and self.count > 0):
            raise ValueError(f"The injection count must be a positive integer, got {self.count!r}.")
        if not (self.spacing_seconds > 0.0 and self.padding_seconds >= 0.0):
            raise ValueError("The spacing must be positive and the padding non-negative, got "
                             f"{self.spacing_seconds} s and {self.padding_seconds} s.")

    def slots(self, segments: np.ndarray) -> np.ndarray:
        """Each slot's start, GPS seconds, in time order."""
        starts = []
        for start, end in np.reshape(segments, (-1, 2)).astype(np.float64):
            usable = (end - start) - 2.0 * self.padding_seconds
            count = int(np.floor(usable / self.spacing_seconds)) if usable > 0 else 0
            starts.append(start + self.padding_seconds + self.spacing_seconds * np.arange(count))
        return np.concatenate(starts) if starts else np.empty(0)

    def passes(self, segments: np.ndarray) -> int:
        """How many passes ``count`` injections take over ``segments``."""
        return -(-self.count // len(self.slots(segments)))

    def times(self, stream: Stream, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Each injection's pass and merger time (GPS), sorted by pass then time."""
        slots = self.slots(segments)
        if len(slots) == 0:
            raise ValueError(f"No segment is longer than twice the {self.padding_seconds} s padding "
                             f"plus one {self.spacing_seconds} s slot: there is nowhere to inject.")
        full, last = divmod(self.count, len(slots))
        chosen = [np.arange(len(slots))] * full
        if last:
            chosen.append(np.sort(stream("last_pass").choice(len(slots), last, replace=False)))
        passes = np.concatenate([np.full(len(c), p, np.int64) for p, c in enumerate(chosen)])
        offsets = stream("tc").uniform(0.25, 0.75, self.count) * self.spacing_seconds
        return passes, slots[np.concatenate(chosen)] + offsets


@dataclass(frozen=True)
class Reach:
    """
    How far a source of given masses can be heard: R(m1, m2), the luminosity distance at
    which the best-case source of those masses (optimal sky and orientation, the most
    favourable spins) has a network optimal SNR of ``network_snr`` in the dataset's noise.
    Nothing beyond R is findable.

    R is tabulated: ``best_case_snr_at_1_gpc[i][j]`` is the best-case network SNR at
    1 Gpc of masses (``mass_grid_msun[i]``, ``mass_grid_msun[j]``) with ``approximant``
    in the lowest of ``noise_profiles`` (``build.best_case_snr_grid``), and
    ``safety_factor`` bounds how much louder the population's own model can be (its
    precession and higher modes). Between grid points R takes the largest of the four
    corners of its cell, so it is never below a neighbouring grid point's:

        R(m1, m2) = 1 Gpc * safety_factor * max(corners) / network_snr.

    Args:
        network_snr: The SNR at the reach.
        safety_factor: The population's best case over ``approximant``'s.
        approximant: The model the grid is computed with.
        noise_profiles: The noise curves (``gf.NoiseProfile`` names) whose lowest the
            grid is computed in.
        mass_grid_msun: The grid's masses, increasing.
        best_case_snr_at_1_gpc: The (G, G) grid, symmetric.
    """

    network_snr: float
    safety_factor: float
    approximant: str
    noise_profiles: Tuple[str, ...]
    mass_grid_msun: Tuple[float, ...]
    best_case_snr_at_1_gpc: Tuple[Tuple[float, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "noise_profiles", tuple(self.noise_profiles))
        object.__setattr__(self, "mass_grid_msun", tuple(float(m) for m in self.mass_grid_msun))
        object.__setattr__(self, "best_case_snr_at_1_gpc",
                           tuple(tuple(float(s) for s in row) for row in self.best_case_snr_at_1_gpc))
        grid, snr = np.asarray(self.mass_grid_msun), np.asarray(self.best_case_snr_at_1_gpc)
        if len(grid) < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError(f"The reach's mass grid must hold at least two increasing masses, got {grid}.")
        if snr.shape != (len(grid), len(grid)) or not np.allclose(snr, snr.T) or not np.all(snr > 0):
            raise ValueError(f"The reach's grid must be a positive, symmetric ({len(grid)}, {len(grid)}) table.")
        if not (self.network_snr > 0 and self.safety_factor >= 1.0):
            raise ValueError(f"The reach needs a positive SNR and a safety factor of at least 1, got "
                             f"{self.network_snr} and {self.safety_factor}.")

    def distance_mpc(self, mass1: np.ndarray, mass2: np.ndarray) -> np.ndarray:
        """R(m1, m2) in Mpc; masses outside the grid are refused."""
        grid, snr = np.asarray(self.mass_grid_msun), np.asarray(self.best_case_snr_at_1_gpc)
        mass1, mass2 = np.asarray(mass1, np.float64), np.asarray(mass2, np.float64)
        if min(mass1.min(), mass2.min()) < grid[0] or max(mass1.max(), mass2.max()) > grid[-1]:
            raise ValueError(f"The reach is tabulated over {grid[0]}-{grid[-1]} solar masses; a mass lies outside.")
        i = np.clip(np.searchsorted(grid, mass1, side="right") - 1, 0, len(grid) - 2)
        j = np.clip(np.searchsorted(grid, mass2, side="right") - 1, 0, len(grid) - 2)
        corners = np.max([snr[i, j], snr[i + 1, j], snr[i, j + 1], snr[i + 1, j + 1]], axis=0)
        return 1000.0 * self.safety_factor * corners / self.network_snr


@dataclass(frozen=True)
class VolumeSet:
    """
    The volume set: each source at a distance d uniform on [0, R(m1, m2)] (``Reach``), the
    distance scaled by its masses' reach as chirp distance scales it by the chirp mass,
    but through merger and ringdown, where the inspiral's M_c^(5/6) does not hold.

    The sources are drawn from q(m, d) = p(m) / R(m) on d < R(m), with sky and orientation
    as the population's; the population measured is p(m) uniform in Euclidean volume,
    and nothing beyond R is found. So each injection carries the importance weight

        w = 4 pi d^2 R(m1, m2)   (Mpc^3)

    and the sensitive volume of a search is V = (1 / N) sum over the found injections of
    w, over all N injections (Tiwari 2018, Class. Quantum Grav. 35 145009, the weighted
    Monte-Carlo estimate of VT). The mean weight over all N is the population's mean
    volume within reach, (4 pi / 3) R^3 averaged over the masses.
    """

    schedule: InjectionSchedule
    reach: Reach

    name = "volume"
    dtype = np.dtype(SOURCE_FIELDS + [("distance", "<f8"), ("reach", "<f8"), ("volume_weight", "<f8")])

    def place(self, sources: Dict[str, np.ndarray], stream: Stream) -> Dict[str, np.ndarray]:
        """Each source's distance (Mpc), its reach (Mpc) and its weight (Mpc^3)."""
        reach = self.reach.distance_mpc(sources["mass1"], sources["mass2"])
        distance = reach * stream("distance").uniform(0.0, 1.0, len(reach))
        return {"distance": distance, "reach": reach, "volume_weight": 4.0 * np.pi * distance ** 2 * reach}


@dataclass(frozen=True)
class SNRSet:
    """
    The SNR set: each source scaled to a network optimal SNR uniform over ``network_snr``,
    in the measured spectrum of the segment it is injected in; its distance follows at
    run time. The range starts at 0, so the efficiency curve falls to the chance level
    at its foot: a check built into every result.
    """

    schedule: InjectionSchedule
    network_snr: Tuple[float, float]

    name = "snr"
    dtype = np.dtype(SOURCE_FIELDS + [("network_snr", "<f8")])

    def __post_init__(self):
        object.__setattr__(self, "network_snr", tuple(float(s) for s in self.network_snr))
        if not 0.0 <= self.network_snr[0] < self.network_snr[1]:
            raise ValueError(f"The SNR range must be non-negative and ordered, got {self.network_snr}.")

    def place(self, sources: Dict[str, np.ndarray], stream: Stream) -> Dict[str, np.ndarray]:
        """Each source's network optimal SNR."""
        return {"network_snr": stream("network_snr").uniform(*self.network_snr, len(sources["mass1"]))}


def injection_table(population: BBHPopulation, injection_set, tree: SeedTree, segments: np.ndarray) -> np.ndarray:
    """One set's injections (``VolumeSet`` or ``SNRSet``): a structured array of its ``dtype``,
    sorted by pass then ``tc``."""
    stream = _streams(tree, injection_set.name)
    passes, tc = injection_set.schedule.times(stream, segments)
    sources = population.draw(stream, injection_set.schedule.count)
    table = np.empty(injection_set.schedule.count, dtype=injection_set.dtype)
    table["pass"], table["tc"] = passes, tc
    for name, values in {**sources, **injection_set.place(sources, stream)}.items():
        table[name] = values
    return table


def table_sha256(table: np.ndarray) -> str:
    """The sha256 of an injection table's bytes."""
    return hashlib.sha256(np.ascontiguousarray(table).tobytes()).hexdigest()
