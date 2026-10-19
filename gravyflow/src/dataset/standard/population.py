"""
A standard dataset's injections: which sources (``BBHPopulation``) and when
(``InjectionSchedule``), each drawn from the dataset's own random streams, so the table
is a function of the dataset's seed, its population file and its segments alone.

Every parameter is drawn from its own stream (``stream/injection/{name}`` of the test
split), so adding a parameter moves no other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from gravyflow.src.rng import SeedTree

#: The split a standard dataset's streams are drawn under: it is test data.
SPLIT = "test"

#: The injection table's fields, in order: the pass it is injected in, then the source.
INJECTION_DTYPE = np.dtype([
    ("pass", "<i8"), ("tc", "<f8"), ("mass1", "<f8"), ("mass2", "<f8"), ("spin1z", "<f8"), ("spin2z", "<f8"),
    ("distance", "<f8"), ("inclination", "<f8"), ("ra", "<f8"), ("dec", "<f8"), ("coa_phase", "<f8"),
    ("polarization", "<f8"),
])


def _stream(tree: SeedTree, name: str) -> np.random.Generator:
    return tree.stream(SPLIT, f"stream/injection/{name}")


@dataclass(frozen=True)
class BBHPopulation:
    """
    A broad binary-black-hole population, placed uniformly in volume out to ``reach_mpc``.

    Args:
        mass_msun: Each component's mass, detector frame, uniform in this range; the
            heavier is ``mass1``.
        spin_z: Each component's dimensionless spin, aligned with the orbit, uniform in
            this range.
        reach_mpc: R: the luminosity distance is uniform in Euclidean volume out to it
            (density proportional to d squared). Beyond R even the population's loudest
            source is too quiet to detect.
        approximant: The waveform model the sources are defined for (ripple's name).
        lower_frequency_hertz: Where each waveform starts.

    The sky position is uniform on the sphere, the inclination uniform in its cosine, the
    coalescence phase uniform in [0, 2 pi) and the polarisation in [0, pi).
    """

    mass_msun: Tuple[float, float]
    spin_z: Tuple[float, float]
    reach_mpc: float
    approximant: str
    lower_frequency_hertz: float

    def __post_init__(self):
        object.__setattr__(self, "mass_msun", tuple(float(m) for m in self.mass_msun))
        object.__setattr__(self, "spin_z", tuple(float(s) for s in self.spin_z))
        low, high = self.mass_msun
        if not 0.0 < low <= high:
            raise ValueError(f"The mass range must be positive and ordered, got {self.mass_msun}.")
        if not -1.0 < self.spin_z[0] <= self.spin_z[1] < 1.0:
            raise ValueError(f"The aligned-spin range must be ordered inside (-1, 1), got {self.spin_z}.")
        if not self.reach_mpc > 0.0:
            raise ValueError(f"The reach must be positive, got {self.reach_mpc} Mpc.")

    def draw(self, tree: SeedTree, count: int) -> Dict[str, np.ndarray]:
        """``count`` sources: every field of ``INJECTION_DTYPE`` but the pass and ``tc``."""
        masses = np.sort(_stream(tree, "mass").uniform(*self.mass_msun, size=(2, count)), axis=0)
        return {
            "mass1": masses[1],
            "mass2": masses[0],
            "spin1z": _stream(tree, "spin1z").uniform(*self.spin_z, count),
            "spin2z": _stream(tree, "spin2z").uniform(*self.spin_z, count),
            "distance": self.reach_mpc * _stream(tree, "distance").uniform(0.0, 1.0, count) ** (1.0 / 3.0),
            "inclination": np.arccos(_stream(tree, "inclination").uniform(-1.0, 1.0, count)),
            "ra": _stream(tree, "ra").uniform(0.0, 2.0 * np.pi, count),
            "dec": np.arcsin(_stream(tree, "dec").uniform(-1.0, 1.0, count)),
            "coa_phase": _stream(tree, "coa_phase").uniform(0.0, 2.0 * np.pi, count),
            "polarization": _stream(tree, "polarization").uniform(0.0, np.pi, count),
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

    def times(self, tree: SeedTree, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Each injection's pass and merger time (GPS), sorted by pass then time."""
        slots = self.slots(segments)
        if len(slots) == 0:
            raise ValueError(f"No segment is longer than twice the {self.padding_seconds} s padding "
                             f"plus one {self.spacing_seconds} s slot: there is nowhere to inject.")
        full, last = divmod(self.count, len(slots))
        chosen = [np.arange(len(slots))] * full
        if last:
            chosen.append(np.sort(_stream(tree, "last_pass").choice(len(slots), last, replace=False)))
        passes = np.concatenate([np.full(len(c), p, np.int64) for p, c in enumerate(chosen)])
        offsets = _stream(tree, "tc").uniform(0.25, 0.75, self.count) * self.spacing_seconds
        return passes, slots[np.concatenate(chosen)] + offsets


def injection_table(population: BBHPopulation, schedule: InjectionSchedule, tree: SeedTree,
                    segments: np.ndarray) -> np.ndarray:
    """The injections: a structured array of ``INJECTION_DTYPE``, sorted by pass then ``tc``."""
    passes, tc = schedule.times(tree, segments)
    sources = population.draw(tree, schedule.count)
    table = np.empty(schedule.count, dtype=INJECTION_DTYPE)
    table["pass"], table["tc"] = passes, tc
    for name, values in sources.items():
        table[name] = values
    return table


def table_sha256(table: np.ndarray) -> str:
    """The sha256 of an injection table's bytes in ``INJECTION_DTYPE``."""
    return hashlib.sha256(np.ascontiguousarray(table, dtype=INJECTION_DTYPE).tobytes()).hexdigest()
