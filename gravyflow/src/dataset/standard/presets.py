"""
gravyflow's standard datasets: named presets that fix every piece of the data a model
is measured on, so results on one preset are comparable and results on two are never
mixed.

A preset (``StandardDataset``) fixes:

- **Time**: the test split of gravyflow's partition on its observing runs, the time
  every one of its detectors passes GWOSC's data-quality ``flags``, with every published
  event vetoed and no glitch vetoed (``build``). The list is computed once and pinned:
  ``segments.csv`` under ``res/standard/<slug>/``, by sha256.
- **Events**: the published events whose footprints the test split holds, for scoring
  separately (``events.csv``, pinned). They are never in the segments.
- **Rate**: 2048 Hz.
- **Injections**: two sets from one broad binary-black-hole population, on the same
  segments (``population``): the volume set, out to each mass's reach with weights that
  keep the volume exact, and the SNR set, spread evenly in network SNR. Both are fixed
  by the pinned parameter file ``population.json`` and the preset's seed; each table
  they make is pinned by sha256 too, so a change of random streams is refused, never
  served.
- **Gaussian twin** (``gaussian_twin``): coloured Gaussian noise from each segment's
  measured spectrum, over the same segments with the same injections.

``fingerprint`` names all of it. Loading a pinned file checks its sha256 and refuses a
file that differs, by name. The data is not downloaded here: the presets say what the
data is; the validator reads it.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import DEFAULT_GWOSC_FLAGS
from gravyflow.src.dataset.splits import DEFAULT_SPLIT, SplitSpec
from gravyflow.src.dataset.standard.build import sha256
from gravyflow.src.dataset.standard.population import (
    BBHPopulation, InjectionSchedule, Reach, SNRSet, VolumeSet, injection_table, table_sha256,
)
from gravyflow.src.rng import SeedTree
from gravyflow.src.utils.fingerprint import config_fingerprint

#: The presets' version: part of every fingerprint, so a change to what a preset means
#: changes the fingerprint of every preset.
STANDARD_VERSION = 2

#: The pinned files of every preset, under ``res/standard/<slug>/``.
PINNED_FILES = ("segments.csv", "events.csv", "population.json")

#: The injection sets of every preset, by name.
INJECTION_SETS = ("volume", "snr")


class PinnedFileChanged(ValueError):
    """A standard dataset's pinned file, or the table it makes, is not the one pinned."""


@dataclass(frozen=True)
class GaussianTwin:
    """
    How a preset's Gaussian twin is made: on each segment, coloured Gaussian noise with
    that segment's measured spectrum (one detector at a time), on absolute time, coloured
    as ``noise.color_noise`` colours a curve; the same segments and injections as the
    real data.

    Args:
        psd_seconds: Welch's segment length for the measured spectrum.
        psd_overlap: The overlap of Welch's segments, as a fraction.
        psd_average: How Welch's segments are averaged: "median", robust to a glitch.
        low_frequency_hertz: No power below it.
        filter_seconds: The colouring kernel's length.
    """

    psd_seconds: float = 4.0
    psd_overlap: float = 0.5
    psd_average: str = "median"
    low_frequency_hertz: float = 15.0
    filter_seconds: float = 128.0


@dataclass(frozen=True)
class StandardDataset:
    """
    One standard dataset (see the module). Ask for one by name with
    ``gf.standard_dataset``.

    Args:
        name: Its name, as ``STANDARD_DATASETS`` keys it.
        slug: Its directory under ``res/standard`` and its streams' name.
        gwosc_runs: The GWOSC runs its time is drawn from.
        pins: The sha256 of each of ``PINNED_FILES``.
        injections_sha256: The sha256 of each injection set's table (``table_sha256``), by
            set name (``INJECTION_SETS``).
        seed: The root of its random streams.
        detectors: Its detectors, by name.
        sample_rate_hertz: Its sample rate.
        flags: The GWOSC flags every detector passes in its time.
        minimum_segment_seconds: Its shortest segment.
        split: The partition its time is the test split of.
        twin: None for the real data; the twin's settings for its Gaussian twin.
    """

    name: str
    slug: str
    gwosc_runs: Tuple[str, ...]
    pins: Mapping[str, str]
    injections_sha256: Mapping[str, str]
    seed: int = 20261019
    detectors: Tuple[str, ...] = ("H1", "L1")
    sample_rate_hertz: float = 2048.0
    flags: Tuple[str, ...] = DEFAULT_GWOSC_FLAGS
    minimum_segment_seconds: float = 64.0
    split: SplitSpec = field(default=DEFAULT_SPLIT, compare=False)
    twin: Optional[GaussianTwin] = None

    #: The split a standard dataset's time is drawn from.
    split_name = "test"

    def __post_init__(self):
        object.__setattr__(self, "pins", dict(self.pins))
        object.__setattr__(self, "injections_sha256", dict(self.injections_sha256))
        if sorted(self.pins) != sorted(PINNED_FILES):
            raise ValueError(f"A standard dataset pins exactly {PINNED_FILES}, got {sorted(self.pins)}.")
        if sorted(self.injections_sha256) != sorted(INJECTION_SETS):
            raise ValueError(f"A standard dataset pins the tables of exactly {INJECTION_SETS}, "
                             f"got {sorted(self.injections_sha256)}.")

    # --------------------------------------------------------------- variants

    def gaussian_twin(self, settings: GaussianTwin = GaussianTwin()) -> "StandardDataset":
        """Its Gaussian twin: the same segments and injections, Gaussian noise (``GaussianTwin``)."""
        return dataclasses.replace(self, twin=settings)

    @property
    def label(self) -> str:
        """Its name, with "(Gaussian twin)" for a twin."""
        return self.name if self.twin is None else f"{self.name} (Gaussian twin)"

    # ---------------------------------------------------------------- the files

    def _read(self, file: str) -> bytes:
        from gravyflow.src.resources import path

        data = path(f"standard/{self.slug}/{file}").read_bytes()
        if sha256(data) != self.pins[file]:
            raise PinnedFileChanged(
                f"{self.name}'s {file} has sha256 {sha256(data)}, but {self.pins[file]} is pinned: the file "
                "is not the one the preset fixes. Restore it, or rebuild the preset (standard.build) and pin it anew.")
        return data

    @cached_property
    def segments(self) -> np.ndarray:
        """Its segments: sorted (S, 2) int64 GPS seconds, every detector holding each whole."""
        rows = np.loadtxt(io.StringIO(self._read("segments.csv").decode()), delimiter=",", skiprows=1,
                          dtype=np.int64, ndmin=2)
        return rows.reshape(-1, 2)

    @cached_property
    def events(self) -> Tuple[dict, ...]:
        """The published events in its test time, never in its segments: rows of ``build.EVENT_COLUMNS``,
        ``detectors`` a tuple of the detectors whose data holds the event."""
        rows = []
        for row in csv.DictReader(io.StringIO(self._read("events.csv").decode())):
            rows.append({
                "name": row["name"], "gps": float(row["gps"]), "catalog": row["catalog"],
                "confident": row["confident"] == "True", "chirp_mass_source": float(row["chirp_mass_source"]),
                "p_astro": float(row["p_astro"]),
                "detectors": tuple(row["detectors"].split("+")) if row["detectors"] else (),
            })
        return tuple(rows)

    @cached_property
    def _population_file(self) -> dict:
        return json.loads(self._read("population.json"))

    @property
    def population(self) -> BBHPopulation:
        """Its injection population, from its pinned ``population.json``."""
        return BBHPopulation(**self._population_file["population"])

    @property
    def volume_set(self) -> VolumeSet:
        """Its volume set's schedule and reach, from its pinned ``population.json``."""
        volume = self._population_file["volume"]
        return VolumeSet(InjectionSchedule(**volume["schedule"]), Reach(**volume["reach"]))

    @property
    def snr_set(self) -> SNRSet:
        """Its SNR set's schedule and range, from its pinned ``population.json``."""
        snr = self._population_file["snr"]
        return SNRSet(InjectionSchedule(**snr["schedule"]), snr["network_snr"])

    @property
    def tree(self) -> SeedTree:
        """Its random streams: its seed's, under its own name. A twin draws its noise from them too."""
        return SeedTree(self.seed).sub(f"dataset/{self.slug}")

    def _table(self, injection_set) -> np.ndarray:
        table = injection_table(self.population, injection_set, self.tree, self.segments)
        pinned = self.injections_sha256[injection_set.name]
        if table_sha256(table) != pinned:
            raise PinnedFileChanged(
                f"{self.name}'s {injection_set.name} injection table has sha256 {table_sha256(table)}, but {pinned} "
                "is pinned: its random streams or its schedule now draw other injections.")
        table.setflags(write=False)
        return table

    @cached_property
    def volume_injections(self) -> np.ndarray:
        """
        Its volume set, the same for the twin: a structured array of ``VolumeSet.dtype``,
        sorted by pass then merger time. The sensitive volume of a search is the sum of
        ``volume_weight`` over the found injections over their count (``VolumeSet``).
        """
        return self._table(self.volume_set)

    @cached_property
    def snr_injections(self) -> np.ndarray:
        """
        Its SNR set, the same for the twin: a structured array of ``SNRSet.dtype``, sorted
        by pass then merger time, each injection scaled at run time to its
        ``network_snr``.
        """
        return self._table(self.snr_set)

    # -------------------------------------------------------------- identity

    @property
    def livetime_seconds(self) -> int:
        """The seconds its segments hold (every detector holds each)."""
        return int(np.sum(np.diff(self.segments, axis=1)))

    def fingerprint(self) -> str:
        """The sha256 of everything that fixes its data."""
        return config_fingerprint({
            "standard_version": STANDARD_VERSION, "name": self.name, "slug": self.slug,
            "gwosc_runs": self.gwosc_runs, "detectors": self.detectors, "sample_rate_hertz": self.sample_rate_hertz,
            "flags": self.flags, "glitch_veto": "none", "minimum_segment_seconds": self.minimum_segment_seconds,
            "split": self.split.fingerprint(), "split_name": self.split_name, "pins": self.pins,
            "injections_sha256": self.injections_sha256, "seed": self.seed, "twin": self.twin,
        })

    def metadata(self) -> dict:
        """What a result measured on it records."""
        return {
            "dataset": self.label, "fingerprint": self.fingerprint(), "gwosc_runs": list(self.gwosc_runs),
            "detectors": list(self.detectors), "sample_rate_hertz": self.sample_rate_hertz,
            "segments": len(self.segments), "seconds": self.livetime_seconds, "events": len(self.events),
            "injections": {"volume": len(self.volume_injections), "snr": len(self.snr_injections)},
            "seed": self.seed,
            "twin": None if self.twin is None else dataclasses.asdict(self.twin),
        }


#: The standard datasets, by name.
STANDARD_DATASETS: Dict[str, StandardDataset] = {preset.name: preset for preset in (
    StandardDataset(
        name="O4 standard", slug="o4-standard", gwosc_runs=("O4a", "O4b"),
        pins={"segments.csv": "4effed063523cbbee1cc92fea6433528f2b8faec0e019c0104fba32569e09605",
              "events.csv": "540b4e829a6704b0cad611c620f4f12ccd75c067ccddaca6bfa1f92092c38c2e",
              "population.json": "7a3fdb0a166b07ed1264259878a1d7984f3529f997b8cc6f5d08b49a1760b4a9"},
        injections_sha256={"volume": "0e39185c08245f59e41d23554500a97f7ad944b084daf0b774b90aab70216882",
                           "snr": "27c755d62dd620f2bfd05e51b8429e6f98650b88eac541fafa80e88575989f74"}),
    StandardDataset(
        name="O3 open-weights", slug="o3-open-weights", gwosc_runs=("O3a", "O3b"),
        pins={"segments.csv": "0b4ab9ab20af5d857798bd6affa5607c890361b97415a701b86fc540919eb22d",
              "events.csv": "e8f0cc4f6c8f01a90e5e1bc40ad2e65246a4438264112f559990dd1c73d49d1b",
              "population.json": "279f362c0328a32b9bf0cda8a3ebb49b6c6fc371dbe9ee93bb6306fd164bbbcd"},
        injections_sha256={"volume": "4431b2a7f151ab988f9825574d1bf9fb880d7d30b0a8af2e87285f72197b06cb",
                           "snr": "191a14c063bc8c32a439d8d4feb1546abe991846f5c46e53396d55ea6b4d3d41"}),
)}


def standard_dataset(name: str, gaussian_twin: bool = False) -> StandardDataset:
    """
    The standard dataset ``name`` (a key of ``STANDARD_DATASETS``), or its Gaussian twin.
    KeyError naming the presets for another name.
    """
    if name not in STANDARD_DATASETS:
        raise KeyError(f"No standard dataset {name!r}; the standard datasets are {sorted(STANDARD_DATASETS)}.")
    preset = STANDARD_DATASETS[name]
    return preset.gaussian_twin() if gaussian_twin else preset
