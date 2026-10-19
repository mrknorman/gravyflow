"""
MLGWSC-1's datasets, made by gravyflow and streamed: read over any span of their segments,
never written whole (a written 30-day dataset is about 42 GB a file).

``MLGWSC1Stream(1, seed=...)`` is the challenge's dataset 1 (``generate_data.py -d 1`` and
``ds1.ini``, Schäfer et al. 2023, arXiv:2209.11146), drawn from gravyflow's own random
streams: the same distribution as the challenge's data, not its samples (the challenge's
test seeds are not published, and its generators draw from NumPy's legacy streams).

- **Time**: the challenge's segments (``mlgwsc1.restrict_segments``): 30 days by default,
  from segments of at least two hours, at 2048 Hz, on H1 and L1.
- **Noise**: stationary Gaussian noise of aLIGOZeroDetHighPower from 15 Hz on each
  detector, independent between detectors, continuous over each segment
  (``noise.stationary``, as the challenge's ``NoiseGenerator`` makes it).
- **Injections**: ``ds1.ini``'s population (``Population``) at the times
  ``pycbc_create_injections`` schedules over the whole O3a span (each 24 s after the last
  plus a uniform draw up to 6 s), injected where they fall at least 30 s inside a segment
  (``make_injections``' padding). The model is IMRPhenomXPHM with no spin and only its
  (2, 2) mode, which is IMRPhenomXAS (ripple's), conditioned below 20 Hz as PyCBC
  conditions the challenge's (``Injector``), its t = 0 at tc, and projected onto the
  detectors at each injection's GPS time.

``read(start, end)`` is the strain of one span as a ``SearchSegment``, the unit an
external search runs on; ``injections=False`` reads the background (the same noise with
nothing injected), as the challenge's background file holds it. ``injections`` is the
table a sensitivity is scored from, with the challenge's injection file's fields.

Simulated noise reads no real strain, so no ``Reservation`` applies to it (``mode``).
"""

import functools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0

from gravyflow.src.dataset.acquisition import mlgwsc1
from gravyflow.src.dataset.conditioning.detector import IFO, Network, NoiseProfile, shift_waveform
from gravyflow.src.dataset.features.waveforms.cbc import (
    Approximant, _frequency_domain_polarisations, _in_x64, _source_parameters, calc_duration_from_f_min,
)
from gravyflow.src.dataset.noise.stationary import StationaryNoise
from gravyflow.src.external.protocol import SearchSegment
from gravyflow.src.rng import SeedTree
from gravyflow.src.utils.fingerprint import config_fingerprint, library_versions

#: The stream's version: part of its fingerprint, so a change to how it is made changes
#: the fingerprint of every stream.
STREAM_VERSION = 1

#: The split whose streams a challenge dataset draws from: it is test data.
SPLIT = "test"

#: What a simulated dataset reads of real strain: nothing, so no reservation applies.
SIMULATED = "simulated: reads no real strain, so no reservation applies"

#: The injection table's fields, as the challenge's injection file names them.
INJECTION_FIELDS = ("tc", "mass1", "mass2", "ra", "dec", "inclination", "coa_phase", "polarization",
                    "chirp_distance", "distance")

#: The chirp mass the chirp distance is referred to: two 1.4 solar-mass bodies (PyCBC's
#: ``distance_from_chirp_distance_mchirp``, ref_mass 1.4).
REFERENCE_CHIRP_MASS_MSUN = 2.0 ** -0.2 * 1.4


def chirp_mass(mass1, mass2):
    """The chirp mass of component masses, in their units."""
    return (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2


@dataclass(frozen=True)
class Population:
    """
    A dataset's injection population, as its ``ds{n}.ini`` gives it. Dataset 1's: component
    masses uniform in [10, 50] solar masses with mass2 <= mass1, no spin; sky position
    uniform on the sphere; inclination uniform in its cosine; coalescence phase and
    polarisation uniform in [0, 2 pi); chirp distance uniform in volume between 130 and 350
    Mpc, and the luminosity distance from it, by the chirp mass.

    Each parameter is drawn from its own stream (``stream/injection/{name}``), so that adding
    a parameter moves no other.
    """

    mass_msun: Tuple[float, float] = (10.0, 50.0)
    chirp_distance_mpc: Tuple[float, float] = (130.0, 350.0)

    def draw(self, tree: SeedTree, count: int) -> Dict[str, np.ndarray]:
        """``count`` sources: each parameter of ``INJECTION_FIELDS`` but ``tc``, float64."""
        def stream(name):
            return tree.stream(SPLIT, f"stream/injection/{name}")

        # Two uniform masses, labelled heavier first: uniform on the square folded onto
        # mass2 <= mass1 is ds1.ini's constraint, which PyCBC meets by rejection.
        masses = np.sort(stream("mass").uniform(*self.mass_msun, size=(2, count)), axis=0)
        lo, hi = self.chirp_distance_mpc
        chirp_distance = stream("chirp_distance").uniform(lo ** 3, hi ** 3, count) ** (1.0 / 3.0)
        mass1, mass2 = masses[1], masses[0]
        return {
            "mass1": mass1,
            "mass2": mass2,
            "ra": stream("ra").uniform(0.0, 2.0 * np.pi, count),
            "dec": np.arcsin(stream("dec").uniform(-1.0, 1.0, count)),
            "inclination": np.arccos(stream("inclination").uniform(-1.0, 1.0, count)),
            "coa_phase": stream("coa_phase").uniform(0.0, 2.0 * np.pi, count),
            "polarization": stream("polarization").uniform(0.0, 2.0 * np.pi, count),
            "chirp_distance": chirp_distance,
            "distance": chirp_distance * (chirp_mass(mass1, mass2) / REFERENCE_CHIRP_MASS_MSUN) ** (5.0 / 6.0),
        }


@dataclass(frozen=True)
class Schedule:
    """
    When injections happen: ``pycbc_create_injections --time-step 24 --time-window 6``
    from the start to the end of the challenge's segment list, as generate_data.py calls
    it. The first is a uniform draw up to ``window_seconds`` after the start, and each
    next one ``step_seconds`` after the last plus another (the script adds the cumulative
    sum of the draws to a grid of steps).
    """

    step_seconds: float = 24.0
    window_seconds: float = 6.0

    def times(self, tree: SeedTree, start: float, end: float) -> np.ndarray:
        """The injection times in [start, end), float64 GPS."""
        count = int(np.ceil((end - start) / self.step_seconds)) + 1
        draws = tree.stream(SPLIT, "stream/injection/tc").uniform(0.0, self.window_seconds, count)
        times = start + self.step_seconds * np.arange(count) + np.cumsum(draws)
        return times[times < end]


#: PyCBC's time-domain conditioning of a frequency-domain model (``waveform.
#: _base_get_td_waveform_from_fd``, which ``get_td_waveform`` uses for IMRPhenomXPHM): the
#: model is taken from a lower frequency, ``f_lower`` lowered in steps of 1 % until the
#: waveform is 1.5 times as long, and rolled up to ``f_lower`` by the rising half of a
#: Kaiser window (beta 8) in frequency (``waveform.utils.fd_taper``).
TAPER_DURATION_FACTOR = 1.5
TAPER_FREQUENCY_STEP = 0.99
TAPER_KAISER_BETA = 8.0


def taper_start_hertz(lower_frequency_hertz: float) -> float:
    """
    Where the challenge's taper starts, by PyCBC's iteration, with the leading-order chirp
    time, tau proportional to f^(-8/3), for PyCBC's IMRPhenomD chirp time: the ratio of
    durations is then the same for every mass, and the start is 16.86 Hz for a 20 Hz
    ``f_lower``, one or two 1 % steps below PyCBC's (17.03 Hz, or 17.20 Hz for some
    masses), where the window is under 2 % (the waveform oracle measures what that costs).
    """
    frequency, ratio = lower_frequency_hertz, 1.0
    while ratio < TAPER_DURATION_FACTOR:
        ratio = (frequency / lower_frequency_hertz) ** (-8.0 / 3.0)
        frequency *= TAPER_FREQUENCY_STEP
    return frequency


@_in_x64
@functools.partial(jax.jit, static_argnames=("approximant", "num_samples", "sample_rate_hertz", "f_ref"))
def _challenge_polarisations(mass1, mass2, distance, inclination, phase, taper_start, taper_end, merger_seconds,
                             approximant, num_samples, sample_rate_hertz, f_ref):
    """
    Plus and cross polarisations, (N, 2, num_samples), of ripple's ``approximant``,
    conditioned as PyCBC conditions the challenge's: zero below ``taper_start``, the rising
    half of a Kaiser window up to ``taper_end``, whole above; the model's own t = 0 at
    ``merger_seconds`` into the buffer.
    """
    frequencies = jnp.linspace(0.0, sample_rate_hertz / 2.0, num_samples // 2 + 1, dtype=jnp.float64)
    no_spin = jnp.zeros(jnp.shape(mass1) + (3,))
    parameters = _source_parameters(mass1, mass2, no_spin, no_spin, jnp.zeros_like(mass1),
                                    jnp.zeros_like(mass1), distance, phase, inclination)
    # The Kaiser window's rising half: 1 / I0(beta) at the start, 1 at the end.
    x = jnp.clip((taper_end - frequencies) / (taper_end - taper_start), 0.0, 1.0)
    window = i0(TAPER_KAISER_BETA * jnp.sqrt(1.0 - x ** 2)) / i0(TAPER_KAISER_BETA)
    below = frequencies < taper_start

    def one(source):
        plus, cross = _frequency_domain_polarisations(approximant, frequencies, source, f_ref, merger_seconds)
        # jnp.where, not a product, so that no non-finite model value below the start leaks.
        return jnp.stack([jnp.fft.irfft(jnp.where(below, 0.0, h * window), n=num_samples)
                          for h in (plus, cross)]) * sample_rate_hertz

    return jax.vmap(one)(parameters)


@dataclass(frozen=True)
class Injector:
    """
    How a dataset's sources become strain, as the challenge's do (generate_data.py's
    ``make_injections``: PyCBC's ``InjectionSet.apply``): ``approximant`` (ripple's), whole
    from ``lower_frequency_hertz`` and conditioned below it as PyCBC's ``get_td_waveform``
    conditions a frequency-domain model (``taper_start_hertz``), its phase referred to
    ``reference_frequency_hertz``; the model's t = 0 at the source's ``tc`` at the Earth's
    centre, projected onto each detector at that GPS time and moved to its arrival.

    This is not gravyflow's own CBC conditioning (``waveforms.cbc``, whose taper is wider and
    whose merger is the envelope's peak): the challenge's data is defined by PyCBC's, and
    its band starts at 15 Hz, inside gravyflow's taper.

    Each waveform is made on a buffer of twice ``window_seconds`` and the last
    ``window_seconds`` kept, with t = 0 at the kept window's centre sample.
    """

    approximant: Approximant = Approximant.IMRPhenomXAS
    lower_frequency_hertz: float = 20.0
    reference_frequency_hertz: float = 20.0
    window_seconds: float = 32.0

    #: Waveforms made at once: one shape, so one compilation.
    BATCH = 16

    def check_holds(self, population: Population) -> None:
        """
        Refuse a window too short for the population's longest waveform: the lightest
        binary's leading-order time to merger from the taper's start must leave a quarter of
        the half window spare, for the higher orders and the taper's own spread.
        """
        lightest = population.mass_msun[0]
        longest = float(calc_duration_from_f_min(lightest, lightest, taper_start_hertz(self.lower_frequency_hertz)))
        if longest > 0.75 * self.window_seconds / 2.0:
            raise ValueError(
                f"A {lightest} + {lightest} solar-mass waveform runs {longest:.1f} s from its taper's start to "
                f"its merger: more than three quarters of half the {self.window_seconds} s window.")

    def polarisations(self, sources: Dict[str, np.ndarray], sample_rate_hertz: float) -> np.ndarray:
        """``BATCH`` sources' plus and cross polarisations, (BATCH, 2, window samples), float64."""
        from gravyflow.src.rng import platform

        kept = int(self.window_seconds * sample_rate_hertz)
        with jax.default_device(platform.cbc_device()):
            buffer = _challenge_polarisations(
                sources["mass1"], sources["mass2"], sources["distance"], sources["inclination"],
                sources["coa_phase"], taper_start_hertz(self.lower_frequency_hertz), self.lower_frequency_hertz,
                2.0 * self.window_seconds - (kept - kept // 2) / sample_rate_hertz,
                approximant=self.approximant, num_samples=2 * kept, sample_rate_hertz=sample_rate_hertz,
                f_ref=self.reference_frequency_hertz)
        return np.asarray(buffer)[..., -kept:]

    def strain(self, sources: Dict[str, np.ndarray], network: Network, sample_rate_hertz: float
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Each source's strain on each detector, (N, D, K) float32 for K = window samples, and
        the absolute sample of each window's first sample, (N,) int64.
        """
        count = len(sources["tc"])
        samples = int(self.window_seconds * sample_rate_hertz)
        strains = []
        for first in range(0, count, self.BATCH):
            # The last batch is filled out with its last source, and the fill dropped.
            rows = np.minimum(np.arange(first, first + self.BATCH), count - 1)
            batch = {name: values[rows] for name, values in sources.items()}
            angles = [batch[name].astype(np.float32) for name in ("ra", "dec", "polarization")]
            antenna = np.asarray(network.get_antenna_pattern(*angles, gps_time_seconds=batch["tc"]))
            delay = np.asarray(network.get_time_delay(*angles[:2], gps_time_seconds=batch["tc"]))
            projected = np.einsum("ndp,npk->ndk", antenna, self.polarisations(batch, sample_rate_hertz))
            # t = 0 is on the window's centre sample; that sample is put on the one nearest
            # tc, and the rest of tc, with each detector's delay, is a shift.
            centre = np.rint(batch["tc"] * sample_rate_hertz)
            shift = (batch["tc"] - centre / sample_rate_hertz)[:, None] + delay
            shifted = np.asarray(shift_waveform(projected.astype(np.float32), sample_rate_hertz,
                                                shift.astype(np.float32)))
            strains.append(shifted[: count - first])
        if not strains:
            return np.zeros((0, len(network.ifos), samples), np.float32), np.zeros(0, np.int64)
        return np.concatenate(strains), self.window_starts(sources["tc"], sample_rate_hertz)

    def window_starts(self, tc: np.ndarray, sample_rate_hertz: float) -> np.ndarray:
        """The absolute sample each source's window starts on: its centre on the sample nearest tc."""
        return np.rint(tc * sample_rate_hertz).astype(np.int64) - int(self.window_seconds * sample_rate_hertz) // 2


@dataclass(frozen=True)
class DatasetSpec:
    """What makes one MLGWSC-1 dataset: its noise, its population, its schedule and its injector."""

    noise: StationaryNoise
    population: Population = Population()
    schedule: Schedule = Schedule()
    injector: Injector = Injector()
    #: No injection within this of a segment's edges (make_injections' padding_start and _end).
    padding_seconds: float = 30.0

    def __post_init__(self):
        self.injector.check_holds(self.population)


#: The datasets gravyflow makes, by the challenge's number.
DATASETS: Dict[int, DatasetSpec] = {
    1: DatasetSpec(StationaryNoise(NoiseProfile.ZERO_DET_HIGH_POWER, sample_rate_hertz=2048.0,
                                   low_frequency_hertz=15.0, filter_seconds=128.0)),
}


@dataclass(frozen=True)
class MLGWSC1Stream:
    """
    One of MLGWSC-1's datasets, streamed: every sample a function of ``seed`` and its GPS
    time (``fingerprint`` names the stream).

    Args:
        dataset: The challenge's dataset number; gravyflow makes dataset 1.
        seed: The root of its random streams (``gf.SeedTree``; None is ``gf.Defaults.seed``).
        start_offset_seconds: The seconds of the challenge's time skipped first
            (generate_data.py's --start-offset).
        duration_seconds: The seconds it lasts (--duration; 30 days).
    """

    dataset: int = 1
    seed: Optional[int] = None
    start_offset_seconds: int = 0
    duration_seconds: int = mlgwsc1.DURATION_SECONDS
    _readers: Dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"gravyflow makes MLGWSC-1's dataset{'s' if len(DATASETS) > 1 else ''} "
                             f"{', '.join(map(str, DATASETS))}, not dataset {self.dataset}.")

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    @property
    def detectors(self) -> Tuple[IFO, ...]:
        return (IFO.H1, IFO.L1)

    @property
    def sample_rate_hertz(self) -> float:
        return self.spec.noise.sample_rate_hertz

    @property
    def mode(self) -> str:
        """How it relates to real strain: simulated, so no reservation applies."""
        return SIMULATED

    @cached_property
    def tree(self) -> SeedTree:
        """Its random streams: the root's, under the dataset's own name."""
        return SeedTree(self.seed).sub(f"dataset/mlgwsc1-ds{self.dataset}")

    @cached_property
    def segments(self) -> np.ndarray:
        """Its segments, (S, 2) integer GPS seconds."""
        return mlgwsc1.restrict_segments(self.start_offset_seconds, self.duration_seconds)

    @cached_property
    def schedule(self) -> np.ndarray:
        """
        Every source the challenge's span schedules, a structured array of
        ``INJECTION_FIELDS`` sorted by ``tc``: the same for any offset and duration, as the
        challenge's one injection file is.
        """
        start, end = mlgwsc1.segments()[0, 0], mlgwsc1.segments()[-1, 1]
        times = self.spec.schedule.times(self.tree, start, end)
        sources = {"tc": times, **self.spec.population.draw(self.tree, len(times))}
        table = np.empty(len(times), dtype=[(name, np.float64) for name in INJECTION_FIELDS])
        for name in INJECTION_FIELDS:
            table[name] = sources[name]
        return table

    @cached_property
    def injections(self) -> np.ndarray:
        """The sources injected: those ``padding_seconds`` or more inside one of its segments."""
        tc, padding = self.schedule["tc"], self.spec.padding_seconds
        inside = np.zeros(len(tc), bool)
        for start, end in self.segments:
            inside |= (tc >= start + padding) & (tc <= end - padding)
        return self.schedule[inside]

    def read(self, start: float, end: float, *, injections: bool = True) -> SearchSegment:
        """
        The strain from GPS ``start`` to ``end``, inside one of its segments and on its
        sample grid: the foreground (noise and injections), or with ``injections=False``
        the background (the same noise alone).
        """
        rate = self.sample_rate_hertz
        first, last = start * rate, end * rate
        if first != int(first) or last != int(last) or not last > first:
            raise ValueError(f"A read is a span on the {rate:g} Hz sample grid, with its end after its start; "
                             f"got {start} to {end}.")
        if not np.any((self.segments[:, 0] <= start) & (end <= self.segments[:, 1])):
            raise ValueError(f"{start} to {end} is not inside one of MLGWSC-1 dataset {self.dataset}'s segments.")
        first, count = int(first), int(last - first)
        strain = np.stack([self._reader(ifo).read(first, count) for ifo in self.detectors])
        if injections:
            strain += self._signals(first, count)
        return SearchSegment(strain, tuple(ifo.name for ifo in self.detectors), start, rate)

    def chunks(self, seconds: float, *, injections: bool = True) -> Iterator[SearchSegment]:
        """Every segment, read in order in pieces of ``seconds`` (the last of each shorter)."""
        for start, end in self.segments.tolist():
            for piece in np.arange(start, end, seconds):
                yield self.read(float(piece), float(min(piece + seconds, end)), injections=injections)

    def fingerprint(self) -> str:
        """The sha256 of everything that decides its samples."""
        return config_fingerprint({
            "stream_version": STREAM_VERSION, "dataset": self.dataset, "root": self.tree.root,
            "prefix": self.tree.prefix, "start_offset_seconds": self.start_offset_seconds,
            "duration_seconds": self.duration_seconds, "spec": self.spec,
        })

    def metadata(self) -> dict:
        """What a result scored on it records: the fingerprint, mode, seed, extent and libraries."""
        return {
            "dataset": f"mlgwsc1-ds{self.dataset}", "fingerprint": self.fingerprint(), "mode": self.mode,
            "seed": self.tree.metadata(), "segments": len(self.segments),
            "seconds": int(np.sum(np.diff(self.segments, axis=1))), "injections": len(self.injections),
            "libraries": library_versions(),
        }

    def _reader(self, ifo: IFO):
        if ifo not in self._readers:
            self._readers[ifo] = self.spec.noise.reader(self.tree, SPLIT, ifo)
        return self._readers[ifo]

    def _signals(self, first: int, count: int) -> np.ndarray:
        """The injections' strain over samples ``first`` to ``first + count``, (D, count)."""
        rate, injector = self.sample_rate_hertz, self.spec.injector
        window = int(injector.window_seconds * rate)
        starts = injector.window_starts(self.injections["tc"], rate)
        overlapping = self.injections[(starts < first + count) & (starts + window > first)]
        out = np.zeros((len(self.detectors), count))
        if len(overlapping) == 0:
            return out
        sources = {name: overlapping[name] for name in INJECTION_FIELDS}
        strains, starts = injector.strain(sources, self._network, rate)
        for strain, start in zip(strains, starts.tolist()):
            lo, hi = max(start, first), min(start + strain.shape[-1], first + count)
            if hi > lo:
                out[:, lo - first:hi - first] += strain[:, lo - start:hi - start]
        return out

    @cached_property
    def _network(self) -> Network:
        return Network(list(self.detectors))
