"""
The validator's timing: what a model costs to run over a stream, and what it cost to train.

- **Throughput** (`stream_throughput`): the compute-seconds the model takes per second of
  strain, sliding along a fixed segment at its stride (every window cut, conditioned and
  scored), timed whole over repeated calls after a cold one: median and tail
  (`throughput_summary`).
- **Latency** (`stream_latency`): data to trigger, in a streaming replay one stride of new
  data at a time: the compute of each step (median, p90, p99, max over at least
  `utils.timing.MIN_STEPS` steps) plus the algorithmic wait, the data a trigger needs after
  its time: half the window's onsource, then the clustering's ``cluster_seconds`` (a
  trigger is the loudest within that time, so it is known only once that time has passed).
- **A-B-B-A** (`abba_rounds`): two callables timed in rounds of A, B, B, A, after one cold
  call of each, so a host's drift falls equally on both: each arm's walls, their median and
  tail, and the ratio of medians, which is a ratio on one machine by construction.
- **No ratio across hardware** (`timing_ratio`): timings are compared only when taken on the
  same hardware at the same thread count.
- **Training cost** (`TrainingCost`): measured where we train (`measure_training`), or
  quoted, with its source, for released weights.

Every timing records the machine (`utils.timing.machine_description`), the thread count the
caller declares (the harness cannot set a backend's; for XLA's CPU backend ``XLA_FLAGS`` and
the process's affinity do) and the host's load at the start and end, flagged when busy
(`utils.timing.HostLoad`). ``clock`` is the wall clock (``time.perf_counter``); tests give a
fake one.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from gravyflow.src.utils.timing import (
    LOAD_PER_CORE_LIMIT,
    MIN_STEPS,
    HostLoad,
    Timing,
    TooFewSteps,
    _check_common,
    _LoadWatch,
    _machine,
    _require,
    _wait_for,
    summarise,
)
from .stream import StreamGeometry


def _timed(call: Callable, clock: Callable[[], float], *arguments) -> float:
    """The wall of one whole call by ``clock``, including the wait for an asynchronous result."""
    began = clock()
    _wait_for(call(*arguments))
    return clock() - began


# ------------------------------------------------------------ throughput

def stream_throughput(score: Callable[[np.ndarray, np.ndarray], Any], geometry: StreamGeometry, strain: np.ndarray, *,
                      threads: int, repeats: int = 5, batch_size: int = 64, scope: str = "end_to_end",
                      clock: Callable[[], float] = time.perf_counter, hardware: Optional[str] = None,
                      max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> Timing:
    """
    The model slid along ``strain`` (detectors, samples) at its stride, every window cut and
    scored by ``score(onsource, offsource)`` in batches of ``batch_size``: one cold call, then
    ``repeats`` timed calls (`utils.timing.Timing`, mode ``batch``) over the strain's duration.
    """
    _check_common(threads, scope)
    _require(isinstance(repeats, int) and repeats >= 1, f"repeats must be an int >= 1, got {repeats!r}")
    strain = np.asarray(strain)
    count = geometry.num_windows(strain.shape[-1])
    _require(count >= 1, f"A strain of {strain.shape[-1]} samples holds no window of the model's.")

    def run():
        out = []
        for first in range(0, count, int(batch_size)):
            index = np.arange(first, min(first + int(batch_size), count))
            out.append(score(*geometry.gather(strain, index * geometry.stride_samples)))
        return out

    with _LoadWatch(threads, max_load_per_core) as watch:
        cold = _timed(run, clock)
        walls = tuple(_timed(run, clock) for _ in range(repeats))
    return Timing("batch", scope, walls, cold, strain.shape[-1] / geometry.sample_rate_hertz, threads, None,
                  _machine(hardware), watch.result)


def throughput_summary(timing: Timing) -> Dict[str, float]:
    """A batch timing's compute-seconds per second of strain: median, p90, p99 and max over its calls."""
    return summarise([wall / timing.strain_seconds for wall in timing.walls])


# ------------------------------------------------------------- latency

def algorithmic_wait_seconds(geometry: StreamGeometry, cluster_seconds: float) -> float:
    """The data a trigger needs after its time: half the window's (padded) onsource, then the clustering's wait."""
    return geometry.window.num_onsource_samples / 2.0 / geometry.sample_rate_hertz + float(cluster_seconds)


def stream_latency(score: Callable[[np.ndarray, np.ndarray], Any], geometry: StreamGeometry, strain: np.ndarray, *,
                   cluster_seconds: float, threads: int, scope: str = "end_to_end",
                   clock: Callable[[], float] = time.perf_counter, hardware: Optional[str] = None,
                   max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> Timing:
    """
    Data-to-trigger latency in a streaming replay of ``strain``: each step one stride of new
    data, its window cut and scored alone; the first step is the cold start. The record
    (`utils.timing.Timing`, mode ``streaming``) adds `algorithmic_wait_seconds` to every
    statistic of the step walls.

    Raises:
        TooFewSteps: the strain gives fewer than `MIN_STEPS` steps after the cold one.
    """
    _check_common(threads, scope)
    strain = np.asarray(strain)
    count = geometry.num_windows(strain.shape[-1])
    if count - 1 < MIN_STEPS:
        raise TooFewSteps(f"the replay has {max(count - 1, 0)} steps after the cold start, and a latency needs at "
                          f"least {MIN_STEPS}: lengthen the strain")

    def step(k):
        return score(*geometry.gather(strain, [k * geometry.stride_samples]))

    with _LoadWatch(threads, max_load_per_core) as watch:
        cold = _timed(step, clock, 0)
        walls = tuple(_timed(step, clock, k) for k in range(1, count))
    return Timing("streaming", scope, walls, cold, geometry.stride_seconds * len(walls), threads,
                  algorithmic_wait_seconds(geometry, cluster_seconds), _machine(hardware), watch.result)


# ---------------------------------------------------------------- A-B-B-A

@dataclass(frozen=True)
class AbbaRounds:
    """
    Rounds of A, B, B, A: the order run (``order``), each arm's walls in the order run, the
    cold call of each (``cold``: A's, B's), the thread count, machine and host load.
    """
    order: Tuple[str, ...]
    a_walls: Tuple[float, ...]
    b_walls: Tuple[float, ...]
    cold: Tuple[float, float]
    threads: int
    machine: dict
    load: HostLoad

    def _walls(self, arm: str) -> Tuple[float, ...]:
        if arm not in ("a", "b"):
            raise ValueError(f"arm is 'a' or 'b', got {arm!r}")
        return self.a_walls if arm == "a" else self.b_walls

    def summary(self, arm: str) -> Dict[str, float]:
        """An arm's median, p90, p99 and max wall."""
        return summarise(self._walls(arm))

    def mean(self, arm: str) -> float:
        return statistics.fmean(self._walls(arm))

    @property
    def ratio(self) -> float:
        """B's median wall over A's."""
        return self.summary("b")["median"] / self.summary("a")["median"]

    @property
    def spread_a(self) -> float:
        """How far A's wall moves between its runs on this host in this state: its range."""
        return max(self.a_walls) - min(self.a_walls)

    @property
    def distinguishable(self) -> bool:
        """Whether the arms' means differ by more than A's spread: a difference inside it is no difference."""
        return abs(self.mean("b") - self.mean("a")) > self.spread_a

    def to_dict(self) -> Dict:
        return {"order": list(self.order), "a_walls": list(self.a_walls), "b_walls": list(self.b_walls),
                "cold_seconds": {"a": self.cold[0], "b": self.cold[1]}, "summary_a": self.summary("a"),
                "summary_b": self.summary("b"), "mean_a": self.mean("a"), "mean_b": self.mean("b"),
                "ratio_b_to_a": self.ratio, "spread_a": self.spread_a, "distinguishable": self.distinguishable,
                "threads": self.threads, "machine": self.machine, "host_load": self.load.to_dict()}


def abba_rounds(a: Callable[[], Any], b: Callable[[], Any], *, rounds: int, threads: int,
                clock: Callable[[], float] = time.perf_counter, hardware: Optional[str] = None,
                max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> AbbaRounds:
    """
    Time ``a`` and ``b`` (no-argument callables doing the same job on the same input) in
    ``rounds`` rounds of A, B, B, A, after one cold call of each, recorded apart.
    """
    _check_common(threads)
    _require(isinstance(rounds, int) and rounds >= 1, f"rounds must be an int >= 1, got {rounds!r}")
    order = ("A", "B", "B", "A") * rounds
    walls = {"A": [], "B": []}
    with _LoadWatch(threads, max_load_per_core) as watch:
        cold = (_timed(a, clock), _timed(b, clock))
        for arm in order:
            walls[arm].append(_timed(a if arm == "A" else b, clock))
    return AbbaRounds(order, tuple(walls["A"]), tuple(walls["B"]), cold, threads, _machine(hardware), watch.result)


# ------------------------------------------------- comparing two timings

@dataclass(frozen=True)
class TimingSummary:
    """A timing's statistics (``median`` at least), with its mode, thread count, machine and host load."""
    mode: str
    statistics: Dict[str, float]
    threads: int
    machine: dict
    load: HostLoad


def timing_ratio(a: TimingSummary, b: TimingSummary, statistic: str = "median") -> float:
    """
    ``b``'s statistic over ``a``'s. Refused across different hardware, thread counts or
    modes: a ratio of timings is a ratio on one machine.
    """
    if a.machine.get("hardware") != b.machine.get("hardware"):
        raise ValueError(f"No ratio across different hardware: {a.machine.get('hardware')!r} and "
                         f"{b.machine.get('hardware')!r}.")
    if a.threads != b.threads:
        raise ValueError(f"No ratio across thread counts: {a.threads} and {b.threads}.")
    if a.mode != b.mode:
        raise ValueError(f"No ratio across modes: {a.mode} and {b.mode}.")
    return b.statistics[statistic] / a.statistics[statistic]


def timing_record(throughput: Optional[Timing] = None, latency: Optional[Timing] = None,
                  abba: Optional[AbbaRounds] = None) -> Dict:
    """
    A validator's timing record (`Validator.timing`), as plain JSON types: the throughput
    (its walls, machine, load and runtime records, and the median and tail per second of
    strain), the latency and the A-B-B-A rounds, each None when not measured; the thread
    count and machine of the first given.
    """
    first = next((t for t in (throughput, latency, abba) if t is not None), None)
    if first is None:
        raise ValueError("Give at least one timing.")
    record = {"threads": first.threads, "machine": first.machine, "throughput": None, "latency": None, "abba": None}
    if throughput is not None:
        record["throughput"] = dict(throughput.to_dict(), summary=throughput_summary(throughput))
    if latency is not None:
        record["latency"] = dict(latency.to_dict(), compute=summarise(latency.walls))
    if abba is not None:
        record["abba"] = abba.to_dict()
    return record


# ------------------------------------------------------------ training cost

TRAINING_COST_KINDS = ("measured", "quoted")


@dataclass(frozen=True)
class TrainingCost:
    """
    What a model cost to train: ``measured`` where we train (wall seconds on ``hardware``),
    or ``quoted`` for released weights, in the units its ``source`` gives, never without the
    source.
    """
    kind: str
    value: float
    units: str
    hardware: Optional[str] = None
    source: Optional[str] = None
    epochs: Optional[int] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if self.kind not in TRAINING_COST_KINDS:
            raise ValueError(f"A training cost's kind is one of {TRAINING_COST_KINDS}, got {self.kind!r}.")
        if self.kind == "quoted" and not (self.source and str(self.source).strip()):
            raise ValueError("A quoted training cost needs its source.")
        if not (np.isfinite(self.value) and self.value >= 0):
            raise ValueError(f"A training cost is a finite value >= 0, got {self.value!r}.")

    @classmethod
    def quoted(cls, value: float, units: str, source: str, notes: Optional[str] = None) -> "TrainingCost":
        return cls("quoted", float(value), str(units), source=source, notes=notes)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingCost":
        return cls(**data)


def measure_training(train: Callable[[], Any], *, clock: Callable[[], float] = time.perf_counter,
                     hardware: Optional[str] = None, epochs: Optional[int] = None) -> Tuple[Any, TrainingCost]:
    """Run ``train`` (no arguments) and return its result and its measured cost: the wall seconds of the call."""
    began = clock()
    result = train()
    _wait_for(result)
    wall = clock() - began
    return result, TrainingCost("measured", float(wall), "s", hardware=_machine(hardware)["hardware"], epochs=epochs)


__all__ = [
    "AbbaRounds",
    "TimingSummary",
    "TrainingCost",
    "abba_rounds",
    "algorithmic_wait_seconds",
    "measure_training",
    "stream_latency",
    "stream_throughput",
    "throughput_summary",
    "timing_ratio",
    "timing_record",
]
