"""
Timing a model or a search: throughput, latency, and A-B-B-A comparisons.

Three measurements, for any callable (a Keras model's forward pass, a matched-filter search,
an external search), each by the wall time of the whole call:

- `time_batch`: throughput, the compute-seconds a call takes per second of strain it
  processes, over a fixed segment.
- `time_streaming`: latency, the time from a chunk of data arriving to the step that scores
  it returning (the per-step compute time: median, p90, p99 and max over at least `MIN_STEPS`
  steps), plus the *algorithmic wait* the caller declares: the data the algorithm needs
  after the event, which no amount of compute removes.
- `compare_abba`: two callables compared by the A-B-B-A rule. A single run of each says
  nothing about a difference, since a host's speed drifts. Run A, B, B, A, each its own call,
  report all four walls, both pair means and the spread of the two A runs; a difference
  inside that spread is no difference.

A *compute-second* here is a wall second of the whole call on the machine described, at the
thread count the caller declares: not CPU-seconds, and not comparable across hardware. The
harness cannot set the thread count of a backend (for XLA's CPU backend, ``XLA_FLAGS`` and the
process's affinity do), so it records the count the caller declares. A callable that returns
asynchronously (a JAX array) is waited for: the wall ends when its result is ready.

The *cold start* is the first call of the callable in this process (compilation, caches, page
faults); it is recorded apart and never enters a statistic. A timing taken while the host was
busy is flagged (`HostLoad`), never silently kept as quiet.

Each `Timing` gives its numbers as runtime records (`Timing.records`) in the fields of a model
catalogue's runtime schema: ``quantity``, ``statistic``, ``value``, ``units``, ``normalised``
(compute-seconds per second of strain for throughput, seconds for latency), ``hardware``,
``scope`` (``network`` or ``end_to_end``), ``mode`` (``batch`` or ``streaming``) and
``grade``, here always ``measured``.
"""

import math
import os
import platform as host_platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

#: The fewest steps a streaming replay times (after its cold first step): a p99 or a max
#: over fewer is not a tail.
MIN_STEPS = 1000

#: What a timing's scope may be: the model's forward pass alone, or the whole pipeline
#: (whitening, I/O, clustering).
SCOPES = ("network", "end_to_end")

#: A host is busy when its load average exceeds this many runnable processes per core, beyond
#: the timed work's own threads.
LOAD_PER_CORE_LIMIT = 0.25

#: The `grade` of every record this module makes.
GRADE = "measured"

BATCH_UNITS = "compute-seconds per second of strain"
LATENCY_UNITS = "s"
LATENCY_STATISTICS = ("median", "p90", "p99", "max")


_EMPTY = object()


class TooFewSteps(ValueError):
    """A streaming replay had fewer than `MIN_STEPS` steps after its cold first step."""


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    """
    The nearest-rank q-th percentile (0 < q <= 100) of ``values``: an observed value, never an
    interpolation, so a tail statistic is a step that happened.
    """
    if not values:
        raise ValueError("percentile of no values")
    if not 0 < q <= 100:
        raise ValueError(f"q must be in (0, 100], got {q}")
    ordered = sorted(values)
    rank = max(math.ceil(q / 100 * len(ordered)), 1)
    return ordered[rank - 1]


def summarise(values: Sequence[float]) -> dict:
    """The median, p90, p99 and max of ``values``."""
    return {
        "median": statistics.median(values),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


# ---------------------------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------------------------

def _load_average() -> Optional[float]:
    """The host's one-minute load average, or None where the platform gives none."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class HostLoad:
    """
    The host's one-minute load average at the start and the end of a timing, and whether the
    timing was taken under load.

    The timed work raises the average itself, by up to its own thread count, so a sample counts
    as busy only beyond that: ``loaded`` is True when a sample exceeds
    ``threads + max_per_core * cpu_count``. A host without a load average (``start`` is None)
    is ``loaded=None``: unmeasured, never taken as quiet.
    """
    start: Optional[float]
    end: Optional[float]
    cpu_count: int
    threads: int
    max_per_core: float = LOAD_PER_CORE_LIMIT

    @property
    def loaded(self) -> Optional[bool]:
        samples = [sample for sample in (self.start, self.end) if sample is not None]
        if len(samples) < 2:
            return None
        return max(samples) > self.threads + self.max_per_core * self.cpu_count

    def note(self) -> Optional[str]:
        """A sentence for a record's notes when the host was busy or unmeasured; else None."""
        if self.loaded is None:
            return "Host load could not be measured."
        if self.loaded:
            return (f"Timed under host load (load average {self.start:.2f} at start, {self.end:.2f} "
                    f"at end, on {self.cpu_count} cores).")
        return None

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "cpu_count": self.cpu_count,
                "threads": self.threads, "max_per_core": self.max_per_core, "loaded": self.loaded}


class _LoadWatch:
    """Samples the load average on entry and exit."""

    def __init__(self, threads: int, max_per_core: float):
        self.threads = threads
        self.max_per_core = max_per_core
        self.start: Optional[float] = None

    def __enter__(self):
        self.start = _load_average()
        return self

    def __exit__(self, *exc):
        self.result = HostLoad(self.start, _load_average(), os.cpu_count() or 1,
                               self.threads, self.max_per_core)
        return False


def _cpu_model() -> str:
    """The CPU's marketing name where the platform gives one, else its architecture."""
    system = host_platform.system()
    try:
        if system == "Darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5, check=True).stdout.strip()
            if name:
                return name
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return host_platform.processor() or host_platform.machine()


def machine_description() -> dict:
    """
    What a timing was taken on: the CPU model and core count, the operating system, Python, and
    gravyflow's platform record (JAX's backend and device, ``XLA_FLAGS``, matmul precision).
    ``hardware`` is the one-line form a catalogue's runtime record carries.
    """
    from gravyflow.src.rng.platform import platform_record

    record = platform_record()
    cpu, cores = _cpu_model(), os.cpu_count() or 1
    if record["backend"] == "cpu":
        hardware = f"{cpu}, {cores} cores (CPU)"
    else:
        import jax

        hardware = f"{len(jax.devices())}x {record['device_kind']} ({record['backend']}) on {cpu}"
    return {
        "hardware": hardware,
        "cpu_model": cpu,
        "cpu_count": cores,
        "system": f"{host_platform.system()} {host_platform.release()}",
        "python": host_platform.python_version(),
        "platform": record,
    }


# ---------------------------------------------------------------------------------------------
# One timed call
# ---------------------------------------------------------------------------------------------

def _wait_for(result: Any) -> None:
    """Block until an asynchronous result (a JAX array, or a container of them) is ready."""
    ready = getattr(result, "block_until_ready", None)
    if ready is not None:
        ready()
    elif isinstance(result, (list, tuple)):
        for item in result:
            _wait_for(item)
    elif isinstance(result, dict):
        for item in result.values():
            _wait_for(item)


def _timed(call: Callable, *arguments) -> float:
    """The wall seconds of one whole call, including the wait for an asynchronous result."""
    began = time.perf_counter()
    _wait_for(call(*arguments))
    return time.perf_counter() - began


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_common(threads: int, scope: Optional[str] = None) -> None:
    _require(isinstance(threads, int) and not isinstance(threads, bool) and threads >= 1,
             f"threads is the thread count the caller ran the work at, an int >= 1, got {threads!r}")
    if scope is not None:
        _require(scope in SCOPES, f"scope is one of {SCOPES}, got {scope!r}")


# ---------------------------------------------------------------------------------------------
# The record of a timing
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Timing:
    """
    One timing: what was measured, on what, and how.

    Attributes:
        mode: ``batch`` or ``streaming``.
        scope: ``network`` or ``end_to_end``.
        walls: the wall seconds of each timed call (batch) or step (streaming), in order,
            excluding the cold start.
        cold_seconds: the wall of the first call or step, apart from ``walls``.
        strain_seconds: the strain each batch call processed, or in streaming the strain the
            replay covered (None when the hop was not declared).
        threads: the thread count the caller declared.
        algorithmic_wait_seconds: streaming only, the data the algorithm needs after the event.
        machine: `machine_description`, or the caller's ``hardware`` string alone.
        load: the host's load at the start and end, and whether it was busy.
    """
    mode: str
    scope: str
    walls: Tuple[float, ...]
    cold_seconds: float
    strain_seconds: Optional[float]
    threads: int
    algorithmic_wait_seconds: Optional[float]
    machine: dict
    load: HostLoad

    @property
    def hardware(self) -> str:
        return self.machine["hardware"]

    def records(self) -> List[dict]:
        """
        The timing as catalogue runtime records, one per statistic: throughput as the median
        (``single`` for one call) of compute-seconds per second of strain; latency as the
        median, p90, p99 and max of the step time plus the algorithmic wait, in seconds.
        """
        if self.mode == "batch":
            if len(self.walls) == 1:
                statistics_ = {"single": self.walls[0]}
            else:
                statistics_ = {"median": statistics.median(self.walls)}
            rows = [(statistic, "inference_cost", wall / self.strain_seconds, BATCH_UNITS)
                    for statistic, wall in statistics_.items()]
        else:
            summary = summarise(self.walls)
            rows = [(statistic, "latency", summary[statistic] + self.algorithmic_wait_seconds,
                     LATENCY_UNITS) for statistic in LATENCY_STATISTICS]
        return [{
            "quantity": quantity,
            "statistic": statistic,
            "value": value,
            "units": units,
            "normalised": value,
            "hardware": self.hardware,
            "scope": self.scope,
            "mode": self.mode,
            "grade": GRADE,
            "notes": self.load.note(),
        } for statistic, quantity, value, units in rows]

    def to_dict(self) -> dict:
        """The whole record as plain JSON types: the timing's details and its runtime records."""
        return {
            "mode": self.mode,
            "scope": self.scope,
            "walls": list(self.walls),
            "cold_seconds": self.cold_seconds,
            "strain_seconds": self.strain_seconds,
            "threads": self.threads,
            "algorithmic_wait_seconds": self.algorithmic_wait_seconds,
            "machine": self.machine,
            "host_load": self.load.to_dict(),
            "runtime": self.records(),
        }


def _machine(hardware: Optional[str]) -> dict:
    """The machine description, or, for a caller naming its hardware, that name alone."""
    return machine_description() if hardware is None else {"hardware": hardware}


# ---------------------------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------------------------

def time_batch(run: Callable[[], Any], *, strain_seconds: float, threads: int, scope: str,
               repeats: int = 5, hardware: Optional[str] = None,
               max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> Timing:
    """
    Time ``run``, a call that processes a fixed segment of ``strain_seconds`` of strain, in
    batch mode: once cold, then ``repeats`` times warm, each the wall of the whole call.

    Args:
        run: a callable taking no arguments, over the same segment every call. It returns the
            model's output (waited for if asynchronous) so the wall covers the work.
        strain_seconds: the duration of strain one call processes, per detector network.
        threads: the thread count the work runs at, declared by the caller.
        scope: ``network`` (the forward pass alone) or ``end_to_end``.
        repeats: the warm calls timed (the cold first call is extra).
        hardware: the hardware string for the record, instead of `machine_description`'s.
        max_load_per_core: the busy-host threshold (`HostLoad`).
    """
    _check_common(threads, scope)
    _require(strain_seconds > 0, f"strain_seconds must be positive, got {strain_seconds}")
    _require(isinstance(repeats, int) and repeats >= 1, f"repeats must be an int >= 1, got {repeats!r}")
    with _LoadWatch(threads, max_load_per_core) as watch:
        cold = _timed(run)
        walls = tuple(_timed(run) for _ in range(repeats))
    return Timing("batch", scope, walls, cold, float(strain_seconds), threads, None,
                  _machine(hardware), watch.result)


# ---------------------------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------------------------

def time_streaming(step: Callable[[Any], Any], stream: Iterable, *, algorithmic_wait_seconds: float,
                   threads: int, scope: str, step_strain_seconds: Optional[float] = None,
                   hardware: Optional[str] = None,
                   max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> Timing:
    """
    Time ``step`` over a replay of ``stream``, the data-to-trigger latency of a streaming
    search: the wall of each step, then the median, p90, p99 and max of them plus the
    algorithmic wait. The first chunk's step is the cold start and joins no statistic; the
    replay must give at least `MIN_STEPS` chunks after it.

    The stream's own cost (reading or generating a chunk) is outside the timing: only
    ``step(chunk)``, and the wait for its result, is timed.

    Args:
        step: a callable taking one chunk, scoring it and returning its output (waited for if
            asynchronous). A stateful search carries its state in the callable.
        stream: the chunks, in order.
        algorithmic_wait_seconds: the data the algorithm needs after the event it detects (for
            example, the span of a merger's signal the model must see), declared by the caller.
        threads: the thread count the work runs at, declared by the caller.
        scope: ``network`` or ``end_to_end``.
        step_strain_seconds: the strain each chunk adds (the hop), where known; the record's
            ``strain_seconds`` is then the replay's total.

    Raises:
        TooFewSteps: the replay is shorter than `MIN_STEPS` steps after the cold one.
    """
    _check_common(threads, scope)
    _require(algorithmic_wait_seconds >= 0,
             f"algorithmic_wait_seconds must be >= 0, got {algorithmic_wait_seconds}")
    _require(step_strain_seconds is None or step_strain_seconds > 0,
             f"step_strain_seconds must be positive, got {step_strain_seconds}")
    chunks = iter(stream)
    with _LoadWatch(threads, max_load_per_core) as watch:
        first = next(chunks, _EMPTY)
        if first is _EMPTY:
            raise TooFewSteps(f"the replay is empty; a latency needs at least {MIN_STEPS} steps")
        cold = _timed(step, first)
        walls = [_timed(step, chunk) for chunk in chunks]
    if len(walls) < MIN_STEPS:
        raise TooFewSteps(f"the replay has {len(walls)} steps after the cold start, and a latency "
                          f"needs at least {MIN_STEPS}")
    strain = None if step_strain_seconds is None else step_strain_seconds * len(walls)
    return Timing("streaming", scope, tuple(walls), cold, strain, threads,
                  float(algorithmic_wait_seconds), _machine(hardware), watch.result)


# ---------------------------------------------------------------------------------------------
# A-B-B-A
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AbbaComparison:
    """
    Two callables compared by the A-B-B-A rule: the four walls in the order run, both pair
    means and the spread of the two A runs.

    Attributes:
        a_walls: A's two walls, first and last.
        b_walls: B's two walls, in the middle.
        mean_a, mean_b: the pair means.
        spread_a: the absolute difference of A's two walls: how far one callable's wall moves
            between runs on this host in this state.
        difference: ``mean_b - mean_a``.
        distinguishable: whether ``|difference|`` exceeds ``spread_a``. A difference inside the
            spread is no difference.
        cold_a, cold_b: the wall of each callable's first call, run before the four and apart
            from them.
        threads, machine, load: as in `Timing`.
    """
    a_walls: Tuple[float, float]
    b_walls: Tuple[float, float]
    cold_a: float
    cold_b: float
    threads: int
    machine: dict
    load: HostLoad

    @property
    def mean_a(self) -> float:
        return statistics.fmean(self.a_walls)

    @property
    def mean_b(self) -> float:
        return statistics.fmean(self.b_walls)

    @property
    def spread_a(self) -> float:
        return abs(self.a_walls[0] - self.a_walls[1])

    @property
    def difference(self) -> float:
        return self.mean_b - self.mean_a

    @property
    def distinguishable(self) -> bool:
        return abs(self.difference) > self.spread_a

    @property
    def walls_in_order(self) -> Tuple[float, float, float, float]:
        """The four walls as run: A, B, B, A."""
        return (self.a_walls[0], self.b_walls[0], self.b_walls[1], self.a_walls[1])

    def to_dict(self) -> dict:
        return {
            "order": ["A", "B", "B", "A"],
            "walls": list(self.walls_in_order),
            "a_walls": list(self.a_walls),
            "b_walls": list(self.b_walls),
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "spread_a": self.spread_a,
            "difference": self.difference,
            "distinguishable": self.distinguishable,
            "cold_seconds": {"a": self.cold_a, "b": self.cold_b},
            "threads": self.threads,
            "machine": self.machine,
            "host_load": self.load.to_dict(),
        }


def compare_abba(a: Callable[[], Any], b: Callable[[], Any], *, threads: int,
                 hardware: Optional[str] = None,
                 max_load_per_core: float = LOAD_PER_CORE_LIMIT) -> AbbaComparison:
    """
    Run ``a`` and ``b`` as A, B, B, A, each run its own whole call, after one cold call of each
    (recorded apart), and report the four walls, both pair means and A's spread.

    The two must do the same job on the same input, so that a wall difference is a difference
    between the callables. Run it on a quiet host: the result carries the host's load, and a
    busy host is flagged (`HostLoad.loaded`).

    Args:
        a, b: callables taking no arguments, returning their output (waited for if
            asynchronous).
        threads: the thread count both run at, declared by the caller.
    """
    _check_common(threads)
    with _LoadWatch(threads, max_load_per_core) as watch:
        cold_a, cold_b = _timed(a), _timed(b)
        a1, b1, b2, a2 = _timed(a), _timed(b), _timed(b), _timed(a)
    return AbbaComparison((a1, a2), (b1, b2), cold_a, cold_b, threads, _machine(hardware),
                          watch.result)
