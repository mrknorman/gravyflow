"""
The runner: an external model's adapter, run in its own environment, under a deadline.

External models are written for frameworks that cannot share a process with
gravyflow's JAX and Keras (a PyTorch release, an older TensorFlow), so each runs in an
environment of its own, built from a lock file its card pins by sha256
(``AdapterEnvironment``), and gravyflow imports none of their code. ``ExternalRunner``
mirrors the adapter's own methods across that boundary: for one segment it

1. checks the segment against the card's input contract, and the environment's lock
   against the card's;
2. fetches the card's weights through gravyflow's one fetch (``gf.fetch``), checked
   against their pinned sha256;
3. makes a temporary directory, leaves the request and the segment's strain in it,
   and runs ``python -I -B child.py <directory>`` with the environment's interpreter
   (isolated: no user site, no ``PYTHON*`` variables, no bytecode written), with the
   directory as its working directory and no input;
4. holds it to the deadline: at expiry, and whenever it ends, every process left in
   its session is sent TERM, then KILL after ``GRACE_SECONDS`` (a process that starts
   a session of its own escapes this, as it escapes any session's owner);
5. reads the result back, and removes the directory.

The child refuses an adapter that is not installed (``AdapterMissing``) or not the
card's (``AdapterMismatch``) before the model runs. A search's result names the card's
mode and fingerprint (``ExternalResult``): its statistic is the published one, never
rescored.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from gravyflow.src import resources
from gravyflow.src.dataset.store.tile import file_sha256
from gravyflow.src.detection.triggers import Triggers

from . import child
from .card import ExternalCard, OutputContract
from .protocol import SearchSegment

logger = logging.getLogger(__name__)

#: The adapter's process: run by path in its environment, which has no gravyflow.
CHILD = Path(child.__file__)

#: Seconds between TERM and KILL for what is left of an adapter's session.
GRACE_SECONDS = 5.0

#: Lines of an adapter's log a failure quotes.
LOG_TAIL_LINES = 40


class ExternalRunFailed(RuntimeError):
    """An external model's run did not return its result."""


class ExternalDeadlineExpired(ExternalRunFailed):
    """An external model's run outlived its deadline, and its session was reaped."""


class AdapterMissing(ExternalRunFailed):
    """The environment has no adapter by the name the card gives."""


class AdapterMismatch(ExternalRunFailed):
    """The environment is not the card's: another lock, or another adapter release, or
    an adapter that is not an ``ExternalSearch``."""


@dataclass(frozen=True)
class AdapterEnvironment:
    """
    Where an adapter runs.

    Args:
        python: The environment's interpreter.
        lock: The lock file it was built from; its sha256 is the card's
            ``environment_lock``.
        paths: Directories put first on the adapter's ``sys.path``, for an adapter
            installed into a directory (``pip install --target``) rather than into
            the environment.

    Each is kept as an absolute path, since the adapter runs in a directory of its own.
    """

    python: Path
    lock: Path
    paths: Tuple[Path, ...] = ()

    def __post_init__(self):
        absolute = lambda path: Path(path).expanduser().absolute()  # noqa: E731
        object.__setattr__(self, "python", absolute(self.python))
        object.__setattr__(self, "lock", absolute(self.lock))
        object.__setattr__(self, "paths", tuple(absolute(p) for p in self.paths))


@dataclass(frozen=True, eq=False)
class ExternalResult:
    """
    An external model's triggers on one segment, and what made them.

    Args:
        name: The card's model name.
        mode: The card's mode (``AS_PUBLISHED``).
        card_fingerprint: The card's fingerprint.
        output: The statistic the triggers carry: its signal column and direction.
        triggers: The search's triggers, as its published code returns them.
    """

    name: str
    mode: str
    card_fingerprint: str
    output: OutputContract
    triggers: Triggers


class ExternalRunner:
    """
    Runs ``card``'s adapter in ``environment``, each call in a fresh process held to
    ``deadline_seconds``. ``scratch`` is where its temporary directories are made
    (default: the system's).
    """

    def __init__(self, card: ExternalCard, environment: AdapterEnvironment, deadline_seconds: float,
                 scratch: Optional[Union[str, Path]] = None):
        if not isinstance(card, ExternalCard):
            raise TypeError(f"An ExternalRunner runs an ExternalCard, got {type(card).__name__}.")
        if isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float)) \
                or not deadline_seconds > 0:
            raise ValueError(f"A run's deadline is a number of seconds above zero, got {deadline_seconds!r}.")
        self.card = card
        self.environment = environment
        self.deadline_seconds = float(deadline_seconds)
        self.scratch = None if scratch is None else Path(scratch)

    def search(self, segment: SearchSegment) -> ExternalResult:
        """The adapter's search over ``segment``: its triggers, with the card's mode and fingerprint."""
        def read(directory: Path) -> Triggers:
            with np.load(directory / child.TRIGGERS, allow_pickle=False) as arrays:
                try:
                    return Triggers(*(arrays[name] for name in child.TRIGGER_FIELDS))
                except ValueError as error:
                    raise ExternalRunFailed(f"{self.card.name}'s search returned triggers gravyflow "
                                            f"refuses: {error}") from error

        triggers = self._run(child.SEARCH, segment, read)
        return ExternalResult(self.card.name, self.card.mode, self.card.fingerprint, self.card.output, triggers)

    def preprocess(self, segment: SearchSegment) -> np.ndarray:
        """The input the adapter's model takes for ``segment``."""
        def read(directory: Path) -> np.ndarray:
            return np.load(directory / child.INPUT, allow_pickle=False)

        return self._run(child.PREPROCESS, segment, read)

    def _run(self, action: str, segment: SearchSegment, read):
        card, environment = self.card, self.environment
        if not isinstance(segment, SearchSegment):
            raise TypeError(f"An external model searches a SearchSegment, got {type(segment).__name__}.")
        if segment.detectors != card.input.detectors or segment.sample_rate_hertz != card.input.sample_rate_hertz:
            raise ValueError(
                f"{card.name} reads {card.input.detectors} at {card.input.sample_rate_hertz} Hz, and the segment "
                f"holds {segment.detectors} at {segment.sample_rate_hertz} Hz.")
        for name, path in (("interpreter", environment.python), ("lock file", environment.lock)):
            if not path.is_file():
                raise FileNotFoundError(f"{card.name}'s environment has no {name} at {path}.")
        with open(environment.lock, "rb") as handle:
            lock = file_sha256(handle)
        if lock != card.environment_lock:
            raise AdapterMismatch(
                f"{card.name} runs in the environment locked by sha256 {card.environment_lock}, and "
                f"{environment.lock} has sha256 {lock}.")
        weights = {w.file.name: str(resources.fetch(w.file)) for w in card.weights}
        request = {
            "action": action, "adapter": card.adapter, "adapter_version": card.adapter_version,
            "paths": [str(p) for p in environment.paths], "weights": weights,
            "detectors": list(segment.detectors), "t0": segment.t0, "sample_rate_hertz": segment.sample_rate_hertz,
        }
        with tempfile.TemporaryDirectory(prefix="gravyflow-external-", dir=self.scratch) as name:
            directory = Path(name)
            (directory / child.REQUEST).write_text(json.dumps(request))
            np.save(directory / child.STRAIN, segment.strain, allow_pickle=False)
            log = directory / "log.txt"
            status = _supervise([str(environment.python), "-I", "-B", str(CHILD), str(directory)],
                                directory, log, self.deadline_seconds)
            what = f"{card.name}'s {action} ({card.adapter} {card.adapter_version}, {environment.python})"
            if status is None:
                raise ExternalDeadlineExpired(
                    f"{what} outlived its {self.deadline_seconds:g} s deadline; its session was reaped.\n"
                    + _tail(log))
            outcome_file = directory / child.OUTCOME
            outcome = json.loads(outcome_file.read_text()) if outcome_file.is_file() else {}
            if "refused" in outcome:
                error = AdapterMissing if outcome["refused"] == child.MISSING else AdapterMismatch
                raise error(f"{what} was refused: {outcome['message']}")
            if status != 0:
                raise ExternalRunFailed(f"{what} exited with status {status}.\n" + _tail(log))
            return read(directory)


def _supervise(argv, cwd: Path, log: Path, deadline_seconds: float) -> Optional[int]:
    """
    Run ``argv`` in a session of its own, its output to ``log``; its exit status, or
    None when it outlived ``deadline_seconds``. Its session is reaped either way.
    """
    with open(log, "wb") as output:
        process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=deadline_seconds)
        except subprocess.TimeoutExpired:
            return None
        finally:
            _reap(process)


def _reap(process: subprocess.Popen) -> None:
    """
    TERM every process in ``process``'s session (its process group: ``process`` leads
    it), then KILL those left after ``GRACE_SECONDS``; a group still alive after KILL
    and another grace is named in the log and given up on.
    """
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        if not _signal_group(process.pid, signal_number) or _gone(process, GRACE_SECONDS):
            process.wait()  # its leader has exited, so this returns at once, and reaps it
            return
    logger.warning("Gave up on process group %d after KILL and %g s: it is still alive.",
                   process.pid, GRACE_SECONDS)


def _signal_group(group: int, signal_number: int) -> bool:
    """
    Send ``signal_number`` to the process group ``group``; False when it has no live
    process. Every process in it is this user's, so a refusal of permission means
    the same: macOS refuses to signal a group whose processes have all exited and
    wait to be reaped by their parents.
    """
    try:
        os.killpg(group, signal_number)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _gone(process: subprocess.Popen, seconds: float) -> bool:
    """Whether ``process``'s group empties within ``seconds``; ``process`` itself is reaped as it exits."""
    end = time.monotonic() + seconds
    while True:
        process.poll()
        if not _signal_group(process.pid, 0):
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.05)


def _tail(log: Path) -> str:
    lines = log.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
    return f"The last {len(lines)} lines of its log:\n" + "\n".join(lines)


__all__ = [
    "AdapterEnvironment", "AdapterMismatch", "AdapterMissing", "ExternalDeadlineExpired", "ExternalResult",
    "ExternalRunFailed", "ExternalRunner", "GRACE_SECONDS",
]
