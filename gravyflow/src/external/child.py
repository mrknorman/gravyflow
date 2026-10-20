"""
The process the runner starts in an adapter's environment, as
``python -I -B child.py <directory>``.

It reads the request the runner left in ``directory`` (``REQUEST``, and the
segment's strain, ``STRAIN``, or for ``SCORES`` the strains of several segments,
``STRAINS``), finds the adapter by its entry point, checks its version against the
card's, prepares it with the local paths of its weights, runs the one action asked
for, and leaves the result (``TRIGGERS``, ``INPUT`` or ``WINDOW_SCORES``) and its
outcome (``OUTCOME``) in the directory.

An adapter's environment holds its own framework and not gravyflow, so this file
imports only the standard library and numpy, loads ``protocol.py`` beside it by
path, and keeps to the Python 3.8 grammar (``ADAPTER_PYTHON``). Its failures to
find or accept an adapter are refusals, written to the outcome with their kind;
anything else the adapter raises ends the process with its traceback in the log.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np

#: The oldest Python an adapter's environment may run.
ADAPTER_PYTHON = (3, 8)

#: The files the runner and this process exchange, in the request's directory.
REQUEST, STRAIN, OUTCOME, TRIGGERS, INPUT = "request.json", "strain.npy", "outcome.json", "triggers.npz", "input.npy"
STRAINS, WINDOW_SCORES = "strains.npz", "scores.npz"

#: The actions a request may ask for, one per ``ExternalSearch`` method that takes a
#: segment; ``SCORES`` runs ``scores`` on each of several segments in one process.
SEARCH, PREPROCESS, SCORES = "search", "preprocess", "scores"

#: The arrays of one segment's window scores (``ExternalSearch.scores``).
SCORE_FIELDS = ("time", "stat")

#: The kinds of refusal: no adapter by the card's name, or one that is not the card's.
MISSING, MISMATCH = "missing", "mismatch"

#: The arrays of a search's triggers: ``detection.triggers.FIELDS``, which this file
#: cannot import (it needs h5py); a test holds the two equal.
TRIGGER_FIELDS = ("time", "stat", "var")


class Refusal(Exception):
    """The adapter the request names cannot be run: ``kind`` is MISSING or MISMATCH."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def main(argv=None) -> int:
    """Serve the request in the directory ``argv[0]``; 0 when served, 2 when refused."""
    directory = Path((sys.argv[1:] if argv is None else argv)[0])
    try:
        outcome = serve(json.loads((directory / REQUEST).read_text()), directory)
    except Refusal as refusal:
        _write_outcome(directory, {"refused": refusal.kind, "message": str(refusal)})
        print(f"Refused: {refusal}", file=sys.stderr)
        return 2
    _write_outcome(directory, outcome)
    return 0


def serve(request: dict, directory: Path) -> dict:
    """Run the request's action and write its result into ``directory``; the outcome."""
    if sys.version_info[:2] < ADAPTER_PYTHON:
        raise Refusal(MISMATCH, f"An adapter's environment runs Python {ADAPTER_PYTHON[0]}.{ADAPTER_PYTHON[1]} "
                                f"or later; {sys.executable} is {platform.python_version()}.")
    protocol = _load_protocol()
    sys.path[:0] = request["paths"]
    distribution, entry_point = find_adapter(request["adapter"], protocol.ADAPTER_GROUP)
    if distribution.version != request["adapter_version"]:
        raise Refusal(MISMATCH, f"The card names {request['adapter']!r} {request['adapter_version']}, and "
                                f"this environment's is {distribution.version} (from {distribution.metadata['Name']}).")
    adapter = entry_point.load()()
    if not isinstance(adapter, protocol.ExternalSearch):
        raise Refusal(MISMATCH, f"{entry_point.value} is not an ExternalSearch: it needs prepare, "
                                "preprocess and search.")
    weights = {name: Path(path) for name, path in request["weights"].items()}

    def fetch(name: str) -> Path:
        try:
            return weights[name]
        except KeyError:
            raise KeyError(f"The card names no weights file {name!r}; it names {sorted(weights)}.") from None

    adapter.prepare(fetch)
    detectors, rate = tuple(request["detectors"]), request["sample_rate_hertz"]
    served = {"adapter_version": distribution.version, "python": platform.python_version()}
    if request["action"] == SCORES:
        if not callable(getattr(adapter, "scores", None)):
            raise Refusal(MISMATCH, f"{entry_point.value} has no scores: it gives no window's statistic before "
                                    "its threshold, which the validator reads.")
        arrays = {}
        with np.load(directory / STRAINS) as strains:
            for k, t0 in enumerate(request["t0s"]):
                found = adapter.scores(protocol.SearchSegment(strains[f"strain_{k}"], detectors, t0, rate))
                missing = [name for name in SCORE_FIELDS if not hasattr(found, name)]
                if missing:
                    raise Refusal(MISMATCH, f"{entry_point.value}'s scores returned a {type(found).__name__}, "
                                            f"which has no {missing}: window scores have {SCORE_FIELDS}.")
                arrays.update({f"{name}_{k}": np.asarray(getattr(found, name)) for name in SCORE_FIELDS})
        np.savez(directory / WINDOW_SCORES, **arrays)
        return served
    segment = protocol.SearchSegment(np.load(directory / STRAIN), detectors, request["t0"], rate)
    if request["action"] == SEARCH:
        found = adapter.search(segment)
        missing = [name for name in TRIGGER_FIELDS if not hasattr(found, name)]
        if missing:
            raise Refusal(MISMATCH, f"{entry_point.value}'s search returned a {type(found).__name__}, "
                                    f"which has no {missing}: triggers have {TRIGGER_FIELDS}.")
        np.savez(directory / TRIGGERS, **{name: np.asarray(getattr(found, name)) for name in TRIGGER_FIELDS})
    elif request["action"] == PREPROCESS:
        np.save(directory / INPUT, np.asarray(adapter.preprocess(segment)), allow_pickle=False)
    else:
        raise ValueError(f"No action {request['action']!r}; the actions are {SEARCH!r}, {PREPROCESS!r} and "
                         f"{SCORES!r}.")
    return served


def find_adapter(name: str, group: str):
    """
    The distribution and entry point that publish the adapter ``name`` in ``group``,
    searched in ``sys.path`` order (a distribution found twice counts once, where
    it is found first). Refuses, by name, none and more than one distribution.
    """
    import importlib.metadata

    found, seen = [], set()
    for distribution in importlib.metadata.distributions():
        key = (distribution.metadata["Name"] or "").lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        found += [(distribution, ep) for ep in distribution.entry_points if ep.group == group and ep.name == name]
    if not found:
        raise Refusal(MISSING, f"No adapter named {name!r} in the entry-point group {group!r} of the "
                               f"environment of {sys.executable}: install the package that publishes it there.")
    if len(found) > 1:
        publishers = sorted(f"{d.metadata['Name']} {d.version}" for d, _ in found)
        raise Refusal(MISMATCH, f"More than one distribution publishes the adapter {name!r}: {publishers}.")
    return found[0]


def _load_protocol():
    """``protocol.py`` beside this file, loaded by path: gravyflow is not importable here."""
    spec = importlib.util.spec_from_file_location("gravyflow_external_protocol", Path(__file__).with_name("protocol.py"))
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up by name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_outcome(directory: Path, outcome: dict) -> None:
    (directory / OUTCOME).write_text(json.dumps(outcome))


if __name__ == "__main__":
    sys.exit(main())
