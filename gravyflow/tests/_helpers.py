"""
The suite's shared derivations: tolerances, statistical bounds and goldens.

Every tolerance in the suite is of one stated kind: exact; representation
(``representation_tolerance``: a few machine epsilons of the data's scale);
statistical (``statistical_bound``: k standard deviations of the estimator);
the accuracy of an independent reference; or measured, as a last resort.

A golden is a committed HDF5 file under ``GOLDENS`` with a JSON sidecar of the
same name that records how it was made. ``assert_matches_golden`` compares a
test's arrays with one and fails, naming the path, when the golden or its
sidecar is missing. Goldens are written only by ``python -m
gravyflow.tests.bless <test-id>``, from fakes or seeded synthetic data.
"""
import json
import os
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np

#: The committed goldens, beside the package's other test data.
GOLDENS = Path(__file__).resolve().parents[1] / "res" / "tests" / "goldens"

#: The sources a golden may be blessed from: never real-data acquisition.
BLESSABLE_SOURCES = ("synthetic", "fake")

#: Set by ``gravyflow.tests.bless`` to the node id of the one test it blesses.
BLESS_ENVIRONMENT_VARIABLE = "GRAVYFLOW_BLESS"

#: What the sidecar of every golden records.
SIDECAR_FIELDS = ("test", "source", "seed", "arrays", "commit", "dirty", "versions", "date")

#: The goldens the running bless session collected, written by ``bless`` after
#: the session ends (the suite guard forbids writing the package from a test).
pending_blessings = []


def representation_tolerance(dtype, scale: float, c: float) -> float:
    """
    ``c * eps(dtype) * scale``: the rounding a computation of ``dtype`` values
    of magnitude ``scale`` accumulates. ``c`` comes from the operation: about
    sqrt(N) for a sum of N terms, log2(N) for an FFT of length N.
    """
    return float(c * np.finfo(dtype).eps * scale)


def statistical_bound(sigma: float, k: float = 4.0) -> float:
    """``k`` standard deviations of an estimator whose ``sigma`` the caller derives."""
    return float(k * sigma)


def effective_count(weights) -> float:
    """
    The number of equally weighted terms a weighted mean of independent terms
    averages as many as: ``(sum w)^2 / sum w^2``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(weights) ** 2 / np.sum(weights ** 2))


def sidecar_path(golden: Path) -> Path:
    """The sidecar of ``golden``: the same name with ``.json``."""
    return golden.with_suffix(".json")


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def assert_matches_golden(
        arrays: Mapping[str, np.ndarray],
        path: Path,
        *,
        c: Optional[float] = None,
        source: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
    """
    Compare ``arrays`` with the golden at ``path``, under ``GOLDENS``.

    Integer and boolean arrays must be equal. Floating arrays must agree within
    ``representation_tolerance(dtype, peak of the golden, c)``, so the tolerance
    follows the data's own scale. A missing golden or sidecar fails, naming its
    path, and is never written here: under ``gravyflow.tests.bless``, the
    blessed test's arrays are collected instead, with their ``source`` (one of
    ``BLESSABLE_SOURCES``) and, for synthetic data, the ``seed``.
    """
    path = Path(path)
    if not path.resolve().is_relative_to(GOLDENS):
        raise AssertionError(f"Golden {path} is not under {GOLDENS}.")
    arrays = {name: np.asarray(value) for name, value in arrays.items()}

    if os.environ.get(BLESS_ENVIRONMENT_VARIABLE) == _current_test() != "":
        if source not in BLESSABLE_SOURCES:
            raise AssertionError(
                f"Golden {path}: a golden is blessed only from {BLESSABLE_SOURCES} data; "
                f"the test declares source={source!r}."
            )
        if source == "synthetic" and seed is None:
            raise AssertionError(f"Golden {path}: synthetic data is blessed only with its seed.")
        pending_blessings.append(
            {"path": path, "arrays": arrays, "test": _current_test(), "source": source, "seed": seed}
        )
        return

    if not path.exists():
        raise AssertionError(
            f"Golden {path} is missing: bless it with python -m gravyflow.tests.bless <test-id>."
        )
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        raise AssertionError(f"Golden {path} has no sidecar {sidecar}: bless it again.")
    missing = set(SIDECAR_FIELDS) - set(json.loads(sidecar.read_text()))
    if missing:
        raise AssertionError(f"Sidecar {sidecar} lacks {sorted(missing)}: bless it again.")
    if c is None:
        raise AssertionError(f"Golden {path}: state the representation tolerance's c.")

    with h5py.File(path, "r") as golden:
        assert set(golden) == set(arrays), (
            f"Golden {path} holds {sorted(golden)}; the test gives {sorted(arrays)}."
        )
        for name, current in arrays.items():
            expected = golden[name][()]
            assert current.shape == expected.shape, (
                f"Golden {path}:{name} has shape {expected.shape}; the test gives {current.shape}."
            )
            if np.issubdtype(expected.dtype, np.inexact):
                scale = float(np.max(np.abs(expected), initial=0.0))
                np.testing.assert_allclose(
                    current, expected, rtol=0.0,
                    atol=representation_tolerance(expected.dtype, scale, c),
                    err_msg=f"Golden {path}:{name}",
                )
            else:
                np.testing.assert_array_equal(current, expected, err_msg=f"Golden {path}:{name}")


def write_golden(path: Path, arrays: Mapping[str, np.ndarray], provenance: Mapping) -> None:
    """Write ``arrays`` to the golden at ``path`` and ``provenance`` to its sidecar."""
    missing = set(SIDECAR_FIELDS) - set(provenance)
    if missing:
        raise ValueError(f"Provenance for {path} lacks {sorted(missing)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as golden:
        for name, value in arrays.items():
            golden.create_dataset(name, data=value)
    sidecar_path(path).write_text(json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n")
