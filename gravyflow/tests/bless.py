"""
Bless a test's golden: ``python -m gravyflow.tests.bless <test-id>``.

Runs exactly one test (its pytest node id, such as
``gravyflow/tests/test_utils.py::test_seeded_dataset_batch_matches_its_golden``)
with blessing on, under the suite's usual guards, so it reaches no network. The
test's ``assert_matches_golden`` calls collect their arrays instead of
comparing them; after the session, each golden is written under ``GOLDENS``
with its sidecar: the generating test, its data source and seed, the array
names, dtypes and shapes, the gravyflow commit (and whether the tree was
dirty), the package versions and the date. Commit the golden on its own, as
``bless(<test>): <reason>``, so review reads its diff.
"""
import datetime
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from gravyflow.tests import _helpers

#: The packages a golden's numbers depend on.
VERSIONED_PACKAGES = ("gravyflow", "numpy", "scipy", "jax", "jaxlib", "keras", "h5py")

REPOSITORY = Path(__file__).resolve().parents[2]


class _OneTest:
    """Refuses a selection of other than one test, and turns blessing on for it."""

    def pytest_collection_modifyitems(self, items):
        if len(items) != 1:
            raise pytest.UsageError(
                f"bless takes exactly one test; the selection collected {len(items)}."
            )
        os.environ[_helpers.BLESS_ENVIRONMENT_VARIABLE] = items[0].nodeid


def _git(*arguments) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), *arguments],
        capture_output=True, text=True, check=True, env=environment,
    ).stdout.strip()


def _versions() -> dict:
    versions = {}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main(arguments) -> int:
    if len(arguments) != 1:
        print("usage: python -m gravyflow.tests.bless <test-id>", file=sys.stderr)
        return 2
    status = pytest.main([arguments[0], "-q", "-p", "no:cacheprovider"], plugins=[_OneTest()])
    os.environ.pop(_helpers.BLESS_ENVIRONMENT_VARIABLE, None)
    if status != 0:
        print(f"BLESS REFUSED: the test did not pass (pytest status {int(status)}).", file=sys.stderr)
        return 1
    if not _helpers.pending_blessings:
        print("BLESS REFUSED: the test ran no assert_matches_golden (or was skipped).", file=sys.stderr)
        return 1

    commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    date = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for blessing in _helpers.pending_blessings:
        arrays = blessing["arrays"]
        _helpers.write_golden(blessing["path"], arrays, {
            "test": blessing["test"],
            "source": blessing["source"],
            "seed": blessing["seed"],
            "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                       for name, value in arrays.items()},
            "commit": commit,
            "dirty": dirty,
            "versions": _versions(),
            "date": date,
        })
        print(f"BLESSED {blessing['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
