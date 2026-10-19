"""
Hermetic ground for the leakage tests.

The suite's own guard (``tests/conftest.py``) already gives every test temporary
``HOME``, ``XDG_CACHE_HOME``, ``GRAVYFLOW_CACHE_DIR`` and working directory,
refuses and records every internet connection, switches proxies off, and fails the
run if this process writes into the package or the user cache. On top of it, for
this package (unless the run passes ``--network``):

- connections to this machine are refused too, and every name lookup of another
  host, so no socket opens at all, a proxy on this machine included;
- a test that causes a refused connection it did not declare (through the
  ``refusals`` fixture) fails, since the library would have zero-filled it silently.

``world`` installs a fake world (see ``fakes`` and ``fixtures``).
"""

from __future__ import annotations

from typing import List

import pytest

from gravyflow.tests.conftest import refuse_internet
from gravyflow.tests.test_public_sources import offline_public_sources  # noqa: F401  (offline GWOSC and Zenodo)


@pytest.fixture(scope="package", autouse=True)
def hermetic(suite_guard):
    """The package's stricter socket guard; yields the suite's guard (None with --network)."""
    if not suite_guard.offline:
        yield None
        return
    with pytest.MonkeyPatch.context() as mp:
        refuse_internet(mp, suite_guard.refusals, loopback=False)
        yield suite_guard


class Refusals:
    """A test's declaration that it expects the guard to refuse connections."""

    def __init__(self, guard):
        self.guard = guard
        self.start = len(guard.refusals)

    @property
    def seen(self) -> List[str]:
        return self.guard.refusals[self.start:]


@pytest.fixture
def refusals(hermetic):
    """Declare that this test expects refused connections; read them from ``.seen``."""
    if hermetic is None:
        pytest.fail("This test needs the offline guard; run it without --network.")
    return Refusals(hermetic)


@pytest.fixture(autouse=True)
def no_undeclared_refusals(request, hermetic):
    """A test that triggers a refusal it did not declare fails."""
    if hermetic is None:
        yield
        return
    start = len(hermetic.refusals)
    yield
    if "refusals" in request.fixturenames:
        return
    seen = hermetic.refusals[start:]
    if seen:
        pytest.fail(f"{len(seen)} undeclared network refusal(s), e.g. {seen[:3]}")


@pytest.fixture
def world(tmp_path):
    """
    ``world(fixture)`` installs a fixture layout for the length of a ``with`` block
    and yields its Harness: ``with world(fx.straddle("train", "test")) as harness:``.
    """
    from .fakes import installed

    counter = {"n": 0}

    def install(fixture, cache_dir=None, strain=True):
        counter["n"] += 1
        return installed(fixture, tmp_path / f"world{counter['n']}", cache_dir, strain)

    return install
