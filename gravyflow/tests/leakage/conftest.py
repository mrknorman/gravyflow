"""
Hermetic ground for the leakage tests.

The suite's own guard (``tests/conftest.py``) already refuses and records every
internet connection, unsets proxies, gives the suite a temporary
``GRAVYFLOW_CACHE_DIR``, and fails the run if the package or ``~/.gravyflow``
changes. On top of it, for this package (unless the run passes ``--network``):

- ``HOME``, ``XDG_CACHE_HOME``, ``GRAVYFLOW_CACHE_DIR`` and the working directory
  are temporary, so no user file or ``./generator_data`` is read or written;
- a test that causes a refused connection it did not declare (through the
  ``refusals`` fixture) fails, since the library would have zero-filled it silently;
- ``gf.Defaults`` is restored after every test.

``world`` installs a fake world (see ``fakes`` and ``fixtures``).
"""

from __future__ import annotations

import os
from typing import Dict, List

import pytest

import gravyflow as gf


@pytest.fixture(scope="package", autouse=True)
def hermetic(request, tmp_path_factory, offline_unless_network_option):
    """The package's temporary roots; yields the suite's offline guard (None with --network)."""
    if offline_unless_network_option is None:
        yield None
        return
    root = tmp_path_factory.mktemp("leakage-home")
    previous_cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(root / "home"))
        mp.setenv("XDG_CACHE_HOME", str(root / "xdg-cache"))
        mp.setenv("GRAVYFLOW_CACHE_DIR", str(root / "gravyflow-cache"))
        (root / "work").mkdir()
        os.chdir(root / "work")
        try:
            yield offline_unless_network_option
        finally:
            os.chdir(previous_cwd)


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


def _defaults_state() -> Dict:
    return {
        key: value for key, value in vars(gf.Defaults).items()
        if not key.startswith("__") and not isinstance(value, (staticmethod, classmethod))
        and not callable(value)
    }


@pytest.fixture(autouse=True)
def restore_defaults():
    """``gf.Defaults`` is global state: restore it after every test."""
    state = _defaults_state()
    yield
    for key, value in state.items():
        setattr(gf.Defaults, key, value)


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
