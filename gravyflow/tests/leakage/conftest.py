"""
Hermetic ground for the leakage tests.

For the whole package (unless the run passes ``--network``):

- every AF_INET and AF_INET6 connection, and every name lookup of a remote host, is
  refused and recorded; a test that causes a refusal it did not declare (through the
  ``refusals`` fixture) fails;
- ``HOME``, ``XDG_CACHE_HOME``, ``GRAVYFLOW_CACHE_DIR`` and the working directory
  are temporary, so no user cache or ``./generator_data`` is read or written;
- the installed package tree is snapshotted (relative path, size, mtime) and must
  be unchanged when the package's tests finish;
- ``gf.Defaults`` is restored after every test.

The fixtures for the fake world are ``world`` (install a fixture layout) and the
layout builders in ``fixtures``.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import gravyflow as gf

_LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


class SocketGuard:
    """Refuses and records remote connections and name lookups."""

    def __init__(self):
        self.refusals: List[str] = []
        self._connect = socket.socket.connect
        self._connect_ex = socket.socket.connect_ex
        self._getaddrinfo = socket.getaddrinfo

    def _refuse(self, what: str):
        self.refusals.append(what)
        raise ConnectionRefusedError(f"Leakage tests are offline: refused {what}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        guard = self

        def connect(sock, address, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                guard._refuse(f"connect to {address!r}")
            return guard._connect(sock, address, *args, **kwargs)

        def connect_ex(sock, address, *args, **kwargs):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                guard._refuse(f"connect_ex to {address!r}")
            return guard._connect_ex(sock, address, *args, **kwargs)

        def getaddrinfo(host, *args, **kwargs):
            if host not in _LOCAL_HOSTS:
                guard._refuse(f"name lookup of {host!r}")
            return guard._getaddrinfo(host, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def package_snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    """{relative path: (size, mtime_ns)} of every file under ``root`` but bytecode."""
    snapshot = {}
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or not path.is_file():
            continue
        stat = path.stat()
        snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def snapshot_changes(before: Dict, after: Dict) -> List[str]:
    changes = [f"added {p}" for p in sorted(set(after) - set(before))]
    changes += [f"removed {p}" for p in sorted(set(before) - set(after))]
    changes += [f"changed {p}" for p in sorted(set(before) & set(after)) if before[p] != after[p]]
    return changes


@pytest.fixture(scope="package", autouse=True)
def hermetic(request, tmp_path_factory):
    """The package-wide guard: offline, temporary roots, an unchanged package tree."""
    if request.config.getoption("--network"):
        yield None
        return
    root = tmp_path_factory.mktemp("leakage-home")
    before = package_snapshot(Path(gf.PATH))
    guard = SocketGuard()
    previous_cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(root / "home"))
        mp.setenv("XDG_CACHE_HOME", str(root / "xdg-cache"))
        mp.setenv("GRAVYFLOW_CACHE_DIR", str(root / "gravyflow-cache"))
        (root / "work").mkdir()
        os.chdir(root / "work")
        guard.install(mp)
        try:
            yield guard
        finally:
            os.chdir(previous_cwd)
    changes = snapshot_changes(before, package_snapshot(Path(gf.PATH)))
    if changes:
        pytest.fail(f"The leakage tests changed the installed package: {changes[:10]}")


class Refusals:
    """A test's declaration that it expects the guard to refuse connections."""

    def __init__(self, guard: SocketGuard):
        self.guard = guard
        self.start = len(guard.refusals)
        self.expected = True

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
