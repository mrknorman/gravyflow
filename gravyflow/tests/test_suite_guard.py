"""
Self-tests of the suite's own guard (``conftest.py``): its temporary roots, its
offline refusal with proxies off, and the checks that the package and the user
cache are left alone.
"""

import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import requests

from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir
from gravyflow.tests.conftest import (
    PROXY_VARIABLES, WriteAudit, refuse_internet, tree_changes, tree_snapshot,
)

#: A documentation address (RFC 5737): never a real host.
REMOTE = "http://203.0.113.9/"


def test_the_suite_runs_in_temporary_roots(suite_guard):
    root = str(suite_guard.root)
    for name in ("HOME", "XDG_CACHE_HOME", "GRAVYFLOW_CACHE_DIR"):
        assert os.environ[name].startswith(root), name
    assert os.getcwd().startswith(root)
    assert str(Path.home()).startswith(root)
    assert str(user_cache_dir()).startswith(root)


def test_the_guard_switches_proxies_off_and_refuses_through_them(monkeypatch):
    """
    A proxy on this machine, set in the environment or in the system's settings
    (macOS), carries no request past the guard: each request is refused by name
    instead of going to the proxy.
    """
    proxy = "http://127.0.0.1:9"
    for name in PROXY_VARIABLES:
        monkeypatch.setenv(name, proxy)
    system = {"http": proxy, "https": proxy}
    monkeypatch.setattr(urllib.request, "getproxies_macosx_sysconf", lambda: system, raising=False)
    seen = []
    with pytest.MonkeyPatch.context() as guard:
        refuse_internet(guard, seen)
        assert not [name for name in PROXY_VARIABLES if name in os.environ]
        with pytest.raises(urllib.error.URLError):
            urllib.request.build_opener().open(REMOTE, timeout=5)
        with pytest.raises(requests.ConnectionError):
            requests.get(REMOTE, timeout=5)
    assert seen == ["connection to ('203.0.113.9', 80)"] * 2, seen


def test_tree_snapshot_sees_every_change_and_skips_vanished_files(tmp_path, monkeypatch):
    (tmp_path / "a.npy").write_bytes(b"1")
    (tmp_path / "__pycache__").mkdir()
    before = tree_snapshot(tmp_path)
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"ignored")
    assert tree_changes(before, tree_snapshot(tmp_path)) == []
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "cached_event_times.npy").write_bytes(b"2")
    (tmp_path / "a.npy").write_bytes(b"12")
    changes = tree_changes(before, tree_snapshot(tmp_path))
    assert changes == ["added res/cached_event_times.npy", "changed a.npy"]

    # A writer's temporary file, listed and then renamed away before it is read.
    (tmp_path / ".tmp-renamed").write_bytes(b"3")
    stat = Path.stat

    def vanishing(path, *args, **kwargs):
        if path.name == ".tmp-renamed":
            raise FileNotFoundError(path)
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", vanishing)
    assert ".tmp-renamed" not in tree_snapshot(tmp_path)


def test_write_audit_sees_this_process_writes_and_no_one_elses(tmp_path):
    watched, elsewhere = tmp_path / "watched", tmp_path / "elsewhere"
    watched.mkdir()
    elsewhere.mkdir()
    (watched / "old").write_bytes(b"0")
    with WriteAudit({"watched": watched}) as audit:
        (watched / "old").read_bytes()
        (elsewhere / "file").write_bytes(b"1")
        subprocess.run(
            [sys.executable, "-c", f"open({str(watched / 'other')!r}, 'w').write('x')"], check=True
        )
        assert audit.writes == {}, "a read, a write elsewhere or another process's write was seen"
        (watched / "made").mkdir()
        np.save(watched / "made" / "array.npy", np.zeros(1))
        os.replace(watched / "old", watched / "new")
        os.remove(watched / "new")
    assert (watched / "other").exists()
    events = [entry.split(" ")[0] for entry in audit.writes["watched"]]
    # A rename changes both of its paths.
    assert events == ["os.mkdir", "open", "os.rename", "os.rename", "os.remove"], audit.writes
    (watched / "after").write_bytes(b"2")
    assert len(audit.writes["watched"]) == 5, "the audit saw a write after it ended"
