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

import gravyflow

from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir
from gravyflow.tests.conftest import (
    PROXY_VARIABLES, WriteAudit, child_env, refuse_internet, tree_changes, tree_snapshot,
)

pytest_plugins = ["pytester"]  # the guards are proved by running tests under them

#: Test files that start Python without ``child_env``, and why the child still imports the tree under test.
CHILDREN_WITHOUT_CHILD_ENV = {
    "test_package.py": "runs in the checkout's root, the first entry of the path of ``python -c``",
}

#: Run in a fresh interpreter, where no event module is imported yet: the offline
#: catalogue is entered and left, and only then are the event lookups imported.
FIRST_IMPORT_INSIDE_THE_CATALOGUE = """
import sys
from gravyflow.tests.conftest import offline_event_catalogue
assert "gravyflow.src.dataset.features.segment_builders" not in sys.modules
with offline_event_catalogue():
    pass
from gravyflow.src.dataset.features import event, segment_builders
assert segment_builders.get_events_with_params is event.get_events_with_params, (
    "a patched event lookup outlived the offline catalogue")
print("RESTORED")
"""

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


def test_the_offline_catalogue_restores_every_event_lookup_it_patched():
    child = subprocess.run(
        [sys.executable, "-c", FIRST_IMPORT_INSIDE_THE_CATALOGUE],
        capture_output=True, text=True, timeout=300,
        env=child_env(),
    )
    assert child.returncode == 0 and child.stdout.strip() == "RESTORED", child.stderr[-2000:]


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


def test_write_audit_sees_a_write_through_a_symlink(tmp_path):
    watched, elsewhere = tmp_path / "watched", tmp_path / "elsewhere"
    watched.mkdir()
    elsewhere.mkdir()
    (elsewhere / "link").symlink_to(watched)
    with WriteAudit({"watched": watched}) as audit:
        (elsewhere / "link" / "file").write_bytes(b"1")
    assert (watched / "file").exists()
    assert [entry.split(" ")[0] for entry in audit.writes["watched"]] == ["open"], audit.writes


# --- A child process imports the tree under test --------------------------------------

def test_a_child_imports_the_tree_under_test():
    """
    The environment's editable install can point at another checkout of gravyflow (a
    worktree's gates would then judge the main checkout), so a child's ``child_env``
    puts the tree under test first. It runs from the suite's temporary working
    directory, where nothing else puts a gravyflow on the path.
    """
    child = subprocess.run(
        [sys.executable, "-c", "import gravyflow; print(gravyflow.__file__)"],
        capture_output=True, text=True, timeout=300, env=child_env(),
    )
    assert child.returncode == 0, child.stderr[-2000:]
    assert Path(child.stdout.strip().splitlines()[-1]).resolve() == Path(gravyflow.__file__).resolve()


def test_child_env_sets_and_removes_variables_and_keeps_the_callers_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    env = child_env(GRAVYFLOW_CACHE_DIR="/cache", PYTHONWARNINGS=None)
    assert env["GRAVYFLOW_CACHE_DIR"] == "/cache" and "PYTHONWARNINGS" not in env
    tree, *rest = env["PYTHONPATH"].split(os.pathsep)
    assert (Path(tree) / "gravyflow").resolve() == Path(gravyflow.__file__).resolve().parent
    assert rest == ["/somewhere/else"]
    assert os.environ["PYTHONWARNINGS"] == "ignore", "child_env changed this process's environment"


def test_every_test_that_starts_python_pins_the_tree_under_test():
    tests = Path(__file__).parent
    unpinned = sorted(
        path.name for path in tests.glob("test_*.py")
        if "sys.executable" in path.read_text() and "child_env" not in path.read_text()
        and path.name not in CHILDREN_WITHOUT_CHILD_ENV
    )
    assert not unpinned, f"these start Python without child_env: {unpinned}"
    stale = sorted(name for name in CHILDREN_WITHOUT_CHILD_ENV if "sys.executable" not in (tests / name).read_text())
    assert not stale, f"CHILDREN_WITHOUT_CHILD_ENV names files that start no Python: {stale}"


# --- gf.Defaults is restored after every test -----------------------------------------

def test_a_failing_test_cannot_leak_defaults(pytester):
    """A test that changes ``gf.Defaults`` and then fails leaves the next test the defaults it started with."""
    pytester.makeconftest("from gravyflow.tests.conftest import defaults_guard  # noqa: F401\n")
    pytester.makepyfile(test_leak="""
        import gravyflow as gf

        SEED, RATE = gf.Defaults.seed, gf.Defaults.sample_rate_hertz

        def test_a_changes_the_defaults_then_fails():
            gf.Defaults.set(seed=SEED + 1, sample_rate_hertz=2 * RATE)
            assert False, "fails after changing the defaults"

        def test_b_starts_from_the_defaults_test_a_started_from():
            assert (gf.Defaults.seed, gf.Defaults.sample_rate_hertz) == (SEED, RATE)
    """)
    pytester.runpytest_inprocess().assert_outcomes(failed=1, passed=1)
