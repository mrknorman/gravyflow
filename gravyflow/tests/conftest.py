import os
import site
import glob
import sys

# Attempt to find nvidia libraries and set LD_LIBRARY_PATH
# This is required in conftest.py to ensure pytest finds the libraries
# before JAX is initialized by any plugins or test collection.
try:
    site_packages = site.getsitepackages()[0]
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    
    if os.path.exists(nvidia_dir):
        libs = glob.glob(os.path.join(nvidia_dir, '*/lib'))
        path_to_add = ":".join(libs)
        
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if path_to_add not in current_ld:
            os.environ['LD_LIBRARY_PATH'] = f"{current_ld}:{path_to_add}"
except Exception as e:
    pass

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"
import gravyflow as gf
from gravyflow.src.resources import PACKAGE

import warnings
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

def _global_rng_state():
    """NumPy's, Python's and Keras' global random states, comparable with ==."""
    import random
    import numpy as np
    from keras.src.backend.common import global_state

    keras_generator = global_state.get_global_attribute("global_seed_generator")
    keras_state = None if keras_generator is None else np.asarray(keras_generator.state.value).tolist()
    numpy_state = np.random.get_state()
    return (
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        random.getstate(),
        keras_state,
    )


@pytest.fixture(autouse=True)
def global_rng_guard(request):
    """
    The library draws only from its seed tree, never from a global RNG: fail a
    test whose run changed NumPy's, Python's or Keras' global random state.

    A test that draws its own data from a global RNG (or builds a Keras model,
    whose initialisers draw from Keras' global seed) declares it with
    ``@pytest.mark.global_rng``: it starts from a fixed global state instead,
    and is not checked. The mark is per test, never per module, so the guard
    still covers the library code every other test runs.
    """
    import gravyflow as gf
    if request.node.get_closest_marker("global_rng") is not None:
        gf.set_random_seeds(gf.Defaults.seed)
        yield
        return
    before = _global_rng_state()
    yield
    assert _global_rng_state() == before, (
        "The global random state changed during the test: library code must draw "
        "from a gf.SeedTree stream, never a global RNG."
    )


def defaults_state() -> dict:
    """``gf.Defaults``' values, comparable with ==."""
    return {
        key: value for key, value in vars(gf.Defaults).items()
        if not key.startswith("__") and not callable(value) and not isinstance(value, classmethod)
    }


@contextmanager
def restoring_defaults():
    """Leave ``gf.Defaults`` as it was found, whatever the block does or raises."""
    state = defaults_state()
    try:
        yield
    finally:
        for key, value in state.items():
            setattr(gf.Defaults, key, value)


@pytest.fixture(autouse=True)
def defaults_guard():
    """
    ``gf.Defaults`` is global state: restore it after every test, so a test that
    changes it, and then fails before it can undo the change, leaves no test after it
    the wrong defaults (an order-dependent failure far from its cause).
    """
    with restoring_defaults():
        yield


def child_env(**overrides) -> dict:
    """
    The environment for a child Python process, which imports the gravyflow tree under
    test: its root is first on ``PYTHONPATH``, ahead of the environment's editable
    install, which can point at another checkout (so a worktree's gate would judge the
    main checkout's library). ``overrides`` set variables; a ``None`` removes one.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path(PACKAGE).resolve().parent), env.get("PYTHONPATH")])
    )
    for name, value in overrides.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return env


@pytest.fixture
def tree():
    """A seed tree for tests that need one."""
    import gravyflow as gf
    return gf.SeedTree(gf.Defaults.seed)

def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store_true", default=False,
        help="Enable plotting of results"
    )
    parser.addoption(
        "--network", action="store_true", default=False,
        help="Lift the suite's internet refusal, for the online unit (gravyflow/tests/online), "
             "whose tests query GWOSC and Zenodo"
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "global_rng: the test draws its own data from a global RNG (not guarded)"
    )
    # Past acquisition (where data gaps are expected), a NaN or inf the sanitiser
    # replaces is a failure upstream: tests fail on it. Added here rather than in
    # the ini, which pytest reads before the root conftest.py sets XLA's flags:
    # naming the class there imports gravyflow, which starts JAX's backend.
    config.addinivalue_line(
        "filterwarnings",
        "error::gravyflow.src.utils.numerics.NonFiniteWarning:"
        "gravyflow\\.src\\.dataset\\.(dataset|features\\.injection|tools\\.snr)",
    )

#: Environment variables through which a client would send traffic to a proxy.
PROXY_VARIABLES = tuple(
    name for base in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY")
    for name in (base, base.lower())
)

#: The hosts a name lookup may ask for when even loopback is refused: this machine.
LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "::1", "0.0.0.0"})


def refuse_internet(monkeypatch, refusals: Optional[List[str]] = None,
                    loopback: bool = True) -> None:
    """
    Refuse every internet connection, by name, and record each refusal in
    ``refusals`` when given. Every network source the library uses (the GWOSC APIs,
    timeline, event list and strain files, Zenodo) connects through a socket,
    whatever client it goes through (requests, urllib, gwosc, gwpy), so one refusal
    covers them all, and any source added later.

    Proxies are switched off, since a client would otherwise reach the internet
    through a proxy on this machine: the proxy variables are removed, and
    ``NO_PROXY=*`` is set, which urllib and requests read before the system's own
    proxy settings (macOS, Windows), so those are never consulted either.

    With ``loopback=False`` connections to this machine are refused too, and so is
    every name lookup of another host: nothing leaves through a socket at all.
    """
    import socket

    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "*")

    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def refuse(what):
        if refusals is not None:
            refusals.append(what)
        raise ConnectionRefusedError(
            f"Offline test: refused {what}; the network is reachable only in test runs "
            "with --network."
        )

    def check(sock, address):
        host = str(address[0])
        local = host in ("localhost", "::1") or host.startswith("127.")
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not (loopback and local):
            refuse(f"connection to {address}")

    def connect(sock, address, *args, **kwargs):
        check(sock, address)
        return original_connect(sock, address, *args, **kwargs)

    def connect_ex(sock, address, *args, **kwargs):
        check(sock, address)
        return original_connect_ex(sock, address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            refuse(f"name lookup of {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    if not loopback:
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def is_bytecode_cache(path) -> bool:
    """
    Whether ``path`` is the interpreter's bytecode cache: a ``__pycache__`` directory,
    or a file in one (with the temporary name it is written under), or a ``.pyc``.
    The interpreter makes these on the first import of a module (pytest's assertion
    rewriter too), whatever the suite does, so neither the snapshot nor the audit
    counts them as its writes.
    """
    path = Path(os.fsdecode(path))
    return "__pycache__" in path.parts or path.suffix == ".pyc"


def tree_snapshot(root: Path) -> Dict[str, Tuple[int, int]]:
    """
    {relative path: (size, mtime_ns)} of every file under ``root`` but bytecode. A
    file that vanishes while the tree is read (a writer's temporary file, renamed
    into place) is left out.
    """
    snapshot = {}
    if not root.is_dir():
        return snapshot
    for path in root.rglob("*"):
        if is_bytecode_cache(path):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file():
            snapshot[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def tree_changes(before: Dict, after: Dict) -> List[str]:
    """What was added, removed or changed between two ``tree_snapshot`` results."""
    changes = [f"added {p}" for p in sorted(set(after) - set(before))]
    changes += [f"removed {p}" for p in sorted(set(before) - set(after))]
    changes += [f"changed {p}" for p in sorted(set(before) & set(after)) if before[p] != after[p]]
    return changes


class WriteAudit:
    """
    The writes this process makes under watched roots, seen through Python's audit
    hooks (PEP 578): a file opened to write, and a path made, renamed or removed,
    through a symlink too. Another process's writes (a notebook filling the user
    cache meanwhile, or a child process) are not this run's and are not seen. Nor
    are C code that opens files itself (HDF5, sqlite), or a path relative to a
    dir_fd, which the event does not carry; the watched roots are also snapshotted.
    """

    #: Path-changing events, with the argument positions that hold the changed paths.
    EVENTS = {
        "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
        "os.truncate": (0,), "os.symlink": (1,), "os.link": (1,), "shutil.rmtree": (0,),
    }
    WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

    def __init__(self, roots: Dict[str, Path]):
        self.roots = {
            name: tuple({os.path.abspath(root), os.path.realpath(root)})
            for name, root in roots.items()
        }
        self.writes: Dict[str, List[str]] = {}

    def __call__(self, event, args):
        if event == "open":
            if args[2] & self.WRITE_FLAGS:
                self._record(event, args[0])
        elif event in self.EVENTS:
            for index in self.EVENTS[event]:
                self._record(event, args[index])

    def _record(self, event, path):
        if isinstance(path, int):
            return
        try:
            path = os.path.abspath(os.fsdecode(path))
            paths = {path, os.path.realpath(path)}
        except (TypeError, ValueError, OSError):  # a hook must never fail the operation
            return
        if is_bytecode_cache(path):
            return
        for name, roots in self.roots.items():
            if any(p == root or p.startswith(root + os.sep) for p in paths for root in roots):
                self.writes.setdefault(name, []).append(f"{event} {path}")

    def __enter__(self):
        _ACTIVE_AUDITS.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE_AUDITS.remove(self)


#: The WriteAudits in force. An audit hook cannot be removed once added, so one hook,
#: added when this file is imported, serves them all.
_ACTIVE_AUDITS: List[WriteAudit] = []


def _audit_hook(event, args):
    for audit in tuple(_ACTIVE_AUDITS):
        audit(event, args)


sys.addaudithook(_audit_hook)


@dataclass
class SuiteGuard:
    """
    The suite's hermetic state: its temporary roots, whether it is offline, and,
    offline, every refused connection in order.
    """

    root: Path
    offline: bool
    refusals: List[str] = field(default_factory=list)


@pytest.fixture(autouse=True, scope="session")
def suite_guard(request, tmp_path_factory):
    """
    The whole suite runs in temporary roots: ``HOME``, ``XDG_CACHE_HOME``,
    ``GRAVYFLOW_CACHE_DIR`` and the working directory, in every run (with --network
    too), so no test reads or writes a user cache, or a ``./generator_data``, and
    the user cache starts empty. Without --network, every internet connection is
    refused (and recorded) and proxies are off, so no test depends on the network:
    only the online unit, which fails by name offline, asks for it.

    The library resolves its cache root at each use, so under the temporary HOME
    it reaches the user's own cache (``~/.gravyflow``) only by a path taken before
    this fixture ran. When the suite ends, it fails if the WriteAudit saw this
    process write into the installed package or that cache, or if the package
    changed. The cache is snapshotted too, but a change the audit did not see (C
    code such as HDF5 or sqlite, a dir_fd path, a child process) cannot be told
    from another process using the cache meanwhile: this host records no writer.
    So such a change is named in the report and does not fail the suite.
    """
    watched = {"the package": Path(PACKAGE), "~/.gravyflow": Path.home() / ".gravyflow"}
    before = {name: tree_snapshot(path) for name, path in watched.items()}
    root = tmp_path_factory.mktemp("suite")
    guard = SuiteGuard(root, offline=not request.config.getoption("--network"))
    previous_cwd = os.getcwd()
    with WriteAudit(watched) as audit, pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("HOME", str(root / "home"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(root / "xdg-cache"))
        monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(root / "gravyflow-cache"))
        (root / "work").mkdir()
        os.chdir(root / "work")
        if guard.offline:
            refuse_internet(monkeypatch, guard.refusals)
        try:
            yield guard
        finally:
            os.chdir(previous_cwd)

    if cache_changes := tree_changes(before["~/.gravyflow"], tree_snapshot(watched["~/.gravyflow"])):
        request.config.pluginmanager.get_plugin("terminalreporter").write_line(
            f"The user's own cache {watched['~/.gravyflow']} changed during the suite: "
            f"{cache_changes}. A change the write audit did not see is this process's C code "
            "or child, or another process's."
        )
    changed = dict(audit.writes)
    if changes := tree_changes(before["the package"], tree_snapshot(watched["the package"])):
        changed.setdefault("the package", []).extend(changes)
    if changed:
        pytest.fail(f"The test suite wrote outside its temporary roots: {changed}")


@pytest.fixture(autouse=True)
def empty_store():
    """
    Every test starts with an empty store under the suite's cache root, so no test
    reads tiles another test's fake source committed under the same source id. The
    reads ahead a test left running are stopped when it ends (``store.shutdown``),
    so none of them fetches through, or commits into, the next test's patched sources.
    """
    import shutil

    from gravyflow.src.dataset.store import pools

    def clear():
        root = os.environ.get("GRAVYFLOW_CACHE_DIR")
        if root:
            shutil.rmtree(Path(root) / "store", ignore_errors=True)

    clear()
    yield
    if pools.background_threads():
        pools.shutdown()
    clear()


@pytest.fixture
def data_obtainer():
    return gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE,
            gf.DataLabel.GLITCHES]
    )


def usable_over_its_runs(obtainer, ifo):
    """
    Offline data quality: every detector usable over the obtainer's whole runs, so
    every transient record passes it (a stand-in for ``get_all_segment_times``).
    """
    import numpy as np

    return np.array([[min(obtainer.start_gps_times), max(obtainer.end_gps_times)]])


@contextmanager
def gwosc_serving(fetch):
    """
    Serve every ``gf.GWOSC()`` source's strain from ``fetch`` for the block, offline: a
    callable ``fetch(detector, sample_rate_hertz, interval)`` returning a
    ``SegmentStrain``, whose ``rates`` (when it has them) replace GWOSC's native rates.
    Obtainers built before the block are served too, since they hold ``GWOSC()``.
    Its strain is the tiles of ``fetch.source_id`` (default ``"gwosc-test"``), which the
    fetch's strain must be labelled with, of ``fetch.tile_seconds`` when it has them
    (else GWOSC's). Yields ``fetch``. A patch of ``GWOSC``, for
    tests that fake one fetch; the leakage harness installs a source instead.
    """
    from unittest import mock

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(gf.GWOSC, "fetch", lambda source, *args: fetch(*args)))
        name = getattr(fetch, "source_id", "gwosc-test")
        stack.enter_context(mock.patch.object(gf.GWOSC, "source_id", lambda source, gps, rate: name))
        if hasattr(fetch, "rates"):
            rates = tuple(float(rate) for rate in fetch.rates)
            stack.enter_context(mock.patch.object(gf.GWOSC, "rates", lambda source: rates))
        if hasattr(fetch, "tile_seconds"):  # small tiles keep a fake's fetches small
            seconds = float(fetch.tile_seconds)
            stack.enter_context(mock.patch.object(gf.GWOSC, "tile_seconds", lambda source, rate: seconds))
        yield fetch


def no_glitch_vetoes():
    """
    Offline glitch vetoes: no detector has a catalogued glitch to keep a partner
    window from (a patch of ``get_glitch_veto_segments``, which reads Gravity Spy).
    """
    from unittest import mock
    import numpy as np

    return mock.patch.object(gf.TransientDataObtainer, "get_glitch_veto_segments",
                             lambda self, ifo: np.empty((0, 2)))


def catalogued_events(events=()):
    """
    Serve ``events`` (rows with ``name``, ``gps`` and ``catalog``) as the one event
    list, offline: a patch of ``event.get_event_catalogue``, which every event
    veto and the transient index's event exclusion read.
    """
    from unittest import mock
    from gravyflow.src.dataset.features import event as event_module

    table = event_module._event_rows_to_arrays([dict(e) for e in events])
    return mock.patch.object(event_module, "get_event_catalogue", lambda use_cache=True: table)


#: Three O3 events, as the GWTC catalogue lists them.
OFFLINE_EVENTS = [
    {"name": name, "gps": gps, "catalog": "GWTC-3-confident",
     "observing_run": "O3", "p_astro": 1.0}
    for name, gps in (("GW-OFFLINE-A", 1240000000.4), ("GW-OFFLINE-B", 1250000000.4),
                      ("GW-OFFLINE-C", 1260000000.4))
]


@contextmanager
def offline_event_catalogue():
    """
    The event catalogue, its data quality and its strain, offline, for the block:
    the catalogue lists OFFLINE_EVENTS, every record passes data quality, and every
    example reads unit Gaussian strain, seeded by its record's time. Yields the events.

    ``segment_builders`` copies ``get_events_with_params`` from ``event`` when it is
    first imported, so both modules are imported before either is patched: a first
    import inside the block would copy the fake, which would then outlive it.
    """
    from unittest import mock
    import numpy as np
    from gravyflow.src.dataset.features import event as event_module
    from gravyflow.src.dataset.features import segment_builders

    def gaussian_strain(self, record, reads, window_spec):
        rng = np.random.default_rng(int(record.transient_gps_time))
        shape = (len(reads),)
        return (
            rng.standard_normal(shape + (window_spec.num_onsource_samples,)).astype(np.float32),
            rng.standard_normal(shape + (window_spec.num_offsource_samples,)).astype(np.float32),
        )

    def listed(observing_runs=None, event_types=None, event_names=None, **kwargs):
        return [dict(e) for e in OFFLINE_EVENTS if event_names is None or e["name"] in event_names]

    with mock.patch.object(event_module, "get_events_with_params", listed), \
         mock.patch.object(segment_builders, "get_events_with_params", listed), \
         mock.patch.object(gf.TransientDataObtainer, "_read_example", gaussian_strain), \
         mock.patch.object(gf.TransientDataObtainer, "get_all_segment_times", usable_over_its_runs), \
         catalogued_events(OFFLINE_EVENTS):
        yield [dict(e) for e in OFFLINE_EVENTS]


@pytest.fixture
def offline_events():
    """The ``offline_event_catalogue`` for the test. Yields the events."""
    with offline_event_catalogue() as events:
        yield events
