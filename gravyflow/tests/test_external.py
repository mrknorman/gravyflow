"""
External models: a card's fingerprint, and the runner that runs an adapter in its own
environment.

The adapter here is a toy in numpy, installed into a temporary directory as
``pip install --target`` would install it (a module and its ``.dist-info`` with the
entry points), and run by this interpreter, isolated. Its weights file is served at
a ``file://`` URL and fetched into the suite's temporary store; no socket opens.
"""

import ast
import dataclasses
import hashlib
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.src.dataset.store.files import RemoteFile
from gravyflow.src.detection.triggers import FIELDS
from gravyflow.src.external import child, runner

TOY = textwrap.dedent('''
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    RECORD = Path(os.environ["TOY_RECORD"])


    class Toy:
        def prepare(self, fetch):
            RECORD.joinpath("prepare.json").write_text(json.dumps({
                "cwd": os.getcwd(), "isolated": sys.flags.isolated,
                "no_bytecode": sys.flags.dont_write_bytecode, "argv": sys.argv}))
            self.scale = float(np.load(fetch("toy/scale.npy")))

        def preprocess(self, segment):
            return segment.strain * self.scale

        def search(self, segment):
            summed = self.preprocess(segment).sum(axis=0)
            index = np.flatnonzero(summed > 1.0)
            return SimpleNamespace(time=segment.t0 + index / segment.sample_rate_hertz,
                                   stat=summed[index], var=np.full(len(index), 0.25))


    class Sleeper(Toy):
        def search(self, segment):
            straggler = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            RECORD.joinpath("straggler").write_text(str(straggler.pid))
            time.sleep(120)


    class Unrankable(Toy):
        def search(self, segment):
            return SimpleNamespace(time=[1.0], stat=[float("nan")], var=[0.1])


    class Crashes(Toy):
        def search(self, segment):
            raise RuntimeError("the toy model fell over")


    class NotASearch:
        pass
''')

ENTRY_POINTS = """\
[gravyflow.external]
toy = toy_search:Toy
sleeper = toy_search:Sleeper
unrankable = toy_search:Unrankable
crashes = toy_search:Crashes
not-a-search = toy_search:NotASearch
"""

LOCK = b"toy environment: numpy\n"
RATE, T0 = 64.0, 1238166018.0


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """The toy adapter installed under ``tmp_path``, its weights published, its card and environment."""
    target = tmp_path / "target"
    info = target / "toy_search-0.1.0.dist-info"
    info.mkdir(parents=True)
    (target / "toy_search.py").write_text(TOY)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-search\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(ENTRY_POINTS)
    published = tmp_path / "published" / "scale.npy"
    published.parent.mkdir()
    np.save(published, np.float64(2.0))
    data = published.read_bytes()
    weights = RemoteFile("toy/scale.npy", published.as_uri(), hashlib.sha256(data).hexdigest(), len(data),
                         licence=None, redistributable=False)
    (tmp_path / "env.lock").write_bytes(LOCK)
    record = tmp_path / "record"
    record.mkdir()
    monkeypatch.setenv("TOY_RECORD", str(record))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return dataclasses.replace(
        card(weights), environment_lock=hashlib.sha256(LOCK).hexdigest()
    ), gf.AdapterEnvironment(sys.executable, tmp_path / "env.lock", (target,)), scratch, record


def card(file, **changes):
    """The toy's card, its weights ``file``, with ``changes``."""
    fields = dict(
        name="toy", adapter="toy", adapter_version="0.1.0", environment_lock="0" * 64,
        input=gf.InputContract(RATE, ("H1", "L1"), window_seconds=1.0, stride_seconds=0.25,
                               whitening="none: the toy reads raw strain"),
        output=gf.OutputContract("the scaled detector sum", signal_column=0, direction="higher",
                                 threshold=1.0, cluster_seconds=0.0),
        weights=(gf.Weights(file, format="npy", framework="numpy"),),
        training_data="none: the toy is not trained",
        published=(gf.PublishedRow("toy2026, Table 1", "toy data", "sensitive distance", 100.0, "Mpc",
                                   far_per_month=1.0),),
    )
    return gf.ExternalCard(**{**fields, **changes})


def segment():
    strain = np.zeros((2, 256))
    strain[:, 100] = (0.4, 0.3)  # summed and scaled by 2: 1.4, above the toy's threshold
    strain[0, 200] = 0.2  # 0.4: below it
    return gf.SearchSegment(strain, ("H1", "L1"), T0, RATE)


def test_a_card_fingerprints_what_decides_its_results(tmp_path):
    weights = RemoteFile("toy/scale.npy", (tmp_path / "w").as_uri(), "a" * 64, 10, licence=None,
                         redistributable=False)
    base = card(weights)
    assert base.fingerprint == card(weights).fingerprint and len(base.fingerprint) == 64
    # 2048 and 2048.0 are one setting.
    assert card(weights, input=dataclasses.replace(base.input, sample_rate_hertz=64)).fingerprint == base.fingerprint
    changed = {
        "contract": card(weights, input=dataclasses.replace(base.input, window_seconds=2.0)),
        "output": card(weights, output=dataclasses.replace(base.output, signal_column=1)),
        "weights sha256": card(weights, weights=(gf.Weights(
            dataclasses.replace(weights, sha256="b" * 64), format="npy", framework="numpy"),)),
        "weights name": card(weights, weights=(gf.Weights(
            dataclasses.replace(weights, name="toy/other.npy"), format="npy", framework="numpy"),)),
        "weights format": card(weights, weights=(gf.Weights(weights, format="torchscript", framework="numpy"),)),
        "weights framework": card(weights, weights=(gf.Weights(weights, format="npy", framework="numpy==2.0"),)),
        "adapter": card(weights, adapter="toy-2"),
        "adapter version": card(weights, adapter_version="0.1.1"),
        "environment lock": card(weights, environment_lock="1" * 64),
    }
    prints = {name: c.fingerprint for name, c in changed.items()}
    assert base.fingerprint not in prints.values() and len(set(prints.values())) == len(prints), prints
    assert base.mode == gf.AS_PUBLISHED == "as-published"


def test_a_cards_provenance_is_not_its_identity(tmp_path):
    """A corrected citation, a new published row or a restated training set leaves the
    model the same model: only what decides a run's outputs is in the fingerprint."""
    weights = RemoteFile("toy/scale.npy", (tmp_path / "w").as_uri(), "a" * 64, 10, licence=None,
                         redistributable=False)
    base = card(weights)
    row = base.published[0]
    unchanged = {
        "citation": card(weights, published=(dataclasses.replace(row, citation="toy2026, Table 2"),)),
        "value": card(weights, published=(dataclasses.replace(row, value=101.0),)),
        "another row": card(weights, published=(row, dataclasses.replace(row, far_per_month=0.1))),
        "no rows": card(weights, published=()),
        "training data": card(weights, training_data="toy data, 2026 release"),
        "name": card(weights, name="toy, renamed"),
        "weights url": card(weights, weights=(gf.Weights(
            dataclasses.replace(weights, url="https://example.org/scale.npy"), format="npy", framework="numpy"),)),
        "weights licence": card(weights, weights=(gf.Weights(
            dataclasses.replace(weights, licence="MIT", redistributable=True), format="npy", framework="numpy"),)),
    }
    prints = {name: c.fingerprint for name, c in unchanged.items()}
    assert set(prints.values()) == {base.fingerprint}, prints


def test_a_card_refuses_what_it_cannot_honour(tmp_path):
    weights = RemoteFile("toy/scale.npy", (tmp_path / "w").as_uri(), "a" * 64, 10, licence=None,
                         redistributable=False)
    with pytest.raises(ValueError, match="mode is one of"):
        card(weights, mode="softmax-removed")
    with pytest.raises(ValueError, match="lock file's sha256"):
        card(weights, environment_lock="not a hash")
    with pytest.raises(ValueError, match="distinct names"):
        card(weights, weights=(gf.Weights(weights, "npy", "numpy"),) * 2)
    with pytest.raises(ValueError, match="direction"):
        gf.OutputContract("p", 0, "up", 0.5, 0.0)
    with pytest.raises(ValueError, match="signal column"):
        gf.OutputContract("p", -1, "higher", 0.5, 0.0)
    with pytest.raises(ValueError, match="window_seconds must be a finite number above 0"):
        gf.InputContract(2048, ("H1", "L1"), 0.0, 0.1, "welch")
    with pytest.raises(ValueError, match="pass band"):
        gf.InputContract(2048, ("H1", "L1"), 1.0, 0.1, "welch", band_pass_hertz=(500.0, 20.0))
    with pytest.raises(ValueError, match="training_data"):
        card(weights, training_data="")


def test_the_runner_runs_the_adapter_isolated_and_removes_its_directory(toy):
    toy_card, environment, scratch, record = toy
    run = gf.ExternalRunner(toy_card, environment, deadline_seconds=120, scratch=scratch)
    result = run.search(segment())
    assert isinstance(result, gf.ExternalResult) and isinstance(result.triggers, gf.Triggers)
    assert result.mode == "as-published" and result.card_fingerprint == toy_card.fingerprint
    assert result.name == "toy" and result.output == toy_card.output
    # The card's provenance travels with the result, outside its identity.
    assert result.published == toy_card.published and result.training_data == toy_card.training_data
    np.testing.assert_array_equal(result.triggers.time, [T0 + 100 / RATE])
    np.testing.assert_allclose(result.triggers.stat, [1.4])
    np.testing.assert_array_equal(result.triggers.var, [0.25])
    seen = json.loads((record / "prepare.json").read_text())
    assert seen["isolated"] == 1 and seen["no_bytecode"] == 1, seen
    assert Path(seen["cwd"]).parent == scratch and Path(seen["argv"][0]) == runner.CHILD
    np.testing.assert_array_equal(run.preprocess(segment()), 2 * segment().strain)
    assert list(scratch.iterdir()) == [], "the runner left its temporary directory"


def test_the_runner_refuses_an_adapter_by_name(toy):
    toy_card, environment, scratch, _ = toy
    with pytest.raises(gf.AdapterMissing, match=r"No adapter named 'absent' in the entry-point group "
                                                r"'gravyflow.external'"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter="absent"), environment, 60, scratch).search(segment())
    with pytest.raises(gf.AdapterMismatch, match=r"names 'toy' 0\.2\.0, and this environment's is 0\.1\.0"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter_version="0.2.0"), environment, 60,
                          scratch).search(segment())
    with pytest.raises(gf.AdapterMismatch, match="not an ExternalSearch"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter="not-a-search"), environment, 60,
                          scratch).search(segment())
    # Another lock is refused before anything runs.
    with pytest.raises(gf.AdapterMismatch, match="environment locked by sha256 1111"):
        gf.ExternalRunner(dataclasses.replace(toy_card, environment_lock="1" * 64), environment, 60,
                          scratch).search(segment())
    with pytest.raises(FileNotFoundError, match="no interpreter at"):
        gf.ExternalRunner(toy_card, dataclasses.replace(environment, python=scratch / "python"), 60,
                          scratch).search(segment())
    with pytest.raises(ValueError, match=r"reads \('H1', 'L1'\) at 64.0 Hz"):
        gf.ExternalRunner(toy_card, environment, 60, scratch).search(
            gf.SearchSegment(np.zeros((1, 64)), ("H1",), T0, RATE))
    assert list(scratch.iterdir()) == []


def test_a_failed_run_says_why(toy):
    toy_card, environment, scratch, _ = toy
    with pytest.raises(gf.ExternalRunFailed, match="(?s)exited with status 1.*the toy model fell over"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter="crashes"), environment, 60, scratch).search(segment())
    with pytest.raises(gf.ExternalRunFailed, match="returned triggers gravyflow refuses: .*NaN"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter="unrankable"), environment, 60,
                          scratch).search(segment())
    assert list(scratch.iterdir()) == []


def test_the_deadline_reaps_the_adapters_whole_session(toy):
    toy_card, environment, scratch, record = toy
    started = time.monotonic()
    with pytest.raises(gf.ExternalDeadlineExpired, match="outlived its 3 s deadline"):
        gf.ExternalRunner(dataclasses.replace(toy_card, adapter="sleeper"), environment, 3, scratch).search(segment())
    assert time.monotonic() - started < 3 + runner.GRACE_SECONDS + 30
    straggler = int((record / "straggler").read_text())
    for _ in range(100):  # bounded: 10 s for the orphan to be reaped by init, then give up
        try:
            os.kill(straggler, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"The adapter's child process {straggler} outlived the deadline.")
    assert list(scratch.iterdir()) == []


def test_the_child_serves_the_runners_request_in_process(toy, monkeypatch):
    # The same request, served by the child in this process (where coverage sees it).
    toy_card, environment, scratch, _ = toy
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(runner, "_supervise", lambda argv, cwd, log, deadline: child.main([argv[-1]]))
    try:
        result = gf.ExternalRunner(toy_card, environment, 60, scratch).search(segment())
        np.testing.assert_array_equal(result.triggers.time, [T0 + 100 / RATE])
        with pytest.raises(gf.AdapterMissing, match="'absent'"):
            gf.ExternalRunner(dataclasses.replace(toy_card, adapter="absent"), environment, 60,
                              scratch).search(segment())
    finally:
        for name in ("toy_search", "gravyflow_external_protocol"):
            sys.modules.pop(name, None)


def test_the_child_keeps_to_what_an_adapters_environment_has():
    # The child and the protocol it loads run beside an adapter's own framework, on
    # Python 3.8 or later, with numpy and nothing else of gravyflow's.
    assert child.TRIGGER_FIELDS == FIELDS
    for path in (runner.CHILD, runner.CHILD.with_name("protocol.py")):
        tree = ast.parse(path.read_text(), feature_version=child.ADAPTER_PYTHON)
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert imported <= set(sys.stdlib_module_names) | {"numpy"}, (path.name, imported)


def test_a_segment_is_built_from_a_sources_strain():
    strains = {d: SegmentStrain(np.arange(8.0) * (i + 1), T0, RATE, "test") for i, d in enumerate(("H1", "L1"))}
    built = gf.SearchSegment.from_strains(strains)
    assert built.detectors == ("H1", "L1") and built.t0 == T0 and built.duration_seconds == 8 / RATE
    np.testing.assert_array_equal(built["L1"], np.arange(8.0) * 2)
    with pytest.raises(KeyError, match="no detector 'V1'"):
        built["V1"]
    strains["L1"] = SegmentStrain(np.zeros(8), T0 + 1, RATE, "test")
    with pytest.raises(ValueError, match="share their first sample"):
        gf.SearchSegment.from_strains(strains)
    with pytest.raises(ValueError, match="NaN"):
        gf.SearchSegment(np.full((1, 4), np.nan), ("H1",), T0, RATE)

