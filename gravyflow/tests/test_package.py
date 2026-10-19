"""
The package as users install and import it: what ``import gravyflow`` does to
the process, the public API it names, its version, and the wheel it builds.

The import checks run in a fresh interpreter, since this one has long since
imported everything.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
import zipfile
from pathlib import Path

import pytest

import gravyflow as gf

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gravyflow"

#: What the extras provide; the core install has none of them.
EXTRA_PACKAGES = (
    "gwpy", "gwdatafind", "gwosc", "requests", "pandas",
    "bokeh", "panel", "holoviews", "datashader",
    "matplotlib", "IPython", "ipykernel", "pytest", "lal", "lalsimulation",
)
#: The modules whose names need the viz extra: the plots, the validator's
#: dashboard, and the example models, which bring their recipe's validation.
VIZ_MODULES = ("gravyflow.src.utils.plotting", "gravyflow.src.validate", "gravyflow.src.model.examples")

#: Make the named top-level packages absent, as if they were not installed: a
#: None in sys.modules makes their import raise and importlib.util.find_spec
#: return None.
BLOCKER = """
import sys
for name in {names!r}:
    sys.modules[name] = None
"""


def run_python(source: str, env: dict = None, timeout: float = 300) -> str:
    """Run ``source`` in a fresh interpreter on this checkout; return its stdout."""
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT, env=os.environ.copy() if env is None else env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert process.returncode == 0, process.stderr[-4000:]
    return process.stdout


def without_keras_backend() -> dict:
    """This environment, with no Keras backend chosen and JAX's platform pinned to the CPU."""
    env = os.environ.copy()
    for name in ("KERAS_BACKEND", "TF_CPP_MIN_LOG_LEVEL", "XLA_PYTHON_CLIENT_PREALLOCATE"):
        env.pop(name, None)
    env["JAX_PLATFORMS"] = "cpu"
    return env


def documented_api() -> list:
    """The names `docs/api.md` documents: the first backticked name of each list item."""
    text = (ROOT / "docs" / "api.md").read_text()
    return re.findall(r"^- `([A-Za-z_][A-Za-z0-9_]*)`", text, flags=re.MULTILINE)


def test_import_sets_nothing_global():
    report = json.loads(run_python("""
        import json, logging, os, sys, warnings
        environ, filters, modules = dict(os.environ), list(warnings.filters), set(sys.modules)
        levels = {name: logger.level for name, logger in logging.root.manager.loggerDict.items()
                  if isinstance(logger, logging.Logger)}
        import gravyflow
        print(json.dumps({
            "environ": {k: v for k, v in os.environ.items() if environ.get(k) != v},
            "removed": sorted(set(environ) - set(os.environ)),
            "filters": len(warnings.filters) - len(filters),
            "loggers": sorted(name for name, logger in logging.root.manager.loggerDict.items()
                              if isinstance(logger, logging.Logger) and name != "gravyflow"
                              and logger.level != levels.get(name, logging.NOTSET)),
            "modules": sorted({name.split(".")[0] for name in set(sys.modules) - modules}
                              - set(sys.stdlib_module_names)),
        }))
    """, env=without_keras_backend()).splitlines()[-1])
    assert report["environ"] == {} and report["removed"] == [], report
    assert report["filters"] == 0, report
    assert report["loggers"] == [], report
    # Nothing outside the standard library is imported until a name is used.
    assert report["modules"] == ["gravyflow"], report


def test_using_names_sets_nothing_global():
    report = json.loads(run_python("""
        import json, os, sys
        os.environ["KERAS_BACKEND"] = "jax"
        # The libraries gravyflow imports set variables of their own as they load:
        # import them first, so what changes below is gravyflow's alone.
        import astropy.time, bokeh.plotting, gwdatafind, gwpy.timeseries, h5py, holoviews, jax, keras
        import ripplegw, scipy.signal
        before = dict(os.environ), jax.config.jax_enable_x64, jax.config.jax_platforms
        import gravyflow as gf
        for name in gf.__all__:
            getattr(gf, name)
        after = dict(os.environ), jax.config.jax_enable_x64, jax.config.jax_platforms
        print(json.dumps({
            "environ": {k: v for k, v in after[0].items() if before[0].get(k) != v},
            "jax": [before[1:], after[1:]],
        }))
    """, env=without_keras_backend()).splitlines()[-1])
    assert report["environ"] == {}, report
    assert report["jax"][0] == report["jax"][1], report


def test_a_name_refuses_without_the_jax_backend():
    output = run_python("""
        import gravyflow as gf
        try:
            gf.Dataset
        except ImportError as error:
            print("refused:", error)
    """, env=without_keras_backend())
    assert "refused:" in output and "KERAS_BACKEND=jax" in output, output


def test_core_runs_without_any_extra():
    # Blocks every extra's packages, then builds two batches of a synthetic
    # dataset: coloured noise (the bundled noise profiles) with WNB injections.
    output = run_python(BLOCKER.format(names=EXTRA_PACKAGES) + textwrap.dedent("""
        import os
        os.environ["KERAS_BACKEND"] = "jax"
        import numpy as np
        import gravyflow as gf
        modules = gf._MODULES
        core = [name for name in gf.__all__ if not modules[name].startswith(""" + repr(VIZ_MODULES) + """)]
        for name in core:
            getattr(gf, name)
        dataset = gf.Dataset(
            seed=7,
            group="train",
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.L1]),
            waveform_generators=[gf.WNBGenerator(
                scaling_method=gf.ScalingMethod(gf.Distribution(value=10.0), gf.ScalingTypes.SNR),
                duration_seconds=gf.Distribution(value=0.2),
                min_frequency_hertz=gf.Distribution(value=50.0),
                max_frequency_hertz=gf.Distribution(value=150.0),
            )],
            sample_rate_hertz=512.0,
            offsource_duration_seconds=4.0,
            num_examples_per_batch=2,
            steps_per_epoch=2,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ScalingTypes.SNR],
        )
        shapes = [np.asarray(dataset[i][0]["WHITENED_ONSOURCE"]).shape for i in range(2)]
        assert all(np.isfinite(np.asarray(dataset[i][0]["WHITENED_ONSOURCE"])).all() for i in range(2))
        print("batches", shapes, len(core), "of", len(gf.__all__))
    """), env=without_keras_backend())
    assert "batches [(2, 1, " in output, output


def test_every_public_name_resolves_as_the_first_access():
    # Each name is the first gravyflow name used: every gravyflow module is dropped
    # from sys.modules before it, so an import cycle that only one first access
    # meets (gf.GlitchType through acquisition's package once did) fails by name.
    # Third-party modules stay loaded, which only makes the loop fast.
    output = run_python("""
        import os, sys
        os.environ["KERAS_BACKEND"] = "jax"
        import gravyflow
        failed = []
        for name in gravyflow.__all__:
            for module in [m for m in sys.modules if m == "gravyflow" or m.startswith("gravyflow.")]:
                del sys.modules[module]
            import gravyflow as gf
            try:
                getattr(gf, name)
            except Exception as error:
                failed.append(f"{name}: {type(error).__name__}: {error}")
        print("resolved", len(gravyflow.__all__), "failed", failed)
    """)
    assert "failed []" in output, output


def test_the_default_source_is_gwosc_on_first_read():
    # Defaults.source is made as it is first read, so it is GWOSC() even when
    # nothing has imported the acquisition package that defines GWOSC.
    output = run_python("""
        import os, sys
        os.environ["KERAS_BACKEND"] = "jax"
        import gravyflow as gf
        defaults = gf.Defaults
        before = "gravyflow.src.dataset.acquisition.sources" in sys.modules
        source = defaults.source
        from gravyflow.src.dataset.acquisition.sources import GWOSC
        print("default", before, type(source) is GWOSC, source == GWOSC(), defaults.source is source)
    """, env=without_keras_backend())
    assert "default False True True True" in output, output


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", gf.__version__), gf.__version__
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "gravyflow.__version__"}


def test_the_public_api_is_the_documented_list():
    documented = documented_api()
    assert len(documented) == len(set(documented)), "a name is documented twice"
    assert sorted(documented) == sorted(gf.__all__)
    assert dir(gf) == sorted(gf.__all__)
    for name in gf.__all__:
        assert getattr(gf, name) is not None, name
    with pytest.raises(AttributeError, match="no attribute 'estimate_psd'"):
        gf.estimate_psd


def test_every_gf_name_the_docs_use_resolves():
    # A doc that names `gf.TransientObtainer` or `gf.EventType` teaches a name that
    # does not exist: each dotted `gf.` path in docs/*.md must resolve.
    unresolved, used = [], 0
    for doc in sorted((ROOT / "docs").glob("*.md")):
        for number, line in enumerate(doc.read_text().splitlines(), start=1):
            for path in re.findall(r"\bgf\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", line):
                used += 1
                target = gf
                for part in path.split("."):
                    target = getattr(target, part, None)
                    if target is None:
                        unresolved.append(f"{doc.name}:{number}: gf.{path}")
                        break
    assert used > 50, f"found only {used} gf names in the docs"
    assert not unresolved, unresolved


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    """The wheel built from a copy of this checkout (the build writes beside its source)."""
    root = tmp_path_factory.mktemp("wheel")
    source = root / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(PACKAGE, source / "gravyflow",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    process = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
         "--wheel-dir", str(root / "dist"), str(source)],
        capture_output=True, text=True, timeout=300,
    )
    assert process.returncode == 0, process.stdout[-2000:] + process.stderr[-2000:]
    (path,) = (root / "dist").glob("gravyflow-*.whl")
    return path


def test_the_wheel_ships_the_package_data(wheel):
    names = set(zipfile.ZipFile(wheel).namelist())
    shipped = lambda pattern: {n for n in names if re.fullmatch(pattern, n)}
    for directory, suffix in (("noise_profiles", "csv"), ("template_banks", "json")):
        expected = {f"gravyflow/res/{directory}/{p.name}" for p in (PACKAGE / "res" / directory).glob(f"*.{suffix}")}
        assert expected and expected <= names, sorted(expected - names)
    from gravyflow.src import resources
    for path in (PACKAGE / "res" / "model").glob("*.keras"):
        assert (f"gravyflow/res/model/{path.name}" in names) == (f"model/{path.name}" not in resources.MANIFEST), path.name
    assert not shipped(r"gravyflow/tests/.*"), "the tests are not shipped"
    assert not shipped(r"gravyflow/res/tests/.*"), "the test data is not shipped"


def test_the_wheel_declares_the_extras(wheel):
    archive = zipfile.ZipFile(wheel)
    (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
    text = archive.read(metadata).decode()
    assert re.search(rf"^Version: {re.escape(gf.__version__)}$", text, re.MULTILINE)
    extras = set(re.findall(r"^Provides-Extra: (\S+)$", text, re.MULTILINE))
    assert extras == {"realdata", "viz", "test", "notebooks", "cuda", "tpu"}, extras
    requirements = re.findall(r"^Requires-Dist: (.+)$", text, re.MULTILINE)
    core = {re.split(r"[ <>=!;\[]", r)[0].lower() for r in requirements if "extra ==" not in r}
    assert core.isdisjoint({"gitpython", "sqlalchemy", "psycopg2-binary", "pycbc", "jax-metal"}), core
    assert core.isdisjoint(p.lower() for p in EXTRA_PACKAGES), core
    assert any(r.startswith("datashader") and 'extra == "viz"' in r for r in requirements), requirements
    assert any(r.startswith("jax[tpu]") and 'extra == "tpu"' in r for r in requirements), requirements


def test_large_weights_are_release_downloads(tmp_path, monkeypatch):
    from gravyflow.src import resources
    from gravyflow.src.dataset.store.files import RemoteFile, RemoteFileMismatch, RemoteFileMissing
    # A bundled model is found in the package.
    assert resources.weights("gabbard_2017.keras") == resources.RESOURCES / "model" / "gabbard_2017.keras"
    # A manifest entry is found only once fetched, explicitly, into the store's cache,
    # checked against its pinned size and sha256.
    payload = tmp_path / "remote" / "large.keras"
    payload.parent.mkdir()
    payload.write_bytes(b"weights")
    entry = RemoteFile("model/large.keras", payload.as_uri(), hashlib.sha256(b"weights").hexdigest(), 7,
                       licence="Apache-2.0", redistributable=True)
    monkeypatch.setitem(resources.MANIFEST, entry.name, entry)
    with pytest.raises(RemoteFileMissing, match=r"gravyflow.fetch\('model/large.keras'\)"):
        resources.weights("large.keras")
    path = resources.fetch("model/large.keras")
    assert path.read_bytes() == b"weights" and resources.weights("large.keras") == path
    assert str(path).startswith(os.environ["GRAVYFLOW_CACHE_DIR"]) and path.name == "large.keras"
    # A download whose checksum differs is refused and not kept.
    monkeypatch.setitem(resources.MANIFEST, entry.name, RemoteFile(
        entry.name, entry.url, "0" * 64, 7, licence="Apache-2.0", redistributable=True))
    with pytest.raises(RemoteFileMismatch, match="sha256"):
        resources.fetch("model/large.keras")
    assert not resources.MANIFEST[entry.name].path().exists()
    with pytest.raises(FileNotFoundError, match="no trained weights named 'absent.keras'"):
        resources.weights("absent.keras")
