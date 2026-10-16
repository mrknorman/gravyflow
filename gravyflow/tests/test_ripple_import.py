"""
gravyflow takes ripple only as the installed ``ripplegw`` package.

No source file imports a module named ``ripple`` (a vendored checkout or an
alias of ``ripplegw``), and none edits ``sys.path`` or registers ``ripple`` in
``sys.modules``. The scan covers every Python file in the repository and the
code cells of every notebook.
"""
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIPPED_DIRECTORIES = {".git", "build", "dist", "__pycache__", ".ipynb_checkpoints"}

NOTEBOOK_PATTERNS = [
    re.compile(r"^\s*import\s+ripple(\s|\.|,|$)", re.MULTILINE),
    re.compile(r"^\s*from\s+ripple(\s|\.)", re.MULTILINE),
    re.compile(r"sys\s*\.\s*path\s*(\.\s*(insert|append|extend)\b|\+=|=[^=])"),
    re.compile(r"sys\s*\.\s*modules\s*\[\s*['\"]ripple['\"]\s*\]\s*="),
]


def _source_files(suffix):
    for path in sorted(REPO_ROOT.rglob(f"*{suffix}")):
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in SKIPPED_DIRECTORIES or part.endswith(".egg-info") for part in parts):
            continue
        yield path


def _is_ripple(module_name):
    return module_name == "ripple" or module_name.startswith("ripple.")


def _is_sys_attribute(node, attribute):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _violations(tree):
    """Yield (line, description) for every forbidden construct in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_ripple(alias.name):
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_ripple(node.module):
                yield node.lineno, f"from {node.module} import ..."
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic_import = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
            )
            if (
                dynamic_import
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _is_ripple(node.args[0].value)
            ):
                yield node.lineno, f"dynamic import of {node.args[0].value}"
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"insert", "append", "extend"}
                and _is_sys_attribute(func.value, "path")
            ):
                yield node.lineno, f"sys.path.{func.attr}(...)"
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_sys_attribute(target, "path"):
                    yield node.lineno, "assignment to sys.path"
                if isinstance(target, ast.Subscript) and _is_sys_attribute(target.value, "modules"):
                    key = target.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str) and _is_ripple(key.value):
                        yield node.lineno, f"sys.modules[{key.value!r}] = ..."


def test_scan_finds_the_package():
    files = list(_source_files(".py"))
    assert REPO_ROOT / "gravyflow" / "src" / "dataset" / "features" / "waveforms" / "cbc.py" in files


def test_the_scan_flags_each_forbidden_construct():
    source = "\n".join([
        "import ripple",
        "from ripple.waveforms import IMRPhenomD",
        "import sys",
        "sys.path.insert(0, 'somewhere')",
        "sys.modules['ripple'] = None",
        "import ripplegw",
        "from ripplegw import waveform",
        "sys.path += ['elsewhere']",
        "importlib.import_module('ripple.waveforms')",
    ])
    lines = sorted(line for line, _ in _violations(ast.parse(source)))
    assert lines == [1, 2, 4, 5, 8, 9]


def test_no_python_file_imports_ripple_or_edits_sys_path():
    found = []
    for path in _source_files(".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, description in _violations(tree):
            found.append(f"{path.relative_to(REPO_ROOT)}:{line}: {description}")
    assert not found, "\n".join(found)


def test_no_notebook_imports_ripple_or_edits_sys_path():
    found = []
    for path in _source_files(".ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for pattern in NOTEBOOK_PATTERNS:
                if pattern.search(source):
                    found.append(f"{path.relative_to(REPO_ROOT)} cell {index}: {pattern.pattern}")
    assert not found, "\n".join(found)


def test_importing_gravyflow_loads_no_ripple_module():
    import gravyflow  # noqa: F401
    import ripplegw  # noqa: F401

    loaded = sorted(name for name in sys.modules if _is_ripple(name))
    assert loaded == []
