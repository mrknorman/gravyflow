"""
A static census of the shipped examples: the notebooks' code cells, the
tutorials, the example scripts and the example models, parsed, never run.

Rule: every ``gf.<name>`` they use, in code or in code printed from a string,
exists in the library, so no example teaches an API that is not there.
"""

import ast
import json
import re
from pathlib import Path

import gravyflow as gf

REPOSITORY = Path(gf.__file__).resolve().parent.parent
EXAMPLES = REPOSITORY / "example_notebooks"
MODELS = Path(gf.__file__).resolve().parent / "src" / "model" / "examples"
PRINTED = re.compile(r"\bgf((?:\.[A-Za-z_]\w*)+)")


def example_sources():
    """(label, source) for every code cell and example file; refuses an empty census."""
    if not EXAMPLES.is_dir():
        raise FileNotFoundError(f"The examples are not at {EXAMPLES}; the census needs the repository.")
    sources = []
    for notebook in sorted(EXAMPLES.glob("*.ipynb")):
        cells = json.loads(notebook.read_text())["cells"]
        for i, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                # IPython magics and shell escapes are not Python.
                code = "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in code.splitlines())
                sources.append((f"{notebook.name} cell {i}", code))
    for path in sorted([*EXAMPLES.glob("*.py"), *EXAMPLES.glob("tutorials/*.py"), *MODELS.glob("*.py")]):
        sources.append((str(path.relative_to(REPOSITORY.parent if MODELS in path.parents else REPOSITORY)),
                        path.read_text()))
    if len(sources) < 20:
        raise AssertionError(f"Only {len(sources)} example sources found; the census would be hollow.")
    return sources


def gf_names(source):
    """Every dotted ``gf.a.b`` chain in the code, and in code printed from string literals."""
    tree = ast.parse(source)
    chains = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts, value = [node.attr], node.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name) and value.id == "gf":
                chains.add(".".join(reversed(parts)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            chains.update(match.lstrip(".") for match in PRINTED.findall(node.value))
    return chains


def missing(chain):
    """The first prefix of ``gf.<chain>`` that does not resolve, or None."""
    obj, path = gf, "gf"
    for part in chain.split("."):
        path = f"{path}.{part}"
        if not hasattr(obj, part):
            return path
        obj = getattr(obj, part)
    return None


def test_examples_use_only_existing_gf_names():
    """Every ``gf.<name>`` an example uses exists."""
    found, absent = 0, []
    for label, source in example_sources():
        try:
            chains = gf_names(source)
        except SyntaxError as error:
            raise AssertionError(f"{label} does not parse: {error}") from error
        found += len(chains)
        absent += [f"{label}: {path}" for path in sorted({missing(c) for c in chains} - {None})]
    assert found > 100, f"only {found} gf names found; the census would be hollow"
    assert not absent, "Examples use gf names that do not exist:\n  " + "\n  ".join(absent)


def test_census_sees_a_missing_name():
    """The census itself: a planted missing name, in code and in printed code, is found."""
    chains = gf_names('x = gf.NoSuchObtainer()\nprint("ds = gf.Dataset(noise=gf.NoSuchNoise)")')
    assert {missing(c) for c in chains} - {None} == {"gf.NoSuchObtainer", "gf.NoSuchNoise"}
