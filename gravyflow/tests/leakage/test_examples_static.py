"""
A static census of the shipped examples: the notebooks' code cells, the
tutorials, the example scripts and the example models, parsed, never run.

Rules:
- every gravyflow name they use exists in the library: ``gf.<name>`` chains,
  names imported from gravyflow's modules and their attributes, under any
  alias, ``getattr(gf, "<name>")``, and ``gf.<name>`` in printed text;
- every block of code they print parses, so a reader can copy it.
"""

import ast
import importlib
import json
import re
import textwrap
from pathlib import Path

import gravyflow as gf

REPOSITORY = Path(gf.__file__).resolve().parent.parent
EXAMPLES = REPOSITORY / "example_notebooks"
MODELS = Path(gf.__file__).resolve().parent / "src" / "model" / "examples"
PRINTED = re.compile(r"\bgf((?:\.[A-Za-z_]\w*)+)")
# A printed line that starts code: a comment, a statement keyword, or a name
# (dotted or subscripted) assigned, augmented or called. Prose never starts so.
CODE_LINE = re.compile(
    r"^(#|@|(?:from|import|for|while|with|if|elif|else|try|except|def|class|return|print)\b"
    r"|[A-Za-z_][\w.]*(?:\[[^\]]*\])?\s*(?:=(?!=)|\+=|-=|\())")


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


def _dotted(node):
    """``a.b.c`` for a chain of attributes on a name, as (name, [b, c]); None otherwise."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return (node.id, list(reversed(parts))) if isinstance(node, ast.Name) else None


def gravyflow_names(source):
    """
    Every gravyflow path the code uses, as ``gravyflow.a.b``: attribute
    chains on a name bound to gravyflow or to anything imported from it
    (``gf`` is always bound, since a notebook cell relies on an earlier one),
    ``getattr`` of such a name with a literal string, and ``gf.<name>`` in
    string literals.
    """
    tree = ast.parse(source)
    bound = {"gf": "gravyflow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "gravyflow":
                    # `import gravyflow.x` binds `gravyflow`; `import gravyflow.x as y` binds y to gravyflow.x.
                    bound[alias.asname or "gravyflow"] = alias.name if alias.asname else "gravyflow"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "gravyflow":
            for alias in node.names:
                if alias.name != "*":
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    paths = set(bound.values()) - {"gravyflow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _dotted(node)
            if chain and chain[0] in bound:
                paths.add(".".join([bound[chain[0]], *chain[1]]))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            chain = _dotted(node.args[0])
            if chain and chain[0] in bound:
                paths.add(".".join([bound[chain[0]], *chain[1], node.args[1].value]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            paths.update("gravyflow" + match for match in PRINTED.findall(node.value))
    return paths


def missing(path):
    """The first prefix of a ``gravyflow.a.b`` path that does not resolve, or None."""
    parts = path.split(".")
    obj, here = importlib.import_module(parts[0]), parts[0]
    for part in parts[1:]:
        here = f"{here}.{part}"
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        try:
            obj = importlib.import_module(here) if isinstance(obj, type(gf)) else None
        except ModuleNotFoundError:
            obj = None
        if obj is None:
            return here
    return None


def printed_code_blocks(source):
    """
    Every block of code the source prints. In each string given to
    ``print``, a block starts at a line that starts code (`CODE_LINE`) and
    either opens a paragraph or is indented under the line before it (as code
    under "Example:" is). It runs on through blank lines and every line
    indented deeper than its first, or as deep and itself code or a closing
    bracket, and ends at any other line. Returns (line, block) pairs, the
    block dedented.
    """
    blocks = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
            continue
        for argument in node.args:
            if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                continue
            block, base, previous = [], 0, None     # previous: the last line's indent, None after a blank
            for line in argument.value.splitlines() + ["<end>"]:
                text, depth = line.strip(), len(line) - len(line.lstrip())
                if block:
                    if not text or depth > base or (depth == base and (CODE_LINE.match(text) or text[0] in ")]}")):
                        block.append(line)
                        continue
                    blocks.append((argument.lineno, textwrap.dedent("\n".join(block)).strip("\n")))
                    block = []
                if text and CODE_LINE.match(text) and (previous is None or depth > previous):
                    block, base = [line], depth
                previous = depth if text else None
    return blocks


def test_examples_use_only_existing_gravyflow_names():
    """Every gravyflow name an example uses exists."""
    found, absent = 0, []
    for label, source in example_sources():
        try:
            paths = gravyflow_names(source)
        except SyntaxError as error:
            raise AssertionError(f"{label} does not parse: {error}") from error
        found += len(paths)
        absent += [f"{label}: {path}" for path in sorted({missing(p) for p in paths} - {None})]
    assert found > 100, f"only {found} gravyflow names found; the census would be hollow"
    assert not absent, "Examples use gravyflow names that do not exist:\n  " + "\n  ".join(absent)


def test_printed_code_parses():
    """Every block of code an example prints parses."""
    found, broken = 0, []
    for label, source in example_sources():
        for line, block in printed_code_blocks(source):
            found += 1
            try:
                ast.parse(block)
            except SyntaxError as error:
                broken.append(f"{label}, string at line {line}, block {block.splitlines()[0]!r}: "
                              f"{error.msg} at its line {error.lineno}")
    assert found > 20, f"only {found} printed code blocks found; the census would be hollow"
    assert not broken, "Examples print code that does not parse:\n  " + "\n  ".join(broken)


def test_census_sees_a_missing_name():
    """The census itself: planted missing names, however they are reached, are found."""
    source = textwrap.dedent('''
        import gravyflow
        import gravyflow as gfl
        from gravyflow import NoSuchC
        from gravyflow.src.validate import NoSuchD, Validator as V
        x = gf.NoSuchObtainer()
        y = getattr(gf, "NoSuchE")
        gravyflow.NoSuchF()
        gfl.NoSuchG()
        V.no_such_method()
        gf.Dataset, gfl.NoiseObtainer, V.validate
        print("ds = gf.Dataset(noise=gf.NoSuchNoise)")
    ''')
    assert {missing(p) for p in gravyflow_names(source)} - {None} == {
        "gravyflow.NoSuchObtainer", "gravyflow.NoSuchNoise", "gravyflow.NoSuchC",
        "gravyflow.src.validate.NoSuchD", "gravyflow.NoSuchE", "gravyflow.NoSuchF", "gravyflow.NoSuchG",
        "gravyflow.src.validate.Validator.no_such_method",
    }


def test_census_sees_printed_code_that_does_not_parse():
    """The census itself: a printed block that does not parse is found, and prose is not code."""
    source = textwrap.dedent('''
        print("""
        Build the dataset like this:

            data = gf.Dataset(noise_obtainer=noise, group="train", ...)

        Then draw from it.
        """)
        print("""
        # A whole printed script
        noise = gf.NoiseObtainer(
            ifos=[gf.IFO.L1],
        )

        data = gf.Dataset(
            noise_obtainer=noise,
        )
        """)
        print("Returns the time series, for each detector.")
    ''')
    blocks = printed_code_blocks(source)
    assert len(blocks) == 2, blocks
    bad, good = blocks
    assert bad[1].strip().startswith("data = gf.Dataset(")
    try:
        ast.parse(bad[1])
    except SyntaxError as error:
        assert "positional argument follows keyword argument" in error.msg
    else:
        raise AssertionError("the planted block parsed")
    ast.parse(good[1])
