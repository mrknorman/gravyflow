"""
The example notebooks as committed: their code and prose only. An output is a
snapshot of one run on one machine, so a committed one goes stale with the code
and bloats every diff.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[2] / "example_notebooks").rglob("*.ipynb"))


def test_the_notebooks_are_found():
    assert NOTEBOOKS, "No example notebook found: the check below would pass over nothing."


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[path.name for path in NOTEBOOKS])
def test_a_committed_notebook_carries_no_output(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    run = [index for index, cell in enumerate(cells)
           if cell["cell_type"] == "code" and (cell.get("outputs") or cell.get("execution_count") is not None)]
    assert not run, f"{notebook.name}: cells {run} hold outputs or execution counts; strip them before committing."
