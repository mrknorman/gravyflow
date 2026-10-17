"""
The whole suite's coverage against its floor:
``python -m gravyflow.tests.coverage_floor --label NAME RUN [RUN ...]``.

The suite runs in parts, each measuring coverage into its own data file,
``.coverage.<RUN>`` in the working directory. This combines exactly those files
(a missing one is refused by name, never treated as covering nothing), reports
the total, and compares it with the floor, ``[tool.coverage.report]
fail_under`` in ``pyproject.toml``, its one home. Below the floor it prints
``NAME FAIL <total> (floor <floor>)`` and exits 1; otherwise ``NAME PASS <total>
(floor <floor>)``, and, when the total exceeds the floor by more than
``RATCHET_MARGIN``, ``RAISE FLOOR TO <total - RATCHET_SLACK>``: the change that
sees it raises the floor in the same commit. The data files are kept, so the
report can be read again with ``coverage report``.
"""
import argparse
import io
import sys
from pathlib import Path

import coverage
from coverage.results import should_fail_under

#: How far above the floor the total may sit before the floor is raised.
RATCHET_MARGIN = 1.0
#: How far below the total a raised floor sits, so run-to-run jitter never fails it.
RATCHET_SLACK = 0.5


def main(arguments) -> int:
    parser = argparse.ArgumentParser(prog="python -m gravyflow.tests.coverage_floor")
    parser.add_argument("--label", required=True, help="the verdict line's first word")
    parser.add_argument("runs", nargs="+", help="the parts whose .coverage.<run> files to combine")
    options = parser.parse_args(arguments)

    data_files = [Path(f".coverage.{run}").resolve() for run in options.runs]
    missing = [str(path) for path in data_files if not path.is_file()]
    if missing:
        print(f"{options.label} REFUSED: no coverage data at {', '.join(missing)}", file=sys.stderr)
        return 1

    measure = coverage.Coverage()
    measure.combine([str(path) for path in data_files], strict=True, keep=True)
    measure.save()
    total = measure.report(file=io.StringIO())
    precision = measure.config.precision
    floor = measure.config.fail_under

    verdict = "FAIL" if should_fail_under(total, floor, precision) else "PASS"
    print(f"{options.label} {verdict} {total:.{precision}f} (floor {floor:.{precision}f})")
    if verdict == "PASS" and total > floor + RATCHET_MARGIN:
        print(f"RAISE FLOOR TO {total - RATCHET_SLACK:.{precision}f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
