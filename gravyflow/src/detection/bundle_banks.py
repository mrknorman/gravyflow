"""
Place the template banks bundled with gravyflow and write them to
res/template_banks, or check the bundled files.

    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks                  # place and write every bank
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks NAME             # one bank, by its file name
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks --check NAME     # place again and compare
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks --coverage NAME  # check its coverage

Each file stores its placement's key (`BankPlacement.key`): the match space,
the noise curve's hash, the waveform model, the placement settings and the
hash of the gravyflow code that places it. `TemplateBank.place` loads a bundled
bank only when that key equals the key of the placement asked for, so after a
change to any of them (including any edit to the CBC generator or the
template-bank module) the bank is placed again, with a warning, until this
script is rerun. Each bank is named by its range, detector and noise curve.

What the key cannot name, such as a dependency's change within its release
series, is caught two ways. `--check` places the bank again and compares the
whole record: about three minutes for each window-mode bank. The streaming
filter's aligned-spin bank takes about an hour to place, so `--coverage`
checks it instead with the waveforms of today's environment: of
COVERAGE_INJECTIONS draws from its placement's proposal (a fixed seed), each
matched with its nearest templates, at most COVERAGE_BELOW may fall below the
minimal match.
"""

import argparse
import json
import sys
import time
from functools import partial

import numpy as np

from gravyflow.src.dataset.conditioning.detector import IFO
from gravyflow.src.detection.matched_filter import default_bank_placement
from gravyflow.src.detection.stream import STREAM_BANK_MAX_TEMPLATES, stream_bank_placement
from gravyflow.src.detection.template_grid import BUNDLED_BANK_DIRECTORY, TemplateBank
from gravyflow.src.resources import PACKAGE

# The bundled banks, by file name: the window-mode MatchedFilter's default for
# each LIGO detector, on that detector's measured O3 curve, and the streaming
# filter's default, on aLIGO's O4 high projection.
BUNDLED = {
    "bbh_5_95_mm0.97_2048hz_1.5s_20hz_h1_o3actual.json": partial(default_bank_placement, ifo=IFO.H1),
    "bbh_5_95_mm0.97_2048hz_1.5s_20hz_l1_o3actual.json": partial(default_bank_placement, ifo=IFO.L1),
    "bbh_aligned_5_95_s0.99_mm0.97_2048hz_24s_20hz_o4high.json": stream_bank_placement,
}

# The coverage check: placement stops with about 1% of its proposal's measure
# below the minimal match, so of 400 draws about 4 fall below it; more than 12
# has probability 1e-4 for a sound bank (binomial, 1%) and 0.5 for a bank
# leaving 3% uncovered.
COVERAGE_INJECTIONS = 400
COVERAGE_BELOW = 12
COVERAGE_SEED = 20261019
COVERAGE_NEIGHBOURS = 32


def coverage(bank: TemplateBank) -> np.ndarray:
    """The fitting factors (a lower bound: nearest templates only) of the coverage check's draws."""
    m1, m2, s1, s2 = bank.placement.proposal(np.random.default_rng(COVERAGE_SEED), COVERAGE_INJECTIONS)
    return bank.fitting_factors(m1, m2, None, s1, s2, neighbours=COVERAGE_NEIGHBOURS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Place the template banks bundled with gravyflow.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Place again and compare with the bundled file")
    mode.add_argument("--coverage", action="store_true", help="Check the bundled bank's coverage")
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help=f"the banks, by file name (default every one): {', '.join(BUNDLED)}")
    args = parser.parse_args(argv)
    unknown = [name for name in args.names if name not in BUNDLED]
    if unknown:
        parser.error(f"no bundled bank is named {', '.join(unknown)}")
    directory = PACKAGE / BUNDLED_BANK_DIRECTORY
    current = True
    for name in args.names or BUNDLED:
        path = directory / name
        start = time.time()
        if args.coverage:
            bank = TemplateBank.from_dict(json.loads(path.read_text()))
            if bank.to_dict()["arguments"] != json.loads(json.dumps(BUNDLED[name]().key())):
                print(f"{name}: stale: its key is not today's placement's; place it again")
                current = False
                continue
            ff = coverage(bank)
            below = int(np.sum(ff < bank.minimal_match))
            print(f"{name}: {below} of {len(ff)} draws below the minimal match (at most {COVERAGE_BELOW}); "
                  f"median {np.median(ff):.4f}, 1st percentile {np.percentile(ff, 1):.4f}, "
                  f"in {time.time() - start:.0f} s")
            current = current and below <= COVERAGE_BELOW
            continue
        bank = TemplateBank.place(BUNDLED[name](), max_templates=STREAM_BANK_MAX_TEMPLATES, use_bundled=False)
        record = bank.to_dict()
        print(f"{name}: placed {bank.num_templates} templates from {bank.candidates_tried} candidates in "
              f"{time.time() - start:.0f} s (final acceptance {bank.final_acceptance:.4f})")
        if args.check:
            same = path.is_file() and json.loads(path.read_text()) == record
            print(f"{name}: identical to a fresh placement: {same}")
            current = current and same
        else:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1) + "\n")
            print(f"{name}: written")
    return 0 if current else 1


if __name__ == "__main__":
    sys.exit(main())
