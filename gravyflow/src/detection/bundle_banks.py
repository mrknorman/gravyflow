"""
Place the template banks bundled with gravyflow and write them to
res/template_banks, or check the bundled files against a fresh placement.

    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks               # place and write every bank
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks --check       # place and compare every bank
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks --check NAME  # one bank, by its file name

Each file stores its placement's key (`BankPlacement.key`): the match space,
the noise curve's hash, the waveform model, the placement settings and the
hash of the gravyflow code that places it (`placement_code`: the syntax of the
template-bank module and of the CBC generator's code it uses, so a comment or
docstring does not count). `TemplateBank.place` loads a bundled bank only when
that key equals the key of the placement asked for, so after a change to any of
them the bank is placed again, with a warning, until this script is rerun. `--check` also catches what the key cannot name, such as a
dependency's change within its release series. Each bank is named by its
detector and noise curve; placing one takes about three minutes on a CPU.
"""

import argparse
import json
import sys
import time
from functools import partial

from gravyflow.src.dataset.conditioning.detector import IFO
from gravyflow.src.detection.matched_filter import default_bank_placement
from gravyflow.src.detection.template_grid import BUNDLED_BANK_DIRECTORY, TemplateBank
from gravyflow.src.resources import PACKAGE

# The bundled banks, by file name: the default MatchedFilter's for each LIGO
# detector, on that detector's measured O3 curve.
BUNDLED = {
    "bbh_5_95_mm0.97_2048hz_1.5s_20hz_h1_o3actual.json": partial(default_bank_placement, ifo=IFO.H1),
    "bbh_5_95_mm0.97_2048hz_1.5s_20hz_l1_o3actual.json": partial(default_bank_placement, ifo=IFO.L1),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Place the template banks bundled with gravyflow.")
    parser.add_argument("--check", action="store_true", help="Compare with the bundled files instead of writing")
    parser.add_argument("names", nargs="*", metavar="NAME",
                        help=f"the banks to place, by file name (default every one): {', '.join(BUNDLED)}")
    args = parser.parse_args(argv)
    unknown = [name for name in args.names if name not in BUNDLED]
    if unknown:
        parser.error(f"no bundled bank is named {', '.join(unknown)}")
    directory = PACKAGE / BUNDLED_BANK_DIRECTORY
    current = True
    for name in args.names or BUNDLED:
        start = time.time()
        bank = TemplateBank.place(BUNDLED[name](), use_bundled=False)
        record = bank.to_dict()
        print(f"{name}: placed {bank.num_templates} templates from {bank.candidates_tried} candidates in "
              f"{time.time() - start:.0f} s (final acceptance {bank.final_acceptance:.4f})")
        path = directory / name
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
