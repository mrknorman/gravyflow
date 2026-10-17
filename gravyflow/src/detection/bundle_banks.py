"""
Place the template banks bundled with gravyflow and write them to
res/template_banks, or check the bundled files against a fresh placement.

    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks          # place and write
    KERAS_BACKEND=jax python -m gravyflow.src.detection.bundle_banks --check  # place and compare

Each file stores its placement's key (`BankPlacement.key`): the match space,
the noise curve's hash, the waveform model, the placement settings and the
hash of the gravyflow code that places it. `TemplateBank.place` loads a bundled
bank only when that key equals the key of the placement asked for, so after a
change to any of them (including any edit to the CBC generator or the
template-bank module) the bank is placed again, with a warning, until this
script is rerun. `--check` also catches what the key cannot name, such as a
dependency's change within its release series. Placing the default bank takes
about two minutes on a CPU.
"""

import argparse
import json
import sys
import time

from gravyflow.src.detection.matched_filter import default_bank_placement
from gravyflow.src.detection.template_grid import BUNDLED_BANK_DIRECTORY, TemplateBank
from gravyflow.src.resources import PACKAGE

# The bundled banks, by file name: the default MatchedFilter's.
BUNDLED = {
    "bbh_5_95_mm0.97_2048hz_1.5s_20hz_l1.json": default_bank_placement,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Place the template banks bundled with gravyflow.")
    parser.add_argument("--check", action="store_true", help="Compare with the bundled files instead of writing")
    args = parser.parse_args(argv)
    directory = PACKAGE / BUNDLED_BANK_DIRECTORY
    current = True
    for name, placement in BUNDLED.items():
        start = time.time()
        bank = TemplateBank.place(placement(), use_bundled=False)
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
