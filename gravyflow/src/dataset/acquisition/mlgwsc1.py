"""
MLGWSC-1, the first machine-learning gravitational-wave search mock data challenge
(Schäfer, Zelenka, Nitz, Wang et al. 2023, arXiv:2209.11146), as a data source.

Its files are entries of gravyflow's manifest of pinned remote files
(``resources.MANIFEST``), fetched only when asked (``gravyflow.fetch``). The segment
list ships with gravyflow too, so its reservation holds offline.

Its fourth dataset is real O3a noise from 362 segments (``segments.csv``): H1 as
recorded, L1 shifted within each segment by up to 240 s. So that no model
gravyflow trains or selects has seen that strain, ``MLGWSC1_O3A`` reserves it from
every train and validate draw by default (``gf.SplitSpec.reserved``): H1 over each
segment, and L1 over each segment widened by 240 s on both sides. The
challenge-faithful mode, ``gf.SplitSpec(reserved=())``, trains on it as the
challenge's entries could; every fingerprint says which mode a result came from.
"""

import csv

import numpy as np

from gravyflow.src.dataset.reservation import Reservation

#: The challenge's segment list, by its manifest name.
SEGMENTS = "mlgwsc1/segments.csv"

#: How far the challenge shifts L1 within a segment, at most, in seconds.
L1_GUARD_SECONDS = 240.0


def segments() -> np.ndarray:
    """The challenge's 362 O3a segments, (362, 2) GPS, from the shipped ``segments.csv``."""
    from gravyflow.src.resources import path

    with open(path(SEGMENTS), newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["", "start", "end"]:
        raise ValueError(f"{SEGMENTS} has the header {rows[0]}, not the challenge's ['', 'start', 'end'].")
    return np.array([[int(start), int(end)] for _, start, end in rows[1:]], dtype=np.float64)


#: MLGWSC-1's test strain, held out of train and validate by default.
MLGWSC1_O3A = Reservation.from_segments(
    "mlgwsc1-o3a", segments(), {"H1": 0.0, "L1": L1_GUARD_SECONDS})
