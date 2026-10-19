"""
MLGWSC-1, the first machine-learning gravitational-wave search mock data challenge
(Schäfer, Zelenka, Nitz, Wang et al. 2023, arXiv:2209.11146), as a data source.

Its files are entries of gravyflow's manifest of pinned remote files
(``resources.MANIFEST``), fetched only when asked (``gravyflow.fetch``). The segment
list ships with gravyflow too, so its reservation holds offline.

Its fourth dataset is real O3a noise from 362 segments (``segments.csv``): H1 as
recorded, L1 shifted by up to 240 s within each segment. The shift moves L1 strain
from one place in a segment to another, never in from outside it, so every sample the
challenge reads lies inside a segment on both detectors. So that no model gravyflow
trains or selects has seen that strain, ``MLGWSC1_O3A`` reserves exactly those
segments, on H1 and L1, from every train and validate draw by default
(``gf.SplitSpec.reserved``). The challenge-faithful mode,
``gf.SplitSpec(reserved=())``, trains on it as the challenge's entries could; every
fingerprint says which mode a result came from.
"""

import csv

import numpy as np

from gravyflow.src.dataset.reservation import Reservation

#: The challenge's segment list, by its manifest name.
SEGMENTS = "mlgwsc1/segments.csv"


def segments() -> np.ndarray:
    """The challenge's 362 O3a segments, (362, 2) GPS, from the shipped ``segments.csv``."""
    from gravyflow.src.resources import path

    with open(path(SEGMENTS), newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["", "start", "end"]:
        raise ValueError(f"{SEGMENTS} has the header {rows[0]}, not the challenge's ['', 'start', 'end'].")
    return np.array([[int(start), int(end)] for _, start, end in rows[1:]], dtype=np.float64)


#: MLGWSC-1's test strain, held out of train and validate by default.
MLGWSC1_O3A = Reservation("mlgwsc1-o3a", tuple(dict.fromkeys(("H1", "L1"), segments()).items()))
