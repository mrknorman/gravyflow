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

Its simulated datasets (1 to 3) take their time from the same list
(``restrict_segments``); ``mlgwsc1_stream`` makes dataset 1 on it.
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

#: A dataset's length: 30 days (generate_data.py's --duration default).
DURATION_SECONDS = 2_592_000

#: The shortest segment a simulated dataset (1 to 3) takes (get_noise's min_segment_duration).
MIN_SEGMENT_SECONDS = 7200


def restrict_segments(start_offset: int = 0, duration: int = DURATION_SECONDS,
                      min_segment_duration: int = MIN_SEGMENT_SECONDS) -> np.ndarray:
    """
    The segments of a simulated MLGWSC-1 dataset, (S, 2) integer GPS: generate_data.py's
    ``restrict_segments`` over ``segments()`` with no slide buffer, as ``get_noise`` calls it
    for datasets 1 to 3, line for line (its edge cases included, so the time is the
    challenge's). Segments shorter than ``min_segment_duration`` are skipped, the first
    ``start_offset`` seconds of the rest are skipped, and ``duration`` seconds are taken;
    the last segment may be shorter. A request beyond the 362 segments is refused, where
    the challenge would warn and return less.
    """
    if start_offset < 0 or duration <= 0:
        raise ValueError(f"A dataset starts at an offset of 0 s or more and lasts more than 0 s; "
                         f"got {start_offset} s and {duration} s.")
    past, kept = 0, []
    for start, end in segments().astype(np.int64).tolist():
        if past - start_offset >= duration:
            continue
        length = end - start
        if min_segment_duration is not None and length < min_segment_duration:
            continue
        if past + length < start_offset:
            past += length
            continue
        if past < start_offset:
            start += start_offset - past
            length = end - start
            past = start_offset
            if min_segment_duration is not None and length < min_segment_duration:
                continue
        if past + length > start_offset + duration:
            end -= past + length - (start_offset + duration)
            length = end - start
        kept.append([start, end])
        past += length
    if past < start_offset + duration:
        raise ValueError(
            f"MLGWSC-1's segments hold {past - start_offset} s after an offset of {start_offset} s, "
            f"not the {duration} s asked for.")
    # ligo.segments' coalesce: segments that touch become one.
    coalesced = [kept[0]]
    for start, end in kept[1:]:
        if start <= coalesced[-1][1]:
            coalesced[-1][1] = max(coalesced[-1][1], end)
        else:
            coalesced.append([start, end])
    return np.array(coalesced, dtype=np.int64)
