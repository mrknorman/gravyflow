"""
Partner times: where a transient's example reads a detector the transient was not
seen in.

A glitch is seen in one detector. Reading the other detectors at its time would
give every example noise that is coincident across detectors, a correlation no
single-detector glitch has and a model could learn. Each other detector an example
needs is read instead at a partner time t' of its own (``partner_time``):

- inside ``allowed``: that detector's usable data, less its own glitches and every
  event, inside the record's split (the obtainer builds it);
- inside one block of the record's split, so a partner is in that split whatever
  the split's fractions, and near the record in time: a glitch's own block, and an
  event's nearest such block in its observing run, since an event is in ``test``
  whatever its own block's split;
- with its centre at least ``separation`` from the transient's time;
- on the partner grid (``PARTNER_GRID_HERTZ``), chosen by a hash of ``key`` (the split's fingerprint,
  the transient's time and the detector): the same partner on every build, from no
  random state.

A record with no partner time on a detector it needs is not served.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.dataset.splits import hash_uniform
from .gwosc_data import subtract_segments

#: The grid a partner's centre lies on, in hertz. It fixes which partner a key draws, so
#: it belongs to the plan; a read centres any time on its own rate's grid.
PARTNER_GRID_HERTZ = 4096.0


def partner_time(
    allowed: np.ndarray,
    gps: float,
    before: float,
    after: float,
    block: Tuple[float, float],
    separation: float,
    key: Sequence,
    sample_rate_hertz: float,
) -> Optional[float]:
    """
    The partner time of a transient at ``gps``, or None when there is none.

    Args:
        allowed: Sorted, coalesced (N, 2) GPS intervals a partner's span may read.
        gps: The transient's time.
        before, after: The span a read takes, before and after its centre.
        block: The GPS interval [start, end) the partner's span lies in.
        separation: The least distance between the partner's centre and ``gps``.
        key: What the choice is a hash of.
        sample_rate_hertz: The grid the partner's centre lies on.

    Returns:
        A centre on the grid whose span ``[t' - before, t' + after]`` lies inside
        ``allowed`` and ``block``, with ``|t' - gps| >= separation``; each such grid
        point equally likely over ``key``.
    """
    allowed = np.reshape(np.asarray(allowed, dtype=np.float64), (-1, 2))
    start, end = float(block[0]), float(block[1])
    pieces = allowed[np.searchsorted(allowed[:, 1], start, side="right"):
                     np.searchsorted(allowed[:, 0], end, side="left")]
    centres = np.stack([
        np.maximum(pieces[:, 0], start) + before,
        np.minimum(pieces[:, 1], end) - after,
    ], axis=1)
    centres = subtract_segments(centres, [[gps - separation, gps + separation]])
    first = np.ceil(centres[:, 0] * sample_rate_hertz)
    counts = np.maximum(np.floor(centres[:, 1] * sample_rate_hertz) - first + 1.0, 0.0)
    total = float(np.sum(counts))
    if total == 0.0:
        return None
    k = min(int(hash_uniform(*key) * total), int(total) - 1)
    ends = np.cumsum(counts)
    piece = int(np.searchsorted(ends, k, side="right"))
    sample = first[piece] + (k - (ends[piece] - counts[piece]))
    return float(sample / sample_rate_hertz)
