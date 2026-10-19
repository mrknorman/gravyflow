"""
Reserved time: stretches of GPS time, per detector, held out of every train and
validate draw of real data, so that no model gravyflow trains or selects has seen
a benchmark's test strain. ``SplitSpec.reserved`` holds them; test keeps the time.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import coalesce_segments


def _detector_name(detector) -> str:
    """A detector's name, from a ``gf.IFO`` or its name."""
    return detector if isinstance(detector, str) else detector.name


@dataclass(frozen=True)
class Reservation:
    """
    Time held out of every train and validate draw of real data: the strain a
    benchmark tests on, so that no model gravyflow trains or selects has seen it.
    Test keeps it.

    Args:
        name: What the reservation is called in ``SplitSpec.mode`` and in the
            reasons records are left out.
        intervals: ((detector, (N, 2) GPS intervals), ...), each detector's reserved
            time, the detector a ``gf.IFO`` or its name, each named once; kept by
            name, sorted, coalesced and in detector order. A detector not named has
            no reserved time. A reservation that reserves no time is refused: it
            would name a mode while holding nothing out.
    """

    name: str
    # Out of the repr: hundreds of intervals would swamp every message naming a split.
    intervals: Tuple[Tuple[str, Tuple[Tuple[float, float], ...]], ...] = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"A reservation's name must be a non-empty string, got {self.name!r}.")
        intervals = {}
        for detector, times in self.intervals:
            name = _detector_name(detector)
            if name in intervals:
                raise ValueError(f"Reservation {self.name!r} names detector {name} twice; name each once.")
            intervals[name] = coalesce_segments(np.reshape(np.asarray(times, dtype=np.float64), (-1, 2)))
        object.__setattr__(self, "intervals", tuple(
            (name, tuple(map(tuple, intervals[name].tolist()))) for name in sorted(intervals)
        ))
        # Derived once, here, so a reservation is the same object (pickle-equal)
        # before and after any use: each detector's time as an array, every
        # detector's under None, none under ''; and the time's sha256. Pickling
        # rebuilds a reservation from its fields (__reduce__), so these are derived
        # again, read-only, rather than restored writeable.
        time = {name: np.array(times, dtype=np.float64).reshape(-1, 2) for name, times in self.intervals}
        time[None] = coalesce_segments(np.concatenate([np.empty((0, 2))] + list(time.values())))
        time[""] = np.empty((0, 2))
        if len(time[None]) == 0:  # coalescing drops intervals of zero length
            raise ValueError(
                f"Reservation {self.name!r} reserves no time on any detector, so it would hold nothing out; "
                "for no reservation, leave it out of SplitSpec.reserved."
            )
        for array in time.values():
            array.flags.writeable = False
        object.__setattr__(self, "_time", time)
        object.__setattr__(self, "_sha256", hashlib.sha256(json.dumps(self.intervals).encode()).hexdigest())

    def __reduce__(self):
        return type(self), (self.name, self.intervals)

    def of(self, detector=None) -> np.ndarray:
        """
        ``detector``'s reserved time ((N, 2) GPS, read-only); every detector's,
        coalesced, for None.
        """
        if detector is None:
            return self._time[None]
        return self._time.get(_detector_name(detector), self._time[""])

    @property
    def sha256(self) -> str:
        """The sha256 of the reserved time, its identity in fingerprints."""
        return self._sha256

    def specification(self) -> Dict[str, str]:
        """What fingerprints read (``utils.fingerprint.canonical``): its name and time's sha256."""
        return {"name": self.name, "sha256": self.sha256}
