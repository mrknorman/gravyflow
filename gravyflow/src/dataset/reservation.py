"""
Reserved time: stretches of GPS time, per detector, held out of every train and
validate draw of real data, so that no model gravyflow trains or selects has seen
a benchmark's test strain. ``SplitSpec.reserved`` holds them; test keeps the time.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

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
        intervals: {detector name: (N, 2) GPS intervals}, each detector's reserved
            time; kept sorted, coalesced and in detector order. A detector not named
            has no reserved time.
    """

    name: str
    intervals: Tuple[Tuple[str, Tuple[Tuple[float, float], ...]], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"A reservation's name must be a non-empty string, got {self.name!r}.")
        intervals = dict(self.intervals)
        object.__setattr__(self, "intervals", tuple(
            (_detector_name(detector), tuple(map(tuple, coalesce_segments(intervals[detector]).tolist())))
            for detector in sorted(intervals, key=_detector_name)
        ))
        # Derived once, here, so a reservation is the same object (pickle-equal)
        # before and after any use: each detector's time as an array, every
        # detector's under None, none under ''; and the time's sha256.
        time = {name: np.array(times, dtype=np.float64).reshape(-1, 2) for name, times in self.intervals}
        time[None] = coalesce_segments(np.concatenate([np.empty((0, 2))] + list(time.values())))
        time[""] = np.empty((0, 2))
        for array in time.values():
            array.flags.writeable = False
        object.__setattr__(self, "_time", time)
        object.__setattr__(self, "_sha256", hashlib.sha256(json.dumps(self.intervals).encode()).hexdigest())

    @classmethod
    def from_segments(cls, name: str, segments, guards: Mapping[str, float]) -> "Reservation":
        """
        ``segments`` ((N, 2) GPS) reserved on each detector of ``guards``, widened on
        both sides by that detector's guard in seconds.
        """
        segments = np.reshape(np.asarray(segments, dtype=np.float64), (-1, 2))
        if any(not guard >= 0.0 for guard in guards.values()):
            raise ValueError(f"A reservation's guards must be non-negative, got {dict(guards)}.")
        return cls(name, tuple(
            (detector, segments + np.array([-float(guard), float(guard)])) for detector, guard in guards.items()
        ))

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
