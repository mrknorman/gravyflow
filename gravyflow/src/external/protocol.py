"""
What an external search implements, and the segment it searches.

An external model (a published search with released weights, written for another
framework) is reached through an adapter: an object with the three methods of
``ExternalSearch``, published by its package under the entry-point group
``ADAPTER_GROUP``. gravyflow never imports an adapter: the runner
(``external.runner``) starts it in its own environment (``external.child``), where
this module is loaded by path, beside numpy and nothing else of gravyflow's.

So this module imports only the standard library and numpy, and keeps to the Python
3.8 grammar, the oldest an adapter's environment may run (``child.ADAPTER_PYTHON``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Tuple, runtime_checkable

import numpy as np

#: The entry-point group adapters are published under, each by a name a card gives.
ADAPTER_GROUP = "gravyflow.external"


@dataclass(frozen=True, eq=False)
class SearchSegment:
    """
    The unit an external search runs on: one stretch of strain, every detector over
    the same samples, searched whole.

    Args:
        strain: The strain, shape (detectors, samples), taken as float64, read-only.
        detectors: The detectors' names, one per row, in order (``("H1", "L1")``).
        t0: The GPS time of the first sample, in seconds.
        sample_rate_hertz: The sampling rate.
    """

    strain: np.ndarray
    detectors: Tuple[str, ...]
    t0: float
    sample_rate_hertz: float

    def __post_init__(self):
        strain = np.array(self.strain, dtype=np.float64)
        detectors = tuple(self.detectors)
        if strain.ndim != 2 or strain.shape[0] != len(detectors) or strain.shape[1] == 0:
            raise ValueError(
                f"A segment's strain is (detectors, samples) with one row per detector {detectors}, "
                f"got shape {strain.shape}.")
        if len(set(detectors)) != len(detectors) or not all(isinstance(d, str) and d for d in detectors):
            raise ValueError(f"A segment's detectors are distinct names, got {detectors}.")
        if not np.isfinite(strain).all():
            raise ValueError("A segment's strain holds a NaN or an infinity.")
        if not float(self.sample_rate_hertz) > 0:
            raise ValueError(f"A segment's sample rate must be above zero, got {self.sample_rate_hertz}.")
        strain.setflags(write=False)
        object.__setattr__(self, "strain", strain)
        object.__setattr__(self, "detectors", detectors)
        object.__setattr__(self, "t0", float(self.t0))
        object.__setattr__(self, "sample_rate_hertz", float(self.sample_rate_hertz))

    @property
    def duration_seconds(self) -> float:
        """Its length in seconds: samples over the rate."""
        return self.strain.shape[1] / self.sample_rate_hertz

    def __getitem__(self, detector: str) -> np.ndarray:
        """The strain of the detector named ``detector``."""
        try:
            return self.strain[self.detectors.index(detector)]
        except ValueError:
            raise KeyError(f"The segment has no detector {detector!r}; it has {self.detectors}.") from None

    @classmethod
    def from_strains(cls, strains: Mapping[str, Any]) -> "SearchSegment":
        """
        The segment of one strain per detector, in the mapping's order: each value
        has ``data``, ``t0`` and ``sample_rate_hertz`` (a data source's
        ``SegmentStrain``), and all share their first sample, rate and length.
        """
        if not strains:
            raise ValueError("A segment needs the strain of at least one detector.")
        values = list(strains.values())
        starts = {float(s.t0) for s in values}
        rates = {float(s.sample_rate_hertz) for s in values}
        lengths = {len(s.data) for s in values}
        if len(starts) != 1 or len(rates) != 1 or len(lengths) != 1:
            raise ValueError(
                "A segment's detectors share their first sample, rate and length; got starts "
                f"{sorted(starts)}, rates {sorted(rates)} and lengths {sorted(lengths)}.")
        return cls(np.stack([np.asarray(s.data) for s in values]), tuple(strains), starts.pop(), rates.pop())


@runtime_checkable
class ExternalSearch(Protocol):
    """
    An external model as gravyflow runs it: an adapter publishes a class with these
    methods under ``ADAPTER_GROUP``; the runner makes one with no arguments.

    - ``prepare(fetch)`` loads the model. ``fetch(name)`` is the local path of the
      weights file its card names ``name``, fetched by gravyflow, pinned by sha256;
      an adapter reads its weights from nowhere else.
    - ``preprocess(segment)`` is the input the model sees for ``segment``, as one
      array: the hook a reproduction compares with the published code's input.
    - ``search(segment)`` runs the published search over the whole segment and
      returns its triggers: anything with ``time``, ``stat`` and ``var`` arrays
      (``detection.triggers.Triggers``, or a named tuple where gravyflow is not
      installed). ``stat`` is the published statistic as published.
    """

    def prepare(self, fetch: Callable[[str], Path]) -> None:
        ...

    def preprocess(self, segment: SearchSegment) -> np.ndarray:
        ...

    def search(self, segment: SearchSegment) -> Any:
        ...


__all__ = ["ADAPTER_GROUP", "ExternalSearch", "SearchSegment"]
