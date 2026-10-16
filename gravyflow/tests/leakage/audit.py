"""
Collection and assertion helpers for decoded footprints.

``collect`` draws batches eagerly, decodes every row and records each set's
footprints. Each assertion refuses an empty input and a set with too few examples,
and reports what it found (the minimum gap, the offending rows), never a bare bool.
Interval arithmetic uses gwpy's SegmentList, the system the library uses.

A footprint is per detector: [offsource first sample, padded-onsource end), held as
integer sample indices at the fixture's served rate and converted to GPS seconds
exactly (the served rate is a power of two).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from gwpy.segments import Segment, SegmentList

from . import fixtures as fx
from .fakes import Harness
from .provenance import Window, as_seconds, batch_arrays, footprints, ifo_names

#: The fewest examples a set may have before an assertion will judge it.
MIN_EXAMPLES = 8


@dataclass
class Records:
    """The decoded footprints of one set of batches."""

    name: str
    split: str
    ifos: Tuple[str, ...]
    footprints: np.ndarray  # (N, I, 2) int64 sample indices at the served rate
    sample_rate_hertz: float
    start_labels: Optional[np.ndarray] = None  # (N, I) START_GPS_TIME, when returned
    fetched: int = 0
    warm: bool = False

    @property
    def num_examples(self) -> int:
        return int(self.footprints.shape[0])

    def seconds(self) -> np.ndarray:
        return np.asarray(self.footprints, dtype=np.float64) / self.sample_rate_hertz

    def segments(self, ifo: Optional[str] = None) -> SegmentList:
        """The footprints of one detector (or all), coalesced, in GPS seconds."""
        secs = self.seconds()
        columns = range(len(self.ifos)) if ifo is None else [self.ifos.index(ifo)]
        return SegmentList(
            Segment(float(a), float(b)) for d in columns for a, b in secs[:, d]
        ).coalesce()


def collect(
    harness: Harness,
    sets: Mapping[str, Tuple[str, Iterable]],
    n_batches: int,
    ifos: Sequence,
    window: Window,
    onsource_cropped: bool,
    stride: int = 1,
    warm: bool = False,
    adjacent: bool = True,
) -> Dict[str, Records]:
    """
    Draw ``n_batches`` from each iterator in ``sets`` ({name: (split, iterator)}),
    eagerly, and decode them.

    Refuses: a fetch error; on a cold build, a set that fetched nothing; on a warm
    build (``warm=True``), a set that fetched anything.
    """
    import gravyflow as gf

    if not sets:
        raise AssertionError("collect: no sets to collect")
    names = ifo_names(ifos)
    out = {}
    for name, (split, iterator) in sets.items():
        before = len(harness.source.snapshot()[0])
        rows_on, rows_off, labels = [], [], []
        iterator = iter(iterator)
        for _ in range(n_batches):
            batch = next(iterator)
            arrays = batch_arrays(batch, [gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE])
            rows_on.append(arrays[gf.ReturnVariables.ONSOURCE])
            rows_off.append(arrays[gf.ReturnVariables.OFFSOURCE])
            try:
                labels.append(batch_arrays(batch, [gf.ReturnVariables.START_GPS_TIME])[
                    gf.ReturnVariables.START_GPS_TIME])
            except KeyError:
                pass
        fetches, errors = harness.source.snapshot()
        if errors:
            raise AssertionError(f"{name}: the fake refused {len(errors)} fetch(es): {errors[:3]}")
        fetched = len(fetches) - before
        if not warm and fetched == 0:
            raise AssertionError(f"{name}: a cold build fetched nothing; its data did not come from the fake")
        if warm and fetched != 0:
            raise AssertionError(f"{name}: a warm build fetched {fetched} time(s); it should read its cache")
        prints = footprints(
            np.concatenate(rows_on), np.concatenate(rows_off), names, harness.encoding,
            window, onsource_cropped, stride, adjacent,
        )
        out[name] = Records(
            name, split, names, prints, harness.encoding.sample_rate_hertz,
            np.concatenate(labels) if labels else None, fetched, warm,
        )
    return out


def _check(records: Iterable[Records], minimum: int = MIN_EXAMPLES) -> List[Records]:
    records = list(records)
    if not records:
        raise AssertionError("no records to judge")
    for record in records:
        if record.num_examples < minimum:
            raise AssertionError(
                f"{record.name} has {record.num_examples} examples; at least {minimum} are needed"
            )
    return records


def _as_segment_lists(intervals) -> Dict[str, SegmentList]:
    """{ifo: SegmentList} from {ifo: [(a, b), ...] | SegmentList | (N, 2) array}."""
    out = {}
    for ifo, value in intervals.items():
        out[ifo] = SegmentList(Segment(float(a), float(b)) for a, b in value).coalesce()
    return out


def fetch_intervals(harness: Harness) -> Dict[str, np.ndarray]:
    """Every served fetch, per detector: (M, 2) int64 [first, stop) sample indices."""
    fetches, _ = harness.source.snapshot()
    out: Dict[str, List[Tuple[int, int]]] = {}
    for fetch in fetches:
        out.setdefault(fetch.ifo, []).append((fetch.first, fetch.stop))
    return {ifo: np.array(v, dtype=np.int64) for ifo, v in out.items()}


def assert_inside_one_fetch(records: Iterable[Records], harness: Harness) -> None:
    """Every footprint lies inside one served fetch of its own detector."""
    intervals = fetch_intervals(harness)
    if not intervals:
        raise AssertionError("the fetch log is empty")
    bad = []
    for record in _check(records):
        for d, ifo in enumerate(record.ifos):
            served = intervals.get(ifo)
            prints = record.footprints[:, d]
            if served is None:
                bad.append((record.name, ifo, "no fetch"))
                continue
            inside = (served[None, :, 0] <= prints[:, None, 0]) & (prints[:, None, 1] <= served[None, :, 1])
            for b in np.flatnonzero(~inside.any(axis=1)):
                bad.append((record.name, int(b), ifo, tuple(as_seconds(prints[b], harness.encoding))))
    if bad:
        raise AssertionError(f"{len(bad)} footprint(s) outside every fetch of their detector: {bad[:5]}")


def assert_inside(records: Iterable[Records], intervals) -> None:
    """Every footprint lies inside one interval of its own detector ({ifo: intervals})."""
    allowed = _as_segment_lists(intervals)
    bad = []
    for record in _check(records):
        secs = record.seconds()
        for d, ifo in enumerate(record.ifos):
            for b, (a, z) in enumerate(secs[:, d]):
                if Segment(float(a), float(z)) not in allowed.get(ifo, SegmentList()):
                    bad.append((record.name, b, ifo, (float(a), float(z))))
    if bad:
        raise AssertionError(f"{len(bad)} footprint(s) outside the allowed intervals: {bad[:5]}")


def assert_in_split(records: Iterable[Records], origin: str) -> None:
    """Every footprint lies in one block of the golden table, of its set's split."""
    bad = []
    for record in _check(records):
        secs = record.seconds()
        for b, d in np.ndindex(secs.shape[:2]):
            a, z = secs[b, d]
            try:
                split = fx.golden_split(origin, a, z)
            except AssertionError as error:
                bad.append((record.name, b, record.ifos[d], str(error)))
                continue
            if split != record.split:
                bad.append((record.name, b, record.ifos[d], f"in a {split} block", (float(a), float(z))))
    if bad:
        raise AssertionError(f"{len(bad)} footprint(s) outside their split's blocks: {bad[:5]}")


def _distance(first: SegmentList, second: SegmentList) -> Tuple[float, Optional[Tuple]]:
    best, pair = np.inf, None
    for a in first:
        for b in second:
            gap = max(float(b[0]) - float(a[1]), float(a[0]) - float(b[1]))
            if gap < best:
                best, pair = gap, (tuple(map(float, a)), tuple(map(float, b)))
    return best, pair


def assert_split_gap(records: Iterable[Records], min_gap_seconds: float) -> None:
    """
    Every footprint of one split lies at least ``min_gap_seconds`` from every
    footprint of another split, across detectors. Needs two or more splits.
    """
    records = _check(records)
    by_split: Dict[str, SegmentList] = {}
    for record in records:
        by_split.setdefault(record.split, SegmentList()).extend(record.segments())
    if len(by_split) < 2:
        raise AssertionError(f"assert_split_gap needs two splits, got {sorted(by_split)}")
    bad = []
    for a, b in combinations(sorted(by_split), 2):
        gap, pair = _distance(by_split[a].coalesce(), by_split[b].coalesce())
        if gap < min_gap_seconds:
            bad.append((a, b, gap, pair))
    if bad:
        raise AssertionError(
            f"splits closer than {min_gap_seconds} s (split, split, minimum gap, closest pair): {bad}"
        )


def assert_avoids(subject, forbidden, ifos: Optional[Sequence[str]] = None) -> None:
    """
    Nothing in ``subject`` intersects ``forbidden``.

    ``subject``: Records (footprints) or {ifo: intervals} (a served-time set).
    ``forbidden``: {ifo: intervals}, or intervals forbidden to every detector.
    """
    if isinstance(subject, Mapping):
        subject_lists = _as_segment_lists(subject)
        if not subject_lists or not any(len(v) for v in subject_lists.values()):
            raise AssertionError("assert_avoids: the served-time set is empty")
        detectors = sorted(subject_lists)
    else:
        records = _check(subject if isinstance(subject, (list, tuple)) else [subject])
        detectors = sorted({ifo for r in records for ifo in r.ifos})
        subject_lists = {
            ifo: SegmentList([s for r in records if ifo in r.ifos for s in r.segments(ifo)]).coalesce()
            for ifo in detectors
        }
    if not isinstance(forbidden, Mapping):
        forbidden = {ifo: forbidden for ifo in detectors}
    banned = _as_segment_lists(forbidden)
    if not any(len(v) for v in banned.values()):
        raise AssertionError("assert_avoids: the forbidden set is empty; the check would be vacuous")
    bad = []
    for ifo in (ifos or detectors):
        overlap = (subject_lists.get(ifo, SegmentList()) & banned.get(ifo, SegmentList())).coalesce()
        if abs(overlap) > 0:
            bad.append((ifo, float(abs(overlap)), [tuple(map(float, s)) for s in overlap[:4]]))
    if bad:
        raise AssertionError(f"forbidden time served (ifo, seconds, first overlaps): {bad}")


def assert_warm_equals_cold(cold: Records, warm: Records) -> None:
    """A warm build fetched nothing and decodes to exactly the cold build's footprints."""
    _check([cold, warm])
    if cold.warm or not warm.warm:
        raise AssertionError("assert_warm_equals_cold needs a cold record and a warm one")
    if cold.fetched == 0 or warm.fetched != 0:
        raise AssertionError(f"fetches: cold {cold.fetched} (must be > 0), warm {warm.fetched} (must be 0)")
    if cold.footprints.shape != warm.footprints.shape or np.any(cold.footprints != warm.footprints):
        differ = int(np.sum(np.any(cold.footprints != warm.footprints, axis=-1))) \
            if cold.footprints.shape == warm.footprints.shape else "shape"
        raise AssertionError(f"the warm build's footprints differ from the cold build's ({differ} rows)")


__all__ = [
    "MIN_EXAMPLES", "Records", "collect", "fetch_intervals", "assert_inside_one_fetch",
    "assert_warm_equals_cold",
    "assert_inside", "assert_in_split", "assert_split_gap", "assert_avoids",
]
