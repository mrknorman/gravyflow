"""
TransientIndex - canonical index of transient examples.

Manages collections of TransientSegments with:
- Deduplication
- Split assignment by footprint (SplitSpec.assign)
- Efficient iteration
- Serialization to/from disk
"""

from collections import Counter
from dataclasses import replace
from typing import List, Iterator, Dict, Optional, Set, Tuple
from pathlib import Path

import numpy as np

from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.event import SourceType
from gravyflow.src.dataset.acquisition.base import DataLabel, ObservingRun
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.dataset.splits import EXCLUDED, SplitSpec, as_split_spec, require_split_fingerprint
from gravyflow.src.utils.gps import gps_to_key

#: Seconds of data the library may read around a transient's time, each side: a
#: record's span (``segment_builders.DEFAULT_PADDING``).
TRANSIENT_HALF_SPAN_SECONDS = 48.2

#: A record's catalogue measurements, saved and loaded with it: a glitch's, and an
#: event's parameters (its masses set its footprint). NaN where the catalogue has none.
MEASUREMENTS = ("snr", "peak_freq", "duration", "ml_confidence", "mass1", "mass2", "distance", "p_astro")


def event_footprints(gps, chirp_mass_source) -> np.ndarray:
    """
    Each event's footprint, (N, 2): ``[t - max(48.2, tau + 1), t + 48.2]``, where tau
    is the event's in-band time from 20 Hz (``EVENT_VETO_CHIRP_TIME_FACTOR`` times the
    chirp time of its source-frame chirp mass; ``UNKNOWN_EVENT_CHIRP_MASS_SOLAR`` when
    it has none).
    """
    from gravyflow.src.dataset.features.event import (
        EVENT_VETO_CHIRP_TIME_FACTOR, UNKNOWN_EVENT_CHIRP_MASS_SOLAR, chirp_time_seconds,
    )
    gps = np.atleast_1d(np.asarray(gps, dtype=np.float64))
    chirp_mass = np.atleast_1d(np.asarray(chirp_mass_source, dtype=np.float64))
    chirp_mass = np.where(np.isfinite(chirp_mass) & (chirp_mass > 0), chirp_mass,
                          UNKNOWN_EVENT_CHIRP_MASS_SOLAR)
    in_band = EVENT_VETO_CHIRP_TIME_FACTOR * chirp_time_seconds(chirp_mass)
    before = np.maximum(TRANSIENT_HALF_SPAN_SECONDS, in_band + 1.0)
    return np.stack([gps - before, gps + TRANSIENT_HALF_SPAN_SECONDS], axis=1).reshape(-1, 2)


def transient_footprint(record: TransientSegment) -> Tuple[float, float]:
    """
    The GPS interval the library may read for ``record``, whatever the dataset's
    window: ``TRANSIENT_HALF_SPAN_SECONDS`` each side of a glitch, and an event's
    footprint (``event_footprints``). A record's split is this interval's.
    """
    t = float(record.transient_gps_time)
    if record.label == DataLabel.EVENTS:
        m1, m2 = float(record.mass1), float(record.mass2)
        chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2 if m1 > 0 and m2 > 0 else np.nan
        start, end = event_footprints([t], [chirp_mass])[0]
        return float(start), float(end)
    return t - TRANSIENT_HALF_SPAN_SECONDS, t + TRANSIENT_HALF_SPAN_SECONDS


def _overlaps(intervals: np.ndarray, start: float, end: float) -> bool:
    """Whether [start, end] meets any of the sorted, coalesced (N, 2) ``intervals``."""
    i = int(np.searchsorted(intervals[:, 1], start, side="left"))
    return i < len(intervals) and intervals[i, 0] <= end


class TransientIndex:
    """
    Canonical index of transient examples (events and glitches).
    
    This is the single source of truth for what examples exist and how
    they should be sampled. All downstream processing (caching, batching,
    label lookup) should key off this index.
    
    Example:
        # Build index from records
        events = build_event_records(observing_runs=[ObservingRun.O3])
        glitches = build_glitch_records(ifos=[IFO.L1], observing_runs=[ObservingRun.O3])
        
        index = TransientIndex(events + glitches)
        index.dedupe()
        index.assign_splits(gf.SplitSpec())
        
        # Iterate over training examples
        for record in index.iter(group="train"):
            print(record.transient_gps_time, record.kind)
    """
    
    def __init__(self, records: List[TransientSegment] = None):
        """
        Initialize index with optional records.
        
        Args:
            records: Initial list of TransientSegments. Can be added later.
        """
        self._groups: Dict[int, str] = {}  # gps_key → split name
        # The SplitSpec fingerprint the splits were assigned by, and the records
        # left out of every split, by reason.
        self.split_fingerprint: Optional[str] = None
        self.excluded: Dict[str, int] = {}
        self.records: List[TransientSegment] = list(records) if records else []
        
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, index: int) -> TransientSegment:
        """Get segment by index."""
        return self.records[index]
    
    def __iter__(self) -> Iterator[TransientSegment]:
        return iter(self.records)
    
    
    def add(self, record: TransientSegment) -> None:
        """Add a single record."""
        self.records.append(record)
    
    def extend(self, records: List[TransientSegment]) -> None:
        """Add multiple records."""
        self.records.extend(records)
    
    def dedupe(self) -> int:
        """
        Remove duplicate records based on GPS key + metadata.
        
        Two records are duplicates if they have the same:
        - GPS key (10ms precision)
        - label (EVENTS/GLITCHES)
        - kind (GlitchType/SourceType)
        - seen_in (list of IFOs where observed)
        
        Returns:
            Number of duplicates removed.
        """
        seen: Set[tuple] = set()
        unique = []
        
        for record in self.records:
            key = (
                record.gps_key,
                record.label,
                record.kind,
                tuple(sorted(ifo.name for ifo in record.seen_in))
            )
            if key not in seen:
                seen.add(key)
                unique.append(record)
        
        removed = len(self.records) - len(unique)
        self.records = unique
        return removed
    
    def sort(self) -> None:
        """Sort records by GPS time ascending."""
        self.records.sort(key=lambda r: r.transient_gps_time)
    
    def assign_splits(
            self,
            spec: Optional[SplitSpec] = None,
            event_footprints_gps: Optional[np.ndarray] = None,
            train_on_real_events: bool = False,
        ) -> Dict[str, int]:
        """
        Assign each record its split: ``spec.assign`` of its footprint
        (``transient_footprint``), so a record's split never depends on a seed,
        the runs, the detectors or the window.

        - Real events are evaluation data: every event goes to ``test``, whatever
          its block, unless ``train_on_real_events``, when an event too takes its
          footprint's split.
        - A record whose footprint straddles an edge between splits, or lies in a
          guard, is in no split.
        - No glitch holds an event: a glitch whose footprint meets an event's
          footprint (``event_footprints_gps``: every catalogued event, (N, 2)), or,
          outside ``test``, comes within ``spec.guard_seconds`` of one, is in no
          split.

        Returns the records left out, by reason (also ``self.excluded``).
        """
        from gravyflow.src.dataset.acquisition.gwosc_data import coalesce_segments

        spec = as_split_spec(spec)
        events = np.empty((0, 2)) if event_footprints_gps is None else np.reshape(
            np.asarray(event_footprints_gps, dtype=np.float64), (-1, 2))
        guard = spec.guard_seconds
        touching = coalesce_segments(events) if len(events) else events
        # Events are test data: the guard keeps every other split's glitches away.
        near = coalesce_segments(events + np.array([-guard, guard])) if len(events) else events

        groups: Dict[int, str] = {}
        excluded: Counter = Counter()
        for record in self.records:
            start, end = transient_footprint(record)
            if record.label == DataLabel.EVENTS and not train_on_real_events:
                groups[record.gps_key] = "test"
                continue
            split = spec.assign(start, end)
            if split == EXCLUDED:
                excluded["straddles a split edge or guard"] += 1
                continue
            if record.label != DataLabel.EVENTS and len(events) and (
                    _overlaps(touching, start, end)
                    or (split != "test" and _overlaps(near, start, end))):
                excluded["holds or neighbours a real event"] += 1
                continue
            groups[record.gps_key] = split
        self._groups = groups
        self.split_fingerprint = spec.fingerprint()
        self.excluded = dict(excluded)
        return self.excluded
    
    def iter(
        self,
        group: Optional[str] = None,
        label: Optional[DataLabel] = None,
    ) -> Iterator[TransientSegment]:
        """
        The records, in index order, optionally filtered. A split's served order is
        drawn per pass when its batches are served, never here.

        Args:
            group: The split whose records to keep (test, validate or train; others
                raise KeyError); None keeps every record, of every split and none.
            label: Filter to specific label (EVENTS or GLITCHES).

        Yields:
            TransientSegments matching filters.
        """
        records_iter = iter(self.records)
        if group is not None:
            SplitSpec.check_name(group)
            if self.split_fingerprint is None:
                raise ValueError("Splits not assigned. Call assign_splits() first.")
            records_iter = (r for r in records_iter if self._groups.get(r.gps_key) == group)
        if label is not None:
            records_iter = (r for r in records_iter if r.label == label)
        yield from records_iter

    def to_segments(
        self, 
        padding_seconds: float,
        num_ifos: int = 1
    ) -> np.ndarray:
        """
        Convert records to segment array for existing download API.
        
        Args:
            padding_seconds: Padding around center time.
            num_ifos: Number of IFOs (for shape).
            
        Returns:
            Array of shape (N, num_ifos, 2) with [start, end] bounds.
        """
        n = len(self.records)
        segments = np.zeros((n, num_ifos, 2), dtype=np.float64)
        
        for i, record in enumerate(self.records):
            start = record.transient_gps_time - padding_seconds
            end = record.transient_gps_time + padding_seconds
            segments[i, :, 0] = start
            segments[i, :, 1] = end
        
        return segments
    
    def filter_by_names(self, names: List[str]) -> "TransientIndex":
        """
        Filter to only segments matching event names.
        
        Used for named event requests (e.g., event_names=["GW150914"]).
        This enables a single code path for all transient acquisition -
        named events simply apply a filter to the index.
        
        Args:
            names: List of event names to keep.
            
        Returns:
            New TransientIndex containing only matching segments.
        """
        name_set = set(names)
        filtered = [r for r in self.records if r.name in name_set]
        
        new_index = TransientIndex(filtered)
        # Preserve group assignments for matching segments
        new_index._groups = {k: v for k, v in self._groups.items() 
                           if k in {r.gps_key for r in filtered}}
        new_index.split_fingerprint = self.split_fingerprint
        new_index.excluded = dict(self.excluded)
        return new_index
    
    def to_numpy_segments(self, num_ifos: int = 1) -> np.ndarray:
        """
        Lazy conversion to numpy format with caching.
        
        Returns numpy array format compatible with valid_segments API.
        Cached to avoid recomputation on repeated calls.
        
        Args:
            num_ifos: Number of IFOs to expand to.
            
        Returns:
            Array of shape (N, num_ifos, 2) with [start_gps, end_gps] bounds.
        """
        # Check cache
        cache_key = f"numpy_segments_{num_ifos}"
        if hasattr(self, '_numpy_cache') and cache_key in self._numpy_cache:
            return self._numpy_cache[cache_key]
        
        # Initialize cache if needed
        if not hasattr(self, '_numpy_cache'):
            self._numpy_cache = {}
        
        # Build numpy array
        n = len(self)
        segments_2d = np.zeros((n, 2), dtype=np.float64)
        
        for i in range(n):
            seg = self[i]
            segments_2d[i, 0] = seg.start_gps_time
            segments_2d[i, 1] = seg.end_gps_time
        
        # Expand for IFOs: (N, 2) -> (N, 1, 2) -> (N, num_ifos, 2)
        expanded = np.expand_dims(segments_2d, axis=1)
        result = np.repeat(expanded, num_ifos, axis=1)
        
        # Cache
        self._numpy_cache[cache_key] = result
        return result
    
    def invalidate_cache(self) -> None:
        """Invalidate cached numpy representations after modifications."""
        if hasattr(self, '_numpy_cache'):
            self._numpy_cache.clear()
    
    def to_gps_array(self) -> np.ndarray:
        """Get array of all GPS times."""
        return np.array([r.transient_gps_time for r in self.records], dtype=np.float64)
    
    def get_group_counts(self) -> Dict[str, int]:
        """Get count of records per group."""
        from collections import Counter
        if not self._groups:
            return {}
        return dict(Counter(self._groups.values()))
    
    def get_kind_counts(self) -> Dict[str, int]:
        """Get count of records per kind."""
        from collections import Counter
        return dict(Counter(str(r.kind) for r in self.records))
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def save(self, path: Path) -> None:
        """
        Save index to NPZ file.
        
        Args:
            path: Output file path.
        """
        path = Path(path)
        
        n = len(self.records)
        
        # Extract arrays
        gps_times = np.array([r.transient_gps_time for r in self.records], dtype=np.float64)
        start_times = np.array([r.start_gps_time for r in self.records], dtype=np.float64)
        end_times = np.array([r.end_gps_time for r in self.records], dtype=np.float64)
        labels = np.array([r.label.value for r in self.records], dtype=np.int8)
        kinds = np.array([r.kind.value for r in self.records], dtype=np.int32)
        runs = np.array([list(ObservingRun).index(r.observing_run) for r in self.records], dtype=np.int8)
        weights = np.array([r.weight for r in self.records], dtype=np.float32)
        
        # Handle optional fields - encode seen_in as bitmasks
        import gravyflow as gf  # Local import to avoid circular dependency
        from gravyflow.src.dataset.acquisition.transient_segment import ifos_to_bitmask
        ifo_masks = np.array([
            ifos_to_bitmask(r.seen_in)
            for r in self.records
        ], dtype=np.uint8)
        
        confidences = np.array([
            r.confidence.value if r.confidence else -1
            for r in self.records
        ], dtype=np.int8)
        
        # Names as separate array (variable length strings)
        names = np.array([r.name if r.name else "" for r in self.records], dtype=str)

        # Split assignments, with the spec they were made by.
        if self.split_fingerprint is None:
            raise ValueError("Splits not assigned: an index is saved with its split. Call assign_splits() first.")
        group_keys = np.array(list(self._groups.keys()), dtype=np.int64)
        group_values = np.array(list(self._groups.values()), dtype=str)
        
        np.savez_compressed(
            path,
            gps_times=gps_times,
            start_times=start_times,
            end_times=end_times,
            labels=labels,
            kinds=kinds,
            runs=runs,
            weights=weights,
            ifo_masks=ifo_masks,
            confidences=confidences,
            names=names,
            group_keys=group_keys,
            group_values=group_values,
            split_fingerprint=np.array(self.split_fingerprint),
            **{field: np.array([getattr(r, field) for r in self.records], dtype=np.float64)
               for field in MEASUREMENTS},
        )
    
    @classmethod
    def load(cls, path: Path, spec: Optional[SplitSpec] = None) -> "TransientIndex":
        """
        Load an index from an NPZ file. Refuses an index split by another spec
        than ``spec`` (None: the default split), or recording none, any object
        array, and an index saved without its records' measurements (``MEASUREMENTS``).
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        require_split_fingerprint(
            str(data["split_fingerprint"]) if "split_fingerprint" in data else None,
            as_split_spec(spec), f"Transient index {path}",
        )
        missing = [field for field in MEASUREMENTS if field not in data]
        if missing:
            raise ValueError(
                f"Transient index {path} was saved without its records' {', '.join(missing)}: "
                "build the index again and save it."
            )
        
        gps_times = data["gps_times"]
        labels = data["labels"]
        kinds = data["kinds"]
        runs = data["runs"]
        weights = data["weights"]
        ifo_masks = data["ifo_masks"]
        confidences = data["confidences"]
        names = data["names"]
        measurements = {field: data[field] for field in MEASUREMENTS}
        
        records = []
        import gravyflow as gf  # Local import to avoid circular dependency
        for i in range(len(gps_times)):
            # Decode enums
            label = DataLabel(labels[i])
            # Decode kind based on label type
            if label == DataLabel.GLITCHES:
                kind = GlitchType(int(kinds[i]))
            elif label == DataLabel.EVENTS:
                kind = SourceType(int(kinds[i]))
            else:
                kind = None
            run = list(ObservingRun)[runs[i]]
            
            # Decode seen_in from bitmask
            from gravyflow.src.dataset.acquisition.transient_segment import bitmask_to_ifos
            from gravyflow.src.utils.gps import gps_to_key
            seen_in = bitmask_to_ifos(int(ifo_masks[i]))
            
            confidence = EventConfidence(confidences[i]) if confidences[i] >= 0 else None
            name = names[i] if names[i] else None
            
            gps_time = float(gps_times[i])
            
            start_gps = float(data["start_times"][i])
            end_gps = float(data["end_times"][i])
            
            record = TransientSegment(
                gps_key=gps_to_key(gps_time),
                transient_gps_time=gps_time,
                start_gps_time=start_gps,
                end_gps_time=end_gps,
                label=label,
                kind=kind,
                observing_run=run,
                seen_in=seen_in,
                confidence=confidence,
                name=name,
                weight=float(weights[i]),
                **{field: float(values[i]) for field, values in measurements.items()},
            )
            records.append(record)
        
        index = cls(records)
        
        index._groups = dict(zip(data["group_keys"].tolist(), data["group_values"].tolist()))
        index.split_fingerprint = str(data["split_fingerprint"])
        
        return index
