"""
TransientIndex - canonical index of transient examples.

Manages collections of TransientSegments with:
- Deduplication
- Deterministic train/val/test group assignment
- Class balancing via weights
- Efficient iteration
- Serialization to/from disk
"""

import hashlib
from dataclasses import replace
from typing import List, Iterator, Dict, Optional, Set
from pathlib import Path

import numpy as np

from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.event import SourceType
from gravyflow.src.dataset.acquisition.base import DataLabel, ObservingRun
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.utils.gps import gps_to_key


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
        index.assign_groups({"train": 0.8, "val": 0.1, "test": 0.1})
        index.apply_balancing()
        
        # Iterate over training examples
        for record in index.iter(group="train"):
            print(record.transient_gps_time, record.kind)
    """
    
    def __init__(self, records: List[TransientSegment] = None, lazy: bool = False):
        """
        Initialize index with optional records.
        
        Args:
            records: Initial list of TransientSegments. Can be added later.
            lazy: If True, store segments in compressed numpy array format
                  to reduce memory usage (40 bytes vs 120 bytes per segment).
                  Segments are inflated on-demand with LRU caching.
        """
        self._lazy = lazy
        self._groups: Dict[int, str] = {}  # gps_key → group name
        
        if lazy and records:
            # Store in compressed format
            from gravyflow.src.dataset.features.compressed_segments import segments_to_compressed_array
            self._compressed = segments_to_compressed_array(records)
            self._inflation_cache = {}  # LRU cache: index → TransientSegment
            from gravyflow.src.dataset.config import TransientDefaults
            self._cache_maxsize = TransientDefaults.LAZY_CACHE_MAXSIZE
            self.records = None  # Not used in lazy mode
        else:
            # Standard mode: store full objects
            self.records: List[TransientSegment] = list(records) if records else []
            self._compressed = None
            self._inflation_cache = None
        
    def __len__(self) -> int:
        if self._lazy:
            return len(self._compressed)
        return len(self.records)
    
    
    def __getitem__(self, index: int) -> TransientSegment:
        """Get segment by index (supports lazy inflation)."""
        if self._lazy:
            # Check cache first
            if index in self._inflation_cache:
                return self._inflation_cache[index]
            
            # Inflate from compressed
            from gravyflow.src.dataset.features.compressed_segments import compressed_to_segment
            segment = compressed_to_segment(self._compressed[index])
            
            # Add to cache (LRU eviction)
            if len(self._inflation_cache) >= self._cache_maxsize:
                # Remove oldest (first inserted)
                self._inflation_cache.pop(next(iter(self._inflation_cache)))
            self._inflation_cache[index] = segment
            
            return segment
        else:
            return self.records[index]
    
    def __iter__(self) -> Iterator[TransientSegment]:
        if self._lazy:
            # Iterate via __getitem__ (lazy inflation)
            for i in range(len(self)):
                yield self[i]
        else:
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
    
    def assign_groups(
        self, 
        groups: Dict[str, float],
        seed: int = 42
    ) -> None:
        """
        Assign records to groups using deterministic hash-based assignment.
        
        This ensures stable group membership across runs and when new
        data is added.
        
        Args:
            groups: Dict mapping group name to proportion (e.g., {"train": 0.8, "val": 0.1, "test": 0.1})
            seed: Random seed mixed into hash for reproducibility.
        """
        # Normalize proportions
        total = sum(groups.values())
        normalized = {k: v / total for k, v in groups.items()}
        
        # Build cumulative distribution
        group_names = list(normalized.keys())
        cumulative = []
        cum = 0.0
        for name in group_names:
            cum += normalized[name]
            cumulative.append(cum)
        
        # Assign each record based on hash of GPS key
        self._groups = {}
        for record in self.records:
            # Create stable hash from GPS key and seed
            hash_input = f"{record.gps_key}:{seed}".encode()
            h = hashlib.md5(hash_input).hexdigest()
            u = int(h, 16) / (2**128)  # Uniform [0, 1)
            
            # Find group
            for i, threshold in enumerate(cumulative):
                if u <= threshold:
                    self._groups[record.gps_key] = group_names[i]
                    break
    
    def apply_balancing(self, by: str = "kind") -> None:
        """
        Set weights for class balancing.
        
        Weights are set inversely proportional to class frequency,
        so rare classes get higher weights.
        
        Args:
            by: Field to balance by ("kind" or "label").
        """
        from collections import Counter
        
        # Count occurrences
        if by == "kind":
            counts = Counter(r.kind for r in self.records)
        elif by == "label":
            counts = Counter(r.label for r in self.records)
        else:
            raise ValueError(f"Unknown balance field: {by}. Use 'kind' or 'label'.")
        
        if not counts:
            return
        
        max_count = max(counts.values())
        
        # Update weights
        new_records = []
        for record in self.records:
            if by == "kind":
                key = record.kind
            else:  # by == "label"
                key = record.label
            
            weight = max_count / counts[key]
            new_records.append(replace(record, weight=weight))
        
        self.records = new_records
    
    def iter(
        self, 
        group: Optional[str] = None,
        label: Optional[DataLabel] = None,
        ifos_filter: Optional[List["gf.IFO"]] = None,
        shuffle: bool = False,
        seed: int = None
    ) -> Iterator[TransientSegment]:
        """
        Iterate over records with optional filtering.
        
        Args:
            group: Filter to specific group (e.g., "train").
            label: Filter to specific label (EVENTS or GLITCHES).
            ifos_filter: Filter to specific IFOs (segments must overlap with these IFOs).
            shuffle: If True, shuffle records before iterating.
            seed: Random seed for shuffling.
            
        Yields:
            TransientSegments matching filters.
        """
        # Get all records (lazy inflation if needed)
        # For lazy mode, use a generator to avoid materializing all at once
        if self._lazy:
            def lazy_generator():
                for i in range(len(self)):
                    yield self[i]
            records_iter = lazy_generator()
        else:
            records_iter = iter(self.records)
        
        # Apply filters as generator pipeline
        if group is not None:
            if self._groups is None:
                raise ValueError("Groups not assigned. Call assign_groups() first.")
            records_iter = (r for r in records_iter if self._groups.get(r.gps_key) == group)
        
        # Filter by label
        if label is not None:
            records_iter = (r for r in records_iter if r.label == label)
        
        # Filter by IFO(s) - check if segment's seen_in overlaps with filter
        if ifos_filter is not None:
            records_iter = (r for r in records_iter if any(ifo in r.seen_in for ifo in ifos_filter))
        
        # Shuffle if requested (must materialize for shuffling)
        if shuffle:
            filtered = list(records_iter)
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(filtered))
            yield from (filtered[i] for i in indices)
        else:
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
        names = np.array([r.name if r.name else "" for r in self.records], dtype=object)
        
        # Group assignments
        group_keys = np.array(list(self._groups.keys()), dtype=np.int64)
        group_values = np.array(list(self._groups.values()), dtype=object)
        
        np.savez_compressed(
            path,
            gps_times=gps_times,
            labels=labels,
            kinds=kinds,
            runs=runs,
            weights=weights,
            ifo_masks=ifo_masks,
            confidences=confidences,
            names=names,
            group_keys=group_keys,
            group_values=group_values
        )
    
    @classmethod
    def load(cls, path: Path) -> "TransientIndex":
        """
        Load index from NPZ file.
        
        Args:
            path: Input file path.
            
        Returns:
            Loaded TransientIndex.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        gps_times = data["gps_times"]
        labels = data["labels"]
        kinds = data["kinds"]
        runs = data["runs"]
        weights = data["weights"]
        ifo_masks = data["ifo_masks"]
        confidences = data["confidences"]
        names = data["names"]
        
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
            
            confidence = list(EventConfidence)[confidences[i]] if confidences[i] >= 0 else None
            name = names[i] if names[i] else None
            
            gps_time = float(gps_times[i])
            padding = 16.0  # Default padding
            
            record = TransientSegment(
                gps_key=gps_to_key(gps_time),
                transient_gps_time=gps_time,
                start_gps_time=gps_time - padding,
                end_gps_time=gps_time + padding,
                label=label,
                kind=kind,
                observing_run=run,
                seen_in=seen_in,
                confidence=confidence,
                name=name,
                weight=float(weights[i])
            )
            records.append(record)
        
        index = cls(records)
        
        # Restore group assignments (if they exist in the file)
        index._groups = {}  # Always initialize
        if "group_keys" in data and len(data["group_keys"]) > 0:
            index._groups = dict(zip(
                data["group_keys"].tolist(),
                data["group_values"].tolist()
            ))
        
        return index
