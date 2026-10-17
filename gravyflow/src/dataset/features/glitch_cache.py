"""
Standardized glitch data cache with bulk array storage.

This module provides efficient HDF5 storage and loading for glitch data
using multi-dimensional arrays instead of per-transient datasets.

File Format (HDF5):
    glitches/
        onsource: (N, IFOs, samples)     # All onsource data
        offsource: (N, IFOs, samples)    # All offsource data
        gps_times: (N,)                   # GPS time for each glitch
        labels: (N,)                      # GlitchType integer labels
        metadata/
            sample_rate_hertz: float
            onsource_duration: float
            offsource_duration: float
            ifo_names: list of strings
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import threading

logger = logging.getLogger(__name__)
import os
import numpy as np
import h5py
from keras import ops

from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from gravyflow.src.utils.gps import gps_to_key, gps_array_to_keys, as_gps_times
from gravyflow.src.dataset.config import TransientDefaults

# Re-export from TransientDefaults for internal use
CACHE_SAMPLE_RATE_HERTZ = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
CACHE_ONSOURCE_DURATION = TransientDefaults.CACHE_ONSOURCE_DURATION
CACHE_OFFSOURCE_DURATION = TransientDefaults.CACHE_OFFSOURCE_DURATION
CACHE_PADDING_DURATION = TransientDefaults.CACHE_PADDING_DURATION

from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class CacheChunk:
    """Represents a loaded chunk of cache data."""
    start_idx: int
    end_idx: int
    onsource: np.ndarray
    offsource: np.ndarray
    gps: np.ndarray
    labels: np.ndarray


class TransientCache:
    """
    Manages transient data cache (events and glitches) with efficient bulk I/O.
    
    All data is stored at maximum supported parameters and resampled/cropped
    at load time to match the requested configuration.
    
    Args:
        path: Path to the HDF5 cache file
        mode: File access mode ('r' for read, 'w' for write, 'a' for append)
    
    Example:
        # Writing cache
        cache = TransientCache(Path("transients_O3.h5"), mode='w')
        cache.save_all(onsource, offsource, gps_times, labels,
                      sample_rate_hertz=4096.0, ifo_names=['L1'])
        
        # Reading cache
        cache = TransientCache(Path("transients_O3.h5"), mode='r')
        cache.validate_request(2048.0, 2.0, 16.0)  # Raises if unsatisfiable
        onsource, offsource, gps, labels = cache.get_batch(
            np.arange(4), sample_rate_hertz=2048.0, onsource_duration=2.0,
            offsource_duration=16.0)  # Windows by read_windows
    """
    
    @staticmethod
    def read_windows(
        onsource,
        offsource,
        stored_rate: float,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        rows=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The (padded) onsource and offsource windows of a request, from stored ones: the
        one geometry of every transient window, whatever reads it.

        A stored offsource and its stored onsource are one stretch of strain, the
        offsource first, with the transient on the onsource's centre sample
        (``centre_index``). The request's onsource is centred on that sample, and its
        offsource ends where the onsource begins, as for noise, so it may reach into
        the stored onsource. Sample counts are WindowSpec's (``even_samples``).

        ``onsource`` and ``offsource`` are arrays (or HDF5 datasets) whose last axis is
        samples at ``stored_rate``; ``rows``, when given, selects their first axis (and
        an HDF5 read then holds only the samples needed). The rate is lowered by
        keeping every n-th sample, in phase with the transient's sample.

        Raises:
            ValueError: if the request needs upsampling, a rate that does not divide the
                stored one, or more strain than is stored.
        """
        from gravyflow.src.utils.tensor import centre_index, even_samples

        if sample_rate_hertz > stored_rate:
            raise ValueError(
                f"Cannot upsample from {stored_rate}Hz to {sample_rate_hertz}Hz. "
                "Upsampling is not supported."
            )
        if stored_rate % sample_rate_hertz != 0:
            raise ValueError(
                f"Sample rates must be integer multiples. "
                f"Cannot downsample from {stored_rate}Hz to {sample_rate_hertz}Hz. "
                f"Ratio {stored_rate / sample_rate_hertz:.4f} is not an integer."
            )
        ratio = int(stored_rate // sample_rate_hertz)
        num_onsource = even_samples(onsource_duration, sample_rate_hertz)
        num_offsource = even_samples(offsource_duration, sample_rate_hertz)

        # Indices along the stretch: the stored offsource, then the stored onsource.
        stored_offsource = int(offsource.shape[-1])
        stored_total = stored_offsource + int(onsource.shape[-1])
        onsource_start = stored_offsource + centre_index(onsource.shape[-1]) \
            - centre_index(num_onsource) * ratio
        start = onsource_start - num_offsource * ratio
        stop = onsource_start + num_onsource * ratio
        if start < 0 or stop > stored_total:
            raise ValueError(
                f"A {onsource_duration} s onsource with a {offsource_duration} s offsource "
                f"needs more strain than is stored ({stored_total / stored_rate} s)."
            )

        def take(store, lo, hi):
            return np.asarray(store[..., lo:hi] if rows is None else store[rows, :, lo:hi])

        parts = []
        if start < stored_offsource:
            parts.append(take(offsource, start, min(stop, stored_offsource)))
        if stop > stored_offsource:
            parts.append(take(onsource, max(start - stored_offsource, 0), stop - stored_offsource))
        stretch = np.concatenate(parts, axis=-1)[..., ::ratio]
        return stretch[..., num_offsource:], stretch[..., :num_offsource]

    def __init__(self, path: Path, mode: str = 'r'):
        self.path = Path(path)
        self.mode = mode
        self._metadata: Optional[Dict] = None
        
        # Chunked memory cache for large datasets
        # Loads a subset of samples into memory with LRU-style eviction
        self._chunk_in_memory = False
        self._max_chunks = 50  # Keep up to 50 chunks (covers ~250k items, ~4-8GB RAM)
        self._chunks: OrderedDict[int, CacheChunk] = OrderedDict() # Map start_idx -> Chunk
        
        self._chunk_size: int = 5000    # Default chunk size (configurable)
        
        # Chunk request parameters (for crop/resample consistency)
        self._chunk_sample_rate: Optional[float] = None
        self._chunk_ons_dur: Optional[float] = None
        self._chunk_off_dur: Optional[float] = None
        
        # Thread-safe GPS index access
        self._gps_lock = threading.RLock()
        self._gps_index: Optional[Dict[int, int]] = None
        
        # Write buffer for batched appends (reduces HDF5 lock contention)
        self._write_buffer: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
        self._write_buffer_size: int = 50  # Flush after this many samples
        
    @property
    def exists(self) -> bool:
        """Check if cache file exists."""
        return self.path.exists()
    
    def reset(self) -> None:
        """Delete cache file and clear in-memory state."""
        if self.path.exists():
            try:
                os.remove(self.path)
            except OSError as e:
                logger.warning(f"Failed to remove cache file {self.path}: {e}")
        
        self._gps_index = None
        self._metadata = None
    
    def get_metadata(self) -> Dict:
        """Load and cache metadata from file."""
        if self._metadata is not None:
            return self._metadata
            
        if not self.exists:
            raise FileNotFoundError(f"Cache file not found: {self.path}")
            
        with h5py.File(self.path, 'r') as f:
            if 'glitches' not in f:
                raise ValueError(f"Invalid cache format: 'glitches' group not found")
            
            grp = f['glitches']
            self._metadata = {
                'sample_rate_hertz': float(grp.attrs.get('sample_rate_hertz', CACHE_SAMPLE_RATE_HERTZ)),
                'onsource_duration': float(grp.attrs.get('onsource_duration', CACHE_ONSOURCE_DURATION)),
                'offsource_duration': float(grp.attrs.get('offsource_duration', CACHE_OFFSOURCE_DURATION)),
                'ifo_names': list(grp.attrs.get('ifo_names', [])),
                'num_glitches': int(grp['gps_times'].shape[0]),
                'num_ifos': int(grp['onsource'].shape[1]),
            }
        return self._metadata
    
    def get_attr(self, key: str, default=None):
        """Get an attribute from the cache file metadata."""
        if not self.exists:
            return default
            
        with h5py.File(self.path, 'r') as f:
            if 'glitches' not in f:
                return default
            return f['glitches'].attrs.get(key, default)

    def set_attr(self, key: str, value):
        """Set an attribute in the cache file metadata."""
        if not self.exists:
            raise FileNotFoundError(f"Cache file not initialized: {self.path}")
            
        with h5py.File(self.path, 'a') as f:
            if 'glitches' not in f:
                raise ValueError("Invalid cache format")
            f['glitches'].attrs[key] = value

    def get_last_gps(self) -> float:
        """Get the last stored GPS time, or -1.0 if empty."""
        if not self.exists:
            return -1.0
            
        with h5py.File(self.path, 'r') as f:
            if 'glitches' not in f or 'gps_times' not in f['glitches']:
                return -1.0
            
            dset = f['glitches']['gps_times']
            if dset.shape[0] == 0:
                return -1.0
            
            # Read last element
            return float(dset[-1])
    
    def _build_gps_index(self) -> Dict[int, int]:
        """Build a dictionary mapping GPS integer keys to their indices in the cache."""
        if not self.exists:
            return {}
            
        with h5py.File(self.path, 'r') as f:
            if 'glitches' not in f or 'gps_times' not in f['glitches']:
                return {}
            
            gps_times = f['glitches']['gps_times'][:]
            # Use integer keys at 0.01s (10ms) precision for exact matching
            index = {gps_to_key(gps): i for i, gps in enumerate(gps_times)}
            return index
    
    def has_gps(self, gps_time: float) -> bool:
        """Check if a GPS time exists in the cache using exact integer key match."""
        if not self.exists:
            return False
        
        with self._gps_lock:
            # Build index on first call (lazy)
            if self._gps_index is None:
                self._gps_index = self._build_gps_index()
            
            # Exact match using integer key
            return gps_to_key(gps_time) in self._gps_index
    
    def get_closest_gps(self, gps_time: float) -> Optional[float]:
        """Get the cached GPS time matching the integer key, or None if not found."""
        if not self.exists:
            return None
        
        with self._gps_lock:
            if self._gps_index is None:
                self._gps_index = self._build_gps_index()
            
            # Exact match using integer key
            key = gps_to_key(gps_time)
            if key in self._gps_index:
                return gps_time  # Return the original time since key matched
            
            return None
    
    def get_by_gps(
        self,
        gps_time: float,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Get a single glitch by GPS time.
        
        Returns:
            Tuple of (onsource, offsource, gps_time, label) or None if not found
            or if the request cannot be satisfied (e.g., upsampling required).
        """
        if not self.has_gps(gps_time):
            return None
        
        # Validate that request can be satisfied (no upsampling)
        if sample_rate_hertz is not None:
            try:
                self.validate_request(
                    sample_rate_hertz,
                    onsource_duration or self.get_metadata()['onsource_duration'],
                    offsource_duration or self.get_metadata()['offsource_duration']
                )
            except ValueError:
                # Cache can't satisfy this request (e.g., upsampling needed)
                return None
        
        # Use integer key for index lookup
        key = gps_to_key(gps_time)
        if key not in self._gps_index:
            return None
        idx = self._gps_index[key]
        ons, offs, gps, labels = self.get_batch(
            np.array([idx]),
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration
        )
        return ons[0], offs[0], gps[0], labels[0]

    def has_key(self, gps_key: int) -> bool:
        """
        Check if a GPS key exists in the cache (no float conversion needed).
        
        Args:
            gps_key: Integer GPS key at 10ms precision
            
        Returns:
            True if key exists in cache
        """
        if not self.exists:
            return False
        
        with self._gps_lock:
            # Build index on first call (lazy)
            if self._gps_index is None:
                self._gps_index = self._build_gps_index()
            
            return gps_key in self._gps_index
    
    def get_by_key(
        self,
        gps_key: int,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Get a single glitch by GPS key (optimized - no conversion needed).
        
        Args:
            gps_key: Integer GPS key at 10ms precision
            sample_rate_hertz: Desired sample rate
            onsource_duration: Desired onsource duration  
            offsource_duration: Desired offsource duration
            
        Returns:
            Tuple of (onsource, offsource, gps_time, label) or None if not found
        """
        if not self.has_key(gps_key):
            return None
        
        # Validate that request can be satisfied
        if sample_rate_hertz is not None:
            try:
                self.validate_request(
                    sample_rate_hertz,
                    onsource_duration or self.get_metadata()['onsource_duration'],
                    offsource_duration or self.get_metadata()['offsource_duration']
                )
            except ValueError:
                return None
        
        # Direct index lookup (no conversion!)
        idx = self._gps_index[gps_key]
        ons, offs, gps, labels = self.get_batch(
            np.array([idx]),
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration
        )
        return ons[0], offs[0], gps[0], labels[0]

    
    def get_indices_for_gps(
        self, 
        gps_times: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bulk resolve GPS times to cache indices using exact integer key matching.
        
        Args:
            gps_times: List or array of GPS times to look up.
            
        Returns:
            found_indices: Array of indices in the cache (for found items).
            missing_mask: Boolean mask of same length as input gps_times, 
                          True where item was NOT found.
        """
        if not self.exists:
             return np.array([]), np.ones(len(gps_times), dtype=bool)
        
        with self._gps_lock:
            # Ensure index exists
            if self._gps_index is None:
                self._gps_index = self._build_gps_index()
                 
            gps_times = np.asarray(gps_times)
            indices = np.full(len(gps_times), -1, dtype=np.int32)
            missing_mask = np.ones(len(gps_times), dtype=bool)
            
            # Exact O(1) lookups using integer keys
            for i, gps in enumerate(gps_times):
                key = gps_to_key(gps)
                if key in self._gps_index:
                    indices[i] = self._gps_index[key]
                    missing_mask[i] = False
                    
            return indices[~missing_mask], missing_mask

    def get_all_gps_times(self) -> np.ndarray:
        """Return all GPS times currently in the cache as float seconds."""
        if not self.exists:
            return np.array([])
        
        with self._gps_lock:
            if self._gps_index is None:
                self._gps_index = self._build_gps_index()
            
            # Convert integer keys back to GPS times (floats)
            from gravyflow.src.utils.gps import key_to_gps
            keys = sorted(self._gps_index.keys())
            return np.array([key_to_gps(k) for k in keys])
    
    def append_single(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_time: float,
        label: int,
        gps_key: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Buffer a single glitch for later batch append.
        
        Writes are buffered to reduce HDF5 file operations and lock contention.
        Call flush_write_buffer() to persist buffered data, or let it auto-flush
        when the buffer reaches _write_buffer_size.
        
        Args:
            onsource: Array of shape (IFOs, samples)
            offsource: Array of shape (IFOs, samples)
            gps_time: GPS time of the glitch (for storage and display)
            label: GlitchType/SourceType integer label
            gps_key: Optional GPS key (10ms precision). If not provided, computed from gps_time.
            metadata: Optional dict with extended metadata:
                - data_label: int (DataLabel enum value, -1 if missing)
                - seen_in: int (bitmask, 0 if missing)
                - snr, peak_freq, duration, ml_confidence: float (NaN if missing)
                - mass1, mass2, distance, p_astro: float (NaN if missing, events only)
        """
        if metadata is None:
            metadata = {}
        
        # Compute GPS key if not provided
        if gps_key is None:
            gps_key = gps_to_key(gps_time)
        
        # Skip if already in cache or buffer (duplicate protection)
        if self.has_key(gps_key):
            return
        if any(gps_to_key(item[2]) == gps_key for item in self._write_buffer):
            return
        
        # Add to buffer with metadata
        self._write_buffer.append((onsource, offsource, gps_time, label, metadata))
        
        # Auto-flush when buffer is full
        if len(self._write_buffer) >= self._write_buffer_size:
            self.flush_write_buffer()
    
    def flush_write_buffer(self) -> None:
        """
        Flush all buffered writes to disk in a single HDF5 operation.
        
        This reduces file lock contention and I/O overhead by batching
        multiple samples into one append operation.
        """
        if not self._write_buffer:
            return  # Nothing to flush
        
        # Stack buffered data
        onsource_batch = np.stack([item[0] for item in self._write_buffer], axis=0)
        offsource_batch = np.stack([item[1] for item in self._write_buffer], axis=0)
        gps_times = np.array([item[2] for item in self._write_buffer])
        labels = np.array([item[3] for item in self._write_buffer])
        
        # Extract extended metadata (item[4] is metadata dict)
        def get_meta(key, default):
            return np.array([item[4].get(key, default) for item in self._write_buffer])
        
        metadata_arrays = {
            'data_label': get_meta('data_label', -1).astype(np.int8),
            'seen_in': get_meta('seen_in', 0).astype(np.uint8),
            'snr': get_meta('snr', float('nan')).astype(np.float32),
            'peak_freq': get_meta('peak_freq', float('nan')).astype(np.float32),
            'duration': get_meta('duration', float('nan')).astype(np.float32),
            'ml_confidence': get_meta('ml_confidence', float('nan')).astype(np.float32),
            'mass1': get_meta('mass1', float('nan')).astype(np.float32),
            'mass2': get_meta('mass2', float('nan')).astype(np.float32),
            'distance': get_meta('distance', float('nan')).astype(np.float32),
            'p_astro': get_meta('p_astro', float('nan')).astype(np.float32),
        }
        
        n_buffered = len(self._write_buffer)
        self._write_buffer.clear()
        
        # Single batch write
        try:
            self.append(
                onsource=onsource_batch,
                offsource=offsource_batch,
                gps_times=gps_times,
                labels=labels,
                metadata=metadata_arrays
            )
            logger.debug(f"Flushed {n_buffered} samples to cache")
        except Exception as e:
            logger.warning(f"Failed to flush write buffer ({n_buffered} samples): {e}")
            # Re-add to buffer for retry (but limit retries to prevent infinite loop)
            # For now, just log and lose the data rather than risk memory issues
    
    @property
    def write_buffer_count(self) -> int:
        """Return number of items in write buffer (not yet persisted)."""
        return len(self._write_buffer)
    
    def validate_request(
        self, 
        sample_rate_hertz: float, 
        onsource_duration: float, 
        offsource_duration: float
    ) -> None:
        """
        Validate that a request can be satisfied from this cache.
        
        Raises ValueError if any parameter exceeds cached limits.
        
        Args:
            sample_rate_hertz: Requested sample rate
            onsource_duration: Requested onsource window duration
            offsource_duration: Requested offsource window duration
        """
        meta = self.get_metadata()
        
        if sample_rate_hertz > meta['sample_rate_hertz']:
            raise ValueError(
                f"Requested sample rate {sample_rate_hertz}Hz exceeds "
                f"cache max {meta['sample_rate_hertz']}Hz. Cannot upsample."
            )
        
        if onsource_duration > meta['onsource_duration']:
            raise ValueError(
                f"Requested onsource duration {onsource_duration}s exceeds "
                f"cache max {meta['onsource_duration']}s. Cannot extend."
            )
        
        if offsource_duration > meta['offsource_duration']:
            raise ValueError(
                f"Requested offsource duration {offsource_duration}s exceeds "
                f"cache max {meta['offsource_duration']}s. Cannot extend."
            )
    
    
    def enable_chunked_mode(
        self,
        chunk_size: int = 5000,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> None:
        """
        Enable chunked memory mode for large datasets.
        
        Loads a subset of samples into memory and streams new chunks
        when requested samples are outside the current chunk.
        
        Args:
            chunk_size: Number of samples per chunk (default 5000, ~1-2GB)
            sample_rate_hertz: Target sample rate for loaded data
            onsource_duration: Target onsource duration
            offsource_duration: Target offsource duration
        """
        self._chunk_size = chunk_size
        self._chunk_sample_rate = sample_rate_hertz
        self._chunk_ons_dur = onsource_duration
        self._chunk_off_dur = offsource_duration
        self._chunk_in_memory = True
        
        # Clear existing chunks
        self._chunks.clear()
        
        # Load first chunk centered at index 0 (or simply aligned)
        self._load_chunk_around_index(0)
        
        logger.info(
            f"Enabled chunked mode: chunk_size={chunk_size}, max_chunks={self._max_chunks}, "
            f"total_samples={self.get_metadata()['num_glitches']}"
        )
    
    def _load_chunk_around_index(self, target_idx: int) -> None:
        """
        Load a chunk of data covering the target index.
        
        Chunks are aligned to chunk_size boundaries to maximize reuse.
        Uses LRU eviction if _max_chunks is exceeded.
        """
        meta = self.get_metadata()
        n_total = meta['num_glitches']
        
        if n_total == 0:
            return
        
        # Align start to chunk boundary for canonical chunks
        # e.g., if chunk_size=5000, idx=500 -> start=0; idx=5001 -> start=5000
        start = (target_idx // self._chunk_size) * self._chunk_size
        start = max(0, min(start, n_total - 1)) # Clamp
        
        end = min(n_total, start + self._chunk_size)
        
        # Check if already loaded (should be checked by caller, but safe to re-check)
        if start in self._chunks:
            self._chunks.move_to_end(start)
            return

        # Prepare to load
        stored_rate = meta['sample_rate_hertz']
        target_rate = self._chunk_sample_rate or stored_rate
        target_ons_dur = self._chunk_ons_dur or meta['onsource_duration']
        target_off_dur = self._chunk_off_dur or meta['offsource_duration']
        
        # Read this chunk's windows (read_windows)
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            chunk_onsource, chunk_offsource = self.read_windows(
                grp['onsource'], grp['offsource'], stored_rate, target_rate,
                target_ons_dur, target_off_dur, rows=slice(start, end),
            )
            chunk_gps = grp['gps_times'][start:end]
            chunk_labels = grp['labels'][start:end]
        
        # Create chunk object
        new_chunk = CacheChunk(
            start_idx=start,
            end_idx=end,
            onsource=chunk_onsource,
            offsource=chunk_offsource,
            gps=chunk_gps,
            labels=chunk_labels
        )
        
        # Add to cache (LRU logic)
        if len(self._chunks) >= self._max_chunks:
            # Evict oldest (first item)
            self._chunks.popitem(last=False)
            
        self._chunks[start] = new_chunk
        self._chunks.move_to_end(start) # Mark as most recently used
        
        logger.debug(f"Loaded chunk [{start}:{end}] ({end-start} samples). Cache has {len(self._chunks)} chunks.")
    
    def get_from_chunk(
        self,
        idx: int,
        target_ifos: Optional[List[str]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Get a sample from the chunked memory cache.
        
        If the sample is outside current chunks, loads a new chunk via LRU.
        
        Args:
            idx: Global index in the cache
            target_ifos: Optional list of IFO names to filter channels
            
        Returns:
            Tuple of (onsource, offsource, gps_time, label) or None if invalid
        """
        if not self._chunk_in_memory:
            return None
        
        # Determine likely start index of the chunk containing idx
        # Assuming aligned chunks:
        chunk_start = (idx // self._chunk_size) * self._chunk_size
        
        # Check if chunk is loaded
        chunk = self._chunks.get(chunk_start)
        
        if chunk is None:
            # Need to load it
            self._load_chunk_around_index(idx)
            # Re-fetch (load handles alignment guarantees)
            # Note: _load_chunk_around_index aligns the chunk start
            # If idx was large and n_total small, start might be different?
            # But the formula matches load logic.
            chunk = self._chunks.get(chunk_start)
            
            # Corner case: if n_total < chunk_size, chunk_start=0.
            if chunk is None:
                 # Try finding any chunk covering this index (scan)
                 # Should be rare if logic matches
                 for s, c in self._chunks.items():
                     if c.start_idx <= idx < c.end_idx:
                         chunk = c
                         break
        else:
            # Mark simple hit as used
            self._chunks.move_to_end(chunk_start)
            
        if chunk is None or not (chunk.start_idx <= idx < chunk.end_idx):
            # Not found or index out of bounds entirely
            return None
        
        # Convert global index to chunk-local index
        local_idx = idx - chunk.start_idx
        
        onsource = chunk.onsource[local_idx]
        offsource = chunk.offsource[local_idx]
        
        # Apply IFO filtering if requested
        if target_ifos:
            meta = self.get_metadata()
            stored_ifos = meta.get('ifo_names', [])
            if stored_ifos:
                try:
                    channel_indices = [stored_ifos.index(ifo) for ifo in target_ifos]
                    onsource = onsource[channel_indices]
                    offsource = offsource[channel_indices]
                except ValueError:
                    return None
        
        return (
            onsource,
            offsource,
            float(chunk.gps[local_idx]),
            int(chunk.labels[local_idx])
        )
    
    def get_batch_from_chunk(
        self,
        indices: np.ndarray,
        target_ifos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get multiple samples from chunked cache, batching reads efficiently.
        
        Loads necessary chunks via LRU logic and extracts samples.
        
        Args:
            indices: Array of global indices
            target_ifos: Optional IFO filter
            
        Returns:
            Tuple of (onsource, offsource, gps_times, labels, found_mask)
            where found_mask[i] is True if indices[i] was found
        """
        if not self._chunk_in_memory:
            # Fallback to disk reads
            found_mask = np.ones(len(indices), dtype=bool)
            return (*self.get_batch(indices, 
                                     sample_rate_hertz=self._chunk_sample_rate,
                                     onsource_duration=self._chunk_ons_dur,
                                     offsource_duration=self._chunk_off_dur,
                                     target_ifos=target_ifos), found_mask)
        
        # Prepare output arrays (use stored metadata to guess size or dynamic?)
        # We need size from first chunk or metadata
        meta = self.get_metadata()
        
        # Determine likely start index of chunks
        chunk_starts = (indices // self._chunk_size) * self._chunk_size
        unique_starts = np.unique(chunk_starts)
        
        # Ensure chunks are loaded (LRU)
        for start in unique_starts:
            if start not in self._chunks:
                self._load_chunk_around_index(start)
        
        # Extract data
        found_list = []
        ons_list = []
        off_list = []
        gps_list = []
        lbl_list = []
        found_mask = np.zeros(len(indices), dtype=bool)
        
        # This loop could be optimized but efficient enough for typical batch sizes (32)
        for i, (idx, start) in enumerate(zip(indices, chunk_starts)):
            chunk = self._chunks.get(start)
            if chunk and chunk.start_idx <= idx < chunk.end_idx:
                 # Found!
                 local_idx = idx - chunk.start_idx
                 
                 ons = chunk.onsource[local_idx]
                 off = chunk.offsource[local_idx]
                 
                 # IFO filter
                 if target_ifos:
                        stored_ifos = meta.get('ifo_names', [])
                        try:
                            # (Re-resolving IFO indices per sample is slow, verify outside loop ideally)
                            c_idxs = [stored_ifos.index(ifo) for ifo in target_ifos]
                            ons = ons[c_idxs]
                            off = off[c_idxs]
                        except ValueError:
                            continue # Invalid IFO
                            
                 ons_list.append(ons)
                 off_list.append(off)
                 gps_list.append(float(chunk.gps[local_idx]))
                 lbl_list.append(int(chunk.labels[local_idx]))
                 found_mask[i] = True
        
        # If no hits, returns empty arrays (check shape)
        if not ons_list:
             return (np.array([]), np.array([]), np.array([]), np.array([]), found_mask)
             
        return (
            np.stack(ons_list),
            np.stack(off_list),
            np.array(gps_list),
            np.array(lbl_list),
            found_mask
        )

    
    def save_all(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_times: np.ndarray,
        labels: np.ndarray,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        ifo_names: List[str],
        compression: str = 'gzip',
        compression_opts: int = 4
    ) -> None:
        """
        Save all glitches to cache in a single write.
        
        Args:
            onsource: Array of shape (N, IFOs, samples)
            offsource: Array of shape (N, IFOs, samples)
            gps_times: Array of shape (N,)
            labels: Array of shape (N,) with GlitchType integer values
            sample_rate_hertz: Sample rate of the data
            onsource_duration: Duration of onsource windows
            offsource_duration: Duration of offsource windows
            ifo_names: List of IFO names (e.g., ['L1', 'H1'])
            compression: HDF5 compression algorithm
            compression_opts: Compression level (0-9)
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.path, 'w') as f:
            grp = f.create_group('glitches')
            
            # Store data with chunking for efficient access
            grp.create_dataset(
                'onsource', 
                data=onsource.astype(np.float32),
                chunks=(min(64, onsource.shape[0]), onsource.shape[1], onsource.shape[2]),
                compression=compression,
                compression_opts=compression_opts
            )
            grp.create_dataset(
                'offsource', 
                data=offsource.astype(np.float32),
                chunks=(min(64, offsource.shape[0]), offsource.shape[1], offsource.shape[2]),
                compression=compression,
                compression_opts=compression_opts
            )
            grp.create_dataset('gps_times', data=gps_times.astype(np.float64))
            grp.create_dataset('labels', data=labels.astype(np.int32))
            
            # Store metadata as attributes
            grp.attrs['sample_rate_hertz'] = sample_rate_hertz
            grp.attrs['onsource_duration'] = onsource_duration
            grp.attrs['offsource_duration'] = offsource_duration
            grp.attrs['ifo_names'] = ifo_names
            grp.attrs['version'] = 1  # For future format changes
            
        logger.info(
            f"Saved {len(gps_times)} glitches to {self.path} "
            f"({onsource.nbytes / 1e6:.1f}MB onsource, "
            f"{offsource.nbytes / 1e6:.1f}MB offsource)"
        )
        
        # Clear cached metadata
        self._metadata = None
    
    def initialize_file(
        self,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        ifo_names: List[str],
        num_ifos: int,
        onsource_samples: int,
        offsource_samples: int,
        compression: str = 'gzip',
        compression_opts: int = 4
    ) -> None:
        """
        Initialize HDF5 file with empty resizable datasets for incremental saving.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(self.path, 'w') as f:
            grp = f.create_group('glitches')
            
            # Create resizable datasets (maxshape=None on first dimension)
            grp.create_dataset(
                'onsource', 
                shape=(0, num_ifos, onsource_samples),
                maxshape=(None, num_ifos, onsource_samples),
                dtype=np.float32,
                chunks=(1, num_ifos, onsource_samples),
                compression=compression,
                compression_opts=compression_opts
            )
            grp.create_dataset(
                'offsource', 
                shape=(0, num_ifos, offsource_samples),
                maxshape=(None, num_ifos, offsource_samples),
                dtype=np.float32,
                chunks=(1, num_ifos, offsource_samples),
                compression=compression,
                compression_opts=compression_opts
            )
            grp.create_dataset(
                'gps_times', 
                shape=(0,), 
                maxshape=(None,), 
                dtype=np.float64
            )
            grp.create_dataset(
                'labels', 
                shape=(0,), 
                maxshape=(None,), 
                dtype=np.int32
            )
            
            # Extended metadata datasets (v2)
            grp.create_dataset('data_label', shape=(0,), maxshape=(None,), dtype=np.int8)
            grp.create_dataset('seen_in', shape=(0,), maxshape=(None,), dtype=np.uint8)
            grp.create_dataset('snr', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('peak_freq', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('duration', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('ml_confidence', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('mass1', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('mass2', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('distance', shape=(0,), maxshape=(None,), dtype=np.float32)
            grp.create_dataset('p_astro', shape=(0,), maxshape=(None,), dtype=np.float32)
            
            # Store metadata
            grp.attrs['sample_rate_hertz'] = sample_rate_hertz
            grp.attrs['onsource_duration'] = onsource_duration
            grp.attrs['offsource_duration'] = offsource_duration
            grp.attrs['ifo_names'] = ifo_names
            grp.attrs['version'] = 2
            
        self._metadata = None
        
    def append(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_times: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Append a batch of glitches to the existing cache file.
        
        Uses incremental GPS index update instead of full invalidation
        for O(1) instead of O(N) index maintenance.
        
        Args:
            onsource: Array of shape (N, IFOs, samples)
            offsource: Array of shape (N, IFOs, samples)
            gps_times: Array of shape (N,)
            labels: Array of shape (N,)
            metadata: Optional dict of metadata arrays, keys:
                data_label, seen_in, snr, peak_freq, duration, 
                ml_confidence, mass1, mass2, distance, p_astro
        """
        if not self.exists:
            raise FileNotFoundError(f"Cache file not initialized: {self.path}")
        
        n_new = len(gps_times)
        
        # Default metadata with sentinel values
        if metadata is None:
            metadata = {}
        
        default_metadata = {
            'data_label': np.full(n_new, -1, dtype=np.int8),
            'seen_in': np.zeros(n_new, dtype=np.uint8),
            'snr': np.full(n_new, np.nan, dtype=np.float32),
            'peak_freq': np.full(n_new, np.nan, dtype=np.float32),
            'duration': np.full(n_new, np.nan, dtype=np.float32),
            'ml_confidence': np.full(n_new, np.nan, dtype=np.float32),
            'mass1': np.full(n_new, np.nan, dtype=np.float32),
            'mass2': np.full(n_new, np.nan, dtype=np.float32),
            'distance': np.full(n_new, np.nan, dtype=np.float32),
            'p_astro': np.full(n_new, np.nan, dtype=np.float32),
        }
        
        # Merge provided metadata with defaults
        for key in default_metadata:
            if key not in metadata:
                metadata[key] = default_metadata[key]
            
        with h5py.File(self.path, 'a') as f:
            grp = f['glitches']
            
            n_current = grp['gps_times'].shape[0]
            n_total = n_current + n_new
            
            # Resize and append core data
            grp['onsource'].resize(n_total, axis=0)
            grp['onsource'][n_current:] = onsource.astype(np.float32)
            
            grp['offsource'].resize(n_total, axis=0)
            grp['offsource'][n_current:] = offsource.astype(np.float32)
            
            grp['gps_times'].resize(n_total, axis=0)
            grp['gps_times'][n_current:] = gps_times.astype(np.float64)
            
            grp['labels'].resize(n_total, axis=0)
            grp['labels'][n_current:] = labels.astype(np.int32)
            
            # Append extended metadata
            for key, arr in metadata.items():
                if key in grp:
                    grp[key].resize(n_total, axis=0)
                    grp[key][n_current:] = arr
            
        # Incremental GPS index update (O(n_new) instead of O(N) rebuild)
        with self._gps_lock:
            self._metadata = None  # Metadata (count) changed
            
            # If index exists, update incrementally; otherwise leave None for lazy rebuild
            if self._gps_index is not None:
                for i, gps in enumerate(gps_times):
                    key = gps_to_key(gps)
                    self._gps_index[key] = n_current + i

    
    def get_batch(
        self,
        indices: np.ndarray,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
        target_ifos: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a batch of glitches by indices, as windows of the request
        (``read_windows``): only the samples they need are read.

        Args:
            indices: Array of glitch indices to load
            sample_rate_hertz: Target sample rate
            onsource_duration: Target (padded) onsource duration
            offsource_duration: Target offsource duration
            target_ifos: Optional list of IFO names to retrieve (must be subset of cached IFOs)

        Returns:
            Tuple of (onsource, offsource, gps_times, labels)
        """
        meta = self.get_metadata()
        stored_rate = meta['sample_rate_hertz']
        stored_ifos = meta.get('ifo_names', [])

        # HDF5 fancy indexing requires sorted indices for efficiency and correctness
        # Sort indices, read, then restore original order
        indices = np.asarray(indices)
        sort_order = np.argsort(indices)
        sorted_indices = indices[sort_order]
        unsort_order = np.argsort(sort_order)  # Inverse permutation

        # Determine channel indices if filtering
        channel_indices = slice(None)
        if target_ifos:
            if not stored_ifos:
                # Fallback for old caches without ifo_names
                if len(target_ifos) == 1:
                     channel_indices = [0] # Assumption
                else: 
                     channel_indices = slice(None)
            else:
                try:
                    channel_indices = [stored_ifos.index(ifo) for ifo in target_ifos]
                except ValueError as e:
                    raise ValueError(f"Requested IFO not in cache (Cache: {stored_ifos}, Request: {target_ifos})") from e

        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            onsource, offsource = self.read_windows(
                grp['onsource'], grp['offsource'], stored_rate,
                sample_rate_hertz or stored_rate,
                onsource_duration or meta['onsource_duration'],
                offsource_duration or meta['offsource_duration'],
                rows=sorted_indices,
            )
            gps_times = grp['gps_times'][sorted_indices]
            labels = grp['labels'][sorted_indices]

        # Apply channel filtering
        if target_ifos:
            onsource = onsource[:, channel_indices, :]
            offsource = offsource[:, channel_indices, :]

        # Restore original order
        return (onsource[unsort_order], offsource[unsort_order],
                gps_times[unsort_order], labels[unsort_order])


def generate_transient_cache_path(data_directory: Optional[Path] = None, prefix: str = "transient") -> Path:
    """
    Generate standardized cache file path.

    Different prefixes are used for different transient types (e.g., "glitch" for
    glitches, "event" for GW events), resulting in separate cache files.

    Args:
        data_directory: Base directory for cache files. If None, the user cache
            (``gwosc_data.user_cache_dir``: ``GRAVYFLOW_CACHE_DIR``, else
            ``~/.gravyflow/cache``), the one root every gravyflow cache uses, so
            the same cache is found whatever the working directory.
        prefix: Filename prefix (e.g. "transient", "glitch", "event")

    Returns:
        Path like: data_directory/{prefix}_cache.h5
    """
    if data_directory is None:
        from gravyflow.src.dataset.acquisition.gwosc_data import (
            user_cache_dir, warn_of_moved_cache,
        )

        path = user_cache_dir() / f"{prefix}_cache.h5"
        # Before the one cache root, this cache ignored GRAVYFLOW_CACHE_DIR.
        warn_of_moved_cache(path, Path.home() / ".gravyflow" / "cache" / path.name)
        return path

    return Path(data_directory) / f"{prefix}_cache.h5"
