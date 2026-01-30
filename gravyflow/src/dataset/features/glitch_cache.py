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
from gravyflow.src.utils.gps import gps_to_key, gps_array_to_keys
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
        cache.validate_request(2048.0, 1.0, 16.0)  # Raises if unsatisfiable
        data = cache.load_all()  # Single bulk read
    """
    
    @staticmethod
    def crop_and_resample(
        data: np.ndarray,
        current_rate: float,
        target_rate: float,
        target_duration: float
    ) -> np.ndarray:
        """
        Crop and resample data along the last axis (samples).
        
        Handles arbitrary leading dimensions (Batch, IFOs, etc).
        Only supports integer downsampling.
        
        Raises:
            ValueError: If target_rate > current_rate or rates are not integer multiples.
        """
        if data.size == 0:
            return data
            
        # 1. Resample (downsample only)
        if target_rate != current_rate:
            if target_rate > current_rate:
                raise ValueError(
                    f"Cannot upsample from {current_rate}Hz to {target_rate}Hz. "
                    "Upsampling is not supported."
                )
            
            # Validate integer multiple relationship
            if current_rate % target_rate != 0:
                raise ValueError(
                    f"Sample rates must be integer multiples. "
                    f"Cannot downsample from {current_rate}Hz to {target_rate}Hz. "
                    f"Ratio {current_rate/target_rate:.4f} is not an integer."
                )
            
            ratio = int(current_rate / target_rate)
            if ratio > 1:
                data = data[..., ::ratio]
                
        # 2. Crop
        target_samples = int(target_duration * target_rate)
        current_samples = data.shape[-1]
        
        if current_samples > target_samples:
            # Center crop
            start = (current_samples - target_samples) // 2
            data = data[..., start : start + target_samples]
            
        return data

    def __init__(self, path: Path, mode: str = 'r'):
        self.path = Path(path)
        self.mode = mode
        self._metadata: Optional[Dict] = None
        
        # In-memory cache for fast serving
        self._in_memory = False
        self._mem_onsource: Optional[np.ndarray] = None
        self._mem_offsource: Optional[np.ndarray] = None
        self._mem_gps: Optional[np.ndarray] = None
        self._mem_labels: Optional[np.ndarray] = None
        self._mem_sample_rate: Optional[float] = None
        self._mem_ons_dur: Optional[float] = None
        self._mem_off_dur: Optional[float] = None
        
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
        self._in_memory = False
        self._mem_onsource = None
        self._mem_offsource = None
    
    @property
    def in_memory(self) -> bool:
        """Check if data is loaded to memory."""
        return self._in_memory
    
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
    
    
    def load_all(
        self,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load entire cache to memory in a single read.
        
        Optionally resample/crop to requested parameters.
        
        Args:
            sample_rate_hertz: Target sample rate (must be <= cached rate)
            onsource_duration: Target onsource duration (must be <= cached)
            offsource_duration: Target offsource duration (must be <= cached)
            
        Returns:
            Dict with keys: 'onsource', 'offsource', 'gps_times', 'labels'
        """
        meta = self.get_metadata()
        
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            
            # Load raw data
            onsource = grp['onsource'][:]
            offsource = grp['offsource'][:]
            gps_times = grp['gps_times'][:]
            labels = grp['labels'][:]
        
        # Use unified crop/resample logic
        target_rate = sample_rate_hertz or meta['sample_rate_hertz']
        
        # Process onsource
        target_on_dur = onsource_duration or meta['onsource_duration']
        onsource = self.crop_and_resample(
            onsource, 
            meta['sample_rate_hertz'], 
            target_rate, 
            target_on_dur
        )
        
        # Process offsource
        target_off_dur = offsource_duration or meta['offsource_duration']
        offsource = self.crop_and_resample(
            offsource, 
            meta['sample_rate_hertz'], 
            target_rate, 
            target_off_dur
        )
        
        return {
            'onsource': onsource,
            'offsource': offsource,
            'gps_times': gps_times,
            'labels': labels
        }
    
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
        
        # Read raw data for this chunk range
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            chunk_onsource = grp['onsource'][start:end]
            chunk_offsource = grp['offsource'][start:end]
            chunk_gps = grp['gps_times'][start:end]
            chunk_labels = grp['labels'][start:end]
        
        # Apply crop/resample
        chunk_onsource = self.crop_and_resample(
            chunk_onsource, stored_rate, target_rate, target_ons_dur
        )
        chunk_offsource = self.crop_and_resample(
            chunk_offsource, stored_rate, target_rate, target_off_dur
        )
        
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
        Load a batch of glitches by indices.
        
        More memory efficient than load_all() when working with subsets.
        
        Args:
            indices: Array of glitch indices to load
            sample_rate_hertz: Target sample rate
            onsource_duration: Target onsource duration
            offsource_duration: Target offsource duration
            target_ifos: Optional list of IFO names to retrieve (must be subset of cached IFOs)
            
        Returns:
            Tuple of (onsource, offsource, gps_times, labels)
        """
        meta = self.get_metadata()
        stored_rate = meta['sample_rate_hertz']
        stored_ons_dur = meta['onsource_duration']
        stored_off_dur = meta['offsource_duration']
        stored_ifos = meta.get('ifo_names', [])
        
        # Determine target parameters
        target_rate = sample_rate_hertz if sample_rate_hertz else stored_rate
        target_ons_dur = onsource_duration if onsource_duration else stored_ons_dur
        target_off_dur = offsource_duration if offsource_duration else stored_off_dur
        
        # Calculate resample ratio (only supports integer downsampling)
        resample_ratio = int(stored_rate / target_rate) if target_rate != stored_rate else 1
        
        # Calculate sample counts at STORED rate (before resampling)
        stored_ons_samples = int(stored_ons_dur * stored_rate)
        stored_off_samples = int(stored_off_dur * stored_rate)
        
        # Calculate how many stored samples we need to read to get target duration after resampling
        target_ons_samples = int(target_ons_dur * target_rate)
        target_off_samples = int(target_off_dur * target_rate)
        
        # Samples to read at stored rate (will become target_samples after resampling)
        read_ons_samples = target_ons_samples * resample_ratio
        read_off_samples = target_off_samples * resample_ratio
        
        # Center crop indices at stored rate
        ons_start = (stored_ons_samples - read_ons_samples) // 2
        ons_end = ons_start + read_ons_samples
        off_start = (stored_off_samples - read_off_samples) // 2
        off_end = off_start + read_off_samples
        
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
            
            # Read only the slice we need: [sorted_indices, :ifos, start:end]
            # We read all channels then slice in memory for simplicity/safety
            onsource = grp['onsource'][sorted_indices, :, ons_start:ons_end]
            offsource = grp['offsource'][sorted_indices, :, off_start:off_end]
            gps_times = grp['gps_times'][sorted_indices]
            labels = grp['labels'][sorted_indices]
            
        # Apply channel filtering
        if target_ifos:
            onsource = onsource[:, channel_indices, :]
            offsource = offsource[:, channel_indices, :]
        
        # Restore original order
        onsource = onsource[unsort_order]
        offsource = offsource[unsort_order]
        gps_times = gps_times[unsort_order]
        labels = labels[unsort_order]
        
        # Apply resampling if needed (now on much smaller arrays)
        if resample_ratio > 1:
            onsource = onsource[:, :, ::resample_ratio]
            offsource = offsource[:, :, ::resample_ratio]
        
        return onsource, offsource, gps_times, labels


    def stream_batches(
        self,
        batch_size: int,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        scale_factor: float = 1.0,
        seed: int = None,
        allowed_segments: np.ndarray = None
    ):
        """
        Generator that yields random batches from the cache.
        
        Streams batches directly from disk to avoid loading entire cache into memory.
        Supports optional segment filtering for train/val/test splits.
        
        Args:
            batch_size: Number of samples per batch
            sample_rate_hertz: Target sample rate
            onsource_duration: Target onsource duration (including any padding)
            offsource_duration: Target offsource duration
            scale_factor: Amplitude scaling factor
            seed: Random seed for reproducibility
            allowed_segments: Optional (N, 2) array of [start, end] GPS segments to filter by
            
        Yields:
            Tuples of (onsource, offsource, gps_times, labels)
        """
        from numpy.random import default_rng
        
        meta = self.get_metadata()
        n_glitches = meta['num_glitches']
        
        # Get GPS times for filtering
        gps_all = self.get_all_gps_times()
        
        # Build index of valid glitches
        valid_indices = np.arange(n_glitches)
        
        if allowed_segments is not None and len(allowed_segments) > 0:
            # Handle potential extra dimensions
            if allowed_segments.ndim == 3 and allowed_segments.shape[1] == 1:
                allowed_segments = allowed_segments.reshape(-1, 2)
            elif allowed_segments.ndim == 1 and allowed_segments.shape[0] == 2:
                allowed_segments = allowed_segments.reshape(1, 2)
            
            # Filter based on segment overlap
            keep_mask = np.zeros(len(gps_all), dtype=bool)
            for start, end in allowed_segments:
                in_segment = (gps_all >= start) & (gps_all <= end)
                keep_mask |= in_segment
                
            valid_indices = np.where(keep_mask)[0]
            n_glitches = len(valid_indices)
        
        if n_glitches == 0:
            raise ValueError(
                f"No glitches found after filtering! "
                f"(Allowed segments: {len(allowed_segments) if allowed_segments is not None else 0})"
            )
        
        rng = default_rng(seed)
        
        while True:
            # Random batch indices from valid_indices (with replacement for infinite generator)
            batch_idx = rng.choice(n_glitches, size=batch_size, replace=True)
            batch_indices = valid_indices[batch_idx]
            
            # Stream batch from disk
            onsource, offsource, gps_times, labels = self.get_batch(
                batch_indices,
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration=onsource_duration,
                offsource_duration=offsource_duration
            )
            
            # Convert to dict format with (B, I) shape for GPS and labels
            num_ifos = onsource.shape[1]
            gps_expanded = ops.tile(ops.expand_dims(gps_times, axis=-1), (1, num_ifos))
            labels_expanded = ops.tile(ops.expand_dims(labels, axis=-1), (1, num_ifos))
            
            yield {
                RV.ONSOURCE: onsource * scale_factor,
                RV.OFFSOURCE: offsource * scale_factor,
                RV.TRANSIENT_GPS_TIME: gps_expanded,
                RV.DATA_LABEL: ops.full((batch_size, num_ifos), 1, dtype="int32"),  # GLITCHES = 1
                RV.SUB_TYPE: labels_expanded,
            }


def generate_transient_cache_path(data_directory: Optional[Path] = None, prefix: str = "transient") -> Path:
    """
    Generate standardized cache file path.
    
    Creates cache files in ~/.gravyflow/cache/ by default. Different prefixes
    are used for different transient types (e.g., "glitch" for glitches, 
    "event" for GW events), resulting in separate cache files.
    
    The default location is ~/.gravyflow/cache/ to ensure the same cache
    is used regardless of current working directory.
    
    Args:
        data_directory: Base directory for cache files. If None, uses ~/.gravyflow/cache/
        prefix: Filename prefix (e.g. "transient", "glitch", "event")
        
    Returns:
        Path like: data_directory/{prefix}_cache.h5
    """
    if data_directory is None:
        # Use absolute path in home directory to ensure consistent cache location
        data_directory = Path.home() / ".gravyflow" / "cache"
        data_directory.mkdir(parents=True, exist_ok=True)
    
    return data_directory / f"{prefix}_cache.h5"
