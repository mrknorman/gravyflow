"""
Unified caching interface for data acquisition.

This module provides a consistent caching API used by both NOISE and TRANSIENT
modes. The interface abstracts whether data is cached in memory or on disk.

Classes:
    AcquisitionCache: Abstract base class defining the cache interface
    MemoryCache: LRU memory cache for raw segment data (NOISE mode)
    DiskCache: HDF5 disk cache for processed transient windows (TRANSIENT mode)
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AcquisitionCache(ABC):
    """
    Abstract base class for data acquisition caching.
    
    Provides a unified interface for both memory and disk caching strategies.
    All cache implementations use integer GPS keys (10ms precision) for lookups.
    
    Subclasses must implement:
        has(key) - Check if key exists
        get(key, **params) - Retrieve data by key
        put(key, onsource, offsource, **metadata) - Store data
        clear() - Clear all cached data
    """
    
    @abstractmethod
    def has(self, key: int) -> bool:
        """
        Check if a GPS key exists in the cache.
        
        Args:
            key: Integer GPS key (10ms precision, from gps_to_key())
            
        Returns:
            True if key exists in cache
        """
        pass
    
    @abstractmethod
    def get(
        self, 
        key: int,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Retrieve cached data by GPS key.
        
        Args:
            key: Integer GPS key
            sample_rate_hertz: Desired output sample rate (for resampling)
            onsource_duration: Desired onsource duration (for cropping)
            offsource_duration: Desired offsource duration (for cropping)
            
        Returns:
            Tuple of (onsource, offsource, gps_time, label) or None if not found.
            Shapes: onsource/offsource = (IFOs, samples)
        """
        pass
    
    @abstractmethod
    def put(
        self,
        key: int,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_time: float,
        label: int = 0,
        **metadata
    ) -> None:
        """
        Store data in the cache.
        
        Args:
            key: Integer GPS key
            onsource: Array of shape (IFOs, samples)
            offsource: Array of shape (IFOs, samples)
            gps_time: Float GPS time (for reference)
            label: Data label (0=NOISE, 1=GLITCH, 2=EVENT)
            **metadata: Additional metadata to store
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached data."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Return number of items in cache."""
        pass


class MemoryCache(AcquisitionCache):
    """
    LRU memory cache for raw segment data.
    
    Used primarily by NOISE mode to cache downloaded segments and avoid
    re-downloading the same data within a session.
    
    Supports both integer GPS keys and string composite keys.
    
    Args:
        maxsize: Maximum number of items to keep in cache (default: 8)
    
    Note:
        This cache does NOT support resampling or cropping - it stores raw data.
        The sample_rate/duration parameters in get() are ignored.
    """
    
    def __init__(self, maxsize: int = 8):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
    
    def has(self, key) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def get(
        self,
        key,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Optional[any]:
        """
        Get cached data by key.
        
        Note: sample_rate/duration params are ignored - returns raw cached data.
        """
        if key not in self._cache:
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return self._cache[key]
    
    def put(
        self,
        key,
        data: any,
        **metadata
    ) -> None:
        """
        Store data in cache with LRU eviction.
        
        Simplified interface accepts any data and any key type.
        If cache is full, evicts the least recently used item.
        
        Args:
            key: Cache key (int or str)
            data: Data to cache
            **metadata: Ignored (for interface compatibility)
        """
        # Evict oldest if at capacity
        while len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        
        self._cache[key] = data
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)


class DiskCache(AcquisitionCache):
    """
    HDF5 disk cache for processed transient windows.
    
    Used by TRANSIENT mode to persist downloaded and processed event/glitch
    windows to disk. Supports resampling and cropping at retrieval time.
    
    This is a thin wrapper around TransientCache that implements the unified
    AcquisitionCache interface.
    
    Args:
        cache_path: Path to HDF5 cache file
        mode: File mode ('r', 'w', 'a')
        
    Note:
        The underlying TransientCache stores data at maximum parameters
        (sample rate, duration) and resamples/crops on retrieval.
    """
    
    def __init__(self, cache_path: Path, mode: str = 'a'):
        # Import here to avoid circular imports
        from gravyflow.src.dataset.features.glitch_cache import TransientCache
        self._cache = TransientCache(cache_path, mode=mode)
        self._path = cache_path
    
    @property
    def path(self) -> Path:
        """Return the cache file path."""
        return self._path
    
    @property
    def exists(self) -> bool:
        """Check if cache file exists."""
        return self._cache.exists
    
    def has(self, key: int) -> bool:
        """Check if GPS key exists in cache."""
        return self._cache.has_key(key)
    
    def get(
        self,
        key: int,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
        """
        Get cached data by key with optional resampling/cropping.
        
        Returns:
            Tuple of (onsource, offsource, gps_time, label) or None if not found.
        """
        return self._cache.get_by_key(
            key,
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration,
        )
    
    def put(
        self,
        key: int,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_time: float,
        label: int = 0,
        **metadata
    ) -> None:
        """
        Append data to disk cache.
        
        Note: TransientCache handles duplicate detection internally.
        """
        self._cache.append_single(
            onsource=onsource,
            offsource=offsource,
            gps_time=gps_time,
            label=label,
            gps_key=key
        )
    
    def clear(self) -> None:
        """Reset the cache (delete and recreate file)."""
        self._cache.reset()
    
    @property
    def size(self) -> int:
        """Return number of items in cache."""
        if not self._cache.exists:
            return 0
        meta = self._cache.get_metadata()
        return meta.get('num_glitches', 0)
    
    # === DiskCache-specific methods (not in base interface) ===
    
    def initialize(
        self,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        ifo_names: list,
        num_ifos: int,
        onsource_samples: int,
        offsource_samples: int
    ) -> None:
        """
        Initialize the cache file with metadata.
        
        Must be called before first put() if cache doesn't exist.
        """
        self._cache.initialize_file(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration,
            ifo_names=ifo_names,
            num_ifos=num_ifos,
            onsource_samples=onsource_samples,
            offsource_samples=offsource_samples
        )
    
    def validate_request(
        self,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float
    ) -> None:
        """
        Validate that a request can be satisfied.
        
        Raises:
            ValueError: If request exceeds cached limits.
        """
        self._cache.validate_request(
            sample_rate_hertz,
            onsource_duration,
            offsource_duration
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get cache metadata."""
        return self._cache.get_metadata()
    
    def get_all_gps_times(self) -> np.ndarray:
        """Return all GPS times in cache."""
        return self._cache.get_all_gps_times()
    
    # === Passthrough methods for TransientCache-specific functionality ===
    # These allow transient.py to access internal cache state efficiently
    
    @property
    def in_memory(self) -> bool:
        """Check if cache data is loaded to memory."""
        return self._cache.in_memory
    
    def has_key(self, gps_key: int) -> bool:
        """Check if GPS key exists (alias for has())."""
        return self._cache.has_key(gps_key)
    
    @property
    def _gps_index(self) -> Dict[int, int]:
        """Access internal GPS index mapping. For performance optimization only."""
        return self._cache._gps_index
    
    @property
    def _mem_onsource(self) -> np.ndarray:
        """Access in-memory onsource array. For performance optimization only."""
        return self._cache._mem_onsource
    
    @property
    def _mem_offsource(self) -> np.ndarray:
        """Access in-memory offsource array. For performance optimization only."""
        return self._cache._mem_offsource
    
    def get_batch(
        self,
        indices: np.ndarray,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        target_ifos: list = None
    ):
        """
        Get a batch of cached data by indices.
        
        Passthrough to TransientCache.get_batch() for efficient bulk retrieval.
        
        Args:
            indices: Array of cache indices to load
            sample_rate_hertz: Target sample rate
            onsource_duration: Target onsource duration
            offsource_duration: Target offsource duration
            target_ifos: Optional list of IFO names to retrieve
        
        Returns:
            Tuple of (onsource, offsource, gps_times, labels)
        """
        return self._cache.get_batch(
            indices=indices,
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration,
            target_ifos=target_ifos
        )
    
    def append_single(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_time: float,
        label: int,
        gps_key: int = None
    ) -> None:
        """Append single item to cache (passthrough to TransientCache)."""
        self._cache.append_single(
            onsource=onsource,
            offsource=offsource,
            gps_time=gps_time,
            label=label,
            gps_key=gps_key
        )
    
    def reset(self) -> None:
        """Reset cache (alias for clear())."""
        self._cache.reset()
    
    def initialize_file(
        self,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        ifo_names: list,
        num_ifos: int,
        onsource_samples: int,
        offsource_samples: int
    ) -> None:
        """Initialize cache file (alias for initialize())."""
        self._cache.initialize_file(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration,
            ifo_names=ifo_names,
            num_ifos=num_ifos,
            onsource_samples=onsource_samples,
            offsource_samples=offsource_samples
        )
    
    # === Chunked memory cache passthrough properties ===
    
    @property
    def _chunk_in_memory(self) -> bool:
        """Check if chunked cache mode is enabled."""
        return self._cache._chunk_in_memory
    
    @property
    def _gps_lock(self):
        """Access GPS index lock."""
        return self._cache._gps_lock
    
    def enable_chunked_mode(
        self,
        chunk_size: int = 5000,
        sample_rate_hertz: float = None,
        onsource_duration: float = None,
        offsource_duration: float = None
    ) -> None:
        """Enable chunked memory mode for faster cache hits."""
        self._cache.enable_chunked_mode(
            chunk_size=chunk_size,
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration
        )
    
    def get_from_chunk(
        self,
        idx: int,
        target_ifos: list = None
    ):
        """Get sample from chunked memory cache."""
        return self._cache.get_from_chunk(idx, target_ifos=target_ifos)
    
    def get_batch_from_chunk(
        self,
        indices: np.ndarray,
        target_ifos: list = None
    ):
        """Get batch of samples from chunked cache."""
        return self._cache.get_batch_from_chunk(indices, target_ifos=target_ifos)
    
    def _build_gps_index(self):
        """Build GPS index (passthrough)."""
        return self._cache._build_gps_index()
