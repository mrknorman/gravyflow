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

logger = logging.getLogger(__name__)

import numpy as np
import h5py
from keras import ops

from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from gravyflow.src.utils.gps import gps_to_key, gps_array_to_keys

# Maximum supported parameters - data is stored at these settings and 
# can be downsampled/cropped at load time (but not upsampled/extended)
#
# NOTE: Download padding (in segment_builders.py) must be >= onsource_half + offsource_max
# to ensure there's enough data for offsource BEFORE onsource. With 32s onsource (±16s)
# and 32s offsource, we need 48s padding on each side.
CACHE_SAMPLE_RATE_HERTZ = 4096.0  # Max supported sample rate
CACHE_ONSOURCE_DURATION = 32.0   # Max onsource window (seconds) - allows shifting augmentation
CACHE_OFFSOURCE_DURATION = 32.0  # Max offsource window (seconds)
CACHE_PADDING_DURATION = 1.0     # Extra padding for cropping


class GlitchCache:
    """
    Manages standardized glitch data cache with efficient bulk I/O.
    
    All data is stored at maximum supported parameters and resampled/cropped
    at load time to match the requested configuration.
    
    Args:
        path: Path to the HDF5 cache file
        mode: File access mode ('r' for read, 'w' for write, 'a' for append)
    
    Example:
        # Writing cache
        cache = GlitchCache(Path("glitches_O3_L1.h5"), mode='w')
        cache.save_all(onsource, offsource, gps_times, labels,
                      sample_rate_hertz=4096.0, ifo_names=['L1'])
        
        # Reading cache
        cache = GlitchCache(Path("glitches_O3_L1.h5"), mode='r')
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
        
    @property
    def exists(self) -> bool:
        """Check if cache file exists."""
        return self.path.exists()
    
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
        
        # Build index on first call (lazy)
        if not hasattr(self, '_gps_index') or self._gps_index is None:
            self._gps_index = self._build_gps_index()
        
        # Exact match using integer key
        return gps_to_key(gps_time) in self._gps_index
    
    def get_closest_gps(self, gps_time: float) -> Optional[float]:
        """Get the cached GPS time matching the integer key, or None if not found."""
        if not self.exists:
            return None
        
        if not hasattr(self, '_gps_index') or self._gps_index is None:
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
        
        # Build index on first call (lazy)
        if not hasattr(self, '_gps_index') or self._gps_index is None:
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
             
        # Ensure index exists
        if not hasattr(self, '_gps_index') or self._gps_index is None:
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
        
        if not hasattr(self, '_gps_index') or self._gps_index is None:
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
        gps_key: Optional[int] = None
    ) -> None:
        """
        Append a single glitch to the cache.
        
        Args:
            onsource: Array of shape (IFOs, samples)
            offsource: Array of shape (IFOs, samples)
            gps_time: GPS time of the glitch (for storage and display)
            label: GlitchType integer label
            gps_key: Optional GPS key (10ms precision). If not provided, computed from gps_time.
        """
        # Expand dims to (1, IFOs, samples) for append
        self.append(
            onsource=onsource[np.newaxis, ...],
            offsource=offsource[np.newaxis, ...],
            gps_times=np.array([gps_time]),
            labels=np.array([label])
        )
        
        # Update GPS index using key directly (no conversion if provided)
        if gps_key is None:
            gps_key = gps_to_key(gps_time)
        
        if hasattr(self, '_gps_index') and self._gps_index is not None:
            self._gps_index[gps_key] = len(self._gps_index)
        else:
            self._gps_index = None  # Force rebuild on next access
    
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
    
    def preload_to_memory(
        self,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float
    ) -> None:
        """
        Load all cache data to RAM at the specified settings.
        
        After calling this, get_batch() will serve from memory instead of disk.
        Significantly faster but requires all data to fit in RAM.
        
        Args:
            sample_rate_hertz: Target sample rate for loaded data
            onsource_duration: Target onsource duration 
            offsource_duration: Target offsource duration
        """
        if self._in_memory:
            logger.info("Cache already loaded to memory, skipping reload")
            return
        
        self.validate_request(sample_rate_hertz, onsource_duration, offsource_duration)
        
        meta = self.get_metadata()
        num_glitches = meta['num_glitches']
        
        logger.info(f"Loading {num_glitches} glitches to memory at {sample_rate_hertz}Hz, "
                     f"{onsource_duration}s/{offsource_duration}s...")
        
        # Use get_batch to load all with correct cropping/resampling
        # Load in chunks to show progress
        chunk_size = 10000
        all_ons = []
        all_offs = []
        all_gps = []
        all_labels = []
        
        for i in range(0, num_glitches, chunk_size):
            end_idx = min(i + chunk_size, num_glitches)
            indices = np.arange(i, end_idx)
            
            ons, offs, gps, labels = self.get_batch(
                indices,
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration=onsource_duration,
                offsource_duration=offsource_duration
            )
            
            all_ons.append(ons)
            all_offs.append(offs)
            all_gps.append(gps)
            all_labels.append(labels)
            
            if i % 50000 == 0:
                logger.info(f"  Loaded {end_idx}/{num_glitches} glitches...")
        
        self._mem_onsource = np.concatenate(all_ons, axis=0)
        self._mem_offsource = np.concatenate(all_offs, axis=0)
        self._mem_gps = np.concatenate(all_gps, axis=0)
        self._mem_labels = np.concatenate(all_labels, axis=0)
        self._mem_sample_rate = sample_rate_hertz
        self._mem_ons_dur = onsource_duration
        self._mem_off_dur = offsource_duration
        self._in_memory = True
        
        mem_size_mb = (self._mem_onsource.nbytes + self._mem_offsource.nbytes) / 1024 / 1024
        logger.info(f"Cache loaded to memory: {mem_size_mb:.1f} MB, {num_glitches} glitches")
    
    def get_batch_from_memory(
        self,
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a batch from in-memory cache (much faster than disk).
        
        Must call preload_to_memory() first.
        """
        if not self._in_memory:
            raise RuntimeError("Cache not loaded to memory. Call preload_to_memory() first.")
        
        return (
            self._mem_onsource[indices],
            self._mem_offsource[indices],
            self._mem_gps[indices],
            self._mem_labels[indices]
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
            
            # Store metadata
            grp.attrs['sample_rate_hertz'] = sample_rate_hertz
            grp.attrs['onsource_duration'] = onsource_duration
            grp.attrs['offsource_duration'] = offsource_duration
            grp.attrs['ifo_names'] = ifo_names
            grp.attrs['version'] = 1
            
        self._metadata = None
        
    def append(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_times: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """
        Append a batch of glitches to the existing cache file.
        """
        if not self.exists:
            raise FileNotFoundError(f"Cache file not initialized: {self.path}")
            
        with h5py.File(self.path, 'a') as f:
            grp = f['glitches']
            
            n_new = len(gps_times)
            n_current = grp['gps_times'].shape[0]
            n_total = n_current + n_new
            
            # Resize and append
            grp['onsource'].resize(n_total, axis=0)
            grp['onsource'][n_current:] = onsource.astype(np.float32)
            
            grp['offsource'].resize(n_total, axis=0)
            grp['offsource'][n_current:] = offsource.astype(np.float32)
            
            grp['gps_times'].resize(n_total, axis=0)
            grp['gps_times'][n_current:] = gps_times.astype(np.float64)
            
            grp['labels'].resize(n_total, axis=0)
            grp['labels'][n_current:] = labels.astype(np.int32)
            
        self._metadata = None  # Metadata (count) changed
    
    def get_batch(
        self,
        indices: np.ndarray,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration: Optional[float] = None,
        offsource_duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a batch of glitches by indices.
        
        More memory efficient than load_all() when working with subsets.
        
        Args:
            indices: Array of glitch indices to load
            sample_rate_hertz: Target sample rate
            onsource_duration: Target onsource duration
            offsource_duration: Target offsource duration
            
        Returns:
            Tuple of (onsource, offsource, gps_times, labels)
        """
        meta = self.get_metadata()
        stored_rate = meta['sample_rate_hertz']
        stored_ons_dur = meta['onsource_duration']
        stored_off_dur = meta['offsource_duration']
        
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
        
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            
            # Read only the slice we need: [sorted_indices, :ifos, start:end]
            onsource = grp['onsource'][sorted_indices, :, ons_start:ons_end]
            offsource = grp['offsource'][sorted_indices, :, off_start:off_end]
            gps_times = grp['gps_times'][sorted_indices]
            labels = grp['labels'][sorted_indices]
        
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


def generate_glitch_cache_path(
    observing_run: str = None,
    ifo: str = None,
    data_directory: Optional[Path] = None
) -> Path:
    """
    Generate standardized cache file path.
    
    Uses a single cache for all observing runs and IFOs
    to avoid duplicate downloads of the same GPS times.
    
    Args:
        observing_run: Ignored (kept for backward compatibility)
        ifo: Ignored (kept for backward compatibility)
        data_directory: Base directory for cache files
        
    Returns:
        Path like: data_directory/glitch_cache.h5
        
    Note:
        The cache stores glitches from all runs and IFOs together.
        GPS keys ensure no duplicates - same event downloaded once regardless
        of how many IFOs/runs request it.
    """
    if data_directory is None:
        data_directory = Path("./gravyflow_data")
    
    # Single cache for all data
    filename = "glitch_cache.h5"
    return data_directory / filename
