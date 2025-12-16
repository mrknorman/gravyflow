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

import numpy as np
import h5py

# Maximum supported parameters - data is stored at these settings and 
# can be downsampled/cropped at load time (but not upsampled/extended)
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
    
    def _build_gps_index(self) -> Dict[float, int]:
        """Build a dictionary mapping GPS times to their indices in the cache."""
        if not self.exists:
            return {}
            
        with h5py.File(self.path, 'r') as f:
            if 'glitches' not in f or 'gps_times' not in f['glitches']:
                return {}
            
            gps_times = f['glitches']['gps_times'][:]
            # Use rounded GPS times as keys to handle floating point comparison
            return {round(float(gps), 3): i for i, gps in enumerate(gps_times)}
    
    def has_gps(self, gps_time: float, tolerance: float = 0.5) -> bool:
        """Check if a GPS time exists in the cache (with tolerance for slight differences)."""
        if not self.exists:
            return False
        
        # Build index on first call (lazy)
        if not hasattr(self, '_gps_index') or self._gps_index is None:
            self._gps_index = self._build_gps_index()
        
        # Try exact match first
        if round(gps_time, 3) in self._gps_index:
            return True
        
        # Try tolerance-based match
        for cached_gps in self._gps_index.keys():
            if abs(cached_gps - gps_time) < tolerance:
                return True
        return False
    
    def get_closest_gps(self, gps_time: float, tolerance: float = 0.5) -> float:
        """Find the closest cached GPS time within tolerance, or None if not found."""
        if not self.exists:
            return None
        
        if not hasattr(self, '_gps_index') or self._gps_index is None:
            self._gps_index = self._build_gps_index()
        
        # Try exact match first
        rounded = round(gps_time, 3)
        if rounded in self._gps_index:
            return rounded
        
        # Find closest within tolerance
        best_match = None
        best_diff = tolerance
        for cached_gps in self._gps_index.keys():
            diff = abs(cached_gps - gps_time)
            if diff < best_diff:
                best_diff = diff
                best_match = cached_gps
        return best_match
    
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
            Tuple of (onsource, offsource, gps_time, label) or None if not found.
        """
        if not self.has_gps(gps_time):
            return None
        
        closest_gps = self.get_closest_gps(gps_time)
        if closest_gps is None:
            return None
        idx = self._gps_index[closest_gps]
        ons, offs, gps, labels = self.get_batch(
            np.array([idx]),
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration=onsource_duration,
            offsource_duration=offsource_duration
        )
        return ons[0], offs[0], gps[0], labels[0]
    
    def append_single(
        self,
        onsource: np.ndarray,
        offsource: np.ndarray,
        gps_time: float,
        label: int
    ) -> None:
        """
        Append a single glitch to the cache.
        
        Args:
            onsource: Array of shape (IFOs, samples)
            offsource: Array of shape (IFOs, samples)
            gps_time: GPS time of the glitch
            label: GlitchType integer label
        """
        # Expand dims to (1, IFOs, samples) for append
        self.append(
            onsource=onsource[np.newaxis, ...],
            offsource=offsource[np.newaxis, ...],
            gps_times=np.array([gps_time]),
            labels=np.array([label])
        )
        
        # Update GPS index
        if hasattr(self, '_gps_index') and self._gps_index is not None:
            self._gps_index[round(gps_time, 3)] = len(self._gps_index)
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
        import logging
        
        if self._in_memory:
            logging.info("Cache already loaded to memory, skipping reload")
            return
        
        self.validate_request(sample_rate_hertz, onsource_duration, offsource_duration)
        
        meta = self.get_metadata()
        num_glitches = meta['num_glitches']
        
        logging.info(f"Loading {num_glitches} glitches to memory at {sample_rate_hertz}Hz, "
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
                logging.info(f"  Loaded {end_idx}/{num_glitches} glitches...")
        
        self._mem_onsource = np.concatenate(all_ons, axis=0)
        self._mem_offsource = np.concatenate(all_offs, axis=0)
        self._mem_gps = np.concatenate(all_gps, axis=0)
        self._mem_labels = np.concatenate(all_labels, axis=0)
        self._mem_sample_rate = sample_rate_hertz
        self._mem_ons_dur = onsource_duration
        self._mem_off_dur = offsource_duration
        self._in_memory = True
        
        mem_size_mb = (self._mem_onsource.nbytes + self._mem_offsource.nbytes) / 1024 / 1024
        logging.info(f"Cache loaded to memory: {mem_size_mb:.1f} MB, {num_glitches} glitches")
    
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
        
        # Resample if needed
        if sample_rate_hertz is not None and sample_rate_hertz != meta['sample_rate_hertz']:
            ratio = int(meta['sample_rate_hertz'] / sample_rate_hertz)
            onsource = onsource[:, :, ::ratio]
            offsource = offsource[:, :, ::ratio]
        
        # Crop onsource if needed
        if onsource_duration is not None and onsource_duration < meta['onsource_duration']:
            target_rate = sample_rate_hertz or meta['sample_rate_hertz']
            target_samples = int(onsource_duration * target_rate)
            current_samples = onsource.shape[2]
            if current_samples > target_samples:
                # Crop from center
                start = (current_samples - target_samples) // 2
                onsource = onsource[:, :, start:start + target_samples]
        
        # Crop offsource if needed
        if offsource_duration is not None and offsource_duration < meta['offsource_duration']:
            target_rate = sample_rate_hertz or meta['sample_rate_hertz']
            target_samples = int(offsource_duration * target_rate)
            current_samples = offsource.shape[2]
            if current_samples > target_samples:
                # Crop from center
                start = (current_samples - target_samples) // 2
                offsource = offsource[:, :, start:start + target_samples]
        
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
            
        logging.info(
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
        
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            
            # Read only the slice we need: [indices, :ifos, start:end]
            # HDF5 fancy indexing: indices for batch dim, slice for sample dim
            onsource = grp['onsource'][indices, :, ons_start:ons_end]
            offsource = grp['offsource'][indices, :, off_start:off_end]
            gps_times = grp['gps_times'][indices]
            labels = grp['labels'][indices]
        
        # Apply resampling if needed (now on much smaller arrays)
        if resample_ratio > 1:
            onsource = onsource[:, :, ::resample_ratio]
            offsource = offsource[:, :, ::resample_ratio]
        
        return onsource, offsource, gps_times, labels


def generate_glitch_cache_path(
    observing_run: str,
    ifo: str,
    data_directory: Optional[Path] = None
) -> Path:
    """
    Generate standardized cache file path without hash.
    
    Args:
        observing_run: e.g., "O3", "O2"
        ifo: e.g., "L1", "H1"
        data_directory: Base directory for cache files
        
    Returns:
        Path like: data_directory/glitch_cache_O3_L1.h5
    """
    if data_directory is None:
        data_directory = Path("./gravyflow_data")
    
    filename = f"glitch_cache_{observing_run}_{ifo}.h5"
    return data_directory / filename
