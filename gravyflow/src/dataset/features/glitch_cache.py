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
CACHE_ONSOURCE_DURATION = 2.0    # Max onsource window (seconds)
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
        
    @property
    def exists(self) -> bool:
        """Check if cache file exists."""
        return self.path.exists()
    
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
        
        with h5py.File(self.path, 'r') as f:
            grp = f['glitches']
            
            # Load only requested indices
            onsource = grp['onsource'][indices]
            offsource = grp['offsource'][indices]
            gps_times = grp['gps_times'][indices]
            labels = grp['labels'][indices]
        
        # Apply resampling/cropping as in load_all
        if sample_rate_hertz is not None and sample_rate_hertz != meta['sample_rate_hertz']:
            ratio = int(meta['sample_rate_hertz'] / sample_rate_hertz)
            onsource = onsource[:, :, ::ratio]
            offsource = offsource[:, :, ::ratio]
        
        if onsource_duration is not None and onsource_duration < meta['onsource_duration']:
            target_rate = sample_rate_hertz or meta['sample_rate_hertz']
            target_samples = int(onsource_duration * target_rate)
            current_samples = onsource.shape[2]
            if current_samples > target_samples:
                start = (current_samples - target_samples) // 2
                onsource = onsource[:, :, start:start + target_samples]
        
        if offsource_duration is not None and offsource_duration < meta['offsource_duration']:
            target_rate = sample_rate_hertz or meta['sample_rate_hertz']
            target_samples = int(offsource_duration * target_rate)
            current_samples = offsource.shape[2]
            if current_samples > target_samples:
                start = (current_samples - target_samples) // 2
                offsource = offsource[:, :, start:start + target_samples]
        
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
