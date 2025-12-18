"""
GPS time key utilities for efficient lookup operations.

Uses 2 decimal place (0.01s = 10ms) precision integer keys for exact matching.
GPS times span 2015-2040 (~1.1B to ~1.9B), which at 2dp * 100 fits in int64.

Example:
    gps_to_key(1125000000.75)  # → 112500000075
    key_to_gps(112500000075)   # → 1125000000.75
"""

import numpy as np


def gps_to_key(gps: float) -> int:
    """Convert GPS time to integer key at 0.01s (10ms) precision."""
    return int(round(float(gps) * 100))


def key_to_gps(key: int) -> float:
    """Convert integer key back to GPS time."""
    return key / 100.0


def gps_array_to_keys(gps_times: np.ndarray) -> np.ndarray:
    """Vectorized GPS to integer key conversion."""
    return np.round(np.asarray(gps_times) * 100).astype(np.int64)
