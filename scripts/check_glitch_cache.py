#!/usr/bin/env python3
"""
Check a transient cache's integrity: its format, its sizes, and each row's strain.

In the cache format (``TRANSIENT_CACHE_FORMAT``) each detector's window in a row is
either held (every sample finite) or not held (every sample NaN: its strain was never
fetched, or could not be had). Neither is an issue. The issues are a window that is
partly non-finite, or held as all zeros: gravyflow writes neither.

Usage: check_glitch_cache.py [PATH]; by default the glitch cache under the gravyflow
cache root (``GRAVYFLOW_CACHE_DIR``, else ``~/.gravyflow/cache``).
"""
import sys
from pathlib import Path

import h5py
import numpy as np

from gravyflow.src.dataset.features.glitch_cache import (
    TransientCache,
    generate_transient_cache_path,
)

CHUNK_ROWS = 100  # rows read at once: windows of tens of MB each


def check_cache(path: Path) -> bool:
    """Print a report on the cache at ``path``; whether it is a healthy cache of the current format."""
    if not path.exists():
        print(f"Cache file not found: {path}")
        return False

    print(f"Checking: {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    try:
        meta = TransientCache(path).get_metadata()  # refuses a file of another format
    except ValueError as error:
        print(f"Not readable: {error}")
        return False

    ifos = meta["ifo_names"]
    rows = meta["num_glitches"]
    print(f"Sample rate {meta['sample_rate_hertz']} Hz; onsource {meta['onsource_duration']} s; "
          f"offsource {meta['offsource_duration']} s; detectors {ifos}; rows {rows}")
    if rows == 0:
        return True

    held = np.zeros(len(ifos), dtype=int)
    not_held = np.zeros(len(ifos), dtype=int)
    partial = np.zeros(len(ifos), dtype=int)
    zeros = np.zeros(len(ifos), dtype=int)
    with h5py.File(path, "r") as f:
        grp = f["glitches"]
        for start in range(0, rows, CHUNK_ROWS):
            # (rows, detectors, samples), onsource and offsource side by side
            windows = np.concatenate(
                [grp["onsource"][start:start + CHUNK_ROWS], grp["offsource"][start:start + CHUNK_ROWS]],
                axis=-1,
            )
            finite = np.isfinite(windows)
            is_held = finite.all(axis=-1)
            is_not_held = np.isnan(windows).all(axis=-1)
            held += is_held.sum(axis=0)
            not_held += is_not_held.sum(axis=0)
            partial += (~is_held & ~is_not_held).sum(axis=0)
            zeros += (is_held & (windows == 0).all(axis=-1)).sum(axis=0)
        gps_times = grp["gps_times"][:]
        labels = grp["labels"][:]

    print("Per detector: held / not held / partly non-finite / held as zeros")
    for i, ifo in enumerate(ifos):
        print(f"  {ifo}: {held[i]} / {not_held[i]} / {partial[i]} / {zeros[i]}")
    print(f"GPS times {gps_times.min():.3f} to {gps_times.max():.3f}; "
          f"{len(np.unique(gps_times))} distinct of {rows} rows")
    for label, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  label {label}: {count} ({count / rows * 100:.1f}%)")

    issues = {
        "partly non-finite windows": int(partial.sum()),
        "windows held as zeros": int(zeros.sum()),
        "rows repeating a GPS time": rows - len(np.unique(gps_times)),
    }
    found = {name: count for name, count in issues.items() if count}
    print(f"ISSUES DETECTED: {found}" if found else "Cache healthy")
    return not found


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else generate_transient_cache_path(prefix="glitch")
    sys.exit(0 if check_cache(target) else 1)
