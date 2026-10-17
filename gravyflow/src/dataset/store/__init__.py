"""
The store: raw strain as its source publishes it, in write-once tiles on the absolute
GPS grid, shared safely by every process on one root.

- ``tile``: the format, the key, the commit record and its identity checks.
- ``commit``: the write-once commit, the reaper, ``forget``, and ``Store``.
- ``view``: opening a tile, and raw window views.
- ``budget``: the byte accountant behind the memory cap.
- ``pools``: the background threads that fetch strain, and ``shutdown``, which stops them.
- ``read``: the one read path of real strain: tiles ensured, viewed, resampled, the
  margin cut away; the process's store and memory cap; refusals by name.
"""

from .budget import BudgetRefused, BudgetTimeout, ByteBudget
from .commit import (
    Commit, Fetched, ForgetRaced, Store, StoreConflict, StoreWriteError, TileWaitTimeout, reap,
)
from .pools import shutdown, thread_pool
from .tile import (
    STORE_FORMAT, StoreError, TileIncomplete, TileKey, TileMissing, TileRecord, TileRefused,
    Unheld, covering, seal,
)
from .view import Tile, UnheldSamples, open_tile, read_window, verify

__all__ = [
    "STORE_FORMAT", "BudgetRefused", "BudgetTimeout", "ByteBudget", "Commit", "Fetched",
    "ForgetRaced", "Store", "StoreConflict", "StoreError", "StoreWriteError", "Tile",
    "TileIncomplete", "TileKey", "TileMissing", "TileRecord", "TileRefused", "TileWaitTimeout",
    "Unheld", "UnheldSamples", "covering", "open_tile", "read_window", "reap", "seal", "shutdown",
    "thread_pool", "verify",
]
