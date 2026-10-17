"""
The background threads that fetch strain, kept in one registry so one call stops them.

Every pool that fetches strain in the background is made by ``thread_pool``, which
names its threads ``gfstore-<name>_<n>`` and registers the pool. ``shutdown()`` stops
every registered pool and waits, within a bound, for every ``gfstore-`` thread to end,
including the thread of a pool whose owner has already been collected: such a pool
leaves the registry with its owner, but its thread finishes the fetch it is running
first. Nothing else in gravyflow starts a long-lived thread that reads strain.
"""

from __future__ import annotations

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import List

#: The prefix of every background strain thread's name.
THREAD_PREFIX = "gfstore-"
#: How long ``shutdown`` waits for the background threads to end, in seconds.
SHUTDOWN_SECONDS = 120.0

_pools: "weakref.WeakSet[ThreadPoolExecutor]" = weakref.WeakSet()
_lock = threading.Lock()


def thread_pool(name: str, max_workers: int = 1) -> ThreadPoolExecutor:
    """A registered pool of ``max_workers`` threads named ``gfstore-<name>_<n>``."""
    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{THREAD_PREFIX}{name}")
    with _lock:
        _pools.add(pool)
    return pool


def background_threads() -> List[threading.Thread]:
    """The live ``gfstore-`` threads."""
    return [thread for thread in threading.enumerate()
            if thread.name.startswith(THREAD_PREFIX) and thread.is_alive()]


def shutdown(timeout_seconds: float = SHUTDOWN_SECONDS) -> int:
    """
    Stop every registered pool, cancelling the work it has not started, and wait for
    every ``gfstore-`` thread to end. A pool stopped here takes no more work: its owner's
    next background fetch raises ``RuntimeError``. Returns the number of pools stopped.

    Raises:
        TimeoutError: naming the threads still running after ``timeout_seconds``.
    """
    with _lock:
        pools = list(_pools)
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + timeout_seconds
    for thread in background_threads():
        thread.join(max(0.0, deadline - time.monotonic()))
    running = [thread.name for thread in background_threads()]
    if running:
        raise TimeoutError(f"Background strain threads still running after {timeout_seconds:g} s: {running}")
    return len(pools)
