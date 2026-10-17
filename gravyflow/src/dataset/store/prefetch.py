"""
Reading ahead: the batches after the one asked for, read on background threads.

A ``Prefetcher`` belongs to one stream, a dataset's split, and is never shared, so no
split ever waits on, or is served, another's reads. It computes the same pure function
of an address that the caller would, only earlier: a batch is equal at depth 0 or any
depth, on one thread or eight, and a read that fails in the background fails again,
by the same name, when its batch is asked for. Its threads are ``gfstore-prefetch-*``
(``store.pools``), so ``gf.store.shutdown()`` stops them, as it does at exit; every
wait for a read ahead is bounded and refuses by name.
"""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Callable, Dict, Generic, Hashable, Optional, Sequence, TypeVar

from .pools import shutdown, thread_pool

#: How long a batch waits for its read ahead, in seconds.
PREFETCH_WAIT_SECONDS = 900.0

T = TypeVar("T")


class PrefetchTimeout(TimeoutError):
    """A batch read ahead did not finish within its wait."""


class Prefetcher(Generic[T]):
    """
    ``compute(address)``, read ahead: ``get(address, ahead)`` returns it, from the read
    started for it when there was one, and starts the reads of the first ``depth`` of
    ``ahead``. Reads started for addresses no longer ahead are cancelled, so at most
    ``depth`` run or wait at once. Once ``gf.store.shutdown()`` has stopped its threads
    it reads nothing ahead, and computes each address when it is asked for.
    """

    def __init__(self, compute: Callable[[Hashable], T], *, name: str, depth: int, workers: int,
                 wait_seconds: float = PREFETCH_WAIT_SECONDS):
        self._compute, self.name, self.depth, self.workers = compute, name, int(depth), int(workers)
        self.wait_seconds = float(wait_seconds)
        self._futures: Dict[Hashable, Future] = {}
        self._pool = None
        self._stopped = False
        self._lock = threading.Lock()

    def get(self, address: Hashable, ahead: Sequence[Hashable] = ()) -> T:
        with self._lock:
            future = self._futures.pop(address, None)
            wanted = list(ahead)[:self.depth]
            for stale in set(self._futures) - set(wanted):
                self._futures.pop(stale).cancel()
            for later in wanted:
                if later not in self._futures and later != address:
                    started = self._submit(later)
                    if started is None:
                        break
                    self._futures[later] = started
        if future is None or future.cancelled():
            return self._compute(address)
        try:
            return future.result(timeout=self.wait_seconds)
        except FutureTimeout:
            raise PrefetchTimeout(
                f"The read ahead of {self.name} batch {address} did not finish in {self.wait_seconds:g} s."
            ) from None

    def _submit(self, address: Hashable) -> Optional[Future]:
        """The read ahead of ``address``; None once ``store.shutdown`` stopped the threads."""
        if self._stopped:
            return None
        if self._pool is None:
            self._pool = thread_pool(f"prefetch-{self.name}", self.workers)
        try:
            return self._pool.submit(self._compute, address)
        except RuntimeError:  # the pool takes no more work: shutdown() stopped it
            self._stopped, self._pool = True, None
            return None

    def close(self) -> None:
        """Cancel every read ahead and stop the threads."""
        with self._lock:
            for future in self._futures.values():
                future.cancel()
            self._futures.clear()
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def __getstate__(self):
        """A copy reads ahead on threads of its own: reads in flight stay behind."""
        state = dict(self.__dict__)
        state.update(_futures={}, _pool=None, _stopped=False, _lock=None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


_exit_hook: Optional[Callable] = None


def _register_exit() -> None:
    """Stop every background strain thread when the interpreter exits."""
    global _exit_hook
    if _exit_hook is None:
        _exit_hook = atexit.register(shutdown)


_register_exit()
