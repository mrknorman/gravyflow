"""
The byte accountant behind the store's memory cap.

Tiles are memory-mapped and never held, so what a process holds is bounded by what is
in flight: fetch decode buffers, and the windows and coefficients its callers keep.
The accountant counts the bytes its holders declare against a cap and holds no
contents. The cap is per process: N worker processes use N times it.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator

from .tile import StoreError


class BudgetRefused(StoreError, ValueError):
    """One request is larger than the whole cap: refused when configured, never waited on."""


class BudgetTimeout(StoreError, TimeoutError):
    """A request waited its whole bound without the bytes it needs coming free."""


class ByteBudget:
    """
    Admits holders of declared byte counts while the total in flight stays within
    ``capacity``. A request that fits waits, bounded, for earlier holders to release;
    one never larger than the cap is always admitted when nothing is in flight.
    """

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"A byte budget needs a whole number of bytes ≥ 1, got {capacity!r}.")
        self.capacity = capacity
        self._condition = threading.Condition()
        self._in_flight = 0
        self._peak = 0
        self._waits = 0

    def check(self, nbytes: int) -> None:
        """Refuse, by name, a request that could never be admitted."""
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError(f"A byte request is a whole number ≥ 0, got {nbytes!r}.")
        if nbytes > self.capacity:
            raise BudgetRefused(
                f"A request for {nbytes} bytes exceeds the whole memory cap of "
                f"{self.capacity} bytes; raise the cap or lower the fetch's size."
            )

    @contextlib.contextmanager
    def admit(self, nbytes: int, timeout: float) -> Iterator[None]:
        """Hold ``nbytes`` of the budget for the block, waiting at most ``timeout`` s."""
        self.check(nbytes)
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._in_flight + nbytes > self.capacity:
                self._waits += 1
            while self._in_flight + nbytes > self.capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BudgetTimeout(
                        f"A request for {nbytes} bytes waited {timeout} s while {self._in_flight} "
                        f"of {self.capacity} bytes stayed in flight."
                    )
                self._condition.wait(remaining)
            self._in_flight += nbytes
            self._peak = max(self._peak, self._in_flight)
        try:
            yield
        finally:
            with self._condition:
                self._in_flight -= nbytes
                self._condition.notify_all()

    @property
    def in_flight(self) -> int:
        """The bytes held now."""
        return self._in_flight

    @property
    def peak(self) -> int:
        """The most bytes ever held at once."""
        return self._peak

    @property
    def waits(self) -> int:
        """How many admissions had to wait for bytes to come free."""
        return self._waits
