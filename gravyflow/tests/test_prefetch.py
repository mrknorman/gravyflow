"""
Reading ahead (``store.prefetch``): a ``Prefetcher`` computes the same pure function of
an address the caller would, only earlier, on ``gfstore-prefetch-*`` threads. Its result
is equal at any depth and on any number of threads; a read ahead that fails fails again,
by the same name, when its address is asked for; reads no longer ahead are cancelled;
every wait is bounded; and once ``store.shutdown()`` stops its threads it reads each
address when asked, starting none.
"""

import copy
import threading
import time

import numpy as np
import pytest

from gravyflow.src.dataset.store import pools
from gravyflow.src.dataset.store.prefetch import Prefetcher, PrefetchTimeout


def pure(address):
    """A pure function of the address: its own seeded draw."""
    return np.random.default_rng(address).normal(size=16)


class Recorder:
    """``pure``, logging the thread each address was computed on."""

    def __init__(self, compute=pure, fail=()):
        self.compute, self.fail = compute, set(fail)
        self.threads = {}
        self._lock = threading.Lock()

    def __call__(self, address):
        with self._lock:
            self.threads.setdefault(address, []).append(threading.current_thread().name)
        if address in self.fail:
            raise LookupError(f"planted failure at {address}")
        return self.compute(address)


def serve(prefetcher, addresses, depth):
    """Each address in turn, reading the next ``depth`` ahead, as a batcher does."""
    return [prefetcher.get(a, addresses[i + 1:i + 1 + depth]) for i, a in enumerate(addresses)]


@pytest.fixture(autouse=True)
def stopped_after():
    yield
    pools.shutdown()


@pytest.mark.parametrize("depth,workers", [(0, 1), (2, 1), (2, 8), (5, 3)])
def test_results_equal_at_every_depth_and_thread_count(depth, workers):
    addresses = list(range(40))
    recorder = Recorder()
    prefetcher = Prefetcher(recorder, name="train", depth=depth, workers=workers)
    served = serve(prefetcher, addresses, depth)
    for address, value in zip(addresses, served):
        np.testing.assert_array_equal(value, pure(address))
    ahead = sum(name.startswith("gfstore-prefetch-train") for names in recorder.threads.values() for name in names)
    if depth:
        assert ahead > 0, "nothing was read ahead"
    else:
        assert ahead == 0 and not pools.background_threads()
    # Each address computed once: on the thread that read it ahead, or by the caller.
    assert all(len(names) == 1 for names in recorder.threads.values())


def test_a_failure_read_ahead_fails_again_by_name_when_asked():
    recorder = Recorder(fail={3})
    prefetcher = Prefetcher(recorder, name="train", depth=4, workers=2)
    prefetcher.get(0, [1, 2, 3, 4])
    for address in (1, 2):
        np.testing.assert_array_equal(prefetcher.get(address, [address + 1]), pure(address))
    with pytest.raises(LookupError, match="planted failure at 3"):
        prefetcher.get(3, [4])
    # The batch after it is unaffected.
    np.testing.assert_array_equal(prefetcher.get(4, []), pure(4))


def test_reads_no_longer_ahead_are_cancelled():
    release = threading.Event()

    def slow(address):
        if address in (0, 1, 2):
            release.wait(5.0)
        return pure(address)

    prefetcher = Prefetcher(slow, name="train", depth=3, workers=1)
    prefetcher.get(100, [0, 1, 2])
    waiting = dict(prefetcher._futures)
    assert sorted(waiting) == [0, 1, 2]
    # A jump elsewhere: the reads ahead not yet started are no longer wanted.
    np.testing.assert_array_equal(prefetcher.get(10, [11]), pure(10))
    release.set()
    assert sorted(prefetcher._futures) == [11]
    assert waiting[1].cancelled() and waiting[2].cancelled()


def test_a_wait_for_a_read_ahead_is_bounded():
    release = threading.Event()

    def stuck(address):
        if address == 0:
            release.wait(10.0)
        return pure(address)

    prefetcher = Prefetcher(stuck, name="train", depth=1, workers=1, wait_seconds=0.2)
    prefetcher.get(50, [0])  # 0 is read ahead, and sticks
    start = time.monotonic()
    with pytest.raises(PrefetchTimeout, match=r"train batch 0 did not finish in 0.2 s"):
        prefetcher.get(0, [])
    assert time.monotonic() - start < 5.0
    release.set()


def test_after_shutdown_each_address_is_read_when_asked():
    recorder = Recorder()
    prefetcher = Prefetcher(recorder, name="train", depth=2, workers=2)
    serve(prefetcher, list(range(4)), 2)
    assert pools.background_threads()
    pools.shutdown()
    assert not pools.background_threads()
    served = serve(prefetcher, list(range(4, 12)), 2)
    for address, value in zip(range(4, 12), served):
        np.testing.assert_array_equal(value, pure(address))
    assert not pools.background_threads(), "a stopped prefetcher started threads again"


def test_a_copy_reads_ahead_on_its_own_threads():
    prefetcher = Prefetcher(pure, name="train", depth=2, workers=1)
    serve(prefetcher, list(range(3)), 2)
    clone = copy.deepcopy(prefetcher)
    assert clone._futures == {} and clone._pool is None
    np.testing.assert_array_equal(clone.get(7, [8]), pure(7))
