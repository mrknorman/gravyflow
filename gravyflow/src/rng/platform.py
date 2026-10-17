"""
XLA's CPU threading, which decides a batch's bits besides its address.

The seed tree makes every draw a pure function of its address. The arithmetic
that turns draws into strain is not always: XLA's multi-threaded CPU kernels
split a program's work by the threads free when it runs, so whitening, Welch
spectra and projected strain can round differently in the last bits of
float32 from one call to the next, with identical inputs. With
``--xla_cpu_multi_thread_eigen=false`` in ``XLA_FLAGS`` the kernels run on the
calling thread and a batch is bit-identical between calls, processes and
rebuilds (its bits still differ from the multi-threaded ones).

The flag is process-wide and read once, when JAX starts its backend (importing
gravyflow does), so the library never sets it: a script or a test session sets
it in the environment before importing gravyflow.
"""

import os

#: The XLA flag that runs CPU kernels single-threaded, so results are bit-reproducible.
SINGLE_THREADED_CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"


def xla_flags() -> str:
    """This process's ``XLA_FLAGS``, as the environment gives it."""
    return os.environ.get("XLA_FLAGS", "")


def single_threaded_cpu() -> bool:
    """Whether ``XLA_FLAGS`` runs XLA's CPU kernels single-threaded (bit-reproducible)."""
    return SINGLE_THREADED_CPU_FLAG in xla_flags().split()


def multi_threaded_cpu() -> bool:
    """
    Whether XLA runs here on a CPU with multi-threaded kernels, so results vary
    in their last bits from call to call (other backends are not judged).
    """
    import jax

    return jax.default_backend() == "cpu" and not single_threaded_cpu()
