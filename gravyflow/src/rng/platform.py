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
import warnings

#: The XLA flag that runs CPU kernels single-threaded, so results are bit-reproducible.
SINGLE_THREADED_CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"


def xla_flags() -> str:
    """This process's ``XLA_FLAGS``, as the environment gives it."""
    return os.environ.get("XLA_FLAGS", "")


def platform_record() -> dict:
    """
    What, besides a batch's address, decides its bits here: the backend JAX runs
    on and ``XLA_FLAGS``. It is recorded beside data (a materialised set's
    manifest), never hashed into a dataset's fingerprint, so a spec keeps its
    streams on every device. This is the one place the record is made: what
    else decides the bits on a device joins it here.
    """
    import jax

    return {"backend": jax.default_backend(), "xla_flags": xla_flags()}


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


def warn_if_multi_threaded_cpu() -> None:
    """
    Warn a training call (the caller of the caller) that runs on multi-threaded
    CPU kernels (`multi_threaded_cpu`): training amplifies their last-bit
    differences, so the run is not reproducible from its seed, and a small model
    may not train. Every training entry point warns through this one check.
    """
    if multi_threaded_cpu():
        warnings.warn(
            "XLA's CPU kernels are multi-threaded here: the whitened strain differs in its last bits "
            "from call to call, and training amplifies the difference, so this run is not "
            "reproducible from its seed, and a small model may not train. Set "
            f"XLA_FLAGS={SINGLE_THREADED_CPU_FLAG} before starting Python.",
            UserWarning, stacklevel=3,
        )
