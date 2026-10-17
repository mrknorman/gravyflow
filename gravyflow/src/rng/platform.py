"""
What decides a batch's bits besides its address: the library versions, the
JAX backend, and XLA's CPU threading.

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
from typing import Dict

#: The XLA flag that runs CPU kernels single-threaded, so results are bit-reproducible.
SINGLE_THREADED_CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"


def xla_flags() -> str:
    """This process's ``XLA_FLAGS``, as the environment gives it."""
    return os.environ.get("XLA_FLAGS", "")


def single_threaded_cpu() -> bool:
    """Whether ``XLA_FLAGS`` runs XLA's CPU kernels single-threaded (bit-reproducible)."""
    return SINGLE_THREADED_CPU_FLAG in xla_flags().split()


def bit_reproducible() -> bool:
    """
    Whether a batch's bytes are a pure function of its address in this
    process: on a CPU backend, only with single-threaded kernels. Other
    backends are not claimed.
    """
    import jax

    return jax.default_backend() == "cpu" and single_threaded_cpu()


def platform_record() -> Dict[str, object]:
    """Everything besides the specification and address that decides a batch's bits."""
    from importlib.metadata import version

    import jax
    import numpy

    return {
        "gravyflow": version("gravyflow"),
        "numpy": numpy.__version__,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "xla_flags": xla_flags(),
        "single_threaded_cpu": single_threaded_cpu(),
    }
