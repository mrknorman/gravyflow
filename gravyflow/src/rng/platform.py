"""
The platform: what, besides a batch's address, decides its bits here.

The seed tree makes every draw a pure function of its address. The arithmetic
that turns draws into strain is not always: an XLA backend may split a
program's work by the threads free when it runs, or pick its kernels at run
time, so whitening, Welch spectra and projected strain can round differently in
the last bits of float32 from one call to the next, with identical inputs. A
backend is *pure* (`pure`) when its flag in ``XLA_FLAGS`` makes its kernels
give the same bits for the same inputs (`PURE_FLAGS`):

- CPU: ``--xla_cpu_multi_thread_eigen=false`` runs the kernels on the calling
  thread, so a batch is bit-identical between calls, processes and rebuilds
  (its bits still differ from the multi-threaded ones).
- GPU: ``--xla_gpu_deterministic_ops=true`` asks XLA for deterministic kernels
  (not yet verified on a GPU with this version of JAX).
- TPU: no flag is known; whether its kernels are pure is unknown.

The flags are process-wide and read once, when JAX starts its backend
(importing gravyflow does), so the library never sets them: a script or a test
session sets them in the environment before importing gravyflow.

A dataset's CBC stage runs on the host's CPU (`cbc_device`), so its bits
follow the CPU's flag whatever the default backend. Everything that decides the bits on a
device is recorded by `platform_record`, beside data, never in its fingerprint.
"""

import os
import platform as host_platform
import warnings
from typing import Optional

#: The flag in ``XLA_FLAGS`` that makes each backend's kernels pure, by backend.
#: A backend without one (TPU) is of unknown purity.
PURE_FLAGS = {
    "cpu": "--xla_cpu_multi_thread_eigen=false",
    "gpu": "--xla_gpu_deterministic_ops=true",
}


class NoHostBackend(RuntimeError):
    """The process has no CPU backend, which the CBC stage runs on."""


def xla_flags() -> str:
    """This process's ``XLA_FLAGS``, as the environment gives it."""
    return os.environ.get("XLA_FLAGS", "")


def pure(backend: Optional[str] = None) -> Optional[bool]:
    """
    Whether XLA's kernels on ``backend`` (by default JAX's default backend)
    give the same bits for the same inputs here: True when its flag
    (`PURE_FLAGS`) is in ``XLA_FLAGS``, False when it is not, and None for a
    backend whose purity is unknown (TPU, and any backend without a flag).
    """
    if backend is None:
        import jax

        backend = jax.default_backend()
    flag = PURE_FLAGS.get(backend)
    return None if flag is None else flag in xla_flags().split()


def cbc_device():
    """
    The device a dataset's CBC stage runs on: the host's CPU, whatever the
    default backend, so its float64 phase never needs an accelerator's float64.

    Raises:
        NoHostBackend: the process has no CPU backend (``JAX_PLATFORMS``
            without ``cpu``).
    """
    import jax

    try:
        return jax.devices("cpu")[0]
    except RuntimeError as error:
        raise NoHostBackend(
            f"CBC waveforms are generated on the host's CPU, but this process has no CPU backend "
            f"(JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '')!r}): include cpu, e.g. "
            f"JAX_PLATFORMS=tpu,cpu, before starting Python.") from error


def platform_record() -> dict:
    """
    What, besides a batch's address and its dataset's fingerprint (which holds
    the library versions), decides its bits here: the default backend and its
    device kind, ``XLA_FLAGS``, JAX's default matmul precision, the CBC stage's
    device and whether its kernels are pure, and the host's CPU architecture
    (the host stage's libm and SIMD paths). It is recorded beside data (a
    materialised set's manifest, ``seed_metadata``), never hashed into a
    dataset's fingerprint, so a spec keeps its streams on every device. This is
    the one place the record is made: what else decides the bits joins it here.
    """
    import jax

    try:
        cbc = {"device": cbc_device().device_kind, "pure": pure("cpu")}
    except NoHostBackend:
        cbc = {"device": None, "pure": None}
    return {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "xla_flags": xla_flags(),
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "cbc": cbc,
        "architecture": host_platform.machine(),
    }


def warn_if_not_pure() -> None:
    """
    Warn a training or materialising call (the caller of the caller) whose
    default backend is not known to be pure (`pure`): training amplifies
    last-bit differences, so the run is not reproducible from its seed, and a
    small model may not train; a materialised set's bytes are not reproduced by
    another writer. Every such entry point warns through this one check.
    """
    import jax

    backend = jax.default_backend()
    purity = pure(backend)
    if purity:
        return
    if purity is None:
        message = (f"No flag is known to make XLA's {backend} kernels give the same bits from call to "
                   "call, so this run may not be reproducible from its seed.")
    else:
        message = (f"XLA's {backend} kernels are not deterministic here: the whitened strain differs in "
                   "its last bits from call to call, and training amplifies the difference, so this run "
                   "is not reproducible from its seed, and a small model may not train. Set "
                   f"XLA_FLAGS={PURE_FLAGS[backend]} before starting Python.")
    warnings.warn(message, UserWarning, stacklevel=3)
