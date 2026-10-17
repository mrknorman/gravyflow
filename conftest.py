"""
The test session's XLA settings, made before anything imports gravyflow.

The suite's oracles compare bits, so it runs XLA's CPU kernels single-threaded,
as the documented CPU run does: multi-threaded kernels round whitening and Welch
spectra differently from call to call (see gravyflow.src.rng.platform). XLA
reads the flag once, when JAX starts its backend, and importing gravyflow starts
it. pytest loads this file, outside the package, before the package's conftest,
whose import imports gravyflow; so nothing pytest reads before it (the ini's
warning filters) may name a gravyflow class. test_determinism checks the result.
"""

import os
import sys

import pytest

_SINGLE_THREADED_CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"

if _SINGLE_THREADED_CPU_FLAG not in os.environ.get("XLA_FLAGS", "").split():
    if "jax" in sys.modules:
        from jax._src import xla_bridge

        if xla_bridge.backends_are_initialized():
            raise pytest.UsageError(
                "JAX started its backend before the test session could make XLA's CPU kernels "
                f"single-threaded: set XLA_FLAGS={_SINGLE_THREADED_CPU_FLAG}, or keep gravyflow from "
                "being imported before conftest.py (a warning filter naming a gravyflow class does)."
            )
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_SINGLE_THREADED_CPU_FLAG}".strip()
