"""
The versions of the numerics that turn fetched strain into the samples gravyflow serves.

``DATA_VERSIONS`` is their one home. Every data obtainer's ``specification()`` carries
it, so a content fingerprint (``validate.provenance.config_fingerprint``), and every
check that compares one, changes when any of them does: a checkpoint or a validation
written under other numerics is refused. The real-noise segment cache is keyed by it
too, so a stored segment is never served under other numerics.
"""

from types import MappingProxyType

import scipy

DATA_VERSIONS = MappingProxyType({
    # The resampler (``utils/resample.py``): a polyphase Kaiser FIR. Bump it when its
    # design or alignment changes.
    "resampler": "fir-1",
    # SciPy designs the resampler's filter and runs it.
    "scipy": scipy.__version__,
})
