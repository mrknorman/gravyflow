"""
The versions of the numerics that turn fetched strain into the samples gravyflow serves.

``DATA_VERSIONS`` is their one home. Every data obtainer's ``specification()`` carries
it, so a content fingerprint (``validate.provenance.config_fingerprint``), and every
check that compares one, changes when any of them does: a checkpoint or a validation
written under other numerics is refused. Real noise is read from the store, which
keeps raw strain, so a change of the resampler cuts no tile; the store's format names
its root, so a new format never reads an old one.
"""

from types import MappingProxyType

import scipy

from gravyflow.src.dataset.store.tile import STORE_FORMAT

DATA_VERSIONS = MappingProxyType({
    # The resampler (``utils/resample.py``): a polyphase Kaiser FIR. Bump it when its
    # design or alignment changes.
    "resampler": "fir-1",
    # SciPy designs the resampler's filter and runs it.
    "scipy": scipy.__version__,
    # The store's tiles and records (``dataset/store/tile.py``).
    "store": STORE_FORMAT,
})
