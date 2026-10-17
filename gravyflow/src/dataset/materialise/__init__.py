"""
Materialised sets: a dataset's batches written once, in one format, and read
back verified (see `sets`). ``GravyflowDataset.materialise`` writes one and
``load_materialised`` serves a dataset's batches from one.
"""

from gravyflow.src.dataset.materialise.backend import LocalBackend, backend_for
from gravyflow.src.dataset.materialise.format import (
    DEFAULT_SHARD_SIZE, FORMAT, ContractMismatch, FingerprintMismatch, ForeignPlatform, IntegrityError,
    MaterialiseConflict, MaterialiseError, MissingShard, NotIndexAddressed, SplitMismatch, UnknownFormat, decode_batch,
    encode_batch, json_bytes, sha256,
)
from gravyflow.src.dataset.materialise.sets import MaterialisedSet, write_set
