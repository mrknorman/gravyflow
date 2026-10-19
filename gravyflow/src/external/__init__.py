"""
External models: published searches with released weights, run as published.

- ``ExternalCard`` (``card``): what decides a model's results (its input and output
  contracts, its weights pinned by sha256, its adapter's version and environment
  lock, its mode ``AS_PUBLISHED``), which its fingerprint covers; and its provenance,
  the rows its authors published and its training data, which it does not.
- ``ExternalSearch`` and ``SearchSegment`` (``protocol``): the methods an adapter
  implements, and the whole segment it searches.
- ``ExternalRunner`` (``runner``): an adapter run in its own environment as an
  isolated subprocess, under a deadline (``child`` is that process).
- ``release_manifest`` (``licence``): the licence gate every published file passes.

gravyflow imports no adapter, and none of their frameworks: adapters live in their
own packages, published under the entry-point group ``ADAPTER_GROUP``.
"""

from .card import AS_PUBLISHED, ExternalCard, InputContract, OutputContract, PublishedRow, Weights
from .licence import LicenceRefused, release_manifest
from .protocol import ADAPTER_GROUP, ExternalSearch, SearchSegment
from .runner import (
    AdapterEnvironment, AdapterMismatch, AdapterMissing, ExternalDeadlineExpired, ExternalResult,
    ExternalRunFailed, ExternalRunner,
)

__all__ = [
    "ADAPTER_GROUP", "AS_PUBLISHED", "AdapterEnvironment", "AdapterMismatch", "AdapterMissing",
    "ExternalCard", "ExternalDeadlineExpired", "ExternalResult", "ExternalRunFailed", "ExternalRunner",
    "ExternalSearch", "InputContract", "LicenceRefused", "OutputContract", "PublishedRow", "SearchSegment",
    "Weights", "release_manifest",
]
