"""
Content fingerprints: the sha256 of what a configuration value holds, never of
its identity or type alone.

``canonical`` turns a value into a JSON-ready form by content, and
``config_fingerprint`` hashes it. Whatever must change when a setting that
decides data or results changes is fingerprinted here: a dataset's
specification (``GravyflowDataset.fingerprint``), a checkpoint's run
configuration and a validation's settings (``validate.provenance``).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import PurePath
from typing import Any

import numpy as np


class UnfingerprintableError(TypeError):
    """A configuration value has no content a fingerprint can read."""


def canonical(value: Any) -> Any:
    """
    A JSON-ready canonical form of a configuration value.

    An object with a ``specification()`` method (the data obtainers) is kept by what
    that returns: everything that determines the data it serves. Dataclasses,
    mappings, sequences, enums (by qualified name), numpy scalars and arrays, and
    paths are kept by content. Any other object is refused
    (``UnfingerprintableError``): a fingerprint that read only its type would not
    change when its content did.
    """
    if callable(getattr(value, "specification", None)) and not isinstance(value, type):
        return {"__type__": type(value).__qualname__, **canonical(value.specification())}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__type__": type(value).__qualname__,
                **{f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, dict):
        return {str(canonical(k)): canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"__array__": value.dtype.str, "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise UnfingerprintableError(
        f"Cannot fingerprint a {type(value).__module__}.{type(value).__qualname__} by its content; "
        "give it a specification() method, or make it a dataclass."
    )


def config_fingerprint(*values: Any) -> str:
    """sha256 of the canonical JSON of ``values``."""
    text = json.dumps([canonical(v) for v in values], sort_keys=True, allow_nan=True)
    return hashlib.sha256(text.encode()).hexdigest()
