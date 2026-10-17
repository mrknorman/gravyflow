"""
Provenance: content hashes that tie saved results to the model and settings
that produced them.

A checkpoint found at a path proves nothing about what wrote it. Training
checkpoints and validation results therefore record the sha256 of the model
(its architecture and weights) and of the configuration, and whoever reads them
back compares those hashes with the live ones and refuses a mismatch.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import PurePath
from typing import Any

import numpy as np


class StaleCheckpointError(RuntimeError):
    """Saved results were produced by another model or configuration."""


def model_fingerprint(model) -> str:
    """
    sha256 of a Keras model's layer stack (class and configuration, names
    excluded) and of every weight's dtype, shape and bytes, in order.

    Layer names are left out (Keras numbers unnamed layers per session). A
    model saved and loaded again keeps its fingerprint; retraining, a new output
    activation or a different layer changes it.
    """
    import keras

    digest = hashlib.sha256()
    for layer in model.layers:
        config = {k: v for k, v in layer.get_config().items() if k != "name"}
        description = {"class": type(layer).__name__, "config": config}
        digest.update(json.dumps(description, sort_keys=True, default=str).encode())
    for weight in model.weights:
        array = np.ascontiguousarray(keras.ops.convert_to_numpy(weight))
        digest.update(f"{array.dtype.str}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


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


def require_match(kind: str, recorded: str | None, live: str, where: Any) -> None:
    """Refuse saved results whose recorded hash is missing or differs from the live one."""
    if recorded != live:
        raise StaleCheckpointError(
            f"{where} was written for another {kind} (recorded sha256 {recorded or 'none'}, "
            f"live {live}). Use a new path, or delete it to start again."
        )
