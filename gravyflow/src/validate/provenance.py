"""
Provenance: content hashes that tie saved results to the model and settings
that produced them.

A checkpoint found at a path proves nothing about what wrote it. Training
checkpoints and validation results therefore record the sha256 of the model
(its architecture and weights) and of the configuration, and whoever reads them
back compares those hashes with the live ones and refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json

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


def require_match(kind: str, recorded: str | None, live: str, where: Any) -> None:
    """Refuse saved results whose recorded hash is missing or differs from the live one."""
    if recorded != live:
        raise StaleCheckpointError(
            f"{where} was written for another {kind} (recorded sha256 {recorded or 'none'}, "
            f"live {live}). Use a new path, or delete it to start again."
        )
