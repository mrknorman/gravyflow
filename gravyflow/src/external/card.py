"""
An external model's card: everything that decides what its search returns, and the
provenance its results carry: the numbers its authors published, and its training data.

A card names the adapter that runs the model (``protocol.ExternalSearch``, by its
entry-point name), the adapter's version and the sha256 of the environment lock it
runs in; the input the model takes (``InputContract``) and the statistic it returns
(``OutputContract``); its weights, each a ``RemoteFile`` pinned by sha256 with its
licence (``Weights``); the rows its authors published (``PublishedRow``); the data it
was trained on, as declared; and the mode its results are in.

``ExternalCard.fingerprint`` is the model's identity: the sha256, by content
(``utils.fingerprint``), of exactly what decides a run's outputs: the adapter's
entry-point name and version, the environment lock, the two contracts, each weights
file's name (the adapter finds it by name), sha256, format and framework, and the
mode. Another weights file, adapter release, environment or contract is another
fingerprint. The published rows and the declared training data are provenance, not
identity: a corrected citation or a new published row leaves the model the same
model, and every result carries them beside its card's fingerprint. The model's name
and its weights' URLs and licences decide nothing a run returns, so they are not in
the fingerprint either.

The one mode is ``AS_PUBLISHED``: the model run as its authors released it, its
statistic reported as published, with its signal column and direction declared. It
is never rescored as gravyflow scores its own models (the logit of column 1): the
validator (``Validator.validate`` with an ``ExternalRunner``) streams it on raw
strain at its own stride and ranks its windows by the published statistic, in the
card's direction. A variant that changes the published model (a softmax removed)
is another mode, with its own name.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from gravyflow.src.dataset.store.files import RemoteFile
from gravyflow.src.utils.fingerprint import config_fingerprint

#: The mode of a model run as its authors released it.
AS_PUBLISHED = "as-published"

#: Every mode an external model's results may be in.
MODES = (AS_PUBLISHED,)

#: Which way a statistic ranks: larger values, or smaller, mean more likely signal.
HIGHER, LOWER = "higher", "lower"

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _text(owner: str, name: str, value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}'s {name} must be a non-empty string, got {value!r}.")


def _number(instance, field: str, owner: str, above: Optional[float] = None,
            at_least: Optional[float] = None) -> None:
    """Check that ``instance.<field>`` is a finite number in its bounds, and keep it as a
    float, so 2048 and 2048.0 are one setting with one fingerprint."""
    value = getattr(instance, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or (above is not None and not value > above) or (at_least is not None and not value >= at_least):
        bound = f" above {above}" if above is not None else f" not below {at_least}" if at_least is not None else ""
        raise ValueError(f"{owner}'s {field} must be a finite number{bound}, got {value!r}.")
    object.__setattr__(instance, field, float(value))


@dataclass(frozen=True)
class InputContract:
    """
    The input an external model takes, as its published code prepares it.

    Args:
        sample_rate_hertz: The rate it reads strain at.
        detectors: The detectors it reads, in the order it stacks them.
        window_seconds: The length of one window it scores.
        stride_seconds: The step between the starts of consecutive windows.
        whitening: How it whitens: the method, its PSD estimate and their lengths.
        context_seconds: Strain it needs beyond the windows it scores (to estimate a
            PSD, or for a filter's edges), in seconds.
        band_pass_hertz: The pass band it filters to, (low, high), or None.
        normalisation: How it scales a whitened window, or ``"none"``.
        segment_seconds: The shortest segment, centred on a window, in which its search
            scores that window (its slicing and edges included), or None: the window
            with its context either side. The validator scores each injection on a
            segment this long, plus the found test's reach either side.
    """

    sample_rate_hertz: float
    detectors: Tuple[str, ...]
    window_seconds: float
    stride_seconds: float
    whitening: str
    context_seconds: float = 0.0
    band_pass_hertz: Optional[Tuple[float, float]] = None
    normalisation: str = "none"
    segment_seconds: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "detectors", tuple(self.detectors))
        for field in ("sample_rate_hertz", "window_seconds", "stride_seconds"):
            _number(self, field, "An input contract", above=0)
        _number(self, "context_seconds", "An input contract", at_least=0)
        if self.segment_seconds is not None:
            _number(self, "segment_seconds", "An input contract", above=0)
        if not self.detectors or len(set(self.detectors)) != len(self.detectors) \
                or not all(isinstance(d, str) and d for d in self.detectors):
            raise ValueError(f"An input contract's detectors are distinct names, got {self.detectors}.")
        _text("An input contract", "whitening", self.whitening)
        _text("An input contract", "normalisation", self.normalisation)
        if self.band_pass_hertz is not None:
            band = tuple(float(f) for f in self.band_pass_hertz)
            if len(band) != 2 or not 0 <= band[0] < band[1] or not math.isfinite(band[1]):
                raise ValueError(f"An input contract's pass band is (low, high) in Hz, got {self.band_pass_hertz}.")
            object.__setattr__(self, "band_pass_hertz", band)


@dataclass(frozen=True)
class OutputContract:
    """
    The statistic an external model's search returns, as published.

    Args:
        statistic: What the statistic is (``"softmax probability of the signal class"``).
        signal_column: The column of the model's output that is the signal class.
        direction: ``HIGHER`` when larger values mean more likely signal, ``LOWER``
            when smaller ones do.
        threshold: The value a window must pass to become a trigger.
        cluster_seconds: The window its triggers are clustered over (0 for none).
        time_offset_seconds: What it adds to a window's start to give a trigger's time.
    """

    statistic: str
    signal_column: int
    direction: str
    threshold: float
    cluster_seconds: float
    time_offset_seconds: float = 0.0

    def __post_init__(self):
        _text("An output contract", "statistic", self.statistic)
        if isinstance(self.signal_column, bool) or not isinstance(self.signal_column, int) or self.signal_column < 0:
            raise ValueError(f"An output contract's signal column is a column index, got {self.signal_column!r}.")
        if self.direction not in (HIGHER, LOWER):
            raise ValueError(f"An output contract's direction is {HIGHER!r} or {LOWER!r}, got {self.direction!r}.")
        _number(self, "threshold", "An output contract")
        _number(self, "cluster_seconds", "An output contract", at_least=0)
        _number(self, "time_offset_seconds", "An output contract")


@dataclass(frozen=True)
class Weights:
    """
    One file of an external model's released weights.

    Args:
        file: Where its authors publish it (a URL pinned to one commit), its size,
            sha256, licence and whether gravyflow may redistribute it. It is fetched
            only through gravyflow's one fetch (``gf.fetch``) and checked there.
        format: How it is stored (``"torch-state-dict"``, ``"torchscript"``, ``"keras-h5"``).
        framework: The framework release that reads it (``"torch==1.12.1"``).
    """

    file: RemoteFile
    format: str
    framework: str

    def __post_init__(self):
        if not isinstance(self.file, RemoteFile):
            raise ValueError(f"A weights file is a RemoteFile, got {type(self.file).__name__}.")
        _text(f"Weights {self.file.name}", "format", self.format)
        _text(f"Weights {self.file.name}", "framework", self.framework)


@dataclass(frozen=True)
class PublishedRow:
    """
    One number an external model's authors published, which a reproduction must
    match before its results count.

    Args:
        citation: Where it is published (``"gabbard2018, Fig. 3"``).
        dataset: The data it was measured on (``"O3a, H1 and L1"``).
        quantity: What it is (``"sensitive distance"``).
        value: Its value, in ``unit``.
        unit: Its unit (``"Mpc"``).
        far_per_month: The false-alarm rate it is quoted at, per month, if any.
        reading_error: For a value read off a figure, the reading's uncertainty, in
            ``unit``; None for a value printed in a table.
    """

    citation: str
    dataset: str
    quantity: str
    value: float
    unit: str
    far_per_month: Optional[float] = None
    reading_error: Optional[float] = None

    def __post_init__(self):
        for name in ("citation", "dataset", "quantity", "unit"):
            _text("A published row", name, getattr(self, name))
        _number(self, "value", "A published row")
        if self.far_per_month is not None:
            _number(self, "far_per_month", "A published row", above=0)
        if self.reading_error is not None:
            _number(self, "reading_error", "A published row", at_least=0)


@dataclass(frozen=True)
class ExternalCard:
    """
    An external model, as gravyflow runs it.

    Args:
        name: The model's name (``"gabbard-2018"``).
        adapter: The entry-point name its adapter is published under, in
            ``protocol.ADAPTER_GROUP``.
        adapter_version: The version of the distribution that publishes the adapter;
            the runner refuses an environment holding another.
        environment_lock: The sha256 of the lock file of the environment the adapter
            runs in; the runner refuses another lock.
        input: The input the model takes.
        output: The statistic its search returns.
        weights: Its weights files, by distinct names.
        training_data: The data it was trained on, as its authors declare it (so a
            result can name its overlap with the data it is tested on); provenance,
            not in the fingerprint.
        published: The rows it must reproduce; provenance, not in the fingerprint.
        mode: The mode its results are in, one of ``MODES``.
    """

    name: str
    adapter: str
    adapter_version: str
    environment_lock: str
    input: InputContract
    output: OutputContract
    weights: Tuple[Weights, ...]
    training_data: str
    published: Tuple[PublishedRow, ...] = ()
    mode: str = AS_PUBLISHED

    def __post_init__(self):
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "published", tuple(self.published))
        for name in ("name", "adapter", "adapter_version", "training_data"):
            _text("A card", name, getattr(self, name))
        if not isinstance(self.environment_lock, str) or not _SHA256.fullmatch(self.environment_lock):
            raise ValueError(
                f"{self.name}: its environment lock is the lock file's sha256, 64 lowercase hex digits, "
                f"got {self.environment_lock!r}.")
        if not isinstance(self.input, InputContract) or not isinstance(self.output, OutputContract):
            raise ValueError(f"{self.name}: its input and output are an InputContract and an OutputContract.")
        if not all(isinstance(w, Weights) for w in self.weights):
            raise ValueError(f"{self.name}: each of its weights is a Weights.")
        names = [w.file.name for w in self.weights]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: its weights files have distinct names, got {names}.")
        if not all(isinstance(r, PublishedRow) for r in self.published):
            raise ValueError(f"{self.name}: each of its published rows is a PublishedRow.")
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: its mode is one of {MODES}, got {self.mode!r}.")

    @property
    def fingerprint(self) -> str:
        """The sha256, by content, of what decides a run's outputs; never the model's
        name, its weights' URLs or licences, its published rows or its training data."""
        return config_fingerprint({
            "adapter": self.adapter, "adapter_version": self.adapter_version,
            "environment_lock": self.environment_lock, "input": self.input, "output": self.output,
            "weights": [{"name": w.file.name, "sha256": w.file.sha256, "format": w.format,
                         "framework": w.framework} for w in self.weights],
            "mode": self.mode,
        })


__all__ = [
    "AS_PUBLISHED", "ExternalCard", "HIGHER", "InputContract", "LOWER", "MODES", "OutputContract",
    "PublishedRow", "Weights",
]
