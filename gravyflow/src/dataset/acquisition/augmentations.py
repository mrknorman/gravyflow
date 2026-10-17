"""
Augmentations: per-example transforms of a training batch's raw strain.

An obtainer carries its augmentations (``augmentations=``; none by default). A
dataset applies them (``augment``) to each training batch's raw onsource and
offsource, before injection and whitening, so an injected signal is never
transformed and AddNoise is scaled by the strain itself. Evaluation data (the
validate and test splits, the validator's datasets and a GRID pass) is never
augmented, and an obtainer's own batches never are.

Each example draws each augmentation once, from that augmentation's own stream
(``augment/{name}``), apart from the draws that choose its windows, and the draw
applies alike to its onsource and offsource and to every detector. The batch's
AUGMENTATIONS output flags what each example had: the sum of the ``bit`` of each
augmentation applied to it. START_GPS_TIME still gives the window read.

Example:
    obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        augmentations=[
            gf.SignReversal(probability=0.5),
            gf.TimeReversal(probability=0.5),
            gf.RandomShift(probability=0.5, shift_fraction=0.1),
        ],
    )
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, NamedTuple, Optional, Tuple

import numpy as np
from keras import ops

from gravyflow.src.dataset.acquisition.base import DataLabel, SamplingMode, label_kinds

logger = logging.getLogger(__name__)


@dataclass
class Augmentation(ABC):
    """
    One augmentation: applied to each example of a training batch with
    ``probability`` (see ``augment``). ``bit`` is its flag in AUGMENTATIONS.
    """
    probability: float = 0.5

    bit: ClassVar[int]

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"{type(self).__name__}'s probability must be in [0, 1], got {self.probability}."
            )

    @abstractmethod
    def apply(self, onsource: np.ndarray, offsource: np.ndarray, chosen: np.ndarray,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        New ``onsource`` and ``offsource`` (B, I, T), the examples ``chosen`` (B,)
        transformed alike in both, and the examples it applied to (B,), which
        AUGMENTATIONS flags: those chosen, less any it has nothing to apply to
        (AddNoise on an example with no variance). Any parameter is drawn from
        ``rng`` for every example, chosen or not, so no example's draw depends on
        another's.
        """


@dataclass
class SignReversal(Augmentation):
    """
    Negate the strain, x -> -x. A gravitational wave with both polarisations
    negated is the same source with its polarisation angle turned by 90 degrees,
    so it is seen negated in every detector; Gaussian noise is unchanged in law.
    """
    bit: ClassVar[int] = 1

    def apply(self, onsource, offsource, chosen, rng):
        return _where(chosen, -onsource, onsource), _where(chosen, -offsource, offsource), chosen


@dataclass
class TimeReversal(Augmentation):
    """
    Reverse time, x[t] -> x[T - 1 - t]. Stationary Gaussian noise is unchanged
    in law; a chirp is not, so real events refuse it.
    """
    bit: ClassVar[int] = 2

    def apply(self, onsource, offsource, chosen, rng):
        return (_where(chosen, onsource[..., ::-1], onsource), _where(chosen, offsource[..., ::-1], offsource),
                chosen)


@dataclass
class RandomShift(Augmentation):
    """
    Read each chosen training example off centre: its windows move by a whole number
    of samples, up to ``shift_fraction`` of the onsource (with its crop padding)
    either way, so a glitch is no longer at the window's centre. The strain is read
    where the windows lie, never rolled, so START_GPS_TIME stays the first returned
    sample's time; TRANSIENT_GPS_TIME stays the glitch's.

    The transient obtainer moves the read (``draw``), so the request's windows,
    moved by the most, must lie in every record's footprint. Glitches only: real
    events refuse it, and so do noise windows, which are drawn uniformly in time
    already.
    """
    shift_fraction: float = 0.25

    bit: ClassVar[int] = 4

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.shift_fraction <= 0.5:
            raise ValueError(
                f"RandomShift's shift_fraction must be in [0, 0.5], got {self.shift_fraction}: "
                "the transient would leave the onsource."
            )

    def draw(self, streams, num_examples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The draw ``augment`` makes for a batch at ``streams``: which of its examples are
        chosen (B,), and each one's shift as a signed fraction of the most (B,), 0 where
        not chosen, from the augmentation's own stream.
        """
        rng = streams.stream(f"augment/{type(self).__name__}")
        chosen = rng.random(num_examples) < self.probability
        return chosen, np.where(chosen, rng.uniform(-1.0, 1.0, num_examples), 0.0)

    def apply(self, onsource, offsource, chosen, rng):
        # The windows were read where the shift put them (``draw``); nothing moves here.
        return onsource, offsource, chosen


@dataclass
class AddNoise(Augmentation):
    """
    Add Gaussian noise of ``amplitude`` times the example's own offsource
    standard deviation, per detector, to both windows. One scale for both keeps
    the onsource's noise level the offsource's, whose PSD whitens it. An example
    with no variance gets none, so it is not flagged; one whose offsource is not
    finite has no level to scale by and is refused.
    """
    amplitude: float = 0.1

    bit: ClassVar[int] = 8

    def __post_init__(self):
        super().__post_init__()
        if not (np.isfinite(self.amplitude) and self.amplitude >= 0.0):
            raise ValueError(f"AddNoise's amplitude must be finite and >= 0, got {self.amplitude}.")

    def apply(self, onsource, offsource, chosen, rng):
        # In float64: the square of strain (~1e-42) is below float32's normal range.
        std = np.std(offsource, axis=-1, dtype=np.float64, keepdims=True)
        unscalable = chosen & ~np.all(np.isfinite(std), axis=(1, 2))
        if np.any(unscalable):
            raise ValueError(
                f"AddNoise refuses example {', '.join(map(str, np.flatnonzero(unscalable)))} of the batch: "
                "its offsource is not finite, so it has no noise level to scale by."
            )
        scale = np.where(chosen[:, None, None], self.amplitude * std, 0.0)
        noised = ((x + rng.standard_normal(x.shape) * scale).astype(x.dtype) for x in (onsource, offsource))
        return (*noised, np.any(scale > 0.0, axis=(1, 2)))


def _where(chosen: np.ndarray, transformed: np.ndarray, original: np.ndarray) -> np.ndarray:
    return np.where(chosen[:, None, None], transformed, original)


def check_augmentations(augmentations: Optional[Iterable], data_labels=()) -> Tuple[Augmentation, ...]:
    """
    ``augmentations`` (None is none) as a tuple, checked for an obtainer of
    ``data_labels``: each an Augmentation, each kind at most once, since each
    draws from the stream its kind names. Refused: TimeReversal and RandomShift
    where the labels ask for real events, which keep the time order and
    placement they were observed with; and RandomShift on noise, whose windows
    are drawn uniformly in time already. What the labels ask for is read by
    their kinds (``label_kinds``), which refuses anything but a label.
    """
    spec = tuple(augmentations or ())
    kinds = set()
    for augmentation in spec:
        if not isinstance(augmentation, Augmentation):
            raise ValueError(
                f"{augmentation!r} is not an augmentation: give gf.SignReversal, gf.TimeReversal, "
                "gf.RandomShift or gf.AddNoise instances."
            )
        if type(augmentation) in kinds:
            raise ValueError(
                f"{type(augmentation).__name__} is given more than once; each kind draws from its own stream."
            )
        kinds.add(type(augmentation))

    served = label_kinds(data_labels)
    if DataLabel.EVENTS in served:
        moved = [type(a).__name__ for a in spec if isinstance(a, (TimeReversal, RandomShift))]
        if moved:
            raise ValueError(
                f"{' and '.join(moved)} cannot be applied to real events: an event keeps the time "
                "order and placement it was observed with. SignReversal and AddNoise may be."
            )
    if DataLabel.NOISE in served and RandomShift in kinds:
        raise ValueError("RandomShift is refused on noise: noise windows are drawn uniformly in time already.")
    return spec


class Augmented(NamedTuple):
    """A batch's strain after ``augment``, and its examples' AUGMENTATIONS flags."""
    onsource: Any
    offsource: Any
    #: (B,) float32: the sum of the bits of the augmentations each example had.
    applied: np.ndarray


#: The kinds of evaluation data already logged as never augmented, once each per process.
_LOGGED: set = set()


def augment(strain, streams, spec, split: str, labels=(), sampling_mode: Optional[SamplingMode] = None
            ) -> Augmented:
    """
    A batch's raw ``strain``, (onsource, offsource) each (B, I, T), with the
    augmentations ``spec`` applied when it is training data drawn at random:
    ``split`` 'train' and no GRID pass.

    Evaluation data is returned as it is, all flags 0, and logged once. Otherwise
    each example draws each augmentation once, in the spec's order, from the
    ``augment/{name}`` stream of ``streams`` (the batch's address): one uniform
    draw per example chooses it, then its parameters are drawn. The choice
    applies alike to the example's onsource and offsource.

    ``spec`` is checked against the obtainer's ``labels`` here too
    (``check_augmentations``), so a spec changed after configuration is still
    refused. The inputs are never written.
    """
    onsource, offsource = strain
    spec = check_augmentations(spec, labels)
    applied = np.zeros(int(np.shape(onsource)[0]), np.int32)
    if not spec:
        return Augmented(onsource, offsource, applied.astype(np.float32))
    if split != "train" or sampling_mode is SamplingMode.GRID:
        data = "A GRID pass" if split == "train" else f"The '{split}' split"
        if data not in _LOGGED:
            _LOGGED.add(data)
            logger.info(f"{data} is evaluation data, so it is never augmented: "
                        f"{', '.join(type(a).__name__ for a in spec)} not applied.")
        return Augmented(onsource, offsource, applied.astype(np.float32))

    onsource, offsource = np.asarray(onsource), np.asarray(offsource)
    for augmentation in spec:
        rng = streams.stream(f"augment/{type(augmentation).__name__}")
        chosen = rng.random(applied.size) < augmentation.probability
        onsource, offsource, changed = augmentation.apply(onsource, offsource, chosen, rng)
        applied |= np.where(changed, augmentation.bit, 0).astype(np.int32)
    return Augmented(ops.convert_to_tensor(onsource), ops.convert_to_tensor(offsource),
                     applied.astype(np.float32))
