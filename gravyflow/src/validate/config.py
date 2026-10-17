from dataclasses import dataclass, field
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import gravyflow as gf

from .provenance import StaleCheckpointError


@dataclass
class ValidationConfig:
    """
    Settings of a validation run: what is scored, and how it is summarised.

    The validator draws its own data, from ``seed`` and ``group``; it never
    reuses the training dataset's seed. Each stage (the noise background, the
    injection set) draws from its own scope of the seed tree rooted at ``seed``.
    """
    # The validator's own random root, never the training seed.
    seed: int = gf.Defaults.seed
    # The held-out split it draws real data from.
    group: str = "test"

    num_examples: int = 100_000
    batch_size: int = 512

    # Injections: SNR uniform over this range; efficiency in bins of this width.
    snr_range: Tuple[float, float] = (0.0, 15.0)
    snr_bin_width: float = 1.0
    num_worst_per_bin: int = 5

    # Target false-alarm rates (Hz) for the efficiency curves.
    far_thresholds: List[float] = field(
        default_factory=lambda: np.logspace(-1, -4.5, 50).tolist()
    )
    # Two-sided interval level, and the binomial interval for efficiencies.
    confidence: float = 0.9
    efficiency_interval: str = "clopper-pearson"

    # ROC populations: injections at SNR >= default_roc_min_snr, plus extra
    # pools given as (min_snr, max_snr) or a centre with a width of 1.
    default_roc_min_snr: float = 5.0
    extra_roc_pools: List[Union[Tuple[float, float], float]] = field(default_factory=list)

    # Real GW events, scored for evaluation only. They need network access.
    score_real_events: bool = False

    # The background of real noise of several detectors: time slides, detector j
    # slid by j k steps in slide k = 1 .. num_time_slides (`slides.TimeSlides`);
    # the step defaults to twice a window's footprint. Synthetic noise has no
    # time to slide: its background is independent windows.
    num_time_slides: int = 16
    time_slide_step_seconds: Optional[float] = None
    # Above-threshold windows of real noise starting within this of the one before
    # are one trigger; by default one window, so adjacent windows join.
    cluster_seconds: Optional[float] = None

    def __post_init__(self):
        from .slides import TimeSlides

        TimeSlides(self.num_time_slides, self.time_slide_step_seconds)  # refuses bad settings
        if self.cluster_seconds is not None and not self.cluster_seconds >= 0:
            raise ValueError(f"cluster_seconds must be >= 0, got {self.cluster_seconds}.")
        # Pairs are tuples, however they arrive (JSON gives lists).
        self.snr_range = tuple(self.snr_range)
        self.extra_roc_pools = [tuple(p) if isinstance(p, (list, tuple)) else p for p in self.extra_roc_pools]
        gf.SplitSpec.check_name(self.group)
        if self.group == "train":
            raise ValueError("The validator never draws from the training split.")
        lo, hi = self.snr_range
        if not hi > lo:
            raise ValueError(f"snr_range must be increasing, got {self.snr_range}.")

    @property
    def snr_bin_edges(self) -> np.ndarray:
        lo, hi = self.snr_range
        num_bins = max(1, int(np.ceil((hi - lo) / self.snr_bin_width - 1e-9)))
        return lo + self.snr_bin_width * np.arange(num_bins + 1)


def split_record(training, validation) -> Dict[str, str]:
    """
    The split record of a model trained on ``training`` and selected on
    ``validation``, read from that data and never declared beside it: the
    fingerprint of the ``SplitSpec`` both read, and the split that selects the
    model (``validation``'s ``group``). It is the ``split`` of the weights'
    training record (`with_run_record`, kept on disk as the run's
    ``run.json``), which the validator checks (``check_held_out``).

    ``training`` and ``validation`` are gravyflow datasets, or anything naming
    one's ``split_spec`` and ``group`` (the run recipe's batches do). Data whose
    ``split_spec`` is None reads no real data (synthetic noise): it is
    split-neutral, and takes the spec of the real data beside it; a model of
    split-neutral data alone records the default spec.

    Raises:
        ValueError: data that names no split; training data outside the
            training split, or validation data inside it; or training and
            validation data split by different specs.
    """
    named = {}
    for role, data in (("training", training), ("validation", validation)):
        group = getattr(data, "group", None)
        if not hasattr(data, "split_spec") or group is None:
            raise ValueError(
                f"The {role} data ({type(data).__name__}) names no split: a model records the "
                "split of gravyflow data, which names its split_spec and group."
            )
        named[role] = (data.split_spec, gf.SplitSpec.check_name(group))
    (spec, group), (selection_spec, selection) = named["training"], named["validation"]
    if group != "train":
        raise ValueError(f"The training data is the {group!r} split; a model trains on 'train'.")
    if selection == "train":
        raise ValueError("The validation data is the 'train' split; a model is selected on held-out data.")
    if spec is not None and selection_spec is not None and selection_spec.fingerprint() != spec.fingerprint():
        raise ValueError(
            f"The training data is split by SplitSpec {spec.fingerprint()} and the validation data "
            f"by {selection_spec.fingerprint()}: the validation data may be training data."
        )
    spec = spec or selection_spec or gf.DEFAULT_SPLIT
    return {"fingerprint": spec.fingerprint(), "selection_split": selection}


#: Where a Keras model keeps the training record of its weights: JSON, so that
#: Keras tracks nothing in it and nothing changes it in place.
_RUN_RECORD = "_gravyflow_run_record"


def with_run_record(model, record: Optional[dict]):
    """
    Give ``model`` (a Keras model) the training record of its weights, and
    return it. The record is the ``run.json`` of the run that trained them: its
    ``split`` is their data's split record (`split_record`). It belongs to the
    weights, so gravyflow gives one only to weights it trains or loads with
    theirs: weights built afresh carry none, and a Keras file holds none (on
    disk, the record is the run's ``run.json``).
    """
    setattr(model, _RUN_RECORD, None if record is None else json.dumps(record, sort_keys=True))
    return model


def run_record_of(model) -> Optional[dict]:
    """
    The training record ``model``'s weights carry (`with_run_record`): None for
    weights gravyflow never trained or loaded with a record.
    """
    record = getattr(model, _RUN_RECORD, None)
    return None if record is None else json.loads(record)


def require_same_split(recorded: Optional[dict], live: Dict[str, str], where) -> None:
    """
    Refuse to train further what ``where`` holds (trained weights, or a run's
    directory), whose record's split is ``recorded`` (None when it records
    none), on data whose split record (`split_record`) is ``live``.

    Raises:
        StaleCheckpointError: the record names another split, or none.
    """
    if recorded != live:
        raise StaleCheckpointError(
            f"{where} records the split {recorded or 'none'}, but this data's split is {live}: "
            "continuing would train or select the model on another split. Train new weights, "
            "in a new directory."
        )


def check_held_out(group: str, spec: Optional["gf.SplitSpec"], run_record: Optional[dict]) -> None:
    """
    Refuse to score split ``group`` for a model whose training record
    (``run_record``, the record its weights carry: `run_record_of`) names
    another split spec than ``spec``, names ``group`` as the split that selected
    the model, or names no split. ``spec`` None is split-neutral data
    (synthetic noise), which no record's spec can share time with. A model with
    no record at all, one gravyflow never trained (an external one), is scored
    with a warning: the split it trained on is unknown.
    """
    from gravyflow.src.dataset.splits import require_split_fingerprint

    if run_record is None:
        logging.getLogger(__name__).warning(
            "The model has no gravyflow training record (an external model, say), so the split "
            f"it trained on is unknown, and its '{group}' metrics may include data it trained on."
        )
        return
    split = run_record.get("split") or {}
    if spec is not None:
        require_split_fingerprint(split.get("fingerprint"), spec, "The model's training run")
    elif split.get("fingerprint") is None:
        raise ValueError("The model's training run records no split fingerprint, so the split it trained on is unknown.")
    selection = split.get("selection_split")
    if selection is None:
        raise ValueError(
            "The model's training run records no selection split, so the split that selected "
            "the model is unknown."
        )
    if selection == group:
        raise ValueError(
            f"The model was selected on the '{group}' split, so its metrics there are biased: "
            "score a split that played no part in training or selection ('test')."
        )
