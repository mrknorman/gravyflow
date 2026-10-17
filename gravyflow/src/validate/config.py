from dataclasses import dataclass, field
import logging
from typing import List, Optional, Tuple, Union

import numpy as np

import gravyflow as gf


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

    def __post_init__(self):
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


def dataset_split_spec(dataset_args: dict) -> "gf.SplitSpec":
    """The split of the real data ``dataset_args`` read: its IFO obtainer's, else the default."""
    noise = dataset_args.get("noise_obtainer")
    obtainer = getattr(noise, "ifo_data_obtainer", None)
    return getattr(obtainer, "split_spec", None) or gf.DEFAULT_SPLIT


def check_held_out(group: str, spec: "gf.SplitSpec", run_record: Optional[dict]) -> None:
    """
    Refuse to score split ``group`` for a model whose training record
    (``run_record``) names another split spec than ``spec``, or names ``group``
    as the split that selected the model. With no record, warn: the split the
    model trained on is unknown.
    """
    from gravyflow.src.dataset.splits import require_split_fingerprint

    if run_record is None:
        logging.getLogger(__name__).warning(
            "No training record: the split this model trained on is unknown, so its "
            f"'{group}' metrics may include data it trained on."
        )
        return
    split = run_record.get("split") or {}
    require_split_fingerprint(split.get("fingerprint"), spec, "The model's training run")
    if split.get("selection_split") == group:
        raise ValueError(
            f"The model was selected on the '{group}' split, so its metrics there are biased: "
            "score a split that played no part in training or selection ('test')."
        )
