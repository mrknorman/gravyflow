"""
Split membership: which split (train, validate, test) a record or a segment
belongs to.

Membership is a hash of the record's own key against the split fractions,
never a seed, so every split of every run, through every entry point, sees one
partition and no record can be in two splits. A seed only orders a split.

Every entry point (datasets, composed datasets and their pools, NoiseObtainer,
an obtainer called directly, TransientIndex, the validator) that is not given
fractions uses ``DEFAULT_SPLIT_FRACTIONS``: two routes with different defaults
would hash one key into different splits, and train on what the other
evaluates.
"""

import hashlib
from types import MappingProxyType
from typing import Dict, Hashable, Mapping, Optional

# The split of every route that is given no fractions.
DEFAULT_SPLIT_FRACTIONS: Mapping[str, float] = MappingProxyType(
    {"train": 0.8, "validate": 0.1, "test": 0.1}
)

# The split that holds every record.
ALL = "all"


def split_fractions(
        groups: Optional[Mapping[str, float]] = None,
        group_name: Optional[str] = None,
    ) -> Dict[str, float]:
    """
    The fractions a split is drawn from: ``groups``, else the default; for
    ``group_name`` "all", everything. Raises KeyError for a ``group_name``
    the fractions do not name.
    """
    if group_name == ALL:
        return {ALL: 1.0}
    fractions = dict(groups) if groups else dict(DEFAULT_SPLIT_FRACTIONS)
    if group_name is not None and group_name not in fractions:
        raise KeyError(f"Group {group_name} not present in groups dictionary {fractions}.")
    return fractions


def split_of(key: Hashable, fractions: Mapping[str, float]) -> str:
    """
    The split ``key`` belongs to: md5 of ``str(key)`` as a uniform number in
    [0, 1), placed among the cumulative, normalised ``fractions``. The last
    split takes any value above a rounded cumulative sum.
    """
    total = sum(fractions.values())
    u = int(hashlib.md5(str(key).encode()).hexdigest(), 16) / 2**128
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += fraction / total
        if u <= cumulative:
            return name
    return list(fractions)[-1]
