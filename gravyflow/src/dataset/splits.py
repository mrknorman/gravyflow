"""
Split membership: which split (train, validate, test) a record or a segment
belongs to.

Membership is a hash of the record's own key against the split fractions,
never a seed, so every split of every run, through every entry point, sees one
partition and no record can be in two splits. A seed only orders a split.

Every entry point (datasets, composed datasets and their pools, NoiseObtainer,
an obtainer called directly, TransientIndex, the validator) splits by
``DEFAULT_SPLIT_FRACTIONS``, and refuses any other fractions: two routes with
different fractions would hash one key into different splits, and train on
what the other evaluates. Configurable fractions return with a split
specification that every route reads.

Split ``ALL`` is every record. It is no set of fractions of its own: it is
served by not filtering the one partition (``in_split``).
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
    The fractions every split is drawn from: ``DEFAULT_SPLIT_FRACTIONS``, given
    as ``groups`` or not, whichever ``group_name`` is served from them. Raises
    ValueError for any other ``groups`` (other keys, order or values), and
    KeyError for a ``group_name`` that is neither ``ALL`` nor a split the
    fractions name.
    """
    fractions = dict(DEFAULT_SPLIT_FRACTIONS)
    if groups is not None and list(dict(groups).items()) != list(fractions.items()):
        raise ValueError(
            f"Split fractions {dict(groups)} are refused: every route splits by "
            f"gf.DEFAULT_SPLIT_FRACTIONS {fractions}, so that no record is in one "
            "split by one route and in another by another. Leave the fractions "
            "unset; configurable fractions return with the split specification."
        )
    if group_name is not None and group_name != ALL and group_name not in fractions:
        raise KeyError(
            f"Group {group_name} not present in groups dictionary {fractions}, and not {ALL!r}."
        )
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


def in_split(split: Optional[str], group_name: str) -> bool:
    """
    Whether a record of split ``split`` is served for ``group_name``: every
    record for ``ALL``, else the records of that split alone.
    """
    return group_name == ALL or split == group_name
