"""
The registry of stream consumers: every name a draw may be made under.

A consumer name says what a draw is for. The registry makes the names unique
by construction: a name that matches no entry is refused, so a typo or a new
draw cannot silently share another consumer's stream. Generator and pool names
are unique within their dataset, which keeps the names built from them unique.
"""

import re

#: One path segment: a name with no slash.
_SEGMENT = r"[A-Za-z0-9_.:+\-]+"

#: Consumers, as regular expressions over the name after its scopes.
_LEAVES = tuple(re.compile(pattern) for pattern in (
    # Noise: one stream per detector and window, so adding a detector moves
    # no other detector's noise.
    rf"noise/{_SEGMENT}/(on|off)",
    # Real noise drawn at random: each pass's row order (its epoch counts
    # passes), each row's accrual phase, each example's row and its detectors'
    # window starts.
    r"noise/(segment_order|accrual|row)",
    rf"noise/start/{_SEGMENT}",
    # A reader of the caller's own, read in order: its seed.
    r"noise/reader",
    # A transient split's passes: pass p's order and balance, at epoch p.
    r"transient/(order|balance)",
    rf"augment/{_SEGMENT}",
    r"partner/choice",
    # A waveform generator's own draws, one stream per attribute so adding an
    # attribute shifts nothing else; an incoherent generator's components
    # draw under comp/{j}.
    rf"wave/{_SEGMENT}(/comp/[0-9]+)*/(param/{_SEGMENT}(/[0-9]+)?|noise|chance)",
    # The injection layer's draws, per generator.
    rf"inject/{_SEGMENT}/shift",
    rf"(sky|pol|scale)/{_SEGMENT}",
    r"pool/assign",
    # A standard dataset's injection sets, one stream per parameter
    # (gravyflow.src.dataset.standard.population).
    rf"standard/(volume|snr)/{_SEGMENT}",
    # The validator's own draws outside its datasets.
    r"validator/(roc|plot|pad)",
    # A model's hyperparameters, its layers and its build seeds
    # (gravyflow.src.model.hyperparameter.search_stream).
    rf"search(/{_SEGMENT})+",
    # A training run's model: the seeds Keras gives its layers when they are
    # built (initialisers, dropout).
    r"model/build",
))

#: Scopes a consumer may sit under: a named dataset, a pool of a composed
#: dataset, and a stage of the validator.
_SCOPES = (
    re.compile(rf"dataset/{_SEGMENT}/"),
    re.compile(rf"pool/{_SEGMENT}/"),
    re.compile(r"validator/(far|inj)/"),
)

_NAME = re.compile(_SEGMENT)


def check_name(name: str, what: str = "name") -> str:
    """A path segment: a non-empty string with no slash."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(
            f"A {what} must be a non-empty string of letters, digits and _.:+-, got {name!r}."
        )
    return name


def check_consumer(name: str) -> str:
    """``name`` if it is a registered consumer (under any scopes); else ValueError."""
    rest = name
    matched = True
    while matched:
        matched = False
        for scope in _SCOPES:
            found = scope.match(rest)
            if found:
                rest = rest[found.end():]
                matched = True
                break
    if any(leaf.fullmatch(rest) for leaf in _LEAVES):
        return name
    raise ValueError(f"{name!r} is not a registered random-stream consumer.")
