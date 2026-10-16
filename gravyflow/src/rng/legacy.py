"""
The seed API the seed tree replaces, kept for one release with a warning.
"""

import sys
import warnings

#: The modules whose frames a deprecation warning skips: the library itself.
_LIBRARY = "gravyflow.src"


def _caller_stacklevel() -> int:
    """
    The stacklevel of the first frame outside the library, from the caller of
    ``warn_deprecated``. Python shows a DeprecationWarning by default only
    when it is attributed to ``__main__``, so a warning attributed to a
    library frame (a shim reached through a dataclass ``__init__`` or another
    library call) would be dropped.
    """
    level, frame = 2, sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != _LIBRARY and not module.startswith(_LIBRARY + "."):
            return level
        level, frame = level + 1, frame.f_back
    return level


def warn_deprecated(what: str, instead: str) -> None:
    """Warn that ``what`` goes in the next release, and what to use instead."""
    warnings.warn(
        f"{what} is deprecated and will be removed in the next release; {instead}.",
        DeprecationWarning,
        stacklevel=_caller_stacklevel(),
    )
