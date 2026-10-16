"""
The seed API the seed tree replaces, kept for one release with a warning.
"""

import warnings


def warn_deprecated(what: str, instead: str) -> None:
    """Warn that ``what`` goes in the next release, and what to use instead."""
    warnings.warn(
        f"{what} is deprecated and will be removed in the next release; {instead}.",
        DeprecationWarning,
        stacklevel=3,
    )
