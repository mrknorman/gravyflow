from pathlib import Path

from ._helpers import (
    GOLDENS,
    assert_matches_golden,
    effective_count,
    representation_tolerance,
    statistical_bound,
)

PATH = Path(__file__).resolve().parent

#: The name the older golden tests call: it compares, and refuses a missing or
#: unsidecarred golden; it no longer writes one.
compare_and_save_parameters = assert_matches_golden
