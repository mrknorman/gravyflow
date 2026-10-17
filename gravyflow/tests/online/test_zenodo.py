"""
The Gravity Spy release on Zenodo, live: the record still lists the files, checksums,
sizes and licence gravyflow pins, and its smallest file downloads and verifies.
"""

import numpy as np
import requests

import gravyflow as gf
from gravyflow.src.dataset.features.gravity_spy import RELEASE_FILES, ZENODO_RECORD


def test_release_checksums_match_the_record():
    record = requests.get(f"https://zenodo.org/api/records/{ZENODO_RECORD}", timeout=60).json()
    files = {f["key"]: (f["checksum"].split(":")[1], f["size"]) for f in record["files"]}
    assert record["metadata"]["license"]["id"] == "cc-by-4.0"
    assert files == {key: (md5, size) for key, md5, size in RELEASE_FILES.values()}


def test_the_smallest_release_file_downloads_and_verifies():
    """H1 O1 is the smallest file; the library checks its md5 before reading it."""
    smallest = min(RELEASE_FILES, key=lambda entry: RELEASE_FILES[entry][2])
    assert smallest == ("H1", "O1")
    times = gf.get_glitch_times(gf.IFO.H1, observing_run=gf.ObservingRun.O1)
    run = gf.ObservingRun.O1.value
    assert len(times) > 0
    assert np.all((times >= run.start_gps_time) & (times < run.end_gps_time))
