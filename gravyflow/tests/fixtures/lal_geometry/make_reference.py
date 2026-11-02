"""
Write ``reference.json``: lal's geometry of the bundled detectors (L1, H1, V1),
which ``test_detector_reference.py`` checks the Network against without lal.

For each detector, lal's Earth-fixed vertex (``CachedDetectors[...].location``,
metres); and for each of a set of isotropic sky positions at GPS times spread
over the observing runs, lal's arrival-time delay from the geocentre
(``TimeDelayFromEarthCenter``, seconds) and antenna pattern
(``ComputeDetAMResponse`` at ``GreenwichMeanSiderealTime``, F+ and Fx). The sky
angles are float32 values, as a projection takes them; every number is written
in full float64 precision, with lal's version.

Run from the repository root, in an environment with lal:

    python gravyflow/tests/fixtures/lal_geometry/make_reference.py
"""
import json
from pathlib import Path

import lal
import numpy as np

DETECTORS = {"L1": lal.LLO_4K_DETECTOR, "H1": lal.LHO_4K_DETECTOR, "V1": lal.VIRGO_DETECTOR}
NUM_POSITIONS = 64
SEED = 20261102


def main():
    rng = np.random.default_rng(SEED)
    right_ascension = rng.uniform(0.0, 2.0 * np.pi, NUM_POSITIONS).astype(np.float32).astype(np.float64)
    declination = np.arcsin(rng.uniform(-1.0, 1.0, NUM_POSITIONS)).astype(np.float32).astype(np.float64)
    polarization = rng.uniform(0.0, np.pi, NUM_POSITIONS).astype(np.float32).astype(np.float64)
    # Whole milliseconds, so each GPS time is exact as a LIGOTimeGPS and as a float.
    gps = np.round(rng.uniform(1.126e9, 1.40e9, NUM_POSITIONS), 3)

    detectors = {}
    for name, index in DETECTORS.items():
        detector = lal.CachedDetectors[index]
        delays, patterns = [], []
        for ra, dec, psi, t in zip(right_ascension, declination, polarization, gps):
            time = lal.LIGOTimeGPS(float(t))
            delays.append(lal.TimeDelayFromEarthCenter(detector.location, float(ra), float(dec), time))
            patterns.append(list(lal.ComputeDetAMResponse(
                detector.response, float(ra), float(dec), float(psi), lal.GreenwichMeanSiderealTime(time)
            )))
        detectors[name] = {
            "location_meters": [float(value) for value in detector.location],
            "time_delay_seconds": delays,
            "antenna_pattern": patterns,
        }

    reference = {
        "lal_version": lal.__version__,
        "seed": SEED,
        "right_ascension_radians": right_ascension.tolist(),
        "declination_radians": declination.tolist(),
        "polarization_radians": polarization.tolist(),
        "gps_time_seconds": gps.tolist(),
        "detectors": detectors,
    }
    path = Path(__file__).with_name("reference.json")
    path.write_text(json.dumps(reference, indent=1) + "\n")
    print(f"wrote {path} (lal {lal.__version__})")


if __name__ == "__main__":
    main()
