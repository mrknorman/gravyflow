"""
Write ``reference.json``: the digests of what datasets that hold no hard
example draw (each generator alone, an IncoherentGenerator, and per-example
networks of bundled detectors), which ``test_determinism.py`` checks every
later version against, so a change that adds a generator or a stream moves
none of theirs.

A digest covers a batch's drawn labels (the sky, the polarisation, the
scaling target, the generator's parameters and the detectors present): the
values the seed tree's streams give, which NumPy draws to the same bits on
every platform. The strain the draws make passes through floating-point
kernels whose bits may differ between platforms and library versions, so it
is not pinned; ``--full`` prints every case's whole-batch digest instead, to
compare two versions of the library on one host.

Run from the repository root, against the version to pin:

    python gravyflow/tests/fixtures/stream_digests/make_reference.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import gravyflow as gf

RV, WP = gf.ReturnVariables, gf.WaveformParameters
SKY = (WP.RIGHT_ASCENSION_RADIANS, WP.DECLINATION_RADIANS, WP.POLARIZATION_RADIANS, gf.ScalingTypes.SNR)
#: Each case's batches: (split, index).
ADDRESSES = (("train", 0), ("train", 3), ("validate", 1))


def uniform(low, high, dtype=float):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM, dtype=dtype)


def snr():
    return gf.ScalingMethod(uniform(8.0, 20.0), gf.ScalingTypes.SNR)


def wnb():
    return gf.WNBGenerator(scaling_method=snr(), duration_seconds=uniform(0.1, 0.3),
                           min_frequency_hertz=uniform(40.0, 80.0), max_frequency_hertz=uniform(120.0, 200.0))


#: Each case: its generators, network and noise, and the drawn labels it returns.
CASES = {
    "white-noise burst": (lambda: [wnb()], [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE,
                          (WP.DURATION_SECONDS, WP.MIN_FREQUENCY_HERTZ, WP.MAX_FREQUENCY_HERTZ)),
    "binary": (lambda: [gf.CBCGenerator(scaling_method=snr(), mass_1_msun=uniform(20.0, 40.0),
                                        mass_2_msun=uniform(20.0, 40.0), inclination_radians=uniform(0.0, 3.0))],
               [gf.IFO.H1, gf.IFO.L1, gf.IFO.V1], gf.NoiseType.COLORED,
               (WP.MASS_1_MSUN, WP.MASS_2_MSUN, WP.INCLINATION_RADIANS)),
    "sine-Gaussian": (lambda: [gf.SineGaussianGenerator(scaling_method=snr(), frequency_hertz=uniform(60.0, 200.0),
                                                        quality_factor=uniform(4.0, 12.0))],
                      [gf.IFO.L1, gf.IFO.H1], gf.NoiseType.COLORED, (WP.FREQUENCY_HERTZ, WP.QUALITY_FACTOR)),
    "chirplet": (lambda: [gf.ChirpletGenerator(scaling_method=snr(), start_frequency_hertz=uniform(40.0, 80.0),
                                               end_frequency_hertz=uniform(120.0, 200.0), duration_seconds=0.2)],
                 [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE, (WP.START_FREQUENCY_HERTZ, WP.END_FREQUENCY_HERTZ)),
    "ringdown": (lambda: [gf.RingdownGenerator(scaling_method=snr(), frequency_hertz=uniform(100.0, 200.0),
                                               damping_time_seconds=uniform(0.01, 0.05))],
                 [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE, (WP.FREQUENCY_HERTZ, WP.DAMPING_TIME_SECONDS)),
    "periodic wave": (lambda: [gf.PeriodicWaveGenerator(scaling_method=snr(), frequency_hertz=uniform(60.0, 120.0),
                                                        duration_seconds=0.2)],
                      [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE, (WP.FREQUENCY_HERTZ,)),
    "two generators": (lambda: [wnb(), gf.RingdownGenerator(scaling_method=snr(), frequency_hertz=150.0,
                                                            damping_time_seconds=0.02)],
                       [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE, (WP.DURATION_SECONDS,)),
    "incoherent": (lambda: [gf.IncoherentGenerator([wnb(), wnb()])], [gf.IFO.H1, gf.IFO.L1], gf.NoiseType.WHITE, ()),
    "bundled detectors": (lambda: [wnb()], gf.Detectors([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1],
                                                       num_present=uniform(2, 3, int)),
                          gf.NoiseType.WHITE, (RV.DETECTOR_MASK, WP.DURATION_SECONDS)),
}


def dataset(name, group, seed=7):
    generators, ifos, noise, labels = CASES[name]
    return gf.Dataset(
        seed=seed, group=group, noise_obtainer=gf.NoiseObtainer(noise_type=noise), ifos=ifos,
        waveform_generators=generators(), sample_rate_hertz=512.0, onsource_duration_seconds=1.0,
        offsource_duration_seconds=4.0, num_examples_per_batch=4,
        input_variables=[RV.ONSOURCE, RV.OFFSOURCE],
        output_variables=[RV.INJECTIONS, RV.CENTRAL_TIME, *SKY, *labels])


def digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        h.update(np.ascontiguousarray(np.asarray(arrays[key])).tobytes())
    return h.hexdigest()


def digests(name, full=False):
    """The case's digest at each of ADDRESSES: of its drawn labels, or of its whole batch where ``full``."""
    _, _, _, labels = CASES[name]
    drawn = {getattr(key, "name", str(key)) for key in (*SKY, *labels)}
    out = []
    for group, index in ADDRESSES:
        x, y = dataset(name, group)[index]
        arrays = {**x, **y} if full else {key: value for key, value in y.items() if key in drawn}
        out.append(digest(arrays))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", action="store_true", help="print whole-batch digests instead of writing")
    full = parser.parse_args().full
    reference = {name: digests(name, full) for name in CASES}
    if full:
        print(json.dumps(reference, indent=1))
    else:
        path = Path(__file__).with_name("reference.json")
        path.write_text(json.dumps(reference, indent=1) + "\n")
        print(f"wrote {path}")
