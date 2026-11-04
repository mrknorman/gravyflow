"""
Write ``reference.json``: the digests of what datasets that hold no hard
example draw (each generator alone, an IncoherentGenerator, and per-example
networks of bundled detectors), which ``test_determinism.py`` checks every
later version against, so a change that adds a generator or a stream moves
none of theirs.

A digest covers a batch's drawn labels (the sky, the polarisation, the scaling
target, the generator's parameters and the detectors present, and each
component's parameters for an incoherent injection, whose own labels hold them
only where they agree), its placement and its noise: the values the seed
tree's streams give, which NumPy draws to the same bits on every platform, and
JAX's Gaussian noise, whose sign its uniform bits set. The placement is each
example's offset from the window's centre in whole samples, from CENTRAL_TIME
less each detector's delay at the drawn sky (rounded, so the float32
arithmetic between them cannot move it). White offsource noise, returned as it
is drawn, is pinned by each sample's sign, which no platform's rounding of a
Gaussian draw flips. Noise that arithmetic has touched, coloured noise and the
onsource less its injections, is pinned by the sign of its sum over each block
of ``BLOCK`` samples: the sum of independent or correlated noise is near zero,
where rounding could flip it, far more rarely than one sample is. The strain
the draws make passes through floating-point kernels whose bits may differ
between platforms and library versions, so it is not pinned; ``--full`` prints
every case's whole-batch digest instead, to compare two versions of the
library on one host.

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
#: The samples a block of noise that arithmetic has touched sums.
BLOCK = 64
#: What each incoherent component drew, by its tag, split and batch index.
DRAWN = {}


def uniform(low, high, dtype=float):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM, dtype=dtype)


def snr():
    return gf.ScalingMethod(uniform(8.0, 20.0), gf.ScalingTypes.SNR)


def wnb():
    return gf.WNBGenerator(scaling_method=snr(), duration_seconds=uniform(0.1, 0.3),
                           min_frequency_hertz=uniform(40.0, 80.0), max_frequency_hertz=uniform(120.0, 200.0))


class RecordedBurst(gf.WNBGenerator):
    """
    A white-noise burst that keeps what each call draws (``DRAWN``), under its
    ``tag``, the call's split and its batch index: an incoherent injection's
    labels hold its components' draws only where they agree, so its
    components' own are read here, as the dataset's copies of them draw them.
    """

    def generate(self, num_waveforms, sample_rate_hertz, duration_seconds, streams=None):
        waveforms, parameters = super().generate(num_waveforms, sample_rate_hertz, duration_seconds, streams)
        DRAWN[self.tag, streams.split, streams.k] = parameters
        return waveforms, parameters


def recorded_burst(tag):
    burst = RecordedBurst(scaling_method=snr(), duration_seconds=uniform(0.1, 0.3),
                          min_frequency_hertz=uniform(40.0, 80.0), max_frequency_hertz=uniform(120.0, 200.0))
    burst.tag = tag
    return burst


#: An incoherent injection's components' drawn labels.
COMPONENT_LABELS = (WP.DURATION_SECONDS, WP.MIN_FREQUENCY_HERTZ, WP.MAX_FREQUENCY_HERTZ)

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
    "incoherent": (lambda: [gf.IncoherentGenerator([recorded_burst(0), recorded_burst(1)])], [gf.IFO.H1, gf.IFO.L1],
                   gf.NoiseType.WHITE, ()),
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


def placement(y, ifos, sample_rate_hertz=512.0, window_seconds=1.0):
    """
    Each example's reference-time offset from the window's centre, in whole samples, (Batch, Generators): CENTRAL_TIME
    (a fraction of the window) less each present detector's delay at the generator's drawn sky, averaged over them.
    """
    network = gf.Network(list(ifos.ifos) if isinstance(ifos, gf.Detectors) else list(ifos))
    times = np.asarray(y["CENTRAL_TIME"], np.float64)
    offsets = np.empty(times.shape[:2])
    for g in range(times.shape[1]):
        delays = np.asarray(network.get_time_delay(
            np.asarray(y["RIGHT_ASCENSION_RADIANS"])[:, g], np.asarray(y["DECLINATION_RADIANS"])[:, g]), np.float64)
        offsets[:, g] = np.nanmean(times[:, g] * window_seconds - window_seconds / 2.0 - delays, axis=-1)
    return np.round(offsets * sample_rate_hertz).astype(np.int64)


def block_signs(noise):
    """The sign of the sum of each block of ``BLOCK`` samples of ``noise`` (..., Samples), in float64, packed."""
    noise = np.asarray(noise, np.float64)
    blocks = noise.reshape(noise.shape[:-1] + (noise.shape[-1] // BLOCK, BLOCK))
    return np.packbits(np.signbit(np.sum(blocks, axis=-1)))


def pinned(name, x, y, group, index):
    """What the case's digest covers in one batch: its drawn labels, its placement and its noise (see the module)."""
    _, ifos, noise, labels = CASES[name]
    drawn = {getattr(key, "name", str(key)) for key in (*SKY, *labels)}
    arrays = {key: value for key, value in y.items() if key in drawn}
    arrays["placement"] = placement(y, ifos)
    onsource_noise = np.asarray(x["ONSOURCE"], np.float64) - np.sum(np.asarray(y["INJECTIONS"], np.float64), axis=1)
    arrays["onsource noise sign"] = block_signs(onsource_noise)
    if noise is gf.NoiseType.WHITE:
        arrays["noise sign"] = np.packbits(np.signbit(np.asarray(x["OFFSOURCE"])))
    else:
        arrays["noise sign"] = block_signs(x["OFFSOURCE"])
    for (tag, split, k), parameters in sorted(DRAWN.items(), key=lambda item: str(item[0])):
        if split == group and k == index:
            for key in COMPONENT_LABELS:
                arrays[f"component {tag} {key.name}"] = np.asarray(parameters[key])
    return arrays


def digests(name, full=False):
    """The case's digest at each of ADDRESSES: of what it pins (``pinned``), or of its whole batch."""
    out = []
    for group, index in ADDRESSES:
        DRAWN.clear()
        x, y = dataset(name, group)[index]
        out.append(digest({**x, **y} if full else pinned(name, x, y, group, index)))
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
