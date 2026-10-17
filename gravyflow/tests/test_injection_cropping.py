"""
The crop from the generated injection buffer to the onsource window.

InjectionGenerator yields injections of the whole buffer, onsource plus a crop
either side; add_injections_to_onsource projects them and keeps the centre
that matches the onsource it is given.
"""
import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.features.injection import (
    CBCGenerator, InjectionGenerator, ScalingMethod, ScalingTypes
)

SAMPLE_RATE_HERTZ = 2048.0
ONSOURCE_SECONDS = 4.0
CROP_SECONDS = 0.5
NUM_EXAMPLES = 4


def test_injections_are_cropped_to_the_centre_of_the_buffer():
    # A strain-amplitude target, so the scaling does not depend on the noise.
    generator = CBCGenerator(
        mass_1_msun=10.0,
        mass_2_msun=10.0,
        distance_mpc=100.0,
        front_padding_duration_seconds=0.1,
        back_padding_duration_seconds=2.0,
        scaling_method=ScalingMethod(
            value=gf.Distribution(value=1.0e-21, type_=gf.DistributionType.CONSTANT),
            type_=ScalingTypes.HPEAK,
        ),
        network=gf.Network([gf.IFO.L1, gf.IFO.H1]),
    )
    injector = InjectionGenerator(waveform_generators=[generator])
    streams = injector.streams(0)
    injections, masks, parameters = injector.generate(
        streams,
        sample_rate_hertz=SAMPLE_RATE_HERTZ,
        onsource_duration_seconds=ONSOURCE_SECONDS,
        crop_duration_seconds=CROP_SECONDS,
        num_examples_per_batch=NUM_EXAMPLES,
    )

    # The buffer is the onsource window and both crops, exactly.
    buffer_samples = int((ONSOURCE_SECONDS + 2 * CROP_SECONDS) * SAMPLE_RATE_HERTZ)
    onsource_samples = int(ONSOURCE_SECONDS * SAMPLE_RATE_HERTZ)
    assert injections.shape == (1, NUM_EXAMPLES, 2, buffer_samples)

    def inject(num_samples):
        noise = np.zeros((NUM_EXAMPLES, 2, num_samples), np.float32)
        onsource, scaled, _ = injector.add_injections_to_onsource(
            injections, masks, noise, noise,
            parameters_to_return=[],
            onsource_duration_seconds=ONSOURCE_SECONDS,
            injection_parameters=parameters,
            streams=streams,
        )
        assert scaled.shape == (1, NUM_EXAMPLES, 2, num_samples)
        return np.asarray(onsource)

    whole = inject(buffer_samples)
    cropped = inject(onsource_samples)

    # The cropped window is the centre of the whole one, sample for sample, and
    # holds each example's signal.
    start = (buffer_samples - onsource_samples) // 2
    np.testing.assert_array_equal(cropped, whole[..., start:start + onsource_samples])
    assert np.all(np.max(np.abs(cropped), axis=(1, 2)) > 0.0)
