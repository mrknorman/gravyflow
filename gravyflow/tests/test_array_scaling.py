"""
A scaling method whose value is an array of targets: example i of the first
batch is scaled to target i, and is injected.
"""
import numpy as np

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args

SAMPLE_RATE = 2048.0
SNR_TARGETS = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])


def test_each_example_is_injected_at_its_target_snr():
    config = ValidationDatasetConfig(
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=2.0,  # three Welch segments: the PSD minimum
    )
    arguments = validation_dataset_args(noise_type=gf.NoiseType.COLORED, config=config, seed=42)
    generators = arguments["waveform_generators"]
    generator = generators[0] if isinstance(generators, list) else generators
    assert generator.scaling_method.type_ == gf.ScalingTypes.SNR
    generator.injection_chance = 1.0
    generator.scaling_method.value = SNR_TARGETS
    arguments.update(
        waveform_generators=[generator],
        num_examples_per_batch=len(SNR_TARGETS),
        input_variables=[gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.OFFSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS, gf.ScalingTypes.SNR],
    )

    inputs, targets = gf.Dataset(**arguments)[0]

    masks = np.asarray(targets[gf.ReturnVariables.INJECTION_MASKS.name])
    assert masks.shape == (len(SNR_TARGETS), 1)
    np.testing.assert_array_equal(masks, 1.0)

    # The label is the target, in order: exact.
    labels = np.asarray(targets[gf.ScalingTypes.SNR.name])
    np.testing.assert_array_equal(labels.reshape(-1), SNR_TARGETS.astype(np.float32))

    # The injected strain has the target SNR against the offsource the scaler
    # measured it with. Representation: the scaler multiplies by target / SNR
    # in float32 and the SNR is linear in amplitude, so recomputing it differs
    # only by the rounding of the two SNR sums (each a few ulp) and the product.
    injections = np.asarray(inputs[gf.ReturnVariables.INJECTIONS.name])[:, 0]
    offsource = np.asarray(inputs[gf.ReturnVariables.OFFSOURCE.name])
    realised = np.asarray(gf.snr(injections, offsource, SAMPLE_RATE))
    np.testing.assert_allclose(realised, SNR_TARGETS, rtol=16 * np.finfo(np.float32).eps)
