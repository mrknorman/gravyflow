import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import keras
from keras import ops
import jax
import jax.numpy as jnp
import gravyflow as gf
from gravyflow.src.dataset.dataset import GravyflowDataset

def test_gravyflow_dataset_init():
    # Test initialization
    dataset = GravyflowDataset(
        seed=42,
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=2,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ReturnVariables.OFFSOURCE]
    )
    
    assert isinstance(dataset, GravyflowDataset)
    assert len(dataset) == 1000 # Default steps per epoch

def test_gravyflow_dataset_batch_generation():
    # Test batch generation with white noise and no injections
    dataset = GravyflowDataset(
        seed=42,
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=4,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ReturnVariables.OFFSOURCE],
        steps_per_epoch=10
    )
    
    # Get a batch
    inputs, outputs = dataset[0]
    
    # Check keys
    assert gf.ReturnVariables.ONSOURCE.name in inputs
    assert gf.ReturnVariables.OFFSOURCE.name in outputs
    
    # Check shapes
    # Onsource: (Batch, Detectors, Time)
    # Time = (1.0 + 2*0.5) * 1024 = 2048
    onsource = inputs[gf.ReturnVariables.ONSOURCE.name]
    offsource = outputs[gf.ReturnVariables.OFFSOURCE.name]
    
    assert ops.shape(onsource) == (4, 1, 2048)
    assert ops.shape(offsource) == (4, 1, 1024)
    
    # Check types
    assert ops.is_tensor(onsource)
    assert ops.is_tensor(offsource)

# Commenting out injection test for now - needs proper WNBGenerator setup
def test_gravyflow_dataset_with_injections():
    # Test with WNB injections
    wnb_gen = gf.WNBGenerator(
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        min_frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        max_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
            type_=gf.ScalingTypes.SNR
        )
    )
    
    dataset = GravyflowDataset(
        seed=42,
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=2,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        waveform_generators=[wnb_gen],
        input_variables=[gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.INJECTIONS],
        output_variables=[gf.ReturnVariables.OFFSOURCE]
    )
    
    inputs, outputs = dataset[0]
    
    assert gf.ReturnVariables.INJECTIONS.name in inputs
    injections = inputs[gf.ReturnVariables.INJECTIONS.name]
    
    # Injections shape: (NumGenerators, Batch, Detectors, Time)
    # Time = 2048 (same as onsource)
    assert ops.shape(injections) == (1, 2, 1, 2048)

def test_gravyflow_dataset_processing():
    # Test whitening and spectrogram
    dataset = GravyflowDataset(
        seed=42,
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=2,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        input_variables=[
            gf.ReturnVariables.WHITENED_ONSOURCE, 
            gf.ReturnVariables.SPECTROGRAM_ONSOURCE
        ],
        output_variables=[]
    )

    inputs, _ = dataset[0]

    whitened = inputs[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    spectrogram = inputs[gf.ReturnVariables.SPECTROGRAM_ONSOURCE.name]
    
    # Whitened should be cropped to onsource duration (1.0s * 1024 = 1024 samples)
    assert ops.shape(whitened) == (2, 1, 1024)
    
    # Spectrogram shape depends on implementation but should be tensor
    assert ops.is_tensor(spectrogram)
