
import os
import logging
import numpy as np
import jax.numpy as jnp
import gravyflow as gf
from gravyflow.src.dataset.dataset import GravyflowDataset
from gravyflow.src.dataset.features.injection import WNBGenerator, ReturnVariables
from gravyflow.src.dataset.features.injection import WNBGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dataset_shapes():
    print("Verifying Dataset Shapes...")
    
    # 1. Setup Generators
    wnb = WNBGenerator(
        duration_seconds=0.5,
        min_frequency_hertz=50.0,
        max_frequency_hertz=100.0,
        injection_chance=1.0,
        scale_factor=1.0,
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(min_=10.0, max_=10.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        )
    )
    
    # 2. Setup Dataset
    dataset = GravyflowDataset(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        crop_duration_seconds=0.1,
        num_examples_per_batch=4,
        waveform_generators=[wnb],
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
        output_variables=[
            ReturnVariables.ONSOURCE, 
            ReturnVariables.START_GPS_TIME,
            ReturnVariables.INJECTIONS
        ]
    )
    
    # 3. Get Batch
    batch = next(iter(dataset))
    # dataset[0] returns (inputs, outputs) dicts based on variables
    # But wait, __getitem__ returns:
    # return final_onsource, scaled_injections, combined_scaling_params
    # NO, __getitem__ returns (inputs, outputs) based on input_variables and output_variables.
    # We need to check __getitem__ logic in dataset.py to see how it packs them.
    
    # Let's inspect dataset[0] directly
    data = dataset[0]
    
    # data is (inputs, outputs)
    inputs, outputs = data
    
    print("Inputs keys:", inputs.keys())
    print("Outputs keys:", outputs.keys())
    
    # Check ONSOURCE (in inputs or outputs depending on config)
    # We put them in output_variables.
    # Wait, dataset[0] returns structured dicts?
    # Let's see what keys we get.
    
    # Ideally standard logic:
    # onsource: (B, I, S)
    # gps: (B, I)
    
    for key, value in outputs.items():
        print(f"Output {key}: shape={value.shape}")
        
    for key, value in inputs.items():
        print(f"Input {key}: shape={value.shape}")
        
    # Manual check
    # We expect 'onsource' (if ReturnVariables.ONSOURCE used)
    # But ReturnVariables are Enums. The keys in the dict are likely the Enum values or strings?
    # Dataset usually maps them.
    
if __name__ == "__main__":
    verify_dataset_shapes()
