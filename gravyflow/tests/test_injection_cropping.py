
import pytest
import numpy as np
from keras import ops
import gravyflow as gf
from gravyflow.src.dataset.features.injection import InjectionGenerator, RippleGenerator, ScalingMethod, ScalingTypes, ScalingOrdinality

class TestInjectionCropping:
    
    def test_injection_cropping_logic(self):
        # Setup
        sample_rate = 2048.0
        onsource_duration = 4.0
        crop_duration = 0.5
        total_duration = onsource_duration + 2 * crop_duration
        
        # Create a generator with large padding to force large shifts
        # We want to ensure that even with large shifts, we don't get cutoffs
        # if we handle cropping correctly.
        
        # Note: To test this effectively, we'd need to inspect the internal state 
        # or use a mock. But here we will verify the end-to-end flow produces 
        # the correct shape and runs without error.
        
        generator = RippleGenerator(
            mass_1_msun=10.0,
            mass_2_msun=10.0,
            distance_mpc=100.0,
            front_padding_duration_seconds=0.1,
            back_padding_duration_seconds=2.0,
            scaling_method=ScalingMethod(
                value=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT), 
                type_=ScalingTypes.SNR
            ),
            network=gf.Network(
                parameters=[gf.IFO.L1, gf.IFO.H1]
            )
        )
        
        injection_gen = InjectionGenerator(
            waveform_generators=[generator],
            seed=42
        )
        
        # Generate a batch
        # We expect the generator to yield injections that might be longer than total_duration
        # if we implement the fix. But the public API of __call__ yields injections, masks, params.
        # Wait, __call__ yields `injections`. 
        # In the current implementation, `injections` has shape corresponding to `total_duration`.
        # In the NEW implementation, `injections` yielded by `__call__` will be LONGER.
        
        # So this test will fail (or show different behavior) after the change.
        # Currently, `injections` shape is determined by `total_duration` passed to `generate`.
        
        iterator = injection_gen(
            sample_rate_hertz=sample_rate,
            onsource_duration_seconds=onsource_duration,
            crop_duration_seconds=crop_duration,
            num_examples_per_batch=4
        )
        
        injections, masks, params = next(iterator)
        
        # Check shape
        # Current behavior: shape corresponds to 5.0s (4.0 + 2*0.5)
        # expected_samples = int(5.0 * 2048) = 10240
        
        print(f"Injection shape: {injections.shape}")
        
        # Now test add_injections_to_onsource
        # onsource should have shape (Batch, Channels, Time)
        onsource = ops.zeros((4, 2, int(total_duration * sample_rate)))
        offsource = ops.zeros((4, 2, int(total_duration * sample_rate)))
        
        final_onsource, scaled_injections, scaling_params = injection_gen.add_injections_to_onsource(
            injections,
            masks,
            onsource,
            offsource,
            parameters_to_return=[],
            onsource_duration_seconds=onsource_duration
        )
        
        # The final onsource should have the same shape as input onsource
        assert final_onsource.shape == onsource.shape
        
        # If we implement the fix, `injections` from `next(iterator)` will be longer.
        # And `add_injections_to_onsource` will handle cropping.
        
