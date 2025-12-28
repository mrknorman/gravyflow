#!/usr/bin/env python3
"""
Pool-Based Dataset Composition Example

This example demonstrates how to use the new FeaturePool and ComposedDataset
classes to create complex training datasets with mixed data sources.

The example creates a multi-class classifier training dataset with:
- Pure noise (label 0)  
- Noise with CBC injections (label 1)
- Noise with WNB injections (label 1) <- same label as CBC for "signal" class

This demonstrates key features:
1. Multiple pools with different probabilities
2. Multiple pools sharing the same class label
3. Mixing of pure noise and injection pools
"""

import logging
import gravyflow as gf
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_example_pools():
    """
    Create example feature pools for a binary signal/noise classifier.
    
    Returns:
        List of FeaturePool objects
    """
    # Common noise source (white noise for simplicity)
    noise_obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.H1, gf.IFO.L1]
    )
    
    # Scaling method for injections
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=8.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # CBC (Compact Binary Coalescence) generator
    cbc_generator = gf.CBCGenerator(
        scaling_method=scaling_method,
        mass_1_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        injection_chance=1.0  # Always inject in this pool
    )
    
    # WNB (White Noise Burst) generator
    wnb_generator = gf.WNBGenerator(
        scaling_method=scaling_method,
        duration_seconds=gf.Distribution(min_=0.01, max_=0.1, type_=gf.DistributionType.UNIFORM),
        min_frequency_hertz=gf.Distribution(min_=50.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
        max_frequency_hertz=gf.Distribution(min_=200.0, max_=500.0, type_=gf.DistributionType.UNIFORM),
        injection_chance=1.0
    )
    
    # Define feature pools
    pools = [
        # Pure noise pool (50% of examples, label 0)
        gf.FeaturePool(
            name="pure_noise",
            label=0,  # Noise class
            probability=0.50,
            noise_obtainer=noise_obtainer
        ),
        
        # CBC injection pool (30% of examples, label 1)
        gf.FeaturePool(
            name="cbc_injections",
            label=1,  # Signal class
            probability=0.30,
            noise_obtainer=noise_obtainer,
            injection_generators=[cbc_generator]
        ),
        
        # WNB injection pool (20% of examples, label 1 - same as CBC!)
        gf.FeaturePool(
            name="wnb_injections",
            label=1,  # Signal class (shared with CBC)
            probability=0.20,
            noise_obtainer=noise_obtainer,
            injection_generators=[wnb_generator]
        ),
    ]
    
    return pools


def main():
    """Main example demonstrating ComposedDataset usage."""
    
    logger.info("="*60)
    logger.info("Pool-Based Dataset Composition Example")
    logger.info("="*60)
    
    # Create feature pools
    pools = create_example_pools()
    
    logger.info(f"\nCreated {len(pools)} feature pools:")
    for pool in pools:
        logger.info(f"  - {pool.name}: label={pool.label}, probability={pool.probability:.0%}")
    
    # Create composed dataset
    with gf.env():
        dataset = gf.ComposedDataset(
            pools=pools,
            sample_rate_hertz=2048.0,
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=16.0,
            crop_duration_seconds=0.5,
            num_examples_per_batch=32,
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE,
            ],
            output_variables=[
                gf.ReturnVariables.POOL_LABEL,
            ],
            steps_per_epoch=100,
            seed=42
        )
        
        logger.info(f"\nDataset created:")
        logger.info(f"  - Steps per epoch: {len(dataset)}")
        logger.info(f"  - Batch size: {dataset.num_examples_per_batch}")
        logger.info(f"  - Number of unique classes: {dataset.pool_sampler.num_classes}")
        
        # Generate a few batches
        logger.info("\nGenerating sample batches...")
        
        label_counts = {0: 0, 1: 0}
        n_batches = 5
        
        for i in range(n_batches):
            inputs, outputs = dataset[i]
            
            labels = outputs['POOL_LABEL']
            
            for label in labels:
                label_counts[int(label)] += 1
            
            logger.info(f"  Batch {i+1}: shape={inputs['WHITENED_ONSOURCE'].shape}, "
                       f"labels={list(labels)}")
        
        total = sum(label_counts.values())
        logger.info(f"\nLabel distribution across {n_batches} batches:")
        logger.info(f"  - Label 0 (noise): {label_counts[0]} ({label_counts[0]/total:.1%})")
        logger.info(f"  - Label 1 (signal): {label_counts[1]} ({label_counts[1]/total:.1%})")
        
        # Show expected vs actual
        expected_noise = 0.50
        expected_signal = 0.50  # 0.30 + 0.20
        logger.info(f"\nExpected distribution:")
        logger.info(f"  - Noise: {expected_noise:.0%}")
        logger.info(f"  - Signal: {expected_signal:.0%}")
        
        logger.info("\n" + "="*60)
        logger.info("Example complete!")
        logger.info("="*60)


if __name__ == "__main__":
    main()
