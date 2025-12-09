#!/usr/bin/env python
"""
Validation script for Gabbard et al. 2017 model.

This script validates a trained Gabbard2017 model using the universal
validation framework, computing efficiency curves, FAR, and ROC metrics.

Usage:
    python validate_gabbard_2017.py --model path/to/model.keras
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import argparse
from pathlib import Path

import keras

import gravyflow as gf
from gravyflow.src.model.validate import Validator
from gravyflow.src.model.examples.base import validation_dataset_args, ValidationConfig
from gravyflow.src.model.examples.gabbard_2017 import Gabbard2017, Gabbard2017Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Gabbard et al. 2017 BBH detection model"
    )
    parser.add_argument(
        '--output-dir', type=str, default='./validation_results',
        help='Directory to save validation results'
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Name for validation run (defaults to model filename)'
    )
    parser.add_argument(
        "--noise-type", type=str, default="real", choices=["real", "colored"],
        help="Noise type to use for validation (real or colored)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine noise type
    noise_type = gf.NoiseType.REAL
    if args.noise_type == "colored":
        noise_type = gf.NoiseType.COLORED
    
    print("=" * 60)
    print("Gabbard et al. 2017 - Model Validation")
    print("=" * 60)
    
    model = Gabbard2017.pretrained()
    print("Model loaded successfully")
    
    # Model-specific config overrides
    gabbard_config = Gabbard2017Config()
    validation_config = ValidationConfig(
        sample_rate_hertz=gabbard_config.sample_rate_hertz,
        onsource_duration_seconds=gabbard_config.onsource_duration_seconds,
        offsource_duration_seconds=gabbard_config.offsource_duration_seconds,
    )
    
    # Get universal dataset args with model-specific overrides
    dataset_args = validation_dataset_args(
        noise_type=noise_type,
        config=validation_config,
        seed=42
    )
    
    print(f"Noise type: {noise_type.name}")
    print(f"Sample rate: {validation_config.sample_rate_hertz} Hz")
    print(f"Onsource duration: {validation_config.onsource_duration_seconds} s")
    print("=" * 60)
    
    # Setup output paths - include noise type in filenames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noise_suffix = noise_type.name.lower()
    name = f"gabbard_2017_{noise_suffix}"
    checkpoint_path = output_dir / f"{name}_validation.h5"
    plot_path = output_dir / f"{name}_validation.html"
    
    print(f"\nStarting validation...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Plots: {plot_path}")
    
    # Run validation
    validator = Validator.validate(
        model=model,
        name=name,
        dataset_args=dataset_args,
        checkpoint_file_path=checkpoint_path,
    )
    
    # Generate plots
    print("\nGenerating plots...")
    validator.plot(plot_path)
    
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
