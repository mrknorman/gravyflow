#!/usr/bin/env python
"""
Training script for Gabbard et al. 2017 model.

This script trains the Gabbard2017 CNN for binary black hole detection
as described in arXiv:1712.06041v2.

Paper Reference:
    H. Gabbard et al., "Matching matched filtering with deep networks 
    for gravitational-wave astronomy", Phys. Rev. Lett. 120, 141103 (2018).
    arXiv: 1712.06041v2

Training Configuration (from paper and Gabbard PhD thesis):
    - Paper trains "until convergence" on simulated BBH signals
    - Gabbard thesis (2021) reports 10^7 training samples for CVAE models
    - CNN models typically require fewer samples than generative models
    - Default configuration: ~500k training samples over 20 epochs

Usage:
    python train_gabbard_2017.py [--epochs N] [--noise-type colored|real]
"""

import os

# Suppress JAX/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
os.environ['JAX_PLATFORMS'] = 'cuda'  # Skip TPU initialization
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

os.environ['KERAS_BACKEND'] = 'jax'

import argparse
from pathlib import Path

import keras
from keras import ops
import jax.numpy as jnp

import gravyflow as gf
from gravyflow.src.model.examples import Gabbard2017


class MaskAdapter(keras.utils.PyDataset):
    """
    Adapter to reshape injection masks for 2-class classification.
    
    The Gabbard2017 model uses 2-neuron softmax output:
    - Class 0: noise-only
    - Class 1: signal+noise
    
    This adapter converts binary masks to one-hot format.
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        if 'INJECTION_MASKS' in labels:
            mask = labels['INJECTION_MASKS']
            
            # Handle different mask shapes - flatten to (batch,)
            if len(mask.shape) == 2 and mask.shape[1] == 1:
                # Shape: (batch, 1) -> (batch,)
                mask = ops.squeeze(mask, axis=-1)
            elif len(mask.shape) == 2:
                # Shape: (num_generators, batch) -> take first generator
                mask = mask[0]
            elif len(mask.shape) == 3:
                # Shape: (num_generators, batch, time) -> reduce over time
                mask = mask[0]
                mask = jnp.max(mask, axis=-1)
            
            # Convert to int for one-hot encoding
            mask = ops.cast(mask, 'int32')
            
            # Convert to one-hot: [0] -> [1, 0], [1] -> [0, 1]
            # Class 0 = noise-only, Class 1 = signal+noise
            one_hot = keras.ops.one_hot(mask, num_classes=2)
            
            # Return one-hot labels directly (matches 2-neuron softmax output)
            return features, one_hot
        
        return features, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gabbard et al. 2017 BBH detection model"
    )
    # Paper specifies training "until convergence"
    # Gabbard thesis used ~30k epochs with 20k samples/epoch for CVAE
    # For CNN, 20 epochs with ~25k samples/epoch is reasonable
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='Number of training epochs (default: 1000, paper: until convergence)'
    )
    parser.add_argument(
        '--noise-type', choices=['colored', 'real'], default='colored',
        help='Noise type: colored (Gaussian) or real (LIGO data)'
    )
    # Paper: "4 × 10^5 independent timeseries" with 90/5/5 split
    # 360,000 / 32 batch_size = 11,250 steps for training
    parser.add_argument(
        '--steps-per-epoch', type=int, default=11250,
        help='Training steps per epoch (360k samples / 32 batch)'
    )
    parser.add_argument(
        '--validation-steps', type=int, default=625,
        help='Validation steps (20k samples / 32 batch = 625)'
    )
    parser.add_argument(
        '--save-model', type=str, default=None,
        help='Path to save trained model (e.g. model.keras)'
    )
    parser.add_argument(
        '--verbose', type=int, default=1, choices=[0, 1, 2],
        help='Keras verbosity: 0=silent, 1=progress bar, 2=one line per epoch'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Select noise type
    noise_type = (gf.NoiseType.REAL if args.noise_type == 'real' 
                  else gf.NoiseType.COLORED)
    
    print("=" * 60)
    print("Gabbard et al. 2017 - BBH Detection CNN")
    print("arXiv:1712.06041v2")
    print("=" * 60)
    print(f"Noise type: {noise_type.name}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    print("=" * 60)
    
    # Create datasets using paper specifications
    # Paper Section II: Training on simulated BBH signals
    print("\nCreating training dataset...")
    train_dataset = Gabbard2017.dataset(
        noise_type=noise_type,
        seed=42,
        group="train",
        steps_per_epoch=args.steps_per_epoch
    )
    
    print("Creating validation dataset...")
    val_dataset = Gabbard2017.dataset(
        noise_type=noise_type,
        seed=1001,
        group="validate",
        steps_per_epoch=args.validation_steps
    )
    
    # Wrap datasets with mask adapter
    train_dataset = MaskAdapter(train_dataset)
    val_dataset = MaskAdapter(val_dataset)
    
    # Get input shapes from first batch
    print("Getting input shapes...")
    for features, _ in [train_dataset[0]]:
        input_shape_onsource = features["ONSOURCE"].shape[1:]
        input_shape_offsource = features["OFFSOURCE"].shape[1:]
    
    print(f"Onsource shape: {input_shape_onsource}")
    print(f"Offsource shape: {input_shape_offsource}")
    
    # Create model using paper architecture
    print("\nCreating Gabbard2017 model...")
    model = Gabbard2017.model(
        input_shape_onsource=input_shape_onsource,
        input_shape_offsource=input_shape_offsource
    )
    
    # Compile with paper training configuration
    model = Gabbard2017.compile_model(model)
    
    # Print model summary
    model.summary()
    
    # Setup callbacks
    callbacks = []
    
    # Save best weights based on validation accuracy
    model_dir = Path(__file__).parent.parent.parent.parent / "res" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "gabbard_2017.keras"
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping to match "train until convergence"
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        start_from_epoch=10
    )
    callbacks.append(early_stop)
    
    print(f"Best model will be saved to: {best_model_path}")
    
    # Train model
    # Paper: Trained until convergence on simulated data
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    
    # Save model if requested
    if args.save_model:
        print(f"\nSaving model to {args.save_model}...")
        model.save(args.save_model)
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    main()
