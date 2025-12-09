#!/usr/bin/env python
"""
Training script for George & Huerta 2017 Deep Filtering models.

Paper: "Deep Neural Networks to Enable Real-time Multimessenger Astrophysics"
arXiv: 1701.00008v3

Supports three training modes:
- regression: Train predictor for mass estimation
- classification: Train detector for signal vs noise
- pretrain: Full paper strategy (regression → transfer → classification)

Usage:
    # Train with paper's transfer learning strategy
    python -m gravyflow.src.model.examples.train_george_huerta_2017 --model shallow --task pretrain --epochs 20

    # Train classification only
    python -m gravyflow.src.model.examples.train_george_huerta_2017 --model shallow --task classification --epochs 20

    # Train regression only (predictor)
    python -m gravyflow.src.model.examples.train_george_huerta_2017 --model shallow --task regression --epochs 20
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'jax')

import argparse
from pathlib import Path

import keras
from keras import ops
import jax.numpy as jnp
import numpy as np

import gravyflow as gf
from gravyflow.src.model.examples.george_huerta_2017 import (
    GeorgeHuerta2017Shallow,
    GeorgeHuerta2017Deep,
    GeorgeHuerta2017Config
)


class MaskAdapter(keras.utils.PyDataset):
    """Adapter to convert injection masks to 2-class one-hot format."""
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        if 'INJECTION_MASKS' in labels:
            mask = labels['INJECTION_MASKS']
            
            # Flatten to (batch,)
            if len(mask.shape) == 2 and mask.shape[1] == 1:
                mask = ops.squeeze(mask, axis=-1)
            elif len(mask.shape) == 2:
                mask = mask[0]
            elif len(mask.shape) == 3:
                mask = mask[0]
                mask = jnp.max(mask, axis=-1)
            
            # Convert to one-hot
            mask = ops.cast(mask, 'int32')
            one_hot = keras.ops.one_hot(mask, num_classes=2)
            
            return features, one_hot
        
        return features, labels


class RegressionAdapter(keras.utils.PyDataset):
    """Adapter to format mass parameters for regression."""
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, labels = self.dataset[index]
        
        # Stack mass_1 and mass_2 into (batch, 2) tensor
        if 'MASS_1_MSUN' in labels and 'MASS_2_MSUN' in labels:
            mass_1 = labels['MASS_1_MSUN']
            mass_2 = labels['MASS_2_MSUN']
            
            # Ensure they are 1D
            if len(mass_1.shape) > 1:
                mass_1 = ops.squeeze(mass_1)
            if len(mass_2.shape) > 1:
                mass_2 = ops.squeeze(mass_2)
            
            # Stack to (batch, 2)
            masses = ops.stack([mass_1, mass_2], axis=-1)
            
            return features, masses
        
        return features, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train George & Huerta 2017 Deep Filtering model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full paper strategy (regression → classification)
  python -m gravyflow.src.model.examples.train_george_huerta_2017 --model shallow --task pretrain

  # Classification only
  python -m gravyflow.src.model.examples.train_george_huerta_2017 --model shallow --task classification

  # Regression only (predictor)  
  python -m gravyflow.src.model.examples.train_george_huerta_2017 --model deep --task regression
        """
    )
    parser.add_argument(
        '--model', choices=['shallow', 'deep'], default='shallow',
        help='Model variant: shallow (Fig. 5) or deep (Fig. 6)'
    )
    parser.add_argument(
        '--task', choices=['regression', 'classification', 'pretrain'], default='pretrain',
        help='Training task: regression (mass prediction), classification (detection), or pretrain (both)'
    )
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Maximum epochs to train (per phase for pretrain)'
    )
    parser.add_argument(
        '--noise-type', choices=['colored', 'real'], default='colored',
        help='Noise type: colored (Gaussian) or real (LIGO data)'
    )
    parser.add_argument(
        '--steps-per-epoch', type=int, default=7000,
        help='Training steps per epoch'
    )
    parser.add_argument(
        '--validation-steps', type=int, default=700,
        help='Validation steps per epoch'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save models'
    )
    parser.add_argument(
        '--pretrained-weights', type=str, default=None,
        help='Path to pretrained regression weights (for classification with transfer)'
    )
    parser.add_argument(
        '--load-regression', action='store_true',
        help='Load existing regression model instead of training (for pretrain task)'
    )
    parser.add_argument(
        '--verbose', type=int, default=1, choices=[0, 1, 2],
        help='Keras verbosity: 0=silent, 1=progress bar, 2=one line per epoch'
    )
    return parser.parse_args()


def get_model_paths(output_dir: Path, model_name: str):
    """Get save paths for regression and classification models."""
    return {
        'regression': output_dir / f"{model_name.lower()}_regression.keras",
        'classification': output_dir / f"{model_name.lower()}_classification.keras"
    }


def train_regression(ModelClass, model_name: str, args, config, noise_type, output_path: Path):
    """Train regression model (mass prediction)."""
    print("\n" + "=" * 60)
    print(f"PHASE 1: REGRESSION (Mass Prediction)")
    print("=" * 60)
    
    print("\nCreating regression training dataset...")
    train_dataset = ModelClass.dataset_regression(
        noise_type=noise_type,
        config=config,
        seed=42,
        group="train",
        steps_per_epoch=args.steps_per_epoch
    )
    
    # Validation uses fixed SNR range (no curriculum)
    val_config = GeorgeHuerta2017Config(use_curriculum=False)
    
    print("Creating regression validation dataset...")
    val_dataset = ModelClass.dataset_regression(
        noise_type=noise_type,
        config=val_config,
        seed=1001,
        group="validate",
        steps_per_epoch=args.validation_steps
    )
    
    # Wrap with regression adapter
    train_dataset = RegressionAdapter(train_dataset)
    val_dataset = RegressionAdapter(val_dataset)
    
    # Get input shapes
    for features, _ in [train_dataset[0]]:
        input_shape_onsource = features["ONSOURCE"].shape[1:]
        input_shape_offsource = features["OFFSOURCE"].shape[1:]
    
    print(f"ONSOURCE shape: {input_shape_onsource}")
    print(f"OFFSOURCE shape: {input_shape_offsource}")
    
    # Create regression model
    print(f"\nCreating {model_name} regression model...")
    model = ModelClass.model(
        input_shape_onsource=input_shape_onsource,
        input_shape_offsource=input_shape_offsource,
        task="regression"
    )
    
    print("\nModel summary:")
    model.summary()
    
    model = ModelClass.compile_model(model, task="regression")
    
    # Callbacks
    callbacks = []
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(output_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        start_from_epoch=80
    )
    callbacks.append(early_stop)
    
    # Train
    print(f"\nTraining regression model for up to {args.epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    
    # Final evaluation
    print("\nFinal regression evaluation:")
    val_loss, val_mae = model.evaluate(val_dataset, verbose=args.verbose)
    print(f"Validation MAE: {val_mae:.4f} M☉")
    
    print(f"\n✓ Regression model saved to: {output_path}")
    
    return model, input_shape_onsource, input_shape_offsource


def train_classification(ModelClass, model_name: str, args, config, noise_type, 
                         output_path: Path, pretrained_weights: str = None,
                         pretrained_model = None,
                         input_shapes: tuple = None):
    """Train classification model (signal detection)."""
    print("\n" + "=" * 60)
    print(f"PHASE 2: CLASSIFICATION (Signal Detection)")
    print("=" * 60)
    
    print("\nCreating classification training dataset...")
    train_dataset = ModelClass.dataset(
        noise_type=noise_type,
        config=config,
        seed=43,  # Different seed for classification
        group="train",
        steps_per_epoch=args.steps_per_epoch
    )
    
    # Validation uses fixed SNR range (no curriculum)
    val_config = GeorgeHuerta2017Config(use_curriculum=False)
    
    print("Creating classification validation dataset...")
    val_dataset = ModelClass.dataset(
        noise_type=noise_type,
        config=val_config,
        seed=1002,
        group="validate",
        steps_per_epoch=args.validation_steps
    )
    
    # Wrap with mask adapter
    train_dataset = MaskAdapter(train_dataset)
    val_dataset = MaskAdapter(val_dataset)
    
    # Get input shapes
    if input_shapes is None:
        for features, _ in [train_dataset[0]]:
            input_shape_onsource = features["ONSOURCE"].shape[1:]
            input_shape_offsource = features["OFFSOURCE"].shape[1:]
    else:
        input_shape_onsource, input_shape_offsource = input_shapes
    
    print(f"ONSOURCE shape: {input_shape_onsource}")
    print(f"OFFSOURCE shape: {input_shape_offsource}")
    
    # Create classification model
    print(f"\nCreating {model_name} classification model...")
    model = ModelClass.model(
        input_shape_onsource=input_shape_onsource,
        input_shape_offsource=input_shape_offsource,
        task="classification"
    )
    
    # Transfer weights from regression if available
    if pretrained_model is not None:
        # Use in-memory model directly (pretrain mode)
        print("\n🔄 Transferring weights from in-memory regression model")
        regression_model = pretrained_model
        
        # Transfer weights layer by layer (except output)
        transferred = 0
        for reg_layer in regression_model.layers:
            if 'output' in reg_layer.name.lower():
                continue  # Skip output layer
            
            try:
                class_layer = model.get_layer(reg_layer.name)
                weights = reg_layer.get_weights()
                if weights:
                    class_layer.set_weights(weights)
                    transferred += 1
            except (ValueError, KeyError):
                pass
        
        print(f"   ↳ Transferred weights from {transferred} layers")
    
    elif pretrained_weights:
        # Load from disk (standalone classification mode)
        print(f"\n🔄 Loading pretrained model from: {pretrained_weights}")
        regression_model = keras.models.load_model(pretrained_weights)
        
        # Transfer weights layer by layer (except output)
        transferred = 0
        for reg_layer in regression_model.layers:
            if 'output' in reg_layer.name.lower():
                continue  # Skip output layer
            
            try:
                class_layer = model.get_layer(reg_layer.name)
                weights = reg_layer.get_weights()
                if weights:
                    class_layer.set_weights(weights)
                    transferred += 1
            except (ValueError, KeyError):
                pass
        
        print(f"   ↳ Transferred weights from {transferred} layers")
        del regression_model
    
    print("\nModel summary:")
    model.summary()
    
    model = ModelClass.compile_model(model, task="classification")
    
    # Callbacks
    callbacks = []
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(output_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        start_from_epoch=80,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Train
    print(f"\nTraining classification model for up to {args.epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    
    # Final evaluation
    print("\nFinal classification evaluation:")
    val_loss, val_acc = model.evaluate(val_dataset, verbose=args.verbose)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    print(f"\n✓ Classification model saved to: {output_path}")
    
    return model


def main():
    args = parse_args()
    
    # Select model class
    if args.model == 'shallow':
        ModelClass = GeorgeHuerta2017Shallow
        model_name = "GeorgeHuerta2017Shallow"
    else:
        ModelClass = GeorgeHuerta2017Deep
        model_name = "GeorgeHuerta2017Deep"
    
    noise_type = (gf.NoiseType.REAL if args.noise_type == 'real' 
                  else gf.NoiseType.COLORED)
    
    # Enable curriculum learning for training
    config = GeorgeHuerta2017Config(
        use_curriculum=True,
        snr_curriculum_verbose=True
    )
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(gf.__file__).parent / "res" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = get_model_paths(output_dir, model_name)
    
    # Header
    print("=" * 60)
    print(f"George & Huerta 2017 - {model_name}")
    print("Deep Filtering for Real-time Multimessenger Astrophysics")
    print("arXiv:1701.00008v3")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Noise type: {noise_type.name}")
    print(f"Epochs per phase: {args.epochs}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if args.task == 'regression':
        # Regression only
        train_regression(ModelClass, model_name, args, config, noise_type, paths['regression'])
        
    elif args.task == 'classification':
        # Classification only (with optional pretrained weights)
        train_classification(
            ModelClass, model_name, args, config, noise_type, 
            paths['classification'],
            pretrained_weights=args.pretrained_weights
        )
        
    elif args.task == 'pretrain':
        # Full pipeline: regression → classification
        print("\n🎯 FULL PRETRAIN PIPELINE")
        print("   Phase 1: Regression (mass prediction)")
        print("   Phase 2: Transfer → Classification (detection)")
        
        # Phase 1: Regression - load if exists and --load-regression, else train
        if args.load_regression and paths['regression'].exists():
            print(f"\n📂 Loading existing regression model from: {paths['regression']}")
            regression_model = keras.models.load_model(paths['regression'])
            regression_model.summary()
            
            # Get input shapes from loaded model
            onsource_input = regression_model.get_layer('ONSOURCE')
            offsource_input = regression_model.get_layer('OFFSOURCE')
            input_shape_on = onsource_input.output.shape[1:]
            input_shape_off = offsource_input.output.shape[1:]
            print(f"ONSOURCE shape: {input_shape_on}")
            print(f"OFFSOURCE shape: {input_shape_off}")
        else:
            if args.load_regression:
                print(f"\n⚠️  Regression model not found at {paths['regression']}, training from scratch...")
            regression_model, input_shape_on, input_shape_off = train_regression(
                ModelClass, model_name, args, config, noise_type, paths['regression']
            )
        
        # Phase 2: Classification with transfer (use in-memory model)
        train_classification(
            ModelClass, model_name, args, config, noise_type, 
            paths['classification'],
            pretrained_model=regression_model,
            input_shapes=(input_shape_on, input_shape_off)
        )
        
        print("\n" + "=" * 60)
        print("✓ PRETRAIN COMPLETE!")
        print("=" * 60)
        print(f"Regression model:     {paths['regression']}")
        print(f"Classification model: {paths['classification']}")
        print("=" * 60)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
