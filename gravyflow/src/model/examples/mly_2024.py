"""
MLy 2024 - Real-Time Detection of Unmodeled Gravitational-Wave Transients

Paper: "Real-Time Detection of Unmodeled Gravitational-Wave Transients Using
       Convolutional Neural Networks"
Authors: Vasileios Skliris, Michael R. K. Norman, Patrick J. Sutton
arXiv: 2009.14611v4

This module implements the MLy pipeline with two CNN models:
1. MLyCoincidence (Model 1) - ResNet detecting coincident signals
2. MLyCoherence (Model 2) - Dual-branch with correlation input

Key Features:
- Detects generic transients without requiring precise signal models
- Uses Pearson cross-correlation between detectors for coherence
- Combined score = Model1_score × Model2_score

Dataset Configuration:
- 3 detectors: LIGO-Hanford (H), LIGO-Livingston (L), Virgo (V)
- 1 second segments at 1024 Hz
- Whitened with 20 Hz highpass
- Training signals: White Noise Bursts (WNBs)
"""

import os
from typing import Optional, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import keras
from keras.layers import (
    Input, Conv1D, Dense, Flatten, Activation, Add,
    GlobalAveragePooling1D, BatchNormalization, Concatenate
)
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel


@dataclass
class MLyConfig:
    """
    Configuration matching MLy paper specifications.
    
    References:
        Skliris et al., arXiv:2009.14611v4
    """
    # Sampling parameters
    sample_rate_hertz: float = 1024.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0  # For PSD estimation
    
    # Correlation parameters
    max_delay_seconds: float = 0.030  # ±30ms for light travel time
    
    # WNB signal parameters (Section III.B)
    wnb_bandwidth_min_hertz: float = 40.0
    wnb_bandwidth_max_hertz: float = 480.0
    wnb_duration_min_seconds: float = 0.05
    wnb_duration_max_seconds: float = 0.9
    
    # SNR ranges for training
    signal_snr_min: float = 10.0
    signal_snr_max: float = 50.0
    glitch_snr_min: float = 6.0
    glitch_snr_max: float = 70.0
    
    # Virgo PSD rescaling factors (Section IV.D)
    # These lower SNR of training signals in Virgo relative to LIGO
    virgo_rescale_model1: float = 4.0
    virgo_rescale_model2: float = 32.0
    
    # Training parameters
    injection_chance: float = 0.5
    batch_size: int = 32
    
    # Model 1 (Coincidence) architecture
    resnet_filters: Tuple[int, int, int] = (64, 128, 128)
    resnet_kernels: Tuple[Tuple[int, int], ...] = ((42, 36), (42, 36), (42, 36))
    
    # Model 2 (Coherence) architecture
    strain_filters: Tuple[int, int, int] = (56, 128, 214)
    strain_kernels: Tuple[int, int, int] = (32, 24, 16)
    strain_strides: Tuple[int, int, int] = (4, 2, 2)
    corr_filters: Tuple[int, int] = (64, 64)
    corr_kernels: Tuple[int, int] = (8, 5)
    
    # Dense layers
    dense_units: int = 256
    num_output_classes: int = 2


class MLyCoincidence(ExampleModel):
    """
    MLy Coincidence Model (Model 1) - ResNet Architecture
    
    Detects coincident signals appearing in at least two detectors.
    Uses residual blocks with skip connections to prevent vanishing gradients.
    
    Architecture (Figure 1 of paper):
    - 3 Residual blocks with Conv1D layers
    - BatchNormalization + ReLU after each layer
    - Global Average Pooling -> Dense(256) -> Dense(256) -> Softmax(2)
    
    Input: (batch, 1024, 3) - whitened strain from 3 detectors
    Output: (batch, 2) - [P(noise), P(signal)]
    
    Usage:
        >>> model = MLyCoincidence.model()
        >>> dataset = MLyCoincidence.dataset()
    """
    
    config = MLyConfig()
    
    @classmethod
    def model(
        cls,
        input_shape: Optional[Tuple[int, int]] = None,
        config: Optional[MLyConfig] = None
    ) -> keras.Model:
        """
        Create the MLy Coincidence ResNet architecture.
        
        Architecture from Figure 1:
        - 3 Residual blocks, each with 2 Conv1D layers
        - Skip connection adds input to output of each block
        - ReLU activation + BatchNorm after each Conv
        
        Args:
            input_shape: Shape of input (samples, detectors). Default: (1024, 3)
            config: Model configuration.
        
        Returns:
            Keras model ready for compilation.
        """
        cfg = config or cls.config
        
        if input_shape is None:
            num_samples = int(cfg.onsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape = (num_samples, 3)  # 3 detectors
        
        # Input layer
        inputs = Input(shape=input_shape, name="strain")
        x = inputs
        
        # 3 Residual blocks
        for block_idx, (filters, kernels) in enumerate(
            zip(cfg.resnet_filters, cfg.resnet_kernels)
        ):
            block_num = block_idx + 1
            
            # Save input for skip connection
            shortcut = x
            
            # First Conv1D in block
            x = Conv1D(
                filters, kernels[0], 
                padding='same',
                name=f"block{block_num}_conv1"
            )(x)
            x = BatchNormalization(name=f"block{block_num}_bn1")(x)
            x = Activation('relu', name=f"block{block_num}_relu1")(x)
            
            # Second Conv1D in block
            x = Conv1D(
                filters, kernels[1],
                padding='same', 
                name=f"block{block_num}_conv2"
            )(x)
            x = BatchNormalization(name=f"block{block_num}_bn2")(x)
            
            # Skip connection - need to match dimensions
            if shortcut.shape[-1] != filters:
                shortcut = Conv1D(
                    filters, 1, 
                    padding='same',
                    name=f"block{block_num}_shortcut"
                )(shortcut)
            
            x = Add(name=f"block{block_num}_add")([x, shortcut])
            x = Activation('relu', name=f"block{block_num}_relu2")(x)
        
        # Head
        x = GlobalAveragePooling1D(name="global_pool")(x)
        x = Dense(cfg.dense_units, name="dense1")(x)
        x = BatchNormalization(name="dense1_bn")(x)
        x = Activation('relu', name="dense1_relu")(x)
        x = Dense(cfg.dense_units, name="dense2")(x)
        x = BatchNormalization(name="dense2_bn")(x)
        x = Activation('relu', name="dense2_relu")(x)
        
        # Output
        outputs = Dense(
            cfg.num_output_classes, 
            activation='softmax',
            name="output"
        )(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="MLyCoincidence")
        return model
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[MLyConfig] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 5000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create dataset for Model 1 training.
        
        Training ratios from paper (N = 10000):
        - 6N Type 0 (noise)
        - 3N Type 1 (signal)
        - 6N Type 3 (single-detector glitches)
        
        For simplicity, we use WNBGenerator with injection_chance=0.5
        """
        cfg = config or cls.config
        
        # WNB generator for coherent signals
        waveform_generator = gf.WNBGenerator(
            duration_seconds=gf.Distribution(
                min_=cfg.wnb_duration_min_seconds,
                max_=cfg.wnb_duration_max_seconds,
                type_=gf.DistributionType.UNIFORM
            ),
            min_frequency_hertz=gf.Distribution(
                min_=20.0,
                max_=cfg.wnb_bandwidth_min_hertz,
                type_=gf.DistributionType.UNIFORM
            ),
            max_frequency_hertz=gf.Distribution(
                min_=cfg.wnb_bandwidth_min_hertz,
                max_=cfg.wnb_bandwidth_max_hertz + 20.0,
                type_=gf.DistributionType.UNIFORM
            ),
            scaling_method=gf.ScalingMethod(
                value=gf.Distribution(
                    min_=cfg.signal_snr_min,
                    max_=cfg.signal_snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                type_=gf.ScalingTypes.SNR
            ),
            injection_chance=cfg.injection_chance,
        )
        
        # Use 3 detectors - H, L, V
        # Note: Virgo may fall back to colored noise if real data unavailable
        noise_obtainer = gf.NoiseObtainer(
            noise_type=noise_type,
            ifos=[gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]
        )
        
        dataset = gf.GravyflowDataset(
            sample_rate_hertz=cfg.sample_rate_hertz,
            onsource_duration_seconds=cfg.onsource_duration_seconds,
            offsource_duration_seconds=cfg.offsource_duration_seconds,
            noise_obtainer=noise_obtainer,
            waveform_generators=waveform_generator,
            num_examples_per_batch=cfg.batch_size,
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE,
            ],
            output_variables=[
                gf.ReturnVariables.INJECTION_MASKS
            ],
            **kwargs
        )
        
        return dataset
    
    @classmethod
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.001
    ) -> keras.Model:
        """Compile with binary cross-entropy loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class MLyCoherence(ExampleModel):
    """
    MLy Coherence Model (Model 2) - Dual-Branch Architecture
    
    Detects correlation in phase and amplitude between detectors.
    Uses both strain data and Pearson correlation as inputs.
    
    Architecture (Figure 2 of paper):
    - Branch A (Strain): 3 Conv1D layers with stride
    - Branch B (Correlation): 2 Conv1D layers
    - Concatenate + Dense(256) -> Softmax(2)
    
    Inputs:
        - strain: (batch, 1024, 3) - whitened strain
        - correlation: (batch, 60, 3) - Pearson correlation for 3 detector pairs
    
    Output: (batch, 2) - [P(noise), P(signal)]
    
    Usage:
        >>> model = MLyCoherence.model()
        >>> dataset = MLyCoherence.dataset()
    """
    
    config = MLyConfig()
    
    @classmethod
    def model(
        cls,
        input_shape_strain: Optional[Tuple[int, int]] = None,
        input_shape_corr: Optional[Tuple[int, int]] = None,
        config: Optional[MLyConfig] = None
    ) -> keras.Model:
        """
        Create the MLy Coherence dual-branch architecture.
        
        Architecture from Figure 2:
        - Branch A: Strain through 3 strided Conv1D layers
        - Branch B: Correlation through 2 Conv1D layers
        - Concatenate and pass through dense layer
        
        Args:
            input_shape_strain: Shape of strain input (samples, detectors).
            input_shape_corr: Shape of correlation input (delays, pairs).
            config: Model configuration.
        
        Returns:
            Keras model with two inputs.
        """
        cfg = config or cls.config
        
        if input_shape_strain is None:
            num_samples = int(cfg.onsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape_strain = (num_samples, 3)
        
        if input_shape_corr is None:
            # ±30ms at 1024 Hz = ±30 samples = 60 total delays
            # 3 detector pairs: HL, HV, LV
            num_delays = int(2 * cfg.max_delay_seconds * cfg.sample_rate_hertz)
            input_shape_corr = (num_delays, 3)
        
        # Branch A: Strain input
        strain_input = Input(shape=input_shape_strain, name="strain")
        x_strain = strain_input
        
        for i, (filters, kernel, stride) in enumerate(
            zip(cfg.strain_filters, cfg.strain_kernels, cfg.strain_strides)
        ):
            x_strain = Conv1D(
                filters, kernel,
                strides=stride,
                padding='same',
                activation='relu',
                name=f"strain_conv{i+1}"
            )(x_strain)
        
        x_strain = GlobalAveragePooling1D(name="strain_pool")(x_strain)
        
        # Branch B: Correlation input
        corr_input = Input(shape=input_shape_corr, name="correlation")
        x_corr = corr_input
        
        for i, (filters, kernel) in enumerate(
            zip(cfg.corr_filters, cfg.corr_kernels)
        ):
            x_corr = Conv1D(
                filters, kernel,
                padding='same',
                activation='relu',
                name=f"corr_conv{i+1}"
            )(x_corr)
        
        x_corr = GlobalAveragePooling1D(name="corr_pool")(x_corr)
        
        # Merge branches
        x = Concatenate(name="merge")([x_strain, x_corr])
        x = Dense(cfg.dense_units, activation='relu', name="dense")(x)
        
        # Output
        outputs = Dense(
            cfg.num_output_classes,
            activation='softmax',
            name="output"
        )(x)
        
        model = Model(
            inputs=[strain_input, corr_input],
            outputs=outputs,
            name="MLyCoherence"
        )
        return model
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[MLyConfig] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 5000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create dataset for Model 2 training.
        
        Training ratios from paper (N = 10000):
        - 10N Type 0 (noise)
        - 10N Type 1 (signal)  
        - 5N Type 2a/b (incoherent glitches)
        - 5N Type 2c (coincident incoherent)
        
        Returns dataset with both whitened strain and rolling Pearson correlation.
        """
        cfg = config or cls.config
        
        waveform_generator = gf.WNBGenerator(
            duration_seconds=gf.Distribution(
                min_=cfg.wnb_duration_min_seconds,
                max_=cfg.wnb_duration_max_seconds,
                type_=gf.DistributionType.UNIFORM
            ),
            min_frequency_hertz=gf.Distribution(
                min_=20.0,
                max_=cfg.wnb_bandwidth_min_hertz,
                type_=gf.DistributionType.UNIFORM
            ),
            max_frequency_hertz=gf.Distribution(
                min_=cfg.wnb_bandwidth_min_hertz,
                max_=cfg.wnb_bandwidth_max_hertz + 20.0,
                type_=gf.DistributionType.UNIFORM
            ),
            scaling_method=gf.ScalingMethod(
                value=gf.Distribution(
                    min_=cfg.signal_snr_min,
                    max_=cfg.signal_snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                type_=gf.ScalingTypes.SNR
            ),
            injection_chance=cfg.injection_chance,
        )
        
        noise_obtainer = gf.NoiseObtainer(
            noise_type=noise_type,
            ifos=[gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]
        )
        
        dataset = gf.GravyflowDataset(
            sample_rate_hertz=cfg.sample_rate_hertz,
            onsource_duration_seconds=cfg.onsource_duration_seconds,
            offsource_duration_seconds=cfg.offsource_duration_seconds,
            noise_obtainer=noise_obtainer,
            waveform_generators=waveform_generator,
            num_examples_per_batch=cfg.batch_size,
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            input_variables=[
                gf.ReturnVariables.WHITENED_ONSOURCE,
                gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE,
            ],
            output_variables=[
                gf.ReturnVariables.INJECTION_MASKS
            ],
            max_arrival_time_difference_seconds=cfg.max_delay_seconds,
            **kwargs
        )
        
        return dataset
    
    @classmethod
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.001
    ) -> keras.Model:
        """Compile with binary cross-entropy loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class MLy(ExampleModel):
    """
    Combined MLy Pipeline
    
    Combines Coincidence (Model 1) and Coherence (Model 2) by multiplying
    their output scores.
    
    Final score = Model1_score[signal] × Model2_score[signal]
    
    A candidate needs high scores from BOTH models to be detected,
    requiring both coincidence in multiple detectors AND correlation.
    
    Usage:
        >>> model1 = MLy.coincidence_model()
        >>> model2 = MLy.coherence_model()
        >>> score = MLy.combined_score(model1_output, model2_output)
    """
    
    config = MLyConfig()
    
    @classmethod
    def coincidence_model(cls, **kwargs) -> keras.Model:
        """Create Model 1 (Coincidence)."""
        return MLyCoincidence.model(**kwargs)
    
    @classmethod
    def coherence_model(cls, **kwargs) -> keras.Model:
        """Create Model 2 (Coherence)."""
        return MLyCoherence.model(**kwargs)
    
    @classmethod
    def model(cls, config: Optional[MLyConfig] = None) -> Tuple[keras.Model, keras.Model]:
        """
        Create both MLy models.
        
        Returns:
            Tuple of (coincidence_model, coherence_model)
        """
        cfg = config or cls.config
        return (
            MLyCoincidence.model(config=cfg),
            MLyCoherence.model(config=cfg)
        )
    
    @staticmethod
    def combined_score(
        coincidence_output: np.ndarray,
        coherence_output: np.ndarray
    ) -> np.ndarray:
        """
        Compute combined MLy score.
        
        Args:
            coincidence_output: Model 1 output, shape (batch, 2)
            coherence_output: Model 2 output, shape (batch, 2)
        
        Returns:
            Combined scores, shape (batch,), on [0, 1]
        """
        # Take signal class probability (index 1) from each model
        score1 = coincidence_output[:, 1]
        score2 = coherence_output[:, 1]
        return score1 * score2
    
    @classmethod
    def dataset(cls, **kwargs) -> gf.GravyflowDataset:
        """Create dataset suitable for both models."""
        return MLyCoherence.dataset(**kwargs)
    
    @classmethod
    def compile_model(cls, model: keras.Model, **kwargs) -> keras.Model:
        """Compile a single model."""
        return MLyCoincidence.compile_model(model, **kwargs)
