"""
George & Huerta 2017 - Deep Filtering for Gravitational Wave Detection

Paper: "Deep Neural Networks to Enable Real-time Multimessenger Astrophysics"
Authors: Daniel George, E. A. Huerta
arXiv: 1701.00008v3
Published: Physical Review D 97, 044039 (2018)

This module provides two model architectures from the paper:
1. GeorgeHuerta2017Shallow - 3 conv layers (Fig. 5), ~2MB, faster
2. GeorgeHuerta2017Deep - 4 conv layers (Fig. 6), ~23MB, more accurate

Key Innovations:
- Dilated convolutions for temporal aggregation
- Transfer learning from predictor to classifier
- Gradual noise increase training scheme

Dataset Configuration:
- Component masses: 5-75 M☉ (mass ratio 1-10)
- Whitened timeseries, 1 second @ 8192 Hz
- EOB waveforms (non-spinning, quasi-circular)
- Peak shifted randomly within 0.2 seconds
"""

import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import keras
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, 
    Activation, Permute, Softmax, Reshape
)
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel


@dataclass
class GeorgeHuerta2017Config:
    """
    Configuration matching George & Huerta 2017 paper specifications.
    
    References:
        Section III "Method" of arXiv:1701.00008v3
    """
    # Sampling parameters
    sample_rate_hertz: float = 8192.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    
    # Mass distributions (Paper: "individual masses of the BBH... 
    # restricted to lie between 5M☉ and 75M☉" with "mass-ratios 
    # confined between 1 and 10")
    mass_1_min_msun: float = 5.0
    mass_1_max_msun: float = 75.0
    mass_2_min_msun: float = 5.0
    mass_2_max_msun: float = 75.0
    
    # SNR scaling (Paper trains with SNR gradually decreasing from 100 to 5-15)
    # For inference testing, paper uses range 5-15
    snr_min: float = 5.0
    snr_max: float = 15.0
    
    # Curriculum learning (Paper: "trains with SNR gradually decreasing from 100 to 5-15")
    # Enable to match paper's gradual noise increase training scheme
    use_curriculum: bool = True
    snr_curriculum_start_min: float = 80.0
    snr_curriculum_start_max: float = 100.0
    snr_curriculum_num_epochs: int = 80
    snr_curriculum_schedule: 'gf.CurriculumSchedule' = None  # Defaults to LINEAR
    snr_curriculum_verbose: bool = False  # Print SNR range at each epoch
    
    # Training parameters
    injection_chance: float = 0.5  # 50% signal+noise, 50% noise-only
    batch_size: int = 32
    
    # Shallow model architecture from Fig. 5
    # Layer 1: Conv 16 filters, kernel 16, no dilation
    # Layer 2: Conv 32 filters, kernel 8, dilation 4
    # Layer 3: Conv 64 filters, kernel 8, dilation 4
    # Pool size 4, stride 4 after each conv
    # FC: 64, 2
    shallow_conv_configs: tuple = (
        (16, 16, 1),   # filters, kernel, dilation
        (32, 8, 4),    # with dilation
        (64, 8, 4),    # with dilation
    )
    shallow_pool_size: int = 4
    shallow_fc_units: int = 64
    
    # Deep model architecture from Fig. 6
    # Layer 1: Conv 64 filters, kernel 16, dilation 1
    # Layer 2: Conv 128 filters, kernel 16, dilation 2
    # Layer 3: Conv 256 filters, kernel 16, dilation 2
    # Layer 4: Conv 512 filters, kernel 32, dilation 2
    # Pool size 4, stride 4 after each conv
    # FC: 128, 64, 2
    deep_conv_configs: tuple = (
        (64, 16, 1),
        (128, 16, 2),
        (256, 16, 2),
        (512, 32, 2),
    )
    deep_pool_size: int = 4
    deep_fc_units: tuple = (128, 64)
    
    # Output configuration
    num_output_classes: int = 2  # classifier: signal+noise vs noise-only


class GeorgeHuerta2017Shallow(ExampleModel):
    """
    George & Huerta 2017 Shallow Network (Fig. 5)
    
    3 convolutional layers with dilated convolutions, followed by
    2 fully connected layers. Outputs 2 classes (signal vs noise)
    or 2 parameters (component masses) depending on task.
    
    Size: ~2MB
    Speed: 6.7ms CPU, 106μs GPU per 1-second input
    
    Usage:
        >>> model = gf.examples.GeorgeHuerta2017Shallow.model()
        >>> dataset = gf.examples.GeorgeHuerta2017Shallow.dataset()
    
    References:
        D. George & E.A. Huerta, "Deep Neural Networks to Enable 
        Real-time Multimessenger Astrophysics", Phys. Rev. D 97, 044039.
        arXiv:1701.00008v3, Fig. 5
    """
    
    config = GeorgeHuerta2017Config()
    
    @classmethod
    def model(
        cls,
        input_shape_onsource: Optional[Tuple[int, int]] = None,
        input_shape_offsource: Optional[Tuple[int, int]] = None,
        config: Optional[GeorgeHuerta2017Config] = None,
        task: str = "classification"  # or "regression"
    ) -> keras.Model:
        """
        Create the George & Huerta 2017 shallow CNN architecture.
        
        Architecture from Fig. 5:
        - 3 Convolutional layers with ReLU, dilated convolutions
        - MaxPooling (size 4, stride 4) after each conv
        - 2 Fully connected layers
        - Softmax output for classification, linear for regression
        
        Args:
            input_shape_onsource: Shape of onsource input (samples, channels).
            input_shape_offsource: Shape of offsource input for whitening.
            config: Model configuration. Uses paper defaults if None.
            task: "classification" (2-class softmax) or "regression" (mass prediction)
        
        Returns:
            Keras model ready for compilation.
        """
        cfg = config or cls.config
        
        # Default shapes based on paper specs (1 second @ 8192 Hz, 1 channel)
        if input_shape_onsource is None:
            num_samples = int(cfg.onsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape_onsource = (num_samples, 1)
        
        if input_shape_offsource is None:
            num_samples = int(cfg.offsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape_offsource = (num_samples, 1)
        
        # Input layers
        onsource_input = Input(shape=input_shape_onsource, name="ONSOURCE")
        offsource_input = Input(shape=input_shape_offsource, name="OFFSOURCE")
        
        # Whitening layer
        # Paper: "whitened the signals using aLIGO's PSD"
        x = gf.Whiten(
            sample_rate_hertz=cfg.sample_rate_hertz,
            onsource_duration_seconds=cfg.onsource_duration_seconds
        )([onsource_input, offsource_input])
        
        # Permute to (batch, channels, samples) for Conv1D
        x = Permute((2, 1))(x)
        
        # 3 Convolutional blocks from Fig. 5
        for i, (filters, kernel_size, dilation) in enumerate(cfg.shallow_conv_configs):
            layer_num = i + 1
            
            x = Conv1D(
                filters, 
                kernel_size, 
                dilation_rate=dilation,
                name=f"Conv1D_{layer_num}"
            )(x)
            
            # MaxPooling
            x = MaxPooling1D(
                pool_size=cfg.shallow_pool_size, 
                strides=cfg.shallow_pool_size,
                name=f"MaxPool_{layer_num}"
            )(x)
            
            # ReLU activation
            x = Activation('relu', name=f"ReLU_{layer_num}")(x)
        
        # Flatten
        x = Flatten(name="Flatten")(x)
        
        # FC layer 1
        x = Dense(cfg.shallow_fc_units, name="Dense_1")(x)
        x = Activation('relu', name="ReLU_FC_1")(x)
        
        # Output layer
        if task == "classification":
            # 2-class softmax for detection
            output = Dense(cfg.num_output_classes, activation='softmax', 
                          name="output")(x)
        else:
            # Linear output for mass prediction (regression)
            output = Dense(2, activation='linear', name="output")(x)
        
        model = Model(
            inputs=[onsource_input, offsource_input],
            outputs=output,
            name="GeorgeHuerta2017Shallow"
        )
        
        return model
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 7000,  # ~2500 templates * 2.8 noise realizations
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset matching George & Huerta 2017 specifications.
        
        Dataset Configuration from paper:
        - Component masses 5-75 M☉ with mass ratio 1-10
        - ~2500 templates with multiple noise realizations
        - 50% signal+noise, 50% noise-only
        
        Curriculum Learning:
            Enable via config.use_curriculum=True to match the paper's 
            "gradual noise increase training scheme" where SNR decreases
            from 100 to 5-15 over the training epochs.
        
        Args:
            noise_type: Type of noise. Paper used Gaussian.
            config: Dataset configuration. Uses paper defaults if None.
                Set config.use_curriculum=True to enable curriculum learning.
            seed: Random seed for reproducibility.
            group: Dataset group ("train", "validate", "test").
            steps_per_epoch: Number of batches per epoch.
            **kwargs: Additional arguments passed to GravyflowDataset.
        
        Returns:
            GravyflowDataset configured to match paper specifications.
        """
        cfg = config or cls.config
        
        # Mass distributions
        mass_1_distribution = gf.Distribution(
            min_=cfg.mass_1_min_msun,
            max_=cfg.mass_1_max_msun,
            type_=gf.DistributionType.UNIFORM
        )
        
        mass_2_distribution = gf.Distribution(
            min_=cfg.mass_2_min_msun,
            max_=cfg.mass_2_max_msun,
            type_=gf.DistributionType.UNIFORM
        )
        
        # Inclination uniform in [0, pi]
        inclination_distribution = gf.Distribution(
            min_=0.0,
            max_=np.pi,
            type_=gf.DistributionType.UNIFORM
        )
        
        # SNR scaling - use curriculum if enabled
        if cfg.use_curriculum:
            # Curriculum learning as described in paper:
            # "trains with SNR gradually decreasing from 100 to 5-15"
            schedule = cfg.snr_curriculum_schedule or gf.CurriculumSchedule.LINEAR
            
            snr_curriculum = gf.Curriculum(
                start=gf.Distribution(
                    min_=cfg.snr_curriculum_start_min,
                    max_=cfg.snr_curriculum_start_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                end=gf.Distribution(
                    min_=cfg.snr_min,
                    max_=cfg.snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                num_epochs=cfg.snr_curriculum_num_epochs,
                schedule=schedule,
                verbose=cfg.snr_curriculum_verbose,
                name="SNR"
            )
            
            scaling_method = gf.ScalingMethod(
                value=snr_curriculum,
                type_=gf.ScalingTypes.SNR
            )
        else:
            # Fixed SNR range (default for inference/testing)
            scaling_method = gf.ScalingMethod(
                value=gf.Distribution(
                    min_=cfg.snr_min,
                    max_=cfg.snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                type_=gf.ScalingTypes.SNR
            )
        
        # Waveform generator
        # Paper: "EOB waveforms... non-spinning BBHs"
        waveform_generator = gf.CBCGenerator(
            mass_1_msun=mass_1_distribution,
            mass_2_msun=mass_2_distribution,
            inclination_radians=inclination_distribution,
            scaling_method=scaling_method,
            injection_chance=cfg.injection_chance,
            # Zero spin as per paper
            spin_1_in=(0.0, 0.0, 0.0),
            spin_2_in=(0.0, 0.0, 0.0),
        )
        
        # Noise configuration
        noise_obtainer = gf.NoiseObtainer(
            noise_type=noise_type,
            ifos=gf.IFO.L1
        )
        
        # Create dataset
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
                gf.ReturnVariables.ONSOURCE,
                gf.ReturnVariables.OFFSOURCE
            ],
            output_variables=[
                gf.ReturnVariables.INJECTION_MASKS
            ],
            **kwargs
        )
        
        return dataset
    
    @classmethod
    def dataset_regression(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 7000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset for mass regression pre-training.
        
        This follows the paper's transfer learning strategy where models
        are first trained to predict component masses (regression), then
        transferred to classification.
        
        Output Variables:
            - MASS_1_MSUN: Primary component mass
            - MASS_2_MSUN: Secondary component mass
        
        Note:
            For regression, injection_chance is set to 1.0 (always inject)
            since we need signal parameters to predict.
        
        Args:
            noise_type: Type of noise. Paper used Gaussian.
            config: Dataset configuration.
            seed: Random seed for reproducibility.
            group: Dataset group ("train", "validate", "test").
            steps_per_epoch: Number of batches per epoch.
            **kwargs: Additional arguments.
        
        Returns:
            GravyflowDataset configured for mass regression.
        """
        cfg = config or cls.config
        
        # Mass distributions
        mass_1_distribution = gf.Distribution(
            min_=cfg.mass_1_min_msun,
            max_=cfg.mass_1_max_msun,
            type_=gf.DistributionType.UNIFORM
        )
        
        mass_2_distribution = gf.Distribution(
            min_=cfg.mass_2_min_msun,
            max_=cfg.mass_2_max_msun,
            type_=gf.DistributionType.UNIFORM
        )
        
        inclination_distribution = gf.Distribution(
            min_=0.0,
            max_=np.pi,
            type_=gf.DistributionType.UNIFORM
        )
        
        # For regression pre-training, use higher SNR (as per paper)
        # Paper trains predictor at high SNR first
        if cfg.use_curriculum:
            schedule = cfg.snr_curriculum_schedule or gf.CurriculumSchedule.LINEAR
            snr_curriculum = gf.Curriculum(
                start=gf.Distribution(
                    min_=cfg.snr_curriculum_start_min,
                    max_=cfg.snr_curriculum_start_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                end=gf.Distribution(
                    min_=cfg.snr_min,
                    max_=cfg.snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                num_epochs=cfg.snr_curriculum_num_epochs,
                schedule=schedule,
                verbose=cfg.snr_curriculum_verbose,
                name="SNR"
            )
            scaling_method = gf.ScalingMethod(
                value=snr_curriculum,
                type_=gf.ScalingTypes.SNR
            )
        else:
            scaling_method = gf.ScalingMethod(
                value=gf.Distribution(
                    min_=cfg.snr_min,
                    max_=cfg.snr_max,
                    type_=gf.DistributionType.UNIFORM
                ),
                type_=gf.ScalingTypes.SNR
            )
        
        # For regression we ALWAYS inject (need signals to predict masses)
        waveform_generator = gf.CBCGenerator(
            mass_1_msun=mass_1_distribution,
            mass_2_msun=mass_2_distribution,
            inclination_radians=inclination_distribution,
            scaling_method=scaling_method,
            injection_chance=1.0,  # Always inject for regression
            spin_1_in=(0.0, 0.0, 0.0),
            spin_2_in=(0.0, 0.0, 0.0),
        )
        
        noise_obtainer = gf.NoiseObtainer(
            noise_type=noise_type,
            ifos=gf.IFO.L1
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
                gf.ReturnVariables.ONSOURCE,
                gf.ReturnVariables.OFFSOURCE
            ],
            output_variables=[
                gf.WaveformParameters.MASS_1_MSUN,
                gf.WaveformParameters.MASS_2_MSUN
            ],
            **kwargs
        )
        
        return dataset
    
    @classmethod
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.0001,
        task: str = "classification"
    ) -> keras.Model:
        """
        Compile model with training configuration from paper.
        
        Training Configuration:
        - Optimizer: Adam
        - Loss: Cross-entropy (classification) or MAE (regression)
        
        Args:
            model: Keras model to compile.
            learning_rate: Learning rate.
            task: "classification" or "regression"
        
        Returns:
            Compiled model.
        """
        # Paper: "ADAM method as our learning algorithm"
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if task == "classification":
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Paper: "mean absolute relative error loss function"
            model.compile(
                optimizer=optimizer,
                loss='mae',
                metrics=['mse']
            )
        return model


class GeorgeHuerta2017Deep(ExampleModel):
    """
    George & Huerta 2017 Deep Network (Fig. 6)
    
    4 convolutional layers with dilated convolutions, followed by
    3 fully connected layers. More accurate but 5x slower than shallow.
    
    Size: ~23MB
    Speed: 85ms CPU, 535μs GPU per 1-second input
    
    Usage:
        >>> model = gf.examples.GeorgeHuerta2017Deep.model()
        >>> dataset = gf.examples.GeorgeHuerta2017Deep.dataset()
    
    References:
        D. George & E.A. Huerta, "Deep Neural Networks to Enable 
        Real-time Multimessenger Astrophysics", Phys. Rev. D 97, 044039.
        arXiv:1701.00008v3, Fig. 6
    """
    
    config = GeorgeHuerta2017Config()
    
    @classmethod
    def model(
        cls,
        input_shape_onsource: Optional[Tuple[int, int]] = None,
        input_shape_offsource: Optional[Tuple[int, int]] = None,
        config: Optional[GeorgeHuerta2017Config] = None,
        task: str = "classification"
    ) -> keras.Model:
        """
        Create the George & Huerta 2017 deep CNN architecture.
        
        Architecture from Fig. 6:
        - 4 Convolutional layers with ReLU, dilated convolutions
        - MaxPooling (size 4, stride 4) after each conv
        - 3 Fully connected layers (128, 64, 2)
        - Softmax output for classification, linear for regression
        
        Args:
            input_shape_onsource: Shape of onsource input (samples, channels).
            input_shape_offsource: Shape of offsource input for whitening.
            config: Model configuration. Uses paper defaults if None.
            task: "classification" (2-class softmax) or "regression" (mass prediction)
        
        Returns:
            Keras model ready for compilation.
        """
        cfg = config or cls.config
        
        # Default shapes based on paper specs
        if input_shape_onsource is None:
            num_samples = int(cfg.onsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape_onsource = (num_samples, 1)
        
        if input_shape_offsource is None:
            num_samples = int(cfg.offsource_duration_seconds * cfg.sample_rate_hertz)
            input_shape_offsource = (num_samples, 1)
        
        # Input layers
        onsource_input = Input(shape=input_shape_onsource, name="ONSOURCE")
        offsource_input = Input(shape=input_shape_offsource, name="OFFSOURCE")
        
        # Whitening layer
        x = gf.Whiten(
            sample_rate_hertz=cfg.sample_rate_hertz,
            onsource_duration_seconds=cfg.onsource_duration_seconds
        )([onsource_input, offsource_input])
        
        # Permute to (batch, channels, samples) for Conv1D
        x = Permute((2, 1))(x)
        
        # 4 Convolutional blocks from Fig. 6
        for i, (filters, kernel_size, dilation) in enumerate(cfg.deep_conv_configs):
            layer_num = i + 1
            
            x = Conv1D(
                filters, 
                kernel_size, 
                dilation_rate=dilation,
                name=f"Conv1D_{layer_num}"
            )(x)
            
            # MaxPooling
            x = MaxPooling1D(
                pool_size=cfg.deep_pool_size, 
                strides=cfg.deep_pool_size,
                name=f"MaxPool_{layer_num}"
            )(x)
            
            # ReLU activation
            x = Activation('relu', name=f"ReLU_{layer_num}")(x)
        
        # Flatten
        x = Flatten(name="Flatten")(x)
        
        # FC layers (128, 64)
        for i, units in enumerate(cfg.deep_fc_units):
            layer_num = i + 1
            x = Dense(units, name=f"Dense_{layer_num}")(x)
            x = Activation('relu', name=f"ReLU_FC_{layer_num}")(x)
        
        # Output layer
        if task == "classification":
            output = Dense(cfg.num_output_classes, activation='softmax', 
                          name="output")(x)
        else:
            output = Dense(2, activation='linear', name="output")(x)
        
        model = Model(
            inputs=[onsource_input, offsource_input],
            outputs=output,
            name="GeorgeHuerta2017Deep"
        )
        
        return model
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 7000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset matching George & Huerta 2017 specifications.
        Same configuration as shallow model.
        """
        # Delegate to shallow model's dataset method
        return GeorgeHuerta2017Shallow.dataset(
            noise_type=noise_type,
            config=config,
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )
    
    @classmethod
    def dataset_regression(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 7000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset for mass regression pre-training.
        Same configuration as shallow model.
        """
        return GeorgeHuerta2017Shallow.dataset_regression(
            noise_type=noise_type,
            config=config,
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )
    
    @classmethod
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.001,
        task: str = "classification"
    ) -> keras.Model:
        """
        Compile model with training configuration from paper.
        Same configuration as shallow model.
        """
        return GeorgeHuerta2017Shallow.compile_model(model, learning_rate, task)
