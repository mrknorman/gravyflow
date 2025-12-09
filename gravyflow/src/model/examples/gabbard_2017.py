"""
Gabbard et al. 2017 - Binary Black Hole Detection CNN

Paper: "Matching matched filtering with deep networks for gravitational-wave astronomy"
Authors: H. Gabbard, M. Williams, F. Hayes, C. Messenger
arXiv: 1712.06041v2
DOI: 10.1103/PhysRevLett.120.141103

This module provides the model architecture, dataset configuration, and
training procedure as described in the paper.

Model Architecture (Table I):
- 6 Convolutional layers with ELU activation
- MaxPooling after layers 2, 4, 6 (sizes 8, 6, 4)
- 2 Dense layers with 50% dropout
- 2-neuron Softmax output (signal+noise vs noise-only)

Dataset Configuration (Section "Simulation details"):
- Component masses: 5-95 M☉ (m1 > m2)
- Mass distribution: log(m) prior
- Whitened timeseries, 1 second @ 8192 Hz
- Merger position: random in [0.75, 0.95] fractional range
- SNR evaluated in central 1 sec with Tukey window
- 0.5s padding before and after for edge effects
"""

import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import keras
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout, ELU, Permute, Softmax
)
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel


@dataclass
class Gabbard2017Config:
    """
    Configuration matching Gabbard et al. 2017 paper specifications.
    
    References:
        Table I and Section "Simulation details" of arXiv:1712.06041v2
    """
    # Sampling parameters
    sample_rate_hertz: float = 8192.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    
    # Mass distributions (Paper: "component black hole masses in the range 
    # from 5M☉ to 95M☉, m1 > m2" with "m1,2 ~ log m1,2")
    mass_1_min_msun: float = 5.0
    mass_1_max_msun: float = 95.0
    mass_2_min_msun: float = 5.0
    mass_2_max_msun: float = 95.0
    
    # SNR scaling (Paper: "optimal SNR defined as" eq. 1, values 1-10 in tests)
    snr_min: float = 8.0
    snr_max: float = 15.0
    
    # Merger window (Paper: "peak amplitude randomly positioned within 
    # the fractional range [0.75, 0.95]")
    # Note: SNR window [0.75, 0.95] is for matched-filter evaluation, not training
    merger_window_start_fraction: float = 0.75
    merger_window_end_fraction: float = 0.95
    
    # Training parameters
    injection_chance: float = 0.5  # 50% signal+noise, 50% noise-only
    batch_size: int = 32
    
    # Model architecture from Table I
    # Layer 1: Conv 8 filters, kernel 64
    # Layer 2: Conv 8 filters, kernel 32, MaxPool 8
    # Layer 3: Conv 16 filters, kernel 32
    # Layer 4: Conv 16 filters, kernel 16, MaxPool 6
    # Layer 5: Conv 32 filters, kernel 16
    # Layer 6: Conv 32 filters, kernel 16, MaxPool 4
    # Layer 7: Dense 64, Dropout 0.5
    # Layer 8: Dense 64, Dropout 0.5
    # Layer 9: Dense 2, Softmax
    conv_configs: tuple = (
        (8, 64, None),    # Layer 1: filters, kernel, maxpool
        (8, 32, 8),       # Layer 2
        (16, 32, None),   # Layer 3
        (16, 16, 6),      # Layer 4
        (32, 16, None),   # Layer 5
        (32, 16, 4),      # Layer 6
    )
    dense_units: int = 64
    dropout_rate: float = 0.5
    num_output_classes: int = 2  # Paper uses 2 neurons with Softmax


class Gabbard2017(ExampleModel):
    """
    Gabbard et al. 2017 Binary Black Hole Detection Model
    
    Provides the CNN architecture, dataset configuration, and training
    procedure described in arXiv:1712.06041v2.
    
    Usage:
        >>> model = gf.examples.Gabbard2017.model()
        >>> dataset = gf.examples.Gabbard2017.dataset()
        >>> model = gf.examples.Gabbard2017.pretrained()
    
    References:
        H. Gabbard et al., "Matching matched filtering with deep networks 
        for gravitational-wave astronomy", Phys. Rev. Lett. 120, 141103 (2018).
        arXiv:1712.06041v2
    """
    
    # Default configuration matching paper specifications
    config = Gabbard2017Config()
    
    @classmethod
    def model(
        cls,
        input_shape_onsource: Optional[Tuple[int, int]] = None,
        input_shape_offsource: Optional[Tuple[int, int]] = None,
        config: Optional[Gabbard2017Config] = None
    ) -> keras.Model:
        """
        Create the Gabbard et al. 2017 CNN architecture.
        
        Architecture from Table I:
        - 6 Convolutional layers with ELU activation
        - MaxPooling after layers 2, 4, 6 (sizes 8, 6, 4)
        - 2 Dense layers (64 units each) with 50% dropout
        - 2-neuron Softmax output
        
        Args:
            input_shape_onsource: Shape of onsource input (samples, channels).
            input_shape_offsource: Shape of offsource input for whitening.
            config: Model configuration. Uses paper defaults if None.
        
        Returns:
            Compiled Keras model ready for training.
        """
        cfg = config or cls.config
        
        # Default shapes based on paper specs (1 second @ 8192 Hz, 1 IFO)
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
        # Paper: "whitened simulated gravitational-wave timeseries"
        x = gf.Whiten(
            sample_rate_hertz=cfg.sample_rate_hertz,
            onsource_duration_seconds=cfg.onsource_duration_seconds
        )([onsource_input, offsource_input])
        
        # Permute to (batch, channels, samples) for Conv1D
        x = Permute((2, 1))(x)
        
        # 6 Convolutional blocks from Table I
        for i, (filters, kernel_size, pool_size) in enumerate(cfg.conv_configs):
            layer_num = i + 1
            
            # Convolution with 'same' padding to preserve dimensions
            x = Conv1D(filters, kernel_size, padding='valid', 
                       name=f"Conv1D_{layer_num}")(x)
            x = ELU(name=f"ELU_{layer_num}")(x)
            
            # MaxPooling (only on layers 2, 4, 6)
            if pool_size is not None:
                x = MaxPooling1D(pool_size=pool_size, strides=pool_size, 
                                 padding="same", name=f"MaxPool_{layer_num}")(x)
        
        # Flatten
        x = Flatten(name="Flatten")(x)
        
        # Dense layer 7: 64 units, ELU, Dropout 0.5
        x = Dense(cfg.dense_units, name="Dense_7")(x)
        x = ELU(name="ELU_7")(x)
        x = Dropout(cfg.dropout_rate, name="Dropout_7")(x)
        
        # Dense layer 8: 64 units, ELU, Dropout 0.5
        x = Dense(cfg.dense_units, name="Dense_8")(x)
        x = ELU(name="ELU_8")(x)
        x = Dropout(cfg.dropout_rate, name="Dropout_8")(x)
        
        # Output layer 9: 2 neurons with Softmax
        # Paper: "each neuron gives the inferred probability that the input 
        # data belongs to the noise or signal+noise class"
        output = Dense(cfg.num_output_classes, activation='softmax', 
                       name="output")(x)
        
        model = Model(
            inputs=[onsource_input, offsource_input],
            outputs=output,
            name="Gabbard2017"
        )
        
        return model
    
    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[Gabbard2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 11250,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Create a dataset matching Gabbard et al. 2017 specifications.
        
        Dataset Configuration from paper:
        - Component masses 5-95 M☉ with log(m) distribution
        - m1 > m2 constraint
        - Merger position random in [0.75, 0.95] fractional range
        - 50% signal+noise, 50% noise-only
        
        Args:
            noise_type: Type of noise. Paper used Gaussian.
            config: Dataset configuration. Uses paper defaults if None.
            seed: Random seed for reproducibility.
            group: Dataset group ("train", "validate", "test").
            steps_per_epoch: Number of batches per epoch.
            **kwargs: Additional arguments passed to GravyflowDataset.
        
        Returns:
            GravyflowDataset configured to match paper specifications.
        """
        cfg = config or cls.config
        
        # Mass distributions with log(m) prior
        # Paper: "m1,2 ~ log m1,2"
        mass_1_distribution = gf.Distribution(
            min_=cfg.mass_1_min_msun,
            max_=cfg.mass_1_max_msun,
            type_=gf.DistributionType.LOG
        )
        
        mass_2_distribution = gf.Distribution(
            min_=cfg.mass_2_min_msun,
            max_=cfg.mass_2_max_msun,
            type_=gf.DistributionType.LOG
        )
        
        # Inclination: "cosine of inclination uniform on [-1, 1]"
        # This means cos(iota) ~ Uniform[-1, 1], so iota ~ arccos(U)
        # Gravyflow doesn't have this, use uniform in [0, pi] as approximation
        inclination_distribution = gf.Distribution(
            min_=0.0,
            max_=np.pi,
            type_=gf.DistributionType.UNIFORM
        )
        
        # SNR scaling
        scaling_method = gf.ScalingMethod(
            value=gf.Distribution(
                min_=cfg.snr_min,
                max_=cfg.snr_max,
                type_=gf.DistributionType.UNIFORM
            ),
            type_=gf.ScalingTypes.SNR
        )
        
        # Waveform generator
        # Paper: "IMRPhenomD type waveform" with "zero spin"
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
    def compile_model(
        cls,
        model: keras.Model,
        learning_rate: float = 0.002
    ) -> keras.Model:
        """
        Compile model with training configuration from paper.
        
        Training Configuration:
        - Optimizer: Nadam (Adam with Nesterov momentum)
        - Learning rate: 0.002
        - Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8
        - Loss: Categorical cross-entropy (for 2-class softmax)
        
        Args:
            model: Keras model to compile.
            learning_rate: Learning rate (paper uses 0.002).
        
        Returns:
            Compiled model.
        """
        # Paper: "adaptive moment estimation with incorporated Nesterov 
        # momentum with a learning rate of 0.002, β1 = 0.9, β2 = 0.999, 
        # ε = 10−8"
        optimizer = keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Paper uses binary cross-entropy (Eq. 2) but with 2-class softmax
        # This is equivalent to categorical cross-entropy with 2 classes
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @classmethod
    def pretrained(cls) -> keras.Model:
        """
        Load pre-trained Gabbard2017 model.
        
        Args:
            weights_path: Path to weights file. If None, uses default location.
        
        Returns:
            Model with pre-trained weights loaded.
        
        Raises:
            FileNotFoundError: If weights file not found.
        """
        model = cls.model()
        
        model_path = Path(__file__).parent.parent.parent.parent / "res" / "model" / "gabbard_2017.h5"
        
        
        keras.models.load_model(str(model_path))
        return model
