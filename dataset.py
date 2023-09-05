from typing import List, Tuple, Union, Dict, Any
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass

import tensorflow as tf

from .noise import NoiseObtainer
from .injection import (cuPhenomDGenerator, InjectionGenerator, 
                        WaveformParameters, WNBGenerator, WaveformParameter,
                        ReturnVariable, ReturnVariables)
from .maths import expand_tensor, Distribution, set_random_seeds, crop_samples
from .whiten import whiten

def get_ifo_data(    
        # Random Seed:
        seed: int = 1000,
        # Temporal components:
        sample_rate_hertz: float = 2048.0,   
        onsource_duration_seconds: float = 1.0,
        offsource_duration_seconds: float = 16.0,
        crop_duration_seconds: float = 0.5,
        # Scale factor:
        scale_factor : float = 1.0E21,
        # Noise: 
        noise_obtainer : NoiseObtainer = None ,
        # Injections:
        injection_generators: List[Union[cuPhenomDGenerator, WNBGenerator]] = None, 
        num_examples_per_generation_batch : int = 2048,
        # Output configuration:
        num_examples_per_batch: int = 1,
        input_variables : List = None,
        output_variables : List = None
    ):
    
    # Set defaults here as if initilised as default arguments objects are global
    if not noise_obtainer:
        noise_obtainer = NoiseObtainer()
        
    if not input_variables:
        input_variables = []
        
    if not output_variables:
        output_variables = []
        
    # Create set with unique elements of input and output variables so that they
    # can be calculated during loop if required:
    variables_to_return = set(input_variables + output_variables)
    
    if not variables_to_return:
        raise ValueError("No return variables requested. What's the point?")
    
    # Set random seeds for Tensorflow and Numpy to ensure deterministic results
    # with the same seed. This means that if the seed is the concerved the
    # dataset produced will be identical:
    set_random_seeds(seed)
    
    # Create Noise Generator:
    noise : Iterator = \
        noise_obtainer.init_generator(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor
        )
    
    # Create Injection Generator    
    waveform_parameters_to_return = [
        item for item in variables_to_return if isinstance(item.value, WaveformParameter)
    ]
    
    injection_generator : InjectionGenerator = \
        InjectionGenerator(
            injection_generators,
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            scale_factor,
            variables_to_return=waveform_parameters_to_return
        )
    
    injections : Iterator = injection_generator.generate()
    
    whitened_injections = None
    amplitudes = None
    snrs = None
    for (onsource, offsource, gps_times), (injections_, mask, parameters) \
        in zip(noise, injections):
                
        if injections_ is not None:
            # Add injections to waveform scaled by inputted SNR config values:
            onsource, scaled_injections, amplitudes, snrs = \
                injection_generator.add_injections_to_onsource(
                    injections_,
                    mask,
                    onsource,
                    variables_to_return=variables_to_return
                ) 
            
            if ReturnVariables.WHITENED_INJECTIONS in variables_to_return:
                whitened_injections = \
                    tf.stack([
                        whiten(
                            scaled_injection_, 
                            offsource, 
                            sample_rate_hertz, 
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5,
                            filter_duration_seconds=1.0
                        ) for scaled_injection_ in scaled_injections
                    ])
        else:
            scaled_injections = None
                
        # Whiten data: 
        if ReturnVariables.WHITENED_ONSOURCE in variables_to_return:
            whitened_onsource = \
                whiten(
                    onsource, 
                    offsource, 
                    sample_rate_hertz, 
                    fft_duration_seconds=1.0,
                    overlap_duration_seconds=0.5,
                    filter_duration_seconds=1.0
                )
            # Crop to remove edge effects, crop with or without whitening to
            # ensure same data is retrieve in both cases
            whitened_onsource = crop_samples(
                whitened_onsource, 
                onsource_duration_seconds, 
                sample_rate_hertz
            )
        
        else:
            whitened_onsource = None
        
        if ReturnVariables.ONSOURCE in variables_to_return:
            # Crop to remove edge effects, crop with or without whitening to
            # ensure same data is retrieve in both cases
            onsource = crop_samples(
                onsource, 
                onsource_duration_seconds, 
                sample_rate_hertz
            )
                
        # Construct dictionary:
        input_dict = create_variable_dictionary(
            input_variables,
            onsource,
            whitened_onsource,
            offsource,
            gps_times,
            scaled_injections,
            whitened_injections,
            mask,
            snrs,
            amplitudes,
            parameters
        )
        output_dict = create_variable_dictionary(
            output_variables,
            onsource,
            whitened_onsource,
            offsource,
            gps_times,
            scaled_injections,
            whitened_injections,
            mask,
            snrs,
            amplitudes,
            parameters
        )
        
        yield (input_dict, output_dict)

def create_variable_dictionary(
    return_variables: List[Union[ReturnVariables, WaveformParameters]],
    onsource : tf.Tensor,
    whitened_onsource : tf.Tensor,
    offsource : tf.Tensor,
    gps_times : tf.Tensor,
    injections : tf.Tensor,
    whitened_injections : tf.Tensor,
    mask : tf.Tensor,
    snrs : tf.Tensor,
    amplitudes : tf.Tensor,
    injection_parameters : Dict
    ):

    operations = {
        ReturnVariables.ONSOURCE: lambda: tf.cast(onsource, tf.float16),
        ReturnVariables.WHITENED_ONSOURCE: lambda: tf.cast(whitened_onsource, tf.float16),
        ReturnVariables.OFFSOURCE: lambda: tf.cast(offsource, tf.float16),
        ReturnVariables.GPS_TIME: lambda: tf.cast(gps_times, tf.float64),
        ReturnVariables.INJECTIONS: lambda: injections,
        ReturnVariables.WHITENED_INJECTIONS: lambda: whitened_injections,
        ReturnVariables.INJECTION_MASKS: lambda: mask,
        ReturnVariables.SNR: lambda: snrs,
        ReturnVariables.AMPLITUDE: lambda: amplitudes
    }
    
    for key, value in injection_parameters.items():
        if key in return_variables:
            operations[key] = lambda key=key: injection_parameters[key]
            
    return_dict = {key: operations[key]() for key in return_variables if key in operations}
        
    return {key: operations[key]() for key in return_variables if key in operations}