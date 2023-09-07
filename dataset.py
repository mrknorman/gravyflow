from typing import List, Tuple, Union, Dict, Any
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.data import Options

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
    if noise_obtainer is None:
        noise_obtainer = NoiseObtainer()
        
    if input_variables is None:
        input_variables = []
        
    if output_variables is None:
        output_variables = []
        
    if injection_generators is None:
        injection_generators = []
        
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
                
        if len(injection_generators):
            
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
            whitened_onsource = tf.cast(whitened_onsource, tf.float16)
        
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
            onsource = tf.cast(onsource, tf.float16)
            
        if ReturnVariables.OFFSOURCE in variables_to_return:
            offsource = tf.cast(offsource, tf.float16)
            
        if ReturnVariables.GPS_TIME in variables_to_return:
            gps_times = tf.cast(gps_times, tf.float64),
                
        # Construct dictionary:
        input_dict, output_dict = [
            create_variable_dictionary(
                var_list,
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
            ) for var_list in [input_variables, output_variables]
        ]
                
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
        ReturnVariables.ONSOURCE: onsource,
        ReturnVariables.WHITENED_ONSOURCE: whitened_onsource,
        ReturnVariables.OFFSOURCE: offsource,
        ReturnVariables.GPS_TIME: gps_times,
        ReturnVariables.INJECTIONS: injections,
        ReturnVariables.WHITENED_INJECTIONS: whitened_injections,
        ReturnVariables.INJECTION_MASKS: mask,
        ReturnVariables.SNR: snrs,
        ReturnVariables.AMPLITUDE: amplitudes
    }

    # Extend operations with any relevant keys from injection_parameters
    operations.update({key: value for key, value in injection_parameters.items() if key in return_variables})

    return {key.name: operations[key] for key in return_variables if key in operations}

def get_ifo_dataset(
        seed: int = 1000,
        sample_rate_hertz: float = 2048.0,
        onsource_duration_seconds: float = 1.0,
        offsource_duration_seconds: float = 16.0,
        crop_duration_seconds: float = 0.5,
        scale_factor: float = 1.0E21,
        noise_obtainer: NoiseObtainer = None,
        injection_generators: List[Union[cuPhenomDGenerator, WNBGenerator]] = None,
        num_examples_per_generation_batch: int = 2048,
        num_examples_per_batch: int = 1,
        input_variables: List = None,
        output_variables: List = None
) -> tf.data.Dataset:
    """
    Generates a TensorFlow dataset with Interferometer data.
    
    Parameters:
        seed (int): Random seed.
        sample_rate_hertz (float): Sample rate in Hz.
        onsource_duration_seconds (float): On-source duration in seconds.
        offsource_duration_seconds (float): Off-source duration in seconds.
        crop_duration_seconds (float): Crop duration in seconds.
        scale_factor (float): Scale factor.
        noise_obtainer (NoiseObtainer): Object to obtain noise.
        injection_generators (list): List of injection generators.
        num_examples_per_generation_batch (int): Number of examples per generation batch.
        num_examples_per_batch (int): Number of examples per batch.
        input_variables (list): List of input variables.
        output_variables (list): List of output variables.
    
    Returns:
        tf.data.Dataset: TensorFlow Dataset object.
    """
    
    if input_variables is None:
        input_variables = []
        
    if output_variables is None:
        output_variables = []
    
    # Set defaults here as if initilised as default arguments objects are global
    if injection_generators is None:
        injection_generators = []
    elif not isinstance(injection_generators, list):
        injection_generators = [injection_generators]

    num_onsource_samples = int(onsource_duration_seconds * sample_rate_hertz)
    num_offsource_samples = int(offsource_duration_seconds * sample_rate_hertz)
    num_injection_configs = len(injection_generators)

    output_signature_dict = {
        ReturnVariables.ONSOURCE.name:
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_onsource_samples), 
                dtype=tf.float16
            ),
        ReturnVariables.WHITENED_ONSOURCE.name: 
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_onsource_samples),
                dtype=tf.float16
            ),
        ReturnVariables.OFFSOURCE.name: 
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_offsource_samples), 
                dtype=tf.float16
            ),
        ReturnVariables.GPS_TIME.name: 
            tf.TensorSpec(
                shape=(num_examples_per_batch,), 
                dtype=tf.int64
            ),
        ReturnVariables.INJECTIONS.name: 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch, 
                    num_onsource_samples
                ),
                dtype=tf.float16
            ),
        ReturnVariables.WHITENED_INJECTIONS.name: 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch, 
                    num_onsource_samples
                ),
                dtype=tf.float16
            ),
        ReturnVariables.INJECTION_MASKS.name: 
            tf.TensorSpec(
                shape=(num_injection_configs, num_examples_per_batch), 
                dtype=tf.bool
            ),
        ReturnVariables.SNR.name:
            tf.TensorSpec(
                shape=(num_injection_configs, num_examples_per_batch), 
                dtype=tf.float64
            ),
        ReturnVariables.AMPLITUDE.name: 
            tf.TensorSpec(
                shape=(num_injection_configs, num_examples_per_batch), 
                dtype=tf.float64
            )
    }
    
    parameters_to_return = {
        item for item in (input_variables + output_variables) if \
        isinstance(item.value, WaveformParameter)
    }

    output_signature_dict.update({
        item.name: tf.TensorSpec(
            shape=(
                num_injection_configs, 
                num_examples_per_batch * item.value.shape[-1]
            ),
            dtype=tf.float32
        ) for item in parameters_to_return
    })

    output_signature = (
        {k.name: output_signature_dict[k.name] for k in input_variables},
        {k.name: output_signature_dict[k.name] for k in output_variables}
    )

    generator = lambda: get_ifo_data(
        seed=seed,
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        scale_factor=scale_factor,
        noise_obtainer=noise_obtainer,
        injection_generators=injection_generators,
        num_examples_per_generation_batch=num_examples_per_generation_batch,
        num_examples_per_batch=num_examples_per_batch,
        input_variables=input_variables,
        output_variables=output_variables
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA

    return tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=output_signature
    ).with_options(options)

def extract_data_from_indicies(
        dataset : tf.data.Dataset,
        indicies : list, 
        num_examples_per_batch : int
    ) -> list:
    
    indicies : List = sorted(indicies) 
    
    dataset_elements : List = []
    current_index : int = 0
    for batch_index, (in_dict, out_dict) in enumerate(dataset):
        # Calculate the range of global indices for this batch
        start_index = batch_index * num_examples_per_batch
        end_index = (batch_index + 1) * num_examples_per_batch

        # Find the worst examples in the current batch
        while current_index < len(indicies) and \
            indicies[current_index] < end_index:
            
            # Calculate in-batch index
            in_batch_index = indicies[current_index] % num_examples_per_batch  
            
            # Extract the corresponding data from in_dict and out_dict using 
            # in_batch_index
            example_element = \
                {key: value[in_batch_index] for key, value in in_dict.items()}
            out_element = \
                {key: value[0][in_batch_index] for key, value in out_dict.items()}
            
            for key, value in out_element.items():
                example_element[key] = value
            
            dataset_elements.append(example_element)

            current_index += 1  # Move to the next worst index
            
    return dataset_elements

def group_split_dataset(
    generator_args : dict,
    group_name : str,
    num_examples : int
    ):
    
    num_batches = num_examples//generator_args["num_examples_per_batch"]
    
    args = generator_args.copy()
    args.update({"group_name" : group_name})
    return get_ifo_dataset(**args).take(num_batches)