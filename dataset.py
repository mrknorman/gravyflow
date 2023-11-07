from typing import List, Tuple, Union, Dict, Any
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass
from warnings import warn

import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.data import Options

import gravyflow as gf

def validate_noise_settings(
        noise_obtainer: gf.NoiseObtainer, 
        variables_to_return: List[
            Union[
                gf.gf.WaveformParameters, 
                gf.ReturnVariables
            ]
        ]
    ) -> None:
    """
    Validates noise settings and emits warnings or errors as appropriate.
    
    Parameters
    ----------
    noise_obtainer : gf.NoiseObtainer
        The object containing noise settings.
    variables_to_return : Union[List[str], List[Enum]]
        List of variables expected to be returned.
        
    Raises
    ------
    ValueError
        If incompatible settings are found.
    """
    
    # Validate noise type and ifo_data_obtainer
    if noise_obtainer.noise_type in [gf.NoiseType.WHITE, gf.NoiseType.COLORED]:
        
        if noise_obtainer.ifo_data_obtainer is not None:
            warn(
                ("Noise is not REAL or PSEUDO-REAL, yet data obtainer is"
                 " defined."), 
                UserWarning
            )
            
        if gf.ReturnVariables.GPS_TIME in variables_to_return:
            warn(
                "Cannot return GPS time from simulated Noise defaulting to -1",
                UserWarning
            )
    
    # Validate whitening for white noise
    if noise_obtainer.noise_type is gf.NoiseType.WHITE:
    
        if gf.ReturnVariables.WHITENED_INJECTIONS in variables_to_return or \
           gf.ReturnVariables.WHITENED_ONSOURCE in variables_to_return:
            
            warn(
                "Whitening requested for WHITE NOISE.", 
                UserWarning
            )
            
def get_max_arrival_time_difference(
    injection_generators : List
    ) -> float:
    
    max_arival_time_difference_seconds : float = 0.01
    
    max_arrival_time_differences = []
    for config in injection_generators: 
        if (config is not None) and (config.network is not None):
            max_arrival_time_differences.append(
                config.network.max_arrival_time_difference_seconds
            )        
    
    if len(max_arrival_time_differences):
        max_arrival_time_differences = tf.stack(max_arrival_time_differences)
        max_arival_time_difference_seconds = tf.reduce_max(
            max_arrival_time_differences
        )
    
    return max_arival_time_difference_seconds

def data(    
        # Random Seed:
        seed: int = None,
        # Temporal components:
        sample_rate_hertz: float = None,   
        onsource_duration_seconds: float = None,
        offsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        # Scale factor:
        scale_factor : float = None,
        # Noise: 
        noise_obtainer : gf.NoiseObtainer = None,
        group : str = "train",
        # Injections:
        injection_generators: List[
            Union[gf.cuPhenomDGenerator, gf.WNBGenerator]
        ] = None, 
        num_examples_per_generation_batch : int = None,
        # Output configuration:
        num_examples_per_batch: int = None,
        input_variables : List[
            Union[gf.gf.WaveformParameters, gf.ReturnVariables]
        ] = None,
        output_variables : List[
            Union[gf.gf.WaveformParameters, gf.ReturnVariables]
        ] = None
    ):

    if seed is None:
        seed = gf.Defaults.seed
    if sample_rate_hertz is None:
        sample_rate_hertz = gf.Defaults.sample_rate_hertz
    if onsource_duration_seconds is None:
        onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
    if offsource_duration_seconds is None:
        offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
    if crop_duration_seconds is None:
        crop_duration_seconds = gf.Defaults.crop_duration_seconds
    if scale_factor is None:
        scale_factor = gf.Defaults.scale_factor
    if num_examples_per_generation_batch is None:
        num_examples_per_generation_batch = gf.Defaults.num_examples_per_generation_batch
    if num_examples_per_batch is None:
        num_examples_per_batch = gf.Defaults.num_examples_per_batch
    
    # Set gf.Defaults here as if initilised as default arguments objects are global
    if noise_obtainer is None:
        noise_obtainer = gf.NoiseObtainer()
        
    if input_variables is None:
        input_variables = []
        
    if output_variables is None:
        output_variables = []
        
    if injection_generators is None:
        injection_generators = []
                
    if not isinstance(injection_generators, list):
        injection_generators = [injection_generators]
    
    # Create set with unique elements of input and output variables so that they
    # can be calculated during loop if required:
    variables_to_return = set(input_variables + output_variables)
    
    if not variables_to_return:
        raise ValueError("No return variables requested. What's the point?")

    validate_noise_settings(noise_obtainer, variables_to_return)
    
    # Set random seeds for Tensorflow and Numpy to ensure deterministic results
    # with the same seed. This means that if the seed is the concerved the
    # dataset produced will be identical:
    gf.set_random_seeds(seed)
    
    # Create Noise Generator:
    noise : Iterator = \
        noise_obtainer(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            scale_factor,
            group
        )
    
    # Create Injection Generator: 
    waveform_parameters_to_return = [
        item for item in variables_to_return if isinstance(
            item.value, gf.WaveformParameter
        )
    ]
    
    injection_generator : gf.InjectionGenerator = \
        gf.InjectionGenerator(
            injection_generators,
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            num_examples_per_generation_batch,
            num_examples_per_batch,
            variables_to_return=waveform_parameters_to_return
        )
    
    injections : Iterator = injection_generator.generate()
    
    whitened_injections = None
    for (onsource, offsource, gps_times), (injections_, mask, parameters) \
        in zip(noise, injections):
                
        if len(injection_generators):
                        
            # Add injections to waveform scaled by inputted SNR config values:
            onsource, scaled_injections, scaling_parameters = \
                injection_generator.add_injections_to_onsource(
                    injections_,
                    mask,
                    onsource,
                    variables_to_return=variables_to_return
                ) 
            
            for key, value in scaling_parameters.items():
                if key in variables_to_return:
                    parameters[key] = value
            
            if gf.ReturnVariables.WHITENED_INJECTIONS in variables_to_return:
                                
                whitened_injections = tf.stack([
                        gf.whiten(
                            scaled_injection_, 
                            offsource, 
                            sample_rate_hertz, 
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5,
                            filter_duration_seconds=1.0
                        ) for scaled_injection_ in scaled_injections
                    ])
                                
                whitened_injections = gf.replace_nan_and_inf_with_zero(
                    whitened_injections
                )

        else:
            scaled_injections = None
                
        # Whiten data: 
        if (gf.ReturnVariables.WHITENED_ONSOURCE in variables_to_return) or \
        (gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in variables_to_return) \
        or (gf.ReturnVariables.SPECTROGRAM_ONSOURCE in variables_to_return):
            
            whitened_onsource = gf.whiten(
                onsource, 
                offsource, 
                sample_rate_hertz, 
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5,
                filter_duration_seconds=1.0
            )
            
            # Crop to remove edge effects, crop with or without whitening to
            # ensure same data is retrieve in both cases
            whitened_onsource = gf.crop_samples(
                whitened_onsource, 
                onsource_duration_seconds, 
                sample_rate_hertz
            )
            
            tf.debugging.check_numerics(
                whitened_onsource, 
                f"NaN detected in whitened_onsource after cast."
            )
            
            if (
                gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE \
                in variables_to_return
            ):
                max_arival_time_difference_seconds: float = \
                    get_max_arrival_time_difference(injection_generators)
                
                rolling_pearson_onsource = gf.rolling_pearson(
                    whitened_onsource,
                    max_arival_time_difference_seconds,
                    sample_rate_hertz
                )
            else:
                rolling_pearson_onsource = None
                
            if (gf.ReturnVariables.SPECTROGRAM_ONSOURCE in variables_to_return):

                spectrogram_onsource = gf.spectrogram(whitened_onsource)
            else:
                spectrogram_onsource = None
                
            whitened_onsource = tf.cast(whitened_onsource, tf.float16)
            whitened_onsource = gf.replace_nan_and_inf_with_zero(
                whitened_onsource
            )

        else:
            whitened_onsource = None
            rolling_pearson_onsource = None
            spectrogram_onsource = None
        
        if gf.ReturnVariables.ONSOURCE in variables_to_return:
            onsource = tf.cast(onsource, tf.float16)
            onsource = gf.replace_nan_and_inf_with_zero(onsource)

            tf.debugging.check_numerics(
                onsource, 
                f"NaN detected in onsource after cast."
            )
            
        if gf.ReturnVariables.OFFSOURCE in variables_to_return:
            offsource = tf.cast(offsource, tf.float16)
            offsource = gf.replace_nan_and_inf_with_zero(offsource)

            tf.debugging.check_numerics(
                offsource, 
                f"NaN detected in offsource after cast."
            )
            
        if gf.ReturnVariables.GPS_TIME in variables_to_return:
            gps_times = tf.cast(gps_times, tf.float64)
            
        if gf.ReturnVariables.INJECTION_MASKS in variables_to_return:
            mask = tf.cast(mask, tf.float32)
                
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
                rolling_pearson_onsource,
                spectrogram_onsource,
                parameters
            ) for var_list in [input_variables, output_variables]
        ]
                
        yield (input_dict, output_dict)

def create_variable_dictionary(
    return_variables: List[Union[gf.ReturnVariables, gf.gf.WaveformParameters]],
    onsource : tf.Tensor,
    whitened_onsource : tf.Tensor,
    offsource : tf.Tensor,
    gps_times : tf.Tensor,
    injections : tf.Tensor,
    whitened_injections : tf.Tensor,
    mask : tf.Tensor,
    rolling_pearson_onsource : tf.Tensor,
    spectrogram_onsource : tf.Tensor,
    injection_parameters : Dict
    ) -> Dict:

    operations = {
        gf.ReturnVariables.ONSOURCE: onsource,
        gf.ReturnVariables.WHITENED_ONSOURCE: whitened_onsource,
        gf.ReturnVariables.OFFSOURCE: offsource,
        gf.ReturnVariables.GPS_TIME: gps_times,
        gf.ReturnVariables.INJECTIONS: injections,
        gf.ReturnVariables.WHITENED_INJECTIONS: whitened_injections,
        gf.ReturnVariables.INJECTION_MASKS: mask,
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE: rolling_pearson_onsource,
        gf.ReturnVariables.SPECTROGRAM_ONSOURCE: spectrogram_onsource
    }

    # Extend operations with any relevant keys from injection_parameters
    operations.update(
        {
            key: value for key, value in injection_parameters.items() \
            if key in return_variables
        }
    )

    return {
        key.name: operations[key] for key in return_variables \
        if key in operations
    }

def Dataset(
        seed: int = None,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        offsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        scale_factor: float = None,
        noise_obtainer: gf.NoiseObtainer = None,
        group : str = "train",
        injection_generators: List[
            Union[gf.cuPhenomDGenerator, gf.WNBGenerator]
        ] = None,
        num_examples_per_generation_batch: int = None,
        num_examples_per_batch: int = None,
        input_variables: List = None,
        output_variables: List = None
    ) -> tf.data.Dataset:
    
    """
    Generates a TensorFlow dataset with Interferometer data.
    
    Parameters:
        seed (int): 
            Random seed.
        sample_rate_hertz (float): 
            Sample rate in Hz.
        onsource_duration_seconds (float): 
            On-source duration in seconds.
        offsource_duration_seconds (float):
            Off-source duration in seconds.
        crop_duration_seconds (float):
            Crop duration in seconds.
        scale_factor (float):
            Scale factor.
        noise_obtainer (gf.NoiseObtainer): 
            Object to obtain noise.
        injection_generators (list):
            List of injection generators.
        num_examples_per_generation_batch (int):
            Number of examples per generation batch.
        num_examples_per_batch (int):
            Number of examples per batch.
        input_variables (list):
            List of input variables.
        output_variables (list):
            List of output variables.
    
    Returns:
        tf.data.Dataset: TensorFlow Dataset object.
    """

    if seed is None:
        seed = gf.Defaults.seed
    if sample_rate_hertz is None:
        sample_rate_hertz = gf.Defaults.sample_rate_hertz
    if onsource_duration_seconds is None:
        onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
    if offsource_duration_seconds is None:
        offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
    if crop_duration_seconds is None:
        crop_duration_seconds = gf.Defaults.crop_duration_seconds
    if scale_factor is None:
        scale_factor = gf.Defaults.scale_factor
    if num_examples_per_generation_batch is None:
        num_examples_per_generation_batch = gf.Defaults.num_examples_per_generation_batch
    if num_examples_per_batch is None:
        num_examples_per_batch = gf.Defaults.num_examples_per_batch
    
    if input_variables is None:
        input_variables = []
        
    if output_variables is None:
        output_variables = []
    
    # Set gf.Defaults here as if initilised as default arguments objects are global
    if injection_generators is None:
        injection_generators = []
    elif not isinstance(injection_generators, list):
        injection_generators = [injection_generators]

    num_cropped_samples = int(onsource_duration_seconds * sample_rate_hertz)
    num_onsource_samples = int((onsource_duration_seconds + 2*crop_duration_seconds) * sample_rate_hertz)
    num_offsource_samples = int(offsource_duration_seconds * sample_rate_hertz)
    num_injection_configs = len(injection_generators)
    
    num_detectors = 1
    if injection_generators: 
        if injection_generators[0].network is not None:
            num_detectors = injection_generators[0].network.num_detectors 
        else:
            num_detectors = 1
    elif noise_obtainer is not None:
        num_detectors = len(noise_obtainer.ifos)
    
    if num_detectors is None:
        num_detectors = 1
        
    max_arival_time_difference_seconds: float = \
        get_max_arrival_time_difference(injection_generators)

    max_arival_time_difference_samples : float = \
        int(max_arival_time_difference_seconds * sample_rate_hertz)
    
    if num_detectors == 1:
        onsource_shape = (num_examples_per_batch, num_onsource_samples)
        cropped_shape = (num_examples_per_batch, num_cropped_samples)
        offsource_shape = (num_examples_per_batch, num_offsource_samples)
        detectors_shape = (num_examples_per_batch,)
        injections_shape = (
                num_injection_configs, 
                num_examples_per_batch, 
                num_cropped_samples
            )
        per_injection_shape = (num_injection_configs, num_examples_per_batch)
        pearson_shape = (
            num_examples_per_batch, 
            max_arival_time_difference_samples
        )
    else:
        onsource_shape = (
            num_examples_per_batch, num_detectors, num_onsource_samples
        )
        cropped_shape = (
            num_examples_per_batch, num_detectors, num_cropped_samples
        )
        offsource_shape = (
            num_examples_per_batch, num_detectors, num_offsource_samples
        )
        detectors_shape = (
            num_examples_per_batch, num_detectors
        )
        injections_shape = (
                num_injection_configs, 
                num_examples_per_batch,
                num_detectors,
                num_cropped_samples
            )
        per_injection_shape = (
            num_injection_configs, num_examples_per_batch
        )        
        pearson_shape = (
            num_examples_per_batch, 
            num_detectors * (num_detectors - 1) // 2,
            2*max_arival_time_difference_samples
        )
        
    spectrogram_shape = gf.spectrogram_shape(cropped_shape)

    output_signature_dict = {
        gf.ReturnVariables.ONSOURCE.name:
            tf.TensorSpec(
                shape=onsource_shape, 
                dtype=tf.float16
            ),
        gf.ReturnVariables.WHITENED_ONSOURCE.name: 
            tf.TensorSpec(
                shape=cropped_shape,
                dtype=tf.float16
            ),
        gf.ReturnVariables.OFFSOURCE.name: 
            tf.TensorSpec(
                shape=offsource_shape, 
                dtype=tf.float16
            ),
        gf.ReturnVariables.GPS_TIME.name: 
            tf.TensorSpec(
                shape=detectors_shape, 
                dtype=tf.int64
            ),
        gf.ReturnVariables.INJECTIONS.name: 
            tf.TensorSpec(
                shape=injections_shape,
                dtype=tf.float16
            ),
        gf.ReturnVariables.WHITENED_INJECTIONS.name: 
            tf.TensorSpec(
                shape=injections_shape,
                dtype=tf.float16
            ),
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE.name:
            tf.TensorSpec(
                shape=pearson_shape,
                dtype=tf.float32
            ), 
        gf.ReturnVariables.SPECTROGRAM_ONSOURCE.name: 
            tf.TensorSpec(
                shape=spectrogram_shape, 
                dtype=tf.float32
            ),
        gf.ReturnVariables.INJECTION_MASKS.name: 
            tf.TensorSpec(
                shape=per_injection_shape, 
                dtype=tf.float32
            )
    }
    
    parameters_to_return = {
        item for item in (input_variables + output_variables) if \
        (isinstance(item.value, gf.WaveformParameter) or isinstance(
            item.value, gf.ScalingType)
        )
    }
    
    if not injection_generators:
        keys_to_remove = {
            gf.ReturnVariables.INJECTION_MASKS, 
            gf.ReturnVariables.INJECTIONS, 
            gf.ReturnVariables.WHITENED_INJECTIONS
        }

        parameters_to_return -= keys_to_remove
        input_variables = [
            item for item in input_variables if item not in keys_to_remove
        ]
        output_variables = [
            item for item in output_variables if item not in keys_to_remove
        ]
    
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

    generator = lambda: data(
        seed=seed,
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        scale_factor=scale_factor,
        noise_obtainer=noise_obtainer,
        group=group,
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
        # Calculate the range of global indices for this batch:
        start_index = batch_index * num_examples_per_batch
        end_index = (batch_index + 1) * num_examples_per_batch
        
        # Find the worst examples in the current batch:
        while current_index < len(indicies) and \
            indicies[current_index] < end_index:
                        
            # Calculate in-batch index:
            in_batch_index = indicies[current_index] % num_examples_per_batch  
                                    
            # Extract the corresponding data from in_dict and out_dict using 
            # in_batch_index:
            example_element = \
            {
                key: value[in_batch_index[0]] for key, value in in_dict.items()
            }
                        
            out_element = \
            {
                key: value[0][
                    in_batch_index[0]
                ] for key, value in out_dict.items()
            }
                        
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
    return dataset(**args).take(num_batches)