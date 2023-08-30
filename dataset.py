from .noise import NoiseObtainer
from typing import List, Tuple, Union, Dict, Any

def get_ifo_data(    
        # Random Seed:
        seed: int = 1000,
        # Temporal components:
        sample_rate_hertz: float = 2048.0,   
        onsource_duration_seconds: float = 1.0,
        offsource_duarion_seconds: float = 16.0,
        maximum_segment_duration_seconds : float = 2048.0,
        window_duration_seconds : float = 1.0,
        # Noise: 
        noise_obtainer : NoiseObtainer = None ,
        # Injections:
        injection_configs: List = None, 
        injection_scale_factor: float = 1.0E20,
        # Data conditioning:
        apply_whitening: bool = False,
        # Outpu configuration:
        num_examples_per_batch: int = 1,
        input_keys : List[str] = None,
        output_keys : List[str] = None
    ):
    
    # Set defaults here as if initilised as default arguments objects are global
    if not noise_obtainer:
        noise_obtainer = NoiseObtainer()
        
    if not injection_configs:
        injection_configs = []
        
    if not input_keys:
        input_keys = []
        
    if not output_keys:
        output_keys = []
    
    # Ensure variable types:
    data_directory = Path(data_directory)
    
    # Ensure output directory exists and is of expected type:
    ensure_directory_exists(data_directory)
    
    # Set random seeds for Tensorflow and Numpy to ensure deterministic results
    # with the same seed. This means that if the seed is the concerved the
    # dataset produced will be identical:
    set_random_seeds(seed)

    # Setup noise config:
    noise_obtainer.setup_noise()

    '''
    # Get segments which contain gravityspy glitches:
    glitch_segments = get_all_glitch_segments(ifo)

    # Convert segments list to tensors
    glitch_segments_start, glitch_segments_end = zip(*glitch_segments)
    glitch_segments_start = \
        tf.constant(glitch_segments_start, dtype=tf.float32)
    glitch_segments_end = tf.constant(glitch_segments_end, dtype=tf.float32)
    '''
    
    # Covert the onsource_duration parameter into TensorFlow constant for use
    # in subsequent tensorflow calculations:
    onsource_duration_seconds_tf = \
        tf.constant(onsource_duration_seconds, dtype=tf.float32)
    
    noise = \
        noise_obtainer.get_noise(
            sample_rate_hertz,
            onsource_duration_seconds,
            num_examples_per_batch,
            scale_factor = 1.0
        )
    
    #injections = \
    
    for onsource, offsource, gps_times in noise:
        print(onsource.numpy())

        quit()
        
"""
    example_index = 0
    with open_hdf5_file(segment_filename) as segment_file:
            
            # Set common parameters before entering the loop
            common_args = {
                "sample_rate_hertz": \
                    {
                        "value": sample_rate_hertz, 
                         "distribution_type": "constant"
                    },
                "duration_seconds": \
                    {
                        "value": onsource_duration_seconds + fduration, 
                         "distribution_type": "constant"
                    }
            }
            
            #Generate injections:
            injections = []
            injection_masks = []
            injection_parameters = []
            snrs = []
            for config in injection_configs:    
    
                injections_, injection_parameters_, injection_masks_ = \
                    generate_injections(
                        config,
                        current_max_batch_count*num_examples_per_batch, 
                        common_args,
                        fduration,
                        sample_rate_hertz,
                        onsource_duration_seconds,
                        num_examples_per_batch
                    )
                
                length_one_args = ['num_waveforms', 'sample_rate_hertz', 'duration_seconds']
                length_three_args = ['spin_1_in', 'spin_2_in']
            
                _injection_parameters = {
                    key: value for key, value in injection_parameters_.items() 
                    if key in input_keys + output_keys
                }
                
                reduced_injection_params = {}
                for key, parameter in _injection_parameters.items():
                    if key not in length_one_args + length_three_args:    
                        parameter = tf.convert_to_tensor(parameter)

                        parameter = expand_tensor(
                            parameter, 
                            injection_masks
                        )

                        reduced_injection_params[key] = \
                            parameter
                
                snrs_ = \
                    generate_snrs(
                        injection_masks_,
                        num_examples_per_batch,
                        config,
                        example_index
                    )
                
                snrs.append(
                    batch_tensor(snrs_,  num_examples_per_batch)
                )
                                
                injection_parameters.append(
                    batch_injection_parameters(
                        reduced_injection_params, 
                        num_examples_per_batch,
                        injections_.shape[0]
                    )
                )
                
                injections.append(
                    batch_tensor(injections_, num_examples_per_batch)
                )    
            
                injection_masks.append(
                    batch_tensor(injection_masks_, num_examples_per_batch)
                )
            
            injection_parameters = np.array(injection_parameters)
            injection_masks = tf.stack(injection_masks)
            snrs = tf.stack(snrs)
            
            for batch_index in range(current_max_batch_count):
                
                example_index += num_examples_per_batch
                
                num_onsource_samples = \
                    int(
                        (onsource_duration_seconds + fduration)
                        *sample_rate_hertz
                    )
                num_offsource_samples = \
                    int(offsource_duarion_seconds * sample_rate_hertz)
                
                batched_onsource, batched_offsource, batched_gps_times = \
                    random_subsection(
                        current_segment_data.data,
                        current_segment_data.dt,
                        current_segment_data.t0,
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch
                    )
                
                batched_gps_times = batched_gps_times + fduration/0.5
                
                cropped_injections = []
                amplitudes = []
                for injection_index, config in enumerate(injection_configs):
                                        
                    scaled_injections = \
                        scale_to_snr(
                            injections[injection_index][batch_index], 
                            batched_onsource, 
                            snrs[injection_index][batch_index],
                            sample_rate_hertz=sample_rate_hertz,
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5
                        )
                    
                    scaled_injections = \
                        replace_nan_and_inf_with_zero(scaled_injections)
                    
                    amplitudes.append(
                        tf.reduce_max(tf.abs(scaled_injections), axis=1)*100
                    )
                    
                    batched_onsource += scaled_injections

                    cropped_injections.append(
                        crop_samples(
                            scaled_injections, 
                            onsource_duration_seconds, 
                            sample_rate_hertz
                        )
                    )
                
                # Whiten data: 
                whitened_injections = None
                if apply_whitening:
                    batched_onsource = \
                        whiten(
                            batched_onsource, 
                            batched_offsource, 
                            sample_rate_hertz, 
                            fftlength = 1.0,
                            overlap = 0.5,
                            fduration = fduration
                        )
                    
                    if ("whitened_injections" in input_keys) or \
                        ("whitened_injections" in output_keys):
                        whitened_injections = [
                            whiten(
                                injections, 
                                batched_offsource, 
                                sample_rate_hertz, 
                                fftlength = 1.0,
                                overlap = 0.5,
                                fduration = fduration
                            ) for injections in cropped_injections
                        ]
                    
                # Crop to remove edge effects, crop with or without whitening to
                # ensure same data is retrieve in both cases
                batched_onsource = crop_samples(
                    batched_onsource, 
                    onsource_duration_seconds, 
                    sample_rate_hertz
                )
                
                '''
                glitch_overlap = \
                    check_glitch_overlap(
                        batched_gps_times, 
                        onsource_duration_seconds_tf, 
                        glitch_segments_start, 
                        glitch_segments_end
                    )
                '''
                
                #Get shapes of injection configs so they can be shaped for return
                parameter_shapes = get_cbc_parameter_shapes(
                    num_examples_per_batch
                )
                
                input_dict = \
                    create_dict(
                        input_keys,
                        batched_onsource,
                        batched_offsource,
                        batched_gps_times,
                        cropped_injections,
                        whitened_injections,
                        injection_masks,
                        snrs,
                        amplitudes,
                        injection_parameters,
                        batch_index,
                        parameter_shapes
                    )
                
                output_dict = \
                    create_dict(
                        output_keys,
                        batched_onsource,
                        batched_offsource,
                        batched_gps_times,
                        cropped_injections,
                        whitened_injections,
                        injection_masks,
                        snrs,
                        amplitudes,
                        injection_parameters,
                        batch_index,
                        parameter_shapes
                    )
                                
                yield (input_dict, output_dict)

def get_ifo_data_generator(
    time_interval: Union[tuple, ObservingRun], 
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,        
    input_keys = [
        "onsource", 
        "offsource", 
        "gps_time", 
        "injections", 
        "snr"],
    output_keys = [
        "onsource", 
        "offsource", 
        "gps_time", 
        "injections", 
        "snr"],
    **kwargs  # Capture all other arguments
    ):
    
    num_examples_per_batch = kwargs.get('num_examples_per_batch', 1)
    num_onsource_samples = \
        int(kwargs.get('onsource_duration_seconds', 1.0)*sample_rate_hertz)
    num_offsource_samples = \
        int(kwargs.get('offsource_duration_seconds', 16.0)*sample_rate_hertz)
    num_injection_configs = len(kwargs.get('injection_configs', {}))
    
    output_signature_dict = {
        'onsource'       : \
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_onsource_samples), 
                dtype=tf.float16
            ),
        'offsource' : \
            tf.TensorSpec(
                shape=(num_examples_per_batch, num_offsource_samples), 
                dtype=tf.float16
            ),
        'gps_time' : 
            tf.TensorSpec(
                shape=(num_examples_per_batch), 
                dtype=tf.int64
            ),
        'injections' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch, 
                    num_onsource_samples
                ), 
                dtype=tf.float16
            ),
        'whitened_injections' :
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch, 
                    num_onsource_samples
                ), 
                dtype=tf.float16
            ),
        'injection_masks' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs, 
                    num_examples_per_batch
                ), 
                dtype=tf.bool
            ),
        'snr' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs,
                    num_examples_per_batch
                ), 
                dtype=tf.float64
            ),
        'amplitude' : 
            tf.TensorSpec(
                shape=(
                    num_injection_configs,
                    num_examples_per_batch
                ), 
                dtype=tf.float64
            ),
    }
    
    parameter_shapes = get_cbc_parameter_shapes(
        num_examples_per_batch
    )
    
    # Add injection parameters to output options
    for key, value in parameter_shapes.items():
        output_signature_dict[key] =  \
            tf.TensorSpec(
                shape=(
                    num_injection_configs,
                    value
                ), 
                dtype=tf.float32
            )   
    
    output_signature = (
        {k: output_signature_dict[k] for k in input_keys},
        {k: output_signature_dict[k] for k in output_keys}
    )
    
    generator = lambda: \
        get_ifo_data(
            time_interval, 
            data_labels, 
            ifo, 
            sample_rate_hertz, 
            input_keys = input_keys, 
            output_keys = output_keys,
            **kwargs
        )
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
        
    return tf.data.Dataset.from_generator(
            generator = generator,
            output_signature = output_signature
        ).with_options(options)

def extract_data_from_indicies(
        dataset,
        indicies : list, 
        num_examples_per_batch : int
    ) -> list:
    
    indicies = sorted(indicies) 
    
    dataset_elements = []
    current_index = 0
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
    
    return get_ifo_data_generator(**args).take(num_batches)
"""

def set_random_seeds(
    seed : int = 100
    ):
    
    """
    Set random seeds for Tensorflow, Numpy, and Core Python to ensure 
    deterministic results with the same seed. This means that if the seed is the 
    concerved the dataset produced will be identical.
    
    Args
    ---
    
    seed : int
        Random seed which will be used to set both Numpy and TensorFlow seeds
    
    """
    
    # Set tensorflow random seed:
    tf.random.set_seed(seed)
    
    # Set Numpy random seed:
    np.random.seed(seed)
    
    # Set core Python.random seed just in case, I don't think its used:
    random.seed(10)