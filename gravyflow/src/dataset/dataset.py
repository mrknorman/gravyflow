from typing import List, Tuple, Union, Dict, Any, Iterator, Optional, Set
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass
from warnings import warn
import logging
import traceback

import tensorflow as tf
from tensorflow.data.experimental import AutoShardPolicy
from tensorflow.data import Options

import gravyflow as gf

def validate_noise_settings(
        noise_obtainer: gf.NoiseObtainer, 
        variables_to_return: List[
            Union[
                gf.WaveformParameters, 
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
    waveform_generators : List
    ) -> float:
    
    max_arival_time_difference_seconds : float = 0.01
    
    max_arrival_time_differences = []

    if isinstance(waveform_generators, list):
        for generator in waveform_generators:
            if (generator is not None) and (generator.network is not None):
                max_arrival_time_differences.append(
                    generator.network.max_arrival_time_difference_seconds
                )     
    elif isinstance(waveform_generators, dict):
        for generator in waveform_generators.values(): 
            if (generator is not None) and (generator["generator"].network is not None):
                max_arrival_time_differences.append(
                    generator["generator"].network.max_arrival_time_difference_seconds
                )     
    
    if len(max_arrival_time_differences):
        max_arrival_time_differences = tf.stack(max_arrival_time_differences)
        max_arival_time_difference_seconds = tf.reduce_max(
            max_arrival_time_differences
        )
    
    return max_arival_time_difference_seconds

class data:
    """
    A class to generate data for gravitational wave detection training.

    This generator produces batches of data, combining noise and simulated
    gravitational wave signals, along with various processed forms of this data.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        offsource_duration_seconds: Optional[float] = None,
        crop_duration_seconds: Optional[float] = None,
        scale_factor: Optional[float] = None,
        noise_obtainer: Optional[gf.NoiseObtainer] = None,
        group: str = "train",
        waveform_generators: Optional[List[Union[gf.cuPhenomDGenerator, gf.WNBGenerator]]] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        mask_history: Optional[List] = None
    ):
        """
        Initialize the DataGenerator with the given parameters.

        :param seed: Random seed for reproducibility
        :param sample_rate_hertz: Sampling rate of the data
        :param onsource_duration_seconds: Duration of the on-source data
        :param offsource_duration_seconds: Duration of the off-source data
        :param crop_duration_seconds: Duration to crop the data
        :param scale_factor: Scaling factor for the data
        :param noise_obtainer: Object to obtain noise data
        :param group: Group identifier (e.g., "train", "test")
        :param waveform_generators: List of waveform generators
        :param num_examples_per_generation_batch: Number of examples per generation batch
        :param num_examples_per_batch: Number of examples per output batch
        :param input_variables: List of input variables to return
        :param output_variables: List of output variables to return
        :param mask_history: List to store injection masks
        """
        # Set default values
        self.seed = seed if seed is not None else gf.Defaults.seed
        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else gf.Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else gf.Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else gf.Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else gf.Defaults.crop_duration_seconds
        self.scale_factor = scale_factor if scale_factor is not None else gf.Defaults.scale_factor
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else gf.Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else gf.Defaults.num_examples_per_batch
        
        self.noise_obtainer = noise_obtainer if noise_obtainer is not None else gf.NoiseObtainer()
        self.group = group
        self.input_variables = input_variables if input_variables is not None else []
        self.output_variables = output_variables if output_variables is not None else []
        self.mask_history = mask_history
        
        # Initialize waveform generators
        self.waveform_generators = waveform_generators if waveform_generators is not None else []
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        # Set network for waveform generators if not provided
        self._set_generator_networks()

        # Validate settings
        self._validate_settings()

        # Set random seeds
        gf.set_random_seeds(self.seed)

        # Create noise and injection generators
        self._create_generators()

    def _set_generator_networks(self):
        """Set the network for waveform generators if not provided."""
        if isinstance(self.waveform_generators, list):
            for generator in self.waveform_generators:
                if generator.network is None: 
                    generator.network = gf.Network(self.noise_obtainer.ifos)
        elif isinstance(self.waveform_generators, dict):
            for generator in self.waveform_generators.values():
                if generator["generator"].network is None: 
                    generator["generator"].network = gf.Network(self.noise_obtainer.ifos)

    def _validate_settings(self):
        """Validate the generator settings."""
        self.variables_to_return = set(self.input_variables + self.output_variables)
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        validate_noise_settings(self.noise_obtainer, self.variables_to_return)

    def _create_generators(self):
        """Create noise and injection generators."""
        self.noise = self.noise_obtainer(
            sample_rate_hertz=self.sample_rate_hertz,
            onsource_duration_seconds=self.onsource_duration_seconds,
            crop_duration_seconds=self.crop_duration_seconds,
            offsource_duration_seconds=self.offsource_duration_seconds,
            num_examples_per_batch=self.num_examples_per_batch,
            scale_factor=self.scale_factor,
            group=self.group,
            seed=self.seed
        )

        waveform_parameters_to_return = [
            item for item in self.variables_to_return if isinstance(
                item.value, gf.WaveformParameter
            )
        ]
        self.injection_generator = gf.InjectionGenerator(
            waveform_generators=self.waveform_generators,
            parameters_to_return=waveform_parameters_to_return,
            seed=self.seed
        )
        self.injections = self.injection_generator(
            sample_rate_hertz=self.sample_rate_hertz,
            onsource_duration_seconds=self.onsource_duration_seconds,
            crop_duration_seconds=self.crop_duration_seconds,
            num_examples_per_generation_batch=self.num_examples_per_generation_batch,
            num_examples_per_batch=self.num_examples_per_batch,
        )

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate the next batch of data.

        :return: A tuple containing two dictionaries:
                 1. Input dictionary with requested input variables
                 2. Output dictionary with requested output variables
        """
        while True:
            try:
                onsource, offsource, gps_times = next(self.noise)
            except Exception as e:
                logging.info(f"Noise generation failed: {e}\nTraceback: {traceback.format_exc()}")
                raise Exception()

            try:
                injections, mask, parameters = next(self.injections)
            except Exception as e:
                logging.info(f"Injection generation failed: {e}\nTraceback: {traceback.format_exc()}")
                raise Exception()

            if self.waveform_generators:
                try:

                    onsource, scaled_injections, scaling_parameters = self.injection_generator.add_injections_to_onsource(
                        injections,
                        mask,
                        onsource,
                        parameters_to_return=self.variables_to_return
                    )

                except Exception as e:
                    logging.error(f"Failed to add injections to onsource: {e}\nTraceback: {traceback.format_exc()}")
                    continue

                if onsource is None:
                    logging.error("Onsource is None!")
                    continue

                if offsource is None:
                    logging.error("Offsource is None!")
                    continue

                for key, value in scaling_parameters.items():
                    if key in self.variables_to_return:
                        parameters[key] = value

                whitened_injections = self._process_whitened_injections(scaled_injections, offsource)
            else:
                scaled_injections = None
                whitened_injections = None

            whitened_onsource, rolling_pearson_onsource, spectrogram_onsource = self._process_onsource(onsource, offsource)

            onsource = self._process_raw_onsource(onsource)
            offsource = self._process_offsource(offsource)
            gps_times = self._process_gps_times(gps_times)
            mask = self._process_mask(mask)

            input_dict, output_dict = self._create_output_dictionaries(
                onsource, whitened_onsource, offsource, gps_times, scaled_injections,
                whitened_injections, mask, rolling_pearson_onsource, spectrogram_onsource, parameters
            )

            return input_dict, output_dict

    def _process_whitened_injections(self, scaled_injections: tf.Tensor, offsource: tf.Tensor) -> Optional[tf.Tensor]:
        """Process whitened injections if required."""
        if gf.ReturnVariables.WHITENED_INJECTIONS in self.variables_to_return:
            # Define a function that whitens a single injection.
            whitened_injections = gf.whiten(
                scaled_injections, 
                offsource, 
                self.sample_rate_hertz, 
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5,
                filter_duration_seconds=1.0
            )
            # Replace NaN and Inf values in the resulting tensor before returning
            return gf.replace_nan_and_inf_with_zero(whitened_injections)
        return None

    def _process_onsource(self, onsource: tf.Tensor, offsource: tf.Tensor) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Process onsource data, including whitening, rolling Pearson, and spectrogram generation."""
        if (gf.ReturnVariables.WHITENED_ONSOURCE in self.variables_to_return) or \
           (gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in self.variables_to_return) or \
           (gf.ReturnVariables.SPECTROGRAM_ONSOURCE in self.variables_to_return):

            whitened_onsource = gf.whiten(
                onsource, 
                offsource, 
                self.sample_rate_hertz, 
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5,
                filter_duration_seconds=1.0
            )
            
            whitened_onsource = gf.crop_samples(
                whitened_onsource, 
                self.onsource_duration_seconds, 
                self.sample_rate_hertz
            )
                        
            rolling_pearson_onsource = self._calculate_rolling_pearson(whitened_onsource)
            spectrogram_onsource = self._calculate_spectrogram(whitened_onsource)
            
            whitened_onsource = tf.cast(whitened_onsource, tf.float16)
            whitened_onsource = gf.replace_nan_and_inf_with_zero(whitened_onsource)

            return whitened_onsource, rolling_pearson_onsource, spectrogram_onsource
        return None, None, None

    def _calculate_rolling_pearson(self, whitened_onsource: tf.Tensor) -> Optional[tf.Tensor]:
        """Calculate rolling Pearson correlation if required."""
        if gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in self.variables_to_return:
            max_arrival_time_difference_seconds = get_max_arrival_time_difference(self.waveform_generators)
            return gf.rolling_pearson(
                whitened_onsource,
                max_arrival_time_difference_seconds,
                self.sample_rate_hertz
            )
        return None

    def _calculate_spectrogram(self, whitened_onsource: tf.Tensor) -> Optional[tf.Tensor]:
        """Calculate spectrogram if required."""
        if gf.ReturnVariables.SPECTROGRAM_ONSOURCE in self.variables_to_return:
            return gf.spectrogram(whitened_onsource)
        return None

    def _process_raw_onsource(self, onsource: tf.Tensor) -> Optional[tf.Tensor]:
        """Process raw onsource data if required."""
        if gf.ReturnVariables.ONSOURCE in self.variables_to_return:
            onsource = tf.cast(onsource, tf.float32)
            onsource = gf.replace_nan_and_inf_with_zero(onsource)
            return onsource
        return None

    def _process_offsource(self, offsource: tf.Tensor) -> Optional[tf.Tensor]:
        """Process offsource data if required."""
        if gf.ReturnVariables.OFFSOURCE in self.variables_to_return:
            offsource = tf.cast(offsource, tf.float32)
            offsource = gf.replace_nan_and_inf_with_zero(offsource)
            return offsource
        return None

    def _process_gps_times(self, gps_times: tf.Tensor) -> Optional[tf.Tensor]:
        """Process GPS times if required."""
        if gf.ReturnVariables.GPS_TIME in self.variables_to_return:
            return tf.cast(gps_times, tf.float64)
        return None

    def _process_mask(self, mask: tf.Tensor) -> Optional[tf.Tensor]:
        """Process injection masks if required."""
        if gf.ReturnVariables.INJECTION_MASKS in self.variables_to_return:
            mask = tf.cast(mask, tf.float32)
            #if self.mask_history is not None:
                #self.mask_history.append(mask)
            return mask
        return None

    def _create_output_dictionaries(self, *args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_dict = create_variable_dictionary(self.input_variables, *args)
        output_dict = create_variable_dictionary(self.output_variables, *args)
        
        # Remove keys with None values to avoid unnecessary memory allocation.
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        
        return input_dict, output_dict

def create_variable_dictionary(
    return_variables: List[Union[gf.ReturnVariables, gf.WaveformParameters]],
    onsource : tf.Tensor,
    whitened_onsource : tf.Tensor,
    offsource : tf.Tensor,
    gps_times : tf.Tensor,
    injections : tf.Tensor,
    whitened_injections : tf.Tensor,
    mask : tf.Tensor,
    rolling_pearson_onsource : tf.Tensor,
    spectrogram_onsource : tf.Tensor,
    injection_parameters : Optional[Dict]
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
    if injection_parameters:
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

def ensure_even(number):
    if number % 2 != 0:
        number -= 1
    return number

def Dataset(
        seed: int = None,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        offsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        scale_factor: float = None,
        noise_obtainer: gf.NoiseObtainer = None,
        group: str = "train",
        waveform_generators: Union[
            List[gf.WaveformGenerator], Dict[str, gf.WaveformGenerator]
        ] = None,
        num_examples_per_generation_batch: int = None,
        num_examples_per_batch: int = None,
        input_variables: List = None,
        output_variables: List = None,
        mask_history=None
    ) -> tf.data.Dataset:
    """
    Generates a TensorFlow dataset with interferometer data, pre-instantiating heavy
    sub-generators so that they are not re-initialized each epoch.
    
    Parameters:
        seed (int): Random seed.
        sample_rate_hertz (float): Sample rate in Hz.
        onsource_duration_seconds (float): On-source duration in seconds.
        offsource_duration_seconds (float): Off-source duration in seconds.
        crop_duration_seconds (float): Crop duration in seconds.
        scale_factor (float): Scale factor.
        noise_obtainer (gf.NoiseObtainer): Object to obtain noise.
        group (str): Data group (e.g. "train").
        waveform_generators (list): List (or dict) of injection generators.
        num_examples_per_generation_batch (int): Number of examples per generation batch.
        num_examples_per_batch (int): Number of examples per batch.
        input_variables (list): List of input variables.
        output_variables (list): List of output variables.
        mask_history: (Optional) List for storing mask history.
    
    Returns:
        tf.data.Dataset: TensorFlow Dataset object.
    """

    # Set default values if not provided.
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
    if waveform_generators is None:
        waveform_generators = []
    elif not isinstance(waveform_generators, list) and not isinstance(waveform_generators, dict):
        waveform_generators = [waveform_generators]

    # Compute sample counts and shapes.
    num_cropped_samples = ensure_even(int(onsource_duration_seconds * sample_rate_hertz))
    num_onsource_samples = ensure_even(int((onsource_duration_seconds + 2 * crop_duration_seconds) * sample_rate_hertz))
    num_offsource_samples = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))
    num_waveform_generators = len(waveform_generators)

    num_detectors = 1
    if isinstance(waveform_generators, list):
        if waveform_generators:
            if waveform_generators[0].network is not None:
                num_detectors = waveform_generators[0].network.num_detectors
            elif noise_obtainer is not None:
                num_detectors = len(noise_obtainer.ifos)
        elif noise_obtainer is not None:
            num_detectors = len(noise_obtainer.ifos)
    elif isinstance(waveform_generators, dict):
        if waveform_generators:
            for generator in waveform_generators.values():
                if generator["generator"].network is not None:
                    num_detectors = generator["generator"].network.num_detectors
                elif noise_obtainer is not None:
                    num_detectors = len(noise_obtainer.ifos)
        elif noise_obtainer is not None:
            num_detectors = len(noise_obtainer.ifos)
    if num_detectors is None:
        num_detectors = 1

    max_arrival_time_difference_seconds = get_max_arrival_time_difference(waveform_generators)
    max_arival_time_difference_samples = int(max_arrival_time_difference_seconds * sample_rate_hertz)

    onsource_shape = (num_examples_per_batch, num_detectors, num_onsource_samples)
    cropped_shape = (num_examples_per_batch, num_detectors, num_cropped_samples)
    offsource_shape = (num_examples_per_batch, num_detectors, num_offsource_samples)
    detectors_shape = (num_examples_per_batch, num_detectors)
    injections_shape = (num_waveform_generators, num_examples_per_batch, num_detectors, num_cropped_samples)
    per_injection_shape = (num_waveform_generators, num_examples_per_batch)
    pearson_shape = (num_examples_per_batch,
                     num_detectors * (num_detectors - 1) // 2,
                     2 * max_arival_time_difference_samples)
    spectrogram_shape = gf.spectrogram_shape(cropped_shape)

    output_signature_dict = {
        gf.ReturnVariables.ONSOURCE.name: tf.TensorSpec(shape=onsource_shape, dtype=tf.float32),
        gf.ReturnVariables.WHITENED_ONSOURCE.name: tf.TensorSpec(shape=cropped_shape, dtype=tf.float16),
        gf.ReturnVariables.OFFSOURCE.name: tf.TensorSpec(shape=offsource_shape, dtype=tf.float32),
        gf.ReturnVariables.GPS_TIME.name: tf.TensorSpec(shape=detectors_shape, dtype=tf.int64),
        gf.ReturnVariables.INJECTIONS.name: tf.TensorSpec(shape=injections_shape, dtype=tf.float16),
        gf.ReturnVariables.WHITENED_INJECTIONS.name: tf.TensorSpec(shape=injections_shape, dtype=tf.float16),
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE.name: tf.TensorSpec(shape=pearson_shape, dtype=tf.float32),
        gf.ReturnVariables.SPECTROGRAM_ONSOURCE.name: tf.TensorSpec(shape=spectrogram_shape, dtype=tf.float32),
        gf.ReturnVariables.INJECTION_MASKS.name: tf.TensorSpec(shape=per_injection_shape, dtype=tf.float32)
    }

    # Determine additional parameter shapes.
    parameters_to_return = {
        item for item in (input_variables + output_variables)
        if isinstance(item.value, gf.WaveformParameter) or isinstance(item.value, gf.ScalingType)
    }
    if not waveform_generators:
        keys_to_remove = {gf.ReturnVariables.INJECTION_MASKS, gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.WHITENED_INJECTIONS}
        parameters_to_return -= keys_to_remove
        input_variables = [item for item in input_variables if item not in keys_to_remove]
        output_variables = [item for item in output_variables if item not in keys_to_remove]

    output_signature_dict.update({
        item.name: tf.TensorSpec(
            shape=(num_waveform_generators, num_examples_per_batch * item.value.shape[-1]),
            dtype=tf.float32
        ) for item in parameters_to_return
    })

    output_signature = (
        {k.name: output_signature_dict[k.name] for k in input_variables},
        {k.name: output_signature_dict[k.name] for k in output_variables}
    )

    # Pre-instantiate heavy sub-generators.
    if noise_obtainer is None:
        noise_obtainer = gf.NoiseObtainer()
    # Create the noise generator once.
    noise_gen = noise_obtainer(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        scale_factor=scale_factor,
        group=group,
        seed=seed
    )
    # Pre-instantiate the injection generator.
    waveform_parameters_to_return = [
        item for item in (input_variables + output_variables)
        if isinstance(item.value, gf.WaveformParameter)
    ]
    inj_gen = gf.InjectionGenerator(
        waveform_generators=waveform_generators,
        parameters_to_return=waveform_parameters_to_return,
        seed=seed
    )

    # Create a single data generator instance and override its sub-generators.
    data_instance = data(
        seed=seed,
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        scale_factor=scale_factor,
        noise_obtainer=noise_obtainer,  # used to instantiate default noise generator
        group=group,
        waveform_generators=waveform_generators,
        num_examples_per_generation_batch=num_examples_per_generation_batch,
        num_examples_per_batch=num_examples_per_batch,
        input_variables=input_variables,
        output_variables=output_variables,
        mask_history=mask_history
    )
    # Overwrite with our pre-instantiated infinite generators.
    data_instance.noise = noise_gen
    data_instance.injection_generator = inj_gen
    data_instance.injections = inj_gen(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        num_examples_per_generation_batch=num_examples_per_generation_batch,
        num_examples_per_batch=num_examples_per_batch,
    )

    # Define a generator function that continuously yields batches from our single instance.
    def generator():
        while True:
            yield next(data_instance)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA

    return tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=output_signature
    ).with_options(options).prefetch(tf.data.AUTOTUNE)

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
        while current_index < len(indicies) and indicies[current_index] < end_index:
                        
            # Calculate in-batch index:
            in_batch_index = indicies[current_index] % num_examples_per_batch  
                                    
            # Extract the corresponding data from in_dict and out_dict using 
            # in_batch_index:
            example_element = {
                key: value[in_batch_index[0]] for key, value in in_dict.items()
            }
                        
            out_element = {
                key: value[0][
                    in_batch_index[0]
                ] for key, value in out_dict.items()
            }
                        
            for key, value in out_element.items():
                example_element[key] = value
            
            dataset_elements.append(example_element)

            current_index += 1  # Move to the next worst index
                
    return dataset_elements