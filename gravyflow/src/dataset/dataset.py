from typing import List, Tuple, Union, Dict, Any, Iterator, Optional, Set
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass
from warnings import warn
import logging
import traceback

import numpy as np
import keras
from keras import ops
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
        # Use ops.stack and ops.max instead of tf
        max_arrival_time_differences = ops.stack(max_arrival_time_differences)
        max_arival_time_difference_seconds = ops.max(
            max_arrival_time_differences
        )
    
    return max_arival_time_difference_seconds

class GravyflowDataset(keras.utils.PyDataset):
    """
    A Keras PyDataset to generate data for gravitational wave detection training.
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
        waveform_generators: Optional[List[Union[gf.RippleGenerator, gf.WNBGenerator]]] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        mask_history: Optional[List] = None,
        steps_per_epoch: int = 1000,
        workers: int = 1,
        use_multiprocessing: bool = False,
        max_queue_size: int = 10
    ):
        # Initialize PyDataset
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)
        
        self.steps_per_epoch = steps_per_epoch

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

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate the next batch of data.
        """
        # Note: 'index' is ignored because data generation is infinite/random.
        # In a multiprocessing setup, each worker will call this.
        # Since self.noise and self.injections are iterators, we need to be careful.
        # PyDataset with workers=1 (default) runs in a thread pool, sharing the instance.
        # Iterators are not thread-safe.
        # However, for now, let's assume single-threaded or manage locks if needed.
        # Or better, we can rely on the fact that next() is atomic enough for simple generators,
        # but complex ones might have issues.
        # For full safety with workers > 1, we might need to re-instantiate generators per worker.
        
        try:
            onsource, offsource, gps_times = next(self.noise)
        except Exception as e:
            logging.info(f"Noise generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Noise generation failed: {e}")

        try:
            injections, mask, parameters = next(self.injections)
        except Exception as e:
            logging.info(f"Injection generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Injection generation failed: {e}")

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
                # Return empty or raise? PyDataset expects a batch.
                raise e

            if onsource is None:
                raise ValueError("Onsource is None!")

            if offsource is None:
                raise ValueError("Offsource is None!")

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

    def _process_whitened_injections(self, scaled_injections, offsource):
        """Process whitened injections if required."""
        if gf.ReturnVariables.WHITENED_INJECTIONS in self.variables_to_return:
            whitened_injections = gf.whiten(
                scaled_injections, 
                offsource, 
                self.sample_rate_hertz, 
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5,
                filter_duration_seconds=1.0
            )
            return gf.replace_nan_and_inf_with_zero(whitened_injections)
        return None

    def _process_onsource(self, onsource, offsource):
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
            
            whitened_onsource = ops.cast(whitened_onsource, "float16")
            whitened_onsource = gf.replace_nan_and_inf_with_zero(whitened_onsource)

            return whitened_onsource, rolling_pearson_onsource, spectrogram_onsource
        return None, None, None

    def _calculate_rolling_pearson(self, whitened_onsource):
        """Calculate rolling Pearson correlation if required."""
        if gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in self.variables_to_return:
            max_arrival_time_difference_seconds = get_max_arrival_time_difference(self.waveform_generators)
            return gf.rolling_pearson(
                whitened_onsource,
                max_arrival_time_difference_seconds,
                self.sample_rate_hertz
            )
        return None

    def _calculate_spectrogram(self, whitened_onsource):
        """Calculate spectrogram if required."""
        if gf.ReturnVariables.SPECTROGRAM_ONSOURCE in self.variables_to_return:
            return gf.spectrogram(whitened_onsource)
        return None

    def _process_raw_onsource(self, onsource):
        """Process raw onsource data if required."""
        if gf.ReturnVariables.ONSOURCE in self.variables_to_return:
            onsource = ops.cast(onsource, "float32")
            onsource = gf.replace_nan_and_inf_with_zero(onsource)
            return onsource
        return None

    def _process_offsource(self, offsource):
        """Process offsource data if required."""
        if gf.ReturnVariables.OFFSOURCE in self.variables_to_return:
            offsource = ops.cast(offsource, "float32")
            offsource = gf.replace_nan_and_inf_with_zero(offsource)
            return offsource
        return None

    def _process_gps_times(self, gps_times):
        """Process GPS times if required."""
        if gf.ReturnVariables.GPS_TIME in self.variables_to_return:
            return ops.cast(gps_times, "float64")
        return None

    def _process_mask(self, mask):
        """Process injection masks if required."""
        if gf.ReturnVariables.INJECTION_MASKS in self.variables_to_return:
            mask = ops.cast(mask, "float32")
            return mask
        return None

    def _create_output_dictionaries(self, *args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_dict = create_variable_dictionary(self.input_variables, *args)
        output_dict = create_variable_dictionary(self.output_variables, *args)
        
        # Remove keys with None values
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        
        return input_dict, output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.__getitem__(0)

def create_variable_dictionary(
    return_variables: List[Union[gf.ReturnVariables, gf.WaveformParameters]],
    onsource,
    whitened_onsource,
    offsource,
    gps_times,
    injections,
    whitened_injections,
    mask,
    rolling_pearson_onsource,
    spectrogram_onsource,
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

# Legacy wrapper for compatibility
def Dataset(**kwargs):
    return GravyflowDataset(**kwargs)

# Legacy class alias
data = GravyflowDataset