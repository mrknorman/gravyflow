from typing import List, Tuple, Union, Dict, Any, Iterator, Optional, Set
from enum import Enum, auto
import copy
from pathlib import Path
from dataclasses import dataclass
from warnings import warn
import logging
import traceback

logger = logging.getLogger(__name__)

import numpy as np
import keras
from keras import ops
import jax
import gravyflow as gf
from gravyflow.src.dataset.conditioning.whiten import whiten
from gravyflow.src.dataset.tools.psd import SpectralConfig, estimate_psd, MIN_PSD_SEGMENTS
from gravyflow.src.utils.tensor import crop_samples, replace_nan_and_inf_with_zero, set_random_seeds
from gravyflow.src.dataset.config import Defaults
from gravyflow.src.dataset.conditioning.pearson import rolling_pearson
from gravyflow.src.dataset.conditioning.conditioning import spectrogram
from gravyflow.src.utils.numerics import ensure_even, ensure_list
from gravyflow.src.utils.gps import as_gps_times

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
    # Skip validation for TransientObtainer which doesn't have noise_type
    if not hasattr(noise_obtainer, 'noise_type'):
        return  # TransientObtainer or similar - skip validation
    
    if noise_obtainer.noise_type in [gf.NoiseType.WHITE, gf.NoiseType.COLORED]:
        
        if noise_obtainer.ifo_data_obtainer is not None:
            warn(
                ("Noise is not real or PSEUDO-REAL, yet data obtainer is"
                 " defined."), 
                UserWarning
            )
            
        if gf.ReturnVariables.START_GPS_TIME in variables_to_return:
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

WHITENED_VARIABLES = (
    gf.ReturnVariables.WHITENED_ONSOURCE,
    gf.ReturnVariables.WHITENED_INJECTIONS,
    gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE,
    gf.ReturnVariables.SPECTROGRAM_ONSOURCE,
)


def _ifo_names(ifos) -> List[str]:
    return [ifo.name if hasattr(ifo, "name") else str(ifo) for ifo in ensure_list(ifos)]


def check_network_matches_ifos(network: "gf.Network", ifos) -> None:
    """
    Raise ValueError unless `network` projects onto exactly `ifos`, in order. A
    network built from geometry alone (no IFO list) is checked by its detector
    count.
    """
    expected = _ifo_names(ifos)
    network_ifos = getattr(network, "ifos", None)
    if network_ifos is not None:
        if _ifo_names(network_ifos) != expected:
            raise ValueError(
                f"A waveform generator's network projects onto {_ifo_names(network_ifos)}, "
                f"but the noise has detectors {expected}; they must be the same, in order."
            )
        return
    num_detectors = getattr(network, "num_detectors", None)
    if num_detectors is not None and int(np.asarray(num_detectors).ravel()[0]) != len(expected):
        raise ValueError(
            f"A waveform generator's network has {int(np.asarray(num_detectors).ravel()[0])} detectors, "
            f"but the noise has {len(expected)} ({expected})."
        )


def lowest_generator_frequency(generator) -> Optional[float]:
    """The lowest frequency a generator's parameters allow, where it declares one."""
    lowest = None
    for name in ("min_frequency_hertz", "frequency_hertz", "start_frequency_hertz", "end_frequency_hertz"):
        value = getattr(generator, name, None)
        if value is None:
            continue
        if isinstance(value, gf.Distribution):
            if value.type_ == gf.DistributionType.CONSTANT:
                value = value.value
            elif value.min_ is not None:
                value = value.min_
            elif value.possible_values is not None:
                value = min(value.possible_values)
            else:
                continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        lowest = value if lowest is None else min(lowest, value)
    return lowest


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
        waveform_generators: Optional[List[Union[gf.CBCGenerator, gf.WNBGenerator]]] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        mask_history: Optional[List] = None,
        steps_per_epoch: int = 1000,
        sampling_mode: Optional["gf.SamplingMode"] = None,
        spectral_config: Optional[SpectralConfig] = None
    ):
        """
        spectral_config: the band, PSD estimator and whitening filter shared by
            SNR scaling, SNR labels and whitening (gf.SpectralConfig; default
            band 20 Hz to Nyquist, 1 s / 0.5 s median Welch PSD, 1 s filter).
            One PSD is estimated per batch from the offsource and reused by all
            three.
        """
        # Initialize PyDataset (single-threaded to avoid JAX/fork deadlocks)
        super().__init__(workers=0)
        
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
        self.ifos = ensure_list(ifos) if ifos is not None else None
        
        # Default to RANDOM mode if not specified
        self.sampling_mode = sampling_mode if sampling_mode is not None else gf.SamplingMode.RANDOM
        self.input_variables = input_variables if input_variables is not None else []
        self.output_variables = output_variables if output_variables is not None else []
        self.mask_history = mask_history
        self.spectral_config = spectral_config if spectral_config is not None else SpectralConfig()
        
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

        # Configure any Curriculum objects in waveform generators
        self._configure_curricula()

    def _configure_curricula(self):
        """Auto-configure any Curriculum objects in waveform generators."""
        generators = []
        if isinstance(self.waveform_generators, list):
            generators = self.waveform_generators
        elif isinstance(self.waveform_generators, dict):
            generators = [g["generator"] for g in self.waveform_generators.values()]
        
        for generator in generators:
            if hasattr(generator, 'scaling_method') and generator.scaling_method is not None:
                value = generator.scaling_method.value
                # Duck-type check for Curriculum (has configure method)
                if hasattr(value, 'configure'):
                    value.configure(steps_per_epoch=self.steps_per_epoch)

    def _noise_ifos(self) -> List:
        """The detectors the noise will have: the dataset's `ifos`, else the obtainer's."""
        if self.ifos is not None:
            return list(self.ifos)
        ifos = getattr(self.noise_obtainer, "ifos", None)
        return ensure_list(ifos) if ifos is not None else [gf.IFO.L1]

    def _set_generator_networks(self):
        """
        Give every generator a network of exactly the noise's detectors, in order.

        A generator without a network is copied and the copy gets one built from
        the noise's detectors, so the caller's generator is never modified and can
        serve another dataset. A generator with its own network must match the
        noise's detectors; a mismatch would broadcast one projection into several
        channels or mislabel them, so it raises.
        """
        noise_ifos = self._noise_ifos()

        def bind(generator):
            if generator.network is None:
                generator = copy.copy(generator)
                generator.network = gf.Network(noise_ifos)
            else:
                check_network_matches_ifos(generator.network, noise_ifos)
            return generator

        if isinstance(self.waveform_generators, list):
            self.waveform_generators = [bind(generator) for generator in self.waveform_generators]
        elif isinstance(self.waveform_generators, dict):
            bound = {}
            for name, entry in self.waveform_generators.items():
                entry = dict(entry)
                entry["generator"] = bind(entry["generator"])
                bound[name] = entry
            self.waveform_generators = bound

    def _validate_settings(self):
        """Validate the generator settings."""
        self.variables_to_return = set(self.input_variables + self.output_variables)
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        validate_noise_settings(self.noise_obtainer, self.variables_to_return)
        self._validate_padding_settings()
        self._validate_spectral_settings()

    def _generator_list(self) -> List:
        if isinstance(self.waveform_generators, list):
            return list(self.waveform_generators)
        if isinstance(self.waveform_generators, dict):
            return [g["generator"] for g in self.waveform_generators.values()]
        return []

    def _validate_spectral_settings(self):
        """
        Check the windows against the PSD estimate and the whitening filter, once,
        at construction:
        - whitening needs a crop of at least half the filter on each side of the
          onsource (so the padded onsource is at least as long as the filter),
          since the filter corrupts that much at each edge;
        - SNR scaling and whitening need an offsource holding at least
          MIN_PSD_SEGMENTS Welch segments.
        """
        config = self.spectral_config
        fs = self.sample_rate_hertz
        config.validate_band(fs)

        self._needs_whitening = any(
            variable in self.variables_to_return for variable in WHITENED_VARIABLES
        )
        has_snr_scaling = any(
            getattr(getattr(generator, "scaling_method", None), "type_", None) == gf.ScalingTypes.SNR
            for generator in self._generator_list()
        )
        # SNR labels of HRSS/HPEAK-scaled injections are measured against the
        # batch's spectrum too, so they share the one definition.
        has_snr_label = self._generator_list() and gf.ScalingTypes.SNR in self.variables_to_return
        self._needs_spectrum = self._needs_whitening or has_snr_scaling or bool(has_snr_label)

        if self._needs_whitening:
            ntaps = config.num_taps(fs)
            crop_samples_each_side = int(round(self.crop_duration_seconds * fs))
            if crop_samples_each_side < ntaps // 2:
                raise ValueError(
                    f"crop_duration_seconds ({self.crop_duration_seconds}) must be at least half of "
                    f"the whitening filter ({config.filter_duration_seconds} s): the filter corrupts "
                    f"{config.filter_duration_seconds / 2:g} s at each edge of the window."
                )
            if config.fft_duration_seconds > config.filter_duration_seconds:
                warn(
                    f"The PSD segments ({config.fft_duration_seconds} s) are longer than the "
                    f"whitening filter ({config.filter_duration_seconds} s), which cannot realise "
                    "the PSD's finer features (narrow lines are not removed).",
                    UserWarning
                )

        if self._needs_spectrum:
            num_offsource = int(round(self.offsource_duration_seconds * fs))
            nperseg = config.nperseg(fs)
            noverlap = config.noverlap(fs)
            num_segments = 0 if num_offsource < nperseg else (num_offsource - nperseg) // (nperseg - noverlap) + 1
            if num_segments < MIN_PSD_SEGMENTS:
                raise ValueError(
                    f"offsource_duration_seconds ({self.offsource_duration_seconds}) gives "
                    f"{num_segments} Welch segment(s) of {config.fft_duration_seconds} s; the PSD "
                    f"needs at least {MIN_PSD_SEGMENTS}."
                )

        if config.lower_frequency_hertz:
            for generator in self._generator_list():
                lowest = lowest_generator_frequency(generator)
                if lowest is not None and lowest < config.lower_frequency_hertz:
                    warn(
                        f"{type(generator).__name__} can produce power at {lowest:g} Hz, below the "
                        f"band's lower edge ({config.lower_frequency_hertz:g} Hz). That power is "
                        "removed by whitening and not counted in the SNR, so SNR scaling can "
                        "make such signals very loud in the raw onsource.",
                        UserWarning
                    )

    def _batch_spectrum(self, offsource):
        """The batch's one PSD estimate, shared by scaling, labels and whitening."""
        return estimate_psd(offsource, self.sample_rate_hertz, self.spectral_config)

    def _validate_padding_settings(self):
        """Validate that padding does not exceed half of onsource duration."""
        if self.waveform_generators:
            generators = []
            if isinstance(self.waveform_generators, list):
                generators = self.waveform_generators
            elif isinstance(self.waveform_generators, dict):
                generators = [g["generator"] for g in self.waveform_generators.values()]
            
            for generator in generators:
                front = generator.front_padding_duration_seconds
                back = generator.back_padding_duration_seconds
                
                # Condition: padding <= onsource / 2
                # If padding is larger, the signal center (placed at center of window)
                # could be shifted outside the cropped onsource window.
                max_padding = max(front, back)
                limit = self.onsource_duration_seconds / 2.0
                
                if max_padding > limit:
                    warn(
                        f"Padding ({max_padding}s) exceeds half of onsource duration ({limit}s). "
                        "This may cause the signal center to be shifted outside the cropped view.",
                        UserWarning
                    )

    def _create_generators(self):
        """Create noise and injection generators."""
        # Build base kwargs for noise obtainer
        noise_kwargs = {
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            'crop_duration_seconds': self.crop_duration_seconds,
            'offsource_duration_seconds': self.offsource_duration_seconds,
            'num_examples_per_batch': self.num_examples_per_batch,
            'scale_factor': self.scale_factor,
            'group': self.group,
            'seed': self.seed,
        }
        # Add ifos to noise_kwargs if provided
        if self.ifos is not None:
            noise_kwargs['ifos'] = self.ifos

        # Only pass sampling_mode to NoiseObtainer (not TransientObtainer)
        if hasattr(self.noise_obtainer, 'noise_type'):
            noise_kwargs['sampling_mode'] = self.sampling_mode
        
        self.noise = self.noise_obtainer(**noise_kwargs)

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
        
        try:
            batch_data = next(self.noise)
            
            # Handle dict format (new) or tuple format (legacy)
            if isinstance(batch_data, dict):
                onsource = batch_data[gf.ReturnVariables.ONSOURCE]
                offsource = batch_data[gf.ReturnVariables.OFFSOURCE]
                # Use START_GPS_TIME if available, fall back to TRANSIENT_GPS_TIME for transient mode
                gps_times = batch_data.get(
                    gf.ReturnVariables.START_GPS_TIME,
                    batch_data.get(gf.ReturnVariables.TRANSIENT_GPS_TIME)
                )
                feature_labels = batch_data.get(gf.ReturnVariables.SUB_TYPE)
                # Transient-specific outputs
                glitch_type = batch_data.get(gf.ReturnVariables.GLITCH_TYPE)
                source_type = batch_data.get(gf.ReturnVariables.SOURCE_TYPE)
                data_label = batch_data.get(gf.ReturnVariables.DATA_LABEL)
            elif len(batch_data) == 4:
                onsource, offsource, gps_times, feature_labels = batch_data
                glitch_type, source_type, data_label = None, None, None
            else:
                onsource, offsource, gps_times = batch_data
                feature_labels = None
                glitch_type, source_type, data_label = None, None, None
        except Exception as e:
            logger.info(f"Noise generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Noise generation failed: {e}")

        try:
            injections, mask, parameters = next(self.injections)
        except Exception as e:
            logger.info(f"Injection generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Injection generation failed: {e}")

        # One PSD per batch: scaling, labels and whitening all read this estimate.
        spectrum = self._batch_spectrum(offsource) if self._needs_spectrum else None

        if self.waveform_generators:
            try:
                onsource, scaled_injections, scaling_parameters = self.injection_generator.add_injections_to_onsource(
                    injections,
                    mask,
                    onsource,
                    spectrum if spectrum is not None else offsource,
                    parameters_to_return=self.variables_to_return,
                    onsource_duration_seconds=self.onsource_duration_seconds,
                    injection_parameters=parameters,
                    scale_factor=self.scale_factor
                )
            except Exception as e:
                logger.error(f"Failed to add injections to onsource: {e}\nTraceback: {traceback.format_exc()}")
                # Return empty or raise? PyDataset expects a batch.
                raise e

            if onsource is None:
                raise ValueError("Onsource is None!")

            if offsource is None:
                raise ValueError("Offsource is None!")

            # The labels are the injection layer's: targets, and the values
            # they imply for the scaling types not applied; NaN where absent.
            for key, value in scaling_parameters.items():
                if key in self.variables_to_return:
                    parameters[key] = value

            whitened_injections = self._process_whitened_injections(scaled_injections, spectrum)
            injections = self._process_injections(scaled_injections)
        else:
            scaled_injections = None
            whitened_injections = None
            injections = None



        whitened_onsource, rolling_pearson_onsource, spectrogram_onsource = self._process_onsource(onsource, spectrum)

        onsource = self._process_raw_onsource(onsource)
        offsource = self._process_offsource(offsource)
        gps_times = self._process_gps_times(gps_times)
        mask = self._process_mask(mask)

        input_dict, output_dict = self._create_output_dictionaries(
            onsource, whitened_onsource, offsource, gps_times, injections,
            whitened_injections, mask, rolling_pearson_onsource, spectrogram_onsource, parameters,
            feature_labels=feature_labels,
            glitch_type=glitch_type,
            source_type=source_type,
            data_label=data_label
        )

        return input_dict, output_dict

    def _process_whitened_injections(self, scaled_injections, spectrum):
        """Whiten the injections (G, B, I, T) with the batch's PSD, if required."""
        if gf.ReturnVariables.WHITENED_INJECTIONS in self.variables_to_return:
            whitened_injections = whiten(scaled_injections, spectrum, self.sample_rate_hertz)
            whitened_injections = gf.crop_samples(
                whitened_injections,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            whitened_injections = ops.cast(whitened_injections, "float32")
            return gf.replace_nan_and_inf_with_zero(whitened_injections)
        return None

    def _process_injections(self, scaled_injections):
        """Process raw injections if required."""
        if gf.ReturnVariables.INJECTIONS in self.variables_to_return:
            # Crop raw injections to match onsource duration
            injections = gf.crop_samples(
                scaled_injections,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            injections = ops.cast(injections, "float32")
            return gf.replace_nan_and_inf_with_zero(injections)
        return None

    def _process_onsource(self, onsource, spectrum):
        """Process onsource data, including whitening, rolling Pearson, and spectrogram generation."""
        if self._needs_whitening:
            whitened_onsource = whiten(onsource, spectrum, self.sample_rate_hertz)
            whitened_onsource = gf.crop_samples(
                whitened_onsource,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            # float32, scrubbed before any cast: a float16 cast turned values
            # above 65504 into inf and then silently into 0.
            whitened_onsource = gf.replace_nan_and_inf_with_zero(
                ops.cast(whitened_onsource, "float32")
            )

            rolling_pearson_onsource = self._calculate_rolling_pearson(whitened_onsource)
            spectrogram_onsource = self._calculate_spectrogram(whitened_onsource)

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
            onsource = gf.crop_samples(
                onsource, 
                self.onsource_duration_seconds, 
                self.sample_rate_hertz
            )
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
        if gf.ReturnVariables.START_GPS_TIME in self.variables_to_return:
            return as_gps_times(gps_times)
        return None

    def _process_mask(self, mask):
        """Process injection masks if required."""
        if gf.ReturnVariables.INJECTION_MASKS in self.variables_to_return:
            mask = ops.cast(mask, "float32")
            return mask
        return None

    def _create_output_dictionaries(self, *args, feature_labels=None, glitch_type=None, source_type=None, data_label=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_dict = create_variable_dictionary(
            self.input_variables, *args, 
            feature_labels=feature_labels,
            glitch_type=glitch_type,
            source_type=source_type,
            data_label=data_label
        )
        output_dict = create_variable_dictionary(
            self.output_variables, *args,
            feature_labels=feature_labels,
            glitch_type=glitch_type,
            source_type=source_type,
            data_label=data_label
        )
        
        # Remove keys with None values
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        
        return input_dict, output_dict

def create_variable_dictionary(
    return_variables: List[Union[gf.ReturnVariables, gf.WaveformParameters, gf.ScalingTypes]],
    onsource,
    whitened_onsource,
    offsource,
    gps_times,
    injections,
    whitened_injections,
    mask,
    rolling_pearson_onsource,
    spectrogram_onsource,
    injection_parameters : Optional[Dict],
    feature_labels = None,
    glitch_type = None,
    source_type = None,
    data_label = None
    ) -> Dict:
    """
    Create dictionary of requested return variables.
    
    Supports ReturnVariables, WaveformParameters, and ScalingTypes. Every
    WaveformParameter and ScalingType comes from ``injection_parameters``; the
    ScalingTypes (SNR, HRSS, HPEAK) are the labels the injection layer
    recorded, never recomputed here.
    
    Args:
        feature_labels: Labels for SUB_TYPE classification (from TransientObtainer).
        glitch_type: GLITCH_TYPE labels (from TransientObtainer).
        source_type: SOURCE_TYPE labels (from TransientObtainer).
        data_label: DATA_LABEL values (from TransientObtainer).
    """
    from keras import ops
    
    operations = {
        gf.ReturnVariables.ONSOURCE: onsource,
        gf.ReturnVariables.WHITENED_ONSOURCE: whitened_onsource,
        gf.ReturnVariables.OFFSOURCE: offsource,
        gf.ReturnVariables.START_GPS_TIME: gps_times,
        gf.ReturnVariables.INJECTIONS: injections,
        gf.ReturnVariables.WHITENED_INJECTIONS: whitened_injections,
        gf.ReturnVariables.INJECTION_MASKS: mask,
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE: rolling_pearson_onsource,
        gf.ReturnVariables.SPECTROGRAM_ONSOURCE: spectrogram_onsource,
        gf.ReturnVariables.SUB_TYPE: feature_labels,
        # Transient-specific outputs
        gf.ReturnVariables.GLITCH_TYPE: glitch_type,
        gf.ReturnVariables.SOURCE_TYPE: source_type,
        gf.ReturnVariables.DATA_LABEL: data_label,
    }

    # Add WaveformParameters from injection_parameters
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

# Legacy wrapper for compatibility
def Dataset(**kwargs):
    return GravyflowDataset(**kwargs)

# Legacy class alias
data = GravyflowDataset


# =============================================================================
# COMPOSED DATASET
# =============================================================================

class ComposedDataset(keras.utils.PyDataset):
    """
    A dataset composed of multiple FeaturePools with different probabilities and labels.
    
    Each pool can be a different data source (noise, glitches, events, injections)
    with its own class label and sampling probability. Multiple pools can share
    the same label for hierarchical classification.
    
    Example:
        pools = [
            gf.FeaturePool(
                name="pure_noise",
                label=0,
                probability=0.5,
                noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ...)
            ),
            gf.FeaturePool(
                name="glitches",
                label=1,
                probability=0.3,
                transient_obtainer=gf.TransientDataObtainer(...)
            ),
            gf.FeaturePool(
                name="cbc_signals",
                label=2,
                probability=0.2,
                noise_obtainer=gf.NoiseObtainer(...),
                injection_generators=[cbc_generator]
            ),
        ]
        
        dataset = gf.ComposedDataset(
            pools=pools,
            sample_rate_hertz=2048.0,
            onsource_duration_seconds=1.0,
            num_examples_per_batch=32,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.POOL_LABEL]
        )
    """
    
    def __init__(
        self,
        pools: List["gf.FeaturePool"],
        seed: Optional[int] = None,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        offsource_duration_seconds: Optional[float] = None,
        crop_duration_seconds: Optional[float] = None,
        scale_factor: Optional[float] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        steps_per_epoch: int = 1000,
        group: str = "train"
    ):
        """
        Initialize the composed dataset.
        
        Args:
            pools: List of FeaturePool objects defining data sources and labels.
            seed: Random seed for reproducibility.
            sample_rate_hertz: Sample rate in Hz.
            onsource_duration_seconds: Duration of onsource window.
            offsource_duration_seconds: Duration of offsource window.
            crop_duration_seconds: Duration to crop from onsource edges.
            scale_factor: Scaling factor for data.
            num_examples_per_generation_batch: Batch size for waveform generation.
            num_examples_per_batch: Batch size for returned data.
            input_variables: Variables to include in input dict.
            output_variables: Variables to include in output dict.
            steps_per_epoch: Number of batches per epoch.
            group: Dataset split (train/val/test).
        """
        super().__init__(workers=0)
        
        from gravyflow.src.dataset.pools import PoolSampler
        
        self.steps_per_epoch = steps_per_epoch
        self.group = group
        
        # Set default values
        self.seed = seed if seed is not None else Defaults.seed
        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else Defaults.crop_duration_seconds
        self.scale_factor = scale_factor if scale_factor is not None else Defaults.scale_factor
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else Defaults.num_examples_per_batch
        self.ifos = ifos if ifos is not None else [gf.IFO.L1]
        
        self.input_variables = input_variables if input_variables is not None else []
        self.output_variables = output_variables if output_variables is not None else []
        self.variables_to_return = set(self.input_variables + self.output_variables)
        
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        
        # Initialize pool sampler
        self.pools = pools
        self.pool_sampler = PoolSampler(pools, seed=self.seed)
        
        # Set random seeds
        gf.set_random_seeds(self.seed)
        
        # Initialize generators for each pool
        self._pool_generators = {}
        self._pool_injection_generators = {}
        self._initialize_pool_generators()
    
    def _initialize_pool_generators(self):
        """Initialize data generators for each pool."""
        for i, pool in enumerate(self.pools):
            generator_kwargs = {
                'sample_rate_hertz': self.sample_rate_hertz,
                'onsource_duration_seconds': self.onsource_duration_seconds,
                'crop_duration_seconds': self.crop_duration_seconds,
                'offsource_duration_seconds': self.offsource_duration_seconds,
                'num_examples_per_batch': self.num_examples_per_batch,
                'ifos': self.ifos,
                'scale_factor': self.scale_factor,
                'group': self.group,
                'seed': self.seed + i,  # Unique seed per pool
            }
            
            if pool.is_noise_pool:
                # Add sampling_mode for noise obtainer
                generator_kwargs['sampling_mode'] = gf.SamplingMode.RANDOM
                self._pool_generators[i] = pool.noise_obtainer(**generator_kwargs)
            else:
                # Transient obtainer doesn't need sampling_mode
                self._pool_generators[i] = pool.transient_obtainer(**generator_kwargs)
            
            # Initialize injection generator if pool has injections
            if pool.has_injections:
                waveform_parameters_to_return = [
                    item for item in self.variables_to_return 
                    if isinstance(item.value, gf.WaveformParameter)
                ]
                
                # Set network for generators if not set
                for gen in pool.injection_generators:
                    if gen.network is None:
                        gen.network = gf.Network(pool.noise_obtainer.ifos)
                
                injection_gen = gf.InjectionGenerator(
                    waveform_generators=pool.injection_generators,
                    parameters_to_return=waveform_parameters_to_return,
                    seed=self.seed + i + 1000
                )
                self._pool_injection_generators[i] = injection_gen(
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
        Generate the next batch of data from multiple pools.
        """
        # Get pool assignments for this batch
        pool_assignments = self.pool_sampler.get_pool_batch_sizes(self.num_examples_per_batch)
        
        # Storage for batch data
        batch_onsource = []
        batch_offsource = []
        batch_pool_labels = []
        batch_gps_times = []
        batch_glitch_type = []
        batch_source_type = []
        batch_data_label = []
        batch_scaled_injections = []
        batch_parameters = {}
        
        # Collect per-position data
        position_data = {}
        
        for pool_idx, (count, positions) in pool_assignments.items():
            pool = self.pools[pool_idx]
            generator = self._pool_generators[pool_idx]
            
            # Generate data from this pool
            try:
                batch_data = next(generator)
                
                # Handle dict format (new) or tuple format (legacy)
                if isinstance(batch_data, dict):
                    onsource = batch_data[gf.ReturnVariables.ONSOURCE]
                    offsource = batch_data[gf.ReturnVariables.OFFSOURCE]
                    gps_times = batch_data.get(
                        gf.ReturnVariables.START_GPS_TIME,
                        batch_data.get(gf.ReturnVariables.TRANSIENT_GPS_TIME)
                    )
                    glitch_type = batch_data.get(gf.ReturnVariables.GLITCH_TYPE)
                    source_type = batch_data.get(gf.ReturnVariables.SOURCE_TYPE)
                    data_label = batch_data.get(gf.ReturnVariables.DATA_LABEL)
                else:
                    if len(batch_data) >= 3:
                        onsource, offsource, gps_times = batch_data[:3]
                    else:
                        onsource, offsource = batch_data[:2]
                        gps_times = None
                    glitch_type, source_type, data_label = None, None, None
            except Exception as e:
                logger.error(f"Data generation failed for pool '{pool.name}': {e}")
                raise
            
            # Apply injections if pool has them
            scaled_injections = None
            parameters = {}
            if pool.has_injections and pool_idx in self._pool_injection_generators:
                try:
                    injections, mask, parameters = next(self._pool_injection_generators[pool_idx])
                    
                    injection_gen = gf.InjectionGenerator(
                        waveform_generators=pool.injection_generators,
                        parameters_to_return=[],
                        seed=self.seed + pool_idx
                    )
                    injection_gen.sample_rate_hertz = self.sample_rate_hertz
                    
                    onsource, scaled_injections, scaling_params = injection_gen.add_injections_to_onsource(
                        injections,
                        mask,
                        onsource,
                        offsource,
                        parameters_to_return=self.variables_to_return,
                        onsource_duration_seconds=self.onsource_duration_seconds,
                        injection_parameters=parameters
                    )
                    parameters.update(scaling_params)
                except Exception as e:
                    logger.error(f"Injection generation failed for pool '{pool.name}': {e}")
                    raise
            
            # Map data to positions
            for local_idx, global_pos in enumerate(positions):
                if local_idx < ops.shape(onsource)[0]:
                    position_data[global_pos] = {
                        'onsource': onsource[local_idx],
                        'offsource': offsource[local_idx],
                        'pool_label': pool.label,
                        'gps_time': gps_times[local_idx] if gps_times is not None else -1.0,
                        'glitch_type': glitch_type[local_idx] if glitch_type is not None else -1,
                        'source_type': source_type[local_idx] if source_type is not None else -1,
                        'data_label': data_label[local_idx] if data_label is not None else pool.label,
                        'scaled_injections': scaled_injections[..., local_idx, :, :] if scaled_injections is not None else None,
                        'parameters': {k: v[local_idx] if ops.is_tensor(v) else v for k, v in parameters.items()}
                    }
        
        # Assemble final batch in order
        for pos in range(self.num_examples_per_batch):
            if pos in position_data:
                data = position_data[pos]
                batch_onsource.append(data['onsource'])
                batch_offsource.append(data['offsource'])
                batch_pool_labels.append(data['pool_label'])
                batch_gps_times.append(data['gps_time'])
                batch_glitch_type.append(data['glitch_type'])
                batch_source_type.append(data['source_type'])
                batch_data_label.append(data['data_label'])
                if data['scaled_injections'] is not None:
                    batch_scaled_injections.append(data['scaled_injections'])
                for k, v in data['parameters'].items():
                    if k not in batch_parameters:
                        batch_parameters[k] = []
                    batch_parameters[k].append(v)
        
        # Stack into tensors
        onsource = ops.stack(batch_onsource, axis=0)
        offsource = ops.stack(batch_offsource, axis=0)
        pool_labels = ops.convert_to_tensor(batch_pool_labels, dtype="int32")
        gps_times = as_gps_times(batch_gps_times)
        glitch_type = ops.convert_to_tensor(batch_glitch_type, dtype="int32")
        source_type = ops.convert_to_tensor(batch_source_type, dtype="int32")
        data_label = ops.convert_to_tensor(batch_data_label, dtype="int32")
        
        scaled_injections = None
        if batch_scaled_injections:
            try:
                scaled_injections = ops.stack(batch_scaled_injections, axis=0)
            except:
                scaled_injections = None
        
        # Stack parameters
        for k, v in batch_parameters.items():
            try:
                batch_parameters[k] = ops.stack(v, axis=0)
            except:
                pass
        
        # Process data (whitening, etc.)
        whitened_onsource = None
        if gf.ReturnVariables.WHITENED_ONSOURCE in self.variables_to_return:
            whitened_onsource = whiten(onsource, offsource, self.sample_rate_hertz)
            whitened_onsource = gf.crop_samples(
                whitened_onsource,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            whitened_onsource = ops.cast(whitened_onsource, "float16")
            whitened_onsource = gf.replace_nan_and_inf_with_zero(whitened_onsource)
        
        # Build return dictionaries
        operations = {
            gf.ReturnVariables.ONSOURCE: gf.crop_samples(onsource, self.onsource_duration_seconds, self.sample_rate_hertz) if gf.ReturnVariables.ONSOURCE in self.variables_to_return else None,
            gf.ReturnVariables.WHITENED_ONSOURCE: whitened_onsource,
            gf.ReturnVariables.OFFSOURCE: offsource if gf.ReturnVariables.OFFSOURCE in self.variables_to_return else None,
            gf.ReturnVariables.START_GPS_TIME: gps_times if gf.ReturnVariables.START_GPS_TIME in self.variables_to_return else None,
            gf.ReturnVariables.POOL_LABEL: pool_labels,
            gf.ReturnVariables.GLITCH_TYPE: glitch_type if gf.ReturnVariables.GLITCH_TYPE in self.variables_to_return else None,
            gf.ReturnVariables.SOURCE_TYPE: source_type if gf.ReturnVariables.SOURCE_TYPE in self.variables_to_return else None,
            gf.ReturnVariables.DATA_LABEL: data_label if gf.ReturnVariables.DATA_LABEL in self.variables_to_return else None,
        }
        
        # Add waveform parameters
        operations.update(batch_parameters)
        
        input_dict = {
            key.name: operations[key] for key in self.input_variables
            if key in operations and operations[key] is not None
        }
        output_dict = {
            key.name: operations[key] for key in self.output_variables
            if key in operations and operations[key] is not None
        }
        
        return input_dict, output_dict
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.__getitem__(0)