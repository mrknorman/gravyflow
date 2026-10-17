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
from gravyflow.src.utils.tensor import (
    crop_samples, replace_nan_and_inf_with_zero, window_samples
)
from gravyflow.src.utils.shapes import BatchContract
from gravyflow.src.dataset.config import Defaults, ignore_scale_factor
from gravyflow.src.dataset.noise.noise import index_addressed, obtainer_batcher
from gravyflow.src.dataset.features.injection import InjectionGenerator, check_scaling_matches_noise
from gravyflow.src.rng import BatchStreams, SeedTree, check_coordinate, check_name, check_split
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
                "Simulated noise has no GPS time: START_GPS_TIME is NaN.",
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
            
def noise_ifos(ifos, noise_obtainer) -> List:
    """
    The detectors a dataset's noise has, and so its generators' network: the
    dataset's own ``ifos``, else its noise obtainer's, else L1.
    """
    if ifos is not None:
        return ensure_list(ifos)
    obtainer_ifos = getattr(noise_obtainer, "ifos", None)
    return ensure_list(obtainer_ifos) if obtainer_ifos is not None else [gf.IFO.L1]


def get_max_arrival_time_difference(ifos: List[gf.IFO]) -> float:
    """
    The lag range of ``ROLLING_PEARSON_ONSOURCE`` (s): the largest light travel
    time between ``ifos``, the detectors the dataset uses (``noise_ifos``; every
    generator's network is bound to them). Noise-only and injection batches
    therefore have the same shape.

    A Python float, because `rolling_pearson` takes it as a static argument.
    """
    return float(gf.Network(ensure_list(ifos)).max_arrival_time_difference_seconds)

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


class EpochMode(Enum):
    """How a dataset's batches change from epoch to epoch."""
    FRESH = auto()   # New draws every epoch: batch k of epoch e is addressed by (e, k).
    REPEAT = auto()  # The same batches every epoch: epoch 0's.


class _KerasEpoch(keras.callbacks.Callback):
    """Sets a dataset's epoch to Keras' epoch, before Keras reads its batches."""

    def __init__(self, dataset: "IndexedBatches"):
        super().__init__()
        self.dataset = dataset

    def on_train_begin(self, logs=None):
        # With fit(steps_per_epoch=) other than the dataset's length, Keras
        # keeps one iterator across epochs and fetches ahead of
        # on_epoch_begin, so an epoch would train batches addressed to the
        # epoch before.
        steps = self.params.get("steps")
        if steps is not None and steps != len(self.dataset):
            raise ValueError(
                f"fit(steps_per_epoch={steps}) differs from the dataset's {len(self.dataset)} batches "
                "per epoch, which would serve one epoch's batches in another. Give the dataset "
                "steps_per_epoch instead, and leave fit's unset."
            )

    def on_epoch_begin(self, epoch, logs=None):
        self.dataset.epoch = epoch


class IndexedBatches:
    """
    Batches addressed by index: ``ds[k]`` is a pure function of the dataset's
    specification, its seed tree, its split, its effective epoch and k.

    The training split (EpochMode.FRESH, the default for ``group="train"``)
    draws fresh batches every epoch. Other splits (EpochMode.REPEAT) always
    serve epoch 0, so an evaluation set is the same in every epoch and every
    run.

    The epoch moves on with ``on_epoch_end``: a loop over ``ds[k]`` or
    ``iter(ds)`` calls it after each pass, and ``next(ds)``, a cursor over
    (epoch, k), calls it itself. Under ``model.fit`` the epoch is Keras' own,
    from ``ds.epoch_callback()``.

    REAL, PSEUDO_REAL and transient noise are still read in order, whatever
    the address (``seed_metadata["index_addressed"]`` is False): their batches
    depend on the order they are asked for, so their evaluation sets are not
    yet fixed across epochs or runs.
    """

    def _init_batches(self, seed, group: str, steps_per_epoch: int, epoch_mode: Optional[EpochMode],
                      name: Optional[str] = None):
        self.name = None if name is None else check_name(name, "dataset name")
        self.tree = SeedTree.of(seed)
        if self.name is not None:
            self.tree = self.tree.sub(f"dataset/{self.name}")
        self.seed = self.tree.root
        self.group = check_split(group)
        self.steps_per_epoch = steps_per_epoch
        if epoch_mode is None:
            epoch_mode = EpochMode.FRESH if group == "train" else EpochMode.REPEAT
        self.epoch_mode = epoch_mode
        self._epoch = 0
        self._cursor = 0
        self._epoch_from_keras = False
        self._warned_keras_epoch = False

    @property
    def epoch(self) -> int:
        """The training epoch: it selects the batches and any Curriculum's distribution."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        """Set the epoch, e.g. to resume a run (Keras' ``initial_epoch``)."""
        self._epoch = check_coordinate(epoch, "epoch")

    @property
    def _batch_epoch(self) -> int:
        return self._epoch if self.epoch_mode is EpochMode.FRESH else 0

    @property
    def _schedule_epoch(self) -> Optional[int]:
        """The epoch a Curriculum is read at: training's, else its end (None)."""
        return self._epoch if self.epoch_mode is EpochMode.FRESH else None

    def streams(self, index: int) -> BatchStreams:
        """The streams of batch ``index`` at the current epoch."""
        index = check_coordinate(index, "batch index")
        if index >= len(self):
            raise IndexError(f"Batch {index} is outside the {len(self)} batches of an epoch.")
        return self.tree.batch(self.group, self._batch_epoch, index)

    def __len__(self):
        return self.steps_per_epoch

    def epoch_callback(self) -> keras.callbacks.Callback:
        """
        The callback that gives this dataset Keras' epoch:
        ``model.fit(ds, ..., callbacks=[ds.epoch_callback()])``.

        Keras also calls ``on_epoch_end`` when ``fit`` starts, before its
        first epoch, so counting those calls would serve Keras epoch e the
        batches (and Curriculum) of epoch e + 1. Only a callback is told
        Keras' epoch: with it, Keras epoch e, ``initial_epoch`` included,
        serves epoch e. So a second ``fit`` without ``initial_epoch`` starts
        again at epoch 0 and serves the first's batches; continue a run with
        ``initial_epoch``. ``fit(steps_per_epoch=)`` must be unset or the
        dataset's length (it is refused otherwise).
        """
        self._epoch_from_keras = True
        return _KerasEpoch(self)

    def on_epoch_begin(self):
        """Keras calls this before each epoch's batches; a plain loop never does."""
        if (
            self.epoch_mode is EpochMode.FRESH
            and not self._epoch_from_keras
            and not self._warned_keras_epoch
        ):
            self._warned_keras_epoch = True
            warn(
                f"The '{self.group}' split draws fresh batches each epoch, but Keras "
                "is driving it without its epoch: Keras calls on_epoch_end once "
                "before the first epoch, so each Keras epoch would serve the next "
                "epoch's batches. Pass callbacks=[dataset.epoch_callback()] to fit.",
                UserWarning,
                stacklevel=2,
            )

    def on_epoch_end(self):
        if self.epoch_mode is EpochMode.FRESH:
            self._epoch += 1

    def __next__(self):
        if self._cursor >= len(self):
            self._cursor = 0
            self.on_epoch_end()
        batch = self[self._cursor]
        self._cursor += 1
        return batch

    @property
    def seed_metadata(self) -> dict:
        """Where the data's randomness came from, to record beside a model."""
        return {
            **self.tree.metadata(),
            "split": self.group,
            "split_fingerprint": self.split_spec.fingerprint(),
            "epoch_mode": self.epoch_mode.name,
            "index_addressed": self.index_addressed,
        }

    @property
    def split_spec(self) -> "gf.SplitSpec":
        """
        How the real data this dataset reads is split: the spec of the IFO data
        obtainers it reads through (a NoiseObtainer's, or a transient obtainer
        itself), else the default. This is the one computation of a split:
        everything else that needs one, the validator and the training runs
        included, reads it from data.
        """
        specs = {getattr(getattr(obtainer, "ifo_data_obtainer", obtainer), "split_spec", None)
                 for obtainer in self._obtainers()}
        specs.discard(None)
        if len(specs) > 1:
            raise ValueError(f"A dataset's obtainers split time differently: {sorted(s.fingerprint() for s in specs)}.")
        return specs.pop() if specs else gf.DEFAULT_SPLIT

    @property
    def index_addressed(self) -> bool:
        """Whether every batch is a pure function of its address (see the class docstring)."""
        return all(index_addressed(obtainer) for obtainer in self._obtainers())


class GravyflowDataset(IndexedBatches, keras.utils.PyDataset):
    """
    A Keras PyDataset to generate data for gravitational wave detection training.

    Randomness: one seed tree, rooted at ``seed`` (default ``gf.Defaults.seed``),
    with every draw addressed by (split ``group``, epoch, batch k, consumer);
    see IndexedBatches, which also says which noise is not yet read by
    address. Under ``model.fit``, pass ``callbacks=[ds.epoch_callback()]`` for
    a training split. Different splits never share a stream, so train,
    validate and test built from one seed are independent.

    Within one split, datasets of one seed draw the same streams (common
    random numbers): a noise-only dataset and a signal dataset share their
    noise, so an ablation compares like with like. Datasets that must be
    independent, such as the classes of a classifier, take distinct
    ``name``s, each a scope of the seed tree, or are the pools of one
    ComposedDataset. The dataset works
    on its own copies of the obtainer and generators it is given, so the
    caller's objects are never changed and can serve other datasets.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        offsource_duration_seconds: Optional[float] = None,
        crop_duration_seconds: Optional[float] = None,
        scale_factor: None = None,
        noise_obtainer: Optional[gf.NoiseObtainer] = None,
        group: str = None,
        waveform_generators: Optional[List[Union[gf.CBCGenerator, gf.WNBGenerator]]] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        mask_history: Optional[List] = None,
        steps_per_epoch: int = 1000,
        sampling_mode: Optional["gf.SamplingMode"] = None,
        spectral_config: Optional[SpectralConfig] = None,
        epoch_mode: Optional[EpochMode] = None,
        name: Optional[str] = None,
    ):
        """
        seed: the root of the seed tree, an integer >= 0 or a gf.SeedTree.
        name: the dataset's scope in the seed tree (letters, digits and
            _.:+-). Datasets of one seed and split with different names draw
            independent streams; unnamed, they share them.
        epoch_mode: EpochMode.FRESH (default for group="train") or REPEAT.
        spectral_config: the band, PSD estimator and whitening filter shared by
            SNR scaling, SNR labels and whitening (gf.SpectralConfig; default
            band 20 Hz to Nyquist, 1 s / 0.5 s median Welch PSD, 1 s filter).
            One PSD is estimated per batch from the offsource and reused by all
            three.
        scale_factor: deprecated and ignored. Every input, output and label is
            physical strain (or derived from it: whitened data, SNRs).
        """
        ignore_scale_factor(scale_factor)
        # Initialize PyDataset (single-threaded to avoid JAX/fork deadlocks)
        super().__init__(workers=0)
        self._init_batches(seed, group, steps_per_epoch, epoch_mode, name)

        # Set default values
        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else gf.Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else gf.Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else gf.Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else gf.Defaults.crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else gf.Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else gf.Defaults.num_examples_per_batch
        
        # The dataset's own copies: the caller's objects are never changed.
        self.noise_obtainer = copy.deepcopy(noise_obtainer) if noise_obtainer is not None else gf.NoiseObtainer()
        self.ifos = ensure_list(ifos) if ifos is not None else None
        
        # Default to RANDOM mode if not specified
        self.sampling_mode = sampling_mode if sampling_mode is not None else gf.SamplingMode.RANDOM
        self.input_variables = input_variables if input_variables is not None else []
        self.output_variables = output_variables if output_variables is not None else []
        self.mask_history = mask_history
        self.spectral_config = spectral_config if spectral_config is not None else SpectralConfig()
        
        # Initialize waveform generators
        self.waveform_generators = copy.deepcopy(waveform_generators) if waveform_generators is not None else []
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        # Set network for waveform generators if not provided
        self._set_generator_networks()

        # Validate settings
        self._validate_settings()
        self.batch_contract = self._batch_contract()

        # Create noise and injection generators
        self._create_generators()

    def _noise_ifos(self) -> List:
        """The detectors the noise will have (``noise_ifos``)."""
        return noise_ifos(self.ifos, self.noise_obtainer)

    def _set_generator_networks(self):
        """
        Give every generator a network of exactly the noise's detectors, in order.

        A generator without a network (the dataset's own copy) gets one built
        from the noise's detectors. A generator with its own network must match
        the noise's detectors; a mismatch would broadcast one projection into
        several channels or mislabel them, so it raises.
        """
        noise_ifos = self._noise_ifos()

        def bind(generator):
            if generator.network is None:
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
        check_scaling_matches_noise(self.noise_obtainer, self._generator_list())
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

    def _batch_contract(self) -> BatchContract:
        """The shapes and dtypes every batch of this dataset is checked against."""
        return BatchContract(
            batch=self.num_examples_per_batch,
            generators=len(self._generator_list()),
            ifos=len(self._noise_ifos()),
            onsource_samples=window_samples(self.onsource_duration_seconds, self.sample_rate_hertz),
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
                if getattr(generator, "scaling_method", None) is None:
                    raise ValueError(
                        f"{type(generator).__name__} has no scaling_method; give it a "
                        "gf.ScalingMethod to inject it into a dataset."
                    )
                # None means "as far as the window allows".
                front = generator.front_padding_duration_seconds or 0.0
                back = generator.back_padding_duration_seconds or 0.0

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
            'sampling_mode': self.sampling_mode,
        }
        # Add ifos to noise_kwargs if provided
        if self.ifos is not None:
            noise_kwargs['ifos'] = self.ifos

        self.noise = obtainer_batcher(self.noise_obtainer, self.tree, self.group, **noise_kwargs)

        waveform_parameters_to_return = [
            item for item in self.variables_to_return if isinstance(
                item.value, gf.WaveformParameter
            )
        ]
        self.injection_generator = gf.InjectionGenerator(
            waveform_generators=self.waveform_generators,
            parameters_to_return=waveform_parameters_to_return,
        )

    def _obtainers(self) -> List:
        return [self.noise_obtainer]

    def __getitem__(self, index) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Batch ``index`` of the current epoch: a pure function of its address
        when the noise is (see IndexedBatches).
        """
        streams = self.streams(index)
        try:
            batch_data = self.noise(streams)
            
            # Handle dict format (new) or tuple format (legacy)
            if isinstance(batch_data, dict):
                onsource = batch_data[gf.ReturnVariables.ONSOURCE]
                offsource = batch_data[gf.ReturnVariables.OFFSOURCE]
                # The first returned onsource sample's time, in every mode.
                gps_times = batch_data.get(gf.ReturnVariables.START_GPS_TIME)
                feature_labels = batch_data.get(gf.ReturnVariables.SUB_TYPE)
                # Transient-specific outputs
                glitch_type = batch_data.get(gf.ReturnVariables.GLITCH_TYPE)
                source_type = batch_data.get(gf.ReturnVariables.SOURCE_TYPE)
                data_label = batch_data.get(gf.ReturnVariables.DATA_LABEL)
                # A composed dataset's noise names each example's pool.
                pool_label = batch_data.get(gf.ReturnVariables.POOL_LABEL)
            elif len(batch_data) == 4:
                onsource, offsource, gps_times, feature_labels = batch_data
                glitch_type, source_type, data_label, pool_label = None, None, None, None
            else:
                onsource, offsource, gps_times = batch_data
                feature_labels = None
                glitch_type, source_type, data_label, pool_label = None, None, None, None
        except Exception as e:
            logger.info(f"Noise generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Noise generation failed: {e}")

        try:
            injections, mask, parameters = self.injection_generator.generate(
                streams,
                sample_rate_hertz=self.sample_rate_hertz,
                onsource_duration_seconds=self.onsource_duration_seconds,
                crop_duration_seconds=self.crop_duration_seconds,
                num_examples_per_batch=self.num_examples_per_batch,
            )
        except Exception as e:
            logger.info(f"Injection generation failed: {e}\nTraceback: {traceback.format_exc()}")
            raise Exception(f"Injection generation failed: {e}")

        # One PSD per batch: scaling, labels and whitening all read this estimate.
        spectrum = self._batch_spectrum(offsource) if self._needs_spectrum else None

        try:
            onsource, scaled_injections, labels = self.injection_generator.add_injections_to_onsource(
                injections,
                mask,
                onsource,
                spectrum if spectrum is not None else offsource,
                parameters_to_return=self.variables_to_return,
                onsource_duration_seconds=self.onsource_duration_seconds,
                injection_parameters=parameters,
                streams=streams,
                epoch=self._schedule_epoch,
                steps_per_epoch=len(self),
            )
        except Exception as e:
            logger.error(f"Failed to add injections to onsource: {e}\nTraceback: {traceback.format_exc()}")
            raise e

        whitened_injections = self._process_whitened_injections(scaled_injections, spectrum)
        injections = self._process_injections(scaled_injections)
        labels = self._batch_labels(labels, mask)

        whitened_onsource, rolling_pearson_onsource, spectrogram_onsource = self._process_onsource(onsource, spectrum)

        onsource = self._process_raw_onsource(onsource)
        offsource = self._process_offsource(offsource)
        gps_times = self._process_gps_times(gps_times)

        input_dict, output_dict = self._create_output_dictionaries(
            onsource, whitened_onsource, offsource, gps_times, injections,
            whitened_injections, labels[gf.ReturnVariables.INJECTION_MASKS],
            rolling_pearson_onsource, spectrogram_onsource, labels,
            feature_labels=self._process_transient_label(feature_labels),
            glitch_type=self._process_transient_label(glitch_type),
            source_type=self._process_transient_label(source_type),
            data_label=self._process_transient_label(data_label),
            pool_label=self._process_transient_label(pool_label),
        )
        self.batch_contract.check(
            (self.input_variables, self.output_variables), input_dict, output_dict
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
            return _batch_first(gf.replace_nan_and_inf_with_zero(
                whitened_injections, gf.ReturnVariables.WHITENED_INJECTIONS.name
            ))
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
            return _batch_first(gf.replace_nan_and_inf_with_zero(
                injections, gf.ReturnVariables.INJECTIONS.name
            ))
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
                ops.cast(whitened_onsource, "float32"), gf.ReturnVariables.WHITENED_ONSOURCE.name
            )

            rolling_pearson_onsource = self._calculate_rolling_pearson(whitened_onsource)
            spectrogram_onsource = self._calculate_spectrogram(whitened_onsource)

            return whitened_onsource, rolling_pearson_onsource, spectrogram_onsource
        return None, None, None

    def _calculate_rolling_pearson(self, whitened_onsource):
        """Calculate rolling Pearson correlation if required."""
        if gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in self.variables_to_return:
            max_arrival_time_difference_seconds = get_max_arrival_time_difference(
                self._noise_ifos()
            )
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
            return gf.replace_nan_and_inf_with_zero(onsource, gf.ReturnVariables.ONSOURCE.name)
        return None

    def _process_offsource(self, offsource):
        """Process offsource data if required."""
        if gf.ReturnVariables.OFFSOURCE in self.variables_to_return:
            offsource = ops.cast(offsource, "float32")
            return gf.replace_nan_and_inf_with_zero(offsource, gf.ReturnVariables.OFFSOURCE.name)
        return None

    def _process_gps_times(self, gps_times):
        """GPS times (B, I), float64, if required; NaN where the noise has none."""
        if gf.ReturnVariables.START_GPS_TIME in self.variables_to_return:
            if gps_times is None:
                return np.full(self.batch_contract.shape(gf.ReturnVariables.START_GPS_TIME), np.nan)
            return as_gps_times(gps_times)
        return None

    @staticmethod
    def _process_transient_label(label):
        """A class label (a transient's ids per detector, a pool's) as float32, like every label."""
        return None if label is None else ops.cast(label, "float32")

    def _batch_labels(self, labels, mask):
        """
        The batch's labels, batch-first: INJECTION_MASKS (B, G),
        INJECTION_PRESENT (B,), and every per-generator label requested,
        (B, G) + its own shape, float32. A label is NaN where generator g
        injected nothing into example b (the injection layer's rule), and
        wholly NaN when no generator draws it.
        """
        mask = ops.transpose(ops.cast(mask, "float32"))
        batch = {
            gf.ReturnVariables.INJECTION_MASKS: mask,
            gf.ReturnVariables.INJECTION_PRESENT: ops.cast(ops.any(mask > 0, axis=1), "float32"),
        }
        for key in self.variables_to_return:
            if isinstance(key, (gf.WaveformParameters, gf.ScalingTypes)) \
                    or key is gf.ReturnVariables.CENTRAL_TIME:
                value = labels.get(key)
                batch[key] = _batch_first(value) if value is not None else \
                    ops.full(self.batch_contract.shape(key), np.nan, dtype="float32")
        return batch

    def _create_output_dictionaries(self, *args, **labels) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_dict = create_variable_dictionary(self.input_variables, *args, **labels)
        output_dict = create_variable_dictionary(self.output_variables, *args, **labels)
        
        # Remove keys with None values
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        
        return input_dict, output_dict

def _batch_first(value):
    """A generator-first (G, B, ...) array as batch-first (B, G, ...)."""
    return ops.moveaxis(ops.convert_to_tensor(value), 0, 1)


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
    data_label = None,
    pool_label = None
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
        pool_label: POOL_LABEL values (a ComposedDataset's pools).
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
        gf.ReturnVariables.POOL_LABEL: pool_label,
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

def _noise_defaults() -> Dict:
    """
    What an example carries in a composed batch where its pool's source gives
    no such output (synthetic noise): no GPS time, and the labels a transient
    obtainer gives a partner detector, noise with no type.
    """
    return {
        gf.ReturnVariables.START_GPS_TIME: (np.nan, np.float64),
        gf.ReturnVariables.DATA_LABEL: (gf.DataLabel.NOISE.value, np.int32),
        gf.ReturnVariables.SUB_TYPE: (-1, np.int32),
        gf.ReturnVariables.GLITCH_TYPE: (-1, np.int32),
        gf.ReturnVariables.SOURCE_TYPE: (-1, np.int32),
    }


class _PoolInjections(InjectionGenerator):
    """
    A composed dataset's injection layer: the generators of every pool, in
    pool order, each drawing from its pool's streams (``pool/{name}/``) and
    injecting only into the examples assigned to its pool. Elsewhere its
    waveform is zero and its mask 0, so its labels there are NaN.
    """

    def __init__(self, pools, pool_of, generators, assignment, parameters_to_return):
        super().__init__(
            waveform_generators=generators,
            parameters_to_return=parameters_to_return,
            scopes=[f"pool/{pools[index].name}" for index in pool_of],
        )
        self._pool_of = np.asarray(pool_of, dtype=int)
        self._assignment = assignment

    def generate(self, streams, **window):
        injections, masks, parameters = super().generate(streams, **window)
        # (G, B): generator g's pool holds example b.
        assigned = (self._pool_of[:, None] == self._assignment(streams)[None, :]).astype(np.float32)
        rows = assigned.reshape(assigned.shape + (1,) * (len(ops.shape(injections)) - 2))
        return injections * rows, masks * assigned, parameters


class ComposedDataset(GravyflowDataset):
    """
    A dataset composed of FeaturePools, each a source of examples with its
    own class label (POOL_LABEL) and sampling probability.

    Each example's pool is drawn from the batch's ``pool/assign`` stream, and
    example b is row b of its pool's batch. Every draw of a pool sits under
    ``pool/{name}/``, so pools never share a stream, and a pool's injection
    generators inject into its own examples only.

    Otherwise a composed batch is a GravyflowDataset's, built by the same code
    (see GravyflowDataset and IndexedBatches): one PSD per batch for scaling,
    labels and whitening, and the batch contract, where the generators G are
    every pool's, in pool order; a generator is absent (mask 0, labels NaN)
    from the other pools' examples. Where a pool's source gives no GPS time or
    transient labels (synthetic noise), its examples have NaN, DATA_LABEL
    NOISE and -1 for the types, as a transient obtainer labels a partner
    detector. With one pool, a composed dataset serves exactly what a
    GravyflowDataset of seed ``gf.SeedTree(seed).sub(f"pool/{name}")`` does.

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
        scale_factor: None = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        steps_per_epoch: int = 1000,
        group: str = None,
        epoch_mode: Optional[EpochMode] = None,
        spectral_config: Optional[SpectralConfig] = None,
    ):
        """
        Args:
            pools: the FeaturePools the examples are drawn from.
            seed: The root of the seed tree (an integer >= 0 or a gf.SeedTree).
            sample_rate_hertz: Sample rate in Hz.
            onsource_duration_seconds: Duration of onsource window.
            offsource_duration_seconds: Duration of offsource window.
            crop_duration_seconds: Duration to crop from onsource edges.
            scale_factor: Deprecated and ignored: the data are physical strain.
            num_examples_per_generation_batch: Batch size for waveform generation.
            num_examples_per_batch: Batch size for returned data.
            ifos: the detectors of every pool; default the detectors the
                pools' sources share, else L1.
            input_variables: Variables to include in input dict.
            output_variables: Variables to include in output dict.
            steps_per_epoch: Number of batches per epoch.
            group: Dataset split (train/val/test).
            epoch_mode: EpochMode.FRESH (default for group="train") or REPEAT.
            spectral_config: as GravyflowDataset's.
        """
        from gravyflow.src.dataset.pools import PoolSampler

        ignore_scale_factor(scale_factor)
        # PyDataset's own initialiser: GravyflowDataset's builds from one obtainer.
        keras.utils.PyDataset.__init__(self, workers=0)
        self._init_batches(seed, group, steps_per_epoch, epoch_mode)

        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else Defaults.crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else Defaults.num_examples_per_batch
        self.ifos = ensure_list(ifos) if ifos is not None else None
        self.sampling_mode = gf.SamplingMode.RANDOM
        self.input_variables = input_variables if input_variables is not None else []
        self.output_variables = output_variables if output_variables is not None else []
        self.mask_history = None
        self.spectral_config = spectral_config if spectral_config is not None else SpectralConfig()

        # The dataset's own copies: the caller's pools are never changed.
        self.pools = copy.deepcopy(pools)
        self.pool_sampler = PoolSampler(self.pools)
        self._generator_pools = [
            index for index, pool in enumerate(self.pools) for _ in pool.injection_generators or []
        ]
        self.waveform_generators = [
            generator for pool in self.pools for generator in pool.injection_generators or []
        ]

        self._set_generator_networks()
        self._validate_settings()
        self.batch_contract = self._batch_contract()
        self._create_generators()

    def _obtainers(self) -> List:
        return [
            pool.noise_obtainer if pool.is_noise_pool else pool.transient_obtainer
            for pool in self.pools
        ]

    def _noise_ifos(self) -> List:
        """
        The detectors of every pool's noise: the dataset's ``ifos``, else the
        detectors the pools' sources share, else L1.
        """
        if self.ifos is not None:
            return self.ifos
        sources = {}
        for obtainer in self._obtainers():
            ifos = getattr(obtainer, "ifos", None)
            if ifos is not None:
                sources[tuple(_ifo_names(ifos))] = ensure_list(ifos)
        if len(sources) > 1:
            raise ValueError(
                f"The pools' sources have different detectors, {sorted(sources)}: give the "
                "dataset the ifos every pool should have."
            )
        return next(iter(sources.values()), [gf.IFO.L1])

    def _validate_settings(self):
        """GravyflowDataset's checks, with each pool's source checked as its noise."""
        self.variables_to_return = set(self.input_variables + self.output_variables)
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        for obtainer in self._obtainers():
            validate_noise_settings(obtainer, self.variables_to_return)
        self._validate_padding_settings()
        self._validate_spectral_settings()

    def _create_generators(self):
        """Each pool's noise batcher, the mixture of them, and the pools' injection layer."""
        window = {
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            'crop_duration_seconds': self.crop_duration_seconds,
            'offsource_duration_seconds': self.offsource_duration_seconds,
            'num_examples_per_batch': self.num_examples_per_batch,
            'ifos': self._noise_ifos(),
            'sampling_mode': self.sampling_mode,
        }
        self._pool_noise = [
            obtainer_batcher(obtainer, self.tree.sub(f"pool/{pool.name}"), self.group, **window)
            for pool, obtainer in zip(self.pools, self._obtainers())
        ]
        self.noise = self._mixed_noise
        self.injection_generator = _PoolInjections(
            self.pools, self._generator_pools, self.waveform_generators, self._assignment,
            parameters_to_return=[
                item for item in self.variables_to_return if isinstance(item.value, gf.WaveformParameter)
            ],
        )

    def _assignment(self, streams: BatchStreams) -> np.ndarray:
        """Each example's pool index, drawn from the batch's ``pool/assign`` stream."""
        return self.pool_sampler.sample_pool_indices(
            self.num_examples_per_batch, streams.stream("pool/assign")
        )

    def _mixed_noise(self, streams: BatchStreams) -> Dict:
        """
        The batch's noise, as an obtainer's batch: example b is row b of its
        pool's batch (drawn from ``pool/{name}/``), with its pool's label.
        Only the pools the batch holds are read.
        """
        assignment = self._assignment(streams)
        defaults = _noise_defaults()
        mixed = {}
        for index in np.unique(assignment):
            pool = self.pools[index]
            batch = self._pool_noise[index](streams.sub(f"pool/{pool.name}"))
            rows = assignment == index
            per_detector = tuple(np.shape(batch[gf.ReturnVariables.ONSOURCE])[:2])
            for key in (gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE, *defaults):
                if key in batch:
                    value = np.asarray(batch[key])
                else:
                    fill, dtype = defaults[key]
                    value = np.full(per_detector, fill, dtype=dtype)
                if key not in mixed:
                    mixed[key] = np.empty((len(assignment),) + value.shape[1:], dtype=value.dtype)
                mixed[key][rows] = value[rows]
        labels = np.array([pool.label for pool in self.pools], dtype=np.float32)
        mixed[gf.ReturnVariables.POOL_LABEL] = labels[assignment]
        return mixed
