from __future__ import annotations

# Built-In imports:
from dataclasses import dataclass
from typing import Tuple, Optional, ClassVar
import logging
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, List, Dict, Type, List
import json
from copy import copy, deepcopy
from dataclasses import replace
from itertools import count
from warnings import warn

# Library imports:
import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import jax
# Local imports:
import gravyflow as gf
from gravyflow.src.utils.shapes import (
    BATCH, GENERATOR, IFO, SAMPLES, OFFSOURCE_SAMPLES, ANY
)
from gravyflow.src.utils.tensor import replace_nan_and_inf_with_zero
from gravyflow.src.rng import BatchStreams, SeedTree, check_name, resolve_streams
from gravyflow.src.rng.legacy import warn_deprecated
from gravyflow.src.dataset.config import ignore_scale_factor
from gravyflow.src.dataset.features.waveforms.cbc import (
    _generate_cbc_window,
    Approximant
)

class ScalingOrdinality(Enum):
    BEFORE_PROJECTION = auto()
    AFTER_PROJECTION = auto()

@dataclass
class ScalingType:
    index : int
    ordinality : ScalingOrdinality
    # The shape of one label, per example and generator.
    shape: tuple = ()

class ScalingTypes(Enum):
    SNR = ScalingType(1, ScalingOrdinality.AFTER_PROJECTION)
    HRSS = ScalingType(2,  ScalingOrdinality.BEFORE_PROJECTION)
    HPEAK = ScalingType(3, ScalingOrdinality.BEFORE_PROJECTION)
    
    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member

@dataclass
class ScalingMethod:
    value : Union['gf.Distribution', 'gf.Curriculum', np.ndarray]
    type_ : ScalingTypes
    
    def scale(
        self,
        injections, 
        onsource,
        scaling_parameters,
        sample_rate_hertz : float,
        onsource_duration_seconds: float = None,
        measure = None
        ):
        """
        Scale ``injections`` to the targets ``scaling_parameters``.

        HRSS and HPEAK act on (Batch, 2, Time) polarisations, and give them the
        target in the units the target is given in. SNR acts on projected
        (Batch, IFOs, Time) strain against ``onsource``, the background, and
        is a ratio of same-unit quantities.

        Args:
            measure: the current hrss or hpeak of each injection, if it was
                measured elsewhere (for example on a longer buffer); measured
                on ``injections`` when None. Not used for SNR.

        Returns:
            (scaled injections, the factor k applied to each example, (Batch,)).
            Examples with nothing to scale get k = 0.
        """
        match self.type_:
            
            case ScalingTypes.SNR:
                return gf.scale_to_snr(
                    injections, 
                    onsource,
                    scaling_parameters,
                    sample_rate_hertz,
                    onsource_duration_seconds=onsource_duration_seconds,
                    return_factor=True
                )

            case ScalingTypes.HRSS:
                if measure is None:
                    measure = calculate_hrss(injections, sample_rate_hertz)
                    
            case ScalingTypes.HPEAK:
                if measure is None:
                    measure = calculate_hpeak(injections)

            case _:
                raise ValueError(f"Scaling type {self.type_} not recognised.")
        
        factor = _ratio(scaling_parameters, measure)
        return injections * _per_example(factor, injections), factor

def check_scaling_matches_noise(noise_obtainer, generators) -> None:
    """
    Refuse a strain-amplitude target (HRSS or HPEAK) for injections into WHITE
    noise. WHITE noise is dimensionless (unit variance, no detector), so a strain
    target (~1e-21) would be invisible in it, with every example still labelled
    as injected. SNR targets are measured against the noise itself and are fine.
    """
    if getattr(noise_obtainer, "noise_type", None) is not gf.NoiseType.WHITE:
        return
    for generator in generators or []:
        type_ = getattr(getattr(generator, "scaling_method", None), "type_", None)
        if type_ is not None and type_.value.ordinality == ScalingOrdinality.BEFORE_PROJECTION:
            raise ValueError(
                f"{type(generator).__name__} scales to {type_.name}, a strain amplitude, but "
                "WHITE noise is dimensionless (unit variance), so the injection would be "
                "invisible in it. Scale to SNR (gf.ScalingTypes.SNR) for WHITE noise, or use "
                "noise in strain (COLORED, PSEUDO_REAL or REAL)."
            )


def _per_example(values, like):
    """Reshape a (Batch,) vector to broadcast against ``like``'s (Batch, ...)."""
    rank = len(ops.shape(like))
    return ops.reshape(values, (-1,) + (1,) * (rank - 1))

def _ratio(target, current):
    """target / current, 0 where current is 0 (nothing to scale)."""
    target = ops.reshape(ops.convert_to_tensor(target, dtype="float32"), (-1,))
    current = ops.reshape(ops.convert_to_tensor(current, dtype="float32"), (-1,))
    positive = current > 0
    return ops.where(
        positive, target / ops.where(positive, current, 1.0), 0.0
    )

def _at_gain(value, gain, power):
    """
    A drawn parameter's value for the source scaled by ``gain``: value x
    gain ** power, per example (0 where the gain is 0 and power < 0).
    """
    if power == 0:
        return value
    factor = gain ** power if power > 0 else _ratio(1.0, gain ** -power)
    return ops.convert_to_tensor(value, dtype="float32") * _per_example(factor, value)

def _peak_normalise(injection):
    """(injection / m, m), with m = max |injection| per example (1 where 0)."""
    injection = ops.convert_to_tensor(injection, dtype="float32")
    axes = tuple(range(1, len(ops.shape(injection))))
    peak = ops.max(ops.abs(injection), axis=axes, keepdims=True)
    peak = ops.where(peak > 0, peak, 1.0)
    return injection / peak, ops.reshape(peak, (-1,))

def calculate_hrss(
    injection,
    sample_rate_hertz : float
    ):
    """
    The root-sum-square strain of each example: sqrt(sum_t (h+^2 + hx^2) dt).

    ``injection`` is (Batch, 2, Time) polarisations, (Batch, Detectors, 2,
    Time) per-detector polarisations (an IncoherentGenerator's, whose hrss is
    the root-sum-square over its detectors), or (Batch, Time); every axis but
    the first is summed. The LAL/burst convention, as
    ``XLALMeasureHrss``. Computed as m * hrss(h / m), with m = max |h|, so
    physical strain (~1e-21) does not underflow in float32.

    Returns: (Batch,), in the units of ``injection`` x sqrt(s).
    """
    normalised, peak = _peak_normalise(injection)
    axes = tuple(range(1, len(ops.shape(normalised))))
    return peak * ops.sqrt(
        ops.sum(normalised * normalised, axis=axes) / sample_rate_hertz
    )

def calculate_hpeak(
    injection
    ):
    """
    The peak strain of each example: max_t sqrt(h+^2 + hx^2).

    ``injection`` is (Batch, 2, Time) polarisations, (Batch, Detectors, 2,
    Time) per-detector polarisations (the loudest detector's peak), or
    (Batch, Time). The LAL/burst convention, as ``XLALMeasureHPeak``;
    float32-safe as ``calculate_hrss``.

    Returns: (Batch,), in the units of ``injection``.
    """
    normalised, peak = _peak_normalise(injection)
    if len(ops.shape(normalised)) > 2:
        # The polarisations are always the second-last axis.
        magnitude = ops.sqrt(ops.sum(normalised * normalised, axis=-2))
    else:
        magnitude = ops.abs(normalised)
    axes = tuple(range(1, len(ops.shape(magnitude))))
    return peak * ops.max(magnitude, axis=axes)

def scale_to_hrss(
    injection, 
    desired_hrss,
    sample_rate_hertz : float
    ):
    """
    Scale (Batch, 2, Time) polarisations to an hrss of ``desired_hrss``
    (scalar or (Batch,)). Zero injections stay zero.
    """
    factor = _ratio(desired_hrss, calculate_hrss(injection, sample_rate_hertz))
    return injection * _per_example(factor, injection)

def scale_to_hpeak(
    injection, 
    desired_hpeak
    ):
    """
    Scale (Batch, 2, Time) polarisations to an hpeak of ``desired_hpeak``
    (scalar or (Batch,)). Zero injections stay zero.
    """
    factor = _ratio(desired_hpeak, calculate_hpeak(injection))
    return injection * _per_example(factor, injection)

@dataclass 
class ReturnVariable:
    index : int
    # The documented shape of the batch output, in gravyflow.src.utils.shapes'
    # axis symbols (see BatchContract), and its dtype.
    shape: tuple
    dtype: str = "float32"

class ReturnVariables(Enum):
    """
    The outputs a dataset can return. Every output is batch-first, in the
    shape shown (B examples, G waveform generators, I detectors, T onsource
    and T_off offsource samples), float32 but START_GPS_TIME (float64).
    Per-generator labels (INJECTION_MASKS, CENTRAL_TIME, and every
    WaveformParameter and ScalingType) are NaN where generator g injected
    nothing into example b, except INJECTION_MASKS, which is 0 there.

    GravyflowDataset and ComposedDataset check every batch against this
    contract; a composed dataset's generators G are every pool's.
    """
    # Core data outputs
    ONSOURCE = ReturnVariable(0, (BATCH, IFO, SAMPLES))
    WHITENED_ONSOURCE = ReturnVariable(1, (BATCH, IFO, SAMPLES))
    OFFSOURCE = ReturnVariable(2, (BATCH, IFO, OFFSOURCE_SAMPLES))
    INJECTIONS = ReturnVariable(4, (BATCH, GENERATOR, IFO, SAMPLES))
    WHITENED_INJECTIONS = ReturnVariable(5, (BATCH, GENERATOR, IFO, SAMPLES))
    # 1 where generator g injected into example b, else 0.
    INJECTION_MASKS = ReturnVariable(6, (BATCH, GENERATOR))
    # 1 where any generator injected into the example, else 0: the
    # signal-or-noise classification label.
    INJECTION_PRESENT = ReturnVariable(18, (BATCH,))
    ROLLING_PEARSON_ONSOURCE = ReturnVariable(7, (BATCH, ANY))
    SPECTROGRAM_ONSOURCE = ReturnVariable(8, (BATCH, ANY))
    
    # GPS Times (disambiguated)
    START_GPS_TIME = ReturnVariable(13, (BATCH, IFO), "float64")     # Segment window start (real noise/transient); NaN for simulated noise
    TRANSIENT_GPS_TIME = ReturnVariable(14, (BATCH, IFO), "float64") # Event/glitch center (transient mode only)
    # The event time: the time most recognised as the event's, chosen per
    # source type, as it arrives at each detector, as a fraction of the
    # onsource window. It is each waveform's reference time:
    # - a CBC: its merger, the peak of its envelope (|analytic h+|^2 +
    #   |analytic hx|^2), for every approximant; not the model's own
    #   coalescence time, which lies up to tens of milliseconds from it;
    # - a supernova, once a supernova generator exists: its explosion;
    # - a generic burst: its peak, the envelope's centre for a sine-Gaussian,
    #   chirplet, WNB or periodic wave, and a ringdown's start.
    CENTRAL_TIME = ReturnVariable(9, (BATCH, GENERATOR, IFO))
    
    # Labels and Types (transient mode), per detector
    DATA_LABEL = ReturnVariable(11, (BATCH, IFO))         # DataLabel enum: NOISE=0, GLITCHES=1, EVENTS=2
    SUB_TYPE = ReturnVariable(10, (BATCH, IFO))           # GlitchType.value (if glitch) or SourceType.value (if event)
    GLITCH_TYPE = ReturnVariable(16, (BATCH, IFO))        # GlitchType.value if glitch, -1 otherwise
    SOURCE_TYPE = ReturnVariable(12, (BATCH, IFO))        # SourceType.value if event, -1 otherwise
    EVENT_TYPE = ReturnVariable(15, (BATCH, IFO))         # EventConfidence enum value (if event)
    POOL_LABEL = ReturnVariable(17, (BATCH,))  # FeaturePool class label (ComposedDataset)
    # The augmentations each example had, as a bitmask: the sum of the bit of
    # each one applied (SignReversal 1, TimeReversal 2, RandomShift 4,
    # AddNoise 8; ``gf.SignReversal.bit``). 0 for every example of evaluation
    # data, which is never augmented. START_GPS_TIME still gives the window read.
    AUGMENTATIONS = ReturnVariable(19, (BATCH,))
    
    # 1 where two or more of the example's detectors are read at one time (real
    # noise and transients: the gf.Coincidence opt-in, or a real event seen in
    # several detectors), else 0.
    COINCIDENT = ReturnVariable(20, (BATCH,))

    def __lt__(self, other):
        return self.value.index < other.value.index

@dataclass
class WaveformGenerator:
    """
    Base of the waveform generators.

    Placement: each waveform's reference time (its peak, or a CBC's merger) is
    placed at an offset from the onsource window's centre drawn uniformly from
    [-front_padding_duration_seconds, +back_padding_duration_seconds]; None
    means as far as the window allows. Unless allow_partially_out_of_window is
    True, the range is narrowed per example so that the whole signal (its
    ``get_signal_extent_seconds``), delayed to any detector, stays inside the
    window.

    Sky: right_ascension_radians, declination_radians and polarization_radians
    are LAL's equatorial angles and polarisation angle, defined at
    reference_gps_time_seconds. Each is a float, a gf.Distribution, or None for
    an isotropic draw (polarisation uniform in [0, pi)).

    Randomness: ``generate(n, sample_rate_hertz, duration_seconds, streams)``
    draws every value from ``streams`` (a gf.BatchStreams), one stream per
    attribute, so a batch is a pure function of its address and the generator
    is never changed by generating. ``name`` names the generator's streams in a
    dataset (default: its position and class), so reordering named generators
    moves none of their streams.
    """
    scaling_method : Union[ScalingMethod, None] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : Optional[float] = None
    back_padding_duration_seconds : Optional[float] = None
    network : Union[List[gf.IFO], gf.Network, Path, None] = None
    allow_partially_out_of_window : bool = False
    right_ascension_radians : Union[float, gf.Distribution, None] = None
    declination_radians : Union[float, gf.Distribution, None] = None
    polarization_radians : Union[float, gf.Distribution, None] = None
    reference_gps_time_seconds : Optional[float] = None
    name : Optional[str] = None

    distributed_attributes : Union[Tuple, None] = None

    # Sky attributes, drawn by the injection layer from the generator's sky
    # and polarisation streams.
    sky_attributes : ClassVar[Tuple[str, ...]] = (
        "right_ascension_radians",
        "declination_radians",
        "polarization_radians",
    )

    def __post_init__(self):
        if self.name is not None:
            check_name(self.name, "generator name")
        self.network = self.init_network(self.network, self.geometry_stream())
        self.convert_all_distributions()

        for attribute in self.sky_attributes:
            value = getattr(self, attribute, None)
            if value is None or isinstance(value, gf.Distribution):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(self, attribute, gf.Distribution(
                    value=float(value), type_=gf.DistributionType.CONSTANT
                ))
            else:
                raise TypeError(
                    f"{attribute} should be a float, a gf.Distribution or None."
                )

    @property
    def stream_label(self) -> str:
        """The generator's name in stream paths when it is called directly."""
        return self.name or type(self).__name__

    def resolve_streams(self, streams, seed=None) -> BatchStreams:
        """
        The streams ``generate`` draws from: ``streams`` as given; else, for a
        direct call, the deterministic default streams (of ``seed``, deprecated,
        or ``gf.Defaults.seed``).
        """
        if isinstance(streams, (int, np.integer)) and not isinstance(streams, bool):
            streams, seed = None, streams
        if seed is not None:
            warn_deprecated(
                f"{type(self).__name__}.generate(seed=)",
                "pass streams (a gf.BatchStreams), or none for the default streams"
            )
        return resolve_streams(streams, seed, f"wave/{self.stream_label}")

    def geometry_stream(self) -> np.random.Generator:
        """
        The stream a network given as a config path draws any random geometry
        from, once, when the generator is built: the generator's direct
        ``param/network`` stream, so one config always gives one network.
        """
        return resolve_streams(None, None, f"wave/{self.stream_label}").stream("param/network")

    def sample_attribute(self, attribute : str, num_samples : int, streams : BatchStreams):
        """``num_samples`` draws of one attribute, from its own stream, as float64."""
        return np.asarray(
            getattr(self, attribute).sample(num_samples, streams.stream(f"param/{attribute}")),
            dtype=np.float64
        )

    def get_max_generated_duration(self):
        return 0.0

    def get_signal_extent_seconds(self, parameters):
        """
        How far each waveform extends before and after its reference time, in
        seconds: a pair of scalars or (batch,) arrays. By default, half the
        maximum generated duration either side.
        """
        half = 0.5 * float(self.get_max_generated_duration())
        return half, half

    def placement_range_seconds(self, window_seconds : float):
        """
        The configured range of the reference time's offset from the window's
        centre, (low, high) in seconds, before any per-example narrowing.
        """
        low = self.front_padding_duration_seconds
        high = self.back_padding_duration_seconds
        low = window_seconds / 2.0 if low is None else float(low)
        high = window_seconds / 2.0 if high is None else float(high)
        return -low, high

    def detector_margin_seconds(self, network=None) -> float:
        """
        The largest delay between the geocentre and a detector of `network`
        (default the generator's), in seconds: how far a detector's reference
        time can lie from the placed one. 0.0 without a network.
        """
        network = self.network if network is None else network
        if network is None:
            return 0.0
        locations = np.asarray(network.location, dtype=np.float64)
        return float(np.max(np.linalg.norm(locations, axis=-1))) / 299792458.0

    def reference_time_window_seconds(self, window_seconds : float, network=None):
        """
        Where the reference time (a CBC's merger) can fall in any detector of
        `network` (default the generator's), as (start, end) in seconds from
        the window's start: the placement range around the centre, widened by
        the detector margin, inside the window.
        """
        low, high = self.placement_range_seconds(window_seconds)
        margin = self.detector_margin_seconds(network)
        centre = window_seconds / 2.0
        return max(centre + low - margin, 0.0), min(centre + high + margin, float(window_seconds))

    @classmethod
    def init_network(cls, network, rng : Optional[np.random.Generator] = None):
        """
        ``network`` as a gf.Network: a list of IFOs, or a config path whose
        random geometry is drawn from ``rng``.
        """
        match network:
            case list():
                network = gf.Network(network)

            case Path():
                network = gf.Network.load(network, rng=rng)
                
            case None | gf.Network():
                pass
            
            case _:
                raise TypeError(
                    ("Unable to initiate network with this type: "
                    f"{type(network)}.")
                )
                
        return network
    
    def copy(self):
        return deepcopy(self)
    
    @classmethod
    def load(
        cls,
        path: Path, 
        scaling_method: ScalingMethod = None,
        network : Union[List[gf.IFO], gf.Network, Path] = None
    ) -> Type[cls]:

        replacements = {
            "pi": np.pi,
            "2*pi": 2.0 * np.pi,
            "constant": gf.DistributionType.CONSTANT,
            "uniform": gf.DistributionType.UNIFORM,
            "normal": gf.DistributionType.NORMAL,
            "hrss" : ScalingTypes.HRSS,
            "hpeak" : ScalingTypes.HPEAK,
            "snr" : ScalingTypes.SNR
        }

        with path.open("r") as file:
            config = json.load(file)

        gf.replace_placeholders(
            config, 
            replacements
        )
            
        if scaling_method is not None:
            config["scaling_method"] = scaling_method
            
            if "scaling_distribution" in config:
                config.pop("scaling_distribution")
            if "scaling_type" in config:
                config.pop("scaling_type")
            
        elif "scaling_type" in config and "scaling_distribution" in config:
            config["scaling_method"] = ScalingMethod(
                gf.Distribution(
                    **config.pop("scaling_distribution"),
                ),
                config.pop("scaling_type")
            )
        else:
            raise ValueError("Missing Scaling Type!")
                        
        generator = None
        waveform_cls = None
        match config.pop("type"):
            case 'PhenomD': 
                waveform_cls = CBCGenerator
            case 'WNB':
                waveform_cls = WNBGenerator
            case _:
                raise ValueError("This waveform type is not implemented.")
                
        generator = waveform_cls(
            scaling_method=config.pop("scaling_method"),
            network=network,
            injection_chance=config.pop("injection_chance"),
            front_padding_duration_seconds=config.pop(
                "front_padding_duration_seconds"
            ),
            back_padding_duration_seconds=config.pop(
                "back_padding_duration_seconds"
            ),
            **{k: gf.Distribution(**v) for k, v in config.items()},
        )

        return generator

    def convert_to_distribution(
        self, 
        attribute : Union[int, float, str, gf.Distribution]
    ):
        if not isinstance(attribute, gf.Distribution):
            if isinstance(attribute, list) or isinstance(attribute, tuple): 
                value = []
                for element in attribute:
                    if isinstance(element, gf.Distribution):
                        value.append(element)
                    else:
                        value.append(
                            gf.Distribution(value=element, type_=gf.DistributionType.CONSTANT)
                        )
                return value
            else:
                return gf.Distribution(value=attribute, type_=gf.DistributionType.CONSTANT)
        else:
            return attribute

    def convert_all_distributions(self):
        if self.distributed_attributes is not None:
            for attribute in self.distributed_attributes:
                converted = self.convert_to_distribution(
                    getattr(self, attribute)
                )
                setattr(self, attribute, converted)
    
    def apply_injection_chance(self, waveforms, streams : BatchStreams):
        """
        Zero each waveform with probability 1 - injection_chance, drawn from
        the generator's ``chance`` stream. An integer seed for ``streams`` is
        deprecated: it draws the direct streams of that root.
        """
        if isinstance(streams, (int, np.integer)) and not isinstance(streams, bool):
            warn_deprecated(
                f"{type(self).__name__}.apply_injection_chance(waveforms, seed)",
                "pass the batch's streams"
            )
            # The direct streams of that root, without generate(seed=)'s warning:
            # the caller did not call generate.
            streams = resolve_streams(None, int(streams), f"wave/{self.stream_label}")
        if self.injection_chance >= 1.0:
            return waveforms
            
        num_waveforms = ops.shape(waveforms)[0]
        mask = jax.random.bernoulli(
            streams.key("chance"), p=self.injection_chance, shape=(num_waveforms,)
        )
        mask = ops.cast(mask, waveforms.dtype)
        
        # Reshape for broadcasting: (Batch, 1, 1)
        # Assuming waveforms is (Batch, Channels, Time)
        mask = ops.reshape(mask, (num_waveforms, 1, 1))
        
        return waveforms * mask

    def ensure_list_of_floats(
        self,
        name : str,
        value : Union[int, float, gf.Distribution],
    ):
        if not isinstance(value, gf.Distribution):
            if isinstance(value, list) or isinstance(value, tuple):
                if (len(value) == 3):
                    new_value = []
                    for element, label in zip(value, ("x", "y", "z")):
                        new_value.append(self.ensure_float(
                            f"{name}_{label}",
                            element
                        ))
                    return tuple(new_value)
                else:
                    raise ValueError(f"{name} should contain three elements, x, y, and z.")
            else:
                raise TypeError(f"{name} should be list or tuple.")
        else:
            if value.dtype == int:
                logging.warn(f"{name} should not have dtype = int, automatically adjusting.")
                return replace(value, dtype=float)
            else:
                return value


    def ensure_float(
        self,
        name : str,
        value : Union[int, float, gf.Distribution],
    ):
        if not isinstance(value, gf.Distribution):
            if isinstance(value, int):
                logging.warn(f"{name} should be float not int, automatically adjusting.")
                return float(value)
            elif isinstance(value, float):
                return value
            else:
                raise TypeError(f"{name} should be float or gf.Distribution object.")    
        else:
            if value.dtype == int:
                logging.warn(f"{name} should not have dtype = int, automatically adjusting.")
                return replace(value, dtype=float)
            else:
                return value

@dataclass 
class WaveformParameter:
    index : str
    # The shape of one value, per example and generator: a dataset returns
    # (B, G) + shape.
    shape: tuple = ()
    # How the parameter goes with the waveform's amplitude: the source's
    # amplitude is proportional to parameter ** (1 / amplitude_power), so when
    # scaling multiplies the generated amplitude by a gain, the injected
    # source's value is the drawn one times gain ** amplitude_power. 0: the
    # parameter does not depend on the amplitude.
    amplitude_power: int = 0
    
class WaveformParameters(Enum):
    
    # CBC parameters:
    MASS_1_MSUN = WaveformParameter(100)
    MASS_2_MSUN = WaveformParameter(101)
    INCLINATION_RADIANS = WaveformParameter(102)
    # A CBC's amplitude goes as 1 / distance.
    DISTANCE_MPC = WaveformParameter(103, amplitude_power=-1)
    REFERENCE_ORBITAL_PHASE_IN = WaveformParameter(104)
    ASCENDING_NODE_LONGITUDE = WaveformParameter(105)
    ECCENTRICITY = WaveformParameter(106)
    MEAN_PERIASTRON_ANOMALY = WaveformParameter(107)
    SPIN_1_IN = WaveformParameter(108, (3,))
    SPIN_2_IN = WaveformParameter(109, (3,))
    LAMBDA_1 = WaveformParameter(110)
    LAMBDA_2 = WaveformParameter(111)
    
    # WNB paramters:
    DURATION_SECONDS = WaveformParameter(201)
    MIN_FREQUENCY_HERTZ = WaveformParameter(202)
    MAX_FREQUENCY_HERTZ = WaveformParameter(203)
    
    # Simple waveform parameters:
    FREQUENCY_HERTZ = WaveformParameter(301)
    # A burst's amplitude: its hrss (sine-Gaussian, chirplet, ringdown) or its
    # peak (periodic wave).
    AMPLITUDE = WaveformParameter(302, amplitude_power=1)
    PHASE_RADIANS = WaveformParameter(303)
    QUALITY_FACTOR = WaveformParameter(304)
    TAU_SECONDS = WaveformParameter(305)
    START_FREQUENCY_HERTZ = WaveformParameter(306)
    END_FREQUENCY_HERTZ = WaveformParameter(307)
    CHIRP_RATE = WaveformParameter(308)
    DAMPING_TIME_SECONDS = WaveformParameter(309)
    # LAL's polarisation-ellipse eccentricity: 1 linear, 0 circular.
    POLARIZATION_ECCENTRICITY = WaveformParameter(310)

    # Sky and polarisation (LAL conventions), per generator:
    RIGHT_ASCENSION_RADIANS = WaveformParameter(401)
    DECLINATION_RADIANS = WaveformParameter(402)
    POLARIZATION_RADIANS = WaveformParameter(403)

    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member
    
    def __lt__(self, other):
        return self.value.index < other.value.index

def is_not_inherited(obj, attribute):
    return attribute in obj.__dict__ or attribute in obj.__class__.__annotations__

def get_support_bounds(value, name : str = "value"):
    """
    The (low, high) support of a float or gf.Distribution, as floats.

    CONSTANT gives its value twice and CHOICE the extremes of its values; the
    bounded types give min_ and max_. A NORMAL without bounds gives -inf and
    +inf, so a caller that needs a finite bound can say which parameter lacks
    one.
    """
    if not isinstance(value, gf.Distribution):
        if isinstance(value, (tuple, list)):
            raise TypeError(f"{name} is a vector; take the bounds of each component.")
        return float(value), float(value)

    type_ = value.type_
    if isinstance(type_, str):
        type_ = gf.DistributionType[type_.upper()]

    match type_:
        case gf.DistributionType.CONSTANT:
            return float(value.value), float(value.value)
        case gf.DistributionType.CHOICE:
            if not value.possible_values:
                raise ValueError(f"{name} is a CHOICE distribution with no possible_values.")
            return float(np.min(value.possible_values)), float(np.max(value.possible_values))
        case gf.DistributionType.NORMAL:
            low = -np.inf if value.min_ is None else float(value.min_)
            high = np.inf if value.max_ is None else float(value.max_)
            return low, high
        case _:
            if value.min_ is None or value.max_ is None:
                raise ValueError(f"{name} is a {type_.name} distribution without min_ and max_.")
            return float(value.min_), float(value.max_)


def _finite_support(generator, name):
    low, high = get_support_bounds(getattr(generator, name), name)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(
            f"{type(generator).__name__}.{name} has unbounded support; give its "
            "NORMAL distribution explicit min_ and max_ bounds."
        )
    return low, high


# A WNB's duration times its bandwidth must reach this, or its Hann envelope
# spreads the burst's energy well outside the labelled band.
WNB_MIN_TIME_BANDWIDTH = 4.0
# Rounds of redrawing below-floor time-bandwidth products before giving up.
_WNB_MAX_REDRAWS = 100

@dataclass
class WNBGenerator(WaveformGenerator):
    """
    White noise bursts: Gaussian noise band-limited to [min, max] frequency
    under a Hann envelope of full width ``duration_seconds`` (see gf.wnb).

    The band of each burst is the sampled min and max frequencies in order,
    with the top clipped to Nyquist; the labels are that realised band. Draws
    whose duration times bandwidth is below ``min_time_bandwidth`` are redrawn,
    so the population is the configured one conditioned on
    duration * bandwidth >= min_time_bandwidth.

    ``polarization_eccentricity`` is LAL's polarisation-ellipse eccentricity:
    0 (the default) gives two independent, equal-power polarisations, 1 plus
    only.
    """
    duration_seconds : Union[float, gf.Distribution] = 0.5
    min_frequency_hertz: Union[float, gf.Distribution] = 16.0
    max_frequency_hertz: Union[float, gf.Distribution] = 1024.0
    polarization_eccentricity: Union[float, gf.Distribution] = 0.0
    min_time_bandwidth : float = WNB_MIN_TIME_BANDWIDTH

    distributed_attributes : Tuple[str] = (
        "duration_seconds",
        "min_frequency_hertz",
        "max_frequency_hertz",
        "polarization_eccentricity"
    )

    def __post_init__(self):

        self.duration_seconds = self.ensure_float("duration_seconds", self.duration_seconds)
        self.min_frequency_hertz = self.ensure_float("min_frequency_hertz", self.min_frequency_hertz)
        self.max_frequency_hertz = self.ensure_float("max_frequency_hertz", self.max_frequency_hertz)
        self.polarization_eccentricity = self.ensure_float(
            "polarization_eccentricity", self.polarization_eccentricity
        )

        super().__post_init__()
        self._validate_parameters()

    def _validate_parameters(self):
        name = type(self).__name__
        duration_low, duration_high = _finite_support(self, "duration_seconds")
        if duration_low <= 0.0:
            raise ValueError(f"{name}.duration_seconds must be positive; its support reaches {duration_low}.")
        frequencies = [
            _finite_support(self, "min_frequency_hertz"),
            _finite_support(self, "max_frequency_hertz"),
        ]
        if min(low for low, _ in frequencies) < 0.0:
            raise ValueError(f"{name} frequencies must not be negative.")
        eccentricity_low, eccentricity_high = _finite_support(self, "polarization_eccentricity")
        if eccentricity_low < 0.0 or eccentricity_high > 1.0:
            raise ValueError(f"{name}.polarization_eccentricity must lie in [0, 1].")
        widest = max(high for _, high in frequencies) - min(low for low, _ in frequencies)
        if duration_high * widest < self.min_time_bandwidth:
            raise ValueError(
                f"{name}: no draw can reach duration * bandwidth >= "
                f"{self.min_time_bandwidth} (at most {duration_high} s x {widest} Hz)."
            )

    def get_max_generated_duration(self):
        return float(_finite_support(self, "duration_seconds")[1])

    def get_signal_extent_seconds(self, parameters):
        half = np.asarray(parameters[WaveformParameters.DURATION_SECONDS]) / 2.0
        return half, half

    def _draw(self, num_waveforms, nyquist, streams):
        """Durations and realised bands, redrawn until each meets the floor."""
        rngs = {
            attribute: streams.stream(f"param/{attribute}")
            for attribute in ("duration_seconds", "min_frequency_hertz", "max_frequency_hertz")
        }

        def sample(attribute, count):
            return np.asarray(getattr(self, attribute).sample(count, rngs[attribute]), dtype=np.float64)

        duration = sample("duration_seconds", num_waveforms)
        first = sample("min_frequency_hertz", num_waveforms)
        second = sample("max_frequency_hertz", num_waveforms)

        def band():
            return np.minimum(first, second), np.minimum(np.maximum(first, second), nyquist)

        for _ in range(_WNB_MAX_REDRAWS):
            low, high = band()
            short = duration * (high - low) < self.min_time_bandwidth
            count = int(np.sum(short))
            if count == 0:
                return duration, low, high
            duration[short] = sample("duration_seconds", count)
            first[short] = sample("min_frequency_hertz", count)
            second[short] = sample("max_frequency_hertz", count)

        raise ValueError(
            f"{type(self).__name__}: after {_WNB_MAX_REDRAWS} redraws, {count} bursts "
            f"still have duration * bandwidth < {self.min_time_bandwidth}; widen the "
            "duration or frequency distributions."
        )

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        streams : Optional[BatchStreams] = None,
        *,
        seed : Optional[int] = None
    ):

        if (num_waveforms > 0):

            streams = self.resolve_streams(streams, seed)
            name = type(self).__name__
            nyquist = sample_rate_hertz / 2.0
            if _finite_support(self, "min_frequency_hertz")[0] >= nyquist \
                    and _finite_support(self, "max_frequency_hertz")[0] >= nyquist:
                raise ValueError(
                    f"{name}: every band starts at or above the Nyquist frequency "
                    f"{nyquist} Hz, so the bursts would be empty."
                )
            if max(_finite_support(self, "min_frequency_hertz")[1],
                   _finite_support(self, "max_frequency_hertz")[1]) > nyquist:
                warn(
                    f"{name}: band edges above the Nyquist frequency {nyquist} Hz are "
                    "clipped to it, and labelled as clipped.", UserWarning
                )

            duration, low, high = self._draw(num_waveforms, nyquist, streams)
            eccentricity = self.sample_attribute("polarization_eccentricity", num_waveforms, streams)

            if np.any(duration > duration_seconds):
                warn(
                    f"{name}: durations above the {duration_seconds} s buffer are "
                    "clipped to it.", UserWarning
                )
                duration = np.minimum(duration, duration_seconds)

            parameters = {
                "duration_seconds" : ops.convert_to_tensor(duration, dtype="float32"),
                "min_frequency_hertz" : ops.convert_to_tensor(low, dtype="float32"),
                "max_frequency_hertz" : ops.convert_to_tensor(high, dtype="float32"),
                "polarization_eccentricity" : ops.convert_to_tensor(eccentricity, dtype="float32"),
            }

            waveforms = gf.wnb(
                num_waveforms,
                sample_rate_hertz,
                duration_seconds,
                **parameters,
                key=streams.key("noise")
            )

            waveforms = ensure_last_dim_even(waveforms)
            waveforms = self.apply_injection_chance(waveforms, streams)

            enum_parameters = {
                WaveformParameters.get(key) : value for key, value in parameters.items()
            }

            return waveforms, enum_parameters

def ensure_last_dim_even(tensor):
    last_dim = ops.shape(tensor)[-1]
    is_even = (last_dim % 2) == 0
    
    if is_even:
        return tensor
    else:
        return tensor[..., :-1]

@dataclass
class CBCGenerator(WaveformGenerator):
    mass_1_msun : Union[float, gf.Distribution] = 30.0
    mass_2_msun : Union[float, gf.Distribution] = 30.0
    inclination_radians : Union[float, gf.Distribution] = 0.0
    distance_mpc : Union[float, gf.Distribution] = 50.0
    reference_orbital_phase_in : Union[float, gf.Distribution] = 0.0
    ascending_node_longitude : Union[float, gf.Distribution] = 0.0
    eccentricity : Union[float, gf.Distribution] = 0.0
    mean_periastron_anomaly : Union[float, gf.Distribution] = 0.0
    spin_1_in : Union[Tuple[float], gf.Distribution] = (0.0, 0.0, 0.0,)
    spin_2_in : Union[Tuple[float], gf.Distribution] = (0.0, 0.0, 0.0,)
    lambda_1 : Union[float, gf.Distribution] = 0.0
    lambda_2 : Union[float, gf.Distribution] = 0.0
    approximant : Approximant = Approximant.IMRPhenomD
    min_frequency_hertz : Union[float, gf.Distribution] = 20.0
    
    distributed_attributes : Tuple[str] = (
        "mass_1_msun",
        "mass_2_msun",
        "inclination_radians",
        "distance_mpc",
        "reference_orbital_phase_in",
        "ascending_node_longitude",
        "eccentricity",
        "mean_periastron_anomaly",
        "spin_1_in",
        "spin_2_in",
        "lambda_1",
        "lambda_2",
        "min_frequency_hertz"
    )

    def __post_init__(self):
        self.mass_1_msun = self.ensure_float("mass_1_msun", self.mass_1_msun)
        self.mass_2_msun = self.ensure_float("mass_2_msun", self.mass_2_msun)

        self.inclination_radians = self.ensure_float("inclination_radians", self.inclination_radians)
        self.distance_mpc = self.ensure_float("distance_mpc", self.distance_mpc)
        self.reference_orbital_phase_in = self.ensure_float("reference_orbital_phase_in", self.reference_orbital_phase_in)
        self.ascending_node_longitude = self.ensure_float("ascending_node_longitude", self.ascending_node_longitude)
        self.eccentricity = self.ensure_float("eccentricity", self.eccentricity)
        self.mean_periastron_anomaly = self.ensure_float("mean_periastron_anomaly", self.mean_periastron_anomaly)
        self.lambda_1 = self.ensure_float("lambda_1", self.lambda_1)
        self.lambda_2 = self.ensure_float("lambda_2", self.lambda_2)
        self.min_frequency_hertz = self.ensure_float("min_frequency_hertz", self.min_frequency_hertz)

        self.spin_1_in = self.ensure_list_of_floats("spin_1_in", self.spin_1_in)
        self.spin_2_in = self.ensure_list_of_floats("spin_2_in", self.spin_2_in)

        # Parameter validation warnings
        self._validate_parameters()

        super().__post_init__()
    
    def _validate_parameters(self):
        """Validate CBC parameters and warn if outside sensible ranges."""
        def get_range(val):
            """Get min/max from a value or distribution (+-inf if unbounded)."""
            return get_support_bounds(val)

        self._validate_source_model()

        # Mass validation (sensible range: 1-200 Msun for CBC)
        m1_min, m1_max = get_range(self.mass_1_msun)
        m2_min, m2_max = get_range(self.mass_2_msun)
        
        if m1_max > 200 or m2_max > 200:
            warn(
                f"CBCGenerator: Mass values >200 Msun may produce NaN waveforms. "
                f"mass_1 range: [{m1_min}, {m1_max}], mass_2 range: [{m2_min}, {m2_max}]"
            )
        if m1_min < 1 or m2_min < 1:
            warn(
                f"CBCGenerator: Mass values <1 Msun may produce invalid waveforms. "
                f"mass_1 range: [{m1_min}, {m1_max}], mass_2 range: [{m2_min}, {m2_max}]"
            )
        
        # Distance validation (sensible range: 1-10000 Mpc)
        d_min, d_max = get_range(self.distance_mpc)
        if d_max > 10000:
            warn(
                f"CBCGenerator: Distance >10000 Mpc may produce very weak signals. "
                f"distance range: [{d_min}, {d_max}] Mpc"
            )
        if d_min < 1:
            warn(
                f"CBCGenerator: Distance <1 Mpc is unrealistically close. "
                f"distance range: [{d_min}, {d_max}] Mpc"
            )
        
        # Inclination validation (sensible range: 0 to pi)
        inc_min, inc_max = get_range(self.inclination_radians)
        if inc_max > 3.15 or inc_min < 0:
            warn(
                f"CBCGenerator: Inclination should be in [0, pi] radians. "
                f"inclination range: [{inc_min}, {inc_max}]"
            )
        
        # Eccentricity validation (sensible range: 0 to 1)
        ecc_min, ecc_max = get_range(self.eccentricity)
        if ecc_max >= 1 or ecc_min < 0:
            warn(
                f"CBCGenerator: Eccentricity should be in [0, 1). "
                f"eccentricity range: [{ecc_min}, {ecc_max}]"
            )
        
        # Minimum frequency validation (sensible range: 5-100 Hz)
        f_min, f_max = get_range(self.min_frequency_hertz)
        if f_min < 5:
            warn(
                f"CBCGenerator: min_frequency <5 Hz may require very long waveforms. "
                f"min_frequency range: [{f_min}, {f_max}] Hz"
            )

    # Approximants whose spins may point out of the orbital plane's normal,
    # and those that read the tidal deformabilities.
    PRECESSING_APPROXIMANTS : ClassVar[Tuple] = (Approximant.IMRPhenomPv2,)
    TIDAL_APPROXIMANTS : ClassVar[Tuple] = (
        Approximant.TaylorF2, Approximant.IMRPhenomD_NRTidalv2
    )

    def _validate_source_model(self):
        """
        Refuse parameters the chosen approximant would ignore or could not
        represent, so that every label describes the waveform generated.

        - No offered approximant models eccentricity, the ascending-node
          longitude or the mean periastron anomaly: each must be 0.
        - Tidal deformabilities must be 0 unless the approximant is tidal.
        - Spins: a single Distribution is the aligned component (z) with no
          in-plane spin; a tuple is (x, y, z). In-plane components must be 0
          for an aligned-spin approximant, and every spin the supports allow
          must satisfy |chi| <= 1.
        """
        name = type(self).__name__
        approximant = Approximant(self.approximant) \
            if not isinstance(self.approximant, Approximant) else self.approximant

        def is_zero(value, attribute):
            return get_support_bounds(value, attribute) == (0.0, 0.0)

        for attribute in ("eccentricity", "ascending_node_longitude", "mean_periastron_anomaly"):
            if not is_zero(getattr(self, attribute), attribute):
                raise ValueError(
                    f"{name}.{attribute} must be 0: no offered approximant models it, "
                    "so it would be labelled but not generated."
                )
        if approximant not in self.TIDAL_APPROXIMANTS:
            for attribute in ("lambda_1", "lambda_2"):
                if not is_zero(getattr(self, attribute), attribute):
                    raise ValueError(
                        f"{name}.{attribute} must be 0 for {approximant.value}, which "
                        "has no tidal deformability; use a tidal approximant."
                    )

        for attribute in ("spin_1_in", "spin_2_in"):
            value = getattr(self, attribute)
            if isinstance(value, gf.Distribution):
                components = (0.0, 0.0, value)
            else:
                components = tuple(value)
            bounds = [
                get_support_bounds(component, f"{attribute}[{index}]")
                for index, component in enumerate(components)
            ]
            if approximant not in self.PRECESSING_APPROXIMANTS and \
                    any(bound != (0.0, 0.0) for bound in bounds[:2]):
                raise ValueError(
                    f"{name}.{attribute} has in-plane (x, y) components, which "
                    f"{approximant.value} ignores; give only the aligned component, or "
                    "use a precessing approximant."
                )
            largest = np.sqrt(sum(max(abs(low), abs(high)) ** 2 for low, high in bounds))
            if not largest <= 1.0:
                raise ValueError(
                    f"{name}.{attribute} can exceed the Kerr bound |chi| <= 1 (its "
                    f"components' supports reach |chi| = {largest:.3g})."
                )

    def get_max_generated_duration(self):
        """
        The window length a CBC source needs beyond the one requested: none.

        CBC waveforms are generated on a buffer sized from the requested window
        alone, and each begins no earlier than the buffer's start (tapered
        below its band edge, ``min_frequency_hertz`` or, where the buffer is
        shorter than the inspiral, a higher frequency), so any window holds its
        source's merger and the inspiral within it. The whole inspiral from
        ``min_frequency_hertz`` is never generated.
        """
        return 0.0
    
    def generate(
            self,
            num_waveforms : int,
            sample_rate_hertz : float,
            duration_seconds : float,
            streams : Optional[BatchStreams] = None,
            *,
            seed : Optional[int] = None
        ):
        
        if (num_waveforms > 0):
            
            streams = self.resolve_streams(streams, seed)
            parameters = {}
            for attribute in self.distributed_attributes:
                value = getattr(self, attribute)
                
                parameter = None
                try:
                    parameter = WaveformParameters.get(attribute)
                except:
                    try:
                        parameter = ScalingTypes.get(attribute) 
                    except:
                        continue

                if attribute in ("spin_1_in", "spin_2_in"):
                    # (num_waveforms, 3): one row per source. A single
                    # Distribution is the aligned (z) component.
                    if isinstance(value, gf.Distribution):
                        components = [
                            np.zeros(num_waveforms),
                            np.zeros(num_waveforms),
                            self.sample_attribute(attribute, num_waveforms, streams),
                        ]
                    else:
                        # Each component of a vector attribute has its own stream.
                        components = [
                            np.asarray(element.sample(
                                num_waveforms, streams.stream(f"param/{attribute}/{j}")
                            ), dtype=np.float64)
                            for j, element in enumerate(value)
                        ]
                    parameters[attribute] = np.stack(components, axis=-1)
                else:
                    parameters[attribute] = self.sample_attribute(attribute, num_waveforms, streams)

            waveforms = None

            # Label the heavier body 1, and carry each body's spin and tidal
            # deformability with its mass.
            swap = parameters["mass_1_msun"] < parameters["mass_2_msun"]
            for first, second in (
                ("mass_1_msun", "mass_2_msun"),
                ("spin_1_in", "spin_2_in"),
                ("lambda_1", "lambda_2"),
            ):
                one, two = parameters[first], parameters[second]
                mask = swap[:, None] if one.ndim == 2 else swap
                parameters[first] = np.where(mask, two, one)
                parameters[second] = np.where(mask, one, two)
            for key in list(parameters):
                parameters[key] = ops.convert_to_tensor(parameters[key], dtype="float32")

            # Wrap inclination to [0, pi] to avoid LAL waveform issues with values like 20.0
            if "inclination_radians" in parameters:
                inc = parameters["inclination_radians"]
                inc = ops.convert_to_tensor(inc, dtype="float32")
                # Modulo 2*pi
                inc = inc % (2.0 * np.pi)
                # Reflect if > pi
                inc = ops.where(inc > np.pi, 2.0 * np.pi - inc, inc)
                parameters["inclination_radians"] = inc
            
            # Add duration to parameters
            parameters["duration_seconds"] = ops.full((num_waveforms,), duration_seconds)

            # No waveform has content below its minimum frequency.
            f_min = parameters.get("min_frequency_hertz", 20.0)
            f_min = ops.convert_to_tensor(f_min, dtype="float32")
            
            # Create kwargs for CBC generator, removing parameters it doesn't accept
            cbc_kwargs = parameters.copy()
            if "min_frequency_hertz" in cbc_kwargs:
                cbc_kwargs.pop("min_frequency_hertz")
            if "duration_seconds" in cbc_kwargs:
                cbc_kwargs.pop("duration_seconds")

            # The merger is at the centre of the window.
            waveforms = _generate_cbc_window(
                    num_waveforms=num_waveforms, 
                    sample_rate_hertz=sample_rate_hertz, 
                    window_seconds=duration_seconds,
                    min_frequency_hertz=f_min,
                    approximant=self.approximant,
                    **cbc_kwargs
            )
            
            waveforms = ensure_last_dim_even(waveforms)
            waveforms = self.apply_injection_chance(waveforms, streams)
            
            # Convert parameters to Enum keys for return
            enum_parameters = {}
            for key, value in parameters.items():
                try:
                    enum_key = WaveformParameters.get(key)
                    enum_parameters[enum_key] = value
                except ValueError:
                    try:
                        enum_key = ScalingTypes.get(key)
                        enum_parameters[enum_key] = value
                    except ValueError:
                        # Fallback for keys that don't map to Enums
                        enum_parameters[key] = value
        
            return waveforms, enum_parameters


class IncoherentGenerator(WaveformGenerator):
    """
    One injection with its own waveform in each detector: component j's
    polarisations are projected onto detector j alone, so its network has
    as many detectors as it has components.

    The components are one signal: one presence (drawn once, from this
    generator's ``chance`` stream, with component 0's injection_chance), one
    placement, sky and scaling (component 0's scaling method, padding and
    allow_partially_out_of_window), one mask value per example. HRSS and
    HPEAK measure the whole injection: the root-sum-square over the
    detectors, and the loudest detector's peak. Each component draws its
    waveform from its own streams (``comp/{j}``).

    ``generate`` returns (Batch, Detectors, 2, Time) polarisations and one
    value per example for each parameter: the components' common value, NaN
    where they differ or a component does not draw it (an injection whose
    detectors hold bursts of different durations has no one duration).
    """
    component_generators : List[WaveformGenerator]

    def __init__(self, component_generators, name : Optional[str] = None):
        self.component_generators = list(component_generators)
        if not self.component_generators:
            raise ValueError("An IncoherentGenerator needs a component generator for each detector.")
        self.name = None if name is None else check_name(name, "generator name")
        first = self.component_generators[0]
        self.scaling_method = first.scaling_method
        self.injection_chance = first.injection_chance
        self.front_padding_duration_seconds = first.front_padding_duration_seconds
        self.back_padding_duration_seconds = first.back_padding_duration_seconds
        self.allow_partially_out_of_window = first.allow_partially_out_of_window
        self.network = first.network

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        # Checked on every assignment, a dataset's binding included: a network
        # of another size would broadcast components onto the wrong detectors.
        network = self.init_network(network, self.geometry_stream())
        if network is not None:
            num_detectors = int(np.asarray(network.num_detectors).ravel()[0])
            if num_detectors != len(self.component_generators):
                raise ValueError(
                    f"An IncoherentGenerator projects component j onto detector j: its "
                    f"{len(self.component_generators)} components need a network of as many "
                    f"detectors, not {num_detectors}."
                )
        self._network = network

    def get_max_generated_duration(self):
        return max(float(component.get_max_generated_duration()) for component in self.component_generators)

    def get_signal_extent_seconds(self, parameters):
        """The extent every component's signal reaches, recorded by ``generate``."""
        extent = np.asarray(parameters[SIGNAL_EXTENT])
        return extent[..., 0], extent[..., 1]

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds : float,
        streams : Optional[BatchStreams] = None,
        *,
        seed : Optional[int] = None
    ):
        """
        ``num_waveforms`` incoherent injections: (Batch, Detectors, 2, Time)
        polarisations and their parameters (see the class).
        """
        streams = self.resolve_streams(streams, seed)
        waveforms, parameters, before, after = [], [], [], []
        for j, component in enumerate(self.component_generators):
            unconditional = copy(component)
            unconditional.injection_chance = 1.0
            component_waveforms, component_parameters = unconditional.generate(
                num_waveforms, sample_rate_hertz, duration_seconds, streams.sub(f"comp/{j}")
            )
            # The presence is this generator's chance stream, so it is shared.
            waveforms.append(self.apply_injection_chance(component_waveforms, streams))
            parameters.append(component_parameters)
            extent = component.get_signal_extent_seconds(component_parameters)
            before.append(np.broadcast_to(np.asarray(extent[0], dtype=np.float64), (num_waveforms,)))
            after.append(np.broadcast_to(np.asarray(extent[1], dtype=np.float64), (num_waveforms,)))

        shapes = sorted({tuple(ops.shape(waveform)) for waveform in waveforms})
        if len(shapes) != 1:
            raise ValueError(
                f"An IncoherentGenerator's components generated waveforms of different shapes, "
                f"{shapes}; they cannot be one injection."
            )
        common = _common_values(parameters)
        common[SIGNAL_EXTENT] = np.stack([np.max(before, axis=0), np.max(after, axis=0)], axis=-1)
        return ops.stack(waveforms, axis=1), common


def _common_values(parameters_list):
    """
    One value per example of every parameter in ``parameters_list`` (one dict
    per component): the components' common value, NaN where they differ or a
    component lacks the parameter.
    """
    keys = []
    for parameters in parameters_list:
        keys.extend(key for key in parameters if key not in keys)
    common = {}
    for key in keys:
        values = [parameters.get(key) for parameters in parameters_list]
        if any(value is None for value in values):
            like = np.asarray(next(value for value in values if value is not None), dtype=np.float64)
            common[key] = np.full(like.shape, np.nan)
            continue
        values = np.stack([np.asarray(value, dtype=np.float64) for value in values])
        common[key] = np.where(np.all(values == values[0], axis=0), values[0], np.nan)
    return common

def roll_vector_zero_padding(vector, shift):
    """
    Rolls a vector by shift, filling with zeros instead of wrapping.
    vector: (Batch, Time) or (Batch, Channels, Time)
    shift: (Batch,) or int
    """
    def roll_one(vec, s):
        s = s.astype("int32")
        length = vec.shape[-1]
        indices = jnp.arange(length)
        shifted_indices = indices - s
        valid = (shifted_indices >= 0) & (shifted_indices < length)
        clamped_indices = jnp.clip(shifted_indices, 0, length - 1)
        # jnp.take is slightly different from ops.take?
        # ops.take(vec, indices, axis=-1)
        # jnp.take(vec, indices, axis=-1)
        val = jnp.take(vec, clamped_indices, axis=-1)
        if len(vec.shape) > 1:
            valid = jnp.expand_dims(valid, 0)
        return val * valid.astype(vec.dtype)

    return jax.vmap(roll_one)(vector, shift)

def generate_mask(vector):
    """
    Generates a mask where vector is non-zero.
    """
    return ops.cast(
        ops.logical_and(
            ops.not_equal(vector, 0.0),
            ops.logical_not(ops.isnan(vector))
        ),
        "float32"
    )

def batch_injection_parameters(parameters_list):
    """
    Combines a list of parameter dictionaries into a single dictionary.
    Handles missing keys by padding with NaNs.
    """
    if not parameters_list:
        return {}
        
    # Get union of all keys
    all_keys = set().union(*(d.keys() for d in parameters_list))
    batched = {}
    
    for key in all_keys:
        values = []
        for p in parameters_list:
            if key in p:
                values.append(p[key])
            else:
                # If key is missing, we need to append a placeholder.
                # We need to know the shape and dtype of the other values to create a compatible placeholder.
                # We can look at other values in the list (forward or backward).
                
                # Find a valid example
                valid_example = next((d[key] for d in parameters_list if key in d), None)
                
                if valid_example is not None:
                    # Create NaN tensor of same shape/dtype
                    if ops.is_tensor(valid_example):
                        shape = ops.shape(valid_example)
                        # Create full of NaNs
                        # ops.full might not support NaN for all backends/dtypes directly, 
                        # but we can multiply ones by NaN.
                        nan_val = float('nan')
                        placeholder = ops.ones_like(valid_example) * nan_val
                    else:
                        placeholder = None # Or float('nan')?
                    
                    values.append(placeholder)
                else:
                    # Should not happen if key is in union
                    values.append(None)

        try:
            # Check if any values are None (non-tensor placeholders)
            if any(v is None for v in values):
                 batched[key] = values
            else:
                batched[key] = ops.stack(values, axis=0)
        except:
            batched[key] = values
            
    return batched

# The sky keys add_injections_to_onsource read before its sky labels were
# WaveformParameters.
RENAMED_SKY_KEYS = {
    "right_ascension": WaveformParameters.RIGHT_ASCENSION_RADIANS,
    "declination": WaveformParameters.DECLINATION_RADIANS,
    "polarization": WaveformParameters.POLARIZATION_RADIANS,
}

def refuse_renamed_sky_keys(injection_parameters):
    """
    Raise if ``injection_parameters`` gives a sky angle under its old name,
    which would otherwise be ignored and the sky drawn instead. (The one
    place the old names are handled, so it can become a deprecation warning.)
    """
    for old, new in RENAMED_SKY_KEYS.items():
        if old in injection_parameters:
            raise KeyError(
                f'The injection parameter "{old}" is now gf.WaveformParameters.{new.name}; '
                "give the sky under that key."
            )

def _common_layout(waveforms_list):
    """
    Each generator's waveforms, (Batch, 2, Time) polarisations or (Batch,
    Detectors, 2, Time) per-detector ones, in one layout to stack: the
    first, unless a generator has a waveform per detector, when every
    coherent generator's polarisations are repeated in each detector (which
    projects exactly as they do).
    """
    detectors = sorted({ops.shape(waveforms)[1] for waveforms in waveforms_list
                        if len(ops.shape(waveforms)) == 4})
    if not detectors:
        return waveforms_list
    if len(detectors) > 1:
        raise ValueError(f"Generators with a waveform per detector disagree on the detectors: {detectors}.")
    return [
        waveforms if len(ops.shape(waveforms)) == 4 else ops.broadcast_to(
            ops.expand_dims(waveforms, 1),
            (ops.shape(waveforms)[0], detectors[0]) + tuple(ops.shape(waveforms)[1:])
        )
        for waveforms in waveforms_list
    ]

def _absent_as_nan(labels, present, num_examples):
    """
    One generator's labels as float32 (Batch, ...) arrays, NaN where it
    injected nothing (``present`` False). A None value (a key the generator
    did not draw) is left None.
    """
    present = ops.reshape(ops.convert_to_tensor(present), (-1,))
    result = {}
    for key, value in labels.items():
        if value is None:
            result[key] = None
            continue
        value = ops.convert_to_tensor(value, dtype="float32")
        if len(ops.shape(value)) == 0:
            value = ops.broadcast_to(value, (num_examples,))
        mask = ops.reshape(present, (-1,) + (1,) * (len(ops.shape(value)) - 1))
        result[key] = ops.where(mask, value, np.float32(np.nan))
    return result

def _stack_generator_labels(labels_list):
    """
    Per-generator label dicts as (Generators, Batch, ...) arrays; a key some
    generators lack is NaN for them.
    """
    keys = []
    for labels in labels_list:
        keys.extend(key for key in labels if key not in keys)
    stacked = {}
    for key in keys:
        values = [labels.get(key) for labels in labels_list]
        like = next((value for value in values if value is not None), None)
        if like is not None:
            stacked[key] = ops.stack(
                [ops.full_like(like, np.nan) if value is None else value for value in values], axis=0
            )
    return stacked

# Keys of the source measures InjectionGenerator records with each batch.
SOURCE_HRSS = "source_hrss"
SOURCE_HPEAK = "source_hpeak"
# An IncoherentGenerator's per-example (before, after) extent, (Batch, 2) seconds.
SIGNAL_EXTENT = "signal_extent_seconds"

class InjectionGenerator:
    def __init__(
        self,
        waveform_generators: List[WaveformGenerator],
        parameters_to_return: List = None,
        seed: int = None,
        scale_factor: None = None,
        scopes: Optional[List[str]] = None,
    ):
        """
        Randomness: ``generate(streams, ...)`` and
        ``add_injections_to_onsource(..., streams=)`` draw everything from the
        batch's streams: each generator's own draws under ``wave/{g}``, its
        placement from ``inject/{g}/shift``, its sky from ``sky/{g}`` and
        ``pol/{g}``, and its scaling targets from ``scale/{g}``, where {g} is
        its ``name`` or "{index}:{class}", its index counted within its
        scope. Each generator draws from its scope of the streams (``scopes``,
        one per generator; default the streams themselves): a composed
        dataset scopes each pool's generators to ``pool/{name}``. Nothing is
        drawn from state kept between calls.

        Called directly, ``__call__`` iterates over batches k = 0, 1, ...;
        batch k is drawn from ``streams(k)``, which its
        ``add_injections_to_onsource`` call takes too, so each batch gets its
        own sky and targets.

        Args:
            seed: Deprecated. The root of the streams of the iterator
                ``__call__`` returns (default ``gf.Defaults.seed``).
            scale_factor: Deprecated and ignored: injections are physical strain.
        """
        ignore_scale_factor(scale_factor)
        self.waveform_generators = waveform_generators
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        if parameters_to_return is None:
            self.parameters_to_return = []
        else:
            self.parameters_to_return = parameters_to_return
        
        if seed is not None:
            warn_deprecated("InjectionGenerator(seed=)", "pass streams to generate()")
        self._tree = SeedTree(seed)
        self.scopes = [""] * len(self.waveform_generators) if scopes is None else list(scopes)
        if len(self.scopes) != len(self.waveform_generators):
            raise ValueError(
                f"{len(self.scopes)} scopes were given for {len(self.waveform_generators)} generators."
            )
        in_scope = {}
        self.labels = []
        for generator, scope in zip(self.waveform_generators, self.scopes):
            index = in_scope.get(scope, 0)
            in_scope[scope] = index + 1
            self.labels.append(generator.name or f"{index}:{type(generator).__name__}")
        if len(set(zip(self.scopes, self.labels))) != len(self.labels):
            raise ValueError(f"Waveform generator names must be unique, got {self.labels}.")
        self.sample_rate_hertz = gf.Defaults.sample_rate_hertz # Default
        
    def __call__(
        self,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        num_examples_per_generation_batch: int = None,
        num_examples_per_batch: int = None
    ):
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if crop_duration_seconds is None:
            crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if num_examples_per_generation_batch is None:
            num_examples_per_generation_batch = gf.Defaults.num_examples_per_generation_batch
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch

        for k in count():
            yield self.generate(
                self.streams(k),
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration_seconds=onsource_duration_seconds,
                crop_duration_seconds=crop_duration_seconds,
                num_examples_per_batch=num_examples_per_batch,
            )

    def streams(self, k: Optional[int] = None) -> BatchStreams:
        """The streams of the direct iterator's batch ``k`` (None: one direct batch)."""
        return self._tree.direct(k)

    def _generator_streams(self, streams: BatchStreams, index: int) -> BatchStreams:
        """Generator ``index``'s view of a batch's streams: its scope of them."""
        return streams.sub(self.scopes[index]) if self.scopes[index] else streams

    def generate(
        self,
        streams : BatchStreams,
        sample_rate_hertz : float,
        onsource_duration_seconds : float,
        crop_duration_seconds : float,
        num_examples_per_batch : int,
    ):
        """
        One batch of unscaled injections, drawn from ``streams``: a pure
        function of the batch's address.

        Returns (injections, masks (G, B), parameters). The injections are
        (G, B, 2, S) polarisations; when a generator has a waveform per
        detector (an IncoherentGenerator), (G, B, I, 2, S), every coherent
        generator's repeated in each detector, which projects exactly as its
        own polarisations do.
        """
        self.sample_rate_hertz = sample_rate_hertz
        injections_list = []
        masks_list = []
        parameters_list = []

        # Use WindowSpec for consistent total_onsource calculation
        window_spec = gf.WindowSpec.from_params(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds
        )
        total_duration = window_spec.total_onsource_duration_seconds

        # Placement. Each generator puts its waveforms' reference time (the
        # peak, or a CBC's merger) at sample N // 2 of the buffer it
        # generates. The window kept from that buffer is chosen so that the
        # reference time lands at an offset from the onsource window's
        # centre drawn uniformly from the generator's placement range,
        # narrowed per example unless the generator allows partial signals
        # (see WaveformGenerator). "shift" records the realised offset, in
        # samples.
        ranges = [
            generator.placement_range_seconds(onsource_duration_seconds)
            for generator in self.waveform_generators
        ]
        max_shift = max((max(abs(low), abs(high)) for low, high in ranges), default=0.0)
        max_delay = 0.0
        for generator in self.waveform_generators:
            if generator.network is not None:
                 max_delay = max(max_delay, float(generator.network.max_arrival_time_difference_seconds))

        # Room for the shifts and the detector delays either side.
        extra_duration = 2.0 * (max_shift + max_delay) + 1.0

        # Calculate max signal duration required by generators
        max_signal_duration = 0.0
        for generator in self.waveform_generators:
            max_signal_duration = max(max_signal_duration, generator.get_max_generated_duration())

        for index, (generator, label, (range_low, range_high)) in enumerate(
            zip(self.waveform_generators, self.labels, ranges)
        ):
            generator_streams = self._generator_streams(streams, index)
            max_abs_shift = int(max(abs(range_low), abs(range_high)) * sample_rate_hertz)
            extended_duration = max(total_duration, max_signal_duration) + extra_duration + 2.0 * max_abs_shift / sample_rate_hertz

            waveforms, params = generator.generate(
                num_examples_per_batch,
                sample_rate_hertz,
                extended_duration,
                generator_streams.sub(f"wave/{label}")
            )
            # Non-finite samples (a waveform model failing at some
            # parameters) are reported and count as no signal.
            waveforms = replace_nan_and_inf_with_zero(
                waveforms, f"{type(generator).__name__} waveforms"
            )

            mask = generate_mask(waveforms)

            # The source's hrss and hpeak, on the whole generated buffer,
            # before the window is cut from it.
            params[SOURCE_HRSS] = calculate_hrss(waveforms, sample_rate_hertz)
            params[SOURCE_HPEAK] = calculate_hpeak(waveforms)

            # One flag per example (0.0 or 1.0), whatever the waveform's layout.
            mask = ops.max(mask, axis=tuple(range(1, len(ops.shape(mask)))))
            masks_list.append(mask)

            offsets = self._draw_offsets_seconds(
                generator, params, range_low, range_high,
                onsource_duration_seconds, num_examples_per_batch,
                generator_streams.stream(f"inject/{label}/shift")
            )

            # Slice so the reference sample, N_ext // 2, lands at
            # target_length / 2 + offset of the kept window.
            extended_samples = waveforms.shape[-1]
            reference_sample = extended_samples // 2
            target_length = int(total_duration * sample_rate_hertz)
            slice_starts = np.rint(
                reference_sample - target_length / 2.0 - offsets * sample_rate_hertz
            )
            slice_starts = np.clip(slice_starts, 0, extended_samples - target_length)
            shifts = reference_sample - slice_starts - target_length / 2.0
            slice_starts = jnp.asarray(slice_starts, dtype="int32")

            def slice_one(w, s):
                # w: (..., Time)
                # s: scalar int
                # We slice along the last axis
                rank = len(w.shape)
                zeros = [jnp.array(0, dtype="int32")] * (rank - 1)
                start_indices = tuple(zeros + [s.astype("int32")])

                slice_sizes = w.shape[:-1] + (target_length,)

                return jax.lax.dynamic_slice(w, start_indices, slice_sizes)

            waveforms = jax.vmap(slice_one)(waveforms, slice_starts)

            # Update coalescence time
            # The merger time in the generated waveform is `coalescence_time`.
            # In the sliced waveform, the time axis is shifted.
            # `t_sliced = t_gen - slice_start`.
            # So `tc_sliced = tc_gen - slice_start`.
            # slice_start is in samples. Convert to seconds.
            if "coalescence_time" in params:
                # params["coalescence_time"] is (Batch,)
                # slice_starts is (Batch,)
                params["coalescence_time"] -= slice_starts / sample_rate_hertz

            params["shift"] = jnp.asarray(shifts, dtype="float32")
            injections_list.append(waveforms)
            parameters_list.append(params)

        if injections_list:
            injections = ops.stack(_common_layout(injections_list), axis=0)
            masks = ops.stack(masks_list, axis=0)
        else:
            # No generators: no injections, and every example is noise.
            target_length = int(total_duration * sample_rate_hertz)
            injections = ops.zeros((0, num_examples_per_batch, 2, target_length), "float32")
            masks = ops.zeros((0, num_examples_per_batch), "float32")

        batched_params = batch_injection_parameters(parameters_list)

        return injections, masks, batched_params

    def _draw_offsets_seconds(
        self,
        generator,
        parameters,
        range_low : float,
        range_high : float,
        window_seconds : float,
        num_examples : int,
        rng : np.random.Generator
    ):
        """
        Each example's reference-time offset from the window's centre, seconds.

        Uniform over [range_low, range_high]. Unless the generator allows
        partial signals, each example's range is narrowed so that its signal,
        from ``before`` seconds ahead of its reference time to ``after``
        seconds behind it, delayed by up to the largest detector-to-geocentre
        delay, lies inside the window. A signal too long for that is centred,
        with a warning.
        """
        uniform = rng.random(num_examples)
        low = np.full(num_examples, float(range_low))
        high = np.full(num_examples, float(range_high))

        if not getattr(generator, "allow_partially_out_of_window", False):
            before, after = generator.get_signal_extent_seconds(parameters)
            before = np.broadcast_to(np.asarray(before, dtype=np.float64), (num_examples,))
            after = np.broadcast_to(np.asarray(after, dtype=np.float64), (num_examples,))
            margin = generator.detector_margin_seconds()
            fits_low = before + margin - window_seconds / 2.0
            fits_high = window_seconds / 2.0 - after - margin
            low = np.maximum(low, fits_low)
            high = np.minimum(high, fits_high)

            too_long = low > high
            if np.any(too_long):
                warn(
                    f"{type(generator).__name__}: {int(np.sum(too_long))} of "
                    f"{num_examples} signals are longer than the {window_seconds} s "
                    "window allows; they are centred and extend outside it.",
                    UserWarning
                )
                centre = np.clip(0.5 * (fits_low + fits_high), range_low, range_high)
                low = np.where(too_long, centre, low)
                high = np.where(too_long, centre, high)

        return low + (high - low) * uniform

    def add_injections_to_onsource(
        self,
        injections,
        mask,
        onsource,
        offsource,
        parameters_to_return,
        onsource_duration_seconds: float = None,
        injection_parameters: Optional[Dict] = None,
        scale_factor: None = None,
        streams: Optional[BatchStreams] = None,
        epoch: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ):
        """
        Scale each generator's injections, project them, and add them to ``onsource``.

        Randomness: sky and polarisation are drawn from the batch's
        ``sky/{g}`` and ``pol/{g}`` streams, scaling targets from
        ``scale/{g}`` (a Curriculum at ``epoch``; None is its end, the
        evaluation distribution). A list of targets is read in order: batch
        k of epoch e (``streams``' coordinates) takes the next examples from
        position e x ``steps_per_epoch`` + k, cycling, so training epochs run
        on through the list. ``streams`` are the batch's: a dataset's, or
        ``streams(k)`` for the direct iterator's batch k. Leaving them out is
        deprecated: it draws every call from the same direct streams.

        Units: everything is physical strain. Each example's polarisations
        are first normalised to unit hpeak. HRSS and HPEAK: scaled to the
        target, projected and cropped. SNR: projected, cropped and scaled
        against ``offsource``. ``scale_factor`` is deprecated and ignored.

        Returns:
            onsource: with the injections added.
            scaled_injections: (Generators, Batch, IFOs, Time), strain.
            labels: every label of every generator, generator-first,
                (Generators, Batch) + the label's own shape, float32, and NaN
                where the generator injected nothing into the example:
                - each WaveformParameter the generators drew; those the
                  amplitude depends on (``amplitude_power``: DISTANCE_MPC,
                  AMPLITUDE) take the value the injected amplitude implies,
                  the drawn value times the gain scaling applied to that
                  power;
                - the sky and polarisation the projection used;
                - for each ScalingType requested in ``parameters_to_return``
                  (all of them when it is None) and each type applied: the
                  target for the type applied, and the value it implies for
                  the others (HRSS and HPEAK in physical strain; SNR by the
                  measure SNR scaling uses);
                - CENTRAL_TIME, when requested: (Generators, Batch, IFOs).
        """
        if injection_parameters:
            refuse_renamed_sky_keys(injection_parameters)
        ignore_scale_factor(scale_factor)
        if streams is None:
            warn_deprecated(
                "add_injections_to_onsource without streams",
                "pass the batch's streams (InjectionGenerator.streams(k) for the iterator's batch k)"
            )
            streams = self.streams()

        # A generator without a scaling method is refused, by name, below.
        label_types = {
            generator.scaling_method.type_ for generator in self.waveform_generators
            if getattr(generator, "scaling_method", None) is not None
        }
        if parameters_to_return is None:
            label_types |= set(ScalingTypes)
        else:
            label_types |= {
                item for item in parameters_to_return if isinstance(item, ScalingTypes)
            }
        label_types = [type_ for type_ in ScalingTypes if type_ in label_types]
        drawn_keys = [
            key for key in (injection_parameters or {})
            if isinstance(key, (WaveformParameters, ScalingTypes))
        ]

        scaled_injections_list = []
        labels_list = []
        
        final_onsource = onsource
        
        # Determine if we need to return injection times
        return_injection_times = False
        if parameters_to_return and ReturnVariables.CENTRAL_TIME in parameters_to_return:
            return_injection_times = True

        for i, (generator, label) in enumerate(zip(self.waveform_generators, self.labels)):
            raw_waveforms = injections[i] 
            
            scaling_method = generator.scaling_method
            network = generator.network
            
            num_examples = ops.shape(raw_waveforms)[0]
            
            current_shifts = None
            if injection_parameters and "shift" in injection_parameters:
                shifts_all = injection_parameters["shift"]
                # shifts_all should be (NumGenerators, Batch)
                if len(ops.shape(shifts_all)) > 1 and ops.shape(shifts_all)[0] == len(self.waveform_generators):
                    current_shifts = shifts_all[i]
                else:
                    # Fallback if only 1 generator or not stacked as expected
                    current_shifts = shifts_all
            
            # Sky position and polarisation, LAL conventions, defined at the
            # generator's reference GPS time. In order of precedence: values
            # the generator returned among its parameters, then the
            # generator's own distributions, else an isotropic draw; drawn
            # from the generator's sky and polarisation streams.
            if scaling_method is None:
                raise ValueError(
                    f"{type(generator).__name__} has no scaling_method; give it a "
                    "gf.ScalingMethod before injecting it."
                )
            if network is None:
                raise ValueError(
                    f"{type(generator).__name__} has no network to project onto."
                )

            num_inj = ops.shape(raw_waveforms)[0]
            gps_time_seconds = getattr(generator, "reference_gps_time_seconds", None)

            def given(key):
                if not injection_parameters or key not in injection_parameters:
                    return None
                value = injection_parameters[key]
                if len(ops.shape(value)) > 1 and ops.shape(value)[0] == len(self.waveform_generators):
                    value = value[i]
                return ops.convert_to_tensor(value, dtype="float32")

            def draw(attribute, rng, isotropic):
                distribution = getattr(generator, attribute, None)
                values = isotropic(rng) if distribution is None else distribution.sample(num_inj, rng)
                return ops.convert_to_tensor(np.asarray(values, dtype=np.float32))

            # Isotropic: right ascension uniform in [0, 2 pi), sin(declination)
            # uniform in [-1, 1), polarisation uniform in [0, pi) (it is
            # defined modulo pi).
            generator_streams = self._generator_streams(streams, i)
            sky_rng = generator_streams.stream(f"sky/{label}")
            pol_rng = generator_streams.stream(f"pol/{label}")
            input_ra = given(WaveformParameters.RIGHT_ASCENSION_RADIANS)
            input_dec = given(WaveformParameters.DECLINATION_RADIANS)
            input_pol = given(WaveformParameters.POLARIZATION_RADIANS)
            if input_ra is None:
                input_ra = draw("right_ascension_radians", sky_rng,
                                lambda rng: 2.0 * np.pi * rng.random(num_inj))
            if input_dec is None:
                input_dec = draw("declination_radians", sky_rng,
                                 lambda rng: np.arcsin(2.0 * rng.random(num_inj) - 1.0))
            if input_pol is None:
                input_pol = draw("polarization_radians", pol_rng,
                                 lambda rng: np.pi * rng.random(num_inj))

            generator_labels = {
                key: self._generator_parameter(injection_parameters, key, i)
                for key in drawn_keys
            }
            generator_labels.update({
                WaveformParameters.RIGHT_ASCENSION_RADIANS: input_ra,
                WaveformParameters.DECLINATION_RADIANS: input_dec,
                WaveformParameters.POLARIZATION_RADIANS: input_pol,
            })

            # Arrival time at each detector minus at the geocentre, (Batch, NumDetectors)
            time_delays = network.get_time_delay(
                input_ra, input_dec, gps_time_seconds=gps_time_seconds
            )

            # The event-time label (CENTRAL_TIME): the reference time's
            # position in the onsource window at each detector, as a fraction
            # of the window.
            if return_injection_times and current_shifts is not None:
                dur = onsource_duration_seconds
                if dur is None:
                    # Fallback to calculating from samples
                     dur = ops.shape(onsource)[-1] / self.sample_rate_hertz

                # The realised offset from the window's centre, in samples.
                shift_seconds = ops.cast(current_shifts, "float32") / self.sample_rate_hertz
                t_geocenter = ops.expand_dims(dur / 2.0 + shift_seconds, -1)

                final_time = t_geocenter + time_delays

                # Normalize by duration (0 to 1)
                normalized_time = final_time / dur

                generator_labels[ReturnVariables.CENTRAL_TIME] = normalized_time


            scaling_dist = gf.Curriculum.resolve(scaling_method.value, epoch)

            if isinstance(scaling_dist, (np.ndarray, list)):
                # A fixed list of targets, read in order across epochs.
                position = (streams.k or 0) + (streams.epoch or 0) * (steps_per_epoch or 0)
                first = position * num_examples
                indices = (first + np.arange(num_examples)) % len(scaling_dist)
                target_val = np.asarray(scaling_dist)[indices]
            else:
                target_val = scaling_dist.sample(num_examples, generator_streams.stream(f"scale/{label}"))
            target_val = ops.convert_to_tensor(target_val, dtype="float32")
            
            sample_rate = self.sample_rate_hertz
            
            # Use offsource for scaling if available, otherwise fallback to onsource
            scaling_background = offsource if offsource is not None else onsource
            
            # Determine target length from onsource
            target_num_samples = ops.shape(onsource)[-1]

            def project_and_crop(strain):
                projected = network.project_wave(
                    strain,
                    sample_rate_hertz=sample_rate,
                    right_ascension=input_ra,
                    declination=input_dec,
                    polarization=input_pol,
                    gps_time_seconds=gps_time_seconds
                )
                start = (ops.shape(projected)[-1] - target_num_samples) // 2
                return projected[..., start:start + target_num_samples]

            # InjectionGenerator yields finite waveforms (it reports and
            # zeroes non-finite samples).
            raw = ops.convert_to_tensor(raw_waveforms, dtype="float32")

            # The source's measures, from the whole generated buffer when the
            # generator recorded them.
            source_hrss = self._generator_parameter(injection_parameters, SOURCE_HRSS, i)
            source_hpeak = self._generator_parameter(injection_parameters, SOURCE_HPEAK, i)
            if source_hrss is None:
                source_hrss = calculate_hrss(raw, sample_rate)
            if source_hpeak is None:
                source_hpeak = calculate_hpeak(raw)

            # 1. Unit hpeak, so no shape overflows float32 at any amplitude.
            normalisation = _ratio(1.0, source_hpeak)
            unit = raw * _per_example(normalisation, raw)
            unit_hrss = ops.reshape(source_hrss, (-1,)) * normalisation
            unit_hpeak = ops.where(normalisation > 0, 1.0, 0.0)

            type_ = scaling_method.type_
            if type_.value.ordinality == ScalingOrdinality.BEFORE_PROJECTION:
                # 2. Physical target; 3-4. project, crop.
                measure = unit_hrss if type_ == ScalingTypes.HRSS else unit_hpeak
                physical, factor = scaling_method.scale(
                    unit, None, target_val, sample_rate, measure=measure
                )
                projected = project_and_crop(physical)
            else:
                # 3-4. Project, crop; 5. scale to the SNR against the noise.
                projected, factor = scaling_method.scale(
                    project_and_crop(unit),
                    scaling_background, 
                    target_val,
                    sample_rate,
                    onsource_duration_seconds=onsource_duration_seconds
                )

            # 6. Sum into the noise.
            final_onsource = final_onsource + projected
            scaled_injections_list.append(projected)

            present = normalisation > 0
            if mask is not None:
                present = ops.logical_and(
                    present, ops.reshape(ops.convert_to_tensor(mask[i]), (-1,)) > 0
                )

            # The gain from the generated strain to the injected physical
            # strain: every parameter the amplitude depends on is labelled
            # with the value the injected amplitude implies.
            gain = normalisation * factor
            for key, value in generator_labels.items():
                if isinstance(key, WaveformParameters) and value is not None:
                    generator_labels[key] = _at_gain(value, gain, key.value.amplitude_power)

            for label_type in label_types:
                if label_type == type_:
                    value = target_val
                elif label_type == ScalingTypes.HRSS:
                    value = unit_hrss * factor
                elif label_type == ScalingTypes.HPEAK:
                    value = unit_hpeak * factor
                else:
                    value = gf.snr(
                        projected if onsource_duration_seconds is None else gf.crop_samples(
                            projected, onsource_duration_seconds, sample_rate
                        ),
                        scaling_background,
                        sample_rate
                    )
                generator_labels[label_type] = value

            labels_list.append(_absent_as_nan(generator_labels, present, num_examples))

        if scaled_injections_list:
            scaled_injections = ops.stack(scaled_injections_list, axis=0)
        else:
            scaled_injections = ops.zeros((0,) + tuple(ops.shape(onsource)), "float32")

        return final_onsource, scaled_injections, _stack_generator_labels(labels_list)

    def _generator_parameter(self, injection_parameters, key, index):
        """Generator ``index``'s value of a batched injection parameter, or None."""
        if not injection_parameters or key not in injection_parameters:
            return None
        value = injection_parameters[key]
        if isinstance(value, (list, tuple)):
            value = value[index] if len(value) == len(self.waveform_generators) else None
        elif len(ops.shape(value)) > 1 and ops.shape(value)[0] == len(self.waveform_generators):
            value = value[index]
        elif len(self.waveform_generators) > 1:
            return None
        if value is None:
            return None
        return ops.convert_to_tensor(value, dtype="float32")
