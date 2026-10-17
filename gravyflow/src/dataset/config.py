import logging
import operator
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from gravyflow.src.rng.legacy import warn_deprecated

#: Why a ``scale_factor`` is no longer taken: every array gravyflow reads,
#: returns or caches is strain.
PHYSICAL_STRAIN = (
    "gravyflow works in physical strain: inputs, outputs, labels and caches are strain, "
    "and the internal x1e21 scale (Defaults.scale_factor) is gone"
)


def ignore_scale_factor(scale_factor) -> None:
    """
    The one-release shim for a ``scale_factor=`` argument: a value passed is
    ignored, with a DeprecationWarning (see PHYSICAL_STRAIN).
    """
    if scale_factor is not None:
        warn_deprecated("scale_factor", f"it is ignored: {PHYSICAL_STRAIN}; remove the argument")


def require_positive(name: str, value) -> None:
    """Refuse, by ``name``, a configuration value that is not a finite number above zero."""
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite number above zero, got {value!r}.")


def require_non_negative(name: str, value) -> None:
    """Refuse, by ``name``, a configuration value that is not a finite number of zero or more."""
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be a finite number of zero or more, got {value!r}.")


def require_count(name: str, value) -> int:
    """``value`` as an int, refused by ``name`` unless it is a whole number of one or more."""
    try:
        count = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a whole number of one or more, got {value!r}.") from None
    if isinstance(value, bool) or count < 1:
        raise ValueError(f"{name} must be a whole number of one or more, got {value!r}.")
    return count


def batch_size(num_examples_per_batch=None) -> int:
    """
    The examples per batch: ``num_examples_per_batch``, or ``Defaults``' when it is None.
    Every reader takes its batch size here, so one that can hold no example is refused by
    name as it is built.
    """
    if num_examples_per_batch is None:
        num_examples_per_batch = Defaults.num_examples_per_batch
    return require_count("num_examples_per_batch", num_examples_per_batch)


@dataclass(frozen=True)
class WindowSpec:
    """
    Immutable specification for data acquisition time windows.
    
    This bundles all duration-related parameters that are frequently passed
    together, providing a single source of truth for window configuration.
    
    Terminology:
        - onsource: The final output window duration (e.g., 1.0s for inference)
        - offsource: The background/PSD estimation window (e.g., 16.0s)
        - crop: Extra duration on EACH SIDE of onsource for edge-effect handling
                (whitening artifacts, FFT edge effects, etc.)
        - total_onsource: onsource + (2 * crop) - the raw acquisition window
    
    Example:
        >>> spec = WindowSpec.default()  # Uses Defaults values
        >>> spec = WindowSpec(
        ...     sample_rate_hertz=2048.0,
        ...     onsource_duration_seconds=1.0,
        ...     offsource_duration_seconds=16.0,
        ...     crop_duration_seconds=0.5
        ... )
        >>> spec.total_onsource_duration_seconds  # 2.0
        >>> spec.num_onsource_samples  # 4096
        
        # For JAX JIT functions - use individual properties:
        >>> num_on = spec.num_onsource_samples  # Static int for JIT
    
    Attributes:
        sample_rate_hertz: Data sample rate in Hz
        onsource_duration_seconds: Final onsource window duration in seconds
        offsource_duration_seconds: Background window duration in seconds
        crop_duration_seconds: Cropping duration on each side in seconds
    
    Note:
        This class is frozen (immutable) to ensure it can be used as a 
        dictionary key and to prevent accidental modification.
    """
    sample_rate_hertz: float
    onsource_duration_seconds: float
    offsource_duration_seconds: float
    crop_duration_seconds: float

    def __post_init__(self):
        """A window that holds no sample, or runs backwards, is refused by name."""
        require_positive("sample_rate_hertz", self.sample_rate_hertz)
        require_positive("onsource_duration_seconds", self.onsource_duration_seconds)
        require_non_negative("offsource_duration_seconds", self.offsource_duration_seconds)
        require_non_negative("crop_duration_seconds", self.crop_duration_seconds)

    @classmethod
    def default(cls) -> "WindowSpec":
        """Create WindowSpec from Defaults values."""
        return cls(
            sample_rate_hertz=Defaults.sample_rate_hertz,
            onsource_duration_seconds=Defaults.onsource_duration_seconds,
            offsource_duration_seconds=Defaults.offsource_duration_seconds,
            crop_duration_seconds=Defaults.crop_duration_seconds,
        )
    
    @classmethod
    def from_params(
        cls,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        offsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
    ) -> "WindowSpec":
        """
        Create WindowSpec with partial overrides of defaults.
        
        Any parameter not specified will use the value from Defaults.
        """
        return cls(
            sample_rate_hertz=sample_rate_hertz if sample_rate_hertz is not None else Defaults.sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds if onsource_duration_seconds is not None else Defaults.onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds if offsource_duration_seconds is not None else Defaults.offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds if crop_duration_seconds is not None else Defaults.crop_duration_seconds,
        )

    @property
    def total_onsource_duration_seconds(self) -> float:
        """Onsource duration including crop padding on both ends."""
        return self.onsource_duration_seconds + (self.crop_duration_seconds * 2.0)
    
    @property
    def num_onsource_samples(self) -> int:
        """Number of onsource samples including cropping padding (ensured even)."""
        from gravyflow.src.utils.tensor import even_samples
        return even_samples(self.total_onsource_duration_seconds, self.sample_rate_hertz)

    @property
    def num_offsource_samples(self) -> int:
        """Number of offsource samples (ensured even)."""
        from gravyflow.src.utils.tensor import even_samples
        return even_samples(self.offsource_duration_seconds, self.sample_rate_hertz)
    
    @property
    def num_crop_samples(self) -> int:
        """
        Samples of the padded onsource window before its first returned sample:
        where ``crop_samples`` starts the onsource it keeps.
        """
        from gravyflow.src.utils.tensor import crop_start
        return crop_start(self.num_onsource_samples, self.onsource_duration_seconds, self.sample_rate_hertz)

    def onsource_start_gps(self, window_start_gps):
        """
        ``START_GPS_TIME``: the GPS time of the first sample of the onsource window
        returned after cropping, from the GPS time of the padded window's first sample
        (the owner's ruling, every path: real noise and transients, obtainer and
        Dataset). Float64, whatever the global precision.
        """
        from gravyflow.src.utils.gps import as_gps_times
        return as_gps_times(window_start_gps) + self.num_crop_samples / float(self.sample_rate_hertz)

    def transient_window_first_sample(self, transient_gps) -> np.ndarray:
        """
        The absolute sample index (samples since GPS 0 at ``sample_rate_hertz``) of the
        first sample of a transient's padded onsource window: the window's centre sample
        (``centre_index``) is the transient's sample, the sample of the absolute grid
        nearest the transient's time. Int64, of the shape of ``transient_gps``.
        """
        from gravyflow.src.utils.gps import as_gps_times
        from gravyflow.src.utils.tensor import centre_index
        nearest = np.round(np.asarray(as_gps_times(transient_gps)) * float(self.sample_rate_hertz))
        return nearest.astype(np.int64) - centre_index(self.num_onsource_samples)

    def transient_window_start_gps(self, transient_gps):
        """
        The GPS time of the first sample of a transient's padded onsource window
        (``transient_window_first_sample``), exactly. Float64.
        """
        from gravyflow.src.utils.gps import as_gps_times
        return as_gps_times(self.transient_window_first_sample(transient_gps) / float(self.sample_rate_hertz))

    @property
    def sample_counts(self) -> Tuple[int, int]:
        """
        Return (num_onsource_samples, num_offsource_samples) as a tuple.
        
        Useful when passing to functions that need both values.
        """
        return (self.num_onsource_samples, self.num_offsource_samples)
    
    def with_overrides(self, **kwargs) -> "WindowSpec":
        """
        Create a new WindowSpec with some values overridden.
        
        Example:
            >>> spec2 = spec.with_overrides(onsource_duration_seconds=2.0)
        """
        return WindowSpec(
            sample_rate_hertz=kwargs.get('sample_rate_hertz', self.sample_rate_hertz),
            onsource_duration_seconds=kwargs.get('onsource_duration_seconds', self.onsource_duration_seconds),
            offsource_duration_seconds=kwargs.get('offsource_duration_seconds', self.offsource_duration_seconds),
            crop_duration_seconds=kwargs.get('crop_duration_seconds', self.crop_duration_seconds),
        )


class _DefaultSource:
    """
    ``Defaults.source`` until it is first read: then ``GWOSC()``, which replaces this
    descriptor on the class, so every later read returns that one source.

    It is made on first read because ``acquisition.sources`` imports this module, and
    ``import gravyflow`` imports neither. Assigning ``Defaults.source`` (``Defaults.set``,
    or a test's monkeypatch) replaces the descriptor too.
    """

    def __get__(self, instance, owner):
        from gravyflow.src.dataset.acquisition.sources import GWOSC

        source = GWOSC()
        setattr(owner, "source", source)
        return source


class Defaults:
    seed : int = 1000
    num_examples_per_generation_batch: int = 2048
    num_examples_per_batch: int = 32
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    crop_duration_seconds: float = 0.5
    #: Where an obtainer built without a ``source`` reads real strain: ``GWOSC()``,
    #: made on first read.
    source = _DefaultSource()
    #: The bytes one process may hold in fetches and reads of real strain at once (the
    #: store's accountant). Tiles are memory-mapped and never held, so this bounds what
    #: is in flight; N worker processes use N times it.
    memory_bytes: int = 2 * 1024 ** 3

    # Standard tiers for JIT graph optimization
    # Using these values reduces the number of unique JAX graph compilations
    STANDARD_BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    STANDARD_SAMPLE_RATES = [2048.0, 4096.0, 8192.0, 16384.0]

    @classmethod
    def set(cls, **kwargs):
        """
        Set defaults by name. The values are checked together before any is set, by the
        rules every configuration built from them applies, so a refused call changes none.
        """
        for key in kwargs:
            if key == "scale_factor":
                raise AttributeError(f"Defaults.scale_factor is removed: {PHYSICAL_STRAIN}.")
            if not hasattr(cls, key):
                raise AttributeError(
                    f"{cls.__name__} has no attribute named '{key}'"
                )
        candidate = {name: kwargs.get(name, getattr(cls, name)) for name in (
            "sample_rate_hertz", "onsource_duration_seconds",
            "offsource_duration_seconds", "crop_duration_seconds",
        )}
        WindowSpec(**candidate)
        for name in ("num_examples_per_batch", "num_examples_per_generation_batch"):
            if name in kwargs:
                require_count(name, kwargs[name])
        if "memory_bytes" in kwargs:
            require_count("memory_bytes", kwargs["memory_bytes"])
        for key, value in kwargs.items():
            setattr(cls, key, value)
    
    @staticmethod
    def quantize_batch_size(batch_size: int) -> int:
        """Round up to nearest standard batch size to reduce JAX recompilations."""
        for std_batch in Defaults.STANDARD_BATCH_SIZES:
            if batch_size <= std_batch:
                return std_batch
        logging.debug(f"Batch size {batch_size} exceeds standard tiers, using as-is.")
        return batch_size
    
    @staticmethod
    def quantize_sample_rate(sample_rate: float) -> float:
        """Round to nearest standard sample rate."""
        if sample_rate in Defaults.STANDARD_SAMPLE_RATES:
            return sample_rate
        # Find closest
        closest = min(Defaults.STANDARD_SAMPLE_RATES, key=lambda x: abs(x - sample_rate))
        if abs(closest - sample_rate) > 1.0:
            logging.debug(f"Sample rate {sample_rate} rounded to standard {closest}")
        return closest


class TransientDefaults:
    """Constants of transient records: how their GPS times are keyed and matched."""

    # GPS time precision for a record's key (10 ms)
    GPS_KEY_PRECISION: float = 0.01

    # GPS tolerance for matching transients (half second handles typical rounding)
    GPS_TOLERANCE_SECONDS: float = 0.5
