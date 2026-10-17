"""
GravyFlow: JAX and Keras 3 tools for gravitational-wave data science.

``import gravyflow`` imports nothing beyond the standard library and changes
nothing in the process: no environment variable, warnings filter, logger or
library setting. Each public name (``__all__``, documented in ``docs/api.md``)
imports its module the first time it is used.

gravyflow runs on Keras's JAX backend, which it never chooses for the process:
set ``KERAS_BACKEND=jax`` before Keras is imported.
"""

import importlib
import logging
import os
import sys

__version__ = "1.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

#: Each public name and the module that defines it, one per line, grouped as
#: docs/api.md groups them.
_MODULES = {
    # Configuration and randomness
    "Defaults": "gravyflow.src.dataset.config",
    "WindowSpec": "gravyflow.src.dataset.config",
    "SplitSpec": "gravyflow.src.dataset.splits",
    "DEFAULT_SPLIT": "gravyflow.src.dataset.splits",
    "EXCLUDED": "gravyflow.src.dataset.splits",
    "SeedTree": "gravyflow.src.rng",
    "BatchStreams": "gravyflow.src.rng",
    "set_random_seeds": "gravyflow.src.utils.tensor",
    "Distribution": "gravyflow.src.utils.tensor",
    "DistributionType": "gravyflow.src.utils.tensor",

    # Datasets
    "Dataset": "gravyflow.src.dataset.dataset",
    "GravyflowDataset": "gravyflow.src.dataset.dataset",
    "ComposedDataset": "gravyflow.src.dataset.dataset",
    "EpochMode": "gravyflow.src.dataset.dataset",
    "FeaturePool": "gravyflow.src.dataset.pools",
    "PoolSampler": "gravyflow.src.dataset.pools",
    "Curriculum": "gravyflow.src.dataset.curriculum",
    "CurriculumSchedule": "gravyflow.src.dataset.curriculum",
    "CurriculumProgressCallback": "gravyflow.src.dataset.curriculum",
    "DiversityCallback": "gravyflow.src.dataset.diversity",
    "LabelTrackingDataset": "gravyflow.src.dataset.diversity",
    "compute_diversity_score": "gravyflow.src.dataset.diversity",
    "BatchConfig": "gravyflow.src.utils.numerics",
    "BatchContract": "gravyflow.src.utils.shapes",
    "BatchContractError": "gravyflow.src.utils.shapes",
    "ReturnVariables": "gravyflow.src.dataset.features.injection",

    # Noise and detectors
    "NoiseType": "gravyflow.src.dataset.noise.noise",
    "NoiseObtainer": "gravyflow.src.dataset.noise.noise",
    "Obtainer": "gravyflow.src.dataset.noise.noise",
    "IFO": "gravyflow.src.dataset.conditioning.detector",
    "Network": "gravyflow.src.dataset.conditioning.detector",
    "NoiseProfile": "gravyflow.src.dataset.conditioning.detector",
    "project_wave": "gravyflow.src.dataset.conditioning.detector",
    "greenwich_mean_sidereal_time": "gravyflow.src.dataset.conditioning.detector",
    "REFERENCE_GPS_TIME_SECONDS": "gravyflow.src.dataset.conditioning.detector",

    # Real data
    "IFODataObtainer": "gravyflow.src.dataset.acquisition",
    "NoiseDataObtainer": "gravyflow.src.dataset.acquisition",
    "TransientDataObtainer": "gravyflow.src.dataset.acquisition",
    "IFOData": "gravyflow.src.dataset.acquisition",
    "ObservingRun": "gravyflow.src.dataset.acquisition",
    "DataQuality": "gravyflow.src.dataset.acquisition",
    "DataLabel": "gravyflow.src.dataset.acquisition",
    "SegmentOrder": "gravyflow.src.dataset.acquisition",
    "AcquisitionMode": "gravyflow.src.dataset.acquisition",
    "SamplingMode": "gravyflow.src.dataset.acquisition",
    "Coincidence": "gravyflow.src.dataset.acquisition",
    "SignReversal": "gravyflow.src.dataset.acquisition",
    "TimeReversal": "gravyflow.src.dataset.acquisition",
    "RandomShift": "gravyflow.src.dataset.acquisition",
    "AddNoise": "gravyflow.src.dataset.acquisition",
    "GWOSC_DATA_FLAGS": "gravyflow.src.dataset.acquisition",
    "GWOSC_BURST_FLAGS": "gravyflow.src.dataset.acquisition",
    "GWOSC_CBC_FLAGS": "gravyflow.src.dataset.acquisition",
    "GWOSC_NO_HW_INJECTION_FLAGS": "gravyflow.src.dataset.acquisition",
    "DEFAULT_GWOSC_FLAGS": "gravyflow.src.dataset.acquisition",
    "DataSource": "gravyflow.src.dataset.acquisition",
    "GWOSC": "gravyflow.src.dataset.acquisition",
    "LocalFrames": "gravyflow.src.dataset.acquisition",
    "StrainUnavailable": "gravyflow.src.dataset.acquisition",
    "UpsamplingRefused": "gravyflow.src.dataset.acquisition",
    "GlitchType": "gravyflow.src.dataset.features.glitch",
    "get_glitch_times": "gravyflow.src.dataset.features.glitch",
    "get_glitch_times_with_labels": "gravyflow.src.dataset.features.glitch",
    "get_glitch_segments": "gravyflow.src.dataset.features.glitch",
    "get_glitch_type_from_index": "gravyflow.src.dataset.features.glitch",
    "get_glitch_catalogue": "gravyflow.src.dataset.features.glitch",
    "TransientCache": "gravyflow.src.dataset.features.glitch_cache",
    "TransientIndex": "gravyflow.src.dataset.features.transient_index",
    "EventConfidence": "gravyflow.src.dataset.features.event",
    "SourceType": "gravyflow.src.dataset.features.event",
    "search_events": "gravyflow.src.dataset.features.event",
    "get_event_catalogue": "gravyflow.src.dataset.features.event",
    "get_all_event_times": "gravyflow.src.dataset.features.event",
    "get_confident_event_times": "gravyflow.src.dataset.features.event",
    "get_marginal_event_times": "gravyflow.src.dataset.features.event",
    "get_event_times_by_type": "gravyflow.src.dataset.features.event",
    "get_confident_events_with_params": "gravyflow.src.dataset.features.event",
    "get_event_veto_segments": "gravyflow.src.dataset.features.event",

    # Injections
    "InjectionGenerator": "gravyflow.src.dataset.features.injection",
    "WaveformGenerator": "gravyflow.src.dataset.features.injection",
    "CBCGenerator": "gravyflow.src.dataset.features.injection",
    "Approximant": "gravyflow.src.dataset.features.injection",
    "WNBGenerator": "gravyflow.src.dataset.features.injection",
    "IncoherentGenerator": "gravyflow.src.dataset.features.injection",
    "WaveformParameter": "gravyflow.src.dataset.features.injection",
    "WaveformParameters": "gravyflow.src.dataset.features.injection",
    "ScalingMethod": "gravyflow.src.dataset.features.injection",
    "ScalingType": "gravyflow.src.dataset.features.injection",
    "ScalingTypes": "gravyflow.src.dataset.features.injection",
    "ScalingOrdinality": "gravyflow.src.dataset.features.injection",
    "calculate_hrss": "gravyflow.src.dataset.features.injection",
    "calculate_hpeak": "gravyflow.src.dataset.features.injection",
    "WaveShape": "gravyflow.src.dataset.features.waveforms.simple",
    "PeriodicWaveGenerator": "gravyflow.src.dataset.features.waveforms.simple",
    "SineGaussianGenerator": "gravyflow.src.dataset.features.waveforms.simple",
    "ChirpletGenerator": "gravyflow.src.dataset.features.waveforms.simple",
    "RingdownGenerator": "gravyflow.src.dataset.features.waveforms.simple",
    "wnb": "gravyflow.src.dataset.features.waveforms.wnb",

    # Signal processing
    "psd": "gravyflow.src.dataset.tools.psd",
    "PowerSpectrum": "gravyflow.src.dataset.tools.psd",
    "SpectralConfig": "gravyflow.src.dataset.tools.psd",
    "snr": "gravyflow.src.dataset.tools.snr",
    "scale_to_snr": "gravyflow.src.dataset.tools.snr",
    "whiten": "gravyflow.src.dataset.conditioning.whiten",
    "Whiten": "gravyflow.src.dataset.conditioning.whiten",
    "WhitenPass": "gravyflow.src.dataset.conditioning.whiten",
    "rolling_pearson": "gravyflow.src.dataset.conditioning.pearson",
    "spectrogram": "gravyflow.src.dataset.conditioning.conditioning",
    "spectrogram_shape": "gravyflow.src.dataset.conditioning.conditioning",
    "resample": "gravyflow.src.utils.resample",
    "NonFiniteWarning": "gravyflow.src.utils.numerics",

    # Matched filtering
    "MatchedFilter": "gravyflow.src.detection",
    "MatchedFilterLayer": "gravyflow.src.detection",
    "TemplateBank": "gravyflow.src.detection",
    "TemplateGrid": "gravyflow.src.detection",
    "matched_filter_fft": "gravyflow.src.detection",
    "optimal_snr": "gravyflow.src.detection",

    # Models
    "Gabbard2017": "gravyflow.src.model.examples.gabbard_2017",
    "MatchedFilterBaseline": "gravyflow.src.model.examples.matched_filter_baseline",
    "MatchedFilterBaselineConfig": "gravyflow.src.model.examples.matched_filter_baseline",
    "Model": "gravyflow.src.model.model",
    "BaseLayer": "gravyflow.src.model.model",
    "Reshape": "gravyflow.src.model.model",
    "DenseLayer": "gravyflow.src.model.model",
    "FlattenLayer": "gravyflow.src.model.model",
    "ConvLayer": "gravyflow.src.model.model",
    "PoolLayer": "gravyflow.src.model.model",
    "DropLayer": "gravyflow.src.model.model",
    "BatchNormLayer": "gravyflow.src.model.model",
    "WhitenLayer": "gravyflow.src.model.model",
    "WhitenPassLayer": "gravyflow.src.model.model",
    "Population": "gravyflow.src.model.model",
    "PopulationSector": "gravyflow.src.model.model",
    "HyperParameter": "gravyflow.src.model.genetics",
    "HyperInjectionGenerator": "gravyflow.src.model.genetics",
    "ModelGenome": "gravyflow.src.model.genetics",

    # Validation (the viz extra)
    "Validator": "gravyflow.src.validate",
    "ValidationConfig": "gravyflow.src.validate",

    # Plotting (the viz extra)
    "generate_strain_plot": "gravyflow.src.utils.plotting",
    "generate_psd_plot": "gravyflow.src.utils.plotting",
    "generate_spectrogram": "gravyflow.src.utils.plotting",
    "generate_correlation_plot": "gravyflow.src.utils.plotting",
    "generate_segment_timeline_plot": "gravyflow.src.utils.plotting",
    "generate_example_extraction_plot": "gravyflow.src.utils.plotting",

    # Training, I/O and process utilities
    "crop_samples": "gravyflow.src.utils.tensor",
    "replace_nan_and_inf_with_zero": "gravyflow.src.utils.tensor",
    "ensure_directory_exists": "gravyflow.src.utils.io",
    "replace_placeholders": "gravyflow.src.utils.io",
    "snake_to_capitalized_spaces": "gravyflow.src.utils.io",
    "is_redirected": "gravyflow.src.utils.io",
    "load_history": "gravyflow.src.utils.io",
    "save_dict_to_hdf5": "gravyflow.src.utils.io",
    "model_file": "gravyflow.src.utils.io",
    "CustomHistorySaver": "gravyflow.src.utils.io",
    "EarlyStoppingWithLoad": "gravyflow.src.utils.io",
    "PrintWaitCallback": "gravyflow.src.utils.io",
    "Heart": "gravyflow.src.utils.processes",
    "HeartbeatCallback": "gravyflow.src.utils.processes",
    "Process": "gravyflow.src.utils.processes",
    "Manager": "gravyflow.src.utils.processes",
    "send_email": "gravyflow.src.utils.alert",

    # GPUs (CUDA)
    "env": "gravyflow.src.utils.gpu",
    "setup_cuda": "gravyflow.src.utils.gpu",
    "find_available_GPUs": "gravyflow.src.utils.gpu",
    "get_gpu_memory_info": "gravyflow.src.utils.gpu",
    "get_tf_memory_usage": "gravyflow.src.utils.gpu",
}

__all__ = list(_MODULES)


def _require_jax_backend() -> None:
    """Refuse, naming the fix, unless Keras runs (or will run) on its JAX backend."""
    keras = sys.modules.get("keras")
    backend = keras.backend.backend() if keras is not None else os.environ.get("KERAS_BACKEND")
    if backend != "jax":
        raise ImportError(
            f"gravyflow runs on Keras's JAX backend, and this process's is {backend!r}. "
            "Set KERAS_BACKEND=jax before Keras is imported (for example, `export KERAS_BACKEND=jax`)."
        )


def __getattr__(name: str):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _require_jax_backend()
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
