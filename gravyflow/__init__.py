"""
This module initializes the GravyFlow library, setting up necessary imports and configurations.
"""

# Standard library imports
import warnings

# Suppress specific LAL warning when running in an ipython kernel
warnings.filterwarnings("ignore", category=UserWarning, message="Wswiglal-redir-stdio")

# Disable JAX memory preallocation to avoid conflicts with TensorFlow
import os
import site
import glob
import sys
import logging

# Set up NullHandler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Suppress JAX/CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'  # CUDA for compute, CPU for callbacks
logging.getLogger('jax._src.xla_bridge').setLevel(logging.ERROR)

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# Attempt to find nvidia libraries and set LD_LIBRARY_PATH
try:
    site_packages = site.getsitepackages()[0]
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    
    if os.path.exists(nvidia_dir):
        libs = glob.glob(os.path.join(nvidia_dir, '*/lib'))
        path_to_add = ":".join(libs)
        
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        
        # Filter out conflicting system CUDA paths
        ld_paths = current_ld.split(':') if current_ld else []
        clean_ld_paths = [p for p in ld_paths if '/usr/local/cuda' not in p]
        
        # Prepend nvidia paths
        new_ld_path = f"{path_to_add}:{':'.join(clean_ld_paths)}"
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
except Exception as e:
    pass

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"



# Local application/library specific imports
from .src.dataset.config import Defaults
from .src.utils.tensor import (
    DistributionType, Distribution, randomise_arguments,
    replace_nan_and_inf_with_zero, expand_tensor, batch_tensor,
    crop_samples, rfftfreq, get_element_shape, check_tensor_integrity,
    set_random_seeds
)
from .src.utils.numerics import (
    ensure_even, ensure_list, calculate_sample_counts, AcquisitionParams
)
from .src.utils.shapes import (
    ShapeContract, ShapeEnforcer, 
    Axis_BIS, Axis_BI, Axis_GBIS, Axis_GB,
    Axis_BPS, Axis_GBPS
)
from .src.utils.gpu import (setup_cuda, find_available_GPUs, get_tf_memory_usage, env, 
    get_gpu_memory_info)
from .src.utils.io import (
    open_hdf5_file, ensure_directory_exists, replace_placeholders,
    transform_string, snake_to_capitalized_spaces, is_redirected, load_history,
    CustomHistorySaver, EarlyStoppingWithLoad, PrintWaitCallback, save_dict_to_hdf5,
    get_file_parent_path
)
PATH = get_file_parent_path()
from .src.utils.processes import (Heart, HeartbeatCallback, Process, Manager, 
    explain_exit_code)
from .src.dataset.tools.psd import psd
from .src.dataset.tools.snr import snr, scale_to_snr
from .src.dataset.features.waveforms.wnb import wnb
from .src.dataset.conditioning.conditioning import spectrogram, spectrogram_shape
from .src.model.genetics import HyperParameter, HyperInjectionGenerator, ModelGenome
from .src.utils.git import get_current_repo
from .src.model.model import (
    BaseLayer, Reshape, DenseLayer, FlattenLayer, ConvLayer,
    PoolLayer, DropLayer, BatchNormLayer, WhitenLayer, WhitenPassLayer,
    Model, PopulationSector, Population, calculate_fitness
)
from .src.dataset.conditioning.whiten import whiten, Whiten, WhitenPass
from .src.dataset.conditioning.pearson import rolling_pearson
from .src.dataset.conditioning.detector import IFO, Network, project_wave
from .src.dataset.acquisition import (
    DataQuality, DataLabel, SegmentOrder, AcquisitionMode, SamplingMode, ObservingRun,
    IFOData, IFODataObtainer, NoiseDataObtainer, TransientDataObtainer
)
from .src.dataset.noise.noise import NoiseType, Obtainer, NoiseObtainer
from .src.dataset.curriculum import Curriculum, CurriculumSchedule, CurriculumProgressCallback
from .src.dataset.features.injection import (
    ScalingOrdinality, ScalingType, ScalingTypes, ScalingMethod, ReturnVariables,
    WaveformGenerator, WaveformParameter, WaveformParameters, WNBGenerator,
    IncoherentGenerator, InjectionGenerator,
    roll_vector_zero_padding, generate_mask, is_not_inherited,
    batch_injection_parameters, CBCGenerator, Approximant,
    calculate_hrss, calculate_hpeak
)
from .src.dataset.features.waveforms.simple import (
    WaveShape, PeriodicWaveGenerator, SineGaussianGenerator,
    ChirpletGenerator, RingdownGenerator
)
from .src.dataset.dataset import data, Dataset, GravyflowDataset
from .src.utils.plotting import (
    generate_strain_plot, generate_psd_plot, generate_spectrogram, generate_correlation_plot,
    generate_segment_timeline_plot, generate_example_extraction_plot
)
from .src.validate import Validator, ValidationConfig
from .src.dataset.features.glitch import GlitchType, get_glitch_times, get_glitch_times_with_labels, get_glitch_segments, get_glitch_type_from_index
from .src.dataset.features.glitch_cache import GlitchCache, generate_glitch_cache_path
from .src.dataset.features.event import (
    EventConfidence, SourceType, get_confident_event_times, get_marginal_event_times, 
    get_all_event_times, get_event_times_by_type, get_confident_events_with_params,
    search_events
)
# record.py removed - functionality replaced by TransientSegment
from .src.dataset.features.transient_index import TransientIndex
from .src.utils.alert import send_email
from .src.model.examples.gabbard_2017 import Gabbard2017
from .src.model.examples.matched_filter_baseline import MatchedFilterBaseline, MatchedFilterBaselineConfig
from .src.detection import MatchedFilter, MatchedFilterLayer, TemplateGrid, matched_filter_fft, optimal_snr
from .src.dataset.diversity import DiversityCallback, LabelTrackingDataset, compute_diversity_score

