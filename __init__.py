"""
This module initializes the GravyFlow library, setting up necessary imports and configurations.
"""

# Standard library imports
import warnings

# Suppress specific LAL warning when running in an ipython kernel
warnings.filterwarnings("ignore", category=UserWarning, message="Wswiglal-redir-stdio")

# Conditional import with specific exception handling
try:
    from .cuphenom.python.cuphenom import imrphenomd
except ImportError as e:
    print(f"Failed to import cuphenom because: {e}.")

# Local application/library specific imports
from .config import Defaults
from .tensor_tools import (
    DistributionType, Distribution, randomise_arguments,
    replace_nan_and_inf_with_zero, expand_tensor, batch_tensor,
    crop_samples, rfftfreq, get_element_shape, check_tensor_integrity,
    set_random_seeds
)
from .environment import (setup_cuda, find_available_GPUs, get_tf_memory_usage, env, 
    get_memory_array, get_gpu_utilization_array)
from .io_tools import (
    open_hdf5_file, ensure_directory_exists, replace_placeholders,
    transform_string, snake_to_capitalized_spaces, is_redirected, load_history,
    CustomHistorySaver, EarlyStoppingWithLoad, PrintWaitCallback
)
from .processes import (Heart, HeartbeatCallback, Process, Manager, 
    explain_exit_code)
from .psd import psd
from .snr import snr, scale_to_snr
from .wnb import wnb
from .conditioning import spectrogram, spectrogram_shape
from .genetics import HyperParameter, HyperInjectionGenerator, ModelGenome
from .git import get_current_repo
from .model import (
    BaseLayer, Reshape, DenseLayer, FlattenLayer, ConvLayer,
    PoolLayer, DropLayer, BatchNormLayer, WhitenLayer, WhitenPassLayer,
    Model, PopulationSector, Population
)
from .whiten import whiten, Whiten, WhitenPass
from .pearson import rolling_pearson
from .detector import IFO, Network
from .acquisition import (
    DataQuality, DataLabel, SegmentOrder, AcquisitionMode, ObservingRun,
    IFOData, IFODataObtainer
)
from .noise import NoiseType, NoiseObtainer
from .injection import (
    ScalingOrdinality, ScalingType, ScalingTypes, ScalingMethod, ReturnVariables,
    WaveformGenerator, WaveformParameter, WaveformParameters, WNBGenerator, 
    cuPhenomDGenerator, IncoherentGenerator, InjectionGenerator,
    roll_vector_zero_padding, generate_mask, is_not_inherited, 
    batch_injection_parameters
)
from .dataset import data, Dataset
from .plotting import (
    generate_strain_plot, generate_psd_plot, generate_spectrogram, generate_correlation_plot
)
from .validate import Validator
from .glitch import GlitchType, get_glitch_times, get_glitch_segments
from .alert import send_email