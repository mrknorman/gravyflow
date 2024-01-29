"""
GravyFlow is a Python library designed for gravitational wave data analysis, leveraging TensorFlow's 
GPU capabilities for enhanced performance. It offers GPU-based implementations of essential functions 
commonly used in this field. GravyFlow's toolkit includes features for creating dataset classes, which 
are crucial for the real-time training of machine learning models specifically in gravitational wave 
analysis. This makes it an ideal resource for data scientists and researchers focusing on gravitational wave studies, 
providing an efficient and powerful tool for their computational needs.
"""

#Supress LAL warning when running in ipython kernel:
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Imports from GravyFlow
from .config import Defaults
from .math import (DistributionType, Distribution, 
                   randomise_arguments, replace_nan_and_inf_with_zero,
                   expand_tensor, batch_tensor, crop_samples,
                   rfftfreq, get_element_shape)
from .environment import (setup_cuda, find_available_GPUs, get_tf_memory_usage, 
                          env)
from .io import (open_hdf5_file, ensure_directory_exists, replace_placeholders)
from .processes import *
from .psd import psd
from .snr import snr, scale_to_snr
from .wnb import wnb
from .conditioning import spectrogram, spectrogram_shape
from .genetics import *
from .model import (HyperParameter, hp, ensure_hp, BaseLayer, DenseLayer, ConvLayer, PoolLayer, DropLayer, randomizeLayer, )

try:
    from .cuphenom.python.cuphenom import imrphenomd
except Exception as e:
    print(f"Failed to import cuphenom because {e}.")

from .whiten import whiten, Whiten, WhitenPass
from .pearson import rolling_pearson
from .detector import *
from .acquisition import *
from .noise import *
from .injection import *
from .dataset import *
from .plotting import *
from .validate import *
from .glitch import GlitchType, get_glitch_times, get_glitch_segments
