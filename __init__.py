from dataclasses import dataclass

class Defaults:
    seed : int = 1000
    num_examples_per_generation_batch: int = 2048
    num_examples_per_batch: int = 32
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    crop_duration_seconds: float = 0.5
    scale_factor: float = 1.0E21

    @classmethod
    def set(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"{cls.__name__} has no attribute named '{key}'")

from .maths import *
from .setup import *
from .psd import psd
from .snr import snr, scale_to_snr
from .wnb import wnb
from .conditioning import spectrogram, spectrogram_shape
from .model import *
from .cuphenom.python.cuphenom import imrphenomd
from .whiten import whiten, Whiten
from .pearson import rolling_pearson
from .detector import *
from .acquisition import *
from .noise import *
from .injection import *
from .dataset import *
from .plotting import *
from .validate import *
from .glitch import GlitchType, get_glitch_times, get_glitch_segments
