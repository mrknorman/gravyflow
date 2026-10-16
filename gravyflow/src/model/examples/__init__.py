"""
Gravyflow Example Models Module

Model architectures and datasets from published gravitational-wave deep
learning papers, and the recipe that trains and evaluates them.

Available Models:
- GeorgeHuerta2017Shallow: Shallow CNN from George & Huerta (2017)
- GeorgeHuerta2017Deep: Deep CNN from George & Huerta (2017)
- Gabbard2017: Binary black hole detection CNN from Gabbard et al. (2017)
- MLyCoincidence, MLyCoherence: the burst networks of MLy, Skliris et al. (2024)
- MatchedFilterBaseline: Traditional matched filtering for comparison

Every network takes ``WHITENED_ONSOURCE`` and returns logits.
"""

from gravyflow.src.model.examples.base import (
    ExampleModel,
    ValidationDatasetConfig,
    validation_dataset_args,
)
from gravyflow.src.model.examples.gabbard_2017 import Gabbard2017, Gabbard2017Config
from gravyflow.src.model.examples.george_huerta_2017 import (
    GeorgeHuerta2017Shallow,
    GeorgeHuerta2017Deep,
    GeorgeHuerta2017Config
)
from gravyflow.src.model.examples.mly_2024 import MLy, MLyCoherence, MLyCoincidence, MLyConfig
from gravyflow.src.model.examples.matched_filter_baseline import (
    MatchedFilterBaseline,
    MatchedFilterBaselineConfig
)
from gravyflow.src.model.examples.recipe import (
    DetectionBatches,
    RunDirectory,
    TrainingRun,
    TrainingSchedule,
    materialise,
)

__all__ = [
    'ExampleModel',
    'ValidationDatasetConfig',
    'validation_dataset_args',
    'GeorgeHuerta2017Shallow',
    'GeorgeHuerta2017Deep',
    'GeorgeHuerta2017Config',
    'Gabbard2017',
    'Gabbard2017Config',
    'MLy',
    'MLyCoincidence',
    'MLyCoherence',
    'MLyConfig',
    'MatchedFilterBaseline',
    'MatchedFilterBaselineConfig',
    'DetectionBatches',
    'RunDirectory',
    'TrainingRun',
    'TrainingSchedule',
    'materialise',
]
