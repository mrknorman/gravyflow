"""
Gravyflow Example Models Module

This module provides pre-built model architectures, datasets, and pre-trained
weights from published gravitational wave deep learning papers.

Available Models:
- GeorgeHuerta2017Shallow: Shallow CNN from George & Huerta (2017)
- GeorgeHuerta2017Deep: Deep CNN from George & Huerta (2017)
- Gabbard2017: Binary black hole detection CNN from Gabbard et al. (2017)
- MatchedFilterBaseline: Traditional matched filtering for comparison
"""

from gravyflow.src.model.examples.base import ExampleModel
from gravyflow.src.model.examples.gabbard_2017 import Gabbard2017
from gravyflow.src.model.examples.george_huerta_2017 import (
    GeorgeHuerta2017Shallow,
    GeorgeHuerta2017Deep,
    GeorgeHuerta2017Config
)
from gravyflow.src.model.examples.matched_filter_baseline import (
    MatchedFilterBaseline,
    MatchedFilterBaselineConfig
)

__all__ = [
    'ExampleModel',
    'GeorgeHuerta2017Shallow',
    'GeorgeHuerta2017Deep', 
    'GeorgeHuerta2017Config',
    'Gabbard2017',
    'MatchedFilterBaseline',
    'MatchedFilterBaselineConfig',
]
