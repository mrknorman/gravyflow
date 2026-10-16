"""
Gravyflow Detection Module

Traditional detection methods, as baselines for deep-learning detectors.

- MatchedFilter: a CBC template-bank matched filter on whitened data
- MatchedFilterLayer: the matched filter as a Keras layer (logits out)
- TemplateBank: template banks with a measured minimal match
- TemplateGrid: a plain grid in component masses (no coverage guarantee)
"""

from gravyflow.src.detection.template_grid import TemplateBank, TemplateGrid
from gravyflow.src.dataset.tools.snr import matched_filter_fft, optimal_snr, template_sigma
from gravyflow.src.detection.matched_filter import MatchedFilter, MatchedFilterLayer, find_triggers

__all__ = [
    'MatchedFilter',
    'MatchedFilterLayer',
    'TemplateBank',
    'TemplateGrid',
    'find_triggers',
    'matched_filter_fft',
    'optimal_snr',
    'template_sigma',
]
