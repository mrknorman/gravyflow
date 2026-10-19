"""
Gravyflow Detection Module

Traditional detection methods, as baselines for deep-learning detectors.

- StreamingMatchedFilter: the matched filter over continuous strain, scored
  per detector, with a chi-squared veto; `coincide` forms its network
  triggers (time slides cost only that step)
- MatchedFilter: a CBC template-bank matched filter on whitened windows
- MatchedFilterLayer: the window matched filter as a Keras layer (logits out)
- TemplateBank: template banks, non-spinning or aligned-spin, with a measured minimal match
- TemplateGrid: a plain grid in component masses (no coverage guarantee)
"""

from gravyflow.src.detection.template_grid import TemplateBank, TemplateGrid
from gravyflow.src.dataset.tools.snr import matched_filter_fft, optimal_snr, template_sigma
from gravyflow.src.detection.matched_filter import MatchedFilter, MatchedFilterLayer, find_triggers
from gravyflow.src.detection.stream import (
    Coincidences, DetectorTriggers, Injection, StreamResult, StreamingMatchedFilter, coincide, reweighted_snr)

__all__ = [
    'Coincidences',
    'DetectorTriggers',
    'Injection',
    'MatchedFilter',
    'MatchedFilterLayer',
    'StreamResult',
    'StreamingMatchedFilter',
    'TemplateBank',
    'TemplateGrid',
    'coincide',
    'find_triggers',
    'matched_filter_fft',
    'optimal_snr',
    'reweighted_snr',
    'template_sigma',
]
