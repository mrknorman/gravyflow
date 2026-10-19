"""
Gravyflow Detection Module

Traditional detection methods, as baselines for deep-learning detectors.

- StreamingMatchedFilter: the matched filter over continuous strain, scored
  per detector, with a chi-squared veto; `coincide` forms its network
  triggers (time slides cost only that step); its toolkit, each as PyCBC
  runs it: autogating, the PSD-variation statistic, the sine-Gaussian veto
  (`vetoes`) and per-template background fits (`background`)
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
from gravyflow.src.detection.vetoes import Autogating, PSDVariation, autogate, ranking_statistic
from gravyflow.src.detection.background import TemplateFits, coincident_ranking, fit_templates

__all__ = [
    'Autogating',
    'Coincidences',
    'DetectorTriggers',
    'Injection',
    'MatchedFilter',
    'MatchedFilterLayer',
    'PSDVariation',
    'StreamResult',
    'StreamingMatchedFilter',
    'TemplateBank',
    'TemplateFits',
    'TemplateGrid',
    'autogate',
    'coincide',
    'coincident_ranking',
    'fit_templates',
    'find_triggers',
    'matched_filter_fft',
    'optimal_snr',
    'ranking_statistic',
    'reweighted_snr',
    'template_sigma',
]
