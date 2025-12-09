"""
Gravyflow Detection Module

GPU-accelerated traditional detection methods for baseline comparison
with deep learning approaches.

Available Classes:
- MatchedFilter: GPU matched filtering with on-the-fly ripple templates
- MatchedFilterLayer: Keras layer wrapper for pipeline compatibility
- TemplateGrid: Parameter space coverage for template banks
"""

from gravyflow.src.detection.template_grid import TemplateGrid
from gravyflow.src.detection.snr import matched_filter_fft, optimal_snr, template_sigma
from gravyflow.src.detection.matched_filter import MatchedFilter, MatchedFilterLayer

__all__ = [
    'MatchedFilter',
    'MatchedFilterLayer',
    'TemplateGrid',
    'matched_filter_fft',
    'optimal_snr',
    'template_sigma',
]
