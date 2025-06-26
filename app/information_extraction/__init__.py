"""
Information Extraction Package

This package handles extracting structured information from OCR text output,
specifically designed for work certificates and employment documents.
"""

from .extractor import InformationExtractor
from .models import ExtractedData, ExtractionResult

__all__ = ["InformationExtractor", "ExtractedData", "ExtractionResult"]
