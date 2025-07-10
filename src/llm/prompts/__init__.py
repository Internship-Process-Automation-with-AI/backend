"""
Prompts package for LLM-based work certificate processing.
Contains extraction and evaluation prompts.
"""

from .correction import CORRECTION_PROMPT
from .evaluation import EVALUATION_PROMPT
from .extraction import EXTRACTION_PROMPT
from .validation import VALIDATION_PROMPT

__all__ = [
    "EXTRACTION_PROMPT",
    "EVALUATION_PROMPT",
    "VALIDATION_PROMPT",
    "CORRECTION_PROMPT",
]
