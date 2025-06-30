"""
Prompts package for LLM-based work certificate processing.
Contains extraction and evaluation prompts.
"""

from .evaluation import EVALUATION_PROMPT
from .extraction import EXTRACTION_PROMPT

__all__ = ["EXTRACTION_PROMPT", "EVALUATION_PROMPT"]
