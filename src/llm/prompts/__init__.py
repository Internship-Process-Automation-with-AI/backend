"""
Prompts package for LLM-based work certificate processing.
Contains extraction and evaluation prompts.
"""

from src.llm.prompts.correction import CORRECTION_PROMPT
from src.llm.prompts.evaluation import EVALUATION_PROMPT
from src.llm.prompts.extraction import EXTRACTION_PROMPT
from src.llm.prompts.validation import VALIDATION_PROMPT

__all__ = [
    "EXTRACTION_PROMPT",
    "EVALUATION_PROMPT",
    "VALIDATION_PROMPT",
    "CORRECTION_PROMPT",
]
