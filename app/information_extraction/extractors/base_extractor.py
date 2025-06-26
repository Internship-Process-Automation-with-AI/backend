"""
Base extractor class for information extraction.
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional, Tuple

from ..models import ExtractedData

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for all information extractors."""

    def __init__(self):
        self.confidence_scores = {}

    @abstractmethod
    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type from text."""
        pass

    @abstractmethod
    def extract_employee_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract employee name from text."""
        pass

    @abstractmethod
    def extract_position(self, text: str) -> Tuple[Optional[str], float]:
        """Extract job position/title from text."""
        pass

    @abstractmethod
    def extract_employer(self, text: str) -> Tuple[Optional[str], float]:
        """Extract employer/company name from text."""
        pass

    def extract_dates(self, text: str) -> Tuple[Optional[date], Optional[date], float]:
        """Extract start and end dates from text."""
        # This will be implemented by DateExtractor
        return None, None, 0.0

    def extract_description(self, text: str) -> Tuple[Optional[str], float]:
        """Extract job description/responsibilities from text."""
        # Default implementation
        return None, 0.0

    def calculate_confidence(self, matches: List[str], total_attempts: int) -> float:
        """Calculate confidence score based on matches."""
        if total_attempts == 0:
            return 0.0
        return len(matches) / total_attempts

    def find_keywords_in_context(
        self, text: str, keywords: List[str], context_lines: int = 2
    ) -> List[str]:
        """Find keywords and return surrounding context."""
        lines = text.split("\n")
        matches = []

        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    # Get context lines
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = " ".join(lines[start:end])
                    matches.append(context)

        return matches

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters that might interfere
        text = re.sub(r"[^\w\s\.\,\-\/\(\)]", "", text)
        return text.strip()

    def extract_all(self, text: str) -> ExtractedData:
        """Extract all information from text."""
        text = self.clean_text(text)

        # Extract each field
        doc_type, doc_confidence = self.extract_document_type(text)
        employee_name, name_confidence = self.extract_employee_name(text)
        position, pos_confidence = self.extract_position(text)
        employer, emp_confidence = self.extract_employer(text)
        start_date, end_date, date_confidence = self.extract_dates(text)
        description, desc_confidence = self.extract_description(text)

        # Store confidence scores
        self.confidence_scores = {
            "document_type": doc_confidence,
            "employee_name": name_confidence,
            "position": pos_confidence,
            "employer": emp_confidence,
            "start_date": date_confidence,
            "end_date": date_confidence,
            "description": desc_confidence,
        }

        return ExtractedData(
            document_type=doc_type,
            employee_name=employee_name,
            position=position,
            employer=employer,
            start_date=start_date,
            end_date=end_date,
            description=description,
            confidence_scores=self.confidence_scores,
        )
