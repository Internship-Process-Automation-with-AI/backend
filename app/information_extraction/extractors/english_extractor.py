"""
English-specific information extractor for work certificates.
"""

import logging
import re
from typing import Optional, Tuple

from ..config import ENGLISH_KEYWORDS
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class EnglishExtractor(BaseExtractor):
    """Extractor for English work certificates and employment documents."""

    def __init__(self):
        super().__init__()
        self.keywords = ENGLISH_KEYWORDS

    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type from English text."""
        text_lower = text.lower()
        matches = []

        # Check for work certificate keywords
        for doc_type, keywords in self.keywords["document_types"].items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append(doc_type)
                    break

        if matches:
            return matches[0], 0.9  # High confidence for English documents

        return None, 0.0

    def extract_employee_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract employee name from English text."""
        # Look for patterns like "employee [Name]" or "[Name] worked"
        patterns = [
            r"employee\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+worked",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+was\s+employed",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+served\s+as",
            r"name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                return name, 0.8

        return None, 0.0

    def extract_position(self, text: str) -> Tuple[Optional[str], float]:
        """Extract job position from English text."""
        # Look for position keywords and extract surrounding text
        position_keywords = self.keywords["position_keywords"]
        context_matches = self.find_keywords_in_context(text, position_keywords)

        if context_matches:
            # Extract the most likely position from context
            for context in context_matches:
                # Look for patterns like "position: [Position]" or "role: [Role]"
                patterns = [
                    r"position[:\s]+([^,\n]+)",
                    r"role[:\s]+([^,\n]+)",
                    r"job\s+title[:\s]+([^,\n]+)",
                    r"duties[:\s]+([^,\n]+)",
                    r"responsibilities[:\s]+([^,\n]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        position = match.group(1).strip()
                        # Clean up the position
                        position = re.sub(r"[^\w\s]", "", position).strip()
                        if len(position) > 3:  # Minimum length check
                            return position, 0.7

        return None, 0.0

    def extract_employer(self, text: str) -> Tuple[Optional[str], float]:
        """Extract employer/company name from English text."""
        # Look for company patterns with English suffixes
        company_patterns = [
            r"([A-Z][a-z\s]+(?:Ltd|Inc|Corp|LLC|Company|Limited))",
            r"employer[:\s]+([^,\n]+)",
            r"company[:\s]+([^,\n]+)",
            r"organization[:\s]+([^,\n]+)",
            r"firm[:\s]+([^,\n]+)",
        ]

        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                # Clean up the company name
                company = re.sub(r"[^\w\s]", "", company).strip()
                if len(company) > 3:
                    return company, 0.8

        return None, 0.0

    def extract_description(self, text: str) -> Tuple[Optional[str], float]:
        """Extract job description from English text."""
        # Look for description keywords
        desc_keywords = ["description", "duties", "responsibilities", "work", "tasks"]
        context_matches = self.find_keywords_in_context(
            text, desc_keywords, context_lines=3
        )

        if context_matches:
            # Extract description from context
            for context in context_matches:
                # Look for patterns like "description: [Description]"
                patterns = [
                    r"description[:\s]+([^.\n]+)",
                    r"duties[:\s]+([^.\n]+)",
                    r"responsibilities[:\s]+([^.\n]+)",
                    r"work[:\s]+([^.\n]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                        if len(description) > 10:  # Minimum length
                            return description, 0.6

        return None, 0.0
