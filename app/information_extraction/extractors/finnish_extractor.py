"""
Finnish-specific information extractor for work certificates.
"""

import logging
import re
from typing import Optional, Tuple

from ..config import FINNISH_KEYWORDS
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class FinnishExtractor(BaseExtractor):
    """Extractor for Finnish work certificates and employment documents."""

    def __init__(self):
        super().__init__()
        self.keywords = FINNISH_KEYWORDS

    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type from Finnish text."""
        text_lower = text.lower()
        matches = []

        # Check for work certificate keywords
        for doc_type, keywords in self.keywords["document_types"].items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append(doc_type)
                    break

        if matches:
            return matches[0], 0.9  # High confidence for Finnish documents

        return None, 0.0

    def extract_employee_name(self, text: str) -> Tuple[Optional[str], float]:
        """Extract employee name from Finnish text."""
        # Look for patterns like "työntekijä [Name]" or "henkilö [Name]"
        patterns = [
            r"työntekijä\s+([A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+)",
            r"henkilö\s+([A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+)",
            r"([A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+)\s+on\s+työskennellyt",
            r"([A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+)\s+työskenteli",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                return name, 0.8

        return None, 0.0

    def extract_position(self, text: str) -> Tuple[Optional[str], float]:
        """Extract job position from Finnish text."""
        # Look for position keywords and extract surrounding text
        position_keywords = self.keywords["position_keywords"]
        context_matches = self.find_keywords_in_context(text, position_keywords)

        if context_matches:
            # Extract the most likely position from context
            for context in context_matches:
                # Look for patterns like "tehtävä: [Position]" or "toimi [Position]"
                patterns = [
                    r"tehtävä[:\s]+([^,\n]+)",
                    r"toimi[:\s]+([^,\n]+)",
                    r"työtehtävä[:\s]+([^,\n]+)",
                    r"asema[:\s]+([^,\n]+)",
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
        """Extract employer/company name from Finnish text."""
        # Look for company patterns with Finnish suffixes
        company_patterns = [
            r"([A-ZÄÖÅ][a-zäöå\s]+(?:Oy|Ab|Ltd|Ky|Tmi|Yhtiö))",
            r"työnantaja[:\s]+([^,\n]+)",
            r"yritys[:\s]+([^,\n]+)",
            r"firma[:\s]+([^,\n]+)",
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
        """Extract job description from Finnish text."""
        # Look for description keywords
        desc_keywords = ["kuvaus", "vastuualueet", "tehtävät", "työkuvaus"]
        context_matches = self.find_keywords_in_context(
            text, desc_keywords, context_lines=3
        )

        if context_matches:
            # Extract description from context
            for context in context_matches:
                # Look for patterns like "kuvaus: [Description]"
                patterns = [
                    r"kuvaus[:\s]+([^.\n]+)",
                    r"vastuualueet[:\s]+([^.\n]+)",
                    r"tehtävät[:\s]+([^.\n]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        description = match.group(1).strip()
                        if len(description) > 10:  # Minimum length
                            return description, 0.6

        return None, 0.0
