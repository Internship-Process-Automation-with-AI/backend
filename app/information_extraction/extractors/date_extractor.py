"""
Date extractor for parsing dates from work certificates.
"""

import logging
import re
from datetime import date
from typing import Optional, Tuple

from dateutil import parser

from ..config import DATE_PATTERNS

logger = logging.getLogger(__name__)


class DateExtractor:
    """Extractor for dates from work certificates."""

    def __init__(self):
        self.date_patterns = DATE_PATTERNS

    def extract_dates(
        self, text: str, language: str = "finnish"
    ) -> Tuple[Optional[date], Optional[date], float]:
        """Extract start and end dates from text."""
        start_date = None
        end_date = None
        confidence = 0.0

        # Get patterns for the specified language
        patterns = self.date_patterns.get(language, self.date_patterns["finnish"])

        # Find all date matches
        all_dates = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    # Parse the matched date
                    date_str = match.group(0)
                    parsed_date = self.parse_date(date_str, language)
                    if parsed_date:
                        all_dates.append(parsed_date)
                except Exception as e:
                    logger.debug(f"Failed to parse date {match.group(0)}: {e}")

        if len(all_dates) >= 2:
            # Sort dates and assume first is start, last is end
            all_dates.sort()
            start_date = all_dates[0]
            end_date = all_dates[-1]
            confidence = 0.8
        elif len(all_dates) == 1:
            # Only one date found, assume it's the start date
            start_date = all_dates[0]
            confidence = 0.5

        return start_date, end_date, confidence

    def parse_date(self, date_str: str, language: str) -> Optional[date]:
        """Parse date string to date object."""
        try:
            # Clean the date string
            date_str = date_str.strip()

            # Handle Finnish date format DD.MM.YYYY
            if language == "finnish" and "." in date_str:
                # Convert DD.MM.YYYY to YYYY-MM-DD for parsing
                if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", date_str):
                    day, month, year = date_str.split(".")
                    return date(int(year), int(month), int(day))
                elif re.match(r"\d{1,2}\.\d{1,2}\.\d{2}", date_str):
                    day, month, year = date_str.split(".")
                    # Assume 20xx for 2-digit years
                    full_year = 2000 + int(year) if int(year) < 50 else 1900 + int(year)
                    return date(full_year, int(month), int(day))

            # Handle English month names (e.g., "Oct 10, 2012")
            if language == "english":
                # Month name to number mapping
                month_map = {
                    "jan": 1,
                    "feb": 2,
                    "mar": 3,
                    "apr": 4,
                    "may": 5,
                    "jun": 6,
                    "jul": 7,
                    "aug": 8,
                    "sep": 9,
                    "oct": 10,
                    "nov": 11,
                    "dec": 12,
                }

                # Pattern for "Month DD, YYYY" or "Month DD YYYY"
                month_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})"
                match = re.match(month_pattern, date_str, re.IGNORECASE)
                if match:
                    month_name, day, year = match.groups()
                    month_num = month_map[month_name.lower()]
                    return date(int(year), month_num, int(day))

            # Use dateutil for other formats
            parsed = parser.parse(date_str, dayfirst=(language == "finnish"))
            return parsed.date()

        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            return None

    def extract_work_period(
        self, start_date: Optional[date], end_date: Optional[date]
    ) -> Optional[str]:
        """Calculate work period duration."""
        if not start_date or not end_date:
            return None

        if start_date > end_date:
            # Swap dates if they're in wrong order
            start_date, end_date = end_date, start_date

        # Calculate difference
        delta = end_date - start_date
        years = delta.days // 365
        months = (delta.days % 365) // 30
        days = delta.days % 30

        # Format the period
        parts = []
        if years > 0:
            parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months > 0:
            parts.append(f"{months} month{'s' if months != 1 else ''}")
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")

        return " ".join(parts) if parts else "Less than 1 day"
