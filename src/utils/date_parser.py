"""
Date parsing utilities for handling various date formats including Finnish dates.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def parse_finnish_date(date_str: str) -> Optional[str]:
    """
    Parse Finnish date formats and convert to YYYY-MM-DD format.

    Handles formats like:
    - "2. 6. 2025" -> "2025-06-02"
    - "31. 8. 2025" -> "2025-08-31"
    - "27.11.2009" -> "2009-11-27"
    - "13.2.1984" -> "1984-02-13"

    Args:
        date_str: Date string in various Finnish formats

    Returns:
        Date string in YYYY-MM-DD format, or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # Clean the input string
    date_str = date_str.strip()

    # Handle different Finnish date formats
    patterns = [
        # Format: "D. M. YYYY" (e.g., "2. 6. 2025") - with spaces around dots
        (
            r"^(\d{1,2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{4})$",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        ),
        # Format: "DD.MM.YYYY" (e.g., "27.11.2009")
        (
            r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        ),
        # Format: "D.M.YYYY" (e.g., "2.6.2025")
        (
            r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        ),
        # Format: "DD/MM/YYYY" (e.g., "27/11/2009")
        (
            r"^(\d{1,2})/(\d{1,2})/(\d{4})$",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        ),
        # Format: "D/M/YYYY" (e.g., "2/6/2025")
        (
            r"^(\d{1,2})/(\d{1,2})/(\d{4})$",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
        ),
    ]

    for pattern, formatter in patterns:
        match = re.match(pattern, date_str)
        if match:
            try:
                formatted_date = formatter(match)
                # Validate the date
                datetime.strptime(formatted_date, "%Y-%m-%d")
                logger.info(
                    f"Successfully parsed Finnish date: '{date_str}' -> '{formatted_date}'"
                )
                return formatted_date
            except ValueError as e:
                logger.warning(
                    f"Invalid date after formatting: '{date_str}' -> '{formatted_date}': {e}"
                )
                continue

    # If no pattern matches, try to parse as already formatted YYYY-MM-DD
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        logger.info(f"Date already in correct format: '{date_str}'")
        return date_str
    except ValueError:
        pass

    logger.warning(f"Could not parse date: '{date_str}'")
    return None


def is_future_date(date_str: str, current_date: Optional[date] = None) -> bool:
    """
    Check if a date string represents a future date.

    Args:
        date_str: Date string in YYYY-MM-DD format
        current_date: Current date to compare against (defaults to today)

    Returns:
        True if the date is in the future, False otherwise
    """
    if not date_str:
        return False

    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if current_date is None:
            current_date = date.today()
        return parsed_date > current_date
    except ValueError:
        logger.warning(f"Could not parse date for future check: '{date_str}'")
        return False


def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that start_date is before end_date.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not start_date or not end_date:
        return True, None

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start > end:
            return False, f"Start date {start_date} is after end date {end_date}"

        return True, None
    except ValueError as e:
        return False, f"Invalid date format: {e}"


def parse_and_validate_dates(
    start_date: str, end_date: str, current_date: Optional[date] = None
) -> dict:
    """
    Parse Finnish dates and validate them.

    Args:
        start_date: Start date string (various formats)
        end_date: End date string (various formats)
        current_date: Current date for future date validation

    Returns:
        Dictionary with parsed dates and validation results
    """
    result = {
        "start_date": None,
        "end_date": None,
        "start_date_parsed": None,
        "end_date_parsed": None,
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }

    # Parse start date
    if start_date:
        parsed_start = parse_finnish_date(start_date)
        result["start_date"] = start_date
        result["start_date_parsed"] = parsed_start

        if parsed_start:
            if is_future_date(parsed_start, current_date):
                result["warnings"].append(f"Start date {parsed_start} is in the future")
        else:
            result["errors"].append(f"Could not parse start date: {start_date}")
            result["is_valid"] = False

    # Parse end date
    if end_date:
        parsed_end = parse_finnish_date(end_date)
        result["end_date"] = end_date
        result["end_date_parsed"] = parsed_end

        if parsed_end:
            if is_future_date(parsed_end, current_date):
                result["warnings"].append(f"End date {parsed_end} is in the future")
        else:
            result["errors"].append(f"Could not parse end date: {end_date}")
            result["is_valid"] = False

    # Validate date range
    if result["start_date_parsed"] and result["end_date_parsed"]:
        is_valid_range, error_msg = validate_date_range(
            result["start_date_parsed"], result["end_date_parsed"]
        )
        if not is_valid_range:
            result["errors"].append(error_msg)
            result["is_valid"] = False

    return result
