"""
Finnish OCR Error Correction Module

This module handles correction of common OCR errors in Finnish text,
particularly for work certificates and employment documents.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


class FinnishOCRCorrector:
    """Corrects common OCR errors in Finnish text."""

    def __init__(self):
        """Initialize the Finnish OCR corrector with correction patterns."""
        self.finnish_corrections = self._load_finnish_corrections()
        self.finnish_patterns = self._load_finnish_patterns()

    def _load_finnish_corrections(self) -> Dict[str, str]:
        """Load comprehensive Finnish word corrections."""
        return {
            # Main headers and titles
            "TYONANTAJA": "TYÖNANTAJA",
            "TYONTEKIJA": "TYÖNTEKIJÄ",
            "TYOTODISTUS": "TYÖTODISTUS",
            "TYOSUHTEEN": "TYÖSUHTEEN",
            "TYOSUHTE": "TYÖSUHTE",
            "TYOTEHTAVAT": "TYÖTEHTÄVÄT",
            "TYONTEHTAVAT": "TYÖNTEHTÄVÄT",
            "TYONTEHTAVA": "TYÖNTEHTÄVÄ",
            "TYOTEHTAVA": "TYÖTEHTÄVÄ",
            "TYOSUHDE": "TYÖSUHDE",
            "TYONIMIKE": "TYÖNIMIKE",
            # Common work certificate terms (mixed case)
            "Ty6nantaja": "Työnantaja",
            "Tyéntekija": "Työntekijä",
            "Tydésuhde": "Työsuhde",
            "Tyésuhteen": "Työsuhteen",
            "Ty6taito": "Työtaito",
            "Ty6n suorittamispaikka": "Työn suorittamispaikka",
            "Ty6ntekija": "Työntekijä",
            "Ty6ntekijan": "Työntekijan",
            "Ty6todistus": "Työtodistus",
            "Ty6suhte": "Työsuhte",
            "Ty6suhteen": "Työsuhteen",
            "Ty6ntehtavat": "Työntehtävät",
            "Ty6ntehtava": "Työntehtävä",
            "Ty6tehtavat": "Työtehtävät",
            "Ty6tehtava": "Työtehtävä",
            "TYOVALINE": "TYÖVALINE",
            # Job-related terms (lowercase)
            "ty6ntekija": "työntekijä",
            "ty6nantaja": "työnantaja",
            "ty6todistus": "työtodistus",
            "ty6suhte": "työsuhte",
            "ty6ntehtavat": "työntehtävät",
            "ty6suhteen": "työsuhteen",
            "tyékohde": "työkohde",
            "työtehtavat": "työtehtävät",
            "ty6ntekijan": "työntekijan",
            "kassaty6skentely": "kassatyöskentely",
            "ty6skentely": "työskentely",
            "ty6aika": "työaika",
            "ty6suhteessa": "työsuhteessa",
            # Evaluation and description terms
            "paattymisen": "päättymisen",
            "pyynnosta": "pyynnöstä",
            "pyynnésta": "pyynnöstä",
            "vuosilomasijai": "vuosilomasijai",
            "kayttos": "käyttös",
            "Kaytés": "Käyttös",
            "kiitettava": "kiitettävä",
            "allekirjoitus": "allekirjoitus",
            "kauppias": "kauppias",
            "PAATTYMISEN": "PAATTYMISEN",
            "päättyi": "päättyi",
            "päättynyt": "päättynyt",
            "alkoi": "alkoi",
            "aloitti": "aloitti",
            "aloittanut": "aloittanut",
            # Personal information
            "Henkilétunnus": "Henkilötunnus",
            "henkilétunnus": "henkilötunnus",
            "henkilötunnus": "henkilötunnus",
            "henkilötunnuksella": "henkilötunnuksella",
            # Common OCR substitutions
            "a6": "ä",
            "o6": "ö",
            "A6": "Ä",
            "O6": "Ö",
            "aé": "ä",
            "oé": "ö",
            "Aé": "Ä",
            "Oé": "Ö",
            "aë": "ä",
            "oë": "ö",
            "Aë": "Ä",
            "Oë": "Ö",
            # Word endings that should have ä/ö
            "avat": "ävät",
            "AVAT": "ÄVÄT",
            "ava": "ävä",
            "AVA": "ÄVÄ",
            "avaa": "ävää",
            "AVAA": "ÄVÄÄ",
            "avasta": "ävästä",
            "AVASTA": "ÄVÄSTÄ",
            "avassa": "ävässä",
            "AVASSA": "ÄVÄSSÄ",
            # Common Finnish words with OCR errors
            "ty6": "työ",
            "TY6": "TYÖ",
            "ty6n": "työn",
            "TY6N": "TYÖN",
            "ty6nt": "työnt",
            "TY6NT": "TYÖNT",
            "ty6na": "työnä",
            "TY6NA": "TYÖNÄ",
            "ty6s": "työs",
            "TY6S": "TYÖS",
            "ty6t": "työt",
            "TY6T": "TYÖT",
            "ty6ssä": "työssä",
            "TY6SSÄ": "TYÖSSÄ",
            "ty6stä": "työstä",
            "TY6STÄ": "TYÖSTÄ",
            "ty6tä": "työtä",
            "TY6TÄ": "TYÖTÄ",
            "ty6nä": "työnä",
            "TY6NÄ": "TYÖNÄ",
        }

    def _load_finnish_patterns(self) -> List[str]:
        """Load Finnish word patterns for context detection."""
        return [
            "TYÖ",
            "TYÖN",
            "TYÖNT",
            "TYÖNA",
            "TYÖS",
            "TYÖT",
            "TY6",
            "TY6N",
            "TY6NT",
            "TY6NA",
            "TY6S",
            "TY6T",
        ]

    def add_correction(self, wrong: str, correct: str) -> None:
        """
        Add a new correction pattern.

        Args:
            wrong: The incorrect text to replace
            correct: The correct text to replace with
        """
        self.finnish_corrections[wrong] = correct
        logger.info(f"Added Finnish correction: '{wrong}' -> '{correct}'")

    def add_corrections(self, corrections: Dict[str, str]) -> None:
        """
        Add multiple correction patterns at once.

        Args:
            corrections: Dictionary of wrong -> correct mappings
        """
        self.finnish_corrections.update(corrections)
        logger.info(f"Added {len(corrections)} Finnish corrections")

    def remove_correction(self, wrong: str) -> bool:
        """
        Remove a correction pattern.

        Args:
            wrong: The incorrect text pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        if wrong in self.finnish_corrections:
            del self.finnish_corrections[wrong]
            logger.info(f"Removed Finnish correction: '{wrong}'")
            return True
        return False

    def get_corrections(self) -> Dict[str, str]:
        """Get all current correction patterns."""
        return self.finnish_corrections.copy()

    def clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text by removing common artifacts and fixing line breaks.
        Conservative approach to preserve important information.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Step 1: Fix common line break issues
        # Replace common line break artifacts with proper line breaks
        line_break_fixes = {
            "\r\n": "\n",  # Windows line breaks
            "\r": "\n",  # Old Mac line breaks
            "\f": "\n",  # Form feed
            "\v": "\n",  # Vertical tab
        }

        for wrong, correct in line_break_fixes.items():
            text = text.replace(wrong, correct)

        # Step 2: Remove only obvious OCR artifacts (conservative approach)
        import re

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Only remove lines that are clearly gibberish (very conservative)
            if self._is_obviously_gibberish(line):
                continue

            # Clean the line conservatively
            cleaned_line = self._clean_line_conservatively(line)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)

        # Join lines back together
        text = "\n".join(cleaned_lines)

        # Step 3: Remove control characters and other problematic characters
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Remove multiple consecutive spaces
        text = re.sub(r" +", " ", text)

        # Remove multiple consecutive newlines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Step 4: Fix common OCR spacing issues
        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        # Fix spacing around parentheses
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)

        # Step 5: Final cleanup
        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _is_obviously_gibberish(self, line: str) -> bool:
        """
        Check if a line is obviously gibberish (very conservative approach).

        Args:
            line: Line to check

        Returns:
            True if line is obviously gibberish
        """
        if not line.strip():
            return True

        # Only remove lines that are clearly artifacts
        # 1. Lines that are only special characters (no alphanumeric content)
        if re.match(r"^[—–—_|#$%&*()[]{}<>+=~`!@^\s]+$", line):
            return True

        # 2. Lines that are only repeated characters (like "==========")
        if len(line.strip()) >= 5:
            unique_chars = set(line.strip())
            if (
                len(unique_chars) == 1
                and line.strip()[0] in r"=_\-—–\.\*\#\|\+\~\`\^\@\$\%\&\(\)\[\]\{\}\<\>"
            ):
                return True

        # 3. Lines that are very short and contain only artifacts
        if len(line.strip()) <= 3:
            alphanumeric = sum(1 for c in line if c.isalnum())
            if alphanumeric == 0:
                return True

        # 4. Lines that are mostly artifacts (90% or more)
        total_chars = len(line.strip())
        if total_chars > 0:
            artifact_chars = sum(1 for c in line if c in "—–—_|#$%&*()[]{}<>+=~`!@^")
            if artifact_chars / total_chars > 0.9:  # 90% or more are artifacts
                return True

        return False

    def _clean_line_conservatively(self, line: str) -> str:
        """
        Clean a single line of text conservatively.

        Args:
            line: Line to clean

        Returns:
            Cleaned line
        """
        import re

        # Only remove excessive special characters at the beginning and end
        line = re.sub(r"^[—–—_|#$%&*()\[\]{}<>+=~`!@^\s]+", "", line)
        line = re.sub(r"[—–—_|#$%&*()\[\]{}<>+=~`!@^\s]+$", "", line)

        # Remove trailing spaces
        line = line.rstrip()

        # Clean up multiple spaces that might have been created
        line = re.sub(r"\s+", " ", line)

        return line.strip()

    def correct_text(self, text: str) -> str:
        """
        Correct common OCR errors in Finnish text.

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        if not text:
            return text

        # Step 1: Clean the text first
        cleaned_text = self.clean_ocr_text(text)

        # Step 2: Apply specific word corrections
        corrected_text = cleaned_text
        for wrong, correct in self.finnish_corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)

        # Step 3: Context-aware character corrections (more conservative)
        lines = corrected_text.split("\n")
        corrected_lines = []

        for line in lines:
            # Look for patterns that suggest Finnish words
            if any(
                finnish_word in line.upper() for finnish_word in self.finnish_patterns
            ):
                # This line likely contains Finnish words, apply character corrections
                line = self._apply_context_corrections(line)

            corrected_lines.append(line)

        return "\n".join(corrected_lines)

    def _apply_context_corrections(self, line: str) -> str:
        """Apply context-aware corrections to a line."""
        # Basic character substitutions
        line = line.replace("6", "ö").replace("é", "ö")

        # Common Finnish word patterns
        line = line.replace("TYON", "TYÖN").replace("tyon", "työn")
        line = line.replace("TYONTEKIJA", "TYÖNTEKIJÄ").replace(
            "tyontekija", "työntekijä"
        )
        line = line.replace("TYONANTAJA", "TYÖNANTAJA").replace(
            "tyonantaja", "työnantaja"
        )
        line = line.replace("TYOSUHTE", "TYÖSUHTE").replace("tyosuhte", "työsuhte")
        line = line.replace("TYOTODISTUS", "TYÖTODISTUS").replace(
            "tyotodistus", "työtodistus"
        )
        line = line.replace("TYOTEHTAVAT", "TYÖTEHTÄVÄT").replace(
            "tyotehtavat", "työtehtävät"
        )
        line = line.replace("TYONTEHTAVAT", "TYÖNTEHTÄVÄT").replace(
            "tyontehtavat", "työntehtävät"
        )

        # Look for words ending with 'avat' (should be 'ävät')
        if "avat" in line.lower():
            line = line.replace("avat", "ävät").replace("AVAT", "ÄVÄT")

        # Look for words with 'é' that should be 'ö' in Finnish context
        if "é" in line and any(
            finnish_word in line.lower()
            for finnish_word in ["työ", "työn", "työs", "pyynn", "ty6", "ty6n", "ty6s"]
        ):
            line = line.replace("é", "ö")

        # Look for common Finnish character patterns
        if any(pattern in line.lower() for pattern in ["ty6", "ty6n", "ty6s", "ty6t"]):
            line = line.replace("6", "ö")

        return line

    def get_correction_stats(self) -> Dict:
        """Get statistics about the correction system."""
        return {
            "total_corrections": len(self.finnish_corrections),
            "finnish_patterns": len(self.finnish_patterns),
            "corrections_by_length": {
                len(key): sum(
                    1 for k in self.finnish_corrections.keys() if len(k) == len(key)
                )
                for key in self.finnish_corrections.keys()
            },
        }


# Global instance for easy access
finnish_corrector = FinnishOCRCorrector()


def correct_finnish_ocr_errors(text: str) -> str:
    """
    Convenience function to correct Finnish OCR errors.

    Args:
        text: Raw OCR text

    Returns:
        Corrected text
    """
    return finnish_corrector.correct_text(text)


def clean_ocr_text(text: str) -> str:
    """
    Convenience function to clean OCR text.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    return finnish_corrector.clean_ocr_text(text)


def clean_ocr_text_conservative(text: str) -> str:
    """
    Conservative OCR text cleaning that preserves important information.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Step 1: Fix common line break issues
    line_break_fixes = {
        "\r\n": "\n",  # Windows line breaks
        "\r": "\n",  # Old Mac line breaks
        "\f": "\n",  # Form feed
        "\v": "\n",  # Vertical tab
    }

    for wrong, correct in line_break_fixes.items():
        text = text.replace(wrong, correct)

    # Step 2: Remove only obvious OCR artifacts (very conservative)
    import re

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Only remove lines that are clearly gibberish
        if _is_obviously_gibberish(line):
            continue

        # Clean the line conservatively
        cleaned_line = _clean_line_conservatively(line)
        if cleaned_line.strip():
            cleaned_lines.append(cleaned_line)

    # Join lines back together
    text = "\n".join(cleaned_lines)

    # Step 3: Remove control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Step 4: Fix spacing issues
    text = re.sub(r" +", " ", text)  # Multiple spaces to single
    text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines to max 2
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # Remove spaces before punctuation
    text = re.sub(r"\(\s+", "(", text)  # Fix spacing around parentheses
    text = re.sub(r"\s+\)", ")", text)

    # Step 5: Final cleanup
    return text.strip()


def _is_obviously_gibberish(line: str) -> bool:
    """
    Check if a line is obviously gibberish (very conservative approach).

    Args:
        line: Line to check

    Returns:
        True if line is obviously gibberish
    """
    if not line.strip():
        return True

    # Only remove lines that are clearly artifacts
    # 1. Lines that are only special characters (no alphanumeric content)
    if re.match(r"^[—–—_|#$%&*()[]{}<>+=~`!@^\s]+$", line):
        return True

    # 2. Lines that are only repeated characters (like "==========")
    if len(line.strip()) >= 5:
        unique_chars = set(line.strip())
        if (
            len(unique_chars) == 1
            and line.strip()[0] in r"=_\-—–\.\*\#\|\+\~\`\^\@\$\%\&\(\)\[\]\{\}\<\>"
        ):
            return True

    # 3. Lines that are very short and contain only artifacts
    if len(line.strip()) <= 3:
        alphanumeric = sum(1 for c in line if c.isalnum())
        if alphanumeric == 0:
            return True

    # 4. Lines that are mostly artifacts (90% or more)
    total_chars = len(line.strip())
    if total_chars > 0:
        artifact_chars = sum(1 for c in line if c in "—–—_|#$%&*()[]{}<>+=~`!@^")
        if artifact_chars / total_chars > 0.9:  # 90% or more are artifacts
            return True

    return False


def _clean_line_conservatively(line: str) -> str:
    """
    Clean a single line of text conservatively.

    Args:
        line: Line to clean

    Returns:
        Cleaned line
    """
    import re

    # Only remove excessive special characters at the beginning and end
    line = re.sub(r"^[—–—_|#$%&*()\[\]{}<>+=~`!@^\s]+", "", line)
    line = re.sub(r"[—–—_|#$%&*()\[\]{}<>+=~`!@^\s]+$", "", line)

    # Remove trailing spaces
    line = line.rstrip()

    # Clean up multiple spaces that might have been created
    line = re.sub(r"\s+", " ", line)

    return line.strip()
