"""
Main information extractor that orchestrates extraction from OCR text.
"""

import logging
import time

from .config import EXTRACTION_SETTINGS
from .extractors import DateExtractor, EnglishExtractor, FinnishExtractor
from .models import ExtractedData, ExtractionResult

logger = logging.getLogger(__name__)


class InformationExtractor:
    """Main information extraction engine for work certificates."""

    def __init__(self):
        """Initialize the information extraction system."""
        self.finnish_extractor = FinnishExtractor()
        self.english_extractor = EnglishExtractor()
        self.date_extractor = DateExtractor()

        logger.info("Information Extraction system initialized")

    def detect_language(self, text: str) -> str:
        """Detect the language of the text (Finnish or English)."""
        text_lower = text.lower()

        # Finnish indicators
        finnish_indicators = [
            "työtodistus",
            "työntekijä",
            "työnantaja",
            "tehtävä",
            "työskenteli",
            "alkoi",
            "päättyi",
            "kuvaus",
        ]

        # English indicators
        english_indicators = [
            "work certificate",
            "employee",
            "employer",
            "position",
            "worked",
            "started",
            "ended",
            "description",
        ]

        finnish_count = sum(
            1 for indicator in finnish_indicators if indicator in text_lower
        )
        english_count = sum(
            1 for indicator in english_indicators if indicator in text_lower
        )

        if finnish_count > english_count:
            return "finnish"
        elif english_count > finnish_count:
            return "english"
        else:
            # Default to Finnish for Finnish work certificates
            return "finnish"

    def extract_information(self, text: str) -> ExtractionResult:
        """
        Extract structured information from OCR text.

        Args:
            text: Text extracted from OCR processing

        Returns:
            ExtractionResult with extracted data and confidence scores
        """
        start_time = time.time()

        try:
            # Clean and validate input
            if not text or len(text.strip()) < 10:
                return ExtractionResult(
                    success=False,
                    extracted_data=ExtractedData(),
                    overall_confidence=0.0,
                    processing_time=time.time() - start_time,
                    engine="information_extractor",
                    errors=["Text too short or empty"],
                )

            # Detect language
            language = self.detect_language(text)
            logger.info(f"Detected language: {language}")

            # Choose appropriate extractor
            if language == "finnish":
                extractor = self.finnish_extractor
            else:
                extractor = self.english_extractor

            # Extract basic information
            extracted_data = extractor.extract_all(text)
            extracted_data.language = language

            # Extract dates using specialized date extractor
            start_date, end_date, date_confidence = self.date_extractor.extract_dates(
                text, language
            )
            extracted_data.start_date = start_date
            extracted_data.end_date = end_date

            # Calculate work period
            if start_date and end_date:
                extracted_data.work_period = self.date_extractor.extract_work_period(
                    start_date, end_date
                )

            # Update confidence scores
            if extracted_data.confidence_scores:
                extracted_data.confidence_scores["start_date"] = date_confidence
                extracted_data.confidence_scores["end_date"] = date_confidence

            # Calculate overall confidence
            confidence_scores = extracted_data.confidence_scores or {}
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(
                    confidence_scores
                )
            else:
                overall_confidence = 0.0

            # Determine success
            success = overall_confidence >= EXTRACTION_SETTINGS["min_confidence"]

            processing_time = time.time() - start_time

            logger.info(
                f"Information extraction completed in {processing_time:.2f}s with confidence {overall_confidence:.2f}"
            )

            return ExtractionResult(
                success=success,
                extracted_data=extracted_data,
                overall_confidence=overall_confidence,
                processing_time=processing_time,
                engine="information_extractor",
            )

        except Exception as e:
            logger.error(f"Error in information extraction: {e}")
            processing_time = time.time() - start_time

            return ExtractionResult(
                success=False,
                extracted_data=ExtractedData(),
                overall_confidence=0.0,
                processing_time=processing_time,
                engine="information_extractor",
                errors=[str(e)],
            )

    def get_extraction_stats(self) -> dict:
        """Get statistics about the extraction system."""
        return {
            "supported_languages": ["finnish", "english"],
            "extraction_engines": ["rule_based"],
            "min_confidence_threshold": EXTRACTION_SETTINGS["min_confidence"],
            "max_text_length": EXTRACTION_SETTINGS["max_text_length"],
        }
