"""
Tests for information extraction functionality.
"""

from datetime import date

import pytest

from app.information_extraction import InformationExtractor


class TestInformationExtraction:
    """Test information extraction functionality."""

    @pytest.fixture
    def extractor(self):
        """Create information extractor instance."""
        return InformationExtractor()

    def test_finnish_work_certificate_extraction(self, extractor):
        """Test extraction from Finnish work certificate."""
        finnish_text = """
        TYÖTODISTUS

        Työntekijä Matti Meikäläinen on työskennellyt yrityksessämme
        tehtävässä Software Developer alkaen 15.01.2023 ja päättyen 31.12.2023.

        Työnantaja: Tech Company Oy
        Tehtävä: Kehitti web-sovelluksia ja osallistui projektityöhön.

        Kuvaus: Työntekijä suoritti tehtävänsä hyvin ja oli luotettava.
        """

        result = extractor.extract_information(finnish_text)

        assert result.success
        assert result.extracted_data.language == "finnish"
        assert result.extracted_data.document_type == "work_certificate"
        assert result.extracted_data.employee_name == "Matti Meikäläinen"
        assert result.extracted_data.position == "Software Developer"
        assert result.extracted_data.employer == "Tech Company Oy"
        assert result.extracted_data.start_date == date(2023, 1, 15)
        assert result.extracted_data.end_date == date(2023, 12, 31)
        assert result.extracted_data.work_period == "11 months 16 days"

    def test_english_work_certificate_extraction(self, extractor):
        """Test extraction from English work certificate."""
        english_text = """
        WORK CERTIFICATE

        Employee John Smith worked in our company as a Software Engineer
        from January 15, 2023 to December 31, 2023.

        Employer: Tech Company Ltd
        Position: Developed web applications and participated in project work.

        Description: The employee performed their duties well and was reliable.
        """

        result = extractor.extract_information(english_text)

        assert result.success
        assert result.extracted_data.language == "english"
        assert result.extracted_data.document_type == "work_certificate"
        assert result.extracted_data.employee_name == "John Smith"
        assert result.extracted_data.employer == "Tech Company Ltd"
        assert result.extracted_data.start_date == date(2023, 1, 15)
        assert result.extracted_data.end_date == date(2023, 12, 31)

    def test_language_detection(self, extractor):
        """Test language detection functionality."""
        finnish_text = "työtodistus työntekijä työnantaja"
        english_text = "work certificate employee employer"

        assert extractor.detect_language(finnish_text) == "finnish"
        assert extractor.detect_language(english_text) == "english"

    def test_empty_text_handling(self, extractor):
        """Test handling of empty or short text."""
        result = extractor.extract_information("")

        assert not result.success
        assert result.overall_confidence == 0.0
        assert "Text too short or empty" in result.errors

    def test_extraction_stats(self, extractor):
        """Test extraction statistics."""
        stats = extractor.get_extraction_stats()

        assert "supported_languages" in stats
        assert "finnish" in stats["supported_languages"]
        assert "english" in stats["supported_languages"]
        assert "extraction_engines" in stats
        assert "rule_based" in stats["extraction_engines"]


if __name__ == "__main__":
    pytest.main([__file__])
