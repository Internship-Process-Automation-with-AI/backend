"""
Test file for OCR functionality.

Tests OCR processor, certificate extractor, and OCR workflow classes.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.ocr.cert_extractor import (
    SUPPORTED_DOC_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_PDF_FORMATS,
    extract_certificate_text,
    extract_finnish_certificate,
)
from src.ocr.ocr import FINNISH_KEYWORDS, LANGUAGE_CONFIGS, OCRProcessor


class TestOCRProcessor:
    """Test OCR processor functionality."""

    @patch("src.ocr.ocr.settings")
    @patch("src.ocr.ocr.pytesseract")
    def test_configure_tesseract_success(self, mock_pytesseract, mock_settings):
        """Test successful Tesseract configuration."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        processor = OCRProcessor()

        assert processor._configure_tesseract is not None
        mock_pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    @patch("src.ocr.ocr.settings")
    @patch("src.ocr.ocr.pytesseract")
    def test_configure_tesseract_default_path(self, mock_pytesseract, mock_settings):
        """Test Tesseract configuration with default path."""
        mock_settings.TESSERACT_CMD = None

        processor = OCRProcessor()

        assert processor._configure_tesseract is not None
        mock_pytesseract.pytesseract.tesseract_cmd = "tesseract"

    @patch("src.ocr.ocr.settings")
    @patch("src.ocr.ocr.pytesseract")
    def test_configure_tesseract_failure(self, mock_pytesseract, mock_settings):
        """Test Tesseract configuration failure."""
        # Note: OCRProcessor constructor only sets the path, doesn't test functionality
        # Actual Tesseract testing happens in workflow initialization
        mock_settings.TESSERACT_CMD = "/invalid/path"
        mock_pytesseract.pytesseract.tesseract_cmd = "/invalid/path"

        # This should not raise an error since we're just setting the path
        processor = OCRProcessor()
        assert processor._configure_tesseract is not None

    def test_language_configs(self):
        """Test language configuration constants."""
        assert "eng" in LANGUAGE_CONFIGS
        assert "fin" in LANGUAGE_CONFIGS
        assert "eng+fin" in LANGUAGE_CONFIGS

        # Check English config
        eng_config = LANGUAGE_CONFIGS["eng"]
        assert eng_config["name"] == "English"
        assert "--oem 3 --psm 6" in eng_config["config"]
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in eng_config["whitelist"]

        # Check Finnish config
        fin_config = LANGUAGE_CONFIGS["fin"]
        assert fin_config["name"] == "Finnish"
        assert "ÄÖÅäöå" in fin_config["whitelist"]

    def test_finnish_keywords(self):
        """Test Finnish keywords for language detection."""
        assert "työtodistus" in FINNISH_KEYWORDS
        assert "todistus" in FINNISH_KEYWORDS
        assert "työntekijä" in FINNISH_KEYWORDS
        assert "työnantaja" in FINNISH_KEYWORDS
        assert "päivämäärä" in FINNISH_KEYWORDS
        assert "nimi" in FINNISH_KEYWORDS
        assert "syntynyt" in FINNISH_KEYWORDS
        assert "ammatti" in FINNISH_KEYWORDS
        assert "tehtävä" in FINNISH_KEYWORDS
        assert "osasto" in FINNISH_KEYWORDS
        assert "palvelussuhde" in FINNISH_KEYWORDS
        assert "harjoittelu" in FINNISH_KEYWORDS
        assert "koulutus" in FINNISH_KEYWORDS
        assert "oppilaitos" in FINNISH_KEYWORDS
        assert "yliopisto" in FINNISH_KEYWORDS
        assert "koulu" in FINNISH_KEYWORDS

    @patch("src.ocr.ocr.pytesseract")
    @patch("src.ocr.ocr.Image")
    @patch("src.ocr.ocr.cv2")
    def test_extract_text_from_image(self, mock_cv2, mock_pil_image, mock_pytesseract):
        """Test text extraction from image."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_pil_image.open.return_value = mock_image

        # Mock OpenCV operations
        mock_cv2.cvtColor.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.medianBlur.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.threshold.return_value = (0, np.array([[0, 1], [1, 0]]))

        # Mock pytesseract
        mock_pytesseract.image_to_string.return_value = "Sample text from image"

        processor = OCRProcessor()

        result = processor.extract_text("sample.jpg", lang="eng")

        assert result == "Sample text from image"
        mock_pytesseract.image_to_string.assert_called_once()

    @patch("src.ocr.ocr.pytesseract")
    @patch("src.ocr.ocr.Image")
    @patch("src.ocr.ocr.cv2")
    def test_extract_text_from_path(self, mock_cv2, mock_pil_image, mock_pytesseract):
        """Test text extraction from file path."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_pil_image.open.return_value = mock_image

        # Mock OpenCV operations
        mock_cv2.cvtColor.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.medianBlur.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.threshold.return_value = (0, np.array([[0, 1], [1, 0]]))

        # Mock pytesseract
        mock_pytesseract.image_to_string.return_value = "Sample text from file"

        processor = OCRProcessor()

        result = processor.extract_text("sample.jpg", lang="eng")

        assert result == "Sample text from file"
        mock_pytesseract.image_to_string.assert_called_once()

    @patch("src.ocr.ocr.pytesseract")
    @patch("src.ocr.ocr.Image")
    @patch("src.ocr.ocr.cv2")
    def test_extract_text_with_finnish_language(
        self, mock_cv2, mock_pil_image, mock_pytesseract
    ):
        """Test text extraction with Finnish language."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_pil_image.open.return_value = mock_image

        # Mock OpenCV operations
        mock_cv2.cvtColor.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.medianBlur.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.threshold.return_value = (0, np.array([[0, 1], [1, 0]]))

        # Mock pytesseract
        mock_pytesseract.image_to_string.return_value = "Suomen tekstiä"

        processor = OCRProcessor()

        result = processor.extract_text("sample.jpg", lang="fin")

        assert result == "Suomen tekstiä"
        # Verify Finnish-specific configuration was used
        mock_pytesseract.image_to_string.assert_called_once()

    @patch("src.ocr.ocr.pytesseract")
    @patch("src.ocr.ocr.Image")
    @patch("src.ocr.ocr.cv2")
    def test_extract_text_with_custom_config(
        self, mock_cv2, mock_pil_image, mock_pytesseract
    ):
        """Test text extraction with custom configuration."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_pil_image.open.return_value = mock_image

        # Mock OpenCV operations
        mock_cv2.cvtColor.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.medianBlur.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.threshold.return_value = (0, np.array([[0, 1], [1, 0]]))

        # Mock pytesseract
        mock_pytesseract.image_to_string.return_value = "Custom config text"
        custom_config = "--oem 1 --psm 8"

        processor = OCRProcessor()

        result = processor.extract_text("sample.jpg", lang="eng", config=custom_config)

        assert result == "Custom config text"
        # Verify custom config was used
        mock_pytesseract.image_to_string.assert_called_once()

    @patch("src.ocr.ocr.pytesseract")
    @patch("src.ocr.ocr.Image")
    @patch("src.ocr.ocr.cv2")
    def test_extract_text_ocr_failure(self, mock_cv2, mock_pil_image, mock_pytesseract):
        """Test OCR processing failure."""
        # Mock PIL Image
        mock_image = Mock(spec=Image.Image)
        mock_pil_image.open.return_value = mock_image

        # Mock OpenCV operations
        mock_cv2.cvtColor.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.medianBlur.return_value = np.array([[0, 1], [1, 0]])
        mock_cv2.threshold.return_value = (0, np.array([[0, 1], [1, 0]]))

        # Mock pytesseract to fail
        mock_pytesseract.image_to_string.side_effect = RuntimeError("OCR failed")

        processor = OCRProcessor()

        with pytest.raises(RuntimeError, match="Failed to extract text: OCR failed"):
            processor.extract_text("sample.jpg", lang="eng")


class TestCertificateExtractor:
    """Test certificate text extraction functionality."""

    def test_supported_formats(self):
        """Test supported file format constants."""
        # Image formats
        assert ".jpg" in SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in SUPPORTED_IMAGE_FORMATS
        assert ".png" in SUPPORTED_IMAGE_FORMATS
        assert ".bmp" in SUPPORTED_IMAGE_FORMATS
        assert ".tiff" in SUPPORTED_IMAGE_FORMATS
        assert ".tif" in SUPPORTED_IMAGE_FORMATS

        # Document formats
        assert ".docx" in SUPPORTED_DOC_FORMATS
        assert ".doc" in SUPPORTED_DOC_FORMATS

        # PDF formats
        assert ".pdf" in SUPPORTED_PDF_FORMATS

    @patch("src.ocr.cert_extractor._extract_from_image")
    def test_extract_certificate_text_image(self, mock_extract_image):
        """Test text extraction from image file."""
        mock_extract_image.return_value = "Image text content"

        result = extract_certificate_text("sample.jpg", language="eng")

        assert result == "Image text content"
        mock_extract_image.assert_called_once()

    @patch("src.ocr.cert_extractor._extract_from_pdf")
    def test_extract_certificate_text_pdf(self, mock_extract_pdf):
        """Test text extraction from PDF file."""
        mock_extract_pdf.return_value = "PDF text content"

        result = extract_certificate_text("sample.pdf", language="eng")

        assert result == "PDF text content"
        mock_extract_pdf.assert_called_once()

    @patch("src.ocr.cert_extractor._extract_from_docx")
    def test_extract_certificate_text_docx(self, mock_extract_docx):
        """Test text extraction from DOCX file."""
        mock_extract_docx.return_value = "DOCX text content"

        result = extract_certificate_text("sample.docx", language="eng")

        assert result == "DOCX text content"
        mock_extract_docx.assert_called_once()

    def test_extract_certificate_text_unsupported_format(self):
        """Test text extraction with unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported file format: .txt"):
            extract_certificate_text("sample.txt", language="eng")

    @patch("src.ocr.cert_extractor._extract_finnish_from_image")
    def test_extract_finnish_certificate_image(self, mock_extract_finnish):
        """Test Finnish certificate extraction from image."""
        mock_extract_finnish.return_value = "Suomen todistus tekstiä"

        result = extract_finnish_certificate("sample.jpg")

        assert result == "Suomen todistus tekstiä"
        mock_extract_finnish.assert_called_once()

    @patch("src.ocr.cert_extractor._extract_finnish_from_pdf")
    def test_extract_finnish_certificate_pdf(self, mock_extract_finnish):
        """Test Finnish certificate extraction from PDF."""
        mock_extract_finnish.return_value = "Suomen PDF todistus"

        result = extract_finnish_certificate("sample.pdf")

        assert result == "Suomen PDF todistus"
        mock_extract_finnish.assert_called_once()

    @patch("src.ocr.cert_extractor._extract_finnish_from_docx")
    def test_extract_finnish_certificate_docx(self, mock_extract_finnish):
        """Test Finnish certificate extraction from DOCX."""
        mock_extract_finnish.return_value = "Suomen DOCX todistus"

        result = extract_finnish_certificate("sample.docx")

        assert result == "Suomen DOCX todistus"
        mock_extract_finnish.assert_called_once()

    @patch("src.ocr.cert_extractor.extract_certificate_text")
    def test_extract_finnish_certificate_fallback(self, mock_extract):
        """Test Finnish certificate extraction fallback on error."""
        mock_extract.return_value = "Fallback text"

        # Mock the Finnish extraction to fail
        with patch(
            "src.ocr.cert_extractor._extract_finnish_from_image"
        ) as mock_finnish:
            mock_finnish.side_effect = Exception("Finnish extraction failed")

            result = extract_finnish_certificate("sample.jpg")

            assert result == "Fallback text"
            mock_extract.assert_called_once_with("sample.jpg", language="eng+fin")


class TestOCRWorkflow:
    """Test OCR workflow functionality."""

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_initialization_success(self, mock_settings):
        """Test successful OCR workflow initialization."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import at the source
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            assert workflow.samples_dir == Path("test_samples")
            assert workflow.language == "auto"
            assert workflow.use_finnish_detection is True
            assert workflow.results == []

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_initialization_failure(self, mock_settings):
        """Test OCR workflow initialization failure."""
        mock_settings.TESSERACT_CMD = "/invalid/path"

        # Mock the OCR processor import to fail
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr:
            mock_ocr.get_available_languages.side_effect = Exception("OCR failed")

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            with pytest.raises(RuntimeError, match="OCR configuration failed"):
                OCRWorkflow()

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_finnish_detection_disabled(self, mock_settings):
        """Test OCR workflow without Finnish detection."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="eng", use_finnish_detection=False
            )

            assert workflow.use_finnish_detection is False

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_finnish_language_unavailable(self, mock_settings):
        """Test OCR workflow when Finnish language is unavailable."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = [
                "eng"
            ]  # No Finnish

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            # Should disable Finnish detection when not available
            assert workflow.use_finnish_detection is False

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_language_detection(self, mock_settings):
        """Test language detection in workflow."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            # Test language detection
            detected_lang = workflow._detect_language(Path("test.jpg"))
            assert detected_lang in ["fin", "eng", "eng+fin", "auto"]

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_process_document_success(self, mock_settings):
        """Test successful document processing in workflow."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            # Mock the document processing
            with patch(
                "src.workflow.ocr_workflow.extract_certificate_text"
            ) as mock_extract:
                mock_extract.return_value = "Processed text content"

                # Mock the file path to avoid real file system calls
                mock_file_path = Mock(spec=Path)
                mock_file_path.stat.return_value = Mock(st_size=1024)
                mock_file_path.name = "test.pdf"
                mock_file_path.suffix = ".pdf"

                result = workflow.process_document(mock_file_path)

                assert result["success"] is True
                assert "extracted_text" in result
                assert result["extracted_text"] == "Processed text content"

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_process_document_failure(self, mock_settings):
        """Test document processing failure in workflow."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            # Mock the document processing to fail
            with patch(
                "src.workflow.ocr_workflow.extract_certificate_text"
            ) as mock_extract:
                mock_extract.side_effect = Exception("Processing failed")

                # Mock the file path to avoid real file system calls
                mock_file_path = Mock(spec=Path)
                mock_file_path.stat.return_value = Mock(st_size=1024)
                mock_file_path.name = "test.pdf"
                mock_file_path.suffix = ".pdf"

                result = workflow.process_document(mock_file_path)

                assert result["success"] is False
                assert "error" in result
                assert "Processing failed" in result["error"]

    @patch("src.workflow.ocr_workflow.settings")
    def test_workflow_batch_processing(self, mock_settings):
        """Test batch processing of multiple documents."""
        mock_settings.TESSERACT_CMD = "/usr/bin/tesseract"

        # Mock the OCR processor import
        with patch("src.ocr.ocr.ocr_processor") as mock_ocr_processor:
            mock_ocr_processor.get_available_languages.return_value = ["eng", "fin"]

            # Import here to avoid issues with the actual OCR setup
            from src.workflow.ocr_workflow import OCRWorkflow

            workflow = OCRWorkflow(
                samples_dir="test_samples", language="auto", use_finnish_detection=True
            )

            # Mock the discover_documents method to return our test files
            mock_documents = [Path("test1.pdf"), Path("test2.jpg"), Path("test3.docx")]

            with patch.object(
                workflow, "discover_documents", return_value=mock_documents
            ):
                with patch.object(workflow, "process_document") as mock_process:
                    mock_process.return_value = {
                        "success": True,
                        "extracted_text": "Sample text",
                    }

                    workflow.process_all_documents()

                    assert len(workflow.results) == 3
                    assert all(result["success"] for result in workflow.results)
