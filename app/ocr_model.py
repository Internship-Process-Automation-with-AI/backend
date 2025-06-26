import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytesseract
from google.cloud import vision
from google.cloud.vision_v1 import types

from app.config import settings
from app.utils.docx_processor import DOCXProcessor
from app.utils.image_preprocessing import ImagePreprocessor
from app.utils.pdf_converter import PDFConverter

logger = logging.getLogger(__name__)


class OCRResult:
    """Container for OCR processing results."""

    def __init__(
        self, text: str, confidence: float, engine: str, processing_time: float
    ):
        self.text = text
        self.confidence = confidence
        self.engine = engine
        self.processing_time = processing_time
        self.success = confidence >= settings.OCR_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "engine": self.engine,
            "processing_time": self.processing_time,
            "success": self.success,
        }


class OCRService:
    """Smart OCR service with PyMuPDF text extraction and OCR fallback."""

    def __init__(self):
        """Initialize OCR service with preprocessing and PDF conversion capabilities."""
        self.preprocessor = ImagePreprocessor()
        self.pdf_converter = PDFConverter()
        self.docx_processor = DOCXProcessor()
        self.google_vision_client = None

        # Configure Tesseract
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

        # Initialize Google Vision client if credentials are available
        self._initialize_google_vision()

        logger.info("Smart OCR Service initialized successfully")

    def _initialize_google_vision(self):
        """Initialize Google Vision API client if credentials are available."""
        try:
            if settings.GOOGLE_CLOUD_CREDENTIALS:
                os.environ[
                    "GOOGLE_APPLICATION_CREDENTIALS"
                ] = settings.GOOGLE_CLOUD_CREDENTIALS
                self.google_vision_client = vision.ImageAnnotatorClient()
                logger.info("Google Vision API client initialized")
            else:
                logger.warning(
                    "Google Cloud credentials not provided - Google Vision API unavailable"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision API: {e}")
            self.google_vision_client = None

    def extract_text_from_file(
        self, file_path: str, use_preprocessing: bool = True
    ) -> OCRResult:
        """
        Extract text from a file (image or PDF) using smart processing.

        Args:
            file_path: Path to the file
            use_preprocessing: Whether to apply image preprocessing

        Returns:
            OCRResult object
        """
        start_time = time.time()

        try:
            # Determine file type and process accordingly
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".pdf":
                return self._process_pdf_file_smart(
                    file_path, use_preprocessing, start_time
                )
            elif file_extension == ".docx":
                return self._process_docx_file(file_path, start_time)
            elif file_extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                return self._process_image_file(
                    file_path, use_preprocessing, start_time
                )
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def extract_text_from_bytes(
        self, file_bytes: bytes, file_extension: str, use_preprocessing: bool = True
    ) -> OCRResult:
        """
        Extract text from file bytes using smart processing.

        Args:
            file_bytes: File content as bytes
            file_extension: File extension (e.g., '.pdf', '.png')
            use_preprocessing: Whether to apply image preprocessing

        Returns:
            OCRResult object
        """
        start_time = time.time()

        try:
            file_extension = file_extension.lower()

            if file_extension == ".pdf":
                return self._process_pdf_bytes_smart(
                    file_bytes, use_preprocessing, start_time
                )
            elif file_extension == ".docx":
                return self._process_docx_bytes(file_bytes, start_time)
            elif file_extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                return self._process_image_bytes(
                    file_bytes, use_preprocessing, start_time
                )
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error processing file bytes: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_pdf_file_smart(
        self, pdf_path: str, use_preprocessing: bool, start_time: float
    ) -> OCRResult:
        """Smart PDF processing: try text extraction first, then OCR if needed."""
        try:
            logger.info(f"Smart processing PDF file: {pdf_path}")

            # Read PDF bytes
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            return self._process_pdf_bytes_smart(
                pdf_bytes, use_preprocessing, start_time
            )

        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_pdf_bytes_smart(
        self, pdf_bytes: bytes, use_preprocessing: bool, start_time: float
    ) -> OCRResult:
        """Smart PDF processing: try text extraction first, then OCR if needed."""
        try:
            logger.info("Smart processing PDF bytes")

            # Step 1: Try direct text extraction with PyMuPDF
            try:
                (
                    extracted_text,
                    quality_score,
                ) = self.pdf_converter.extract_text_with_quality_check(pdf_bytes)

                if quality_score > 30.0 and len(extracted_text.strip()) > 25:
                    logger.info(
                        f"Using PyMuPDF text extraction (quality: {quality_score:.1f})"
                    )
                    processing_time = time.time() - start_time

                    return OCRResult(
                        text=extracted_text,
                        confidence=quality_score,
                        engine="pymupdf",
                        processing_time=processing_time,
                    )
                else:
                    logger.info(
                        f"PyMuPDF text quality too low ({quality_score:.1f}), falling back to OCR"
                    )

            except Exception as e:
                logger.warning(
                    f"PyMuPDF text extraction failed: {e}, falling back to OCR"
                )

            # Step 2: Fallback to OCR processing
            return self._process_pdf_with_ocr(pdf_bytes, use_preprocessing, start_time)

        except Exception as e:
            logger.error(f"Error in smart PDF processing: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_pdf_with_ocr(
        self, pdf_bytes: bytes, use_preprocessing: bool, start_time: float
    ) -> OCRResult:
        """Process PDF using OCR (fallback method)."""
        try:
            logger.info("Processing PDF with OCR")

            # Convert PDF to images
            images = self.pdf_converter.convert_pdf_to_images(pdf_bytes)

            # Extract text from all pages
            all_text = []
            total_confidence = 0.0
            page_count = len(images)

            for i, image in enumerate(images):
                logger.info(f"Processing PDF page {i + 1}/{page_count} with OCR")

                # Extract text from this page
                result = self._extract_text_from_image(image, use_preprocessing)

                # If we got text, include it regardless of confidence
                if result.text.strip():
                    all_text.append(result.text)
                    total_confidence += result.confidence
                else:
                    logger.warning(f"No text extracted from page {i + 1}")

            # Calculate average confidence
            avg_confidence = total_confidence / page_count if page_count > 0 else 0.0

            # Combine all text
            combined_text = "\n\n".join(all_text)

            processing_time = time.time() - start_time

            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine=result.engine if "result" in locals() else "unknown",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_docx_file(self, docx_path: str, start_time: float) -> OCRResult:
        """Process DOCX file and extract text."""
        try:
            logger.info(f"Processing DOCX file: {docx_path}")

            # Extract text from DOCX
            extracted_text, quality_score = self.docx_processor.extract_text_from_file(
                docx_path
            )

            processing_time = time.time() - start_time

            return OCRResult(
                text=extracted_text,
                confidence=quality_score,
                engine="python-docx",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing DOCX file: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_docx_bytes(self, docx_bytes: bytes, start_time: float) -> OCRResult:
        """Process DOCX bytes and extract text."""
        try:
            logger.info("Processing DOCX bytes")

            # Extract text from DOCX bytes
            extracted_text, quality_score = self.docx_processor.extract_text_from_docx(
                docx_bytes
            )

            processing_time = time.time() - start_time

            return OCRResult(
                text=extracted_text,
                confidence=quality_score,
                engine="python-docx",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error processing DOCX bytes: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_image_file(
        self, image_path: str, use_preprocessing: bool, start_time: float
    ) -> OCRResult:
        """Process image file and extract text."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            return self._extract_text_from_image(image, use_preprocessing, start_time)

        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _process_image_bytes(
        self, image_bytes: bytes, use_preprocessing: bool, start_time: float
    ) -> OCRResult:
        """Process image bytes and extract text."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image bytes")

            return self._extract_text_from_image(image, use_preprocessing, start_time)

        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _extract_text_from_image(
        self,
        image: np.ndarray,
        use_preprocessing: bool,
        start_time: Optional[float] = None,
    ) -> OCRResult:
        """
        Extract text from image using Tesseract with Google Vision fallback.
        Smart approach: try raw image first, only preprocess if needed.

        Args:
            image: Image as numpy array
            use_preprocessing: Whether to apply image preprocessing
            start_time: Start time for processing (if None, will be calculated)

        Returns:
            OCRResult object
        """
        if start_time is None:
            start_time = time.time()

        try:
            # First, try with raw image (no preprocessing)
            raw_result = self._extract_text_with_tesseract(image)

            # Check if raw image gives good results
            raw_text_length = len(
                raw_result.text.strip().replace(" ", "").replace("\n", "")
            )
            raw_is_good = raw_text_length > 25 and raw_result.confidence > 40.0

            logger.info(
                f"Raw image result: {raw_text_length} chars, {raw_result.confidence:.1f}% confidence"
            )

            # If raw image is good enough, use it
            if raw_is_good:
                logger.info("Raw image gives good results, skipping preprocessing")
                best_result = raw_result
            else:
                logger.info("Raw image not good enough, trying preprocessing")

                # Try with preprocessing if enabled
                if use_preprocessing and settings.IMAGE_PREPROCESSING_ENABLED:
                    processed_image = self.preprocessor.preprocess_image(image)
                    processed_result = self._extract_text_with_tesseract(
                        processed_image
                    )

                    # Compare raw vs processed results
                    processed_text_length = len(
                        processed_result.text.strip().replace(" ", "").replace("\n", "")
                    )

                    logger.info(
                        f"Processed image result: {processed_text_length} chars, {processed_result.confidence:.1f}% confidence"
                    )

                    # Choose the better result based on quality score
                    def text_quality_score(result):
                        clean_text = (
                            result.text.strip().replace(" ", "").replace("\n", "")
                        )
                        text_length = len(clean_text)

                        # Prefer results with higher confidence
                        confidence_bonus = result.confidence / 100.0

                        # Penalize very low confidence results even if they have more text
                        if result.confidence < 30.0:
                            text_length *= 0.5  # Reduce score for low confidence

                        # Calculate final score
                        return text_length * (1 + confidence_bonus)

                    raw_score = text_quality_score(raw_result)
                    processed_score = text_quality_score(processed_result)

                    logger.info(
                        f"Raw score: {raw_score:.1f}, Processed score: {processed_score:.1f}"
                    )

                    if (
                        processed_score > raw_score * 1.2
                    ):  # Processed must be significantly better
                        best_result = processed_result
                        logger.info("Using processed image result")
                    else:
                        best_result = raw_result
                        logger.info(
                            "Using raw image result (preprocessing didn't help)"
                        )
                else:
                    best_result = raw_result

            # If Tesseract fails or has low confidence, try Google Vision
            if not best_result.success and self.google_vision_client:
                logger.info("Tesseract failed, trying Google Vision API")
                google_result = self._extract_text_with_google_vision(image)

                if google_result.success:
                    return google_result
                else:
                    logger.warning("Both Tesseract and Google Vision failed")
                    return best_result
            else:
                return best_result

        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "error", processing_time)

    def _extract_text_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract OCR with multiple strategies."""
        start_time = time.time()

        try:
            # Try multiple Tesseract configurations for better accuracy
            configs = [
                r"--oem 3 --psm 6",  # Default: Assume uniform block of text
                r"--oem 3 --psm 3",  # Fully automatic page segmentation
                r"--oem 3 --psm 4",  # Assume single column of text
                r"--oem 3 --psm 8",  # Single word
                r"--oem 3 --psm 13",  # Raw line
            ]

            best_result = None
            best_confidence = 0.0

            for i, config in enumerate(configs):
                try:
                    # Extract text with this configuration
                    text = pytesseract.image_to_string(image, config=config)

                    # Get confidence data
                    data = pytesseract.image_to_data(
                        image, config=config, output_type=pytesseract.Output.DICT
                    )

                    # Calculate average confidence
                    confidences = [int(conf) for conf in data["conf"]]
                    avg_confidence = (
                        sum(confidences) / len(confidences) if confidences else 0.0
                    )

                    # If we got meaningful text, use this result
                    if text.strip() and len(text.strip()) > 10:
                        # If we got text but confidence is very low, give it a minimum confidence
                        if avg_confidence < 10.0:
                            avg_confidence = 30.0

                        # Keep the best result
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_result = {
                                "text": text.strip(),
                                "confidence": avg_confidence,
                                "config": config,
                            }

                except Exception as e:
                    logger.debug(f"Tesseract config {i} failed: {e}")
                    continue

            # If no good result found, try with default config
            if not best_result:
                text = pytesseract.image_to_string(image, config=r"--oem 3 --psm 6")
                best_result = {
                    "text": text.strip(),
                    "confidence": 20.0,  # Default confidence for extracted text
                    "config": r"--oem 3 --psm 6",
                }

            processing_time = time.time() - start_time

            return OCRResult(
                text=best_result["text"],
                confidence=best_result["confidence"],
                engine="tesseract",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "tesseract_error", processing_time)

    def _extract_text_with_google_vision(self, image: np.ndarray) -> OCRResult:
        """Extract text using Google Vision API."""
        start_time = time.time()

        try:
            # Convert numpy array to bytes
            success, buffer = cv2.imencode(".png", image)
            if not success:
                raise ValueError("Failed to encode image")

            image_bytes = buffer.tobytes()

            # Create Google Vision image object
            vision_image = types.Image(content=image_bytes)

            # Perform text detection
            response = self.google_vision_client.text_detection(image=vision_image)

            if response.error.message:
                raise Exception(f"Google Vision API error: {response.error.message}")

            # Extract text from response
            texts = response.text_annotations
            if not texts:
                return OCRResult("", 0.0, "google_vision", time.time() - start_time)

            # Get the full text (first annotation contains all text)
            full_text = texts[0].description

            # Calculate confidence based on number of detected text blocks
            confidence = min(100.0, len(texts) * 10)  # Simple confidence calculation

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text.strip(),
                confidence=confidence,
                engine="google_vision",
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            processing_time = time.time() - start_time
            return OCRResult("", 0.0, "google_vision_error", processing_time)

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(settings.ALLOWED_EXTENSIONS)

    def is_google_vision_available(self) -> bool:
        """Check if Google Vision API is available."""
        return self.google_vision_client is not None

    def get_processing_stats(self) -> Dict:
        """Get processing statistics and capabilities."""
        return {
            "supported_formats": self.get_supported_formats(),
            "google_vision_available": self.is_google_vision_available(),
            "preprocessing_enabled": settings.IMAGE_PREPROCESSING_ENABLED,
            "confidence_threshold": settings.OCR_CONFIDENCE_THRESHOLD,
            "processing_engines": [
                "pymupdf",
                "python-docx",
                "tesseract",
                "google_vision",
            ],
        }
