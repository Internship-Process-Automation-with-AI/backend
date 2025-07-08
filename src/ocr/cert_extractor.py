"""
Extract plain text from scanned internship certificates (PDF, DOCX, DOC, images).
Supports multi-language processing including Finnish certificates.
"""

from io import BytesIO
from pathlib import Path

import cv2
import docx2txt
import numpy as np
from docx import Document
from pdf2image import convert_from_path
from PIL import Image

from src.ocr.ocr import ocr_processor
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_DOC_FORMATS = {".docx", ".doc"}
SUPPORTED_PDF_FORMATS = {".pdf"}


def extract_certificate_text(
    file_path: str | Path, language: str = "auto", enhance_finnish: bool = False
) -> str:
    """
    Extract plain text from a scanned certificate file (PDF, DOCX, DOC, image).

    Args:
        file_path: Path to the certificate file
        language: Language for OCR ('eng', 'fin', 'eng+fin', 'auto')
        enhance_finnish: Whether to apply Finnish-specific enhancements

    Returns:
        Cleaned plain text extracted from the file

    Raises:
        ValueError: If the file format is unsupported or extraction fails
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    logger.info(
        f"Extracting text from: {file_path} (type: {ext}, language: {language})"
    )

    if ext in SUPPORTED_IMAGE_FORMATS:
        return _extract_from_image(path, language, enhance_finnish)
    if ext in SUPPORTED_PDF_FORMATS:
        return _extract_from_pdf(path, language, enhance_finnish)
    if ext in SUPPORTED_DOC_FORMATS:
        return _extract_from_docx(path, language, enhance_finnish)
    msg = f"Unsupported file format: {ext}"
    raise ValueError(msg)


def extract_finnish_certificate(file_path: str | Path) -> str:
    """
    Specialized function for extracting Finnish certificates with optimized settings.

    Args:
        file_path: Path to the Finnish certificate file

    Returns:
        Cleaned Finnish text extracted from the file
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    logger.info(f"Extracting Finnish certificate text from: {file_path}")

    try:
        if ext in SUPPORTED_IMAGE_FORMATS:
            return _extract_finnish_from_image(path)
        if ext in SUPPORTED_PDF_FORMATS:
            return _extract_finnish_from_pdf(path)
        if ext in SUPPORTED_DOC_FORMATS:
            return _extract_finnish_from_docx(path)
        msg = f"Unsupported file format: {ext}"
        raise ValueError(msg)
    except Exception as e:
        logger.error(f"Finnish extraction failed for {file_path}: {e}")
        # Fallback to regular extraction
        logger.info("Falling back to regular extraction")
        return extract_certificate_text(file_path, language="eng+fin")


def _extract_from_image(
    image_path: Path, language: str = "auto", enhance_finnish: bool = False
) -> str:
    """Extract text from an image file with enhanced word spacing detection."""
    try:
        image = Image.open(image_path)

        # For Finnish language, use minimal preprocessing to preserve character shapes
        if language == "fin" or enhance_finnish:
            text = ocr_processor.extract_text_finnish(image, preprocess=False)
        else:
            # For scanned documents and images, use enhanced word spacing detection
            logger.info(f"Using enhanced word spacing detection for: {image_path.name}")
            text = ocr_processor.extract_text_with_spacing(
                image,
                lang=language if language != "auto" else "eng",
                enhance_word_spacing=True,
            )

            # If word spacing detection fails or gives poor results, try fallback
            if not text or len(text.strip()) < 10:
                logger.warning(
                    "Word spacing detection gave poor results, trying fallback"
                )
                # Convert to grayscale and binarize as fallback
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                _, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                processed = Image.fromarray(thresh)

                # Use appropriate language extraction
                if language == "auto":
                    text = ocr_processor.extract_text(processed, auto_language=True)
                else:
                    text = ocr_processor.extract_text(processed, lang=language)

        return _clean_text(text)
    except Exception as e:
        logger.exception(f"Image OCR failed: {e}")
        raise


def _extract_from_pdf(
    pdf_path: Path, language: str = "auto", enhance_finnish: bool = False
) -> str:
    """Extract text from a PDF by converting each page to an image and running OCR."""
    try:
        images = convert_from_path(str(pdf_path))
        logger.info(f"PDF has {len(images)} pages.")
        text_chunks: list[str] = []

        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}")

            # For Finnish language, use minimal preprocessing
            if language == "fin" or enhance_finnish:
                text = ocr_processor.extract_text_finnish(image, preprocess=False)
            else:
                # For scanned PDFs, use enhanced word spacing detection
                text = ocr_processor.extract_text_with_spacing(
                    image,
                    lang=language if language != "auto" else "eng",
                    enhance_word_spacing=True,
                )

                # If word spacing detection fails, fallback to standard processing
                if not text or len(text.strip()) < 10:
                    logger.warning(
                        f"Page {i + 1}: Word spacing detection gave poor results, trying fallback"
                    )
                    # Standard preprocessing for other languages
                    img_array = np.array(image)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    processed = Image.fromarray(thresh)

                    # Use appropriate language extraction
                    if language == "auto":
                        text = ocr_processor.extract_text(processed, auto_language=True)
                    else:
                        text = ocr_processor.extract_text(processed, lang=language)

            text_chunks.append(text)

        return _clean_text("\n".join(text_chunks))
    except Exception as e:
        logger.exception(f"PDF OCR failed: {e}")
        raise


def _extract_from_docx(
    docx_path: Path, language: str = "auto", enhance_finnish: bool = False
) -> str:
    """Extract text from DOCX/DOC. If images are present, OCR them."""
    try:
        # Try extracting text directly
        text = docx2txt.process(str(docx_path))
        if text.strip():
            return _clean_text(text)
        # If no text, try OCR on embedded images
        doc = Document(str(docx_path))
        text_chunks: list[str] = []
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_data = rel.target_part.blob
                image = Image.open(BytesIO(img_data))
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                processed = Image.fromarray(thresh)

                # Use appropriate language extraction
                if language == "auto":
                    ocr_text = ocr_processor.extract_text(processed, auto_language=True)
                elif language == "fin" or enhance_finnish:
                    ocr_text = ocr_processor.extract_text_finnish(processed)
                else:
                    ocr_text = ocr_processor.extract_text(processed, lang=language)

                text_chunks.append(ocr_text)
        if text_chunks:
            return _clean_text("\n".join(text_chunks))
        return ""
    except Exception as e:
        logger.exception(f"DOCX OCR failed: {e}")
        raise


def _extract_finnish_from_image(image_path: Path) -> str:
    """Extract Finnish text from an image with specialized processing."""
    try:
        image = Image.open(image_path)

        # For Finnish text, use minimal preprocessing to preserve character shapes
        # Heavy preprocessing can damage ä, ö, å recognition
        text = ocr_processor.extract_text_finnish(image, preprocess=False)
        return _clean_text(text)
    except Exception as e:
        logger.exception(f"Finnish image OCR failed: {e}")
        raise


def _extract_finnish_from_pdf(pdf_path: Path) -> str:
    """Extract Finnish text from a PDF with optimized processing."""
    try:
        images = convert_from_path(str(pdf_path))
        logger.info(f"Processing Finnish PDF with {len(images)} pages")
        text_chunks: list[str] = []

        for i, image in enumerate(images):
            logger.info(f"Processing Finnish page {i + 1}")

            # For Finnish text, use minimal preprocessing to preserve character shapes
            # The heavy OpenCV preprocessing actually hurts Finnish character recognition
            text = ocr_processor.extract_text_finnish(image, preprocess=False)
            text_chunks.append(text)

        return _clean_text("\n".join(text_chunks))
    except Exception as e:
        logger.exception(f"Finnish PDF OCR failed: {e}")
        raise


def _extract_finnish_from_docx(docx_path: Path) -> str:
    """Extract Finnish text from DOCX with optimized processing."""
    try:
        # Try extracting text directly first
        text = docx2txt.process(str(docx_path))
        if text.strip():
            return _clean_text(text)

        # If no text, try OCR on embedded images with Finnish optimization
        doc = Document(str(docx_path))
        text_chunks: list[str] = []
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_data = rel.target_part.blob
                image = Image.open(BytesIO(img_data))
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Apply Finnish-optimized preprocessing
                processed_image = ocr_processor._preprocess_image(
                    Image.fromarray(gray), enhance_for_finnish=True
                )

                ocr_text = ocr_processor.extract_text_finnish(processed_image)
                text_chunks.append(ocr_text)

        if text_chunks:
            return _clean_text("\n".join(text_chunks))
        return ""
    except Exception as e:
        logger.exception(f"Finnish DOCX OCR failed: {e}")
        raise


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace, normalize newlines
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
