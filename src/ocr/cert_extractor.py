"""
Extract plain text from scanned internship certificates (PDF, DOCX, DOC, images).
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


def extract_certificate_text(file_path: str | Path) -> str:
    """
    Extract plain text from a scanned certificate file (PDF, DOCX, DOC, image).
    Args:
        file_path: Path to the certificate file
    Returns:
        Cleaned plain text extracted from the file
    Raises:
        ValueError: If the file format is unsupported or extraction fails
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    logger.info(f"Extracting text from: {file_path} (type: {ext})")

    if ext in SUPPORTED_IMAGE_FORMATS:
        return _extract_from_image(path)
    if ext in SUPPORTED_PDF_FORMATS:
        return _extract_from_pdf(path)
    if ext in SUPPORTED_DOC_FORMATS:
        return _extract_from_docx(path)
    msg = f"Unsupported file format: {ext}"
    raise ValueError(msg)


def _extract_from_image(image_path: Path) -> str:
    """Extract text from an image file with preprocessing."""
    try:
        image = Image.open(image_path)
        # Convert to grayscale and binarize
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = Image.fromarray(thresh)
        text = ocr_processor.extract_text(processed)
        return _clean_text(text)
    except Exception as e:
        logger.exception(f"Image OCR failed: {e}")
        raise


def _extract_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF by converting each page to an image and running OCR."""
    try:
        images = convert_from_path(str(pdf_path))
        logger.info(f"PDF has {len(images)} pages.")
        text_chunks: list[str] = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}")
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = Image.fromarray(thresh)
            text = ocr_processor.extract_text(processed)
            text_chunks.append(text)
        return _clean_text("\n".join(text_chunks))
    except Exception as e:
        logger.exception(f"PDF OCR failed: {e}")
        raise


def _extract_from_docx(docx_path: Path) -> str:
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
                ocr_text = ocr_processor.extract_text(processed)
                text_chunks.append(ocr_text)
        if text_chunks:
            return _clean_text("\n".join(text_chunks))
        return ""
    except Exception as e:
        logger.exception(f"DOCX OCR failed: {e}")
        raise


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace, normalize newlines
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
