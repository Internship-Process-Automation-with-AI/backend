"""
OCR utility functions using Tesseract.
"""

from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """OCR processor using Tesseract with proper configuration."""

    def __init__(self):
        """Initialize OCR processor with Tesseract configuration."""
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        """Configure Tesseract executable path."""
        try:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_executable
            logger.info(f"Tesseract configured at: {settings.tesseract_executable}")
        except RuntimeError as e:
            logger.exception(f"Failed to configure Tesseract: {e}")
            raise

    def extract_text(
        self,
        image: str | Path | Image.Image | np.ndarray,
        lang: str = "eng",
        config: str | None = None,
        preprocess: bool = True,
    ) -> str:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image: Image source (file path, PIL Image, or numpy array)
            lang: Language code for OCR (default: 'eng')
            config: Custom Tesseract configuration string
            preprocess: Whether to preprocess the image for better OCR

        Returns:
            Extracted text as string

        Raises:
            RuntimeError: If OCR processing fails
        """
        try:
            # Convert input to PIL Image
            pil_image = self._prepare_image(image, preprocess)

            # Set default config if not provided
            if config is None:
                config = "--oem 3 --psm 6"

            # Extract text
            text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

            return text.strip()

        except Exception as e:
            logger.exception(f"OCR text extraction failed: {e}")
            msg = f"Failed to extract text: {e}"
            raise RuntimeError(msg) from e

    def extract_data(
        self,
        image: str | Path | Image.Image | np.ndarray,
        lang: str = "eng",
    ) -> dict:
        """
        Extract detailed OCR data including bounding boxes and confidence scores.

        Args:
            image: Image source (file path, PIL Image, or numpy array)
            lang: Language code for OCR (default: 'eng')

        Returns:
            Dictionary containing OCR data with text, coordinates, and confidence
        """
        try:
            pil_image = self._prepare_image(image, preprocess=True)

            return pytesseract.image_to_data(
                pil_image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
            )

        except Exception as e:
            logger.exception(f"OCR data extraction failed: {e}")
            msg = f"Failed to extract OCR data: {e}"
            raise RuntimeError(msg) from e

    def get_available_languages(self) -> list[str]:
        """
        Get list of available languages for OCR.

        Returns:
            List of available language codes
        """
        try:
            return pytesseract.get_languages(config="")
        except Exception as e:
            logger.exception(f"Failed to get available languages: {e}")
            return ["eng"]  # Return default if fails

    def _prepare_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        preprocess: bool = True,
    ) -> Image.Image:
        """
        Prepare image for OCR processing.

        Args:
            image: Input image in various formats
            preprocess: Whether to apply preprocessing

        Returns:
            PIL Image ready for OCR
        """
        # Convert to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert OpenCV image (BGR) to PIL (RGB)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            pil_image = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            msg = f"Unsupported image type: {type(image)}"
            raise ValueError(msg)

        if preprocess:
            pil_image = self._preprocess_image(pil_image)

        return pil_image

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing to improve OCR accuracy.

        Args:
            image: PIL Image to preprocess

        Returns:
            Preprocessed PIL Image
        """
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)

        # Convert to grayscale if colored
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply preprocessing techniques
        # 1. Noise removal
        denoised = cv2.medianBlur(gray, 5)

        # 2. Thresholding to get binary image
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL Image
        return Image.fromarray(thresh)


# Global OCR processor instance
ocr_processor = OCRProcessor()
