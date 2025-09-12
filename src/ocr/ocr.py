"""
OCR utility functions using Tesseract with multi-language support.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Language-specific configurations
LANGUAGE_CONFIGS = {
    "eng": {
        "config": "--oem 3 --psm 6",
        "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/() ",
        "name": "English",
    },
    "fin": {
        "config": "--oem 3 --psm 6",
        "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÅäöå0123456789.,:-/() ",
        "name": "Finnish",
    },
    "eng+fin": {
        "config": "--oem 3 --psm 6",
        "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÅäöå0123456789.,:-/() ",
        "name": "English + Finnish",
    },
}

# Finnish-specific words for language detection
FINNISH_KEYWORDS = {
    "työtodistus",
    "todistus",
    "työntekijä",
    "työnantaja",
    "päivämäärä",
    "nimi",
    "syntynyt",
    "ammatti",
    "tehtävä",
    "osasto",
    "palvelussuhde",
    "harjoittelu",
    "koulutus",
    "oppilaitos",
    "yliopisto",
    "koulu",
}


class OCRProcessor:
    """OCR processor using Tesseract with multi-language configuration."""

    def __init__(self):
        """Initialize OCR processor with Tesseract configuration."""
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        """Configure Tesseract executable path."""
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                settings.TESSERACT_CMD or "tesseract"
            )
            logger.info(
                f"Tesseract configured at: {settings.TESSERACT_CMD or 'tesseract'}"
            )
        except RuntimeError as e:
            logger.exception(f"Failed to configure Tesseract: {e}")
            raise

    def extract_text(
        self,
        image: str | Path | Image.Image | np.ndarray,
        lang: str = "eng",
        config: str | None = None,
        preprocess: bool = True,
        auto_language: bool = False,
    ) -> str:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image: Image source (file path, PIL Image, or numpy array)
            lang: Language code for OCR (default: 'eng', options: 'fin', 'eng+fin')
            config: Custom Tesseract configuration string
            preprocess: Whether to preprocess the image for better OCR
            auto_language: Whether to automatically detect and use best language

        Returns:
            Extracted text as string

        Raises:
            RuntimeError: If OCR processing fails
        """
        try:
            # Convert input to PIL Image
            pil_image = self._prepare_image(image, preprocess)

            # Auto-detect language if requested
            if auto_language:
                lang = self._detect_best_language(pil_image)
                logger.info(f"Auto-detected language: {lang}")

            # Get language-specific configuration
            if config is None:
                config = self._get_language_config(lang)

            # Extract text
            text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

            return text.strip()

        except Exception as e:
            logger.exception(f"OCR text extraction failed: {e}")
            msg = f"Failed to extract text: {e}"
            raise RuntimeError(msg) from e

    def extract_text_finnish(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        config: str = None,
        preprocess: bool = False,  # Changed default to False for Finnish
    ) -> str:
        """
        Extract text optimized for Finnish documents.

        Args:
            image: Input image in various formats
            config: Custom Tesseract configuration
            preprocess: Whether to apply image preprocessing (default: False for Finnish)

        Returns:
            Extracted text string
        """
        try:
            # Use Finnish-specific configuration if none provided
            if config is None:
                config = self._get_finnish_config()  # Use optimized Finnish config

            # For Finnish documents, use minimal preprocessing to preserve ä, ö, å characters
            if preprocess:
                # Only apply very light preprocessing that won't damage Finnish characters
                processed_image = self._prepare_image(image, enhance_for_finnish=True)
            else:
                # Use image directly with minimal format normalization only
                if isinstance(image, (str, Path)):
                    processed_image = Image.open(image)
                elif isinstance(image, np.ndarray):
                    processed_image = Image.fromarray(image)
                elif isinstance(image, Image.Image):
                    processed_image = image
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")

                # Only normalize format, no heavy preprocessing
                processed_image = self._normalize_image_format(processed_image)

            # Extract with Finnish language settings and optimized config
            text = pytesseract.image_to_string(
                processed_image, lang="fin", config=config
            )

            return text.strip()

        except Exception as e:
            logger.error(f"Finnish text extraction failed: {e}")
            raise

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
            langs = pytesseract.get_languages(config="")
            logger.info(f"Available Tesseract languages: {langs}")
            return langs
        except Exception as e:
            logger.exception(f"Failed to get available languages: {e}")
            return ["eng"]  # Return default if fails

    def _detect_best_language(self, image: Image.Image) -> str:
        """
        Detect the best language for OCR by trying different options.

        Args:
            image: PIL Image to analyze

        Returns:
            Best language code detected
        """
        try:
            # First try with Finnish to detect Finnish documents
            fin_text = pytesseract.image_to_string(
                image, lang="fin", config="--oem 3 --psm 6"
            )

            # Check for Finnish characters first (most reliable indicator)
            finnish_chars = set("äöå")
            has_finnish_chars = any(char in fin_text.lower() for char in finnish_chars)

            # Check for Finnish keywords
            text_lower = fin_text.lower()
            finnish_score = sum(
                1 for keyword in FINNISH_KEYWORDS if keyword in text_lower
            )

            # If we find Finnish indicators, use Finnish
            if has_finnish_chars or finnish_score > 0:
                logger.info(
                    f"Finnish indicators found: chars={has_finnish_chars}, keywords={finnish_score}"
                )
                return "fin"  # Use Finnish for best results

            # If no Finnish detected, try English
            eng_text = pytesseract.image_to_string(
                image, lang="eng", config="--oem 3 --psm 6"
            )

            # Check English text for Finnish indicators as well
            eng_text_lower = eng_text.lower()
            eng_finnish_score = sum(
                1 for keyword in FINNISH_KEYWORDS if keyword in eng_text_lower
            )
            eng_has_finnish_chars = any(
                char in eng_text_lower for char in finnish_chars
            )

            if eng_finnish_score > 0 or eng_has_finnish_chars:
                logger.info(
                    f"Finnish indicators found in English text: keywords={eng_finnish_score}, chars={eng_has_finnish_chars}"
                )
                return "fin"  # Use Finnish even if detected via English

            return "eng"

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "eng"  # Fallback to English

    def _get_language_config(self, lang: str) -> str:
        """
        Get language-specific Tesseract configuration.

        Args:
            lang: Language code

        Returns:
            Tesseract configuration string
        """
        lang_config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["eng"])

        # Build config with character whitelist
        config = lang_config["config"]
        if "whitelist" in lang_config:
            config += f" -c tessedit_char_whitelist={lang_config['whitelist']}"

        return config

    def _get_finnish_config(self) -> str:
        """
        Get optimized configuration for Finnish text recognition.

        Returns:
            Finnish-optimized Tesseract configuration
        """
        return (
            "--oem 3 --psm 6 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÅäöå0123456789.,:-/()%& "
            "-c load_system_dawg=false "
            "-c load_freq_dawg=false "
            "-c preserve_interword_spaces=1 "
            "-c textord_heavy_nr=1 "
            "-c textord_min_linesize=2.5"
        )

    def _prepare_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        preprocess: bool = True,
        enhance_for_finnish: bool = False,
    ) -> Image.Image:
        """
        Prepare image for OCR processing.

        Args:
            image: Input image in various formats
            preprocess: Whether to apply preprocessing
            enhance_for_finnish: Whether to apply Finnish-specific enhancements

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

        # Always normalize format for pytesseract compatibility
        pil_image = self._normalize_image_format(pil_image)

        if preprocess:
            pil_image = self._preprocess_image(
                pil_image, enhance_for_finnish=enhance_for_finnish
            )

        return pil_image

    def _preprocess_image(
        self, image: Image.Image, enhance_for_finnish: bool = False
    ) -> Image.Image:
        """
        Apply preprocessing to improve OCR accuracy.

        Args:
            image: PIL Image to preprocess
            enhance_for_finnish: Whether to apply Finnish-specific enhancements

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
        if enhance_for_finnish:
            # Very light preprocessing for Finnish documents to preserve ä, ö, å characters
            # 1. Very light contrast enhancement (minimal to avoid character damage)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # 2. Very light noise removal that preserves character details
            denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)

            # 3. Simple thresholding instead of adaptive to preserve character shapes
            _, thresh = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            # Standard preprocessing
            # 1. Noise removal
            denoised = cv2.medianBlur(gray, 5)

            # 2. Thresholding to get binary image
            _, thresh = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Convert back to PIL Image
        return Image.fromarray(thresh)

    def _preprocess_for_word_spacing(self, image: Image.Image) -> Image.Image:
        """
        Specialized preprocessing for better word spacing detection in scanned documents.

        Args:
            image: Input PIL Image

        Returns:
            Processed PIL Image with better word separation
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Apply adaptive threshold for better character separation
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to improve word separation
            # Horizontal kernel to separate words better
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

            # Vertical kernel to connect broken characters
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

            # Apply morphological opening to separate touching characters
            opened = cv2.morphologyEx(
                adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel
            )

            # Apply morphological closing to connect broken parts of characters
            processed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, vertical_kernel)

            # Apply slight dilation to make characters more readable
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            processed = cv2.dilate(processed, kernel, iterations=1)

            return Image.fromarray(processed)

        except Exception as e:
            logger.warning(f"Word spacing preprocessing failed: {e}")
            return self._prepare_image(image)  # Fallback to standard preprocessing

    def extract_text_with_spacing(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        lang: str = "eng",
        enhance_word_spacing: bool = True,
    ) -> str:
        """
        Extract text with enhanced word spacing detection for scanned documents.

        Args:
            image: Input image in various formats
            lang: Language code for OCR
            enhance_word_spacing: Whether to apply word spacing enhancement

        Returns:
            Extracted text with improved word spacing
        """
        try:
            # Convert to PIL Image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Apply word spacing preprocessing if requested
            if enhance_word_spacing:
                processed_image = self._preprocess_for_word_spacing(pil_image)
            else:
                processed_image = self._prepare_image(pil_image)

            # Use optimized configuration for word spacing
            spacing_config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

            # Extract text
            text = pytesseract.image_to_string(
                processed_image, lang=lang, config=spacing_config
            )

            # Post-process to fix remaining spacing issues
            if enhance_word_spacing:
                text = self._fix_word_spacing(text)

            return text.strip()

        except Exception as e:
            logger.error(f"Text extraction with spacing failed: {e}")
            # Fallback to regular extraction
            return self.extract_text(image, lang=lang)

    def _fix_word_spacing(self, text: str) -> str:
        """
        Post-process text to fix missing word spaces using heuristics.

        Args:
            text: Raw OCR text

        Returns:
            Text with improved word spacing
        """
        if not text:
            return text

        import re

        # Apply spacing rules
        result = text

        # Basic word boundary patterns
        patterns = [
            # Add space before uppercase letters that follow lowercase
            (r"([a-z])([A-Z])", r"\1 \2"),
            # Add space between letters and numbers
            (r"([a-zA-Z])(\d)", r"\1 \2"),
            (r"(\d)([a-zA-Z])", r"\1 \2"),
            # Add space before common certificate words
            (
                r"([a-z])(Certificate|Experience|Internship|University|Technologies|Ltd|Mr|Ms|Date|This|That|The|And|Has|Was|Is|To|At|In|On|With|From|For)",
                r"\1 \2",
            ),
            # Add space after common small words
            (
                r"(to|is|of|at|in|on|the|and|has|was|for|with|from|that|this)([A-Z][a-z])",
                r"\1 \2",
            ),
            # Specific patterns for certificates
            (r"([a-z])(successfully|completed|employed|designated)", r"\1 \2"),
            (r"(Mr|Ms|Dr)([A-Z])", r"\1 \2"),
            # Date patterns
            (r"(\d{4})(to|from)", r"\1 \2"),
            (r"(to|from)(\d)", r"\1 \2"),
            # Company patterns
            (r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2"),
        ]

        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        # Clean up multiple spaces
        result = re.sub(r"\s+", " ", result)

        # Fix common OCR mistakes
        fixes = {
            "1s": "is",
            "1n": "in",
            "0f": "of",
            "Mr.": "Mr ",
            "Ms.": "Ms ",
            "Dr.": "Dr ",
        }

        for mistake, correction in fixes.items():
            result = result.replace(mistake, correction)

        return result.strip()

    def _normalize_image_format(self, image: Image.Image) -> Image.Image:
        """
        Normalize image format to ensure pytesseract compatibility.

        Args:
            image: PIL Image to normalize

        Returns:
            Normalized PIL Image compatible with pytesseract
        """
        try:
            # Convert to RGB if necessary (pytesseract works best with RGB or L mode)
            if image.mode not in ("RGB", "L"):
                if image.mode == "RGBA":
                    # Create white background for RGBA images
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(
                        image, mask=image.split()[-1]
                    )  # Use alpha channel as mask
                    image = background
                else:
                    image = image.convert("RGB")

            # Ensure image has proper format attribute
            if not hasattr(image, "format") or image.format is None:
                # Create a new image with proper format
                img_copy = Image.new(image.mode, image.size)
                img_copy.paste(image)
                img_copy.format = "PNG"  # Set a safe format
                return img_copy

            return image

        except Exception as e:
            logger.warning(f"Image normalization failed: {e}")
            # Fallback: convert to RGB and set PNG format
            try:
                normalized = image.convert("RGB")
                normalized.format = "PNG"
                return normalized
            except Exception as fallback_e:
                logger.error(f"Image normalization fallback failed: {fallback_e}")
                raise

    def extract_text_raw_finnish(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> str:
        """
        Extract Finnish text with absolutely no preprocessing for maximum character preservation.

        Args:
            image: Input image in various formats

        Returns:
            Extracted Finnish text string
        """
        try:
            # Convert to PIL Image with minimal handling
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Only normalize format, absolutely no preprocessing
            pil_image = self._normalize_image_format(pil_image)

            # Try Finnish configurations with early exit for speed
            configs = [
                # Config 1: Best performing Finnish config (try first)
                "--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=0",
                # Config 2: Fallback with different PSM
                "--oem 3 --psm 8 -c preserve_interword_spaces=1 -c textord_heavy_nr=0",
            ]

            best_text = ""
            best_score = 0

            for i, config in enumerate(configs):
                try:
                    text = pytesseract.image_to_string(
                        pil_image, lang="fin", config=config
                    ).strip()

                    if text:
                        # Score based on Finnish character count and text length
                        finnish_chars = sum(1 for c in text.lower() if c in "äöå")
                        score = finnish_chars * 10 + len(text)

                        if score > best_score:
                            best_score = score
                            best_text = text

                        # EARLY EXIT: If we get good text on first try, don't waste time on other configs
                        if (
                            i == 0 and len(text) > 400
                        ):  # Substantial text found on best config
                            logger.info(
                                f"Early exit - good Finnish text found: {len(text)} chars, score: {score}"
                            )
                            return text

                except Exception as e:
                    logger.debug(f"Config {config} failed: {e}")
                    continue

            if best_text:
                logger.info(f"Best Finnish extraction score: {best_score}")
                return best_text
            else:
                # Quick fallback to basic Finnish extraction
                fallback_config = "--oem 3 --psm 6"
                return pytesseract.image_to_string(
                    pil_image, lang="fin", config=fallback_config
                ).strip()

        except Exception as e:
            logger.error(f"Raw Finnish text extraction failed: {e}")
            raise


# Global OCR processor instance
ocr_processor = OCRProcessor()
