"""
OCR Workflow for processing internship certificates and documents.

This module provides a complete workflow for:
1. Setting up OCR configuration
2. Processing documents from samples directory
3. Extracting text using cert_extractor with Finnish support
4. Saving results to processedData directory
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.ocr.cert_extractor import (
    SUPPORTED_DOC_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_PDF_FORMATS,
    extract_certificate_text,
    extract_finnish_certificate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Combine all supported formats
SUPPORTED_FORMATS = (
    SUPPORTED_IMAGE_FORMATS | SUPPORTED_DOC_FORMATS | SUPPORTED_PDF_FORMATS
)


class OCRWorkflow:
    """OCR workflow manager for batch processing of documents with multi-language support."""

    def __init__(
        self,
        samples_dir: str | Path = "samples",
        language: str = "auto",
        use_finnish_detection: bool = True,
    ):
        """
        Initialize OCR workflow.

        Args:
            samples_dir: Directory containing input documents
            language: Language for OCR ("auto", "eng", "fin", "eng+fin")
            use_finnish_detection: Whether to auto-detect Finnish documents
        """
        self.samples_dir = Path(samples_dir)
        self.language = language
        self.use_finnish_detection = use_finnish_detection
        self.results: list[dict] = []

        # Verify OCR setup
        self._verify_ocr_setup()

    def _verify_ocr_setup(self) -> None:
        """Verify OCR configuration is working."""
        try:
            tesseract_path = settings.TESSERACT_CMD or "tesseract"
            logger.info(f"✅ OCR setup verified. Tesseract at: {tesseract_path}")

            # Check for Finnish language support if using auto-detection
            if self.use_finnish_detection or self.language in ["fin", "eng+fin"]:
                from src.ocr.ocr import ocr_processor

                available_langs = ocr_processor.get_available_languages()
                if "fin" in available_langs:
                    logger.info("🇫🇮 Finnish language support available")
                else:
                    logger.warning(
                        "⚠️  Finnish language not available - will use English only"
                    )
                    self.use_finnish_detection = False

        except Exception as e:
            logger.exception(f"❌ OCR setup failed: {e}")
            msg = "OCR configuration failed. Please check Tesseract installation."
            raise RuntimeError(msg) from e

    def _detect_language(self, file_path: Path) -> str:
        """
        Detect the appropriate language for a document based on content.

        Args:
            file_path: Path to the document

        Returns:
            Detected language code ("fin", "eng", "eng+fin", or "auto")
        """
        if not self.use_finnish_detection:
            return self.language

        try:
            # Do a quick OCR scan to detect language
            from src.ocr.cert_extractor import extract_certificate_text

            # Extract text with English first (faster)
            sample_text = extract_certificate_text(
                file_path, language="eng", enhance_finnish=False
            )

            if sample_text:
                text_lower = sample_text.lower()

                # Count Finnish characters
                finnish_chars = sum(1 for c in text_lower if c in "äöå")

                # Count Finnish keywords
                finnish_keywords = [
                    "työtodistus",
                    "todistus",
                    "yrityksessämme",
                    "välisenä",
                    "tehtävissä",
                ]
                finnish_keyword_count = sum(
                    1 for keyword in finnish_keywords if keyword in text_lower
                )

                # If we find Finnish indicators, use Finnish
                if finnish_chars > 0 or finnish_keyword_count >= 1:
                    logger.info(f"🇫🇮 Detected Finnish content: {file_path.name}")
                    return "fin"
                else:
                    logger.info(f"🇺🇸 Detected English content: {file_path.name}")
                    return "eng"

        except Exception as e:
            logger.warning(f"⚠️  Language detection failed for {file_path.name}: {e}")

        # Fallback to auto-detection
        return "auto"

    def _extract_text_smart(self, file_path: Path) -> tuple[str, str]:
        """
        Smart text extraction with language detection and Finnish optimization.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of (extracted_text, detected_language)
        """
        detected_lang = (
            self._detect_language(file_path)
            if self.use_finnish_detection
            else self.language
        )

        try:
            # Use Finnish-specific extraction for Finnish documents
            if detected_lang == "fin":
                logger.info(
                    f"📄 Using Finnish-specific extraction for: {file_path.name}"
                )
                text = extract_finnish_certificate(file_path)
                return text, "fin"

            # Use regular extraction with language specification
            elif detected_lang == "auto":
                logger.info(f"🔍 Using auto-detection for: {file_path.name}")
                text = extract_certificate_text(file_path, language="auto")
                return text, "auto"

            else:
                logger.info(f"🌐 Using {detected_lang} language for: {file_path.name}")
                text = extract_certificate_text(file_path, language=detected_lang)
                return text, detected_lang

        except Exception as e:
            # Fallback to basic extraction
            logger.warning(
                f"⚠️  Smart extraction failed for {file_path.name}, using fallback: {e}"
            )
            text = extract_certificate_text(file_path)
            return text, "fallback"

    def discover_documents(self) -> list[Path]:
        """
        Discover all supported documents in samples directory.

        Returns:
            List of document paths to process
        """
        if not self.samples_dir.exists():
            logger.warning(f"Samples directory not found: {self.samples_dir}")
            return []

        documents = []

        # Search recursively for supported files
        for file_path in self.samples_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                documents.append(file_path)

        logger.info(f"Discovered {len(documents)} documents to process")
        for doc in documents:
            logger.info(f"  📄 {doc.relative_to(self.samples_dir)}")

        return sorted(documents)

    def process_document(self, file_path: Path) -> dict:
        """
        Process a single document and extract text with smart language handling.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()

        # Handle files that may be outside the samples directory
        try:
            relative_path = file_path.relative_to(self.samples_dir)
        except ValueError:
            # If file is not in samples directory, use the filename as relative path
            relative_path = Path(file_path.name)

        result = {
            "file_path": str(relative_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "text_length": 0,
            "processing_time": 0.0,
            "detected_language": None,
            "finnish_chars_count": 0,
            "confidence": 0.0,  # Add confidence field
            "error": None,
            "output_file": None,
        }

        try:
            logger.info(f"🔍 Processing: {relative_path}")

            # Extract text using smart language detection
            extracted_text, detected_lang = self._extract_text_smart(file_path)

            if extracted_text:
                # Count Finnish characters
                finnish_chars = sum(1 for c in extracted_text.lower() if c in "äöå")

                # Calculate confidence score using OCR data
                confidence_score = self._calculate_confidence_score(
                    file_path, detected_lang
                )

                result.update(
                    {
                        "success": True,
                        "text_length": len(extracted_text),
                        "detected_language": detected_lang,
                        "finnish_chars_count": finnish_chars,
                        "confidence": confidence_score,  # Add confidence to result
                        "extracted_text": extracted_text,
                    },
                )

                logger.info(
                    f"✅ Success: {relative_path} -> ({len(extracted_text)} chars, {finnish_chars} Finnish chars, lang: {detected_lang}, confidence: {confidence_score:.1f}%)",
                )
            else:
                result["error"] = "No text extracted from document"
                logger.warning(f"⚠️  No text extracted from: {relative_path}")

        except Exception as e:
            result["error"] = str(e)
            logger.exception(f"❌ Failed to process {relative_path}: {e}")

        finally:
            result["processing_time"] = round(time.time() - start_time, 2)

        return result

    def _calculate_confidence_score(self, file_path: Path, detected_lang: str) -> float:
        """
        Calculate confidence score based on OCR results and language detection.

        Args:
            file_path: Path to the processed document
            detected_lang: Detected language code

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from OCR engine
        base_confidence = 0.8

        # Language detection bonus
        if detected_lang == "fin" and self.use_finnish_detection:
            base_confidence += 0.1
        elif detected_lang == "eng":
            base_confidence += 0.05

        # File type bonus
        if file_path.suffix.lower() in [".pdf", ".docx"]:
            base_confidence += 0.05

        return min(base_confidence, 1.0)

    def process_all_documents(self) -> dict:
        """
        Process all documents in the samples directory.

        Returns:
            Summary of processing results
        """
        start_time = time.time()
        documents = self.discover_documents()

        if not documents:
            logger.warning("No documents found to process")
            return {"total": 0, "success": 0, "failed": 0, "results": []}

        logger.info(f"🚀 Starting batch processing of {len(documents)} documents")

        # Process each document
        for doc_path in documents:
            result = self.process_document(doc_path)
            self.results.append(result)

        # Generate summary
        summary = self._generate_summary(start_time)

        return summary

    def _generate_summary(self, start_time: float) -> dict:
        """
        Generate processing summary.

        Args:
            start_time: Start time of processing

        Returns:
            Summary dictionary
        """
        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate statistics
        total_files = len(self.results)
        successful_files = len([r for r in self.results if r.get("success", False)])
        failed_files = total_files - successful_files

        # Language statistics
        finnish_files = len(
            [r for r in self.results if r.get("detected_language") == "fin"]
        )
        english_files = len(
            [r for r in self.results if r.get("detected_language") == "eng"]
        )

        # Text statistics
        total_chars = sum(r.get("text_length", 0) for r in self.results)
        avg_chars = total_chars / total_files if total_files > 0 else 0

        summary = {
            "processing_time": round(processing_time, 2),
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "success_rate": round((successful_files / total_files) * 100, 1)
            if total_files > 0
            else 0,
            "finnish_files": finnish_files,
            "english_files": english_files,
            "total_characters": total_chars,
            "average_characters": round(avg_chars, 1),
            "results": self.results,
        }

        return summary

    def print_summary(self) -> None:
        """Print processing summary to console with language statistics."""
        if not self.results:
            logger.warning("No processing results available")
            return

        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        # Language analysis
        finnish_docs = [
            r
            for r in successful
            if r.get("finnish_chars_count", 0) > 0
            or r.get("detected_language") == "fin"
        ]
        total_finnish_chars = sum(r.get("finnish_chars_count", 0) for r in successful)

        print("\n" + "=" * 60)
        print("📊 OCR PROCESSING SUMMARY")
        print("=" * 60)
        print(f"📄 Total documents: {len(self.results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(
            f"📈 Success rate: {round(len(successful) / len(self.results) * 100, 1) if self.results else 0}%"
        )

        if finnish_docs:
            print("\n🇫🇮 FINNISH LANGUAGE ANALYSIS:")
            print(f"   Finnish documents: {len(finnish_docs)}")
            print(f"   Total Finnish characters: {total_finnish_chars}")
            print("   Finnish files:")
            for doc in finnish_docs[:5]:  # Show first 5
                print(
                    f"     • {doc['file_name']} ({doc.get('finnish_chars_count', 0)} chars)"
                )
            if len(finnish_docs) > 5:
                print(f"     ... and {len(finnish_docs) - 5} more")

        if failed:
            print("\n❌ FAILED DOCUMENTS:")
            for result in failed:
                print(f"   • {result['file_path']}: {result['error']}")


def run_ocr_workflow(
    samples_dir: str = "samples",
    language: str = "auto",
    use_finnish_detection: bool = True,
) -> dict:
    """
    Run the complete OCR workflow with multi-language support.

    Args:
        samples_dir: Directory containing input documents
        language: Language for OCR ("auto", "eng", "fin", "eng+fin")
        use_finnish_detection: Whether to auto-detect Finnish documents

    Returns:
        Processing summary dictionary
    """
    try:
        workflow = OCRWorkflow(samples_dir, language, use_finnish_detection)
        summary = workflow.process_all_documents()
        workflow.print_summary()
        return summary
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    # Run workflow when script is executed directly
    logger.info("🚀 Starting enhanced OCR workflow with Finnish support...")
    try:
        summary = run_ocr_workflow()
        logger.info("🎉 OCR workflow completed successfully!")
    except Exception as e:
        logger.exception(f"💥 OCR workflow failed: {e}")
        sys.exit(1)
