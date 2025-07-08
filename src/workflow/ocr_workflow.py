"""
OCR Workflow for processing internship certificates and documents.

This module provides a complete workflow for:
1. Setting up OCR configuration
2. Processing documents from samples directory
3. Extracting text using cert_extractor with Finnish support
4. Saving results to processedData directory
"""

import json
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
        output_dir: str | Path = "processedData",
        language: str = "auto",
        use_finnish_detection: bool = True,
    ):
        """
        Initialize OCR workflow.

        Args:
            samples_dir: Directory containing input documents
            output_dir: Directory to save processed results
            language: Language for OCR ("auto", "eng", "fin", "eng+fin")
            use_finnish_detection: Whether to auto-detect Finnish documents
        """
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        self.use_finnish_detection = use_finnish_detection
        self.results: list[dict] = []

        # Ensure directories exist
        self._setup_directories()

        # Verify OCR setup
        self._verify_ocr_setup()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for organization
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        logger.info(f"Output directory structure created at: {self.output_dir}")

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
        Detect the appropriate language for a document based on filename and content.

        Args:
            file_path: Path to the document

        Returns:
            Detected language code ("fin", "eng", "eng+fin", or "auto")
        """
        filename_lower = file_path.name.lower()

        # Check for Finnish indicators in filename
        finnish_indicators = [
            "finnish",
            "finn",
            "suomi",
            "työtodistus",
            "todistus",
            "harjoittelu",
            "kesätyö",
            "työ",
        ]

        if any(indicator in filename_lower for indicator in finnish_indicators):
            logger.info(f"🇫🇮 Detected Finnish document from filename: {file_path.name}")
            return "fin"

        # Default to auto-detection
        return self.language

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
            if detected_lang == "fin" or (
                self.use_finnish_detection and "finnish" in file_path.name.lower()
            ):
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

                # Save text to file in document-specific directory
                base_name = file_path.stem
                document_dir = self.output_dir / base_name
                document_dir.mkdir(exist_ok=True)

                output_filename = f"ocr_output_{base_name}.txt"
                output_path = document_dir / output_filename

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                result.update(
                    {
                        "success": True,
                        "text_length": len(extracted_text),
                        "detected_language": detected_lang,
                        "finnish_chars_count": finnish_chars,
                        "output_file": str(output_filename),
                        "extracted_text": extracted_text,
                    },
                )

                logger.info(
                    f"✅ Success: {relative_path} -> {output_filename} ({len(extracted_text)} chars, {finnish_chars} Finnish chars, lang: {detected_lang})",
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

    def _generate_output_filename(self, input_path: Path) -> str:
        """
        Generate output filename for extracted text.

        Args:
            input_path: Original file path

        Returns:
            Generated filename for text output
        """
        # Remove extension and add .txt
        base_name = input_path.stem

        # Handle subdirectories by replacing path separators
        try:
            relative_path = input_path.relative_to(self.samples_dir)
            if relative_path.parent != Path():
                # Include subdirectory in filename
                base_name = (
                    str(relative_path.parent).replace("/", "_").replace("\\", "_")
                    + "_"
                    + base_name
                )
        except ValueError:
            # If file is not in samples directory, use just the filename
            pass

        return f"{base_name}.txt"

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

        # Save detailed report
        self._save_processing_report(summary)

        return summary

    def _generate_summary(self, start_time: float) -> dict:
        """Generate processing summary with language statistics."""
        total_time = time.time() - start_time
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        # Language statistics
        language_stats = {}
        finnish_docs = []
        total_finnish_chars = 0

        for result in successful:
            lang = result.get("detected_language", "unknown")
            language_stats[lang] = language_stats.get(lang, 0) + 1

            finnish_chars = result.get("finnish_chars_count", 0)
            total_finnish_chars += finnish_chars

            if finnish_chars > 0 or lang == "fin":
                finnish_docs.append(
                    {
                        "file": result["file_name"],
                        "finnish_chars": finnish_chars,
                        "language": lang,
                    }
                )

        return {
            "processing_started": datetime.fromtimestamp(start_time).isoformat(),
            "processing_completed": datetime.now().isoformat(),
            "total_processing_time": round(total_time, 2),
            "total_documents": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": round(len(successful) / len(self.results) * 100, 1)
            if self.results
            else 0,
            "total_text_extracted": sum(r["text_length"] for r in successful),
            "total_finnish_characters": total_finnish_chars,
            "finnish_documents_count": len(finnish_docs),
            "language_statistics": language_stats,
            "finnish_documents": finnish_docs,
            "average_processing_time": round(
                sum(r["processing_time"] for r in self.results) / len(self.results),
                2,
            )
            if self.results
            else 0,
            "configuration": {
                "language_mode": self.language,
                "finnish_detection_enabled": self.use_finnish_detection,
            },
            "results": self.results,
        }

    def _save_processing_report(self, summary: dict) -> None:
        """Save detailed processing report with language analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_report_path = (
            self.output_dir / "reports" / f"processing_report_{timestamp}.json"
        )
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save human-readable summary
        summary_path = self.output_dir / "reports" / f"summary_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("OCR PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing completed: {summary['processing_completed']}\n")
            f.write(f"Total documents: {summary['total_documents']}\n")
            f.write(f"Successful: {summary['successful']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Success rate: {summary['success_rate']}%\n")
            f.write(f"Total processing time: {summary['total_processing_time']}s\n")
            f.write(f"Average processing time: {summary['average_processing_time']}s\n")
            f.write(
                f"Total text extracted: {summary['total_text_extracted']} characters\n",
            )

            # Language statistics
            f.write("\nLANGUAGE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Language mode: {summary['configuration']['language_mode']}\n")
            f.write(
                f"Finnish detection: {'Enabled' if summary['configuration']['finnish_detection_enabled'] else 'Disabled'}\n"
            )
            f.write(f"Finnish documents found: {summary['finnish_documents_count']}\n")
            f.write(
                f"Total Finnish characters: {summary['total_finnish_characters']}\n\n"
            )

            if summary["language_statistics"]:
                f.write("Language distribution:\n")
                for lang, count in summary["language_statistics"].items():
                    f.write(f"  {lang}: {count} documents\n")
                f.write("\n")

            # Finnish documents details
            if summary["finnish_documents"]:
                f.write("FINNISH DOCUMENTS:\n")
                f.write("-" * 20 + "\n")
                for doc in summary["finnish_documents"]:
                    f.write(
                        f"🇫🇮 {doc['file']} - {doc['finnish_chars']} Finnish chars (lang: {doc['language']})\n"
                    )
                f.write("\n")

            if summary["failed"] > 0:
                f.write("FAILED DOCUMENTS:\n")
                f.write("-" * 20 + "\n")
                for result in summary["results"]:
                    if not result["success"]:
                        f.write(f"❌ {result['file_path']}: {result['error']}\n")
                f.write("\n")

            f.write("SUCCESSFUL DOCUMENTS:\n")
            f.write("-" * 20 + "\n")
            for result in summary["results"]:
                if result["success"]:
                    lang_info = f" (lang: {result.get('detected_language', 'unknown')})"
                    finnish_info = (
                        f", {result.get('finnish_chars_count', 0)} Finnish chars"
                        if result.get("finnish_chars_count", 0) > 0
                        else ""
                    )
                    f.write(
                        f"✅ {result['file_path']} -> {result['output_file']} ({result['text_length']} chars{finnish_info}{lang_info})\n",
                    )

        logger.info(f"📊 Processing report saved: {json_report_path}")
        logger.info(f"📋 Summary saved: {summary_path}")

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
    output_dir: str = "processedData",
    language: str = "auto",
    use_finnish_detection: bool = True,
) -> dict:
    """
    Run the complete OCR workflow with multi-language support.

    Args:
        samples_dir: Directory containing input documents
        output_dir: Directory to save processed results
        language: Language for OCR ("auto", "eng", "fin", "eng+fin")
        use_finnish_detection: Whether to auto-detect Finnish documents

    Returns:
        Processing summary dictionary
    """
    try:
        workflow = OCRWorkflow(samples_dir, output_dir, language, use_finnish_detection)
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
