"""
OCR Workflow for processing internship certificates and documents.

This module provides a complete workflow for:
1. Setting up OCR configuration
2. Processing documents from samples directory
3. Extracting text using cert_extractor
4. Saving results to processedData directory
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from config.settings import settings
from src.ocr.cert_extractor import (
    SUPPORTED_DOC_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_PDF_FORMATS,
    extract_certificate_text,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Combine all supported formats
SUPPORTED_FORMATS = (
    SUPPORTED_IMAGE_FORMATS | SUPPORTED_DOC_FORMATS | SUPPORTED_PDF_FORMATS
)


class OCRWorkflow:
    """OCR workflow manager for batch processing of documents."""

    def __init__(
        self,
        samples_dir: str | Path = "samples",
        output_dir: str | Path = "processedData",
    ):
        """
        Initialize OCR workflow.

        Args:
            samples_dir: Directory containing input documents
            output_dir: Directory to save processed results
        """
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.results: list[dict] = []

        # Ensure directories exist
        self._setup_directories()

        # Verify OCR setup
        self._verify_ocr_setup()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for organization
        (self.output_dir / "text_files").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        logger.info(f"Output directory structure created at: {self.output_dir}")

    def _verify_ocr_setup(self) -> None:
        """Verify OCR configuration is working."""
        try:
            tesseract_path = settings.tesseract_executable
            logger.info(f"✅ OCR setup verified. Tesseract at: {tesseract_path}")
        except Exception as e:
            logger.exception(f"❌ OCR setup failed: {e}")
            msg = "OCR configuration failed. Please check Tesseract installation."
            raise RuntimeError(
                msg,
            ) from e

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
        Process a single document and extract text.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        relative_path = file_path.relative_to(self.samples_dir)

        result = {
            "file_path": str(relative_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "processed_at": datetime.now().isoformat(),
            "success": False,
            "text_length": 0,
            "processing_time": 0.0,
            "error": None,
            "output_file": None,
        }

        try:
            logger.info(f"🔍 Processing: {relative_path}")

            # Extract text using cert_extractor
            extracted_text = extract_certificate_text(file_path)

            if extracted_text:
                # Save text to file
                output_filename = self._generate_output_filename(file_path)
                output_path = self.output_dir / "text_files" / output_filename

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                result.update(
                    {
                        "success": True,
                        "text_length": len(extracted_text),
                        "output_file": str(output_filename),
                    },
                )

                logger.info(
                    f"✅ Success: {relative_path} -> {output_filename} ({len(extracted_text)} chars)",
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
        relative_path = input_path.relative_to(self.samples_dir)
        if relative_path.parent != Path():
            # Include subdirectory in filename
            base_name = (
                str(relative_path.parent).replace("/", "_").replace("\\", "_")
                + "_"
                + base_name
            )

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
        """Generate processing summary."""
        total_time = time.time() - start_time
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

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
            "average_processing_time": round(
                sum(r["processing_time"] for r in self.results) / len(self.results),
                2,
            )
            if self.results
            else 0,
            "results": self.results,
        }

    def _save_processing_report(self, summary: dict) -> None:
        """Save detailed processing report."""
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
                f"Total text extracted: {summary['total_text_extracted']} characters\n\n",
            )

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
                    f.write(
                        f"✅ {result['file_path']} -> {result['output_file']} ({result['text_length']} chars)\n",
                    )

        logger.info(f"📊 Processing report saved: {json_report_path}")
        logger.info(f"📋 Summary saved: {summary_path}")

    def print_summary(self) -> None:
        """Print processing summary to console."""
        if not self.results:
            logger.warning("No processing results available")
            return

        [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        if failed:
            for _result in failed:
                pass


def run_ocr_workflow(
    samples_dir: str = "samples",
    output_dir: str = "processedData",
) -> dict:
    """
    Run the complete OCR workflow.

    Args:
        samples_dir: Directory containing input documents
        output_dir: Directory to save processed results

    Returns:
        Processing summary dictionary
    """
    try:
        workflow = OCRWorkflow(samples_dir, output_dir)
        summary = workflow.process_all_documents()
        workflow.print_summary()
        return summary
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    # Run workflow when script is executed directly
    logger.info("🚀 Starting OCR workflow...")
    try:
        summary = run_ocr_workflow()
        logger.info("🎉 OCR workflow completed successfully!")
    except Exception as e:
        logger.exception(f"💥 OCR workflow failed: {e}")
        sys.exit(1)
