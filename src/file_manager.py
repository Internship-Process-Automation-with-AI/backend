#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Manager for OAMK Work Certificate Processor
Handles file storage, organization, and cleanup in the uploads directory.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file storage and organization in the uploads directory."""

    def __init__(self, uploads_base_dir: str = "uploads"):
        """
        Initialize the file manager.

        Args:
            uploads_base_dir: Base directory for uploads (relative to backend)
        """
        # Get the backend directory (parent of src)
        backend_dir = Path(__file__).parent.parent
        self.uploads_base = backend_dir / uploads_base_dir
        self.temp_dir = self.uploads_base / "temp"

        # Create directories if they don't exist
        self.uploads_base.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(
            f"File manager initialized with uploads directory: {self.uploads_base}"
        )

    def get_date_based_path(self, date: Optional[datetime] = None) -> Path:
        """
        Get the date-based directory path for organizing files.

        Args:
            date: Date to use (defaults to current date)

        Returns:
            Path to the date-based directory
        """
        if date is None:
            date = datetime.now()

        year_dir = str(date.year)
        month_dir = f"{date.month:02d}"
        day_dir = f"{date.day:02d}"

        date_path = self.uploads_base / year_dir / month_dir / day_dir
        date_path.mkdir(parents=True, exist_ok=True)

        return date_path

    def generate_unique_filename(
        self, original_filename: str, date: Optional[datetime] = None
    ) -> str:
        """
        Generate a unique filename with timestamp prefix.

        Args:
            original_filename: Original filename from user
            date: Date to use for timestamp (defaults to current date)

        Returns:
            Unique filename with timestamp prefix
        """
        if date is None:
            date = datetime.now()

        # Get file extension
        file_ext = Path(original_filename).suffix

        # Generate timestamp prefix
        timestamp = date.strftime("%Y%m%d_%H%M%S")

        # Create unique filename
        base_name = Path(original_filename).stem
        unique_filename = f"{timestamp}_{base_name}{file_ext}"

        return unique_filename

    def save_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        date: Optional[datetime] = None,
    ) -> Tuple[Path, str]:
        """
        Save an uploaded file to the organized uploads directory.

        Args:
            file_content: File content as bytes
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Tuple of (file_path, unique_filename)
        """
        if date is None:
            date = datetime.now()

        # Create document-specific folder (like processedData structure)
        base_name = Path(original_filename).stem
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        document_folder_name = f"{timestamp}_{base_name}"

        # Create document folder in uploads
        document_folder = self.uploads_base / document_folder_name
        document_folder.mkdir(exist_ok=True)

        # Save original file in document folder
        file_path = document_folder / original_filename
        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(f"Saved uploaded file: {original_filename} -> {file_path}")

        return file_path, original_filename

    def save_processing_results(
        self, results: dict, original_filename: str, date: Optional[datetime] = None
    ) -> Tuple[Path, Path]:
        """
        Save processing results (OCR + LLM) alongside the uploaded file.

        Args:
            results: Complete processing results dictionary
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Tuple of (results_path, ocr_text_path)
        """
        if date is None:
            date = datetime.now()

        # Create document-specific folder (same as save_uploaded_file)
        base_name = Path(original_filename).stem
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        document_folder_name = f"{timestamp}_{base_name}"
        document_folder = self.uploads_base / document_folder_name

        # Ensure the document folder exists
        document_folder.mkdir(parents=True, exist_ok=True)

        # Save complete results JSON
        results_filename = f"aiworkflow_output_{base_name}_{timestamp}.json"
        results_path = document_folder / results_filename

        with open(results_path, "w", encoding="utf-8") as f:
            import json

            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save OCR text separately for easy reading
        ocr_text_filename = f"ocr_output_{base_name}.txt"
        ocr_text_path = document_folder / ocr_text_filename

        # Get OCR text from results
        ocr_text = ""
        if "ocr_results" in results:
            # Try to get text from different possible locations
            ocr_results = results["ocr_results"]
            if "extracted_text" in ocr_results:
                ocr_text = ocr_results["extracted_text"]
            elif "text" in ocr_results:
                ocr_text = ocr_results["text"]

        if ocr_text:
            with open(ocr_text_path, "w", encoding="utf-8") as f:
                f.write(ocr_text)

        logger.info(f"Saved processing results: {results_path}")
        logger.info(f"Saved OCR text: {ocr_text_path}")

        return results_path, ocr_text_path

    def get_document_folder(
        self, original_filename: str, date: Optional[datetime] = None
    ) -> Path:
        """
        Get the folder path where a document and its results are stored.

        Args:
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Path to the document folder
        """
        if date is None:
            date = datetime.now()

        # Create document-specific folder path
        base_name = Path(original_filename).stem
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        document_folder_name = f"{timestamp}_{base_name}"
        document_folder = self.uploads_base / document_folder_name

        return document_folder

    def get_document_files(
        self, original_filename: str, date: Optional[datetime] = None
    ) -> dict:
        """
        Get all files related to a specific document.

        Args:
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Dictionary with paths to all related files
        """
        if date is None:
            date = datetime.now()

        date_path = self.get_date_based_path(date)
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        base_name = Path(original_filename).stem

        # Generate expected filenames
        unique_filename = self.generate_unique_filename(original_filename, date)
        results_base = f"{timestamp}_{base_name}"

        files = {
            "folder": date_path,
            "original": date_path / "original" / unique_filename,
            "results_json": date_path
            / "results"
            / f"{results_base}_complete_results.json",
            "ocr_text": date_path / "results" / f"{results_base}_ocr_text.txt",
        }

        return files

    def create_temp_file(self, file_content: bytes, suffix: str = "") -> Path:
        """
        Create a temporary file for processing.

        Args:
            file_content: File content as bytes
            suffix: File suffix (e.g., '.pdf')

        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=self.temp_dir
        )

        with temp_file:
            temp_file.write(file_content)

        logger.info(f"Created temporary file: {temp_file.name}")
        return Path(temp_file.name)

    def cleanup_temp_file(self, file_path: Path) -> bool:
        """
        Clean up a temporary file.

        Args:
            file_path: Path to temporary file

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")

        return False

    def cleanup_old_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.

        Args:
            max_age_hours: Maximum age of temp files in hours

        Returns:
            Number of files cleaned up
        """
        if not self.temp_dir.exists():
            return 0

        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0

        for temp_file in self.temp_dir.iterdir():
            if temp_file.is_file():
                try:
                    if temp_file.stat().st_mtime < cutoff_time:
                        temp_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup old temp file {temp_file}: {e}")

        return cleaned_count

    def get_file_info(self, file_path: Path) -> dict:
        """
        Get information about a stored file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            return {}

        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "filename": file_path.name,
        }

    def list_uploaded_files(
        self, date: Optional[datetime] = None, limit: int = 100
    ) -> list:
        """
        List uploaded files in date-based directory.

        Args:
            date: Date to list files for (defaults to current date)
            limit: Maximum number of files to return

        Returns:
            List of file information dictionaries
        """
        if date is None:
            date = datetime.now()

        date_path = self.get_date_based_path(date)
        original_dir = date_path / "original"

        if not original_dir.exists():
            return []

        files = []
        for file_path in original_dir.iterdir():
            if file_path.is_file() and len(files) < limit:
                files.append(self.get_file_info(file_path))

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created"], reverse=True)

        return files

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics for the uploads directory.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        total_files = 0

        for root, dirs, files in os.walk(self.uploads_base):
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                    total_files += 1
                except Exception:
                    pass

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "uploads_directory": str(self.uploads_base),
        }


# Create a singleton instance
file_manager = FileManager()
