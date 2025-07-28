#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File management utilities for handling uploaded documents and processing results.
Now works with database-stored files instead of file system storage.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """Manages file operations for uploaded documents and processing results."""

    def __init__(self, uploads_base_dir: str = "uploads"):
        """
        Initialize file manager.

        Args:
            uploads_base_dir: Base directory for uploads (kept for backward compatibility)
        """
        self.uploads_base = Path(uploads_base_dir)
        # Note: We no longer create uploads directory since files are stored in database
        # But we keep this for any legacy file operations that might still be needed

    def create_temp_file_from_content(
        self, file_content: bytes, file_extension: str
    ) -> Tuple[Path, str]:
        """
        Create a temporary file from database-stored content for processing.

        Args:
            file_content: File content as bytes from database
            file_extension: File extension (e.g., 'pdf', 'docx')

        Returns:
            Tuple of (temp_file_path, temp_filename)
        """
        try:
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = Path(temp_file.name)
                temp_filename = temp_file_path.name

            logger.info(f"Created temporary file: {temp_filename}")
            return temp_file_path, temp_filename

        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            raise

    def cleanup_temp_file(self, temp_file_path: Path) -> bool:
        """
        Clean up a temporary file.

        Args:
            temp_file_path: Path to the temporary file

        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        try:
            if temp_file_path.exists():
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
            return False

    def get_file_info_from_content(self, file_content: bytes, filename: str) -> dict:
        """
        Get information about a file from its content.

        Args:
            file_content: File content as bytes
            filename: Original filename

        Returns:
            Dictionary with file information
        """
        return {
            "filename": filename,
            "size": len(file_content),
            "size_mb": round(len(file_content) / (1024 * 1024), 2),
            "created": datetime.now(),
            "modified": datetime.now(),
        }

    def validate_file_size(self, file_content: bytes, max_size_mb: int = 10) -> bool:
        """
        Validate file size.

        Args:
            file_content: File content as bytes
            max_size_mb: Maximum file size in MB

        Returns:
            bool: True if file size is valid, False otherwise
        """
        file_size_mb = len(file_content) / (1024 * 1024)
        return file_size_mb <= max_size_mb

    def get_supported_extensions(self) -> set:
        """
        Get list of supported file extensions.

        Returns:
            Set of supported file extensions
        """
        return {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".docx",
            ".doc",
        }

    def validate_file_extension(self, filename: str) -> bool:
        """
        Validate file extension.

        Args:
            filename: Original filename

        Returns:
            bool: True if file extension is supported, False otherwise
        """
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.get_supported_extensions()

    def get_media_type(self, file_extension: str) -> str:
        """
        Get media type for a file extension.

        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')

        Returns:
            str: Media type string
        """
        media_types = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        return media_types.get(file_extension.lower(), "application/octet-stream")

    # Legacy methods kept for backward compatibility but marked as deprecated
    def get_date_based_path(self, date: Optional[datetime] = None) -> Path:
        """
        Get the date-based directory path for organizing files.
        DEPRECATED: Files are now stored in database.

        Args:
            date: Date to use (defaults to current date)

        Returns:
            Path to the date-based directory
        """
        logger.warning(
            "get_date_based_path is deprecated - files are now stored in database"
        )
        if date is None:
            date = datetime.now()

        year_dir = str(date.year)
        month_dir = f"{date.month:02d}"
        day_dir = f"{date.day:02d}"

        date_path = self.uploads_base / year_dir / month_dir / day_dir
        date_path.mkdir(parents=True, exist_ok=True)

        return date_path

    def save_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        date: Optional[datetime] = None,
    ) -> Tuple[Path, str]:
        """
        Save an uploaded file to the organized uploads directory.
        DEPRECATED: Files are now stored in database.

        Args:
            file_content: File content as bytes
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Tuple of (file_path, unique_filename)
        """
        logger.warning(
            "save_uploaded_file is deprecated - files are now stored in database"
        )
        # This method is kept for backward compatibility but should not be used
        # Files should be stored directly in the database using create_certificate
        raise NotImplementedError("Files are now stored in database, not file system")

    def save_processing_results(
        self, results: dict, original_filename: str, date: Optional[datetime] = None
    ) -> Tuple[Path, Path]:
        """
        Save processing results (OCR + LLM) alongside the uploaded file.
        DEPRECATED: Results are now stored in database.

        Args:
            results: Complete processing results dictionary
            original_filename: Original filename from user
            date: Date to use for organization (defaults to current date)

        Returns:
            Tuple of (results_path, ocr_text_path)
        """
        logger.warning(
            "save_processing_results is deprecated - results are now stored in database"
        )
        # This method is kept for backward compatibility but should not be used
        # Results should be stored directly in the database
        raise NotImplementedError("Results are now stored in database, not file system")
