"""
Application settings and configuration.
"""

import os
import platform
import shutil
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logger import get_logger


class TesseractNotFoundError(RuntimeError):
    """
    Tesseract OCR not found.

    Please install Tesseract or set TESSERACT_CMD environment variable.
    """


class TesseractNotConfiguredError(RuntimeError):
    """Tesseract not configured."""


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = "Backend API"
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")

    # Tesseract OCR Configuration
    tesseract_cmd: str | None = Field(
        default=None,
        description="Path to Tesseract executable",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        """Configure Tesseract OCR path based on the operating system."""
        if self.tesseract_cmd:
            # If explicitly set in environment, use that
            logger.info(f"Tesseract path set from environment: {self.tesseract_cmd}")
            return

        if platform.system() == "Windows":
            # Common Windows installation paths
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
                    os.getenv("USERNAME", ""),
                ),
                # Check if it's in PATH
                "tesseract.exe",
            ]

            for path in possible_paths:
                if path == "tesseract.exe":
                    # Check if tesseract is in PATH
                    if shutil.which("tesseract"):
                        self.tesseract_cmd = "tesseract"
                        logger.info("Tesseract found in system PATH.")
                        break
                elif Path(path).exists():
                    self.tesseract_cmd = path
                    logger.info(f"Tesseract found at: {path}")
                    break
        else:
            # Unix-like systems
            tesseract_path = shutil.which("tesseract")
            if tesseract_path:
                self.tesseract_cmd = tesseract_path
                logger.info(f"Tesseract found at: {tesseract_path}")

        if not self.tesseract_cmd:
            logger.error(
                "Tesseract OCR not found. Please install Tesseract or set TESSERACT_CMD environment variable.",
            )
            raise TesseractNotFoundError

    @property
    def tesseract_executable(self) -> str:
        """Get the Tesseract executable path."""
        if not self.tesseract_cmd:
            raise TesseractNotConfiguredError
        return self.tesseract_cmd


logger = get_logger(__name__)

# Global settings instance
settings = Settings()
