from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "OAMK Internship Certificate Processor"

    # OCR Configuration
    TESSERACT_CMD: Optional[str] = None  # Path to tesseract executable
    GOOGLE_CLOUD_CREDENTIALS: Optional[str] = (
        None  # Path to Google Cloud credentials JSON
    )

    # LLM Configuration
    GEMINI_API_KEY: Optional[str] = None  # Gemini API key for LLM evaluation
    GEMINI_MODEL: str = "gemini-2.0-flash"  # Primary Gemini model to use
    GEMINI_FALLBACK_MODELS: List[str] = [
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
    ]

    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
        ".docx",
    }

    # Processing Configuration
    OCR_CONFIDENCE_THRESHOLD: float = 50.0  # Minimum confidence for OCR results
    IMAGE_PREPROCESSING_ENABLED: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from environment variables


# Global settings instance
settings = Settings()
