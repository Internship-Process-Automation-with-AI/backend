"""
Data models for information extraction results.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional


@dataclass
class ExtractedData:
    """Container for extracted information from work certificates."""

    # Document information
    document_type: Optional[str] = None
    language: Optional[str] = None

    # Employee information
    employee_name: Optional[str] = None
    position: Optional[str] = None
    job_title: Optional[str] = None

    # Employer information
    employer: Optional[str] = None
    company_name: Optional[str] = None

    # Employment dates
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    work_period: Optional[str] = None

    # Additional information
    description: Optional[str] = None
    responsibilities: Optional[str] = None
    document_date: Optional[date] = None

    # Confidence scores for each field
    confidence_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_type": self.document_type,
            "language": self.language,
            "employee_name": self.employee_name,
            "position": self.position,
            "job_title": self.job_title,
            "employer": self.employer,
            "company_name": self.company_name,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "work_period": self.work_period,
            "description": self.description,
            "responsibilities": self.responsibilities,
            "document_date": self.document_date.isoformat()
            if self.document_date
            else None,
            "confidence_scores": self.confidence_scores or {},
        }


@dataclass
class ExtractionResult:
    """Result container for information extraction process."""

    success: bool
    extracted_data: ExtractedData
    overall_confidence: float
    processing_time: float
    engine: str
    errors: Optional[list] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "extracted_data": self.extracted_data.to_dict(),
            "overall_confidence": self.overall_confidence,
            "processing_time": self.processing_time,
            "engine": self.engine,
            "errors": self.errors or [],
        }
