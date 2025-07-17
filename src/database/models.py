"""
Data models for OAMK Work Certificate Processor database.

This module defines the data classes and enums for students, certificates, and decisions
using raw SQL operations instead of SQLAlchemy ORM.
"""

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID


class TrainingType(str, enum.Enum):
    """Enumeration for training types."""

    GENERAL = "GENERAL"
    PROFESSIONAL = "PROFESSIONAL"


class DecisionStatus(str, enum.Enum):
    """Enumeration for AI decision statuses."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class ReviewStatus(str, enum.Enum):
    """Enumeration for review statuses."""

    PENDING = "PENDING"
    REVIEWED = "REVIEWED"


@dataclass
class Student:
    """
    Student data class representing students in the system.

    Attributes:
        student_id: Unique identifier for the student (UUID)
        email: Student's email address (must be @students.oamk.fi)
        degree: Student's degree program
        created_at: Timestamp when the student record was created
        updated_at: Timestamp when the student record was last updated
    """

    student_id: UUID
    email: str
    degree: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_id": str(self.student_id),
            "email": self.email,
            "degree": self.degree,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Certificate:
    """
    Certificate data class representing uploaded work certificates.

    Attributes:
        certificate_id: Unique identifier for the certificate (UUID)
        student_id: Foreign key to the student who uploaded the certificate
        training_type: Type of training requested (GENERAL/PROFESSIONAL)
        filename: Original filename of the uploaded certificate
        filetype: File type/extension of the certificate
        filepath: Path or link to the uploaded file
        uploaded_at: Timestamp when the certificate was uploaded
    """

    certificate_id: UUID
    student_id: UUID
    training_type: TrainingType
    filename: str
    filetype: str
    filepath: Optional[str]
    uploaded_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "certificate_id": str(self.certificate_id),
            "student_id": str(self.student_id),
            "training_type": self.training_type.value,
            "filename": self.filename,
            "filetype": self.filetype,
            "filepath": self.filepath,
            "uploaded_at": self.uploaded_at.isoformat(),
        }


@dataclass
class Decision:
    """
    Decision data class representing evaluation decisions on certificates.

    Attributes:
        decision_id: Unique identifier for the decision (UUID)
        certificate_id: Foreign key to the certificate being evaluated
        ocr_output: OCR extracted text from the certificate
        ai_decision: AI decision (ACCEPTED/REJECTED)
        ai_justification: Explanation for the AI decision
        created_at: Timestamp when the decision was made
        student_feedback: Student's feedback for rejected applications
        review_status: Current review status (PENDING/REVIEWED)
        reviewer_comment: Reviewer's comments
        reviewed_at: Timestamp when the review was completed
    """

    decision_id: UUID
    certificate_id: UUID
    ocr_output: Optional[str]
    ai_decision: DecisionStatus
    ai_justification: str
    created_at: datetime
    student_feedback: Optional[str] = None
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewer_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": str(self.decision_id),
            "certificate_id": str(self.certificate_id),
            "ocr_output": self.ocr_output,
            "ai_decision": self.ai_decision.value,
            "ai_justification": self.ai_justification,
            "created_at": self.created_at.isoformat(),
            "student_feedback": self.student_feedback,
            "review_status": self.review_status.value,
            "reviewer_comment": self.reviewer_comment,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


@dataclass
class StudentWithCertificates:
    """
    Extended student data class with their certificates.

    Attributes:
        student_id: Unique identifier for the student
        email: Student's email address
        degree: Student's degree program
        created_at: Timestamp when the student record was created
        updated_at: Timestamp when the student record was last updated
        certificates: List of certificates uploaded by the student
    """

    student_id: UUID
    email: str
    degree: str
    created_at: datetime
    updated_at: datetime
    certificates: List[Certificate]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_id": str(self.student_id),
            "email": self.email,
            "degree": self.degree,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "certificates": [cert.to_dict() for cert in self.certificates],
        }


@dataclass
class ApplicationSummary:
    """
    Summary data class for reviewer dashboard listing applications.

    Attributes:
        decision_id: Unique identifier for the decision
        certificate_id: Unique identifier for the certificate
        student_name: Student's name (derived from email)
        student_email: Student's email address
        student_degree: Student's degree program
        filename: Certificate filename
        training_type: Type of training requested
        ai_decision: AI decision result
        review_status: Current review status
        uploaded_at: When certificate was uploaded
        created_at: When decision was created
        student_feedback: Student's feedback if any
    """

    decision_id: UUID
    certificate_id: UUID
    student_name: str
    student_email: str
    student_degree: str
    filename: str
    training_type: TrainingType
    ai_decision: DecisionStatus
    review_status: ReviewStatus
    uploaded_at: datetime
    created_at: datetime
    student_feedback: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": str(self.decision_id),
            "certificate_id": str(self.certificate_id),
            "student_name": self.student_name,
            "student_email": self.student_email,
            "student_degree": self.student_degree,
            "filename": self.filename,
            "training_type": self.training_type.value,
            "ai_decision": self.ai_decision.value,
            "review_status": self.review_status.value,
            "uploaded_at": self.uploaded_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "student_feedback": self.student_feedback,
        }


@dataclass
class DetailedApplication:
    """
    Detailed application data for reviewer review page.

    Attributes:
        decision: The decision record
        certificate: The certificate record
        student: The student record
    """

    decision: Decision
    certificate: Certificate
    student: Student

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision.to_dict(),
            "certificate": self.certificate.to_dict(),
            "student": self.student.to_dict(),
        }
