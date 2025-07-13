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

    GENERAL = "general"
    PROFESSIONAL = "professional"


class DecisionStatus(str, enum.Enum):
    """Enumeration for decision statuses."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


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
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_id": str(self.student_id),
            "email": self.email,
            "degree": self.degree,
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
        training_type: Type of training requested (general/professional)
        filename: Original filename of the uploaded certificate
        filetype: File type/extension of the certificate
        uploaded_at: Timestamp when the certificate was uploaded
    """

    certificate_id: UUID
    student_id: UUID
    training_type: TrainingType
    filename: str
    filetype: str
    uploaded_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "certificate_id": str(self.certificate_id),
            "student_id": str(self.student_id),
            "training_type": self.training_type.value,
            "filename": self.filename,
            "filetype": self.filetype,
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
        decision: Final decision (accepted/rejected)
        justification: Explanation for the decision
        created_at: Timestamp when the decision was made
        assigned_reviewer: Name/identifier of the reviewer (optional)
    """

    decision_id: UUID
    certificate_id: UUID
    ocr_output: Optional[str]
    decision: DecisionStatus
    justification: str
    created_at: datetime
    assigned_reviewer: Optional[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": str(self.decision_id),
            "certificate_id": str(self.certificate_id),
            "ocr_output": self.ocr_output,
            "decision": self.decision.value,
            "justification": self.justification,
            "created_at": self.created_at.isoformat(),
            "assigned_reviewer": self.assigned_reviewer,
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
