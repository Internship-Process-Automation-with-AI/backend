"""
Data models for OAMK Work Certificate Processor database.

This module defines the data classes and enums for students, certificates, and decisions
using raw SQL operations.
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


# Outcome of the human review step (None = pending)
class ReviewerDecision(str, enum.Enum):
    PASS_ = "PASS"  # Certificate accepted by reviewer
    FAIL = "FAIL"  # Certificate rejected by reviewer


class AppealStatus(str, enum.Enum):
    """Enumeration for appeal statuses."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class Student:
    """
    Student data class representing students in the system.

    Attributes:
        student_id: Unique identifier for the student (UUID)
        email: Student's email address (must be @students.oamk.fi)
        degree: Student's degree program
        first_name: Student's first name
        last_name: Student's last name
    """

    student_id: UUID
    email: str
    degree: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_id": str(self.student_id),
            "email": self.email,
            "degree": self.degree,
            "first_name": self.first_name,
            "last_name": self.last_name,
        }


@dataclass
class Reviewer:
    """
    Reviewer data class representing users who can review certificates.

    Attributes:
        reviewer_id: Unique identifier for the reviewer (UUID)
        email: Reviewer's email address
        first_name: Reviewer's first name
        last_name: Reviewer's last name
        position: Reviewer's position
        department: Reviewer's department
    """

    reviewer_id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    department: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "reviewer_id": str(self.reviewer_id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "position": self.position,
            "department": self.department,
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
    ocr_output: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "certificate_id": str(self.certificate_id),
            "filename": self.filename,
            "training_type": self.training_type.value,
            "uploaded_at": str(self.uploaded_at),
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
        reviewer_id: Unique identifier for the reviewer (UUID)
        reviewer_decision: Outcome of the human review step (None = pending)
        reviewer_comment: Reviewer's comments
        reviewed_at: Timestamp when the review was completed
        appeal_reason: Student's appeal reason
        appeal_status: Appeal status (PENDING/APPROVED/REJECTED)
        appeal_submitted_at: When appeal was submitted
        appeal_reviewer_id: Reviewer handling the appeal
        appeal_review_comment: Appeal reviewer's comments
        appeal_reviewed_at: When appeal was reviewed
    """

    decision_id: UUID
    certificate_id: UUID
    ai_justification: str
    ai_decision: DecisionStatus
    created_at: datetime
    student_feedback: Optional[str] = None
    reviewer_id: Optional[UUID] = None
    reviewer_decision: Optional[ReviewerDecision] = None  # NULL == pending
    reviewer_comment: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    appeal_reason: Optional[str] = None
    appeal_status: Optional[AppealStatus] = None
    appeal_submitted_at: Optional[datetime] = None
    appeal_reviewer_id: Optional[UUID] = None
    appeal_review_comment: Optional[str] = None
    appeal_reviewed_at: Optional[datetime] = None
    # Evaluation details
    total_working_hours: Optional[int] = None
    credits_awarded: Optional[int] = None
    training_duration: Optional[str] = None
    training_institution: Optional[str] = None
    degree_relevance: Optional[str] = None
    supporting_evidence: Optional[str] = None
    challenging_evidence: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": str(self.decision_id),
            "certificate_id": str(self.certificate_id),
            "ai_justification": self.ai_justification,
            "ai_decision": self.ai_decision.value,
            "created_at": self.created_at.isoformat(),
            "student_feedback": self.student_feedback,
            "reviewer_id": str(self.reviewer_id) if self.reviewer_id else None,
            "reviewer_decision": self.reviewer_decision.value
            if self.reviewer_decision
            else None,
            "reviewer_comment": self.reviewer_comment,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "appeal_reason": self.appeal_reason,
            "appeal_status": self.appeal_status.value if self.appeal_status else None,
            "appeal_submitted_at": self.appeal_submitted_at.isoformat()
            if self.appeal_submitted_at
            else None,
            "appeal_reviewer_id": str(self.appeal_reviewer_id)
            if self.appeal_reviewer_id
            else None,
            "appeal_review_comment": self.appeal_review_comment,
            "appeal_reviewed_at": self.appeal_reviewed_at.isoformat()
            if self.appeal_reviewed_at
            else None,
            # Evaluation details
            "total_working_hours": self.total_working_hours,
            "credits_awarded": self.credits_awarded,
            "training_duration": self.training_duration,
            "training_institution": self.training_institution,
            "degree_relevance": self.degree_relevance,
            "supporting_evidence": self.supporting_evidence,
            "challenging_evidence": self.challenging_evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class StudentWithCertificates:
    """
    Extended student data class with their certificates.

    Attributes:
        student_id: Unique identifier for the student
        email: Student's email address
        degree: Student's degree program
        first_name: Student's first name
        last_name: Student's last name
        certificates: List of certificates uploaded by the student
    """

    student_id: UUID
    email: str
    degree: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    certificates: List[Certificate] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "student_id": str(self.student_id),
            "email": self.email,
            "degree": self.degree,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "certificates": [cert.to_dict() for cert in self.certificates]
            if self.certificates
            else [],
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
    uploaded_at: datetime
    created_at: datetime
    reviewer_decision: Optional[ReviewerDecision] = None
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
            "uploaded_at": self.uploaded_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "reviewer_decision": self.reviewer_decision.value
            if self.reviewer_decision
            else None,
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
