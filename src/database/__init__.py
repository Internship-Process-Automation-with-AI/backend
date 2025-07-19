"""
Database package for OAMK Work Certificate Processor.

This package contains database models, configurations, and utilities
for PostgreSQL database operations using raw SQL.
"""

from .database import (
    add_student_feedback,
    check_database_health,
    create_certificate,
    create_database_if_not_exists,
    create_decision,
    create_student,
    get_applications_by_status,
    get_certificate_by_id,
    get_certificates,
    get_database_info,
    get_db,
    get_db_connection,
    get_decision_by_id,
    get_decisions,
    get_detailed_application,
    get_pending_applications,
    get_statistics,
    get_student_by_email,
    get_student_with_certificates,
    test_database_connection,
    update_decision_review,
)
from .init_db import init_database
from .models import (
    ApplicationSummary,
    Certificate,
    Decision,
    DecisionStatus,
    DetailedApplication,
    ReviewerDecision,
    Student,
    StudentWithCertificates,
    TrainingType,
)

__all__ = [
    # Database operations
    "get_db_connection",
    "get_db",
    "create_database_if_not_exists",
    "test_database_connection",
    "check_database_health",
    "get_database_info",
    "init_database",
    # Student operations
    "create_student",
    "get_student_by_id",
    "get_student_by_email",
    "get_students",
    "get_student_with_certificates",
    # Certificate operations
    "create_certificate",
    "get_certificate_by_id",
    "get_certificates",
    # Decision operations
    "create_decision",
    "get_decision_by_id",
    "get_decisions",
    "add_student_feedback",
    "update_decision_review",
    # Reviewer operations
    "get_pending_applications",
    "get_applications_by_status",
    "get_detailed_application",
    # Statistics
    "get_statistics",
    # Models
    "Student",
    "Certificate",
    "Decision",
    "StudentWithCertificates",
    "ApplicationSummary",
    "DetailedApplication",
    "TrainingType",
    "DecisionStatus",
    "ReviewerDecision",
]
