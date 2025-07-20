"""
Database configuration and connection management for OAMK Work Certificate Processor.

This module provides raw SQL database operations using psycopg2 for PostgreSQL.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import psycopg2

from src.config import settings

from .models import (
    AppealStatus,
    ApplicationSummary,
    Certificate,
    Decision,
    DecisionStatus,
    DetailedApplication,
    Reviewer,
    ReviewerDecision,
    Student,
    StudentWithCertificates,
    TrainingType,
)

logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    "host": settings.DATABASE_HOST,
    "port": settings.DATABASE_PORT,
    "database": settings.DATABASE_NAME,
    "user": settings.DATABASE_USER,
    "password": settings.DATABASE_PASSWORD,
}


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    Yields:
        psycopg2.connection: Database connection
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_db():
    """Get database connection for external use."""
    return psycopg2.connect(**DB_CONFIG)


def create_database_if_not_exists() -> bool:
    """
    Create the database if it doesn't exist.

    Returns:
        bool: True if database was created or already exists, False if failed
    """
    try:
        # Connect to default postgres database to create our database
        temp_config = DB_CONFIG.copy()
        temp_config["database"] = "postgres"

        with psycopg2.connect(**temp_config) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (DB_CONFIG["database"],),
                )
                if cur.fetchone():
                    logger.info(f"Database {DB_CONFIG['database']} already exists")
                    return True

                # Create database
                cur.execute(f'CREATE DATABASE "{DB_CONFIG["database"]}"')
                logger.info(f"Created database {DB_CONFIG['database']}")
        return True

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False


def test_database_connection() -> bool:
    """Test database connection."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get database information.

    Returns:
        dict: Database information
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                return {"status": "connected", "version": version}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_database_health() -> dict:
    """
    Check database health and basic statistics.

    Returns:
        dict: Health check results
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if tables exist
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('students', 'certificates', 'decisions')
                    """
                )
                tables = [row[0] for row in cur.fetchall()]

                # Get basic counts
                counts = {}
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    counts[table] = cur.fetchone()[0]

                return {
                    "status": "healthy",
                    "tables": tables,
                    "counts": counts,
                    "timestamp": datetime.now().isoformat(),
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def init_database():
    """Initialize database with schema."""
    create_database_if_not_exists()
    # Schema creation would be handled separately via SQL files


# Raw SQL operations for Students
def create_student(
    email: str,
    degree: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> Student:
    """
    Create a new student record.

    Args:
        email: Student's email address
        degree: Student's degree program
        first_name: Student's first name
        last_name: Student's last name

    Returns:
        Student: Created student object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            student_id = uuid4()
            cur.execute(
                """
                INSERT INTO students (student_id, email, degree, first_name, last_name)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(student_id), email, degree, first_name, last_name),
            )
            conn.commit()

            return Student(
                student_id=student_id,
                email=email,
                degree=degree,
                first_name=first_name,
                last_name=last_name,
            )


def get_student_by_email(email: str) -> Optional[Student]:
    """
    Get student by email.

    Args:
        email: Student's email address

    Returns:
        Optional[Student]: Student object if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id, email, degree, first_name, last_name FROM students WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Student(
                student_id=UUID(row[0]),
                email=row[1],
                degree=row[2],
                first_name=row[3],
                last_name=row[4],
            )


def get_student_by_id(student_id: UUID) -> Optional[Student]:
    """Get student by UUID."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id, email, degree, first_name, last_name FROM students WHERE student_id = %s",
                (str(student_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Student(
                student_id=UUID(row[0]),
                email=row[1],
                degree=row[2],
                first_name=row[3],
                last_name=row[4],
            )


def get_student_with_certificates(
    student_id: UUID,
) -> Optional[StudentWithCertificates]:
    """
    Get student with their certificates.

    Args:
        student_id: Student's UUID

    Returns:
        Optional[StudentWithCertificates]: Student with certificates if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get student
            cur.execute(
                "SELECT student_id, email, degree, first_name, last_name FROM students WHERE student_id = %s",
                (str(student_id),),
            )
            student_row = cur.fetchone()
            if not student_row:
                return None

            # Get certificates
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, filepath, uploaded_at
                FROM certificates WHERE student_id = %s ORDER BY uploaded_at DESC
                """,
                (str(student_id),),
            )
            certificate_rows = cur.fetchall()

            certificates = [
                Certificate(
                    certificate_id=UUID(row[0]),
                    student_id=UUID(row[1]),
                    training_type=TrainingType(row[2]),
                    filename=row[3],
                    filetype=row[4],
                    filepath=row[5],
                    uploaded_at=row[6],
                )
                for row in certificate_rows
            ]

            return StudentWithCertificates(
                student_id=student_row[0],
                email=student_row[1],
                degree=student_row[2],
                first_name=student_row[3],
                last_name=student_row[4],
                certificates=certificates,
            )


# Raw SQL operations for Certificates
def create_certificate(
    student_id: UUID,
    training_type: TrainingType,
    filename: str,
    filetype: str,
    filepath: Optional[str] = None,
) -> Certificate:
    """
    Create a new certificate record.

    Args:
        student_id: Student's UUID
        training_type: Type of training requested
        filename: Original filename
        filetype: File type/extension
        filepath: Path to the uploaded file

    Returns:
        Certificate: Created certificate object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            certificate_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO certificates (certificate_id, student_id, training_type, filename, filetype, filepath, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(certificate_id),
                    str(student_id),
                    training_type.value,
                    filename,
                    filetype,
                    filepath,
                    now,
                ),
            )
            conn.commit()

            return Certificate(
                certificate_id=certificate_id,
                student_id=student_id,
                training_type=training_type,
                filename=filename,
                filetype=filetype,
                filepath=filepath,
                uploaded_at=now,
            )


def get_certificate_by_id(certificate_id: UUID) -> Optional[Certificate]:
    """
    Get certificate by ID.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        Optional[Certificate]: Certificate object if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, filepath, uploaded_at
                FROM certificates WHERE certificate_id = %s
                """,
                (str(certificate_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Certificate(
                certificate_id=UUID(row[0]),
                student_id=UUID(row[1]),
                training_type=TrainingType(row[2]),
                filename=row[3],
                filetype=row[4],
                filepath=row[5],
                uploaded_at=row[6],
            )


def get_certificates(skip: int = 0, limit: int = 100) -> List[Certificate]:
    """
    Get list of certificates with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List[Certificate]: List of certificate objects
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, filepath, uploaded_at
                FROM certificates ORDER BY uploaded_at DESC OFFSET %s LIMIT %s
                """,
                (skip, limit),
            )
            rows = cur.fetchall()

            return [
                Certificate(
                    certificate_id=UUID(row[0]),
                    student_id=UUID(row[1]),
                    training_type=TrainingType(row[2]),
                    filename=row[3],
                    filetype=row[4],
                    filepath=row[5],
                    uploaded_at=row[6],
                )
                for row in rows
            ]


def delete_certificate(certificate_id: UUID) -> bool:
    """
    Delete a certificate and all its associated data.

    Args:
        certificate_id: Certificate ID to delete

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Delete associated decision first (due to foreign key constraint)
                cur.execute(
                    "DELETE FROM decisions WHERE certificate_id = %s",
                    (str(certificate_id),),
                )

                # Delete the certificate
                cur.execute(
                    "DELETE FROM certificates WHERE certificate_id = %s",
                    (str(certificate_id),),
                )

                # Check if any rows were affected
                if cur.rowcount > 0:
                    conn.commit()
                    logger.info(f"Successfully deleted certificate {certificate_id}")
                    return True
                else:
                    logger.warning(f"No certificate found with ID {certificate_id}")
                    return False

            except Exception as e:
                conn.rollback()
                logger.error(f"Error deleting certificate {certificate_id}: {e}")
                raise


# Raw SQL operations for Decisions
def create_decision(
    certificate_id: UUID,
    ai_decision: DecisionStatus,
    ai_justification: str,
    student_feedback: Optional[str] = None,
    total_working_hours: Optional[int] = None,
    credits_awarded: Optional[int] = None,
    training_duration: Optional[str] = None,
    training_institution: Optional[str] = None,
    degree_relevance: Optional[str] = None,
    supporting_evidence: Optional[str] = None,
    challenging_evidence: Optional[str] = None,
    recommendation: Optional[str] = None,
) -> Decision:
    """
    Create a new decision record.

    Args:
        certificate_id: Certificate's UUID
        ai_decision: AI decision status
        justification: Explanation for the decision
        student_feedback: Student's feedback
        review_status: Review status

    Returns:
        Decision: Created decision object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            decision_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO decisions (
                    decision_id, certificate_id, ai_justification, ai_decision, created_at, student_feedback,
                    total_working_hours, credits_awarded, training_duration, training_institution,
                    degree_relevance, supporting_evidence, challenging_evidence, recommendation
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(decision_id),
                    str(certificate_id),
                    ai_justification,
                    ai_decision.value,
                    now,
                    student_feedback,
                    total_working_hours,
                    credits_awarded,
                    training_duration,
                    training_institution,
                    degree_relevance,
                    supporting_evidence,
                    challenging_evidence,
                    recommendation,
                ),
            )
            conn.commit()

            return Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ai_decision=ai_decision,
                ai_justification=ai_justification,
                created_at=now,
                student_feedback=student_feedback,
                reviewer_decision=None,
                reviewed_at=None,
                total_working_hours=total_working_hours,
                credits_awarded=credits_awarded,
                training_duration=training_duration,
                training_institution=training_institution,
                degree_relevance=degree_relevance,
                supporting_evidence=supporting_evidence,
                challenging_evidence=challenging_evidence,
                recommendation=recommendation,
            )


def get_decision_by_id(decision_id: UUID) -> Optional[Decision]:
    """
    Get decision by ID.

    Args:
        decision_id: Decision's UUID

    Returns:
        Optional[Decision]: Decision object if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT decision_id, certificate_id, ocr_output, ai_justification, ai_decision, created_at,
                       student_feedback, reviewer_decision, reviewer_comment, reviewed_at
                FROM decisions WHERE decision_id = %s
                """,
                (str(decision_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Decision(
                decision_id=UUID(row[0]),
                certificate_id=UUID(row[1]),
                ocr_output=row[2],
                ai_justification=row[3],
                ai_decision=DecisionStatus(row[4]),
                created_at=row[5],
                student_feedback=row[6],
                reviewer_decision=ReviewerDecision(row[7]) if row[7] else None,
                reviewed_at=row[9],
            )


def get_decisions(skip: int = 0, limit: int = 100) -> List[Decision]:
    """
    Get list of decisions with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List[Decision]: List of decision objects
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT decision_id, certificate_id, ocr_output, ai_justification, ai_decision, created_at,
                       student_feedback, reviewer_decision, reviewer_comment, reviewed_at
                FROM decisions ORDER BY created_at DESC OFFSET %s LIMIT %s
                """,
                (skip, limit),
            )
            rows = cur.fetchall()

            return [
                Decision(
                    decision_id=UUID(row[0]),
                    certificate_id=UUID(row[1]),
                    ocr_output=row[2],
                    ai_justification=row[3],
                    ai_decision=DecisionStatus(row[4]),
                    created_at=row[5],
                    student_feedback=row[6],
                    reviewer_decision=ReviewerDecision(row[7]) if row[7] else None,
                    reviewed_at=row[9],
                )
                for row in rows
            ]


# New functions for reviewer functionality
def get_pending_applications() -> List[ApplicationSummary]:
    """
    Get all applications with PENDING review status.

    Returns:
        List[ApplicationSummary]: List of pending applications
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.decision_id, d.certificate_id, s.email, s.degree, c.filename, 
                       c.training_type, d.ai_decision, d.reviewer_decision, c.uploaded_at, 
                       d.created_at, d.student_feedback
                FROM decisions d
                JOIN certificates c ON d.certificate_id = c.certificate_id
                JOIN students s ON c.student_id = s.student_id
                WHERE d.reviewer_decision IS NULL
                ORDER BY d.created_at DESC
                """
            )
            rows = cur.fetchall()

            return [
                ApplicationSummary(
                    decision_id=UUID(row[0]),
                    certificate_id=UUID(row[1]),
                    student_name=row[2]
                    .split("@")[0]
                    .replace(".", " ")
                    .title(),  # Extract name from email
                    student_email=row[2],
                    student_degree=row[3],
                    filename=row[4],
                    training_type=TrainingType(row[5]),
                    ai_decision=DecisionStatus(row[6]),
                    reviewer_decision=ReviewerDecision(row[7]) if row[7] else None,
                    uploaded_at=row[8],
                    created_at=row[9],
                    student_feedback=row[10],
                )
                for row in rows
            ]


def get_applications_by_status(
    review_status: ReviewerDecision,
) -> List[ApplicationSummary]:
    """
    Get applications by review status.

    Args:
        review_status: Review status to filter by

    Returns:
        List[ApplicationSummary]: List of applications with the specified status
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.decision_id, d.certificate_id, s.email, s.degree, c.filename, 
                       c.training_type, d.ai_decision, d.reviewer_decision, c.uploaded_at, 
                       d.created_at, d.student_feedback
                FROM decisions d
                JOIN certificates c ON d.certificate_id = c.certificate_id
                JOIN students s ON c.student_id = s.student_id
                WHERE d.reviewer_decision = %s
                ORDER BY d.created_at DESC
                """,
                (review_status.value,),
            )
            rows = cur.fetchall()

            return [
                ApplicationSummary(
                    decision_id=UUID(row[0]),
                    certificate_id=UUID(row[1]),
                    student_name=row[2].split("@")[0].replace(".", " ").title(),
                    student_email=row[2],
                    student_degree=row[3],
                    filename=row[4],
                    training_type=TrainingType(row[5]),
                    ai_decision=DecisionStatus(row[6]),
                    reviewer_decision=ReviewerDecision(row[7]),
                    uploaded_at=row[8],
                    created_at=row[9],
                    student_feedback=row[10],
                )
                for row in rows
            ]


def get_detailed_application(certificate_id: UUID) -> Optional[DetailedApplication]:
    """
    Get detailed application information by certificate ID.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        Optional[DetailedApplication]: Detailed application if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get decision
            cur.execute(
                """
                SELECT decision_id, certificate_id, ocr_output, ai_justification, ai_decision, created_at,
                       student_feedback, reviewer_decision, reviewer_comment, reviewed_at
                FROM decisions WHERE certificate_id = %s
                """,
                (str(certificate_id),),
            )
            decision_row = cur.fetchone()
            if not decision_row:
                return None

            # Get certificate
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, filepath, uploaded_at
                FROM certificates WHERE certificate_id = %s
                """,
                (str(certificate_id),),
            )
            certificate_row = cur.fetchone()
            if not certificate_row:
                return None

            # Get student
            cur.execute(
                """
                SELECT student_id, email, degree, first_name, last_name
                FROM students WHERE student_id = %s
                """,
                (str(certificate_row[1]),),
            )
            student_row = cur.fetchone()
            if not student_row:
                return None

            decision = Decision(
                decision_id=UUID(decision_row[0]),
                certificate_id=UUID(decision_row[1]),
                ocr_output=decision_row[2],
                ai_justification=decision_row[3],
                ai_decision=DecisionStatus(decision_row[4]),
                created_at=decision_row[5],
                student_feedback=decision_row[6],
                reviewer_decision=ReviewerDecision(decision_row[7]),
                reviewer_comment=decision_row[8],
                reviewed_at=decision_row[9],
            )

            certificate = Certificate(
                certificate_id=UUID(certificate_row[0]),
                student_id=UUID(certificate_row[1]),
                training_type=TrainingType(certificate_row[2]),
                filename=certificate_row[3],
                filetype=certificate_row[4],
                filepath=certificate_row[5],
                uploaded_at=certificate_row[6],
            )

            student = Student(
                student_id=UUID(student_row[0]),
                email=student_row[1],
                degree=student_row[2],
                first_name=student_row[3],
                last_name=student_row[4],
            )

            return DetailedApplication(
                decision=decision,
                certificate=certificate,
                student=student,
            )


def update_decision_review(
    certificate_id: UUID,
    reviewer_comment: str,
    reviewer_decision: ReviewerDecision,
    student_feedback: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Add a reviewer comment and mark the decision as *REVIEWED*.

    Args:
        certificate_id: Certificate's UUID.
        reviewer_comment: Free-text comment from the human reviewer.
        student_feedback: Optional student feedback to update at the same time.

    Returns:
        tuple[bool, Optional[str]]: (success, error_message)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # First check if certificate exists and current status
                cur.execute(
                    "SELECT reviewer_decision FROM decisions WHERE certificate_id = %s",
                    (str(certificate_id),),
                )
                result = cur.fetchone()
                if not result:
                    return False, "Certificate not found"

                now = datetime.now()

                cur.execute(
                    """
                    UPDATE decisions 
                    SET reviewer_comment = %s,
                        reviewer_decision = %s,
                        reviewed_at = %s
                    WHERE certificate_id = %s
                    """,
                    (
                        reviewer_comment,
                        reviewer_decision.value,
                        now,
                        str(certificate_id),
                    ),
                )

                if student_feedback is not None:
                    cur.execute(
                        """
                        UPDATE decisions 
                        SET student_feedback = %s
                        WHERE certificate_id = %s
                        """,
                        (student_feedback, str(certificate_id)),
                    )

                conn.commit()
                return True, None

    except Exception as e:
        error_msg = f"Database error in update_decision_review: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def add_student_feedback(
    certificate_id: UUID, student_feedback: str, reviewer_id: Optional[UUID] = None
) -> bool:
    """
    Add student feedback and optionally reviewer ID to a decision.

    Args:
        certificate_id: Certificate's UUID
        student_feedback: Student's feedback
        reviewer_id: Reviewer's UUID (optional)

    Returns:
        bool: True if update successful, False otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if reviewer_id:
                cur.execute(
                    """
                    UPDATE decisions 
                    SET student_feedback = %s, reviewer_id = %s
                    WHERE certificate_id = %s
                    """,
                    (student_feedback, str(reviewer_id), str(certificate_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE decisions 
                    SET student_feedback = %s
                    WHERE certificate_id = %s
                    """,
                    (student_feedback, str(certificate_id)),
                )
            conn.commit()
            return cur.rowcount > 0


def get_statistics() -> dict:
    """
    Get basic statistics.

    Returns:
        dict: Statistics data
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            stats = {}

            # Count students
            cur.execute("SELECT COUNT(*) FROM students")
            stats["total_students"] = cur.fetchone()[0]

            # Count certificates
            cur.execute("SELECT COUNT(*) FROM certificates")
            stats["total_certificates"] = cur.fetchone()[0]

            # Count decisions
            cur.execute("SELECT COUNT(*) FROM decisions")
            stats["total_decisions"] = cur.fetchone()[0]

            # Count by AI decision
            cur.execute(
                "SELECT ai_decision, COUNT(*) FROM decisions GROUP BY ai_decision"
            )
            ai_decisions = {row[0]: row[1] for row in cur.fetchall()}
            stats["ai_decisions"] = ai_decisions

            # Count by reviewer decision (PASS/FAIL) – NULL values (pending) are ignored here
            cur.execute(
                "SELECT reviewer_decision, COUNT(*) FROM decisions GROUP BY reviewer_decision"
            )
            reviewer_decisions = {row[0]: row[1] for row in cur.fetchall()}
            stats["reviewer_decisions"] = reviewer_decisions

            # Count by training type
            cur.execute(
                "SELECT training_type, COUNT(*) FROM certificates GROUP BY training_type"
            )
            training_types = {row[0]: row[1] for row in cur.fetchall()}
            stats["training_types"] = training_types

            return stats


def get_student_certificate_count(student_id: UUID) -> dict:
    """
    Get certificate count by training type for a student.

    Args:
        student_id: Student's UUID

    Returns:
        dict: Count of certificates by training type
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT training_type, COUNT(*) as count
                FROM certificates 
                WHERE student_id = %s
                GROUP BY training_type
                """,
                (str(student_id),),
            )
            rows = cur.fetchall()

            counts = {"GENERAL": 0, "PROFESSIONAL": 0}
            for row in rows:
                counts[row[0]] = row[1]

            return counts


def check_certificate_limits(
    student_id: UUID, training_type: TrainingType
) -> tuple[bool, str]:
    """
    Check if student can upload another certificate of the specified type.

    Args:
        student_id: Student's UUID
        training_type: Type of training requested

    Returns:
        tuple: (is_allowed, error_message)
    """
    counts = get_student_certificate_count(student_id)

    if training_type == TrainingType.GENERAL:
        if counts["GENERAL"] >= 1:
            return (
                False,
                "Student already has 1 general training certificate (maximum allowed)",
            )
    elif training_type == TrainingType.PROFESSIONAL:
        if counts["PROFESSIONAL"] >= 2:
            return (
                False,
                "Student already has 2 professional training certificates (maximum allowed)",
            )

    return True, ""


def create_sample_students():
    sample_data = [
        (
            "alice.smith@students.oamk.fi",
            "Business Information Technology",
            "Alice",
            "Smith",
        ),
        ("bob.johnson@students.oamk.fi", "Nursing", "Bob", "Johnson"),
        ("carol.wilson@students.oamk.fi", "Mechanical Engineering", "Carol", "Wilson"),
        ("david.brown@students.oamk.fi", "International Business", "David", "Brown"),
        ("eve.davis@students.oamk.fi", "Information Technology", "Eve", "Davis"),
        (
            "frank.miller@students.oamk.fi",
            "Energy and Environmental Engineering",
            "Frank",
            "Miller",
        ),
        ("grace.moore@students.oamk.fi", "Social Services", "Grace", "Moore"),
        ("henry.taylor@students.oamk.fi", "Physiotherapy", "Henry", "Taylor"),
        (
            "irene.anderson@students.oamk.fi",
            "Construction Engineering",
            "Irene",
            "Anderson",
        ),
        (
            "jack.thomas@students.oamk.fi",
            "Business Information Technology",
            "Jack",
            "Thomas",
        ),
    ]
    for email, degree, first_name, last_name in sample_data:
        if not get_student_by_email(email):
            create_student(email, degree, first_name, last_name)


# Raw SQL operations for Reviewers


def create_reviewer(
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
):
    """Create a reviewer record if not exists."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            reviewer_id = uuid4()
            cur.execute(
                "INSERT INTO reviewers (reviewer_id, email, first_name, last_name) VALUES (%s, %s, %s, %s) ON CONFLICT (email) DO NOTHING",
                (str(reviewer_id), email, first_name, last_name),
            )
            conn.commit()


def get_reviewer_by_email(email: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT reviewer_id, email, first_name, last_name FROM reviewers WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Reviewer(
                reviewer_id=UUID(row[0]),
                email=row[1],
                first_name=row[2],
                last_name=row[3],
            )


def get_all_reviewers():
    """
    Get all reviewers from the database.

    Returns:
        List[Reviewer]: List of all reviewers
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT reviewer_id, email, first_name, last_name FROM reviewers ORDER BY first_name, last_name"
            )
            rows = cur.fetchall()
            return [
                Reviewer(
                    reviewer_id=UUID(row[0]),
                    email=row[1],
                    first_name=row[2],
                    last_name=row[3],
                )
                for row in rows
            ]


def create_sample_reviewers():
    sample_reviewers = [
        ("laura.koskinen@oamk.fi", "Laura", "Koskinen"),
        ("jukka.virtanen@oamk.fi", "Jukka", "Virtanen"),
        ("emilia.makela@oamk.fi", "Emilia", "Mäkelä"),
        ("antti.lehtinen@oamk.fi", "Antti", "Lehtinen"),
        ("sanna.nieminen@oamk.fi", "Sanna", "Nieminen"),
    ]
    for email, first_name, last_name in sample_reviewers:
        if not get_reviewer_by_email(email):
            create_reviewer(email, first_name, last_name)


def get_certificates_by_reviewer_id(reviewer_id: UUID) -> List[DetailedApplication]:
    """
    Get all certificates assigned to a reviewer.

    Args:
        reviewer_id: Reviewer's UUID

    Returns:
        List[DetailedApplication]: List of detailed applications assigned to the reviewer
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.decision_id, d.certificate_id, c.ocr_output, d.ai_justification, 
                       d.ai_decision, d.created_at, d.student_feedback, d.reviewer_decision, 
                       d.reviewer_comment, d.reviewed_at,
                       c.student_id, c.training_type, c.filename, c.filetype, c.filepath, c.uploaded_at,
                       s.email, s.degree, s.first_name, s.last_name
                FROM decisions d
                JOIN certificates c ON d.certificate_id = c.certificate_id
                JOIN students s ON c.student_id = s.student_id
                WHERE d.reviewer_id = %s
                ORDER BY d.created_at DESC
                """,
                (str(reviewer_id),),
            )
            rows = cur.fetchall()

            return [
                DetailedApplication(
                    decision=Decision(
                        decision_id=UUID(row[0]),
                        certificate_id=UUID(row[1]),
                        ai_justification=row[3],
                        ai_decision=DecisionStatus(row[4]),
                        created_at=row[5],
                        student_feedback=row[6],
                        reviewer_decision=ReviewerDecision(row[7]) if row[7] else None,
                        reviewer_comment=row[8],
                        reviewed_at=row[9],
                        reviewer_id=reviewer_id,
                    ),
                    certificate=Certificate(
                        certificate_id=UUID(row[1]),
                        student_id=UUID(row[10]),
                        training_type=TrainingType(row[11]),
                        filename=row[12],
                        filetype=row[13],
                        filepath=row[14],
                        uploaded_at=row[15],
                        ocr_output=row[2],  # Add ocr_output from certificates table
                    ),
                    student=Student(
                        student_id=UUID(row[10]),
                        email=row[16],
                        degree=row[17],
                        first_name=row[18],
                        last_name=row[19],
                    ),
                )
                for row in rows
            ]


# Raw SQL operations for Appeals (integrated into decisions table)


def submit_appeal(certificate_id: UUID, appeal_reason: str, reviewer_id: UUID) -> bool:
    """
    Submit an appeal by updating the decision record.

    Args:
        certificate_id: Certificate's UUID
        appeal_reason: Student's reason for appealing
        reviewer_id: Reviewer's UUID to assign the appeal to

    Returns:
        bool: True if appeal was submitted successfully
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            now = datetime.now()
            cur.execute(
                """
                UPDATE decisions 
                SET appeal_reason = %s, 
                    appeal_status = %s, 
                    appeal_submitted_at = %s, 
                    appeal_reviewer_id = %s
                WHERE certificate_id = %s
                """,
                (
                    appeal_reason,
                    AppealStatus.PENDING.value,
                    now,
                    str(reviewer_id),
                    str(certificate_id),
                ),
            )
            conn.commit()
            return True


def get_appeal_by_certificate_id(certificate_id: UUID) -> Optional[dict]:
    """
    Get appeal information by certificate ID from decisions table.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        Optional[dict]: Appeal information if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT appeal_reason, appeal_status, appeal_submitted_at, 
                       appeal_reviewer_id, appeal_review_comment, appeal_reviewed_at
                FROM decisions WHERE certificate_id = %s AND appeal_reason IS NOT NULL
                """,
                (str(certificate_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "appeal_reason": row[0],
                "appeal_status": row[1],
                "appeal_submitted_at": row[2],
                "appeal_reviewer_id": row[3],
                "appeal_review_comment": row[4],
                "appeal_reviewed_at": row[5],
            }
