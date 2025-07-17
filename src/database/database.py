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
    ApplicationSummary,
    Certificate,
    Decision,
    DecisionStatus,
    DetailedApplication,
    ReviewStatus,
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
    """
    Test database connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
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

    Returns:
        Student: Created student object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            student_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO students (student_id, email, degree, first_name, last_name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (str(student_id), email, degree, first_name, last_name, now, now),
            )
            conn.commit()

            return Student(
                student_id=student_id,
                email=email,
                degree=degree,
                first_name=first_name,
                last_name=last_name,
                created_at=now,
                updated_at=now,
            )


def get_student_by_id(student_id: UUID) -> Optional[Student]:
    """
    Get student by ID.

    Args:
        student_id: Student's UUID

    Returns:
        Optional[Student]: Student object if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id, email, degree, first_name, last_name, created_at, updated_at FROM students WHERE student_id = %s",
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
                created_at=row[5],
                updated_at=row[6],
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
                "SELECT student_id, email, degree, first_name, last_name, created_at, updated_at FROM students WHERE email = %s",
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
                created_at=row[5],
                updated_at=row[6],
            )


def get_students(skip: int = 0, limit: int = 100) -> List[Student]:
    """
    Get list of students with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List[Student]: List of student objects
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id, email, degree, first_name, last_name, created_at, updated_at FROM students ORDER BY created_at DESC OFFSET %s LIMIT %s",
                (skip, limit),
            )
            rows = cur.fetchall()

            return [
                Student(
                    student_id=UUID(row[0]),
                    email=row[1],
                    degree=row[2],
                    first_name=row[3],
                    last_name=row[4],
                    created_at=row[5],
                    updated_at=row[6],
                )
                for row in rows
            ]


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
                "SELECT student_id, email, degree, first_name, last_name, created_at, updated_at FROM students WHERE student_id = %s",
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
                created_at=student_row[5],
                updated_at=student_row[6],
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


# Raw SQL operations for Decisions
def create_decision(
    certificate_id: UUID,
    ocr_output: Optional[str],
    ai_decision: DecisionStatus,
    ai_justification: str,
    student_feedback: Optional[str] = None,
    review_status: ReviewStatus = ReviewStatus.PENDING,
) -> Decision:
    """
    Create a new decision record.

    Args:
        certificate_id: Certificate's UUID
        ocr_output: OCR extracted text
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
                INSERT INTO decisions (decision_id, certificate_id, ocr_output, ai_justification, ai_decision, created_at, student_feedback, review_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(decision_id),
                    str(certificate_id),
                    ocr_output,
                    ai_justification,
                    ai_decision.value,
                    now,
                    student_feedback,
                    review_status.value,
                ),
            )
            conn.commit()

            return Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ocr_output=ocr_output,
                ai_decision=ai_decision,
                ai_justification=ai_justification,
                created_at=now,
                student_feedback=student_feedback,
                review_status=review_status,
                reviewer_comment=None,
                reviewed_at=None,
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
                       student_feedback, review_status, reviewer_comment, reviewed_at
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
                review_status=ReviewStatus(row[7]),
                reviewer_comment=row[8],
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
                       student_feedback, review_status, reviewer_comment, reviewed_at
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
                    review_status=ReviewStatus(row[7]),
                    reviewer_comment=row[8],
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
                       c.training_type, d.ai_decision, d.review_status, c.uploaded_at, 
                       d.created_at, d.student_feedback
                FROM decisions d
                JOIN certificates c ON d.certificate_id = c.certificate_id
                JOIN students s ON c.student_id = s.student_id
                WHERE d.review_status = 'PENDING'
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
                    review_status=ReviewStatus(row[7]),
                    uploaded_at=row[8],
                    created_at=row[9],
                    student_feedback=row[10],
                )
                for row in rows
            ]


def get_applications_by_status(review_status: ReviewStatus) -> List[ApplicationSummary]:
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
                       c.training_type, d.ai_decision, d.review_status, c.uploaded_at, 
                       d.created_at, d.student_feedback
                FROM decisions d
                JOIN certificates c ON d.certificate_id = c.certificate_id
                JOIN students s ON c.student_id = s.student_id
                WHERE d.review_status = %s
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
                    review_status=ReviewStatus(row[7]),
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
                       student_feedback, review_status, reviewer_comment, reviewed_at
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
                SELECT student_id, email, degree, created_at, updated_at
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
                review_status=ReviewStatus(decision_row[7]),
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
                created_at=student_row[3],
                updated_at=student_row[4],
            )

            return DetailedApplication(
                decision=decision,
                certificate=certificate,
                student=student,
            )


def update_decision_review(
    certificate_id: UUID,
    reviewer_comment: str,
    review_status: ReviewStatus = ReviewStatus.REVIEWED,
    student_feedback: Optional[str] = None,
) -> bool:
    """
    Update decision with reviewer comments and status.

    Args:
        certificate_id: Certificate's UUID
        reviewer_comment: Reviewer's comments
        review_status: New review status
        student_feedback: Updated student feedback (optional)

    Returns:
        bool: True if update successful, False otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            now = datetime.now()

            # Build the update query dynamically based on provided parameters
            update_fields = [
                "reviewer_comment = %s",
                "review_status = %s",
                "reviewed_at = %s",
            ]
            params = [reviewer_comment, review_status.value, now]

            if student_feedback is not None:
                update_fields.append("student_feedback = %s")
                params.append(student_feedback)

            params.append(str(certificate_id))

            query = f"""
                UPDATE decisions 
                SET {", ".join(update_fields)}
                WHERE certificate_id = %s
            """

            cur.execute(query, params)
            conn.commit()

            return cur.rowcount > 0


def add_student_feedback(certificate_id: UUID, student_feedback: str) -> bool:
    """
    Add student feedback to a decision.

    Args:
        certificate_id: Certificate's UUID
        student_feedback: Student's feedback

    Returns:
        bool: True if update successful, False otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
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

            # Count by review status
            cur.execute(
                "SELECT review_status, COUNT(*) FROM decisions GROUP BY review_status"
            )
            review_statuses = {row[0]: row[1] for row in cur.fetchall()}
            stats["review_statuses"] = review_statuses

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
