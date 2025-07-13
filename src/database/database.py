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
    Certificate,
    Decision,
    DecisionStatus,
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
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_db():
    """
    Get database connection for dependency injection.

    Returns:
        psycopg2.connection: Database connection
    """
    return psycopg2.connect(**DB_CONFIG)


def create_database_if_not_exists() -> bool:
    """
    Create the database if it doesn't exist.

    Returns:
        bool: True if database was created or already exists, False on error
    """
    try:
        # Connect to PostgreSQL server (postgres database)
        server_config = DB_CONFIG.copy()
        server_config["database"] = "postgres"

        with psycopg2.connect(**server_config) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (settings.DATABASE_NAME,),
                )

                if not cur.fetchone():
                    # Database doesn't exist, create it
                    cur.execute(f"CREATE DATABASE {settings.DATABASE_NAME}")
                    logger.info(f"Created database: {settings.DATABASE_NAME}")
                else:
                    logger.info(f"Database already exists: {settings.DATABASE_NAME}")

        return True

    except psycopg2.Error as e:
        logger.error(f"Error creating database: {e}")
        return False


def test_database_connection() -> bool:
    """
    Test database connection.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                if cur.fetchone():
                    logger.info("Database connection successful")
                    return True
                return False

    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get database connection information.

    Returns:
        dict: Database connection details (without sensitive information)
    """
    return {
        "host": settings.DATABASE_HOST,
        "port": settings.DATABASE_PORT,
        "database": settings.DATABASE_NAME,
        "user": settings.DATABASE_USER,
        "driver": "psycopg2",
    }


def check_database_health() -> dict:
    """
    Check database health status.

    Returns:
        dict: Database health status information
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Test basic connection
                cur.execute("SELECT 1")

                # Get database version
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]

                return {
                    "status": "healthy",
                    "connection": True,
                    "version": version,
                    "database_info": get_database_info(),
                }

    except psycopg2.Error as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "connection": False,
            "error": str(e),
            "database_info": get_database_info(),
        }


# Raw SQL operations for Students
def create_student(email: str, degree: str) -> Student:
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
                INSERT INTO students (student_id, email, degree, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(student_id), email, degree, now, now),
            )
            conn.commit()

            return Student(
                student_id=student_id,
                email=email,
                degree=degree,
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
                "SELECT student_id, email, degree, created_at, updated_at FROM students WHERE student_id = %s",
                (str(student_id),),
            )

            row = cur.fetchone()
            if row:
                return Student(
                    student_id=row[0],
                    email=row[1],
                    degree=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                )
            return None


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
                "SELECT student_id, email, degree, created_at, updated_at FROM students WHERE email = %s",
                (email,),
            )

            row = cur.fetchone()
            if row:
                return Student(
                    student_id=row[0],
                    email=row[1],
                    degree=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                )
            return None


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
                "SELECT student_id, email, degree, created_at, updated_at FROM students ORDER BY created_at DESC OFFSET %s LIMIT %s",
                (skip, limit),
            )

            return [
                Student(
                    student_id=row[0],
                    email=row[1],
                    degree=row[2],
                    created_at=row[3],
                    updated_at=row[4],
                )
                for row in cur.fetchall()
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
                "SELECT student_id, email, degree, created_at, updated_at FROM students WHERE student_id = %s",
                (str(student_id),),
            )

            student_row = cur.fetchone()
            if not student_row:
                return None

            # Get certificates
            cur.execute(
                "SELECT certificate_id, student_id, training_type, filename, filetype, uploaded_at FROM certificates WHERE student_id = %s ORDER BY uploaded_at DESC",
                (str(student_id),),
            )

            certificates = [
                Certificate(
                    certificate_id=row[0],
                    student_id=row[1],
                    training_type=TrainingType(row[2]),
                    filename=row[3],
                    filetype=row[4],
                    uploaded_at=row[5],
                )
                for row in cur.fetchall()
            ]

            return StudentWithCertificates(
                student_id=student_row[0],
                email=student_row[1],
                degree=student_row[2],
                created_at=student_row[3],
                updated_at=student_row[4],
                certificates=certificates,
            )


# Raw SQL operations for Certificates
def create_certificate(
    student_id: UUID, training_type: TrainingType, filename: str, filetype: str
) -> Certificate:
    """
    Create a new certificate record.

    Args:
        student_id: Student's UUID
        training_type: Type of training requested
        filename: Original filename
        filetype: File type/extension

    Returns:
        Certificate: Created certificate object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            certificate_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO certificates (certificate_id, student_id, training_type, filename, filetype, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(certificate_id),
                    str(student_id),
                    training_type.value,
                    filename,
                    filetype,
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
                "SELECT certificate_id, student_id, training_type, filename, filetype, uploaded_at FROM certificates WHERE certificate_id = %s",
                (str(certificate_id),),
            )

            row = cur.fetchone()
            if row:
                return Certificate(
                    certificate_id=row[0],
                    student_id=row[1],
                    training_type=TrainingType(row[2]),
                    filename=row[3],
                    filetype=row[4],
                    uploaded_at=row[5],
                )
            return None


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
                "SELECT certificate_id, student_id, training_type, filename, filetype, uploaded_at FROM certificates ORDER BY uploaded_at DESC OFFSET %s LIMIT %s",
                (skip, limit),
            )

            return [
                Certificate(
                    certificate_id=row[0],
                    student_id=row[1],
                    training_type=TrainingType(row[2]),
                    filename=row[3],
                    filetype=row[4],
                    uploaded_at=row[5],
                )
                for row in cur.fetchall()
            ]


# Raw SQL operations for Decisions
def create_decision(
    certificate_id: UUID,
    ocr_output: Optional[str],
    decision: DecisionStatus,
    justification: str,
    assigned_reviewer: Optional[str] = None,
) -> Decision:
    """
    Create a new decision record.

    Args:
        certificate_id: Certificate's UUID
        ocr_output: OCR extracted text
        decision: Decision status
        justification: Explanation for the decision
        assigned_reviewer: Optional reviewer identifier

    Returns:
        Decision: Created decision object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            decision_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO decisions (decision_id, certificate_id, ocr_output, decision, justification, created_at, assigned_reviewer)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(decision_id),
                    str(certificate_id),
                    ocr_output,
                    decision.value,
                    justification,
                    now,
                    assigned_reviewer,
                ),
            )
            conn.commit()

            return Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ocr_output=ocr_output,
                decision=decision,
                justification=justification,
                created_at=now,
                assigned_reviewer=assigned_reviewer,
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
                "SELECT decision_id, certificate_id, ocr_output, decision, justification, created_at, assigned_reviewer FROM decisions WHERE decision_id = %s",
                (str(decision_id),),
            )

            row = cur.fetchone()
            if row:
                return Decision(
                    decision_id=row[0],
                    certificate_id=row[1],
                    ocr_output=row[2],
                    decision=DecisionStatus(row[3]),
                    justification=row[4],
                    created_at=row[5],
                    assigned_reviewer=row[6],
                )
            return None


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
                "SELECT decision_id, certificate_id, ocr_output, decision, justification, created_at, assigned_reviewer FROM decisions ORDER BY created_at DESC OFFSET %s LIMIT %s",
                (skip, limit),
            )

            return [
                Decision(
                    decision_id=row[0],
                    certificate_id=row[1],
                    ocr_output=row[2],
                    decision=DecisionStatus(row[3]),
                    justification=row[4],
                    created_at=row[5],
                    assigned_reviewer=row[6],
                )
                for row in cur.fetchall()
            ]


# Statistics functions
def get_statistics() -> dict:
    """
    Get system statistics.

    Returns:
        dict: Statistics about students, certificates, and decisions
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Get totals
            cur.execute("SELECT COUNT(*) FROM students")
            total_students = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM certificates")
            total_certificates = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM decisions")
            total_decisions = cur.fetchone()[0]

            # Get decision statistics
            cur.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'accepted'")
            accepted_decisions = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'rejected'")
            rejected_decisions = cur.fetchone()[0]

            # Get certificate statistics
            cur.execute(
                "SELECT COUNT(*) FROM certificates WHERE training_type = 'professional'"
            )
            professional_certificates = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM certificates WHERE training_type = 'general'"
            )
            general_certificates = cur.fetchone()[0]

            return {
                "total_students": total_students,
                "total_certificates": total_certificates,
                "total_decisions": total_decisions,
                "decisions": {
                    "accepted": accepted_decisions,
                    "rejected": rejected_decisions,
                    "acceptance_rate": (accepted_decisions / total_decisions * 100)
                    if total_decisions > 0
                    else 0,
                },
                "certificates": {
                    "professional": professional_certificates,
                    "general": general_certificates,
                },
            }
