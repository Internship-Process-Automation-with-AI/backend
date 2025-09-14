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
    AdditionalDocument,
    ApplicationSummary,
    Certificate,
    Decision,
    DecisionStatus,
    DetailedApplication,
    DocumentType,
    Reviewer,
    ReviewerDecision,
    Student,
    StudentWithCertificates,
    TrainingType,
    WorkType,
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
                SELECT certificate_id, student_id, training_type, filename, filetype, uploaded_at
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
                    uploaded_at=row[5],
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
    file_content: bytes,
    work_type: WorkType = WorkType.REGULAR,
) -> Certificate:
    """
    Create a new certificate record.

    Args:
        student_id: Student's UUID
        training_type: Type of training (GENERAL/PROFESSIONAL)
        filename: Original filename
        filetype: File type/extension
        file_content: File content as bytes
        work_type: Type of work (REGULAR/SELF_PACED)

    Returns:
        Certificate: Created certificate object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            certificate_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO certificates (certificate_id, student_id, training_type, work_type, filename, filetype, file_content, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(certificate_id),
                    str(student_id),
                    training_type.value,
                    work_type.value,
                    filename,
                    filetype,
                    file_content,
                    now,
                ),
            )
            conn.commit()
            return Certificate(
                certificate_id=certificate_id,
                student_id=student_id,
                training_type=training_type,
                work_type=work_type,
                filename=filename,
                filetype=filetype,
                uploaded_at=now,
                file_content=file_content,
            )


def get_certificate_by_id(certificate_id: UUID) -> Optional[Certificate]:
    """
    Get a certificate by its ID.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        Certificate: Certificate object if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, work_type, filename, filetype, file_content, ocr_output, uploaded_at
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
                work_type=WorkType(row[3]),
                filename=row[4],
                filetype=row[5],
                file_content=row[6],
                ocr_output=row[7],
                uploaded_at=row[8],
            )


def get_certificates(student_id: UUID) -> List[Certificate]:
    """
    Get all certificates for a student.

    Args:
        student_id: Student's UUID

    Returns:
        List[Certificate]: List of certificate objects
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, ocr_output, uploaded_at
                FROM certificates WHERE student_id = %s ORDER BY uploaded_at DESC
                """,
                (str(student_id),),
            )
            rows = cur.fetchall()
            certificates = []
            for row in rows:
                certificates.append(
                    Certificate(
                        certificate_id=UUID(row[0]),
                        student_id=UUID(row[1]),
                        training_type=TrainingType(row[2]),
                        filename=row[3],
                        filetype=row[4],
                        ocr_output=row[5],
                        uploaded_at=row[6],
                        file_content=None,  # Don't load file content for list view
                    )
                )
            return certificates


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
    total_working_hours: Optional[int] = None,
    credits_awarded: Optional[int] = None,
    training_duration: Optional[str] = None,
    training_institution: Optional[str] = None,
    degree_relevance: Optional[str] = None,
    supporting_evidence: Optional[str] = None,
    challenging_evidence: Optional[str] = None,
    recommendation: Optional[str] = None,
    ai_workflow_json: Optional[str] = None,
    company_validation_status: Optional[str] = None,
    company_validation_justification: Optional[str] = None,
    # Name validation parameter
    name_validation_match_result: Optional[str] = None,
    name_validation_explanation: Optional[str] = None,
) -> Decision:
    """
    Create a new decision record.

    Args:
        certificate_id: Certificate's UUID
        ai_decision: AI decision status
        ai_justification: Explanation for the decision
        total_working_hours: Total working hours from certificate
        credits_awarded: Credits awarded (ECTS)
        training_duration: Duration of training
        training_institution: Institution where training was conducted
        degree_relevance: How relevant the training is to the degree
        supporting_evidence: Supporting evidence for the decision
        challenging_evidence: Challenging evidence against the decision
        recommendation: AI recommendation summary
        ai_workflow_json: Complete AI workflow JSON output
        company_validation_status: Overall company validation status
        company_validation_justification: Company validation details and evidence
        name_validation_*: Name validation results from LLM

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
                    decision_id, certificate_id, ai_justification, ai_decision, created_at,
                    total_working_hours, credits_awarded, training_duration, training_institution,
                    degree_relevance, supporting_evidence, challenging_evidence, recommendation, ai_workflow_json,
                    company_validation_status, company_validation_justification,
                    name_validation_match_result, name_validation_explanation
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(decision_id),
                    str(certificate_id),
                    ai_justification,
                    ai_decision.value,
                    now,
                    total_working_hours,
                    credits_awarded,
                    training_duration,
                    training_institution,
                    degree_relevance,
                    supporting_evidence,
                    challenging_evidence,
                    recommendation,
                    ai_workflow_json,
                    company_validation_status,
                    company_validation_justification,
                    name_validation_match_result,
                    name_validation_explanation,
                ),
            )
            conn.commit()

            return Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ai_decision=ai_decision,
                ai_justification=ai_justification,
                created_at=now,
                student_comment=None,  # Always None during initial creation
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
                ai_workflow_json=ai_workflow_json,
                company_validation_status=company_validation_status,
                company_validation_justification=company_validation_justification,
                name_validation_match_result=name_validation_match_result,
                name_validation_explanation=name_validation_explanation,
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
                SELECT decision_id, certificate_id, ai_justification, ai_decision, created_at,
                       student_comment, reviewer_decision, reviewer_comment, reviewed_at,
                       total_working_hours, credits_awarded, training_duration, training_institution,
                       degree_relevance, supporting_evidence, challenging_evidence, recommendation
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
                ai_justification=row[2],
                ai_decision=DecisionStatus(row[3]),
                created_at=row[4],
                student_comment=row[5],
                reviewer_decision=ReviewerDecision(row[6]) if row[6] else None,
                reviewer_comment=row[7],
                reviewed_at=row[8],
                total_working_hours=row[9],
                credits_awarded=row[10],
                training_duration=row[11],
                training_institution=row[12],
                degree_relevance=row[13],
                supporting_evidence=row[14],
                challenging_evidence=row[15],
                recommendation=row[16],
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
                SELECT decision_id, certificate_id, ai_justification, ai_decision, created_at,
                       student_comment, reviewer_decision, reviewer_comment, reviewed_at,
                       total_working_hours, credits_awarded, training_duration, training_institution,
                       degree_relevance, supporting_evidence, challenging_evidence, recommendation
                FROM decisions ORDER BY created_at DESC OFFSET %s LIMIT %s
                """,
                (skip, limit),
            )
            rows = cur.fetchall()

            return [
                Decision(
                    decision_id=UUID(row[0]),
                    certificate_id=UUID(row[1]),
                    ai_justification=row[2],
                    ai_decision=DecisionStatus(row[3]),
                    created_at=row[4],
                    student_comment=row[5],
                    reviewer_decision=ReviewerDecision(row[6]) if row[6] else None,
                    reviewer_comment=row[7],
                    reviewed_at=row[8],
                    total_working_hours=row[9],
                    credits_awarded=row[10],
                    training_duration=row[11],
                    training_institution=row[12],
                    degree_relevance=row[13],
                    supporting_evidence=row[14],
                    challenging_evidence=row[15],
                    recommendation=row[16],
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
                       d.created_at, d.student_comment
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
                    student_comment=row[10],
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
                       d.created_at, d.student_comment
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
                    student_comment=row[10],
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
            # Get decision with all evaluation fields
            cur.execute(
                """
                SELECT decision_id, certificate_id, ai_justification, ai_decision, created_at,
                       student_comment, reviewer_decision, reviewer_comment, reviewed_at,
                       total_working_hours, credits_awarded, training_duration, training_institution,
                       degree_relevance, supporting_evidence, challenging_evidence, recommendation,
                       company_validation_status, company_validation_justification,
                       name_validation_match_result, name_validation_explanation
                FROM decisions WHERE certificate_id = %s
                """,
                (str(certificate_id),),
            )
            decision_row = cur.fetchone()
            if not decision_row:
                return None

            # Get certificate (including ocr_output if needed)
            cur.execute(
                """
                SELECT certificate_id, student_id, training_type, filename, filetype, uploaded_at
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
                ai_justification=decision_row[2],
                ai_decision=DecisionStatus(decision_row[3]),
                created_at=decision_row[4],
                student_comment=decision_row[5],
                reviewer_decision=ReviewerDecision(decision_row[6])
                if decision_row[6]
                else None,
                reviewer_comment=decision_row[7],
                reviewed_at=decision_row[8],
                total_working_hours=decision_row[9],
                credits_awarded=decision_row[10],
                training_duration=decision_row[11],
                training_institution=decision_row[12],
                degree_relevance=decision_row[13],
                supporting_evidence=decision_row[14],
                challenging_evidence=decision_row[15],
                recommendation=decision_row[16],
                company_validation_status=decision_row[17],
                company_validation_justification=decision_row[18],
                name_validation_match_result=(
                    decision_row[19] if len(decision_row) > 19 else None
                ),
                name_validation_explanation=(
                    decision_row[20] if len(decision_row) > 20 else None
                ),
            )

            certificate = Certificate(
                certificate_id=UUID(certificate_row[0]),
                student_id=UUID(certificate_row[1]),
                training_type=TrainingType(certificate_row[2]),
                filename=certificate_row[3],
                filetype=certificate_row[4],
                uploaded_at=certificate_row[5],
            )

            student = Student(
                student_id=UUID(student_row[0]),
                email=student_row[1],
                degree=student_row[2],
                first_name=student_row[3],
                last_name=student_row[4],
            )

            # Get additional documents for self-paced work
            additional_docs = get_additional_documents(certificate_id)

            return DetailedApplication(
                decision=decision,
                certificate=certificate,
                student=student,
                additional_documents=additional_docs if additional_docs else None,
            )


def update_decision_review(
    certificate_id: UUID,
    reviewer_comment: Optional[str],
    reviewer_decision: ReviewerDecision,
    student_comment: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Add a reviewer comment and mark the decision as *REVIEWED*.

    Args:
        certificate_id: Certificate's UUID.
        reviewer_comment: Optional free-text comment from the human reviewer.
        reviewer_decision: PASS or FAIL decision.
        student_comment: Optional student comment to update at the same time.

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

                if student_comment is not None:
                    cur.execute(
                        """
                        UPDATE decisions 
                        SET student_comment = %s
                        WHERE certificate_id = %s
                        """,
                        (student_comment, str(certificate_id)),
                    )

                conn.commit()
                return True, None

    except Exception as e:
        error_msg = f"Database error in update_decision_review: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def add_student_comment_and_reviewer(
    certificate_id: UUID, student_comment: str, reviewer_id: Optional[UUID] = None
) -> bool:
    """
    Add student comment and optionally reviewer ID to a decision.

    Args:
        certificate_id: Certificate's UUID
        student_comment: Student's comment
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
                    SET student_comment = %s, reviewer_id = %s
                    WHERE certificate_id = %s
                    """,
                    (student_comment, str(reviewer_id), str(certificate_id)),
                )
            else:
                cur.execute(
                    """
                    UPDATE decisions 
                    SET student_comment = %s
                    WHERE certificate_id = %s
                    """,
                    (student_comment, str(certificate_id)),
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
    position: Optional[str] = None,
    department: Optional[str] = None,
):
    """
    Create a new reviewer in the database.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reviewers (email, first_name, last_name, position, department)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING reviewer_id
                """,
                (email, first_name, last_name, position, department),
            )
            reviewer_id = cur.fetchone()[0]
            conn.commit()
            return reviewer_id


def get_reviewer_by_email(email: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT reviewer_id, email, first_name, last_name, position, department FROM reviewers WHERE email = %s",
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
                position=row[4],
                department=row[5],
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
                "SELECT reviewer_id, email, first_name, last_name, position, department FROM reviewers ORDER BY first_name, last_name"
            )
            rows = cur.fetchall()
            return [
                Reviewer(
                    reviewer_id=UUID(row[0]),
                    email=row[1],
                    first_name=row[2],
                    last_name=row[3],
                    position=row[4],
                    department=row[5],
                )
                for row in rows
            ]


def create_sample_reviewers():
    sample_reviewers = [
        (
            "laura.koskinen@oamk.fi",
            "Laura",
            "Koskinen",
            "Senior Lecturer",
            "Health Sciences",
        ),
        ("jukka.virtanen@oamk.fi", "Jukka", "Virtanen", "Program Director", "Nursing"),
        (
            "emilia.makela@oamk.fi",
            "Emilia",
            "Mäkelä",
            "Faculty Coordinator",
            "Midwifery",
        ),
        (
            "antti.lehtinen@oamk.fi",
            "Antti",
            "Lehtinen",
            "Department Head",
            "Health Care",
        ),
        (
            "sanna.nieminen@oamk.fi",
            "Sanna",
            "Nieminen",
            "Academic Coordinator",
            "Student Services",
        ),
    ]
    for email, first_name, last_name, position, department in sample_reviewers:
        if not get_reviewer_by_email(email):
            create_reviewer(email, first_name, last_name, position, department)


def get_certificates_by_reviewer_id(reviewer_id: UUID) -> List[DetailedApplication]:
    """
    Get all certificates assigned to a reviewer, including appeal assignments.

    Args:
        reviewer_id: Reviewer's UUID

    Returns:
        List[DetailedApplication]: List of detailed applications assigned to the reviewer
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    d.decision_id, d.certificate_id, d.ai_justification,
                    d.ai_decision, d.created_at, d.student_comment, d.reviewer_decision,
                       d.reviewer_comment, d.reviewed_at,
                    d.reviewer_id,
                    d.total_working_hours, d.credits_awarded, d.training_duration,
                    d.training_institution, d.degree_relevance, d.supporting_evidence,
                    d.challenging_evidence, d.recommendation,
                       c.student_id, c.training_type, c.filename, c.filetype, c.uploaded_at,
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

            applications: List[DetailedApplication] = []
            for row in rows:
                try:
                    # Basic sanity checks
                    if not row[0] or not row[1] or not row[18]:
                        logger.warning(f"Skipping row with NULL values: {row}")
                        continue

                    decision = Decision(
                        decision_id=UUID(row[0]),
                        certificate_id=UUID(row[1]),
                        ai_justification=row[2] or "",
                        ai_decision=DecisionStatus(row[3])
                        if row[3]
                        else DecisionStatus.REJECTED,
                        created_at=row[4],
                        student_comment=row[5],
                        reviewer_decision=ReviewerDecision(row[6]) if row[6] else None,
                        reviewer_comment=row[7],
                        reviewed_at=row[8],
                        reviewer_id=UUID(row[9]) if row[9] else None,
                        total_working_hours=row[10],
                        credits_awarded=row[11],
                        training_duration=row[12],
                        training_institution=row[13],
                        degree_relevance=row[14],
                        supporting_evidence=row[15],
                        challenging_evidence=row[16],
                        recommendation=row[17],
                    )

                    certificate = Certificate(
                        certificate_id=UUID(row[1]),
                        student_id=UUID(row[18]),
                        training_type=TrainingType(row[19])
                        if row[19]
                        else TrainingType.GENERAL,
                        filename=row[20] or "",
                        filetype=row[21] or "",
                        uploaded_at=row[22],
                        ocr_output=None,
                    )

                    student = Student(
                        student_id=UUID(row[18]),
                        email=row[23] or "",
                        degree=row[24] or "",
                        first_name=row[25],
                        last_name=row[26],
                    )

                    applications.append(
                        DetailedApplication(
                            decision=decision,
                            certificate=certificate,
                            student=student,
                        )
                    )
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing row {row}: {e}")
                    continue

            return applications


# Raw SQL operations for Student Comments (simplified appeal process)


def add_student_comment(certificate_id: UUID, student_comment: str) -> bool:
    """
    Add student comment to a decision record.

    Args:
        certificate_id: Certificate's UUID
        student_comment: Student's comment/appeal reason

    Returns:
        bool: True if comment was added successfully
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE decisions 
                SET student_comment = %s
                WHERE certificate_id = %s
                """,
                (
                    student_comment,
                    str(certificate_id),
                ),
            )
            conn.commit()
            return True


def get_student_comment_by_certificate_id(certificate_id: UUID) -> Optional[str]:
    """
    Get student comment by certificate ID from decisions table.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        Optional[str]: Student comment if found, None otherwise
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT student_comment
                FROM decisions WHERE certificate_id = %s AND student_comment IS NOT NULL
                """,
                (str(certificate_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return row[0]


# Raw SQL operations for Additional Documents
def create_additional_document(
    certificate_id: UUID,
    document_type: DocumentType,
    filename: str,
    filetype: str,
    file_content: bytes,
) -> AdditionalDocument:
    """
    Create a new additional document record.

    Args:
        certificate_id: Certificate's UUID this document belongs to
        document_type: Type of document (HOUR_DOCUMENTATION/PROJECT_DETAILS)
        filename: Original filename
        filetype: File type/extension
        file_content: File content as bytes

    Returns:
        AdditionalDocument: Created additional document object
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            document_id = uuid4()
            now = datetime.now()

            cur.execute(
                """
                INSERT INTO additional_documents (document_id, certificate_id, document_type, filename, filetype, file_content, uploaded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(document_id),
                    str(certificate_id),
                    document_type.value,
                    filename,
                    filetype,
                    file_content,
                    now,
                ),
            )
            conn.commit()
            return AdditionalDocument(
                document_id=document_id,
                certificate_id=certificate_id,
                document_type=document_type,
                filename=filename,
                filetype=filetype,
                uploaded_at=now,
                file_content=file_content,
            )


def get_additional_documents(certificate_id: UUID) -> List[AdditionalDocument]:
    """
    Get all additional documents for a certificate.

    Args:
        certificate_id: Certificate's UUID

    Returns:
        List[AdditionalDocument]: List of additional document objects
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, certificate_id, document_type, filename, filetype, file_content, ocr_output, uploaded_at
                FROM additional_documents WHERE certificate_id = %s
                ORDER BY uploaded_at
                """,
                (str(certificate_id),),
            )
            rows = cur.fetchall()

            documents = []
            for row in rows:
                documents.append(
                    AdditionalDocument(
                        document_id=UUID(row[0]),
                        certificate_id=UUID(row[1]),
                        document_type=DocumentType(row[2]),
                        filename=row[3],
                        filetype=row[4],
                        file_content=row[5],
                        ocr_output=row[6],
                        uploaded_at=row[7],
                    )
                )

            return documents


def get_student_identity_by_certificate(certificate_id: UUID) -> Optional[dict]:
    """
    Get student identity information by certificate ID for name validation.

    Args:
        certificate_id: UUID of the certificate

    Returns:
        Dictionary with student identity info or None if not found
        {
            "first_name": str,
            "last_name": str,
            "email": str,
            "full_name": str  # Best-effort full name combination
        }
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Join certificates and students tables to get student info
                cur.execute(
                    """
                    SELECT s.first_name, s.last_name, s.email
                    FROM students s
                    INNER JOIN certificates c ON s.student_id = c.student_id
                    WHERE c.certificate_id = %s
                    """,
                    (str(certificate_id),),
                )

                result = cur.fetchone()
                if not result:
                    logger.warning(
                        f"No student found for certificate ID: {certificate_id}"
                    )
                    return None

                first_name, last_name, email = result

                # Create best-effort full name
                name_parts = []
                if first_name and first_name.strip():
                    name_parts.append(first_name.strip())
                if last_name and last_name.strip():
                    name_parts.append(last_name.strip())

                full_name = " ".join(name_parts) if name_parts else "Unknown"

                return {
                    "first_name": first_name or "",
                    "last_name": last_name or "",
                    "email": email or "",
                    "full_name": full_name,
                }

    except Exception as e:
        logger.error(
            f"Error getting student identity for certificate {certificate_id}: {e}"
        )
        return None
