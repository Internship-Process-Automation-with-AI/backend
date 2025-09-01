"""
Test file for database operations.

Tests the database connection management, CRUD operations for students, certificates, decisions, and reviewers.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import psycopg2
import pytest

from src.database.database import (
    add_student_comment,
    add_student_comment_and_reviewer,
    check_database_health,
    create_certificate,
    create_database_if_not_exists,
    create_decision,
    create_student,
    delete_certificate,
    get_all_reviewers,
    get_certificate_by_id,
    get_certificates_by_reviewer_id,
    get_database_info,
    get_db_connection,
    get_detailed_application,
    get_reviewer_by_email,
    get_student_by_email,
    get_student_by_id,
    get_student_comment_by_certificate_id,
    get_student_with_certificates,
    test_database_connection,
    update_decision_review,
)
from src.database.models import (
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


def create_mock_context_manager(return_value):
    """Helper function to create a mock context manager."""
    mock = Mock()
    mock.__enter__ = Mock(return_value=return_value)
    mock.__exit__ = Mock(return_value=None)
    return mock


class TestDatabaseConnection:
    """Test database connection management."""

    @patch("src.database.database.psycopg2.connect")
    def test_get_db_connection_success(self, mock_connect):
        """Test successful database connection."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_connect.return_value = mock_conn

        with get_db_connection() as conn:
            assert conn == mock_conn

        mock_connect.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.database.database.psycopg2.connect")
    def test_get_db_connection_error(self, mock_connect):
        """Test database connection error handling."""
        mock_connect.side_effect = psycopg2.Error("Connection failed")

        with pytest.raises(psycopg2.Error):
            with get_db_connection():
                pass

    @patch("src.database.database.psycopg2.connect")
    def test_get_db_connection_rollback_on_error(self, mock_connect):
        """Test that rollback is called on connection error."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn

        # Set up context manager for cursor that raises an error
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        # The error should be raised when we try to use the cursor, not just when setting up
        mock_cursor.execute.side_effect = psycopg2.Error("Cursor error")

        with pytest.raises(psycopg2.Error):
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1"
                    )  # Actually execute something to trigger the error

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.database.database.psycopg2.connect")
    def test_test_database_connection_success(self, mock_connect):
        """Test successful database connection test."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_cursor.execute.return_value = None
        mock_connect.return_value = mock_conn

        result = test_database_connection()
        assert result is True

    @patch("src.database.database.psycopg2.connect")
    def test_test_database_connection_failure(self, mock_connect):
        """Test failed database connection test."""
        mock_connect.side_effect = Exception("Connection failed")

        result = test_database_connection()
        assert result is False

    @patch("src.database.database.psycopg2.connect")
    def test_get_database_info_success(self, mock_connect):
        """Test successful database info retrieval."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = ("PostgreSQL 15.0",)
        mock_connect.return_value = mock_conn

        result = get_database_info()
        assert result["status"] == "connected"
        assert "PostgreSQL" in result["version"]

    @patch("src.database.database.psycopg2.connect")
    def test_get_database_info_error(self, mock_connect):
        """Test database info retrieval error."""
        mock_connect.side_effect = Exception("Connection failed")

        result = get_database_info()
        assert result["status"] == "error"
        assert "Connection failed" in result["error"]

    @patch("src.database.database.psycopg2.connect")
    def test_check_database_health_success(self, mock_connect):
        """Test successful database health check."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)

        # Mock table existence check - need to mock the actual SQL queries
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [
            ("students",),
            ("certificates",),
            ("decisions",),
        ]  # Tables

        # Mock the count queries
        mock_cursor.fetchone.side_effect = [
            (5,),
            (10,),
            (8,),
        ]  # Students, certificates, decisions counts

        mock_connect.return_value = mock_conn

        result = check_database_health()
        assert result["status"] == "healthy"
        assert "students" in result["tables"]
        assert "certificates" in result["tables"]
        assert "decisions" in result["tables"]
        assert result["counts"]["students"] == 5
        assert result["counts"]["certificates"] == 10
        assert result["counts"]["decisions"] == 8
        assert "timestamp" in result

    @patch("src.database.database.psycopg2.connect")
    def test_check_database_health_error(self, mock_connect):
        """Test database health check error."""
        mock_connect.side_effect = Exception("Connection failed")

        result = check_database_health()
        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]


class TestStudentOperations:
    """Test student database operations."""

    @patch("src.database.database.get_db_connection")
    def test_create_student_success(self, mock_get_connection):
        """Test successful student creation."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        result = create_student(
            email="test@students.oamk.fi",
            degree="Computer Science",
            first_name="John",
            last_name="Doe",
        )

        assert isinstance(result, Student)
        assert result.email == "test@students.oamk.fi"
        assert result.degree == "Computer Science"
        assert result.first_name == "John"
        assert result.last_name == "Doe"
        assert isinstance(result.student_id, UUID)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("src.database.database.get_db_connection")
    def test_create_student_minimal_fields(self, mock_get_connection):
        """Test student creation with minimal fields."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        result = create_student(email="test@students.oamk.fi", degree="Engineering")

        assert isinstance(result, Student)
        assert result.email == "test@students.oamk.fi"
        assert result.degree == "Engineering"
        assert result.first_name is None
        assert result.last_name is None

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_email_found(self, mock_get_connection):
        """Test getting student by email when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(student_id),
            "test@students.oamk.fi",
            "Computer Science",
            "John",
            "Doe",
        )

        result = get_student_by_email("test@students.oamk.fi")

        assert isinstance(result, Student)
        assert result.student_id == student_id
        assert result.email == "test@students.oamk.fi"
        assert result.degree == "Computer Science"
        assert result.first_name == "John"
        assert result.last_name == "Doe"

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_email_not_found(self, mock_get_connection):
        """Test getting student by email when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_student_by_email("nonexistent@students.oamk.fi")

        assert result is None

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_id_found(self, mock_get_connection):
        """Test getting student by ID when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(student_id),
            "test@students.oamk.fi",
            "Computer Science",
            "John",
            "Doe",
        )

        result = get_student_by_id(student_id)

        assert isinstance(result, Student)
        assert result.student_id == student_id
        assert result.email == "test@students.oamk.fi"
        assert result.degree == "Computer Science"

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_id_not_found(self, mock_get_connection):
        """Test getting student by ID when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_student_by_id(uuid4())

        assert result is None

    @patch("src.database.database.get_db_connection")
    def test_get_student_with_certificates_found(self, mock_get_connection):
        """Test getting student with certificates when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        student_id = uuid4()
        certificate_id = uuid4()
        uploaded_at = datetime.now()

        # Mock student data - first call returns student, second call returns certificates
        mock_cursor.fetchone.return_value = (
            str(student_id),
            "test@students.oamk.fi",
            "Computer Science",
            "John",
            "Doe",
        )
        mock_cursor.fetchall.return_value = [
            (
                str(certificate_id),
                str(student_id),
                "GENERAL",
                "test.pdf",
                "pdf",
                uploaded_at,
            )  # Certificates
        ]

        result = get_student_with_certificates(student_id)

        assert isinstance(result, StudentWithCertificates)
        # The function returns a StudentWithCertificates with direct attributes, not nested objects
        assert result.student_id == str(
            student_id
        )  # The function returns student_id as string, not UUID
        assert result.email == "test@students.oamk.fi"
        assert len(result.certificates) == 1
        assert result.certificates[0].certificate_id == certificate_id
        assert result.certificates[0].filename == "test.pdf"

    @patch("src.database.database.get_db_connection")
    def test_get_student_with_certificates_not_found(self, mock_get_connection):
        """Test getting student with certificates when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_student_with_certificates(uuid4())

        assert result is None


class TestCertificateOperations:
    """Test certificate database operations."""

    @patch("src.database.database.get_db_connection")
    def test_create_certificate_success(self, mock_get_connection):
        """Test successful certificate creation."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        student_id = uuid4()
        file_content = b"test file content"

        result = create_certificate(
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="test.pdf",
            filetype="pdf",
            file_content=file_content,
        )

        assert isinstance(result, Certificate)
        assert result.student_id == student_id
        assert result.training_type == TrainingType.GENERAL
        assert result.filename == "test.pdf"
        assert result.filetype == "pdf"
        assert result.file_content == file_content
        assert isinstance(result.certificate_id, UUID)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("src.database.database.get_db_connection")
    def test_get_certificate_by_id_found(self, mock_get_connection):
        """Test getting certificate by ID when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        mock_cursor.fetchone.return_value = (
            str(certificate_id),
            str(student_id),
            "GENERAL",
            "test.pdf",
            "pdf",
            b"file content",
            "OCR text",
            uploaded_at,
        )

        result = get_certificate_by_id(certificate_id)

        assert isinstance(result, Certificate)
        assert result.certificate_id == certificate_id
        assert result.student_id == student_id
        assert result.filename == "test.pdf"
        assert result.file_content == b"file content"
        assert result.ocr_output == "OCR text"

    @patch("src.database.database.get_db_connection")
    def test_get_certificate_by_id_not_found(self, mock_get_connection):
        """Test getting certificate by ID when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_certificate_by_id(uuid4())

        assert result is None


class TestDecisionOperations:
    """Test decision database operations."""

    @patch("src.database.database.get_db_connection")
    def test_create_decision_success(self, mock_get_connection):
        """Test successful decision creation."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()

        result = create_decision(
            certificate_id=certificate_id,
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            ai_workflow_json='{"test": "data"}',
        )

        assert isinstance(result, Decision)
        assert result.certificate_id == certificate_id
        assert result.ai_justification == "Test justification"
        assert result.ai_decision == DecisionStatus.ACCEPTED
        assert result.ai_workflow_json == '{"test": "data"}'
        assert isinstance(result.decision_id, UUID)

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("src.database.database.get_db_connection")
    def test_get_detailed_application_found(self, mock_get_connection):
        """Test getting detailed application when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        student_id = uuid4()
        decision_id = uuid4()
        uploaded_at = datetime.now()
        created_at = datetime.now()

        # Mock the complex query result - this matches the actual SQL structure
        # The function expects separate rows for decision, certificate, and student
        mock_cursor.fetchone.side_effect = [
            # Decision row - includes all 19 columns including company validation
            (
                str(decision_id),  # decision_id (0)
                str(certificate_id),  # certificate_id (1)
                "Test justification",  # ai_justification (2)
                "ACCEPTED",  # ai_decision (3)
                created_at,  # created_at (4)
                None,  # student_comment (5)
                None,  # reviewer_decision (6)
                None,  # reviewer_comment (7)
                None,  # reviewed_at (8)
                None,  # total_working_hours (9)
                None,  # credits_awarded (10)
                None,  # training_duration (11)
                None,  # training_institution (12)
                None,  # degree_relevance (13)
                None,  # supporting_evidence (14)
                None,  # challenging_evidence (15)
                None,  # recommendation (16)
                "UNVERIFIED",  # company_validation_status (17)
                None,  # company_validation_justification (18)
            ),
            # Certificate row
            (
                str(certificate_id),
                str(student_id),
                "GENERAL",
                "test.pdf",
                "pdf",
                uploaded_at,
            ),
            # Student row
            (
                str(student_id),
                "test@students.oamk.fi",
                "Computer Science",
                "John",
                "Doe",
            ),
        ]

        result = get_detailed_application(certificate_id)

        assert isinstance(result, DetailedApplication)
        assert result.certificate.certificate_id == certificate_id
        assert result.decision.decision_id == decision_id
        assert result.student.first_name == "John"
        assert result.student.last_name == "Doe"


class TestReviewerOperations:
    """Test reviewer database operations."""

    @patch("src.database.database.get_db_connection")
    def test_get_all_reviewers_success(self, mock_get_connection):
        """Test getting all reviewers successfully."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        reviewer1_id = uuid4()
        reviewer2_id = uuid4()

        mock_cursor.fetchall.return_value = [
            (str(reviewer1_id), "reviewer1@oamk.fi", "John", "Smith", "Lecturer", "CS"),
            (
                str(reviewer2_id),
                "reviewer2@oamk.fi",
                "Jane",
                "Doe",
                "Professor",
                "Engineering",
            ),
        ]

        result = get_all_reviewers()

        assert len(result) == 2
        assert result[0].email == "reviewer1@oamk.fi"
        assert result[0].first_name == "John"
        assert result[0].last_name == "Smith"
        assert result[1].email == "reviewer2@oamk.fi"
        assert result[1].first_name == "Jane"
        assert result[1].last_name == "Doe"

    @patch("src.database.database.get_db_connection")
    def test_get_reviewer_by_email_found(self, mock_get_connection):
        """Test getting reviewer by email when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        reviewer_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(reviewer_id),
            "reviewer@oamk.fi",
            "John",
            "Smith",
            "Lecturer",
            "Computer Science",
        )

        result = get_reviewer_by_email("reviewer@oamk.fi")

        assert isinstance(result, Reviewer)
        assert result.reviewer_id == reviewer_id
        assert result.email == "reviewer@oamk.fi"
        assert result.first_name == "John"
        assert result.last_name == "Smith"
        assert result.position == "Lecturer"
        assert result.department == "Computer Science"

    @patch("src.database.database.get_db_connection")
    def test_get_reviewer_by_email_not_found(self, mock_get_connection):
        """Test getting reviewer by email when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_reviewer_by_email("nonexistent@oamk.fi")

        assert result is None


class TestCommentAndReviewOperations:
    """Test comment and review operations."""

    @patch("src.database.database.get_db_connection")
    def test_add_student_comment_success(self, mock_get_connection):
        """Test successful student comment addition."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        comment = "I disagree with this decision"

        result = add_student_comment(certificate_id, comment)

        assert result is True
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("src.database.database.get_db_connection")
    def test_update_decision_review_success(self, mock_get_connection):
        """Test successful decision review update."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        reviewer_decision = ReviewerDecision.PASS_
        reviewer_comment = "Approved after review"

        # Mock the first query that checks if certificate exists
        mock_cursor.fetchone.return_value = (None,)  # No existing reviewer decision

        result, error = update_decision_review(
            certificate_id=certificate_id,
            reviewer_comment=reviewer_comment,
            reviewer_decision=reviewer_decision,
        )

        assert result is True
        assert error is None
        # The function makes 2 SQL operations: one UPDATE and potentially one more if student_comment is provided
        assert mock_cursor.execute.call_count >= 1  # At least one SQL operation
        mock_conn.commit.assert_called_once()

    @patch("src.database.database.get_db_connection")
    def test_delete_certificate_success(self, mock_get_connection):
        """Test successful certificate deletion."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()

        # Mock rowcount to simulate successful deletion
        mock_cursor.rowcount = 1

        result = delete_certificate(certificate_id)

        assert result is True
        assert (
            mock_cursor.execute.call_count == 2
        )  # Two SQL operations (delete decision, delete certificate)
        mock_conn.commit.assert_called_once()


class TestUtilityOperations:
    """Test utility database operations."""

    @patch("src.database.database.get_db_connection")
    def test_get_certificates_by_reviewer_id_success(self, mock_get_connection):
        """Test getting certificates by reviewer ID successfully."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        reviewer_id = uuid4()
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()
        created_at = datetime.now()

        # Mock the complex query result - this matches the actual SQL structure
        mock_cursor.fetchall.return_value = [
            (
                str(uuid4()),  # decision_id
                str(certificate_id),  # certificate_id
                "Test justification",  # ai_justification
                "ACCEPTED",  # ai_decision
                created_at,  # created_at
                None,  # student_comment
                None,  # reviewer_decision
                None,  # reviewer_comment
                None,  # reviewed_at
                str(reviewer_id),  # reviewer_id
                None,  # total_working_hours
                None,  # credits_awarded
                None,  # training_duration
                None,  # training_institution
                None,  # degree_relevance
                None,  # supporting_evidence
                None,  # challenging_evidence
                None,  # recommendation
                str(student_id),  # student_id
                "GENERAL",  # training_type
                "test.pdf",  # filename
                "pdf",  # filetype
                uploaded_at,  # uploaded_at
                "test@students.oamk.fi",  # email
                "Computer Science",  # degree
                "John",  # first_name
                "Doe",  # last_name
            )
        ]

        result = get_certificates_by_reviewer_id(reviewer_id)

        assert len(result) == 1
        assert result[0].certificate.certificate_id == certificate_id
        assert result[0].certificate.filename == "test.pdf"

    @patch("src.database.database.get_db_connection")
    def test_get_student_comment_by_certificate_id_found(self, mock_get_connection):
        """Test getting student comment by certificate ID when found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        mock_cursor.fetchone.return_value = ("I disagree with this decision",)

        result = get_student_comment_by_certificate_id(certificate_id)

        assert result == "I disagree with this decision"

    @patch("src.database.database.get_db_connection")
    def test_get_student_comment_by_certificate_id_not_found(self, mock_get_connection):
        """Test getting student comment by certificate ID when not found."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.fetchone.return_value = None

        result = get_student_comment_by_certificate_id(uuid4())

        assert result is None

    @patch("src.database.database.get_db_connection")
    def test_add_student_comment_and_reviewer_success(self, mock_get_connection):
        """Test successful student comment and reviewer assignment."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        certificate_id = uuid4()
        reviewer_id = uuid4()
        comment = "I disagree with this decision"

        # Mock rowcount to simulate successful update
        mock_cursor.rowcount = 1

        result = add_student_comment_and_reviewer(certificate_id, comment, reviewer_id)

        assert result is True
        assert mock_cursor.execute.call_count == 1  # One SQL operation
        mock_conn.commit.assert_called_once()


class TestDatabaseErrorHandling:
    """Test database error handling scenarios."""

    @patch("src.database.database.get_db_connection")
    def test_create_student_database_error(self, mock_get_connection):
        """Test student creation with database error."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context managers
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_get_connection.return_value = create_mock_context_manager(mock_conn)

        mock_cursor.execute.side_effect = psycopg2.Error("Database error")

        with pytest.raises(psycopg2.Error):
            create_student(email="test@students.oamk.fi", degree="Computer Science")

    @patch("src.database.database.get_db_connection")
    def test_get_student_connection_error(self, mock_get_connection):
        """Test getting student with connection error."""
        mock_get_connection.side_effect = psycopg2.Error("Connection failed")

        with pytest.raises(psycopg2.Error):
            get_student_by_email("test@students.oamk.fi")

    @patch("src.database.database.psycopg2.connect")
    def test_create_database_if_not_exists_success(self, mock_connect):
        """Test successful database creation."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_connect.return_value = create_mock_context_manager(mock_conn)

        # Mock that database doesn't exist initially
        mock_cursor.fetchone.return_value = None

        result = create_database_if_not_exists()

        assert result is True
        mock_cursor.execute.assert_called()

    @patch("src.database.database.psycopg2.connect")
    def test_create_database_if_not_exists_already_exists(self, mock_connect):
        """Test database creation when database already exists."""
        mock_conn = Mock()
        mock_cursor = Mock()

        # Set up context manager for cursor
        mock_conn.cursor.return_value = create_mock_context_manager(mock_cursor)
        mock_connect.return_value = create_mock_context_manager(mock_conn)

        # Mock that database already exists
        mock_cursor.fetchone.return_value = (True,)

        result = create_database_if_not_exists()

        assert result is True
        # Should not call CREATE DATABASE
        mock_cursor.execute.assert_called_once()

    @patch("src.database.database.psycopg2.connect")
    def test_create_database_if_not_exists_error(self, mock_connect):
        """Test database creation with error."""
        mock_connect.side_effect = Exception("Connection failed")

        result = create_database_if_not_exists()

        assert result is False
