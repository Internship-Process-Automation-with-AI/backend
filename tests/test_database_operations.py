"""
Simple Database Operations Tests

Basic tests for database operations focusing on core functionality.
"""

from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import psycopg2
import pytest

from src.database.database import (
    create_certificate,
    get_all_reviewers,
    get_certificate_by_id,
    get_pending_applications,
    get_reviewer_by_email,
    get_student_by_email,
    get_student_by_id,
)
from src.database.models import TrainingType


def setup_mock_cursor():
    """Helper function to set up a mock cursor with context manager support."""
    mock_cursor = Mock()
    mock_cursor.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor.__exit__ = Mock(return_value=None)
    return mock_cursor


class TestDatabaseOperations:
    """Basic database operations tests."""

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_email_success(self, mock_get_db_connection):
        """Test successful student retrieval by email."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(student_id),  # Convert to string
            "test@students.oamk.fi",
            "Bachelor of Engineering",
            "John",
            "Doe",
        )

        result = get_student_by_email("test@students.oamk.fi")

        assert result is not None
        assert result.email == "test@students.oamk.fi"
        assert result.degree == "Bachelor of Engineering"
        assert result.first_name == "John"
        assert result.last_name == "Doe"

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_email_not_found(self, mock_get_db_connection):
        """Test student retrieval when not found."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        mock_cursor.fetchone.return_value = None

        result = get_student_by_email("nonexistent@students.oamk.fi")

        assert result is None

    @patch("src.database.database.get_db_connection")
    def test_get_student_by_id_success(self, mock_get_db_connection):
        """Test successful student retrieval by ID."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(student_id),  # Convert to string
            "test@students.oamk.fi",
            "Bachelor of Engineering",
            "John",
            "Doe",
        )

        result = get_student_by_id(student_id)

        assert result is not None
        assert result.email == "test@students.oamk.fi"

    @patch("src.database.database.get_db_connection")
    def test_get_reviewer_by_email_success(self, mock_get_db_connection):
        """Test successful reviewer retrieval by email."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        reviewer_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(reviewer_id),  # Convert to string
            "reviewer@oamk.fi",
            "Jane",
            "Smith",
        )

        result = get_reviewer_by_email("reviewer@oamk.fi")

        assert result is not None
        assert result.email == "reviewer@oamk.fi"
        assert result.first_name == "Jane"
        assert result.last_name == "Smith"
        # Note: get_reviewer_by_email doesn't return position and department

    @patch("src.database.database.get_db_connection")
    def test_get_all_reviewers_success(self, mock_get_db_connection):
        """Test successful retrieval of all reviewers."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        reviewer1_id = uuid4()
        reviewer2_id = uuid4()
        mock_cursor.fetchall.return_value = [
            (
                str(reviewer1_id),
                "reviewer1@oamk.fi",
                "Jane",
                "Smith",
                "Engineer",
                "Engineering",
            ),
            (
                str(reviewer2_id),
                "reviewer2@oamk.fi",
                "John",
                "Doe",
                "Manager",
                "Management",
            ),
        ]

        result = get_all_reviewers()

        assert len(result) == 2
        assert result[0].email == "reviewer1@oamk.fi"
        assert result[1].email == "reviewer2@oamk.fi"

    @patch("src.database.database.get_db_connection")
    def test_create_certificate_success(self, mock_get_db_connection):
        """Test successful certificate creation."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        certificate_id = uuid4()
        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(certificate_id),  # Convert to string
            str(student_id),  # Convert to string
            "PROFESSIONAL",
            "test.pdf",
            "pdf",
            datetime.now(),
            b"test content",
            None,  # ocr_output
        )

        result = create_certificate(
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
        )

        assert result is not None
        assert result.training_type == TrainingType.PROFESSIONAL
        assert result.filename == "test.pdf"

    @patch("src.database.database.get_db_connection")
    def test_get_certificate_by_id_success(self, mock_get_db_connection):
        """Test successful certificate retrieval by ID."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        certificate_id = uuid4()
        student_id = uuid4()
        mock_cursor.fetchone.return_value = (
            str(certificate_id),  # Convert to string
            str(student_id),  # Convert to string
            "PROFESSIONAL",
            "test.pdf",
            "pdf",
            datetime.now(),
            b"test content",
            None,  # ocr_output
        )

        result = get_certificate_by_id(certificate_id)

        assert result is not None
        assert result.training_type == TrainingType.PROFESSIONAL
        assert result.filename == "test.pdf"

    @patch("src.database.database.get_db_connection")
    def test_get_pending_applications_success(self, mock_get_db_connection):
        """Test successful retrieval of pending applications."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        decision_id = uuid4()
        certificate_id = uuid4()
        mock_cursor.fetchall.return_value = [
            (
                str(decision_id),  # decision_id
                str(certificate_id),  # certificate_id
                "test@students.oamk.fi",  # email
                "Bachelor of Engineering",  # degree
                "test.pdf",  # filename
                "PROFESSIONAL",  # training_type
                "ACCEPTED",  # ai_decision
                None,  # reviewer_decision
                datetime.now(),  # uploaded_at
                datetime.now(),  # created_at
                None,  # student_feedback
            )
        ]

        result = get_pending_applications()

        assert len(result) == 1
        application = result[0]
        assert (
            application.student_name == "Test"
        )  # Extracted from email: test@students.oamk.fi -> test -> Test
        assert application.student_email == "test@students.oamk.fi"
        assert application.training_type == TrainingType.PROFESSIONAL


class TestDatabaseErrorHandling:
    """Test database error handling scenarios."""

    @patch("src.database.database.get_db_connection")
    def test_connection_error_handling(self, mock_get_db_connection):
        """Test handling of connection errors."""
        mock_get_db_connection.side_effect = psycopg2.OperationalError(
            "Connection failed"
        )

        with pytest.raises(psycopg2.OperationalError):
            get_student_by_email("test@students.oamk.fi")

    @patch("src.database.database.get_db_connection")
    def test_database_query_error(self, mock_get_db_connection):
        """Test handling of database query errors."""
        mock_connection = Mock()
        mock_cursor = setup_mock_cursor()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_connection
        mock_get_db_connection.return_value.__exit__.return_value = None

        mock_cursor.execute.side_effect = psycopg2.Error("Query failed")

        with pytest.raises(psycopg2.Error):
            get_student_by_email("test@students.oamk.fi")
