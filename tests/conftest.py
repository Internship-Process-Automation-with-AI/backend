"""
Pytest configuration and fixtures for OAMK Backend tests.
"""

import os

# Add src to path for imports
import sys
import tempfile
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.API.main import app
from src.database.models import (
    Certificate,
    Decision,
    DecisionStatus,
    Reviewer,
    ReviewerDecision,
    Student,
    TrainingType,
)


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create a temporary database file
    temp_db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db_path.close()

    # Set environment variable for test database
    os.environ["DATABASE_URL"] = f"sqlite:///{temp_db_path.name}"

    yield temp_db_path.name

    # Cleanup
    os.unlink(temp_db_path.name)


@pytest.fixture
def sample_student() -> Student:
    """Create a sample student for testing."""
    return Student(
        student_id=uuid.uuid4(),
        email="test.student@students.oamk.fi",
        degree="Bachelor of Engineering",
        first_name="Test",
        last_name="Student",
    )


@pytest.fixture
def sample_reviewer() -> Reviewer:
    """Create a sample reviewer for testing."""
    return Reviewer(
        reviewer_id=uuid.uuid4(),
        email="reviewer@oamk.fi",
        first_name="Test",
        last_name="Reviewer",
        position="Lecturer",
        department="Engineering",
    )


@pytest.fixture
def sample_certificate(sample_student) -> Certificate:
    """Create a sample certificate for testing."""
    return Certificate(
        certificate_id=uuid.uuid4(),
        student_id=sample_student.student_id,
        training_type=TrainingType.GENERAL,
        filename="test_certificate.pdf",
        filetype="pdf",
        uploaded_at=datetime.now(),
        file_content=b"test file content",
        ocr_output="Test OCR output",
    )


@pytest.fixture
def sample_decision(sample_certificate, sample_reviewer) -> Decision:
    """Create a sample decision for testing."""
    return Decision(
        decision_id=uuid.uuid4(),
        certificate_id=sample_certificate.certificate_id,
        ai_justification="Test AI justification",
        ai_decision=DecisionStatus.ACCEPTED,
        created_at=datetime.now(),
        reviewer_id=sample_reviewer.reviewer_id,
        reviewer_decision=ReviewerDecision.PASS_,
        reviewer_comment="Test reviewer comment",
        reviewed_at=datetime.now(),
        total_working_hours=120,
        credits_awarded=4,
        training_duration="3 months",
        training_institution="Test Company",
        degree_relevance="High",
        supporting_evidence="Good evidence",
        challenging_evidence="Some challenges",
        recommendation="Recommend approval",
    )


@pytest.fixture
def mock_ocr_workflow():
    """Mock OCR workflow for testing."""
    with patch("src.workflow.ocr_workflow.OCRWorkflow") as mock:
        mock_instance = Mock()
        mock_instance.extract_text.return_value = "Mock OCR text"
        mock_instance.detect_language.return_value = "fi"
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_llm_orchestrator():
    """Mock LLM orchestrator for testing."""
    with patch("src.workflow.ai_workflow.LLMOrchestrator") as mock:
        mock_instance = Mock()
        mock_instance.is_available.return_value = True
        mock_instance.process_document.return_value = {
            "ai_decision": "ACCEPTED",
            "ai_justification": "Mock AI justification",
            "total_working_hours": 120,
            "credits_awarded": 4,
            "training_duration": "3 months",
            "training_institution": "Test Company",
            "degree_relevance": "High",
            "supporting_evidence": "Good evidence",
            "challenging_evidence": "Some challenges",
            "recommendation": "Recommend approval",
        }
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(b"%PDF-1.4\nTest PDF content")
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    # Create a minimal PNG file
    temp_file.write(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def sample_docx_file():
    """Create a sample DOCX file for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    # Create a minimal DOCX file (ZIP format with minimal content)
    temp_file.write(
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].xmlPK\x01\x02\x14\x00\x14\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].xmlPK\x05\x06\x00\x00\x00\x00\x01\x00\x01\x00/\x00\x00\x00\x1f\x00\x00\x00\x00\x00"
    )
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)
