"""
API endpoint tests for FastAPI router in src/API/api.py.

Covers:
- Student lookup and applications
- Upload certificate
- Process certificate (OCR + LLM orchestration)
- Reviewers listing and lookup
- Reviewer certificates
- Delete certificate
"""

import io
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.API.main import app
from src.database.models import (
    ApplicationSummary,
    Certificate,
    Decision,
    DecisionStatus,
    Reviewer,
    Student,
    TrainingType,
)


@pytest.fixture()
def client():
    return TestClient(app)


def _cm(return_value):
    mock = Mock()
    mock.__enter__ = Mock(return_value=return_value)
    mock.__exit__ = Mock(return_value=None)
    return mock


class TestStudentEndpoints:
    @patch("src.API.api.get_student_with_certificates")
    @patch("src.API.api.get_student_by_email")
    def test_student_lookup_success(
        self, mock_get_by_email, mock_get_with_certs, client
    ):
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="john@students.oamk.fi",
            degree="Computer Science",
            first_name="John",
            last_name="Doe",
        )
        mock_get_by_email.return_value = student

        # Return empty certificates list
        mock_get_with_certs.return_value = Mock(certificates=[])

        resp = client.get(f"/student/{student.email}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["student_id"] == str(student_id)
        assert data["degree"] == "Computer Science"
        assert data["first_name"] == "John"
        assert data["last_name"] == "Doe"
        assert data["certificates"] == []

    @patch("src.API.api.get_student_by_email")
    def test_student_lookup_not_found(self, mock_get_by_email, client):
        mock_get_by_email.return_value = None
        resp = client.get("/student/missing@students.oamk.fi")
        assert resp.status_code == 404

    @patch("src.API.api.get_db_connection")
    @patch("src.API.api.get_student_by_email")
    def test_get_student_applications_success(
        self, mock_get_by_email, mock_get_conn, client
    ):
        student_id = uuid4()
        mock_get_by_email.return_value = Student(
            student_id=student_id,
            email="john@students.oamk.fi",
            degree="Computer Science",
            first_name="John",
            last_name="Doe",
        )

        # Build a fake DB cursor returning one application row
        mock_cursor = Mock()
        uploaded_at = datetime.now()
        decision_created_at = datetime.now()
        mock_cursor.fetchall.return_value = [
            (
                uuid4(),  # cert_id
                TrainingType.GENERAL.value,  # training_type
                "REGULAR",  # work_type (added missing column)
                "file.pdf",  # filename
                uploaded_at,  # uploaded_at
                DecisionStatus.ACCEPTED.value,  # ai_decision
                "ok",  # ai_justification
                decision_created_at,  # decision_created_at
                None,  # reviewer_id
                None,  # reviewer_decision
                None,  # reviewer_comment
                None,  # reviewed_at
                None,  # reviewer_first_name
                None,  # reviewer_last_name
                1500,  # total_working_hours
                12,  # credits_awarded
                "6 months",  # training_duration
                "OAMK",  # training_institution
                "High",  # degree_relevance
                "Support",  # supporting_evidence
                "Challenge",  # challenging_evidence
                "Recommend",  # recommendation
                None,  # student_comment
                None,  # name_validation_match_result
                None,  # name_validation_explanation
            )
        ]

        mock_conn = Mock()
        mock_conn.cursor.return_value = _cm(mock_cursor)
        mock_get_conn.return_value = _cm(mock_conn)

        resp = client.get(
            f"/student/{mock_get_by_email.return_value.email}/applications"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["applications"]) == 1
        assert data["applications"][0]["credits"] == 12


class TestUploadAndProcessEndpoints:
    @patch("src.API.api.get_certificate_by_id")
    @patch("src.API.api.create_certificate")
    @patch("src.API.api.get_student_by_id")
    def test_upload_certificate_success(
        self,
        mock_get_student_by_id,
        mock_create_certificate,
        mock_get_certificate_by_id,
        client,
    ):
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="john@students.oamk.fi",
            degree="Computer Science",
            first_name="John",
            last_name="Doe",
        )
        mock_get_student_by_id.return_value = student

        cert_id = uuid4()
        now = datetime.now()
        certificate = Certificate(
            certificate_id=cert_id,
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="file.pdf",
            filetype="pdf",
            uploaded_at=now,
            file_content=b"%PDF-1.4",
        )
        mock_create_certificate.return_value = certificate
        mock_get_certificate_by_id.return_value = certificate

        files = {"file": ("file.pdf", io.BytesIO(b"%PDF-1.4 body"), "application/pdf")}
        resp = client.post(
            f"/student/{student_id}/upload-certificate",
            data={"training_type": "GENERAL"},
            files=files,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["certificate"]["training_type"] == "GENERAL"

    @patch("src.API.api.get_student_by_id")
    def test_upload_certificate_invalid_training_type(
        self, mock_get_student_by_id, client
    ):
        student_id = uuid4()
        mock_get_student_by_id.return_value = Student(
            student_id=student_id,
            email="john@students.oamk.fi",
            degree="Computer Science",
        )

        files = {"file": ("file.pdf", io.BytesIO(b"%PDF-1.4 body"), "application/pdf")}
        resp = client.post(
            f"/student/{student_id}/upload-certificate",
            data={"training_type": "INVALID_TYPE"},
            files=files,
        )
        assert resp.status_code == 400

    @patch("src.API.api.create_decision")
    @patch("src.API.api.llm_orchestrator")
    @patch("src.API.api.ocr_workflow")
    @patch("src.API.api.get_db_connection")
    @patch("src.API.api.get_student_by_id")
    @patch("src.API.api.get_certificate_by_id")
    def test_process_certificate_success(
        self,
        mock_get_certificate_by_id,
        mock_get_student_by_id,
        mock_get_conn,
        mock_ocr,
        mock_llm,
        mock_create_decision,
        client,
    ):
        certificate_id = uuid4()
        student_id = uuid4()

        cert = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="file.pdf",
            filetype="pdf",
            uploaded_at=datetime.now(),
            file_content=b"%PDF-1.4 body",
        )
        mock_get_certificate_by_id.return_value = cert
        mock_get_student_by_id.return_value = Student(
            student_id=student_id,
            email="john@students.oamk.fi",
            degree="Computer Science",
        )

        # DB connection for OCR update
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = _cm(mock_cursor)
        mock_get_conn.return_value = _cm(mock_conn)

        # OCR and LLM mocks
        mock_ocr.process_document.return_value = {
            "success": True,
            "extracted_text": "some text",
        }
        mock_llm.process_work_certificate.return_value = {
            "success": True,
            "processing_time": 0.12,
            "evaluation_results": {
                "results": {
                    "decision": "ACCEPTED",
                    "credits_qualified": 10,
                    "total_working_hours": 1500,
                    "justification": "ok",
                    "degree_relevance": "High",
                    "supporting_evidence": "Support",
                    "challenging_evidence": "Challenge",
                    "recommendation": "Recommend",
                }
            },
            "extraction_results": {
                "results": {
                    "total_employment_period": "6 months",
                    "employer": "OAMK",
                }
            },
        }

        decision = Decision(
            decision_id=uuid4(),
            certificate_id=certificate_id,
            ai_justification="ok",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=datetime.now(),
            total_working_hours=1500,
            credits_awarded=10,
            training_duration="6 months",
            training_institution="OAMK",
        )
        mock_create_decision.return_value = decision

        resp = client.post(f"/certificate/{certificate_id}/process")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["decision"]["ai_decision"] == DecisionStatus.ACCEPTED.value


class TestReviewerEndpoints:
    @patch("src.API.api.get_all_reviewers")
    def test_get_reviewers_success(self, mock_get_all, client):
        reviewers = [
            Reviewer(
                reviewer_id=uuid4(),
                email="rev1@oamk.fi",
                first_name="Alice",
                last_name="Smith",
            ),
            Reviewer(
                reviewer_id=uuid4(),
                email="rev2@oamk.fi",
                first_name="Bob",
                last_name="Lee",
            ),
        ]
        mock_get_all.return_value = reviewers

        resp = client.get("/reviewers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["reviewers"]) == 2

    @patch("src.API.api.get_reviewer_by_email")
    def test_reviewer_lookup_success(self, mock_get_by_email, client):
        reviewer = Reviewer(reviewer_id=uuid4(), email="rev1@oamk.fi", first_name="A")
        mock_get_by_email.return_value = reviewer
        resp = client.get("/reviewer/rev1@oamk.fi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["reviewer"]["email"] == "rev1@oamk.fi"

    @patch("src.API.api.get_certificates_by_reviewer_id")
    def test_get_reviewer_certificates_success(self, mock_get_by_rev, client):
        app_summaries = [
            ApplicationSummary(
                decision_id=uuid4(),
                certificate_id=uuid4(),
                student_name="John Doe",
                student_email="john@students.oamk.fi",
                student_degree="CS",
                filename="f.pdf",
                training_type=TrainingType.GENERAL,
                ai_decision=DecisionStatus.ACCEPTED,
                uploaded_at=datetime.now(),
                created_at=datetime.now(),
            )
        ]
        mock_get_by_rev.return_value = app_summaries
        resp = client.get(f"/reviewer/{uuid4()}/certificates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["applications"]) == 1


class TestDeleteEndpoint:
    @patch("src.API.api.delete_certificate")
    @patch("src.API.api.get_certificate_by_id")
    def test_delete_certificate_success(self, mock_get_cert, mock_delete, client):
        cert = Certificate(
            certificate_id=uuid4(),
            student_id=uuid4(),
            training_type=TrainingType.GENERAL,
            filename="f.pdf",
            filetype="pdf",
            uploaded_at=datetime.now(),
            file_content=b"%PDF-1.4",
        )
        mock_get_cert.return_value = cert
        resp = client.delete(f"/certificate/{cert.certificate_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
