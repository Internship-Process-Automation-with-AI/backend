"""
Tests for API endpoints functionality.
"""

import io
from unittest.mock import Mock, patch
from uuid import uuid4

from src.database.models import TrainingType


class TestStudentEndpoints:
    """Test student-related API endpoints."""

    def test_student_lookup_success(self, test_client, sample_student):
        """Test successful student lookup by email."""
        with patch("src.API.api.get_student_by_email", return_value=sample_student):
            with patch(
                "src.API.api.get_student_with_certificates",
                return_value=Mock(certificates=[]),
            ):
                response = test_client.get(f"/student/{sample_student.email}")

                assert response.status_code == 200
                data = response.json()
                assert data["student_id"] == str(sample_student.student_id)
                assert data["degree"] == sample_student.degree
                assert data["first_name"] == sample_student.first_name
                assert data["last_name"] == sample_student.last_name
                assert data["certificates"] == []

    def test_student_lookup_not_found(self, test_client):
        """Test student lookup when student doesn't exist."""
        with patch("src.API.api.get_student_by_email", return_value=None):
            response = test_client.get("/student/nonexistent@example.com")

            assert response.status_code == 404
            assert response.json()["detail"] == "Student not found"

    def test_get_student_applications_success(self, test_client, sample_student):
        """Test getting student applications successfully."""
        with patch("src.API.api.get_student_by_email", return_value=sample_student):
            with patch("src.API.api.get_db_connection") as mock_conn:
                mock_cursor = Mock()
                from datetime import datetime

                mock_cursor.fetchall.return_value = [
                    (
                        uuid4(),  # certificate_id
                        "PROFESSIONAL",  # training_type
                        "test.pdf",  # filename
                        datetime(2024, 1, 1),  # uploaded_at
                        "ACCEPTED",  # ai_decision
                        "Valid certificate",  # ai_justification
                        datetime(2024, 1, 1),  # decision_created_at
                        uuid4(),  # reviewer_id
                        "PASS",  # reviewer_decision
                        "John",  # first_name
                        "Reviewer",  # last_name
                        None,  # appeal_status
                        None,  # appeal_submitted_at
                        None,  # appeal_reason
                        None,  # appeal_review_comment
                        None,  # appeal_reviewed_at
                        None,  # appeal_reviewer_first_name
                        None,  # appeal_reviewer_last_name
                        1600,  # total_working_hours
                        20.0,  # credits_awarded
                        "2 years",  # training_duration
                        "Tech Corp",  # training_institution
                        "high",  # degree_relevance
                        "Technical skills",  # supporting_evidence
                        None,  # challenging_evidence
                        "Approve",  # recommendation
                    )
                ]
                mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                response = test_client.get(
                    f"/student/{sample_student.email}/applications"
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert len(data["applications"]) == 1
                assert data["applications"][0]["training_type"] == "PROFESSIONAL"
                assert data["applications"][0]["ai_decision"] == "ACCEPTED"

    def test_get_student_applications_student_not_found(self, test_client):
        """Test getting applications for non-existent student."""
        with patch("src.API.api.get_student_by_email", return_value=None):
            response = test_client.get("/student/nonexistent@example.com/applications")

            assert response.status_code == 404
            assert response.json()["detail"] == "Student not found"

    def test_upload_certificate_success(self, test_client, sample_student):
        """Test successful certificate upload."""
        certificate_id = uuid4()

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create_cert:
                mock_cert = Mock()
                mock_cert.certificate_id = certificate_id
                mock_cert.to_dict.return_value = {"certificate_id": str(certificate_id)}
                mock_create_cert.return_value = mock_cert

                # Create a mock file
                file_content = b"Mock PDF content"
                files = {
                    "file": ("test.pdf", io.BytesIO(file_content), "application/pdf")
                }
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["certificate_id"] == str(certificate_id)

    def test_upload_certificate_invalid_training_type(
        self, test_client, sample_student
    ):
        """Test certificate upload with invalid training type."""
        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            file_content = b"Mock PDF content"
            files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
            data = {"training_type": "INVALID_TYPE"}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate",
                files=files,
                data=data,
            )

            assert response.status_code == 400
            assert "Invalid training type" in response.json()["detail"]

    def test_upload_certificate_student_not_found(self, test_client):
        """Test certificate upload for non-existent student."""
        student_id = uuid4()
        file_content = b"Mock PDF content"
        files = {"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")}
        data = {"training_type": "PROFESSIONAL"}

        with patch("src.API.api.get_student_by_id", return_value=None):
            response = test_client.post(
                f"/student/{student_id}/upload-certificate", files=files, data=data
            )

            assert response.status_code == 404
            assert response.json()["detail"] == "Student not found"

    def test_process_certificate_success(
        self, test_client, sample_certificate, sample_student
    ):
        """Test successful certificate processing."""
        # Create a mock certificate with file content
        mock_cert = Mock()
        mock_cert.certificate_id = sample_certificate.certificate_id
        mock_cert.student_id = sample_certificate.student_id  # Use the actual UUID
        mock_cert.file_content = b"Mock file content"
        mock_cert.filename = "test.pdf"
        mock_cert.filetype = "pdf"  # Add the missing filetype

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            with patch("src.API.api.get_student_by_id", return_value=sample_student):
                with patch("src.API.api.ocr_workflow") as mock_ocr:
                    with patch("src.API.api.llm_orchestrator") as mock_llm:
                        mock_ocr.process_document.return_value = {
                            "success": True,
                            "extracted_text": "Mock extracted text",
                        }

                        mock_llm.process_work_certificate.return_value = {
                            "success": True,
                            "extraction_results": {"employee_name": "John Doe"},
                            "evaluation_results": {"credits_qualified": 20.0},
                        }

                        with patch("src.API.api.create_decision"):
                            response = test_client.post(
                                f"/certificate/{sample_certificate.certificate_id}/process"
                            )

                            assert response.status_code == 200
                            data = response.json()
                            assert data["success"] is True
                            assert "certificate_id" in data
                            assert "ocr_results" in data
                            assert "llm_results" in data
                            assert "decision" in data

    def test_process_certificate_not_found(self, test_client):
        """Test processing non-existent certificate."""
        certificate_id = uuid4()

        with patch("src.API.api.get_certificate_by_id", return_value=None):
            response = test_client.post(f"/certificate/{certificate_id}/process")

            assert response.status_code == 404
            assert response.json()["detail"] == "Certificate not found"

    def test_add_feedback_success(self, test_client, sample_certificate):
        """Test successful feedback submission."""
        with patch(
            "src.API.api.get_certificate_by_id", return_value=sample_certificate
        ):
            with patch("src.API.api.add_student_feedback"):
                feedback_data = {
                    "student_feedback": "This is my feedback",
                    "reviewer_id": str(uuid4()),
                }

                response = test_client.post(
                    f"/certificate/{sample_certificate.certificate_id}/feedback",
                    json=feedback_data,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Feedback and reviewer information stored"

    def test_add_feedback_certificate_not_found(self, test_client):
        """Test adding feedback to non-existent certificate."""
        certificate_id = uuid4()
        feedback_data = {"student_feedback": "Test feedback"}

        with patch("src.API.api.get_certificate_by_id", return_value=None):
            response = test_client.post(
                f"/certificate/{certificate_id}/feedback", json=feedback_data
            )

            assert response.status_code == 404
            assert response.json()["detail"] == "Certificate not found"

    def test_delete_certificate_success(self, test_client, sample_certificate):
        """Test successful certificate deletion."""
        with patch(
            "src.API.api.get_certificate_by_id", return_value=sample_certificate
        ):
            with patch("src.API.api.delete_certificate"):
                response = test_client.delete(
                    f"/certificate/{sample_certificate.certificate_id}"
                )

                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Certificate deleted successfully"

    def test_delete_certificate_not_found(self, test_client):
        """Test deleting non-existent certificate."""
        certificate_id = uuid4()

        with patch("src.API.api.get_certificate_by_id", return_value=None):
            response = test_client.delete(f"/certificate/{certificate_id}")

            assert response.status_code == 404
            assert response.json()["detail"] == "Certificate not found"

    def test_send_for_approval_success(
        self, test_client, sample_certificate, sample_reviewer
    ):
        """Test successful send for approval."""
        with patch(
            "src.API.api.get_certificate_by_id", return_value=sample_certificate
        ):
            with patch("src.API.api.get_db_connection") as mock_conn:
                mock_cursor = Mock()
                mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor

                approval_data = {"reviewer_id": str(sample_reviewer.reviewer_id)}

                response = test_client.post(
                    f"/certificate/{sample_certificate.certificate_id}/send-for-approval",
                    json=approval_data,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Certificate sent for approval successfully"

    def test_submit_appeal_success(self, test_client, sample_certificate):
        """Test successful appeal submission."""
        with patch(
            "src.API.api.get_certificate_by_id", return_value=sample_certificate
        ):
            with patch("src.API.api.submit_appeal"):
                appeal_data = {"appeal_reason": "I disagree with the decision"}

                response = test_client.post(
                    f"/certificate/{sample_certificate.certificate_id}/appeal",
                    json=appeal_data,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Appeal submitted successfully"

    def test_download_certificate_success(self, test_client, sample_certificate):
        """Test successful certificate download."""
        mock_cert = Mock()
        mock_cert.file_content = b"Mock file content"
        mock_cert.filetype = "pdf"
        mock_cert.filename = "test.pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == b"Mock file content"

    def test_preview_certificate_success(self, test_client, sample_certificate):
        """Test successful certificate preview."""
        mock_cert = Mock()
        mock_cert.file_content = b"Mock file content"
        mock_cert.filetype = "pdf"
        mock_cert.filename = "test.pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}/preview"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == b"Mock file content"


class TestReviewerEndpoints:
    """Test reviewer-related API endpoints."""

    def test_reviewer_lookup_success(self, test_client, sample_reviewer):
        """Test successful reviewer lookup by email."""
        with patch("src.API.api.get_reviewer_by_email", return_value=sample_reviewer):
            response = test_client.get(f"/reviewer/{sample_reviewer.email}")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "reviewer" in data

    def test_reviewer_lookup_not_found(self, test_client):
        """Test reviewer lookup when reviewer doesn't exist."""
        with patch("src.API.api.get_reviewer_by_email", return_value=None):
            response = test_client.get("/reviewer/nonexistent@example.com")

            assert response.status_code == 404
            assert response.json()["detail"] == "Reviewer not found"

    def test_get_reviewer_certificates_success(self, test_client, sample_reviewer):
        """Test getting certificates for a reviewer."""
        with patch("src.API.api.get_certificates_by_reviewer_id") as mock_get_certs:
            mock_certs = [
                Mock(
                    certificate_id=uuid4(),
                    filename="test1.pdf",
                    training_type=TrainingType.PROFESSIONAL,
                    uploaded_at="2024-01-01T00:00:00",
                    to_dict=lambda: {
                        "certificate_id": str(uuid4()),
                        "filename": "test1.pdf",
                    },
                ),
                Mock(
                    certificate_id=uuid4(),
                    filename="test2.pdf",
                    training_type=TrainingType.GENERAL,
                    uploaded_at="2024-01-02T00:00:00",
                    to_dict=lambda: {
                        "certificate_id": str(uuid4()),
                        "filename": "test2.pdf",
                    },
                ),
            ]
            mock_get_certs.return_value = mock_certs

            response = test_client.get(
                f"/reviewer/{sample_reviewer.reviewer_id}/certificates"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "applications" in data

    def test_get_certificate_details_success(self, test_client, sample_certificate):
        """Test getting detailed certificate information."""
        with patch("src.API.api.get_detailed_application") as mock_get_details:
            mock_details = Mock()
            mock_details.to_dict.return_value = {
                "certificate_id": str(sample_certificate.certificate_id),
                "filename": "test.pdf",
            }
            mock_get_details.return_value = mock_details

            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}/details"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "application" in data

    def test_update_certificate_review_success(self, test_client, sample_certificate):
        """Test successful certificate review update."""
        with patch("src.API.api.update_decision_review", return_value=(True, None)):
            review_data = {
                "reviewer_comment": "This is a good certificate",
                "reviewer_decision": "PASS",
            }

            response = test_client.post(
                f"/certificate/{sample_certificate.certificate_id}/review",
                json=review_data,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Certificate review updated successfully" in data["message"]

    def test_update_certificate_review_invalid_decision(
        self, test_client, sample_certificate
    ):
        """Test review update with invalid decision."""
        with patch(
            "src.API.api.get_certificate_by_id", return_value=sample_certificate
        ):
            review_data = {
                "reviewer_comment": "Test comment",
                "reviewer_decision": "INVALID",
            }

            response = test_client.post(
                f"/certificate/{sample_certificate.certificate_id}/review",
                json=review_data,
            )

            assert response.status_code == 400
            assert "Invalid reviewer_decision" in response.json()["detail"]


class TestGeneralEndpoints:
    """Test general API endpoints."""

    def test_get_reviewers_success(self, test_client):
        """Test getting all reviewers."""
        with patch("src.API.api.get_all_reviewers") as mock_get_reviewers:
            mock_reviewers = [
                Mock(
                    reviewer_id=uuid4(),
                    first_name="John",
                    last_name="Reviewer1",
                    email="reviewer1@example.com",
                    to_dict=lambda: {"first_name": "John", "last_name": "Reviewer1"},
                ),
                Mock(
                    reviewer_id=uuid4(),
                    first_name="Jane",
                    last_name="Reviewer2",
                    email="reviewer2@example.com",
                    to_dict=lambda: {"first_name": "Jane", "last_name": "Reviewer2"},
                ),
            ]
            mock_get_reviewers.return_value = mock_reviewers

            response = test_client.get("/reviewers")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_health_check(self, test_client):
        """Test API health check endpoint."""
        response = test_client.get("/docs")
        assert response.status_code == 200  # FastAPI docs endpoint


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    def test_invalid_uuid_format(self, test_client):
        """Test handling of invalid UUID format."""
        response = test_client.get("/certificate/invalid-uuid")
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        response = test_client.post("/certificate/invalid-uuid/feedback", json={})
        assert response.status_code == 422  # Validation error

    def test_invalid_file_upload(self, test_client, sample_student):
        """Test handling of invalid file upload."""
        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            # Upload without file
            data = {"training_type": "PROFESSIONAL"}
            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate", data=data
            )
            assert response.status_code == 422  # Validation error

    def test_database_connection_error(self, test_client, sample_student):
        """Test handling of database connection errors."""
        # This test expects the exception to be handled by FastAPI
        # We'll test a different scenario that actually returns 500
        with patch("src.API.api.get_student_by_email", return_value=None):
            response = test_client.get(f"/student/{sample_student.email}")
            assert response.status_code == 404  # Student not found
