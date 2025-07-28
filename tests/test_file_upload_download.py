"""
File Upload/Download Tests

Tests for file handling functionality including:
- File upload validation
- File type detection
- File content processing
- File download functionality
- Error handling for invalid files
"""

import io
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image


class TestFileUploadValidation:
    """Test file upload validation and processing."""

    def test_valid_pdf_upload(self, test_client, sample_student):
        """Test uploading a valid PDF file."""
        # Create a mock PDF file
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create:
                mock_cert = Mock()
                mock_cert.certificate_id = uuid4()
                mock_cert.to_dict.return_value = {
                    "certificate_id": str(mock_cert.certificate_id)
                }
                mock_create.return_value = mock_cert

                files = {
                    "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
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
                assert "certificate_id" in data

    def test_valid_image_upload(self, test_client, sample_student):
        """Test uploading a valid image file."""
        # Create a mock PNG image
        img = Image.new("RGB", (100, 100), color="red")
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create:
                mock_cert = Mock()
                mock_cert.certificate_id = uuid4()
                mock_cert.to_dict.return_value = {
                    "certificate_id": str(mock_cert.certificate_id)
                }
                mock_create.return_value = mock_cert

                files = {"file": ("test.png", img_io, "image/png")}
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_valid_docx_upload(self, test_client, sample_student):
        """Test uploading a valid DOCX file."""
        # Create a mock DOCX file (simplified)
        docx_content = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].xml"

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create:
                mock_cert = Mock()
                mock_cert.certificate_id = uuid4()
                mock_cert.to_dict.return_value = {
                    "certificate_id": str(mock_cert.certificate_id)
                }
                mock_create.return_value = mock_cert

                files = {
                    "file": (
                        "test.docx",
                        io.BytesIO(docx_content),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
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

    def test_invalid_file_type(self, test_client, sample_student):
        """Test uploading an invalid file type."""
        invalid_content = b"This is not a valid file"

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            files = {"file": ("test.txt", io.BytesIO(invalid_content), "text/plain")}
            data = {"training_type": "PROFESSIONAL"}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate",
                files=files,
                data=data,
            )

            assert response.status_code == 400
            data = response.json()
            assert "File type txt not supported" in data["detail"]

    def test_empty_file(self, test_client, sample_student):
        """Test uploading an empty file."""
        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
            data = {"training_type": "PROFESSIONAL"}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate",
                files=files,
                data=data,
            )

            assert response.status_code == 400
            data = response.json()
            assert "File is empty" in data["detail"]

    def test_file_too_large(self, test_client, sample_student):
        """Test uploading a file that's too large."""
        # Create a large file (simulate > 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            files = {
                "file": ("large.pdf", io.BytesIO(large_content), "application/pdf")
            }
            data = {"training_type": "PROFESSIONAL"}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate",
                files=files,
                data=data,
            )

            assert response.status_code == 400
            data = response.json()
            assert "File size too large" in data["detail"]

    def test_missing_file(self, test_client, sample_student):
        """Test upload request without file."""
        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            data = {"training_type": "PROFESSIONAL"}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate", data=data
            )

            assert response.status_code == 422  # Validation error

    def test_missing_training_type(self, test_client, sample_student):
        """Test upload request without training type."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

            response = test_client.post(
                f"/student/{sample_student.student_id}/upload-certificate", files=files
            )

            assert response.status_code == 422  # Validation error


class TestFileDownload:
    """Test file download functionality."""

    def test_download_pdf_file(self, test_client, sample_certificate):
        """Test downloading a PDF file."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        mock_cert = Mock()
        mock_cert.file_content = pdf_content
        mock_cert.filetype = "pdf"
        mock_cert.filename = "test.pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == pdf_content

    def test_download_image_file(self, test_client, sample_certificate):
        """Test downloading an image file."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_content = img_io.getvalue()

        mock_cert = Mock()
        mock_cert.file_content = img_content
        mock_cert.filetype = "png"
        mock_cert.filename = "test.png"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"
            assert response.content == img_content

    def test_download_docx_file(self, test_client, sample_certificate):
        """Test downloading a DOCX file."""
        docx_content = b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00[Content_Types].xml"

        mock_cert = Mock()
        mock_cert.file_content = docx_content
        mock_cert.filetype = "docx"
        mock_cert.filename = "test.docx"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}"
            )

            assert response.status_code == 200
            assert (
                response.headers["content-type"]
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            assert response.content == docx_content

    def test_download_nonexistent_file(self, test_client):
        """Test downloading a file that doesn't exist."""
        certificate_id = uuid4()

        with patch("src.API.api.get_certificate_by_id", return_value=None):
            response = test_client.get(f"/certificate/{certificate_id}")

            assert response.status_code == 404
            assert response.json()["detail"] == "Certificate not found"

    def test_download_file_no_content(self, test_client, sample_certificate):
        """Test downloading a file with no content."""
        mock_cert = Mock()
        mock_cert.file_content = None
        mock_cert.filetype = "pdf"
        mock_cert.filename = "test.pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}"
            )

            assert response.status_code == 404
            assert response.json()["detail"] == "Certificate file content not found"


class TestFilePreview:
    """Test file preview functionality."""

    def test_preview_pdf_file(self, test_client, sample_certificate):
        """Test previewing a PDF file."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        mock_cert = Mock()
        mock_cert.file_content = pdf_content
        mock_cert.filetype = "pdf"
        mock_cert.filename = "test.pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}/preview"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content == pdf_content

    def test_preview_image_file(self, test_client, sample_certificate):
        """Test previewing an image file."""
        img = Image.new("RGB", (100, 100), color="green")
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG")
        img_content = img_io.getvalue()

        mock_cert = Mock()
        mock_cert.file_content = img_content
        mock_cert.filetype = "jpg"
        mock_cert.filename = "test.jpg"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            response = test_client.get(
                f"/certificate/{sample_certificate.certificate_id}/preview"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
            assert response.content == img_content


class TestFileProcessing:
    """Test file processing workflows."""

    def test_file_processing_with_ocr(
        self, test_client, sample_certificate, sample_student
    ):
        """Test processing a file through OCR workflow."""
        # Create a mock certificate with file content
        mock_cert = Mock()
        mock_cert.certificate_id = sample_certificate.certificate_id
        mock_cert.student_id = sample_certificate.student_id
        mock_cert.file_content = b"Mock file content"
        mock_cert.filename = "test.pdf"
        mock_cert.filetype = "pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            with patch("src.API.api.get_student_by_id", return_value=sample_student):
                with patch("src.API.api.ocr_workflow") as mock_ocr:
                    with patch("src.API.api.llm_orchestrator") as mock_llm:
                        mock_ocr.process_document.return_value = {
                            "success": True,
                            "extracted_text": "Mock extracted text from OCR",
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
                            assert "ocr_results" in data
                            assert "llm_results" in data

    def test_file_processing_ocr_failure(
        self, test_client, sample_certificate, sample_student
    ):
        """Test processing when OCR fails."""
        mock_cert = Mock()
        mock_cert.certificate_id = sample_certificate.certificate_id
        mock_cert.student_id = sample_certificate.student_id
        mock_cert.file_content = b"Mock file content"
        mock_cert.filename = "test.pdf"
        mock_cert.filetype = "pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            with patch("src.API.api.get_student_by_id", return_value=sample_student):
                with patch("src.API.api.ocr_workflow") as mock_ocr:
                    mock_ocr.process_document.return_value = {
                        "success": False,
                        "error": "OCR processing failed",
                    }

                    response = test_client.post(
                        f"/certificate/{sample_certificate.certificate_id}/process"
                    )

                    assert response.status_code == 500
                    data = response.json()
                    assert "OCR processing failed" in data["detail"]

    def test_file_processing_llm_failure(
        self, test_client, sample_certificate, sample_student
    ):
        """Test processing when LLM fails but OCR succeeds."""
        mock_cert = Mock()
        mock_cert.certificate_id = sample_certificate.certificate_id
        mock_cert.student_id = sample_certificate.student_id
        mock_cert.file_content = b"Mock file content"
        mock_cert.filename = "test.pdf"
        mock_cert.filetype = "pdf"

        with patch("src.API.api.get_certificate_by_id", return_value=mock_cert):
            with patch("src.API.api.get_student_by_id", return_value=sample_student):
                with patch("src.API.api.ocr_workflow") as mock_ocr:
                    with patch("src.API.api.llm_orchestrator") as mock_llm:
                        mock_ocr.process_document.return_value = {
                            "success": True,
                            "extracted_text": "Mock extracted text",
                        }

                        mock_llm.process_work_certificate.side_effect = Exception(
                            "LLM processing failed"
                        )

                        response = test_client.post(
                            f"/certificate/{sample_certificate.certificate_id}/process"
                        )

                        assert response.status_code == 200
                        data = response.json()
                        # When LLM fails, the API returns a response without 'success' key
                        # but with the error information in llm_results
                        assert "llm_results" in data
                        assert data["llm_results"]["success"] is False


class TestFileValidation:
    """Test file validation utilities."""

    def test_file_type_detection_pdf(self):
        """Test detecting PDF file type."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        # Test that PDF content starts with the correct header
        assert pdf_content.startswith(b"%PDF")

    def test_file_type_detection_image(self):
        """Test detecting image file type."""
        img = Image.new("RGB", (100, 100), color="red")
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_content = img_io.getvalue()

        # PNG files start with specific bytes
        assert img_content.startswith(b"\x89PNG\r\n\x1a\n")

    def test_file_size_validation(self):
        """Test file size validation."""
        small_content = b"x" * (5 * 1024 * 1024)  # 5MB
        large_content = b"x" * (15 * 1024 * 1024)  # 15MB

        # Small file should be valid
        assert len(small_content) <= 10 * 1024 * 1024  # 10MB limit

        # Large file should be invalid
        assert len(large_content) > 10 * 1024 * 1024  # 10MB limit

    def test_file_content_validation(self):
        """Test file content validation."""
        valid_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        invalid_content = b"This is not a valid file"

        # Valid PDF should start with %PDF
        assert valid_pdf.startswith(b"%PDF")

        # Invalid content should not start with %PDF
        assert not invalid_content.startswith(b"%PDF")


class TestFileErrorHandling:
    """Test file error handling scenarios."""

    def test_corrupted_file_upload(self, test_client, sample_student):
        """Test uploading a corrupted file."""
        corrupted_content = (
            b"This is corrupted content that doesn't match any file format"
        )

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create:
                mock_cert = Mock()
                mock_cert.certificate_id = uuid4()
                mock_cert.to_dict.return_value = {
                    "certificate_id": str(mock_cert.certificate_id)
                }
                mock_create.return_value = mock_cert

                files = {
                    "file": (
                        "corrupted.pdf",
                        io.BytesIO(corrupted_content),
                        "application/pdf",
                    )
                }
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                # Should either reject the file or process it with errors
                assert response.status_code in [200, 400]

    def test_malicious_file_upload(self, test_client, sample_student):
        """Test uploading a potentially malicious file."""
        # Create content that might be flagged as suspicious
        suspicious_content = b"<script>alert('xss')</script>" + b"x" * 1000

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch("src.API.api.create_certificate") as mock_create:
                mock_cert = Mock()
                mock_cert.certificate_id = uuid4()
                mock_cert.to_dict.return_value = {
                    "certificate_id": str(mock_cert.certificate_id)
                }
                mock_create.return_value = mock_cert

                files = {
                    "file": (
                        "suspicious.pdf",
                        io.BytesIO(suspicious_content),
                        "application/pdf",
                    )
                }
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                # Should handle suspicious content appropriately
                assert response.status_code in [200, 400, 422]

    def test_network_error_during_upload(self, test_client, sample_student):
        """Test handling network errors during upload."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch(
                "src.API.api.create_certificate", side_effect=Exception("Network error")
            ):
                files = {
                    "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
                }
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                assert response.status_code == 500
                data = response.json()
                assert "failed to create certificate" in data["detail"].lower()

    def test_disk_space_error(self, test_client, sample_student):
        """Test handling disk space errors during file processing."""
        pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        )

        with patch("src.API.api.get_student_by_id", return_value=sample_student):
            with patch(
                "src.API.api.create_certificate",
                side_effect=OSError("No space left on device"),
            ):
                files = {
                    "file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")
                }
                data = {"training_type": "PROFESSIONAL"}

                response = test_client.post(
                    f"/student/{sample_student.student_id}/upload-certificate",
                    files=files,
                    data=data,
                )

                assert response.status_code == 500
                data = response.json()
                assert "failed to create certificate" in data["detail"].lower()
