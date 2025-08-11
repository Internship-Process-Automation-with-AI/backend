"""
Database Models Tests

Tests for database models including:
- Student model validation and serialization
- Certificate model validation and serialization
- Decision model validation and serialization
- Reviewer model validation and serialization
- Model relationships and constraints
- Edge cases and error handling
"""

from datetime import datetime
from uuid import uuid4

from src.database.models import (
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


class TestStudentModel:
    """Test Student model functionality."""

    def test_student_creation_valid(self):
        """Test creating a valid Student instance."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="test.student@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        assert student.student_id == student_id
        assert student.email == "test.student@students.oamk.fi"
        assert student.degree == "Bachelor of Engineering"
        assert student.first_name == "John"
        assert student.last_name == "Doe"

    def test_student_to_dict(self):
        """Test Student to_dict method."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="test.student@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        student_dict = student.to_dict()

        assert student_dict["student_id"] == str(student_id)
        assert student_dict["email"] == "test.student@students.oamk.fi"
        assert student_dict["degree"] == "Bachelor of Engineering"
        assert student_dict["first_name"] == "John"
        assert student_dict["last_name"] == "Doe"

    def test_student_email_validation(self):
        """Test Student email validation."""
        student_id = uuid4()

        # Valid email
        student = Student(
            student_id=student_id,
            email="test.student@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )
        assert student.email == "test.student@students.oamk.fi"

        # Test with different valid email formats
        student.email = "student@oamk.fi"
        assert student.email == "student@oamk.fi"

        student.email = "test123@students.oamk.fi"
        assert student.email == "test123@students.oamk.fi"

    def test_student_degree_validation(self):
        """Test Student degree validation."""
        student_id = uuid4()

        # Valid degrees
        valid_degrees = [
            "Bachelor of Engineering",
            "Master of Engineering",
            "Bachelor of Business Administration",
            "Master of Business Administration",
        ]

        for degree in valid_degrees:
            student = Student(
                student_id=student_id,
                email="test@students.oamk.fi",
                degree=degree,
                first_name="John",
                last_name="Doe",
            )
            assert student.degree == degree

    def test_student_name_validation(self):
        """Test Student name validation."""
        student_id = uuid4()

        # Test with various name formats
        test_cases = [
            ("John", "Doe"),
            ("Mary-Jane", "O'Connor"),
            ("José", "García"),
            ("李", "小明"),
            ("", "Doe"),  # Empty first name
            ("John", ""),  # Empty last name
        ]

        for first_name, last_name in test_cases:
            student = Student(
                student_id=student_id,
                email="test@students.oamk.fi",
                degree="Bachelor of Engineering",
                first_name=first_name,
                last_name=last_name,
            )
            assert student.first_name == first_name
            assert student.last_name == last_name

    def test_student_equality(self):
        """Test Student equality comparison."""
        student_id = uuid4()
        student1 = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        student2 = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        assert student1 == student2

    def test_student_inequality(self):
        """Test Student inequality comparison."""
        student1 = Student(
            student_id=uuid4(),
            email="test1@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        student2 = Student(
            student_id=uuid4(),
            email="test2@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="Jane",
            last_name="Smith",
        )

        assert student1 != student2


class TestCertificateModel:
    """Test Certificate model functionality."""

    def test_certificate_creation_valid(self):
        """Test creating a valid Certificate instance."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
            uploaded_at=uploaded_at,
            ocr_output="Test OCR output",
        )

        assert certificate.certificate_id == certificate_id
        assert certificate.student_id == student_id
        assert certificate.training_type == TrainingType.PROFESSIONAL
        assert certificate.filename == "test.pdf"
        assert certificate.file_content == b"test content"
        assert certificate.filetype == "pdf"
        assert certificate.uploaded_at == uploaded_at
        assert certificate.ocr_output == "Test OCR output"

    def test_certificate_to_dict(self):
        """Test Certificate to_dict method."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
            uploaded_at=uploaded_at,
            ocr_output="Test OCR output",
        )

        cert_dict = certificate.to_dict()

        assert cert_dict["certificate_id"] == str(certificate_id)
        assert cert_dict["filename"] == "test.pdf"
        assert cert_dict["training_type"] == "PROFESSIONAL"
        assert cert_dict["uploaded_at"] == str(uploaded_at)

    def test_certificate_training_type_validation(self):
        """Test Certificate training type validation."""
        certificate_id = uuid4()
        student_id = uuid4()

        # Test all training types
        for training_type in TrainingType:
            certificate = Certificate(
                certificate_id=certificate_id,
                student_id=student_id,
                training_type=training_type,
                filename="test.pdf",
                file_content=b"test content",
                filetype="pdf",
                uploaded_at=datetime.now(),
            )
            assert certificate.training_type == training_type

    def test_certificate_file_validation(self):
        """Test Certificate file validation."""
        certificate_id = uuid4()
        student_id = uuid4()

        # Test various file types
        test_files = [
            ("test.pdf", "pdf", b"PDF content"),
            ("test.png", "png", b"PNG content"),
            ("test.docx", "docx", b"DOCX content"),
            ("test.jpg", "jpg", b"JPG content"),
        ]

        for filename, filetype, content in test_files:
            certificate = Certificate(
                certificate_id=certificate_id,
                student_id=student_id,
                training_type=TrainingType.PROFESSIONAL,
                filename=filename,
                file_content=content,
                filetype=filetype,
                uploaded_at=datetime.now(),
            )
            assert certificate.filename == filename
            assert certificate.filetype == filetype
            assert certificate.file_content == content

    def test_certificate_without_optional_fields(self):
        """Test Certificate creation without optional fields."""
        certificate_id = uuid4()
        student_id = uuid4()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
            uploaded_at=datetime.now(),
            # ocr_output is optional
        )

        assert certificate.certificate_id == certificate_id
        assert certificate.ocr_output is None


class TestDecisionModel:
    """Test Decision model functionality."""

    def test_decision_creation_valid(self):
        """Test creating a valid Decision instance."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()
        created_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification="Valid certificate",
            credits_awarded=20.0,
            total_working_hours=1600,
            training_duration="2 years",
            training_institution="Tech Corp",
            degree_relevance="high",
            supporting_evidence="Technical skills",
            challenging_evidence=None,
            recommendation="Approve",
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.PASS_,
            reviewer_comment="Good certificate",
            created_at=created_at,
        )

        assert decision.decision_id == decision_id
        assert decision.certificate_id == certificate_id
        assert decision.ai_decision == DecisionStatus.ACCEPTED
        assert decision.ai_justification == "Valid certificate"
        assert decision.credits_awarded == 20.0
        assert decision.total_working_hours == 1600
        assert decision.training_duration == "2 years"
        assert decision.training_institution == "Tech Corp"
        assert decision.degree_relevance == "high"
        assert decision.supporting_evidence == "Technical skills"
        assert decision.challenging_evidence is None
        assert decision.recommendation == "Approve"
        assert decision.reviewer_id == reviewer_id
        assert decision.reviewer_decision == ReviewerDecision.PASS_
        assert decision.reviewer_comment == "Good certificate"
        assert decision.created_at == created_at

    def test_decision_to_dict(self):
        """Test Decision to_dict method."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()
        created_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification="Valid certificate",
            credits_awarded=20.0,
            total_working_hours=1600,
            training_duration="2 years",
            training_institution="Tech Corp",
            degree_relevance="high",
            supporting_evidence="Technical skills",
            challenging_evidence=None,
            recommendation="Approve",
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.PASS_,
            reviewer_comment="Good certificate",
            created_at=created_at,
        )

        decision_dict = decision.to_dict()

        assert decision_dict["decision_id"] == str(decision_id)
        assert decision_dict["certificate_id"] == str(certificate_id)
        assert decision_dict["ai_decision"] == "ACCEPTED"
        assert decision_dict["ai_justification"] == "Valid certificate"
        assert decision_dict["credits_awarded"] == 20.0
        assert decision_dict["total_working_hours"] == 1600
        assert decision_dict["training_duration"] == "2 years"
        assert decision_dict["training_institution"] == "Tech Corp"
        assert decision_dict["degree_relevance"] == "high"
        assert decision_dict["supporting_evidence"] == "Technical skills"
        assert decision_dict["challenging_evidence"] is None
        assert decision_dict["recommendation"] == "Approve"
        assert decision_dict["reviewer_id"] == str(reviewer_id)
        assert decision_dict["reviewer_decision"] == "PASS"
        assert decision_dict["reviewer_comment"] == "Good certificate"

    def test_decision_status_validation(self):
        """Test Decision status validation."""
        decision_id = uuid4()
        certificate_id = uuid4()

        # Test all decision statuses
        for status in DecisionStatus:
            decision = Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ai_decision=status,
                ai_justification="Test justification",
                credits_awarded=20.0,
                total_working_hours=1600,
                training_duration="2 years",
                training_institution="Tech Corp",
                degree_relevance="high",
                supporting_evidence="Technical skills",
                challenging_evidence=None,
                recommendation="Approve",
                created_at=datetime.now(),
            )
            assert decision.ai_decision == status

    def test_decision_reviewer_decision_validation(self):
        """Test Decision reviewer decision validation."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()

        # Test all reviewer decisions
        for reviewer_decision in ReviewerDecision:
            decision = Decision(
                decision_id=decision_id,
                certificate_id=certificate_id,
                ai_decision=DecisionStatus.ACCEPTED,
                ai_justification="Test justification",
                credits_awarded=20.0,
                total_working_hours=1600,
                training_duration="2 years",
                training_institution="Tech Corp",
                degree_relevance="high",
                supporting_evidence="Technical skills",
                challenging_evidence=None,
                recommendation="Approve",
                reviewer_id=reviewer_id,
                reviewer_decision=reviewer_decision,
                reviewer_comment="Test comment",
                created_at=datetime.now(),
            )
            assert decision.reviewer_decision == reviewer_decision

    def test_decision_without_optional_fields(self):
        """Test Decision creation without optional fields."""
        decision_id = uuid4()
        certificate_id = uuid4()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification="Test justification",
            credits_awarded=20.0,
            total_working_hours=1600,
            training_duration="2 years",
            training_institution="Tech Corp",
            degree_relevance="high",
            supporting_evidence="Technical skills",
            challenging_evidence=None,
            recommendation="Approve",
            created_at=datetime.now(),
            # reviewer_id, reviewer_decision, appeal_status, etc. are optional
        )

        assert decision.decision_id == decision_id
        assert decision.reviewer_id is None
        assert decision.reviewer_decision is None


class TestReviewerModel:
    """Test Reviewer model functionality."""

    def test_reviewer_creation_valid(self):
        """Test creating a valid Reviewer instance."""
        reviewer_id = uuid4()

        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="reviewer@oamk.fi",
            first_name="Jane",
            last_name="Smith",
            department="Engineering",
            position="Software Engineering",
        )

        assert reviewer.reviewer_id == reviewer_id
        assert reviewer.email == "reviewer@oamk.fi"
        assert reviewer.first_name == "Jane"
        assert reviewer.last_name == "Smith"
        assert reviewer.department == "Engineering"
        assert reviewer.position == "Software Engineering"

    def test_reviewer_to_dict(self):
        """Test Reviewer to_dict method."""
        reviewer_id = uuid4()

        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="reviewer@oamk.fi",
            first_name="Jane",
            last_name="Smith",
            department="Engineering",
            position="Software Engineering",
        )

        reviewer_dict = reviewer.to_dict()

        assert reviewer_dict["reviewer_id"] == str(reviewer_id)
        assert reviewer_dict["email"] == "reviewer@oamk.fi"
        assert reviewer_dict["first_name"] == "Jane"
        assert reviewer_dict["last_name"] == "Smith"
        assert reviewer_dict["department"] == "Engineering"
        assert reviewer_dict["position"] == "Software Engineering"

    def test_reviewer_email_validation(self):
        """Test Reviewer email validation."""
        reviewer_id = uuid4()

        # Valid emails
        valid_emails = ["reviewer@oamk.fi", "jane.smith@oamk.fi", "reviewer123@oamk.fi"]

        for email in valid_emails:
            reviewer = Reviewer(
                reviewer_id=reviewer_id,
                email=email,
                first_name="Jane",
                last_name="Smith",
                department="Engineering",
                position="Software Engineering",
            )
            assert reviewer.email == email

    def test_reviewer_without_optional_fields(self):
        """Test Reviewer creation without optional fields."""
        reviewer_id = uuid4()

        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="reviewer@oamk.fi",
            first_name="Jane",
            last_name="Smith",
            # department and position are optional
        )

        assert reviewer.reviewer_id == reviewer_id
        assert reviewer.email == "reviewer@oamk.fi"
        assert reviewer.department is None
        assert reviewer.position is None


class TestCompositeModels:
    """Test composite models and relationships."""

    def test_student_with_certificates(self):
        """Test StudentWithCertificates model."""
        student_id = uuid4()
        certificate_id = uuid4()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
            uploaded_at=datetime.now(),
        )

        student_with_certs = StudentWithCertificates(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
            certificates=[certificate],
        )

        assert student_with_certs.student_id == student_id
        assert len(student_with_certs.certificates) == 1
        assert student_with_certs.certificates[0] == certificate

    def test_application_summary(self):
        """Test ApplicationSummary model."""
        certificate_id = uuid4()
        uploaded_at = datetime.now()

        summary = ApplicationSummary(
            decision_id=uuid4(),
            certificate_id=certificate_id,
            student_name="John Doe",
            student_email="test@students.oamk.fi",
            student_degree="Bachelor of Engineering",
            filename="test.pdf",
            training_type=TrainingType.PROFESSIONAL,
            ai_decision=DecisionStatus.ACCEPTED,
            uploaded_at=uploaded_at,
            created_at=datetime.now(),
        )

        assert summary.certificate_id == certificate_id
        assert summary.training_type == TrainingType.PROFESSIONAL
        assert summary.filename == "test.pdf"
        assert summary.ai_decision == DecisionStatus.ACCEPTED
        assert summary.uploaded_at == uploaded_at
        assert summary.student_name == "John Doe"
        assert summary.student_email == "test@students.oamk.fi"

    def test_detailed_application(self):
        """Test DetailedApplication model."""
        certificate_id = uuid4()
        student_id = uuid4()

        decision = Decision(
            decision_id=uuid4(),
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification="Valid certificate",
            created_at=datetime.now(),
        )

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            filetype="pdf",
            uploaded_at=datetime.now(),
        )

        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Bachelor of Engineering",
            first_name="John",
            last_name="Doe",
        )

        detailed_app = DetailedApplication(
            decision=decision, certificate=certificate, student=student
        )

        assert detailed_app.decision.certificate_id == certificate_id
        assert detailed_app.certificate.student_id == student_id
        assert detailed_app.certificate.training_type == TrainingType.PROFESSIONAL
        assert detailed_app.decision.ai_decision == DecisionStatus.ACCEPTED
        assert detailed_app.student.email == "test@students.oamk.fi"


class TestEnumValidation:
    """Test enum validation and values."""

    def test_training_type_enum(self):
        """Test TrainingType enum values."""
        assert TrainingType.PROFESSIONAL.value == "PROFESSIONAL"
        assert TrainingType.GENERAL.value == "GENERAL"

        # Test enum membership
        assert "PROFESSIONAL" in [t.value for t in TrainingType]
        assert "GENERAL" in [t.value for t in TrainingType]

    def test_decision_status_enum(self):
        """Test DecisionStatus enum values."""
        assert DecisionStatus.ACCEPTED.value == "ACCEPTED"
        assert DecisionStatus.REJECTED.value == "REJECTED"

        # Test enum membership
        assert "ACCEPTED" in [s.value for s in DecisionStatus]
        assert "REJECTED" in [s.value for s in DecisionStatus]

    def test_reviewer_decision_enum(self):
        """Test ReviewerDecision enum values."""
        assert ReviewerDecision.PASS_.value == "PASS"
        assert ReviewerDecision.FAIL.value == "FAIL"

        # Test enum membership
        assert "PASS" in [d.value for d in ReviewerDecision]
        assert "FAIL" in [d.value for d in ReviewerDecision]


class TestModelEdgeCases:
    """Test model edge cases and error handling."""

    def test_empty_strings_in_models(self):
        """Test models with empty string values."""
        student_id = uuid4()
        certificate_id = uuid4()

        # Student with empty strings
        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="",
            first_name="",
            last_name="",
        )
        assert student.degree == ""
        assert student.first_name == ""
        assert student.last_name == ""

        # Certificate with empty strings
        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="",
            file_content=b"",
            filetype="",
            uploaded_at=datetime.now(),
            ocr_output="",
        )
        assert certificate.filename == ""
        assert certificate.file_content == b""
        assert certificate.filetype == ""
        assert certificate.ocr_output == ""

    def test_none_values_in_models(self):
        """Test models with None values for optional fields."""
        student_id = uuid4()
        certificate_id = uuid4()
        decision_id = uuid4()
        reviewer_id = uuid4()

        # Certificate with None values
        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test.pdf",
            file_content=b"test content",
            filetype="pdf",
            uploaded_at=datetime.now(),
            ocr_output=None,
        )
        assert certificate.ocr_output is None

        # Decision with None values
        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification="Test",
            credits_awarded=20.0,
            total_working_hours=1600,
            training_duration="2 years",
            training_institution="Tech Corp",
            degree_relevance="high",
            supporting_evidence="Technical skills",
            challenging_evidence=None,
            recommendation="Approve",
            created_at=datetime.now(),
            reviewer_id=None,
            reviewer_decision=None,
            reviewer_comment=None,
        )
        assert decision.reviewer_id is None
        assert decision.reviewer_decision is None
        assert decision.reviewer_comment is None

        # Reviewer with None values
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="reviewer@oamk.fi",
            first_name="Jane",
            last_name="Smith",
            department=None,
            position=None,
        )
        assert reviewer.department is None
        assert reviewer.position is None

    def test_large_values_in_models(self):
        """Test models with large values."""
        student_id = uuid4()
        certificate_id = uuid4()
        decision_id = uuid4()

        # Large file content
        large_content = b"x" * (1024 * 1024)  # 1MB

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="large_file.pdf",
            file_content=large_content,
            filetype="pdf",
            uploaded_at=datetime.now(),
        )
        assert len(certificate.file_content) == 1024 * 1024

        # Large text fields
        large_text = "x" * 10000  # 10KB text

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_decision=DecisionStatus.ACCEPTED,
            ai_justification=large_text,
            credits_awarded=20.0,
            total_working_hours=1600,
            training_duration="2 years",
            training_institution="Tech Corp",
            degree_relevance="high",
            supporting_evidence=large_text,
            challenging_evidence=large_text,
            recommendation=large_text,
            created_at=datetime.now(),
        )
        assert len(decision.ai_justification) == 10000
        assert len(decision.supporting_evidence) == 10000
        assert len(decision.challenging_evidence) == 10000
        assert len(decision.recommendation) == 10000

    def test_special_characters_in_models(self):
        """Test models with special characters."""
        student_id = uuid4()
        certificate_id = uuid4()

        # Student with special characters
        student = Student(
            student_id=student_id,
            email="test+tag@students.oamk.fi",
            degree="Bachelor's Degree in Engineering",
            first_name="José María",
            last_name="O'Connor-Smith",
        )
        assert student.email == "test+tag@students.oamk.fi"
        assert student.degree == "Bachelor's Degree in Engineering"
        assert student.first_name == "José María"
        assert student.last_name == "O'Connor-Smith"

        # Certificate with special characters
        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="test-file_2024.pdf",
            file_content=b"test content with special chars: aao",
            filetype="pdf",
            uploaded_at=datetime.now(),
            ocr_output="OCR output with special chars: äöå",
        )
        assert certificate.filename == "test-file_2024.pdf"
        assert certificate.ocr_output == "OCR output with special chars: äöå"
