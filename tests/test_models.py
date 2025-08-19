"""
Test file for database models.

Tests the data classes and enums for students, certificates, decisions, and reviewers.
"""

from datetime import datetime
from uuid import uuid4

from src.database.models import (
    Certificate,
    Decision,
    DecisionStatus,
    Reviewer,
    ReviewerDecision,
    Student,
    TrainingType,
)


class TestTrainingType:
    """Test TrainingType enum."""

    def test_training_type_values(self):
        """Test that TrainingType enum has correct values."""
        assert TrainingType.GENERAL == "GENERAL"
        assert TrainingType.PROFESSIONAL == "PROFESSIONAL"

    def test_training_type_enumeration(self):
        """Test that TrainingType enum can be iterated."""
        types = list(TrainingType)
        assert len(types) == 2
        assert TrainingType.GENERAL in types
        assert TrainingType.PROFESSIONAL in types


class TestDecisionStatus:
    """Test DecisionStatus enum."""

    def test_decision_status_values(self):
        """Test that DecisionStatus enum has correct values."""
        assert DecisionStatus.ACCEPTED == "ACCEPTED"
        assert DecisionStatus.REJECTED == "REJECTED"

    def test_decision_status_enumeration(self):
        """Test that DecisionStatus enum can be iterated."""
        statuses = list(DecisionStatus)
        assert len(statuses) == 2
        assert DecisionStatus.ACCEPTED in statuses
        assert DecisionStatus.REJECTED in statuses


class TestReviewerDecision:
    """Test ReviewerDecision enum."""

    def test_reviewer_decision_values(self):
        """Test that ReviewerDecision enum has correct values."""
        assert ReviewerDecision.PASS_ == "PASS"
        assert ReviewerDecision.FAIL == "FAIL"

    def test_reviewer_decision_enumeration(self):
        """Test that ReviewerDecision enum can be iterated."""
        decisions = list(ReviewerDecision)
        assert len(decisions) == 2
        assert ReviewerDecision.PASS_ in decisions
        assert ReviewerDecision.FAIL in decisions


class TestStudent:
    """Test Student data class."""

    def test_student_creation_with_required_fields(self):
        """Test creating a student with only required fields."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Computer Science",
        )

        assert student.student_id == student_id
        assert student.email == "test@students.oamk.fi"
        assert student.degree == "Computer Science"
        assert student.first_name is None
        assert student.last_name is None

    def test_student_creation_with_all_fields(self):
        """Test creating a student with all fields."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="john.doe@students.oamk.fi",
            degree="Engineering",
            first_name="John",
            last_name="Doe",
        )

        assert student.student_id == student_id
        assert student.email == "john.doe@students.oamk.fi"
        assert student.degree == "Engineering"
        assert student.first_name == "John"
        assert student.last_name == "Doe"

    def test_student_to_dict(self):
        """Test student to_dict method."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Computer Science",
            first_name="John",
            last_name="Doe",
        )

        student_dict = student.to_dict()

        assert student_dict["student_id"] == str(student_id)
        assert student_dict["email"] == "test@students.oamk.fi"
        assert student_dict["degree"] == "Computer Science"
        assert student_dict["first_name"] == "John"
        assert student_dict["last_name"] == "Doe"

    def test_student_to_dict_with_none_values(self):
        """Test student to_dict method with None values."""
        student_id = uuid4()
        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Computer Science",
        )

        student_dict = student.to_dict()

        assert student_dict["first_name"] is None
        assert student_dict["last_name"] is None


class TestReviewer:
    """Test Reviewer data class."""

    def test_reviewer_creation_with_required_fields(self):
        """Test creating a reviewer with only required fields."""
        reviewer_id = uuid4()
        reviewer = Reviewer(reviewer_id=reviewer_id, email="reviewer@oamk.fi")

        assert reviewer.reviewer_id == reviewer_id
        assert reviewer.email == "reviewer@oamk.fi"
        assert reviewer.first_name is None
        assert reviewer.last_name is None
        assert reviewer.position is None
        assert reviewer.department is None

    def test_reviewer_creation_with_all_fields(self):
        """Test creating a reviewer with all fields."""
        reviewer_id = uuid4()
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="prof.smith@oamk.fi",
            first_name="Professor",
            last_name="Smith",
            position="Senior Lecturer",
            department="Computer Science",
        )

        assert reviewer.reviewer_id == reviewer_id
        assert reviewer.email == "prof.smith@oamk.fi"
        assert reviewer.first_name == "Professor"
        assert reviewer.last_name == "Smith"
        assert reviewer.position == "Senior Lecturer"
        assert reviewer.department == "Computer Science"

    def test_reviewer_to_dict(self):
        """Test reviewer to_dict method."""
        reviewer_id = uuid4()
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            email="reviewer@oamk.fi",
            first_name="John",
            last_name="Smith",
            position="Lecturer",
            department="Engineering",
        )

        reviewer_dict = reviewer.to_dict()

        assert reviewer_dict["reviewer_id"] == str(reviewer_id)
        assert reviewer_dict["email"] == "reviewer@oamk.fi"
        assert reviewer_dict["first_name"] == "John"
        assert reviewer_dict["last_name"] == "Smith"
        assert reviewer_dict["position"] == "Lecturer"
        assert reviewer_dict["department"] == "Engineering"

    def test_reviewer_to_dict_with_none_values(self):
        """Test reviewer to_dict method with None values."""
        reviewer_id = uuid4()
        reviewer = Reviewer(reviewer_id=reviewer_id, email="reviewer@oamk.fi")

        reviewer_dict = reviewer.to_dict()

        assert reviewer_dict["first_name"] is None
        assert reviewer_dict["last_name"] is None
        assert reviewer_dict["position"] is None
        assert reviewer_dict["department"] is None


class TestCertificate:
    """Test Certificate data class."""

    def test_certificate_creation_with_required_fields(self):
        """Test creating a certificate with only required fields."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="test.pdf",
            filetype="pdf",
            uploaded_at=uploaded_at,
        )

        assert certificate.certificate_id == certificate_id
        assert certificate.student_id == student_id
        assert certificate.training_type == TrainingType.GENERAL
        assert certificate.filename == "test.pdf"
        assert certificate.filetype == "pdf"
        assert certificate.uploaded_at == uploaded_at
        assert certificate.file_content is None
        assert certificate.ocr_output is None

    def test_certificate_creation_with_all_fields(self):
        """Test creating a certificate with all fields."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()
        file_content = b"test file content"
        ocr_output = "Extracted text from OCR"

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="work_cert.pdf",
            filetype="pdf",
            uploaded_at=uploaded_at,
            file_content=file_content,
            ocr_output=ocr_output,
        )

        assert certificate.certificate_id == certificate_id
        assert certificate.student_id == student_id
        assert certificate.training_type == TrainingType.PROFESSIONAL
        assert certificate.filename == "work_cert.pdf"
        assert certificate.filetype == "pdf"
        assert certificate.uploaded_at == uploaded_at
        assert certificate.file_content == file_content
        assert certificate.ocr_output == ocr_output

    def test_certificate_to_dict(self):
        """Test certificate to_dict method."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="test.pdf",
            filetype="pdf",
            uploaded_at=uploaded_at,
        )

        certificate_dict = certificate.to_dict()

        assert certificate_dict["certificate_id"] == str(certificate_id)
        assert certificate_dict["filename"] == "test.pdf"
        assert certificate_dict["training_type"] == "GENERAL"
        assert certificate_dict["uploaded_at"] == str(uploaded_at)

    def test_certificate_training_type_values(self):
        """Test certificate with different training types."""
        certificate_id = uuid4()
        student_id = uuid4()
        uploaded_at = datetime.now()

        # Test GENERAL training type
        general_cert = Certificate(
            certificate_id=certificate_id,
            student_id=student_id,
            training_type=TrainingType.GENERAL,
            filename="general.pdf",
            filetype="pdf",
            uploaded_at=uploaded_at,
        )
        assert general_cert.training_type == TrainingType.GENERAL

        # Test PROFESSIONAL training type
        professional_cert = Certificate(
            certificate_id=uuid4(),
            student_id=student_id,
            training_type=TrainingType.PROFESSIONAL,
            filename="professional.pdf",
            filetype="pdf",
            uploaded_at=uploaded_at,
        )
        assert professional_cert.training_type == TrainingType.PROFESSIONAL


class TestDecision:
    """Test Decision data class."""

    def test_decision_creation_with_required_fields(self):
        """Test creating a decision with only required fields."""
        decision_id = uuid4()
        certificate_id = uuid4()
        created_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="This is a test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=created_at,
        )

        assert decision.decision_id == decision_id
        assert decision.certificate_id == certificate_id
        assert decision.ai_justification == "This is a test justification"
        assert decision.ai_decision == DecisionStatus.ACCEPTED
        assert decision.created_at == created_at
        assert decision.student_comment is None
        assert decision.reviewer_id is None
        assert decision.reviewer_decision is None
        assert decision.reviewer_comment is None
        assert decision.reviewed_at is None
        assert decision.total_working_hours is None
        assert decision.credits_awarded is None
        assert decision.training_duration is None
        assert decision.training_institution is None
        assert decision.degree_relevance is None
        assert decision.supporting_evidence is None
        assert decision.challenging_evidence is None
        assert decision.recommendation is None
        assert decision.ai_workflow_json is None

    def test_decision_creation_with_all_fields(self):
        """Test creating a decision with all fields."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()
        created_at = datetime.now()
        reviewed_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="Comprehensive AI analysis",
            ai_decision=DecisionStatus.REJECTED,
            created_at=created_at,
            student_comment="I disagree with this decision",
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.FAIL,
            reviewer_comment="Student's appeal is valid",
            reviewed_at=reviewed_at,
            total_working_hours=1500,
            credits_awarded=0,
            training_duration="6 months",
            training_institution="Tech Academy",
            degree_relevance="High relevance to Computer Science",
            supporting_evidence="Strong technical skills demonstrated",
            challenging_evidence="Limited documentation provided",
            recommendation="Requires additional documentation",
            ai_workflow_json='{"analysis": "detailed"}',
        )

        assert decision.decision_id == decision_id
        assert decision.certificate_id == certificate_id
        assert decision.ai_justification == "Comprehensive AI analysis"
        assert decision.ai_decision == DecisionStatus.REJECTED
        assert decision.created_at == created_at
        assert decision.student_comment == "I disagree with this decision"
        assert decision.reviewer_id == reviewer_id
        assert decision.reviewer_decision == ReviewerDecision.FAIL
        assert decision.reviewer_comment == "Student's appeal is valid"
        assert decision.reviewed_at == reviewed_at
        assert decision.total_working_hours == 1500
        assert decision.credits_awarded == 0
        assert decision.training_duration == "6 months"
        assert decision.training_institution == "Tech Academy"
        assert decision.degree_relevance == "High relevance to Computer Science"
        assert decision.supporting_evidence == "Strong technical skills demonstrated"
        assert decision.challenging_evidence == "Limited documentation provided"
        assert decision.recommendation == "Requires additional documentation"
        assert decision.ai_workflow_json == '{"analysis": "detailed"}'

    def test_decision_to_dict(self):
        """Test decision to_dict method."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()
        created_at = datetime.now()
        reviewed_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=created_at,
            student_comment="Test comment",
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.PASS_,
            reviewer_comment="Test review comment",
            reviewed_at=reviewed_at,
            total_working_hours=1200,
            credits_awarded=5,
            training_duration="4 months",
            training_institution="Test Academy",
            degree_relevance="High relevance",
            supporting_evidence="Strong evidence",
            challenging_evidence="No challenges",
            recommendation="Approve",
            ai_workflow_json='{"test": "data"}',
        )

        decision_dict = decision.to_dict()

        assert decision_dict["decision_id"] == str(decision_id)
        assert decision_dict["certificate_id"] == str(certificate_id)
        assert decision_dict["ai_justification"] == "Test justification"
        assert decision_dict["ai_decision"] == "ACCEPTED"
        assert decision_dict["created_at"] == created_at.isoformat()
        assert decision_dict["student_comment"] == "Test comment"
        assert decision_dict["reviewer_id"] == str(reviewer_id)
        assert decision_dict["reviewer_decision"] == "PASS"
        assert decision_dict["reviewer_comment"] == "Test review comment"
        assert decision_dict["reviewed_at"] == reviewed_at.isoformat()
        assert decision_dict["total_working_hours"] == 1200
        assert decision_dict["credits_awarded"] == 5
        assert decision_dict["training_duration"] == "4 months"
        assert decision_dict["training_institution"] == "Test Academy"
        assert decision_dict["degree_relevance"] == "High relevance"
        assert decision_dict["supporting_evidence"] == "Strong evidence"
        assert decision_dict["challenging_evidence"] == "No challenges"
        assert decision_dict["recommendation"] == "Approve"
        assert decision_dict["ai_workflow_json"] == '{"test": "data"}'

    def test_decision_to_dict_with_none_values(self):
        """Test decision to_dict method with None values."""
        decision_id = uuid4()
        certificate_id = uuid4()
        created_at = datetime.now()

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=created_at,
        )

        decision_dict = decision.to_dict()

        assert decision_dict["student_comment"] is None
        assert decision_dict["reviewer_id"] is None
        assert decision_dict["reviewer_decision"] is None
        assert decision_dict["reviewer_comment"] is None
        assert decision_dict["reviewed_at"] is None
        assert decision_dict["total_working_hours"] is None
        assert decision_dict["credits_awarded"] is None
        assert decision_dict["training_duration"] is None
        assert decision_dict["training_institution"] is None
        assert decision_dict["degree_relevance"] is None
        assert decision_dict["supporting_evidence"] is None
        assert decision_dict["challenging_evidence"] is None
        assert decision_dict["recommendation"] is None
        assert decision_dict["ai_workflow_json"] is None

    def test_decision_status_values(self):
        """Test decision with different AI decision statuses."""
        decision_id = uuid4()
        certificate_id = uuid4()
        created_at = datetime.now()

        # Test ACCEPTED decision
        accepted_decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="Accepted justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=created_at,
        )
        assert accepted_decision.ai_decision == DecisionStatus.ACCEPTED

        # Test REJECTED decision
        rejected_decision = Decision(
            decision_id=uuid4(),
            certificate_id=certificate_id,
            ai_justification="Rejected justification",
            ai_decision=DecisionStatus.REJECTED,
            created_at=created_at,
        )
        assert rejected_decision.ai_decision == DecisionStatus.REJECTED

    def test_reviewer_decision_values(self):
        """Test decision with different reviewer decisions."""
        decision_id = uuid4()
        certificate_id = uuid4()
        reviewer_id = uuid4()
        created_at = datetime.now()
        reviewed_at = datetime.now()

        # Test PASS decision
        pass_decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate_id,
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=created_at,
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.PASS_,
            reviewed_at=reviewed_at,
        )
        assert pass_decision.reviewer_decision == ReviewerDecision.PASS_

        # Test FAIL decision
        fail_decision = Decision(
            decision_id=uuid4(),
            certificate_id=certificate_id,
            ai_justification="Test justification",
            ai_decision=DecisionStatus.REJECTED,
            created_at=created_at,
            reviewer_id=reviewer_id,
            reviewer_decision=ReviewerDecision.FAIL,
            reviewed_at=reviewed_at,
        )
        assert fail_decision.reviewer_decision == ReviewerDecision.FAIL


class TestModelIntegration:
    """Test integration between different models."""

    def test_student_certificate_relationship(self):
        """Test that student and certificate can be related via student_id."""
        student_id = uuid4()
        certificate_id = uuid4()

        student = Student(
            student_id=student_id,
            email="test@students.oamk.fi",
            degree="Computer Science",
        )

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=student.student_id,  # Use student's ID
            training_type=TrainingType.GENERAL,
            filename="test.pdf",
            filetype="pdf",
            uploaded_at=datetime.now(),
        )

        assert certificate.student_id == student.student_id
        assert str(certificate.student_id) == str(student.student_id)

    def test_certificate_decision_relationship(self):
        """Test that certificate and decision can be related via certificate_id."""
        certificate_id = uuid4()
        decision_id = uuid4()

        certificate = Certificate(
            certificate_id=certificate_id,
            student_id=uuid4(),
            training_type=TrainingType.GENERAL,
            filename="test.pdf",
            filetype="pdf",
            uploaded_at=datetime.now(),
        )

        decision = Decision(
            decision_id=decision_id,
            certificate_id=certificate.certificate_id,  # Use certificate's ID
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=datetime.now(),
        )

        assert decision.certificate_id == certificate.certificate_id
        assert str(decision.certificate_id) == str(certificate.certificate_id)

    def test_decision_reviewer_relationship(self):
        """Test that decision and reviewer can be related via reviewer_id."""
        reviewer_id = uuid4()
        decision_id = uuid4()

        reviewer = Reviewer(reviewer_id=reviewer_id, email="reviewer@oamk.fi")

        decision = Decision(
            decision_id=decision_id,
            certificate_id=uuid4(),
            ai_justification="Test justification",
            ai_decision=DecisionStatus.ACCEPTED,
            created_at=datetime.now(),
            reviewer_id=reviewer.reviewer_id,  # Use reviewer's ID
            reviewer_decision=ReviewerDecision.PASS_,
            reviewed_at=datetime.now(),
        )

        assert decision.reviewer_id == reviewer.reviewer_id
        assert str(decision.reviewer_id) == str(reviewer.reviewer_id)
