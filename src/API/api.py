import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.database.database import (
    add_student_feedback,
    create_certificate,
    create_decision,
    get_all_reviewers,
    get_certificate_by_id,
    get_certificates_by_reviewer_id,
    get_db_connection,
    get_detailed_application,
    get_reviewer_by_email,
    get_student_by_email,
    get_student_by_id,
    get_student_with_certificates,
    update_decision_review,
)
from src.database.models import DecisionStatus, ReviewerDecision, TrainingType
from src.workflow.ai_workflow import LLMOrchestrator
from src.workflow.ocr_workflow import OCRWorkflow

router = APIRouter()


@router.get("/student/{email}", tags=["student"])
async def student_lookup(email: str):
    """Lookup student by email and return student_id, degree, first_name, last_name, and certificate list."""
    student = get_student_by_email(email)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student_with_certs = get_student_with_certificates(student.student_id)
    cert_list = (
        [cert.to_dict() for cert in student_with_certs.certificates]
        if student_with_certs and student_with_certs.certificates
        else []
    )
    return {
        "student_id": str(student.student_id),
        "degree": student.degree,
        "first_name": student.first_name,
        "last_name": student.last_name,
        "certificates": cert_list,
    }


@router.post("/student/{student_id}/upload-certificate", tags=["student"])
async def upload_certificate(
    student_id: UUID,
    training_type: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a working certificate file for a student (by id)."""
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    try:
        training_type_enum = TrainingType(training_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid training type")
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_location = os.path.join(uploads_dir, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    certificate = create_certificate(
        student_id=student.student_id,
        training_type=training_type_enum,
        filename=file.filename,
        filetype=os.path.splitext(file.filename)[1][1:],
        filepath=file_location,
    )
    return {
        "success": True,
        "certificate_id": str(certificate.certificate_id),
        "certificate": certificate.to_dict(),
    }


ocr_workflow = OCRWorkflow()
llm_orchestrator = LLMOrchestrator()


@router.post("/certificate/{certificate_id}/process", tags=["student"])
async def process_certificate(certificate_id: UUID):
    """Process an uploaded certificate: OCR + LLM evaluation, store decision, return results."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    student = get_student_by_id(cert.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    # OCR processing
    ocr_result = ocr_workflow.process_document(Path(cert.filepath))
    if not ocr_result.get("success"):
        raise HTTPException(status_code=500, detail="OCR processing failed")
    # update certificate table with ocr_output
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE certificates SET ocr_output=%s WHERE certificate_id=%s",
                (ocr_result.get("extracted_text", ""), str(certificate_id)),
            )
            conn.commit()
    # Run LLM evaluation
    cleaned_text = ocr_result.get("extracted_text", "")
    llm_result = llm_orchestrator.process_work_certificate(
        cleaned_text,
        student_degree=student.degree,
        requested_training_type=cert.training_type.value.lower(),
    )
    # Store decision
    decision = create_decision(
        certificate_id=certificate_id,
        ai_decision=DecisionStatus.ACCEPTED
        if llm_result.get("success")
        else DecisionStatus.REJECTED,
        ai_justification=llm_result.get("evaluation_results", {})
        .get("results", {})
        .get("justification", "LLM processing complete"),
    )
    return {
        "ocr_results": ocr_result,
        "llm_results": llm_result,
        "decision": decision.to_dict(),
    }


@router.get("/reviewers", tags=["student"])
async def get_reviewers():
    """Get all reviewers with their names and IDs."""
    try:
        reviewers = get_all_reviewers()
        return {
            "success": True,
            "reviewers": [reviewer.to_dict() for reviewer in reviewers],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching reviewers: {str(e)}"
        )


class FeedbackRequest(BaseModel):
    student_feedback: str
    reviewer_id: Optional[UUID] = None


@router.post("/certificate/{certificate_id}/feedback", tags=["student"])
async def add_feedback_endpoint(certificate_id: UUID, payload: FeedbackRequest):
    """Store student feedback and reviewer ID for a decision/certificate."""
    success = add_student_feedback(
        certificate_id, payload.student_feedback, payload.reviewer_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return {"success": True, "message": "Feedback and reviewer information stored"}


@router.get("/reviewer/{email}", tags=["reviewer"])
async def reviewer_lookup(email: str):
    """Lookup reviewer by email and return reviewer_id and other information."""
    reviewer = get_reviewer_by_email(email)
    if not reviewer:
        raise HTTPException(status_code=404, detail="Reviewer not found")
    return {"success": True, "reviewer": reviewer.to_dict()}


@router.get("/reviewer/{reviewer_id}/certificates", tags=["reviewer"])
async def get_reviewer_certificates(reviewer_id: UUID):
    """Get all certificates assigned to a reviewer."""
    try:
        applications = get_certificates_by_reviewer_id(reviewer_id)
        return {
            "success": True,
            "applications": [app.to_dict() for app in applications],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching reviewer certificates: {str(e)}"
        )


@router.get("/certificate/{certificate_id}", tags=["reviewer"])
async def download_certificate_file(certificate_id: UUID):
    """Return the actual certificate file as a download."""
    cert = get_certificate_by_id(certificate_id)
    if not cert or not cert.filepath:
        raise HTTPException(status_code=404, detail="Certificate file not found")

    return FileResponse(
        path=cert.filepath,
        media_type="application/octet-stream",
        filename=cert.filename,
    )


# If the frontend still needs metadata, expose it under a new path
@router.get("/certificate/{certificate_id}/details", tags=["reviewer"])
async def get_certificate_details(certificate_id: UUID):
    """Get detailed information about a specific certificate (moved from the root path)."""
    application = get_detailed_application(certificate_id)
    if not application:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return {"success": True, "application": application.to_dict()}


class ReviewUpdateRequest(BaseModel):
    """Payload for reviewer to submit their decision."""

    reviewer_comment: str
    reviewer_decision: str  # "PASS" or "FAIL"


@router.post("/certificate/{certificate_id}/review", tags=["reviewer"])
async def update_certificate_review(certificate_id: UUID, payload: ReviewUpdateRequest):
    """Add a reviewer comment and mark the certificate as *REVIEWED*."""

    # Validate decision
    try:
        decision_enum = ReviewerDecision(payload.reviewer_decision.upper())
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid reviewer_decision. Use 'PASS' or 'FAIL'."
        )

    success, error_msg = update_decision_review(
        certificate_id=certificate_id,
        reviewer_comment=payload.reviewer_comment,
        reviewer_decision=decision_enum,
    )
    if not success:
        status_code = 404 if error_msg == "Certificate not found" else 400
        raise HTTPException(status_code=status_code, detail=error_msg)

    return {
        "success": True,
        "message": f"Certificate review updated successfully. Decision set to {decision_enum.value}",
    }
