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
from src.utils.logger import get_logger
from src.workflow.ai_workflow import LLMOrchestrator
from src.workflow.ocr_workflow import OCRWorkflow

router = APIRouter()
logger = get_logger(__name__)


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


@router.get("/student/{email}/applications", tags=["student"])
async def get_student_applications(email: str):
    """Get student applications with decision information."""
    student = get_student_by_email(email)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Get all certificates for the student with their decisions
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    c.certificate_id,
                    c.training_type,
                    c.filename,
                    c.uploaded_at,
                    d.ai_decision,
                    d.ai_justification,
                    d.created_at as decision_created_at
                FROM certificates c
                LEFT JOIN decisions d ON c.certificate_id = d.certificate_id
                WHERE c.student_id = %s
                ORDER BY c.uploaded_at DESC
            """,
                (str(student.student_id),),
            )

            rows = cur.fetchall()

            applications = []
            for row in rows:
                (
                    cert_id,
                    training_type,
                    filename,
                    uploaded_at,
                    ai_decision,
                    ai_justification,
                    decision_created_at,
                ) = row

                # Determine status and credits
                status = "PENDING"
                credits = 0

                if ai_decision:
                    status = ai_decision
                    # For now, set credits based on training type and decision
                    if ai_decision == "ACCEPTED":
                        credits = 30 if training_type == "PROFESSIONAL" else 10

                applications.append(
                    {
                        "certificate_id": str(cert_id),
                        "training_type": training_type,
                        "filename": filename,
                        "status": status,
                        "credits": credits,
                        "submitted_date": uploaded_at.isoformat()
                        if uploaded_at
                        else None,
                        "decision_date": decision_created_at.isoformat()
                        if decision_created_at
                        else None,
                        "justification": ai_justification,
                    }
                )

    return {"success": True, "applications": applications}


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

    # Create timestamped folder
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{file.filename.replace('.', '_')}"

    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Create the specific folder for this upload
    upload_folder = os.path.join(uploads_dir, folder_name)
    os.makedirs(upload_folder, exist_ok=True)

    # Save file in the timestamped folder
    file_location = os.path.join(upload_folder, file.filename)
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

    # Get the folder where the file was uploaded
    file_path = Path(cert.filepath)
    upload_folder = file_path.parent

    # Create timestamp for output files
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # OCR processing
    ocr_result = ocr_workflow.process_document(file_path)
    if not ocr_result.get("success"):
        raise HTTPException(status_code=500, detail="OCR processing failed")

    # Save OCR output to the same folder
    ocr_output_file = upload_folder / f"ocr_output_{file_path.stem}.txt"
    with open(ocr_output_file, "w", encoding="utf-8") as f:
        f.write(ocr_result.get("extracted_text", ""))

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

    try:
        llm_result = llm_orchestrator.process_work_certificate(
            cleaned_text,
            student_degree=student.degree,
            requested_training_type=cert.training_type.value.lower(),
        )

        # Save LLM output to the same folder
        llm_output_file = (
            upload_folder / f"aiworkflow_output_{file_path.stem}_{timestamp}.json"
        )
        with open(llm_output_file, "w", encoding="utf-8") as f:
            import json

            json.dump(llm_result, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"LLM processing failed: {e}")
        # Save error information to the same folder
        error_output_file = (
            upload_folder / f"llm_error_{file_path.stem}_{timestamp}.json"
        )
        error_data = {
            "error": f"LLM processing failed: {str(e)}",
            "timestamp": timestamp,
            "certificate_id": str(certificate_id),
        }
        with open(error_output_file, "w", encoding="utf-8") as f:
            import json

            json.dump(error_data, f, indent=2, ensure_ascii=False)

        # Return OCR results with LLM error
        return {
            "ocr_results": ocr_result,
            "llm_results": {
                "success": False,
                "error": f"LLM processing failed: {str(e)}",
                "evaluation_results": {
                    "results": {
                        "training_hours": 0,
                        "credits": 0,
                        "duration": "Not available",
                        "institution": "Not available",
                        "justification": f"LLM processing failed: {str(e)}",
                    }
                },
            },
            "decision": {
                "ai_decision": "PENDING",
                "ai_justification": f"LLM processing failed: {str(e)}",
            },
        }

    # Store decision using evaluation results directly
    evaluation_results = llm_result.get("evaluation_results", {}).get("results", {})

    decision = create_decision(
        certificate_id=certificate_id,
        ai_decision=DecisionStatus(evaluation_results.get("decision", "REJECTED")),
        ai_justification=evaluation_results.get(
            "justification", "LLM processing complete"
        ),
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
