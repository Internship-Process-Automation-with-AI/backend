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
    delete_certificate,
    get_all_reviewers,
    get_appeal_by_certificate_id,
    get_certificate_by_id,
    get_certificates_by_reviewer_id,
    get_db_connection,
    get_detailed_application,
    get_reviewer_by_email,
    get_student_by_email,
    get_student_by_id,
    get_student_with_certificates,
    submit_appeal,
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
                    d.created_at as decision_created_at,
                    d.reviewer_id,
                    d.reviewer_decision,
                    r.first_name,
                    r.last_name,
                    d.appeal_status,
                    d.appeal_submitted_at,
                    d.appeal_review_comment,
                    d.appeal_reviewed_at,
                    ar.first_name as appeal_reviewer_first_name,
                    ar.last_name as appeal_reviewer_last_name,
                    d.total_working_hours,
                    d.credits_awarded,
                    d.training_duration,
                    d.training_institution,
                    d.degree_relevance,
                    d.supporting_evidence,
                    d.challenging_evidence,
                    d.recommendation
                FROM certificates c
                LEFT JOIN decisions d ON c.certificate_id = d.certificate_id
                LEFT JOIN reviewers r ON d.reviewer_id = r.reviewer_id
                LEFT JOIN reviewers ar ON d.appeal_reviewer_id = ar.reviewer_id
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
                    reviewer_id,
                    reviewer_decision,
                    reviewer_first_name,
                    reviewer_last_name,
                    appeal_status,
                    appeal_submitted_at,
                    appeal_review_comment,
                    appeal_reviewed_at,
                    appeal_reviewer_first_name,
                    appeal_reviewer_last_name,
                    total_working_hours,
                    credits_awarded,
                    training_duration,
                    training_institution,
                    degree_relevance,
                    supporting_evidence,
                    challenging_evidence,
                    recommendation,
                ) = row

                # Determine status and credits
                status = "PENDING"
                credits = 0

                if ai_decision:
                    if reviewer_id and not reviewer_decision:
                        status = "PENDING_FOR_APPROVAL"
                    else:
                        status = ai_decision
                        # For now, set credits based on training type and decision
                        if ai_decision == "ACCEPTED":
                            credits = 30 if training_type == "PROFESSIONAL" else 10

                # Build reviewer name
                reviewer_name = None
                if reviewer_first_name and reviewer_last_name:
                    reviewer_name = f"{reviewer_first_name} {reviewer_last_name}"
                elif reviewer_first_name:
                    reviewer_name = reviewer_first_name
                elif reviewer_last_name:
                    reviewer_name = reviewer_last_name

                # Build appeal reviewer name
                appeal_reviewer_name = None
                if appeal_reviewer_first_name and appeal_reviewer_last_name:
                    appeal_reviewer_name = (
                        f"{appeal_reviewer_first_name} {appeal_reviewer_last_name}"
                    )
                elif appeal_reviewer_first_name:
                    appeal_reviewer_name = appeal_reviewer_first_name
                elif appeal_reviewer_last_name:
                    appeal_reviewer_name = appeal_reviewer_last_name

                # Add appeal information to status if appeal exists
                if appeal_status:
                    if appeal_status == "PENDING":
                        status = "APPEAL_PENDING"
                    elif appeal_status == "APPROVED":
                        status = "APPEAL_APPROVED"
                    elif appeal_status == "REJECTED":
                        status = "APPEAL_REJECTED"

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
                        "ai_decision": ai_decision,
                        "justification": ai_justification,
                        "reviewer_name": reviewer_name,
                        "appeal_status": appeal_status,
                        "appeal_submitted_date": appeal_submitted_at.isoformat()
                        if appeal_submitted_at
                        else None,
                        "appeal_review_comment": appeal_review_comment,
                        "appeal_reviewed_date": appeal_reviewed_at.isoformat()
                        if appeal_reviewed_at
                        else None,
                        "appeal_reviewer_name": appeal_reviewer_name,
                        # Evaluation details
                        "total_working_hours": total_working_hours,
                        "credits_awarded": credits_awarded,
                        "training_duration": training_duration,
                        "training_institution": training_institution,
                        "degree_relevance": degree_relevance,
                        "supporting_evidence": supporting_evidence,
                        "challenging_evidence": challenging_evidence,
                        "recommendation": recommendation,
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

    # Determine credits based on decision
    ai_decision = evaluation_results.get("decision", "REJECTED")
    credits_awarded = 0  # Default to 0 for rejected applications

    if ai_decision == "ACCEPTED":
        # Get credits from LLM response, default to 0 if not provided
        credits_awarded = evaluation_results.get("credits", 0)
        if credits_awarded is None:
            credits_awarded = 0

    decision = create_decision(
        certificate_id=certificate_id,
        ai_decision=DecisionStatus(ai_decision),
        ai_justification=evaluation_results.get(
            "justification", "LLM processing complete"
        ),
        total_working_hours=evaluation_results.get("total_working_hours"),
        credits_awarded=credits_awarded,
        training_duration=evaluation_results.get("duration"),
        training_institution=evaluation_results.get("institution"),
        degree_relevance=evaluation_results.get("degree_relevance"),
        supporting_evidence=evaluation_results.get("supporting_evidence"),
        challenging_evidence=evaluation_results.get("challenging_evidence"),
        recommendation=evaluation_results.get("recommendation"),
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

        # If no reviewers exist, create some sample ones
        if not reviewers:
            from src.database.database import create_sample_reviewers

            create_sample_reviewers()
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


@router.delete("/certificate/{certificate_id}", tags=["student"])
async def delete_certificate_endpoint(certificate_id: UUID):
    """Delete a certificate and its associated data."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    try:
        # Delete the certificate and all associated data
        delete_certificate(certificate_id)

        # Also delete the uploaded file if it exists
        if cert.filepath and os.path.exists(cert.filepath):
            os.remove(cert.filepath)
            # Try to remove the parent directory if it's empty
            parent_dir = os.path.dirname(cert.filepath)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)

        return {"success": True, "message": "Certificate deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting certificate {certificate_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete certificate")


class SendForApprovalRequest(BaseModel):
    reviewer_id: UUID


@router.post("/certificate/{certificate_id}/send-for-approval", tags=["student"])
async def send_for_approval_endpoint(
    certificate_id: UUID, payload: SendForApprovalRequest
):
    """Send a certificate for approval to a specific reviewer."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    try:
        # Update the decision record with the reviewer_id
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE decisions SET reviewer_id = %s WHERE certificate_id = %s",
                    (str(payload.reviewer_id), str(certificate_id)),
                )
                conn.commit()

        return {
            "success": True,
            "message": "Certificate sent for approval successfully",
        }
    except Exception as e:
        logger.error(
            f"Error sending certificate {certificate_id} for approval: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to send for approval")


class AppealRequest(BaseModel):
    appeal_reason: str


@router.post("/certificate/{certificate_id}/appeal", tags=["student"])
async def submit_appeal_endpoint(certificate_id: UUID, payload: AppealRequest):
    """Submit an appeal for a rejected certificate."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    # Check if appeal already exists
    existing_appeal = get_appeal_by_certificate_id(certificate_id)
    if existing_appeal:
        raise HTTPException(
            status_code=400, detail="Appeal already exists for this certificate"
        )

    try:
        # Get the first available reviewer (or you can implement logic to assign to a specific appeals reviewer)
        reviewers = get_all_reviewers()
        if not reviewers:
            raise HTTPException(status_code=500, detail="No reviewers available")

        # Assign to the first reviewer (you can modify this logic)
        assigned_reviewer = reviewers[0]

        # Submit appeal by updating the decision record
        success = submit_appeal(
            certificate_id, payload.appeal_reason, assigned_reviewer.reviewer_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to submit appeal")

        return {
            "success": True,
            "message": "Appeal submitted successfully",
            "assigned_reviewer": assigned_reviewer.to_dict(),
        }
    except Exception as e:
        logger.error(
            f"Error submitting appeal for certificate {certificate_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to submit appeal")


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
