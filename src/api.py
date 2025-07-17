"""
FastAPI Application for OAMK Work Certificate Processor
Provides REST API endpoints for the frontend to interact with the document processing pipeline.
"""

import logging
import os
import sys
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from file_manager import file_manager
from mainpipeline import DocumentPipeline
from pydantic import BaseModel

from src.database.database import (
    add_student_feedback,
    check_certificate_limits,
    check_database_health,
    create_certificate,
    create_decision,
    create_student,
    get_applications_by_status,
    get_certificate_by_id,
    get_certificates,
    get_decision_by_id,
    get_decisions,
    get_detailed_application,
    get_pending_applications,
    get_statistics,
    get_student_by_email,
    get_student_with_certificates,
    get_students,
    init_database,
    update_decision_review,
)
from src.database.models import DecisionStatus, ReviewStatus, TrainingType

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OAMK Work Certificate Processor API",
    description="API for processing work certificates and evaluating academic credits",
    version="1.0.0",
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the document pipeline
pipeline = DocumentPipeline()


# Pydantic models for request/response
class ProcessingRequest(BaseModel):
    student_degree: str
    student_email: str
    training_type: str


class ProcessingResponse(BaseModel):
    success: bool
    file_path: str
    student_degree: str
    student_email: str
    requested_training_type: str
    processing_time: float
    ocr_results: dict
    llm_results: dict
    error: Optional[str] = None


# Database response models
class StudentResponse(BaseModel):
    student_id: UUID
    email: str
    degree: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CertificateResponse(BaseModel):
    certificate_id: UUID
    student_id: UUID
    training_type: str
    filename: str
    filetype: str
    filepath: Optional[str]
    uploaded_at: datetime

    class Config:
        from_attributes = True


class DecisionResponse(BaseModel):
    decision_id: UUID
    certificate_id: UUID
    ocr_output: Optional[str]
    ai_decision: str
    justification: str
    created_at: datetime
    student_feedback: Optional[str]
    review_status: str
    reviewer_comment: Optional[str]
    reviewed_at: Optional[datetime]

    class Config:
        from_attributes = True


class StudentWithCertificates(BaseModel):
    student_id: UUID
    email: str
    degree: str
    created_at: datetime
    updated_at: datetime
    certificates: List[CertificateResponse]

    class Config:
        from_attributes = True


# Reviewer-specific models
class ApplicationSummaryResponse(BaseModel):
    decision_id: UUID
    certificate_id: UUID
    student_name: str
    student_email: str
    student_degree: str
    filename: str
    training_type: str
    ai_decision: str
    review_status: str
    uploaded_at: datetime
    created_at: datetime
    student_feedback: Optional[str]

    class Config:
        from_attributes = True


class DetailedApplicationResponse(BaseModel):
    decision: DecisionResponse
    certificate: CertificateResponse
    student: StudentResponse

    class Config:
        from_attributes = True


class ReviewSubmissionRequest(BaseModel):
    certificate_id: UUID
    reviewer_comment: str
    review_status: str = "REVIEWED"
    student_feedback: Optional[str] = None


class StudentFeedbackRequest(BaseModel):
    certificate_id: UUID
    student_feedback: str


@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup checks."""
    try:
        # Initialize database
        init_database()
        logger.info("Database initialized successfully")

        # Initialize pipeline services
        if pipeline.initialize_services():
            logger.info("Pipeline services initialized successfully")
        else:
            logger.warning(
                "Pipeline services initialization failed - some features may not work"
            )

        # Check if file manager is working
        storage_status = file_manager.get_storage_stats()
        logger.info(f"File manager status: {storage_status}")

        # Check database health
        health = check_database_health()
        logger.info(f"Database health: {health}")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {"message": "OAMK Work Certificate Processor API", "version": "1.0.0"}


@app.get("/api/degrees")
async def get_degree_programs():
    """Get list of supported degree programs."""
    try:
        # Try to use the degree evaluator if available
        if pipeline.degree_evaluator:
            degrees = pipeline.degree_evaluator.get_supported_degree_programs()
            return {"degrees": degrees}
        else:
            # Fallback to importing the data directly
            from llm.degree_evaluator import DegreeEvaluator

            evaluator = DegreeEvaluator()
            degrees = evaluator.get_supported_degree_programs()
            return {"degrees": degrees}
    except Exception as e:
        logger.error(f"Error getting degree programs: {e}")
        # Fallback to static list if all else fails
        degrees = [
            "Bachelor of Engineering (BEng), Information Technology",
            "Insinööri (AMK), tieto- ja viestintätekniikka",
            "Tradenomi (AMK), tietojenkäsittely",
            "Bachelor of Business Administration (BBA), International Business",
            "Tradenomi (AMK), liiketalous",
            "Tradenomi (AMK), liiketalous, verkkokoulutus",
            "Medianomi (AMK)",
            "Musiikkipedagogi (AMK)",
            "Tanssinopettaja (AMK)",
            "Agrologi (AMK), maaseutuelinkeinot",
            "Bachelor of Health Care, Nursing",
            "Bioanalyytikko (AMK)",
            "Ensihoitaja (AMK)",
            "Fysioterapeutti (AMK)",
            "Kätilö (AMK)",
            "Hammashoitaja (AMK)",
            "Kuntoutuksen ohjaaja (AMK)",
            "Laboratoriohoitaja (AMK)",
            "Optometristi (AMK)",
            "Röntgenhoitaja (AMK)",
            "Sairaanhoitaja (AMK)",
            "Sosionomi (AMK)",
            "Suuhygienisti (AMK)",
            "Toimintaterapeutti (AMK)",
            "Sosiaali- ja terveysalan johtaja (AMK)",
            "Terveydenhoitaja (AMK)",
            "Bachelor of Engineering, Energy and Environmental Engineering",
            "Bachelor of Engineering, Mechanical Engineering",
            "Insinööri (AMK), energia- ja ympäristötekniikka",
            "Insinööri (AMK), konetekniikka",
            "Insinööri (AMK), sähkö- ja automaatiotekniikka",
            "Insinööri (AMK), talotekniikka",
            "Insinööri (AMK), rakennus- ja yhdyskuntatekniikka",
            "Rakennusarkkitehti (AMK)",
            "Rakennusmestari (AMK)",
        ]
        return {"degrees": degrees}


@app.post("/api/process", response_model=ProcessingResponse)
async def process_document(
    file: UploadFile = File(...),
    student_degree: str = Form(...),
    student_email: str = Form(...),
    training_type: str = Form(...),
):
    """
    Process a work certificate document for academic credit evaluation.

    This endpoint:
    1. Validates the uploaded file
    2. Stores the document
    3. Creates/gets student record
    4. Creates certificate record
    5. Runs OCR and AI analysis
    6. Stores the decision
    7. Returns comprehensive results
    """
    processing_start = datetime.now()

    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate training type
        try:
            training_type_enum = TrainingType(training_type.upper())
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid training type: {training_type}"
            )

        # Validate email format
        if not student_email.endswith("@students.oamk.fi"):
            raise HTTPException(
                status_code=400, detail="Email must be @students.oamk.fi domain"
            )

        # Read file content
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer for potential reuse

        # Store the uploaded file
        try:
            file_path, saved_filename = file_manager.save_uploaded_file(
                file_content, file.filename
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"File storage failed: {str(e)}"
            )

        # Get or create student
        student = get_student_by_email(student_email)
        if not student:
            student = create_student(student_email, student_degree)
        else:
            # Update student's degree if it has changed
            if student.degree != student_degree:
                # Note: We could add an update_student_degree function if needed
                logger.info(
                    f"Student degree changed from {student.degree} to {student_degree}"
                )

        # Check certificate limits before creating new certificate
        is_allowed, error_message = check_certificate_limits(
            student.student_id, training_type_enum
        )
        if not is_allowed:
            raise HTTPException(status_code=400, detail=error_message)

        # Create certificate record
        certificate = create_certificate(
            student_id=student.student_id,
            training_type=training_type_enum,
            filename=file.filename,
            filetype=file.filename.split(".")[-1].lower()
            if "." in file.filename
            else "unknown",
            filepath=str(file_path),
        )

        # Process the document through the pipeline
        pipeline_result = pipeline.process_document(
            file_path=file_path,
            student_degree=student_degree,
            student_email=student_email,
            training_type=training_type,
        )

        # Extract decision information from pipeline results
        ai_decision = DecisionStatus.ACCEPTED
        justification = "Document processed successfully"
        ocr_output = None

        if (
            "llm_results" in pipeline_result
            and "evaluation_results" in pipeline_result["llm_results"]
        ):
            eval_results = pipeline_result["llm_results"]["evaluation_results"].get(
                "results", {}
            )
            decision_text = eval_results.get("decision", "ACCEPTED").upper()
            ai_decision = (
                DecisionStatus.ACCEPTED
                if decision_text == "ACCEPTED"
                else DecisionStatus.REJECTED
            )
            justification = eval_results.get("justification", justification)

        if "ocr_results" in pipeline_result:
            ocr_output = str(pipeline_result["ocr_results"])

            # Create decision record
            create_decision(
                certificate_id=certificate.certificate_id,
                ocr_output=ocr_output,
                ai_decision=ai_decision,
                ai_justification=justification,
                review_status=ReviewStatus.PENDING,
            )

        processing_time = (datetime.now() - processing_start).total_seconds()

        # Return comprehensive response
        return ProcessingResponse(
            success=True,
            file_path=str(file_path),
            student_degree=student_degree,
            student_email=student_email,
            requested_training_type=training_type,
            processing_time=processing_time,
            ocr_results=pipeline_result.get("ocr_results", {}),
            llm_results=pipeline_result.get("llm_results", {}),
            storage_info={
                "file_path": str(file_path),
                "saved_filename": saved_filename,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        processing_time = (datetime.now() - processing_start).total_seconds()
        return ProcessingResponse(
            success=False,
            file_path="",
            student_degree=student_degree,
            student_email=student_email,
            requested_training_type=training_type,
            processing_time=processing_time,
            ocr_results={},
            llm_results={},
            error=str(e),
        )


@app.post("/api/feedback")
async def add_feedback(request: StudentFeedbackRequest):
    """Add student feedback to a decision."""
    try:
        success = add_student_feedback(request.certificate_id, request.student_feedback)
        if not success:
            raise HTTPException(status_code=404, detail="Certificate not found")

        return {"success": True, "message": "Feedback added successfully"}

    except Exception as e:
        logger.error(f"Failed to add feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reviewer API endpoints
@app.get("/api/review/pending", response_model=List[ApplicationSummaryResponse])
async def get_pending_reviews():
    """Get all applications with PENDING review status."""
    try:
        applications = get_pending_applications()
        return [ApplicationSummaryResponse(**app.to_dict()) for app in applications]
    except Exception as e:
        logger.error(f"Failed to get pending reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/status/{status}", response_model=List[ApplicationSummaryResponse])
async def get_applications_by_review_status(status: str):
    """Get applications by review status."""
    try:
        # Validate status
        if status.upper() not in ["PENDING", "REVIEWED"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid status. Must be 'pending' or 'reviewed'",
            )

        review_status = ReviewStatus(status.upper())
        applications = get_applications_by_status(review_status)
        return [ApplicationSummaryResponse(**app.to_dict()) for app in applications]

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid review status")
    except Exception as e:
        logger.error(f"Failed to get applications by status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/{certificate_id}", response_model=DetailedApplicationResponse)
async def get_application_details(certificate_id: UUID):
    """Get detailed application information by certificate ID."""
    try:
        application = get_detailed_application(certificate_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        return DetailedApplicationResponse(**application.to_dict())

    except Exception as e:
        logger.error(f"Failed to get application details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/submit")
async def submit_review(request: ReviewSubmissionRequest):
    """Submit a review decision for an application."""
    try:
        # Validate review status
        if request.review_status.upper() not in ["REVIEWED"]:
            raise HTTPException(status_code=400, detail="Invalid review status")

        review_status = ReviewStatus(request.review_status.upper())

        success = update_decision_review(
            certificate_id=request.certificate_id,
            reviewer_comment=request.reviewer_comment,
            review_status=review_status,
            student_feedback=request.student_feedback,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Application not found")

        return {"success": True, "message": "Review submitted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Other existing endpoints (updated for new schema)
@app.get("/api/detect-language")
async def detect_language(file: UploadFile = File(...)):
    """Detect the language of the uploaded document."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer

        # Save to temporary location for processing
        temp_result = file_manager.save_uploaded_file(content, file.filename)

        if not temp_result[0]:  # Check if the first element of the tuple is False
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {temp_result[1]}"
            )

        try:
            # Run language detection through the pipeline
            result = pipeline.detect_language(temp_result[0])

            # Clean up temp file
            try:
                os.remove(temp_result[0])
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")

                return {
                    "success": True,
                    "language": result.get("language", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "details": result.get("details", {}),
                }

        except Exception as processing_error:
            # Clean up temp file on error
            try:
                os.remove(temp_result[0])
            except Exception:
                pass
            raise processing_error

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Comprehensive health check endpoint.

    Checks:
    - API status
    - Database connectivity
    - File storage
    - Document pipeline
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {},
    }

    # Check database
    try:
        db_health = check_database_health()
        health_status["checks"]["database"] = db_health
        if db_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # Check file storage
    try:
        storage_info = file_manager.get_storage_stats()
        health_status["checks"]["storage"] = {
            "status": "healthy" if storage_info["available"] else "degraded",
            "info": storage_info,
        }
        if not storage_info["available"]:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["storage"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    # Check document pipeline
    try:
        pipeline_status = pipeline.health_check()
        health_status["checks"]["pipeline"] = pipeline_status
        if pipeline_status.get("status") != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["pipeline"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    return health_status


@app.get("/api/files")
async def list_uploaded_files(date: Optional[str] = None, limit: int = 50):
    """List uploaded files with optional date filtering."""
    try:
        files = file_manager.list_files(date=date, limit=limit)
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/stats")
async def get_storage_stats():
    """Get file storage statistics."""
    try:
        stats = file_manager.get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/document/{filename}")
async def get_document_files(filename: str, date: Optional[str] = None):
    """
    Get all files related to a specific document.

    Args:
        filename: The original document filename
        date: Optional date in YYYY-MM-DD format

    Returns:
        Dictionary containing document files and metadata
    """
    try:
        # Find the document folder
        document_info = file_manager.get_document_info(filename, date)

        if not document_info:
            raise HTTPException(status_code=404, detail="Document not found")

        return document_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Database API endpoints
@app.get("/api/students", response_model=List[StudentResponse])
async def get_students_endpoint(skip: int = 0, limit: int = 100):
    """Get list of students with pagination."""
    try:
        students = get_students(skip=skip, limit=limit)
        return [StudentResponse(**student.to_dict()) for student in students]
    except Exception as e:
        logger.error(f"Failed to get students: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}", response_model=StudentWithCertificates)
async def get_student_endpoint(student_id: UUID):
    """Get student with their certificates."""
    try:
        student = get_student_with_certificates(student_id)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        return StudentWithCertificates(**student.to_dict())
    except Exception as e:
        logger.error(f"Failed to get student: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/email/{email}", response_model=StudentWithCertificates)
async def get_student_by_email_endpoint(email: str):
    """Get student by email with their certificates."""
    try:
        student = get_student_by_email(email)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        # Get student with certificates
        student_with_certs = get_student_with_certificates(student.student_id)
        if not student_with_certs:
            raise HTTPException(status_code=404, detail="Student data not found")

        return StudentWithCertificates(**student_with_certs.to_dict())
    except Exception as e:
        logger.error(f"Failed to get student by email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/{student_id}/certificates/count")
async def get_student_certificate_count_endpoint(student_id: UUID):
    """Get certificate count by training type for a student."""
    try:
        from src.database.database import get_student_certificate_count

        counts = get_student_certificate_count(student_id)
        return {
            "student_id": str(student_id),
            "certificate_counts": counts,
            "limits": {"general": 1, "professional": 2},
            "can_upload_general": counts["GENERAL"] < 1,
            "can_upload_professional": counts["PROFESSIONAL"] < 2,
        }
    except Exception as e:
        logger.error(f"Failed to get certificate count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/students/email/{email}/certificates/count")
async def get_student_certificate_count_by_email_endpoint(email: str):
    """Get certificate count by training type for a student by email."""
    try:
        from src.database.database import get_student_certificate_count

        # Get student by email first
        student = get_student_by_email(email)
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        counts = get_student_certificate_count(student.student_id)
        return {
            "student_id": str(student.student_id),
            "student_email": email,
            "certificate_counts": counts,
            "limits": {"general": 1, "professional": 2},
            "can_upload_general": counts["GENERAL"] < 1,
            "can_upload_professional": counts["PROFESSIONAL"] < 2,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get certificate count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/certificates", response_model=List[CertificateResponse])
async def get_certificates_endpoint(skip: int = 0, limit: int = 100):
    """Get list of certificates with pagination."""
    try:
        certificates = get_certificates(skip=skip, limit=limit)
        return [CertificateResponse(**cert.to_dict()) for cert in certificates]
    except Exception as e:
        logger.error(f"Failed to get certificates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/certificates/{certificate_id}", response_model=CertificateResponse)
async def get_certificate_endpoint(certificate_id: UUID):
    """Get certificate by ID."""
    try:
        certificate = get_certificate_by_id(certificate_id)
        if not certificate:
            raise HTTPException(status_code=404, detail="Certificate not found")

        return CertificateResponse(**certificate.to_dict())
    except Exception as e:
        logger.error(f"Failed to get certificate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions", response_model=List[DecisionResponse])
async def get_decisions_endpoint(skip: int = 0, limit: int = 100):
    """Get list of decisions with pagination."""
    try:
        decisions = get_decisions(skip=skip, limit=limit)
        return [DecisionResponse(**decision.to_dict()) for decision in decisions]
    except Exception as e:
        logger.error(f"Failed to get decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/decisions/{decision_id}", response_model=DecisionResponse)
async def get_decision_endpoint(decision_id: UUID):
    """Get decision by ID."""
    try:
        decision = get_decision_by_id(decision_id)
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")

        return DecisionResponse(**decision.to_dict())
    except Exception as e:
        logger.error(f"Failed to get decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics_endpoint():
    """Get database statistics."""
    try:
        stats = get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
