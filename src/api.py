#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Application for OAMK Work Certificate Processor
Provides REST API endpoints for the frontend to interact with the document processing pipeline.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Local imports - ruff: noqa: E402
from file_manager import file_manager  # noqa: E402
from mainpipeline import DocumentPipeline  # noqa: E402

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


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    logger.info("Initializing document processing pipeline...")
    if not pipeline.initialize_services():
        logger.error("Failed to initialize pipeline services")
        raise RuntimeError("Pipeline initialization failed")
    logger.info("Pipeline initialized successfully")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "OAMK Work Certificate Processor API", "status": "running"}


@app.get("/api/degrees")
async def get_degree_programs():
    """Get list of supported degree programs."""
    try:
        if pipeline.degree_evaluator:
            degrees = pipeline.degree_evaluator.get_supported_degree_programs()
            return {"degrees": degrees}
        else:
            raise HTTPException(
                status_code=500, detail="Degree evaluator not initialized"
            )
    except Exception as e:
        logger.error(f"Error getting degree programs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process", response_model=ProcessingResponse)
async def process_document(
    file: UploadFile = File(...),
    student_degree: str = Form(...),
    student_email: str = Form(...),
    training_type: str = Form(...),
):
    """
    Process a work certificate document.

    Args:
        file: The uploaded document file (PDF, image, etc.)
        student_degree: The student's degree program
        student_email: The student's email address
        training_type: Type of training (general or professional)

    Returns:
        Processing results including OCR and LLM evaluation
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
            )

        # Validate email format
        if not student_email.lower().endswith("@students.oamk.fi"):
            raise HTTPException(
                status_code=400,
                detail="Email must be a valid OAMK student email (@students.oamk.fi)",
            )

        # Validate training type
        if training_type not in ["general", "professional"]:
            raise HTTPException(
                status_code=400,
                detail="Training type must be 'general' or 'professional'",
            )

        # Read file content
        content = await file.read()

        # Use the same timestamp for both file save and processing results
        processing_date = datetime.now()

        # Save uploaded file to organized uploads directory
        saved_file_path, unique_filename = file_manager.save_uploaded_file(
            file_content=content, original_filename=file.filename, date=processing_date
        )

        logger.info(f"Processing file: {file.filename}")
        logger.info(f"Saved to: {saved_file_path}")
        logger.info(f"Student degree: {student_degree}")
        logger.info(f"Student email: {student_email}")
        logger.info(f"Training type: {training_type}")

        # Debug: Check LLM orchestrator status before processing
        if hasattr(pipeline, "orchestrator") and pipeline.orchestrator:
            orchestrator_stats = pipeline.orchestrator.get_stats()
            print(f"🔍 LLM Orchestrator Status: {orchestrator_stats}")
        else:
            print("❌ LLM Orchestrator not initialized")

        # Process the document using the pipeline (directly from saved file)
        results = pipeline.process_document(
            file_path=str(saved_file_path),
            student_degree=student_degree,
            student_email=student_email,
            training_type=training_type,
        )

        # Ensure OCR text is included in results for file manager
        if "ocr_results" in results:
            ocr_results = results["ocr_results"]
            # Get the extracted text from the correct field
            ocr_text = ocr_results.get("extracted_text", "") or ocr_results.get(
                "text", ""
            )
            if ocr_text and "extracted_text" not in ocr_results:
                results["ocr_results"]["extracted_text"] = ocr_text

        # Save processing results alongside the uploaded file
        results_path, ocr_text_path = file_manager.save_processing_results(
            results=results, original_filename=file.filename, date=processing_date
        )

        # Format response
        response_data = {
            "success": True,
            "file_path": file.filename,
            "student_degree": student_degree,
            "student_email": student_email,
            "requested_training_type": training_type,
            "processing_time": results.get("processing_time", 0),
            "ocr_results": {
                "success": results.get("ocr_results", {}).get("success", False),
                "engine": results.get("ocr_results", {}).get("engine", ""),
                "confidence": results.get("ocr_results", {}).get("confidence", 0),
                "processing_time": results.get("ocr_results", {}).get(
                    "processing_time", 0
                ),
                "text_length": results.get("ocr_results", {}).get("text_length", 0),
                "detected_language": results.get("ocr_results", {}).get(
                    "detected_language", ""
                ),
                "finnish_chars_count": results.get("ocr_results", {}).get(
                    "finnish_chars_count", 0
                ),
            },
            "llm_results": results.get("llm_results", {}),
            "storage_info": {
                "document_folder": str(
                    file_manager.get_document_folder(file.filename, processing_date)
                ),
                "results_file": str(results_path),
                "ocr_text_file": str(ocr_text_path),
            },
        }

        logger.info("Document processing completed successfully")
        return ProcessingResponse(**response_data)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/detect-language")
async def detect_language(file: UploadFile = File(...)):
    """
    Detect the language of an uploaded document.

    Args:
        file: The uploaded document file

    Returns:
        Language detection results
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
            )

        # Read file content
        content = await file.read()

        # Save file temporarily for language detection
        processing_date = datetime.now()
        saved_file_path, unique_filename = file_manager.save_uploaded_file(
            file_content=content, original_filename=file.filename, date=processing_date
        )

        try:
            # Use OCR workflow to detect language
            if pipeline.ocr_workflow:
                file_path = Path(saved_file_path)
                detected_lang = pipeline.ocr_workflow._detect_language(file_path)

                # Also try content-based detection
                try:
                    from ocr.cert_extractor import extract_certificate_text

                    quick_text = extract_certificate_text(file_path, language="fin")
                    finnish_chars = sum(1 for c in quick_text.lower() if c in "äöå")

                    return {
                        "filename": file.filename,
                        "filename_based_detection": detected_lang,
                        "content_based_detection": {
                            "finnish_characters": finnish_chars,
                            "suggests_finnish": finnish_chars > 0,
                            "text_preview": quick_text[:200] + "..."
                            if len(quick_text) > 200
                            else quick_text,
                        },
                        "recommended_language": "fin"
                        if finnish_chars > 0
                        else detected_lang,
                    }
                except Exception as e:
                    return {
                        "filename": file.filename,
                        "filename_based_detection": detected_lang,
                        "content_based_detection": {"error": str(e)},
                        "recommended_language": detected_lang,
                    }
            else:
                raise HTTPException(
                    status_code=500, detail="OCR workflow not initialized"
                )
        finally:
            # Clean up the saved file since this is just for language detection
            try:
                if saved_file_path.exists():
                    saved_file_path.unlink()
                    # Also clean up the directory if it's empty
                    document_folder = file_manager.get_document_folder(
                        file.filename, processing_date
                    )
                    if document_folder.exists() and not any(document_folder.iterdir()):
                        document_folder.rmdir()
            except Exception as e:
                logger.warning(f"Failed to cleanup language detection file: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        raise HTTPException(
            status_code=500, detail=f"Language detection failed: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        # Check if pipeline components are available
        ocr_available = pipeline.ocr_workflow is not None
        llm_available = (
            pipeline.orchestrator is not None and pipeline.orchestrator.is_available()
        )
        degree_available = pipeline.degree_evaluator is not None

        return {
            "status": "healthy",
            "components": {
                "ocr_workflow": ocr_available,
                "llm_orchestrator": llm_available,
                "degree_evaluator": degree_available,
            },
            "all_components_healthy": ocr_available
            and llm_available
            and degree_available,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "all_components_healthy": False}


@app.get("/api/files")
async def list_uploaded_files(date: Optional[str] = None, limit: int = 50):
    """List uploaded files with optional date filtering."""
    try:
        # Parse date if provided (format: YYYY-MM-DD)
        target_date = None
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
                )

        files = file_manager.list_uploaded_files(date=target_date, limit=limit)
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/stats")
async def get_storage_stats():
    """Get storage statistics for uploaded files."""
    try:
        stats = file_manager.get_storage_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/document/{filename}")
async def get_document_files(filename: str, date: Optional[str] = None):
    """Get all files related to a specific document."""
    try:
        # Parse date if provided (format: YYYY-MM-DD)
        target_date = None
        if date:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
                )

        # Get document files
        files = file_manager.get_document_files(filename, date=target_date)

        # Check which files exist
        file_status = {}
        for key, file_path in files.items():
            if key == "folder":
                file_status[key] = {
                    "path": str(file_path),
                    "exists": file_path.exists(),
                }
            else:
                file_status[key] = {
                    "path": str(file_path),
                    "exists": file_path.exists(),
                    "size": file_path.stat().st_size if file_path.exists() else 0,
                }

        return {"filename": filename, "date": date or "current", "files": file_status}
    except Exception as e:
        logger.error(f"Error getting document files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
