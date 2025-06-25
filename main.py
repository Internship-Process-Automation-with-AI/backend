import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
import tempfile

from app.config import settings
from app.ocr_service import OCRService, OCRResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="OAMK Internship Certificate OCR Processing System"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR service
ocr_service = OCRService()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting OAMK Internship Certificate Processor")
    logger.info(f"Supported file formats: {ocr_service.get_supported_formats()}")
    logger.info(f"Google Vision API available: {ocr_service.is_google_vision_available()}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "OAMK Internship Certificate Processor",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ocr_service": "available",
        "google_vision": ocr_service.is_google_vision_available()
    }


@app.post("/api/v1/ocr/extract-text")
async def extract_text_from_file(
    file: UploadFile = File(...),
    use_preprocessing: bool = True
):
    """
    Extract text from uploaded file using OCR.
    
    Args:
        file: Uploaded file (PDF, PNG, JPG, etc.)
        use_preprocessing: Whether to apply image preprocessing
        
    Returns:
        JSON response with extracted text and metadata
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {list(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size
        file_size = 0
        file_content = b""
        
        # Read file content
        while chunk := await file.read(8192):
            file_size += len(chunk)
            if file_size > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
                )
            file_content += chunk
        
        logger.info(f"Processing file: {file.filename} ({file_size} bytes)")
        
        # Process file with OCR
        result = ocr_service.extract_text_from_bytes(
            file_content, 
            file_extension, 
            use_preprocessing
        )
        
        # Prepare response
        response_data = {
            "filename": file.filename,
            "file_size": file_size,
            "file_type": file_extension,
            "extracted_text": result.text,
            "confidence": result.confidence,
            "ocr_engine": result.engine,
            "processing_time": result.processing_time,
            "success": result.success,
            "preprocessing_applied": use_preprocessing
        }
        
        if not result.success:
            logger.warning(f"Low confidence OCR result for {file.filename}: {result.confidence}%")
            response_data["warning"] = "Low confidence OCR result. Consider checking the extracted text."
        
        return JSONResponse(content=response_data, status_code=200)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/ocr/extract-text-sync")
async def extract_text_sync(
    file: UploadFile = File(...),
    use_preprocessing: bool = True
):
    """
    Synchronous text extraction endpoint.
    """
    return await extract_text_from_file(file, use_preprocessing)


@app.get("/api/v1/ocr/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_formats": ocr_service.get_supported_formats(),
        "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024)
    }


@app.get("/api/v1/ocr/status")
async def get_ocr_status():
    """Get OCR service status and capabilities."""
    return {
        "service_status": "running",
        "tesseract_available": True,  # Assuming Tesseract is installed
        "google_vision_available": ocr_service.is_google_vision_available(),
        "preprocessing_enabled": settings.IMAGE_PREPROCESSING_ENABLED,
        "confidence_threshold": settings.OCR_CONFIDENCE_THRESHOLD
    }


@app.post("/api/v1/ocr/batch")
async def batch_extract_text(
    files: list[UploadFile] = File(...),
    use_preprocessing: bool = True
):
    """
    Extract text from multiple files in batch.
    
    Args:
        files: List of uploaded files
        use_preprocessing: Whether to apply image preprocessing
        
    Returns:
        JSON response with results for all files
    """
    try:
        results = []
        
        for file in files:
            try:
                # Process each file
                result = await extract_text_from_file(file, use_preprocessing)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "data": result.body.decode() if hasattr(result, 'body') else result
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_files": len(files),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"])
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 