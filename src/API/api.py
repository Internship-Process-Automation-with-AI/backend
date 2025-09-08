import os
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.responses import Response

from src.database.database import (
    add_student_comment,
    add_student_comment_and_reviewer,
    create_additional_document,
    create_certificate,
    create_decision,
    delete_certificate,
    get_additional_documents,
    get_all_reviewers,
    get_certificate_by_id,
    get_certificates_by_reviewer_id,
    get_db_connection,
    get_detailed_application,
    get_reviewer_by_email,
    get_student_by_email,
    get_student_by_id,
    get_student_comment_by_certificate_id,
    get_student_with_certificates,
    update_decision_review,
)
from src.database.models import (
    DecisionStatus,
    DocumentType,
    ReviewerDecision,
    TrainingType,
    WorkType,
)
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
                    c.work_type,
                    c.filename,
                    c.uploaded_at,
                    d.ai_decision,
                    d.ai_justification,
                    d.created_at as decision_created_at,
                    d.reviewer_id,
                    d.reviewer_decision,
                    d.reviewer_comment,
                    d.reviewed_at,
                    r.first_name,
                    r.last_name,
                    d.total_working_hours,
                    d.credits_awarded,
                    d.training_duration,
                    d.training_institution,
                    d.degree_relevance,
                    d.supporting_evidence,
                    d.challenging_evidence,
                    d.recommendation,
                    d.student_comment
                FROM certificates c
                LEFT JOIN decisions d ON c.certificate_id = d.certificate_id
                LEFT JOIN reviewers r ON d.reviewer_id = r.reviewer_id
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
                    work_type,
                    filename,
                    uploaded_at,
                    ai_decision,
                    ai_justification,
                    decision_created_at,
                    reviewer_id,
                    reviewer_decision,
                    reviewer_comment,
                    reviewed_at,
                    reviewer_first_name,
                    reviewer_last_name,
                    total_working_hours,
                    credits_awarded,
                    training_duration,
                    training_institution,
                    degree_relevance,
                    supporting_evidence,
                    challenging_evidence,
                    recommendation,
                    student_comment,
                ) = row

                # Determine status and credits
                status = "PENDING"
                credits = credits_awarded or 0  # Use actual credits from database

                if ai_decision:
                    if reviewer_id and not reviewer_decision:
                        status = "PENDING_FOR_APPROVAL"
                    elif ai_decision in ["ACCEPTED", "REJECTED"] and not reviewer_id:
                        # AI has made a decision but application hasn't been sent for approval
                        status = "AI_PROCESSED_NOT_SENT"
                    else:
                        status = ai_decision

                # Build reviewer name
                reviewer_name = None
                if reviewer_first_name and reviewer_last_name:
                    reviewer_name = f"{reviewer_first_name} {reviewer_last_name}"
                elif reviewer_first_name:
                    reviewer_name = reviewer_first_name
                elif reviewer_last_name:
                    reviewer_name = reviewer_last_name

                # Get additional documents for self-paced work
                additional_documents = []
                if work_type == "SELF_PACED":
                    additional_docs = get_additional_documents(cert_id)
                    additional_documents = (
                        [doc.to_dict() for doc in additional_docs]
                        if additional_docs
                        else []
                    )

                applications.append(
                    {
                        "certificate_id": str(cert_id),
                        "training_type": training_type,
                        "work_type": work_type,
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
                        "reviewer_decision": reviewer_decision,
                        "reviewer_comment": reviewer_comment,
                        "reviewed_date": reviewed_at.isoformat()
                        if reviewed_at
                        else None,
                        # Evaluation details
                        "total_working_hours": total_working_hours,
                        "credits_awarded": credits_awarded,
                        "training_duration": training_duration,
                        "training_institution": training_institution,
                        "degree_relevance": degree_relevance,
                        "supporting_evidence": supporting_evidence,
                        "challenging_evidence": challenging_evidence,
                        "recommendation": recommendation,
                        "student_comment": student_comment,
                        # Additional documents for self-paced work
                        "additional_documents": additional_documents,
                    }
                )

    return {"success": True, "applications": applications}


@router.post("/student/{student_id}/upload-certificate", tags=["student"])
async def upload_certificate(
    student_id: UUID,
    training_type: str = Form(...),
    work_type: str = Form(default="REGULAR"),
    file: UploadFile = File(...),
    additional_documents: List[UploadFile] = File(default=[]),
):
    """Upload a working certificate file for a student (by id)."""
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    try:
        training_type_enum = TrainingType(training_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid training type")

    try:
        work_type_enum = WorkType(work_type.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid work type")

    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Extract file extension properly
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        raise HTTPException(status_code=400, detail="File must have an extension")

    # Remove the dot from extension
    filetype = file_extension[1:].lower()

    # Validate file type
    allowed_extensions = {
        ".pdf",
        ".docx",
        ".doc",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
    }
    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {filetype} not supported. Allowed types: {', '.join(allowed_extensions)}",
        )

    # Read file content
    file_content = await file.read()

    # Validate file size (10MB limit)
    if len(file_content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400, detail="File size too large. Maximum size is 10MB."
        )

    # Validate that we actually have content
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    logger.info(
        f"Uploading file: {file.filename}, type: {filetype}, size: {len(file_content)} bytes"
    )

    # Store file in database instead of file system
    try:
        certificate = create_certificate(
            student_id=student.student_id,
            training_type=training_type_enum,
            work_type=work_type_enum,
            filename=file.filename,
            filetype=filetype,
            file_content=file_content,
        )

        logger.info(f"Certificate created with ID: {certificate.certificate_id}")

        # Handle additional documents for self-paced work
        if work_type_enum == WorkType.SELF_PACED and additional_documents:
            logger.info(
                f"Processing {len(additional_documents)} additional documents for self-paced work"
            )
            for doc in additional_documents:
                if not doc.filename:
                    continue

                # Validate additional document file type
                doc_extension = os.path.splitext(doc.filename)[1]
                if not doc_extension:
                    logger.warning(
                        f"Additional document {doc.filename} has no extension, skipping"
                    )
                    continue

                doc_filetype = doc_extension[1:].lower()
                if doc_extension.lower() not in allowed_extensions:
                    logger.warning(
                        f"Additional document {doc.filename} has unsupported type {doc_filetype}, skipping"
                    )
                    continue

                # Read additional document content
                doc_content = await doc.read()

                # Validate additional document size (10MB limit)
                if len(doc_content) > 10 * 1024 * 1024:  # 10MB
                    logger.warning(
                        f"Additional document {doc.filename} too large ({len(doc_content)} bytes), skipping"
                    )
                    continue

                # Validate that we actually have content
                if len(doc_content) == 0:
                    logger.warning(
                        f"Additional document {doc.filename} is empty, skipping"
                    )
                    continue

                # Create additional document record
                try:
                    additional_doc = create_additional_document(
                        certificate_id=certificate.certificate_id,
                        document_type=DocumentType.HOUR_DOCUMENTATION,
                        filename=doc.filename,
                        filetype=doc_filetype,
                        file_content=doc_content,
                    )
                    logger.info(
                        f"Additional document created: {additional_doc.document_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create additional document {doc.filename}: {e}"
                    )

        # Verify the certificate was created correctly
        verification_cert = get_certificate_by_id(certificate.certificate_id)
        if verification_cert:
            logger.info(
                f"Verification - File content size: {len(verification_cert.file_content) if verification_cert.file_content else 0} bytes"
            )
            logger.info(f"Verification - File type: {verification_cert.filetype}")
            logger.info(f"Verification - Work type: {verification_cert.work_type}")
        else:
            logger.error("Failed to retrieve certificate after creation")

    except Exception as e:
        logger.error(f"Error creating certificate: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create certificate: {str(e)}"
        )

    return {
        "success": True,
        "certificate_id": str(certificate.certificate_id),
        "certificate": certificate.to_dict(),
    }


ocr_workflow = OCRWorkflow(language="auto", use_finnish_detection=True)
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

    # Check if file content exists in database
    if not cert.file_content:
        raise HTTPException(
            status_code=404, detail="Certificate file content not found"
        )

    # Convert memoryview to bytes if necessary
    if hasattr(cert.file_content, "tobytes"):
        file_content = cert.file_content.tobytes()
    else:
        file_content = cert.file_content

    # Create a temporary file for OCR processing
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{cert.filetype}"
    ) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    try:
        # OCR processing using temporary file
        ocr_result = ocr_workflow.process_document(Path(temp_file_path))
        if not ocr_result.get("success"):
            raise HTTPException(status_code=500, detail="OCR processing failed")

        # Update certificate table with ocr_output
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE certificates SET ocr_output=%s WHERE certificate_id=%s",
                    (ocr_result.get("extracted_text", ""), str(certificate_id)),
                )
                conn.commit()

        # Process additional documents if self-paced work
        additional_doc_results = []
        if cert.work_type == WorkType.SELF_PACED:
            logger.info("Processing additional documents for self-paced work")
            additional_docs = get_additional_documents(certificate_id)

            if additional_docs:
                logger.info(
                    f"Found {len(additional_docs)} additional documents to process"
                )

                for doc in additional_docs:
                    try:
                        logger.info(f"Processing additional document: {doc.filename}")

                        # Convert memoryview to bytes if necessary
                        if hasattr(doc.file_content, "tobytes"):
                            doc_file_content = doc.file_content.tobytes()
                        else:
                            doc_file_content = doc.file_content

                        # Create temporary file for additional document OCR
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=f".{doc.filetype}"
                        ) as doc_temp_file:
                            doc_temp_file.write(doc_file_content)
                            doc_temp_file_path = doc_temp_file.name

                        try:
                            # Run OCR on additional document
                            doc_ocr_result = ocr_workflow.process_document(
                                Path(doc_temp_file_path)
                            )

                            if doc_ocr_result.get("success"):
                                # Update additional_documents table with OCR output
                                with get_db_connection() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute(
                                            "UPDATE additional_documents SET ocr_output=%s WHERE document_id=%s",
                                            (
                                                doc_ocr_result.get(
                                                    "extracted_text", ""
                                                ),
                                                str(doc.document_id),
                                            ),
                                        )
                                        conn.commit()

                                # Store OCR result for LLM processing
                                additional_doc_results.append(
                                    {
                                        "filename": doc.filename,
                                        "ocr_text": doc_ocr_result.get(
                                            "extracted_text", ""
                                        ),
                                        "document_type": doc.document_type.value,
                                        "ocr_result": doc_ocr_result,
                                    }
                                )

                                logger.info(
                                    f"✅ Successfully processed additional document: {doc.filename}"
                                )
                                logger.info(
                                    f"   Extracted {len(doc_ocr_result.get('extracted_text', ''))} characters"
                                )
                            else:
                                logger.warning(
                                    f"❌ OCR failed for additional document: {doc.filename}"
                                )
                                logger.warning(
                                    f"   Error: {doc_ocr_result.get('error', 'Unknown error')}"
                                )

                                # Still add to results with empty text for LLM processing
                                additional_doc_results.append(
                                    {
                                        "filename": doc.filename,
                                        "ocr_text": "",
                                        "document_type": doc.document_type.value,
                                        "ocr_result": doc_ocr_result,
                                    }
                                )

                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(doc_temp_file_path)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to clean up temporary file {doc_temp_file_path}: {e}"
                                )

                    except Exception as e:
                        logger.error(
                            f"Failed to process additional document {doc.filename}: {e}"
                        )
                        # Continue processing other documents
                        continue

                logger.info(
                    f"Completed processing {len(additional_doc_results)} additional documents"
                )
            else:
                logger.info("No additional documents found for self-paced work")
        else:
            logger.info("Regular work type - no additional documents to process")

        # Run LLM evaluation
        cleaned_text = ocr_result.get("extracted_text", "")

        try:
            # Enhanced LLM processing for self-paced work with additional documents
            if cert.work_type == WorkType.SELF_PACED and additional_doc_results:
                logger.info("Running enhanced LLM evaluation with additional documents")
                logger.info(
                    f"Additional documents count: {len(additional_doc_results)}"
                )
                logger.info(f"Combined text length: {len(cleaned_text)} characters")

                try:
                    llm_result = llm_orchestrator.process_work_certificate(
                        cleaned_text,
                        student_degree=student.degree,
                        requested_training_type=cert.training_type.value.lower(),
                        work_type=cert.work_type.value,
                        additional_documents=additional_doc_results,
                    )
                    if llm_result is None:
                        logger.error("Enhanced LLM processing returned None result")
                        raise Exception("LLM processing returned None result")
                    logger.info(
                        f"Enhanced LLM processing completed. Success: {llm_result.get('success', False)}"
                    )
                except Exception as llm_error:
                    logger.error(f"Enhanced LLM processing failed: {llm_error}")
                    logger.error(f"LLM error type: {type(llm_error)}")
                    logger.error(f"LLM error details: {str(llm_error)}")
                    raise llm_error
            else:
                logger.info("Running standard LLM evaluation")
                llm_result = llm_orchestrator.process_work_certificate(
                    cleaned_text,
                    student_degree=student.degree,
                    requested_training_type=cert.training_type.value.lower(),
                )

        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            # Return OCR results with LLM error
            return {
                "success": False,
                "error": f"LLM processing failed: {str(e)}",
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

        # Store decision using evaluation results directly from LLM
        evaluation_results = llm_result.get("evaluation_results", {}).get("results", {})
        extraction_results = llm_result.get("extraction_results", {}).get("results", {})

        # Get values directly from LLM results - no decision processing in API
        llm_decision = evaluation_results.get("decision", "PENDING")
        credits_awarded = evaluation_results.get("credits_qualified", 0)
        total_working_hours = evaluation_results.get("total_working_hours", 0)
        training_duration = extraction_results.get(
            "total_employment_period", "Not specified"
        )
        training_institution = extraction_results.get("employer", "Not specified")
        ai_justification = evaluation_results.get(
            "justification", "No justification provided"
        )
        degree_relevance = evaluation_results.get("degree_relevance", "Not specified")
        supporting_evidence = evaluation_results.get("supporting_evidence", "")
        challenging_evidence = evaluation_results.get("challenging_evidence", "")
        recommendation = evaluation_results.get("recommendation", "")

        # Convert LLM decision string to DecisionStatus enum for database storage
        final_decision = (
            DecisionStatus.ACCEPTED
            if llm_decision == "ACCEPTED"
            else DecisionStatus.REJECTED
        )

        # Create complete AI workflow JSON output (like the old aiworkflow_output files)
        complete_workflow_json = {
            "success": True,
            "certificate_id": str(certificate_id),
            "student_degree": student.degree,
            "processing_time": llm_result.get("processing_time", 0),
            "ocr_results": ocr_result,
            "llm_results": llm_result,
            "decision": {
                "ai_decision": llm_decision,  # Use LLM's decision string directly
                "ai_justification": ai_justification,
                "credits_awarded": credits_awarded,
                "total_working_hours": total_working_hours,
                "training_duration": training_duration,
                "training_institution": training_institution,
            },
        }

        # Convert to JSON string
        import json

        ai_workflow_json_string = json.dumps(
            complete_workflow_json, indent=2, ensure_ascii=False
        )

        # Extract company validation data from LLM results
        company_validation_status = "UNVERIFIED"
        company_validation_justification = "No company validation data available"

        # Debug logging - let's see exactly what we're getting
        logger.info("🔍 DEBUGGING COMPANY VALIDATION EXTRACTION")
        logger.info(f"LLM result type: {type(llm_result)}")
        logger.info(f"LLM result keys: {list(llm_result.keys())}")

        # Let's also log the actual content to see what's there
        logger.info(f"LLM result content preview: {str(llm_result)[:500]}...")

        if llm_result.get("success") and "validation_results" in llm_result:
            validation_results = llm_result["validation_results"]
            logger.info("✅ Found validation_results")
            logger.info(f"Validation results type: {type(validation_results)}")
            logger.info(f"Validation results keys: {list(validation_results.keys())}")

            # Check if there's a "results" level (which exists in your JSON structure)
            if "results" in validation_results:
                validation_results_data = validation_results["results"]
                logger.info("✅ Found results level in validation_results")
                logger.info(f"Results keys: {list(validation_results_data.keys())}")

                if "company_validation" in validation_results_data:
                    company_validation = validation_results_data["company_validation"]
                    logger.info("✅ Found company_validation in results")
                    logger.info(f"Company validation type: {type(company_validation)}")
                    logger.info(
                        f"Company validation keys: {list(company_validation.keys())}"
                    )
                    logger.info(f"Company validation content: {company_validation}")

                    # Get overall status for company_validation_status column
                    company_validation_status = company_validation.get(
                        "status", "UNVERIFIED"
                    )
                    logger.info(
                        f"📊 Extracted company validation status: {company_validation_status}"
                    )

                    # Get the entire companies array for company_validation_justification column
                    companies = company_validation.get("companies", [])
                    logger.info(f"🏢 Found {len(companies)} companies to process")
                    logger.info(f"Companies content: {companies}")

                    if companies:
                        # Save the entire companies array as JSON string
                        import json

                        company_validation_justification = json.dumps(
                            companies, indent=2, ensure_ascii=False
                        )
                        logger.info(
                            f"✅ Saved companies array with {len(companies)} companies to justification column"
                        )
                        logger.info(
                            f"Justification preview: {company_validation_justification[:200]}..."
                        )
                    else:
                        company_validation_justification = f"Overall Status: {company_validation_status}\nNo individual company details available."
                        logger.info(
                            "⚠️ No companies found, using fallback justification"
                        )
                else:
                    logger.warning(
                        "❌ company_validation NOT found in validation_results.results"
                    )
                    logger.warning(
                        f"Available keys in results: {list(validation_results_data.keys())}"
                    )
                    # Let's see what's actually in results
                    logger.warning(f"Results content: {validation_results_data}")
            else:
                # Fallback: try the old path (in case the structure changes)
                logger.info("⚠️ No 'results' level found, trying direct access")
                if "company_validation" in validation_results:
                    company_validation = validation_results["company_validation"]
                    logger.info("✅ Found company_validation (direct access)")

                    # Get overall status for company_validation_status column
                    company_validation_status = company_validation.get(
                        "status", "UNVERIFIED"
                    )
                    logger.info(
                        f"📊 Extracted company validation status: {company_validation_status}"
                    )

                    # Get the entire companies array for company_validation_justification column
                    companies = company_validation.get("companies", [])
                    logger.info(f"🏢 Found {len(companies)} companies to process")
                    logger.info(f"Companies content: {companies}")

                    if companies:
                        # Save the entire companies array as JSON string
                        import json

                        company_validation_justification = json.dumps(
                            companies, indent=2, ensure_ascii=False
                        )
                        logger.info(
                            f"✅ Saved companies array with {len(companies)} companies to justification column"
                        )
                        logger.info(
                            f"Justification preview: {company_validation_justification[:200]}..."
                        )
                    else:
                        company_validation_justification = f"Overall Status: {company_validation_status}\nNo individual company details available."
                        logger.info(
                            "⚠️ No companies found, using fallback justification"
                        )
                else:
                    logger.warning(
                        "❌ company_validation NOT found in validation_results (direct access)"
                    )
                    logger.warning(
                        f"Available keys in validation_results: {list(validation_results.keys())}"
                    )
                    # Let's see what's actually in validation_results
                    logger.warning(f"Validation results content: {validation_results}")
        else:
            logger.warning("❌ LLM result missing success or validation_results")
            logger.warning(f"LLM result keys: {list(llm_result.keys())}")
            if "success" in llm_result:
                logger.warning(f"Success value: {llm_result['success']}")
            if "validation_results" in llm_result:
                logger.warning("Validation results found but success is False")

        logger.info("🎯 FINAL RESULTS:")
        logger.info(f"Final company validation status: {company_validation_status}")
        logger.info(
            f"Final justification length: {len(company_validation_justification)}"
        )
        logger.info(
            f"Final justification preview: {company_validation_justification[:200]}..."
        )

        # Create decision record - store LLM results directly
        decision = create_decision(
            certificate_id=certificate_id,
            ai_decision=final_decision,  # Enum for database storage
            ai_justification=ai_justification,
            total_working_hours=total_working_hours,
            credits_awarded=credits_awarded,
            training_duration=training_duration,
            training_institution=training_institution,
            degree_relevance=degree_relevance,
            supporting_evidence=supporting_evidence,
            challenging_evidence=challenging_evidence,
            recommendation=recommendation,
            ai_workflow_json=ai_workflow_json_string,
            company_validation_status=company_validation_status,
            company_validation_justification=company_validation_justification,
        )

        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

        # Prepare response with additional document information
        response = {
            "success": True,
            "certificate_id": str(certificate_id),
            "work_type": cert.work_type.value,
            "ocr_results": ocr_result,
            "llm_results": llm_result,
            "decision": {
                "ai_decision": decision.ai_decision.value,
                "ai_justification": decision.ai_justification,
                "credits_awarded": decision.credits_awarded,
                "total_working_hours": decision.total_working_hours,
                "training_duration": decision.training_duration,
                "training_institution": decision.training_institution,
                "company_validation_status": decision.company_validation_status,
                "company_validation_justification": decision.company_validation_justification,
                "calculation_method": "enhanced_with_additional_documents"
                if (cert.work_type == WorkType.SELF_PACED and additional_doc_results)
                else "standard",
            },
        }

        # Add additional document information if self-paced work
        if cert.work_type == WorkType.SELF_PACED and additional_doc_results:
            response["additional_documents"] = {
                "count": len(additional_doc_results),
                "documents": [
                    {
                        "filename": doc["filename"],
                        "document_type": doc["document_type"],
                        "ocr_success": doc["ocr_result"].get("success", False),
                        "text_length": len(doc["ocr_text"]),
                        "ocr_error": doc["ocr_result"].get("error")
                        if not doc["ocr_result"].get("success")
                        else None,
                    }
                    for doc in additional_doc_results
                ],
            }
            logger.info(
                f"Added {len(additional_doc_results)} additional documents to response"
            )

        return response

    except Exception as e:
        # Clean up temporary file on error
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass

        logger.error(f"Certificate processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/certificate/{certificate_id}/status", tags=["student"])
async def get_certificate_status(certificate_id: UUID):
    """Get the processing status of a certificate."""
    try:
        cert = get_certificate_by_id(certificate_id)
        if not cert:
            raise HTTPException(status_code=404, detail="Certificate not found")

        # Check if decision exists (indicates processing is complete)
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ai_decision, ai_justification, total_working_hours, credits_awarded, training_duration, training_institution FROM decisions WHERE certificate_id = %s",
                    (str(certificate_id),),
                )
                decision_row = cur.fetchone()

        if decision_row:
            # Processing is complete
            return {
                "status": "completed",
                "certificate_id": str(certificate_id),
                "decision": {
                    "ai_decision": decision_row[0].value
                    if decision_row[0]
                    else "PENDING",
                    "ai_justification": decision_row[1] or "No justification provided",
                    "total_working_hours": decision_row[2] or 0,
                    "credits_awarded": decision_row[3] or 0,
                    "training_duration": decision_row[4] or "Not specified",
                    "training_institution": decision_row[5] or "Not specified",
                },
            }
        else:
            # Processing is still in progress
            return {
                "status": "processing",
                "certificate_id": str(certificate_id),
                "message": "Certificate is being processed. Please wait.",
            }

    except Exception as e:
        logger.error(f"Failed to get certificate status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


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
    student_comment: str
    reviewer_id: Optional[UUID] = None


@router.post("/certificate/{certificate_id}/feedback", tags=["student"])
async def add_feedback_endpoint(certificate_id: UUID, payload: FeedbackRequest):
    """Store student comment and reviewer ID for a decision/certificate."""
    success = add_student_comment_and_reviewer(
        certificate_id, payload.student_comment, payload.reviewer_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return {
        "success": True,
        "message": "Student comment and reviewer information stored",
    }


class StudentCommentRequest(BaseModel):
    student_comment: str


@router.post("/certificate/{certificate_id}/student-comment", tags=["student"])
async def add_student_comment_endpoint(
    certificate_id: UUID, payload: StudentCommentRequest
):
    """Add student comment to a rejected certificate (simplified appeal process)."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    # Check if comment already exists
    existing_comment = get_student_comment_by_certificate_id(certificate_id)
    if existing_comment:
        raise HTTPException(
            status_code=400,
            detail="Student comment already exists for this certificate",
        )

    try:
        # Add student comment
        success = add_student_comment(certificate_id, payload.student_comment)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add student comment")

        return {
            "success": True,
            "message": "Student comment added successfully. The application will follow the normal approval process.",
        }
    except Exception as e:
        logger.error(
            f"Error adding student comment for certificate {certificate_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Failed to add student comment")


@router.delete("/certificate/{certificate_id}", tags=["student"])
async def delete_certificate_endpoint(certificate_id: UUID):
    """Delete a certificate and its associated data."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    try:
        # Delete the certificate and all associated data
        delete_certificate(certificate_id)

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
    reviewer_id: Optional[UUID] = None


@router.post("/certificate/{certificate_id}/appeal", tags=["student"])
async def submit_appeal_endpoint(certificate_id: UUID, payload: AppealRequest):
    """Submit an appeal for a rejected certificate (simplified process)."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")

    # Check if student comment already exists
    existing_comment = get_student_comment_by_certificate_id(certificate_id)
    if existing_comment:
        raise HTTPException(
            status_code=400,
            detail="Student comment already exists for this certificate",
        )

    try:
        # Get reviewers for assignment
        reviewers = get_all_reviewers()
        if not reviewers:
            raise HTTPException(status_code=500, detail="No reviewers available")

        # Determine assigned reviewer: use provided reviewer_id if present; otherwise fallback to the first reviewer
        assigned_reviewer_id: UUID
        assigned_reviewer_obj = None
        if payload.reviewer_id:
            assigned_reviewer_id = payload.reviewer_id
            # Try to find full reviewer object for response
            for r in reviewers:
                if str(r.reviewer_id) == str(payload.reviewer_id):
                    assigned_reviewer_obj = r
                    break
        else:
            assigned_reviewer_obj = reviewers[0]
            assigned_reviewer_id = assigned_reviewer_obj.reviewer_id

        # Add student comment and assign reviewer (simplified appeal process)
        success = add_student_comment_and_reviewer(
            certificate_id, payload.appeal_reason, assigned_reviewer_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to submit appeal")

        return {
            "success": True,
            "message": "Appeal submitted successfully. The application will now follow the normal approval process.",
            "assigned_reviewer": (
                assigned_reviewer_obj.to_dict()
                if assigned_reviewer_obj
                else {"reviewer_id": str(assigned_reviewer_id)}
            ),
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


@router.get("/certificate/{certificate_id}", tags=["student"])
async def download_certificate_file(certificate_id: UUID):
    """Return the certificate file for download."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    if not cert.file_content:
        raise HTTPException(
            status_code=404, detail="Certificate file content not found"
        )

    # Convert memoryview to bytes if necessary
    if hasattr(cert.file_content, "tobytes"):
        file_content = cert.file_content.tobytes()
    else:
        file_content = cert.file_content

    # Determine the media type based on file extension
    file_extension = cert.filetype.lower()
    media_type = "application/octet-stream"  # default

    if file_extension in ["pdf"]:
        media_type = "application/pdf"
    elif file_extension in ["jpg", "jpeg"]:
        media_type = "image/jpeg"
    elif file_extension in ["png"]:
        media_type = "image/png"
    elif file_extension in ["doc", "docx"]:
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # Return the file content from database
    return Response(
        content=file_content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={cert.filename}"},
    )


@router.get("/certificate/{certificate_id}/preview", tags=["student"])
async def preview_certificate_file(certificate_id: UUID):
    """Return the certificate file for preview (inline display)."""
    cert = get_certificate_by_id(certificate_id)
    if not cert:
        raise HTTPException(status_code=404, detail="Certificate not found")
    if not cert.file_content:
        raise HTTPException(
            status_code=404, detail="Certificate file content not found"
        )

    # Convert memoryview to bytes if necessary
    if hasattr(cert.file_content, "tobytes"):
        file_content = cert.file_content.tobytes()
    else:
        file_content = cert.file_content

    # Determine the media type based on file extension
    file_extension = cert.filetype.lower()
    media_type = "application/octet-stream"  # default

    if file_extension in ["pdf"]:
        media_type = "application/pdf"
    elif file_extension in ["jpg", "jpeg"]:
        media_type = "image/jpeg"
    elif file_extension in ["png"]:
        media_type = "image/png"
    elif file_extension in ["doc", "docx"]:
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # For preview, we want to serve the file inline without forcing download
    return Response(
        content=file_content,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={cert.filename}"},
    )


# If the frontend still needs metadata, expose it under a new path
@router.get("/certificate/{certificate_id}/details", tags=["reviewer"])
async def get_certificate_details(certificate_id: UUID):
    """Get detailed information about a specific certificate (moved from the root path)."""
    application = get_detailed_application(certificate_id)
    if not application:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return {"success": True, "application": application.to_dict()}


@router.get(
    "/certificate/{certificate_id}/additional-document/{document_id}/preview",
    tags=["reviewer"],
)
async def preview_additional_document(certificate_id: UUID, document_id: UUID):
    """Preview an additional document for a certificate."""
    # Get the additional document
    additional_docs = get_additional_documents(certificate_id)
    document = None

    for doc in additional_docs:
        if doc.document_id == document_id:
            document = doc
            break

    if not document:
        raise HTTPException(status_code=404, detail="Additional document not found")

    if not document.file_content:
        raise HTTPException(status_code=404, detail="Document content not available")

    # Convert memoryview to bytes if necessary (same logic as main certificate preview)
    if hasattr(document.file_content, "tobytes"):
        file_content = document.file_content.tobytes()
    else:
        file_content = document.file_content

    # Determine content type based on file extension
    file_extension = document.filetype.lower()
    media_type = "application/octet-stream"  # default

    if file_extension in ["pdf"]:
        media_type = "application/pdf"
    elif file_extension in ["jpg", "jpeg"]:
        media_type = "image/jpeg"
    elif file_extension in ["png"]:
        media_type = "image/png"
    elif file_extension in ["doc", "docx"]:
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    # For preview, we want to serve the file inline without forcing download
    return Response(
        content=file_content,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={document.filename}"},
    )


class ReviewUpdateRequest(BaseModel):
    """Payload for reviewer to submit their decision."""

    reviewer_comment: Optional[str] = None
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
