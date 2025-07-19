"""
LLM Orchestrator for Work Certificate Processing
Manages a 4-stage process: extraction + structural validation of extraction, evaluation + structural validation of evaluation, validation, and correction, on + structural validation of correction.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import google.generativeai as genai

from src.config import settings
from src.llm.degree_evaluator import DegreeEvaluator
from src.llm.models import (
    validate_evaluation_results,
    validate_extraction_results,
)
from src.llm.prompts import (
    CORRECTION_PROMPT,
    EVALUATION_PROMPT,
    EXTRACTION_PROMPT,
    VALIDATION_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Orchestrates LLM-based work certificate processing using a 4-stage approach."""

    def __init__(self):
        """Initialize the LLM orchestrator with Gemini."""
        self.model = None
        self.model_name = settings.GEMINI_MODEL
        self.current_model_index = 0
        self.available_models = [
            settings.GEMINI_MODEL
        ] + settings.GEMINI_FALLBACK_MODELS
        self.degree_evaluator = DegreeEvaluator()
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini API client with fallback support."""
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API key not provided - LLM processing unavailable")
            return

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._try_initialize_model()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def _try_initialize_model(self, model_index: int = 0) -> bool:
        """Try to initialize a specific model by index."""
        if model_index >= len(self.available_models):
            logger.error("All available models failed to initialize")
            return False

        model_name = self.available_models[model_index]
        try:
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            self.current_model_index = model_index
            logger.info(f"LLM Orchestrator initialized with model: {model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize model {model_name}: {e}")
            return self._try_initialize_model(model_index + 1)

    def _handle_quota_error(self, error: str) -> bool:
        """Handle quota limit errors by trying fallback models."""
        quota_indicators = [
            "quota exceeded",
            "quota limit",
            "rate limit",
            "quota has been exceeded",
            "quota limit exceeded",
            "quota exceeded for quota metric",
            "quota limit exceeded for quota metric",
            "quota exceeded for quota group",
            "quota limit exceeded for quota group",
        ]

        is_quota_error = any(
            indicator in error.lower() for indicator in quota_indicators
        )

        if is_quota_error and self.current_model_index < len(self.available_models) - 1:
            logger.warning(
                f"Quota limit hit for model {self.model_name}, trying fallback..."
            )
            next_model_index = self.current_model_index + 1
            if self._try_initialize_model(next_model_index):
                logger.info(
                    f"Successfully switched to fallback model: {self.model_name}"
                )
                return True
            else:
                logger.error("All fallback models also failed")
                return False

        return False

    def _call_llm_with_fallback(
        self, prompt: str, operation_name: str = "LLM call"
    ) -> Optional[str]:
        """Call LLM with automatic fallback on quota errors."""
        max_retries = len(self.available_models) - 1
        retry_count = 0

        while retry_count <= max_retries:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"{operation_name} failed with model {self.model_name}: {error_msg}"
                )

                if self._handle_quota_error(error_msg):
                    retry_count += 1
                    continue
                else:
                    logger.error(f"{operation_name} failed with all available models")
                    raise e

        return None

    def process_work_certificate(
        self,
        text: str,
        student_degree: str = "Business Administration",
        requested_training_type: str = None,
    ) -> Dict[str, Any]:
        """
        Process a work certificate using the 4-stage LLM approach.

        Args:
            text: Cleaned OCR text from the work certificate
            student_degree: Student's degree program
            requested_training_type: Student's requested training type (general or professional)

        Returns:
            Dictionary with both extraction and evaluation results
        """
        if not self.model:
            return self._error_response("Gemini API not available")

        # Validate input
        validation_error = self._validate_input(text)
        if validation_error:
            return self._error_response(validation_error)

        # Validate degree program
        if not self.degree_evaluator.validate_degree_program(student_degree):
            logger.warning(
                f"Unsupported degree program: {student_degree}, using general criteria"
            )

        start_time = time.time()

        try:
            # Stage 1: Information Extraction
            sanitized_text = self._sanitize_text(text)
            extraction_result = self._extract_information(sanitized_text)

            if not extraction_result.get("success", False):
                return self._error_response(
                    f"Extraction failed: {extraction_result.get('error', 'Unknown error')}",
                    extraction_results=extraction_result,
                    processing_time=time.time() - start_time,
                )

            # Stage 1.5: Structural Validation of Extraction Results
            structural_validation_extraction = validate_extraction_results(
                extraction_result["results"]
            )

            if structural_validation_extraction.validation_passed:
                logger.info("✅ Structural validation passed for extraction")
            else:
                logger.warning(
                    f"Structural validation failed for extraction: {structural_validation_extraction.summary}"
                )
                # Log detailed issues
                for i, issue in enumerate(
                    structural_validation_extraction.issues_found, 1
                ):
                    logger.warning(
                        f"  Extraction Issue {i}: {issue.type} ({issue.severity}) - {issue.description}"
                    )
                # Continue processing but log the issues

            # Stage 2: Academic Evaluation
            evaluation_result = self._evaluate_academically(
                sanitized_text,
                extraction_result["results"],
                student_degree,
                requested_training_type,
            )

            # Stage 2.5: Structural Validation of Evaluation Results
            if evaluation_result.get("success", False):
                structural_validation_evaluation = validate_evaluation_results(
                    evaluation_result["results"]
                )

                if not structural_validation_evaluation.validation_passed:
                    logger.warning(
                        f"Structural validation failed for evaluation: {structural_validation_evaluation.summary}"
                    )
                    # Log detailed issues
                    for i, issue in enumerate(
                        structural_validation_evaluation.issues_found, 1
                    ):
                        logger.warning(
                            f"  Evaluation Issue {i}: {issue.type} ({issue.severity}) - {issue.description}"
                        )
                    # Continue processing but log the issues

            # Stage 3: Validation
            validation_result = self._validate_results(
                sanitized_text,
                extraction_result["results"],
                evaluation_result["results"],
                student_degree,
                requested_training_type,
            )

            # Stage 4: Correction (if needed)
            correction_result = None
            if validation_result.get("success", False) and validation_result[
                "results"
            ].get("requires_correction", False):
                correction_result = self._correct_results(
                    sanitized_text,
                    extraction_result["results"],
                    evaluation_result["results"],
                    validation_result["results"],
                    student_degree,
                    requested_training_type,
                )

            # Stage 4.5: Structural Validation of Correction Results (if correction was performed)
            structural_validation_correction = None
            if correction_result and correction_result.get("success", False):
                # Validate corrected extraction results
                if "extraction_results" in correction_result["results"]:
                    structural_validation_correction_extraction = (
                        validate_extraction_results(
                            correction_result["results"]["extraction_results"]
                        )
                    )

                    if not structural_validation_correction_extraction.validation_passed:
                        logger.warning(
                            f"Structural validation failed for corrected extraction: {structural_validation_correction_extraction.summary}"
                        )
                        # Log detailed issues
                        for i, issue in enumerate(
                            structural_validation_correction_extraction.issues_found, 1
                        ):
                            logger.warning(
                                f"  Corrected Extraction Issue {i}: {issue.type} ({issue.severity}) - {issue.description}"
                            )

                # Validate corrected evaluation results
                if "evaluation_results" in correction_result["results"]:
                    structural_validation_correction_evaluation = (
                        validate_evaluation_results(
                            correction_result["results"]["evaluation_results"]
                        )
                    )

                    if not structural_validation_correction_evaluation.validation_passed:
                        logger.warning(
                            f"Structural validation failed for corrected evaluation: {structural_validation_correction_evaluation.summary}"
                        )
                        # Log detailed issues
                        for i, issue in enumerate(
                            structural_validation_correction_evaluation.issues_found, 1
                        ):
                            logger.warning(
                                f"  Corrected Evaluation Issue {i}: {issue.type} ({issue.severity}) - {issue.description}"
                            )

                # Store correction validation results
                structural_validation_correction = {
                    "extraction": structural_validation_correction_extraction.dict()
                    if "structural_validation_correction_extraction" in locals()
                    else None,
                    "evaluation": structural_validation_correction_evaluation.dict()
                    if "structural_validation_correction_evaluation" in locals()
                    else None,
                }

            return {
                "success": True,
                "processing_time": time.time() - start_time,
                "extraction_results": extraction_result,
                "evaluation_results": evaluation_result,
                "validation_results": validation_result,
                "correction_results": correction_result,
                "structural_validation": {
                    "extraction": structural_validation_extraction.dict()
                    if "structural_validation_extraction" in locals()
                    else None,
                    "evaluation": structural_validation_evaluation.dict()
                    if "structural_validation_evaluation" in locals()
                    else None,
                    "correction": structural_validation_correction,
                },
                "student_degree": student_degree,
                "model_used": self.model_name,
                "stages_completed": {
                    "extraction": extraction_result.get("success", False),
                    "evaluation": evaluation_result.get("success", False),
                    "validation": validation_result.get("success", False),
                    "correction": correction_result.get("success", False)
                    if correction_result
                    else False,
                },
            }

        except Exception as e:
            logger.error(f"Error in LLM orchestration: {e}")
            return self._error_response(
                str(e), processing_time=time.time() - start_time
            )

    def _validate_input(self, text: str) -> str:
        """Validate input text and return error message if invalid."""
        if not text or not isinstance(text, str):
            return f"Invalid text input: {type(text)}"

        if len(text) < 10 or text.strip() == "":
            return "Text too short or empty"

        if text.strip().startswith('"') or text.strip().startswith("{"):
            return "Text appears to be JSON, not document content"

        return None

    def _error_response(self, error: str, **kwargs) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error,
            "extraction_results": kwargs.get("extraction_results"),
            "evaluation_results": kwargs.get("evaluation_results"),
            "processing_time": kwargs.get("processing_time", 0),
            "model_used": self.model_name,
        }

    def _sanitize_text(self, text: str) -> str:
        """Sanitize and clean text before sending to LLM."""
        if not text:
            return ""

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize line endings and clean whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)  # Remove excessive newlines
        text = re.sub(r" +", " ", text)  # Remove excessive spaces

        # Clean lines and join
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Limit length
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Text truncated due to length]"

        return text.strip()

    def _extract_information(self, text: str) -> Dict[str, Any]:
        """Stage 1: Extract basic information from the certificate."""
        stage_start = time.time()

        try:
            prompt = EXTRACTION_PROMPT.format(document_text=text)
            response = self._call_llm_with_fallback(prompt, "extraction")
            results = self._parse_llm_response(response)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"Error in extraction stage: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _validate_results(
        self,
        text: str,
        extracted_info: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        student_degree: str,
        requested_training_type: str = None,
    ) -> Dict[str, Any]:
        """Stage 3: Validate LLM results against original document."""
        stage_start = time.time()

        try:
            # Format data for validation prompt
            extraction_str = json.dumps(extracted_info, indent=2, ensure_ascii=False)
            evaluation_str = json.dumps(
                evaluation_results, indent=2, ensure_ascii=False
            )

            prompt = VALIDATION_PROMPT.format(
                ocr_text=text,
                extraction_results=extraction_str,
                evaluation_results=evaluation_str,
                student_degree=student_degree,
                requested_training_type=requested_training_type or "general",
            )

            response = self._call_llm_with_fallback(prompt, "validation")
            results = self._parse_llm_response(response)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"Error in validation stage: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _correct_results(
        self,
        text: str,
        extracted_info: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        validation_results: Dict[str, Any],
        student_degree: str,
        requested_training_type: str = None,
    ) -> Dict[str, Any]:
        """Stage 4: Correct inaccuracies identified by validation."""
        stage_start = time.time()

        try:
            # Format data for correction prompt
            original_llm_output = {
                "extraction_results": extracted_info,
                "evaluation_results": evaluation_results,
            }
            original_str = json.dumps(original_llm_output, indent=2, ensure_ascii=False)
            validation_str = json.dumps(
                validation_results, indent=2, ensure_ascii=False
            )

            prompt = CORRECTION_PROMPT.format(
                ocr_text=text,
                original_llm_output=original_str,
                validation_results=validation_str,
                student_degree=student_degree,
                requested_training_type=requested_training_type or "general",
            )

            response = self._call_llm_with_fallback(prompt, "correction")
            results = self._parse_llm_response(response)

            # Ensure requested_training_type is preserved in correction results
            if results and "evaluation_results" in results:
                results["evaluation_results"]["requested_training_type"] = (
                    requested_training_type
                )

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"Error in correction stage: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _evaluate_academically(
        self,
        text: str,
        extracted_info: Dict[str, Any],
        student_degree: str,
        requested_training_type: str = None,
    ) -> Dict[str, Any]:
        """Stage 2: Evaluate the certificate for academic credits with degree-specific criteria and requested training type."""
        stage_start = time.time()

        try:
            # Format extracted info and get degree guidelines
            extracted_info_str = json.dumps(
                extracted_info, indent=2, ensure_ascii=False
            )
            degree_guidelines = self.degree_evaluator.get_degree_specific_guidelines(
                student_degree
            )

            # Create evaluation prompt
            prompt = EVALUATION_PROMPT.format(
                extracted_info=extracted_info_str,
                document_text=text,
                student_degree=student_degree,
                degree_specific_guidelines=degree_guidelines,
                requested_training_type=requested_training_type or "general",
            )

            response = self._call_llm_with_fallback(prompt, "evaluation")
            results = self._parse_llm_response(response)

            # Only ensure credit calculations are correct, don't override LLM decisions
            if results:
                # Store the requested training type in the results
                results["requested_training_type"] = requested_training_type

                total_hours = results.get("total_working_hours", 0)
                base_credits = int(total_hours / 27)

                # Store the actual calculated credits (before capping)
                results["credits_calculated"] = float(base_credits)

                # Use requested_training_type for capping
                training_type = requested_training_type or results.get(
                    "training_type", ""
                )
                if training_type == "professional" and base_credits > 30:
                    results["credits_qualified"] = 30.0
                    results["calculation_breakdown"] = (
                        f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 30.0 maximum for professional training"
                    )
                elif training_type == "general" and base_credits > 10:
                    results["credits_qualified"] = 10.0
                    results["calculation_breakdown"] = (
                        f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 10.0 maximum for general training"
                    )
                else:
                    results["credits_qualified"] = float(base_credits)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"Error in evaluation stage: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _parse_llm_response(self, response_text: Optional[str]) -> Dict[str, Any]:
        """Parse the LLM response and extract JSON."""
        if response_text is None:
            logger.error("LLM response is None - all models may have failed")
            return self._create_fallback_response("")

        try:
            response_text = response_text.strip()

            # Find JSON in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx]

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed_json = self._fix_common_json_issues(json_str)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    # Create fallback response
                    return self._create_fallback_response(json_str)

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._create_fallback_response("")

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove text before first { and after last }
        json_str = json_str[json_str.find("{") : json_str.rfind("}") + 1]

        # Fix missing quotes around keys
        json_str = re.sub(r"(\s*)(\w+)(\s*):", r'\1"\2"\3:', json_str)

        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')

        # Fix trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def _create_fallback_response(self, partial_json: str) -> Dict[str, Any]:
        """Create a fallback response when JSON parsing fails."""
        # Try to extract any available information
        employee_name = "Unknown"
        position = "Unknown"
        employer = "Unknown"

        name_match = re.search(r'"employee_name"\s*:\s*"([^"]+)"', partial_json)
        if name_match:
            employee_name = name_match.group(1)

        position_match = re.search(r'"position"\s*:\s*"([^"]+)"', partial_json)
        if position_match:
            position = position_match.group(1)

        employer_match = re.search(r'"employer"\s*:\s*"([^"]+)"', partial_json)
        if employer_match:
            employer = employer_match.group(1)

        return {
            "employee_name": employee_name,
            "position": position,
            "employer": employer,
            "start_date": None,
            "end_date": None,
            "employment_period": "Unknown",
            "document_language": "en",
            "confidence_level": "low",
            "extraction_notes": "Partial JSON parsed, some fields may be missing",
        }

    def is_available(self) -> bool:
        """Check if the LLM orchestrator is available."""
        return self.model is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the LLM orchestrator."""
        return {
            "available": self.is_available(),
            "current_model": self.model_name,
            "current_model_index": self.current_model_index,
            "available_models": self.available_models,
            "fallback_models_remaining": len(self.available_models)
            - self.current_model_index
            - 1,
            "api_key_configured": bool(settings.GEMINI_API_KEY),
            "stages": ["extraction", "evaluation", "validation", "correction"],
        }

    def get_prompt_info(self) -> Dict[str, Any]:
        """Get information about the prompts being used."""
        return {
            "extraction_prompt_length": len(EXTRACTION_PROMPT),
            "evaluation_prompt_length": len(EVALUATION_PROMPT),
            "validation_prompt_length": len(VALIDATION_PROMPT),
            "correction_prompt_length": len(CORRECTION_PROMPT),
            "total_prompt_length": len(EXTRACTION_PROMPT)
            + len(EVALUATION_PROMPT)
            + len(VALIDATION_PROMPT)
            + len(CORRECTION_PROMPT),
        }
