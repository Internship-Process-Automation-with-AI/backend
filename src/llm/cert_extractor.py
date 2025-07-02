"""
LLM Orchestrator for Work Certificate Processing
Manages a two-stage process: extraction followed by evaluation.
"""

import json
import logging
import re
import time
from typing import Any, Dict

import google.generativeai as genai

from src.config import settings
from src.llm.degree_evaluator import DegreeEvaluator
from src.llm.prompts import (
    CORRECTION_PROMPT,
    EVALUATION_PROMPT,
    EXTRACTION_PROMPT,
    VALIDATION_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Orchestrates LLM-based work certificate processing using a two-stage approach."""

    def __init__(self):
        """Initialize the LLM orchestrator with Gemini."""
        self.model = None
        self.model_name = settings.GEMINI_MODEL
        self.degree_evaluator = DegreeEvaluator()
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini API client."""
        if not settings.GEMINI_API_KEY:
            logger.warning("Gemini API key not provided - LLM processing unavailable")
            return

        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"LLM Orchestrator initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None

    def process_work_certificate(
        self, text: str, student_degree: str = "Business Administration"
    ) -> Dict[str, Any]:
        """
        Process a work certificate using the 4-stage LLM approach.

        Args:
            text: Cleaned OCR text from the work certificate
            student_degree: Student's degree program

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

            # Stage 2: Academic Evaluation
            evaluation_result = self._evaluate_academically(
                sanitized_text, extraction_result["results"], student_degree
            )

            # Stage 3: Validation
            validation_result = self._validate_results(
                sanitized_text,
                extraction_result["results"],
                evaluation_result["results"],
                student_degree,
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
                )

            return {
                "success": True,
                "processing_time": time.time() - start_time,
                "extraction_results": extraction_result,
                "evaluation_results": evaluation_result,
                "validation_results": validation_result,
                "correction_results": correction_result,
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
            response = self.model.generate_content(prompt)
            results = self._parse_llm_response(response.text)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response.text,
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
            )

            response = self.model.generate_content(prompt)
            results = self._parse_llm_response(response.text)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response.text,
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
            )

            response = self.model.generate_content(prompt)
            results = self._parse_llm_response(response.text)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response.text,
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
        self, text: str, extracted_info: Dict[str, Any], student_degree: str
    ) -> Dict[str, Any]:
        """Stage 2: Evaluate the certificate for academic credits with degree-specific criteria."""
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
            )

            response = self.model.generate_content(prompt)
            results = self._parse_llm_response(response.text)

            # Apply degree-specific relevance analysis and corrections
            if results and extracted_info:
                self._apply_degree_corrections(results, extracted_info, student_degree)

            return {
                "success": True,
                "processing_time": time.time() - stage_start,
                "results": results,
                "raw_response": response.text,
            }

        except Exception as e:
            logger.error(f"Error in evaluation stage: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _apply_degree_corrections(
        self,
        results: Dict[str, Any],
        extracted_info: Dict[str, Any],
        student_degree: str,
    ):
        """Apply degree-specific corrections to evaluation results."""
        positions = extracted_info.get("positions", [])
        total_relevance_score = 0

        # Calculate average relevance
        for position in positions:
            job_title = position.get("title", "")
            job_description = position.get("responsibilities", "")
            relevance_level, _ = self.degree_evaluator.calculate_relevance_score(
                student_degree, job_title, job_description, ""
            )

            if relevance_level == "high_relevance":
                total_relevance_score += 1.0
            elif relevance_level == "medium_relevance":
                total_relevance_score += 0.5

        avg_relevance_score = total_relevance_score / max(1, len(positions))
        overall_relevance_level = (
            "high_relevance"
            if avg_relevance_score >= 0.6
            else "medium_relevance"
            if avg_relevance_score >= 0.3
            else "low_relevance"
        )

        # Apply corrections based on relevance
        total_hours = results.get("total_working_hours", 0)
        base_credits = int(total_hours / 27)  # Round down

        if overall_relevance_level == "low_relevance":
            self._apply_general_training_corrections(results, base_credits, total_hours)
        else:
            self._apply_professional_training_corrections(
                results, base_credits, total_hours, student_degree
            )

        results["degree_relevance_level"] = overall_relevance_level
        results["student_degree_program"] = student_degree

    def _apply_general_training_corrections(
        self, results: Dict[str, Any], base_credits: int, total_hours: int
    ):
        """Apply corrections for general training classification."""
        results["training_type"] = "general"
        results["degree_relevance"] = "low"

        if base_credits > 10:
            results["credits_qualified"] = 10.0
            results["conclusion"] = (
                "Student receives 10.0 ECTS credits as general training (capped at maximum limit)."
            )
        else:
            results["credits_qualified"] = float(base_credits)
            results["conclusion"] = (
                f"Student receives {base_credits}.0 ECTS credits as general training."
            )

        # Only add fallback if LLM didn't provide any justification
        if not results.get("summary_justification"):
            results["summary_justification"] = (
                "Work experience provides valuable general skills and transferable competencies, but does not directly align with the specific requirements of the degree program."
            )

        # Only add fallback if LLM didn't provide any relevance explanation
        if not results.get("relevance_explanation"):
            results["relevance_explanation"] = (
                "The roles and responsibilities do not sufficiently match the degree-specific criteria for professional training classification."
            )

        results["calculation_breakdown"] = (
            f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 10.0 maximum for general training"
        )

    def _apply_professional_training_corrections(
        self,
        results: Dict[str, Any],
        base_credits: int,
        total_hours: int,
        student_degree: str,
    ):
        """Apply corrections for professional training classification."""
        results["training_type"] = "professional"

        if base_credits > 30:
            results["credits_qualified"] = 30.0
            results["calculation_breakdown"] = (
                f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits, capped at 30.0 maximum for professional training"
            )
            results["conclusion"] = (
                "Student receives 30.0 ECTS credits as professional training. This provides full completion of the degree's practical training component."
            )
        else:
            results["credits_qualified"] = float(base_credits)
            results["calculation_breakdown"] = (
                f"{total_hours} hours / 27 hours per ECTS = {base_credits}.0 credits"
            )
            results["conclusion"] = (
                f"Student receives {base_credits}.0 ECTS credits as professional training."
            )

        results["summary_justification"] = (
            f"Professional experience directly related to {student_degree} with significant skill development and industry-specific knowledge relevant to the degree program."
        )

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response and extract JSON."""
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
            "model": self.model_name,
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
