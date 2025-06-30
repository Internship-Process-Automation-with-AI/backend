"""
LLM Orchestrator for Work Certificate Processing
Manages a two-stage process: extraction followed by evaluation.
"""

import json
import logging
import time
from typing import Any, Dict

import google.generativeai as genai

from src.config import settings
from src.llm.prompts import EVALUATION_PROMPT, EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Orchestrates LLM-based work certificate processing using a two-stage approach."""

    def __init__(self):
        """Initialize the LLM orchestrator with Gemini."""
        self.model = None
        self.model_name = settings.GEMINI_MODEL
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini API client."""
        try:
            if not settings.GEMINI_API_KEY:
                logger.warning(
                    "Gemini API key not provided - LLM processing unavailable"
                )
                return

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"LLM Orchestrator initialized with model: {self.model_name}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Gemini with model {self.model_name}: {e}"
            )
            self.model = None

    def process_work_certificate(self, text: str) -> Dict[str, Any]:
        """
        Process a work certificate using the two-stage LLM approach.

        Args:
            text: Cleaned OCR text from the work certificate

        Returns:
            Dictionary with both extraction and evaluation results
        """
        if not self.model:
            return {
                "success": False,
                "error": "Gemini API not available",
                "extraction_results": None,
                "evaluation_results": None,
            }

        # Validate input text
        if not text or not isinstance(text, str):
            return {
                "success": False,
                "error": f"Invalid text input: {type(text)} - {repr(text)[:100]}",
                "extraction_results": None,
                "evaluation_results": None,
            }

        # Check for suspicious content that might indicate corruption
        if len(text) < 10 or text.strip() == "":
            return {
                "success": False,
                "error": f"Text too short or empty: {repr(text)}",
                "extraction_results": None,
                "evaluation_results": None,
            }

        # Check for JSON-like content that shouldn't be in input text
        if text.strip().startswith('"') or text.strip().startswith("{"):
            return {
                "success": False,
                "error": f"Text appears to be JSON, not document content: {repr(text)[:100]}",
                "extraction_results": None,
                "evaluation_results": None,
            }

        start_time = time.time()

        try:
            # Stage 1: Information Extraction
            logger.info("Starting Stage 1: Information Extraction")
            sanitized_text = self._sanitize_text_for_llm(text)
            extraction_result = self._extract_information(sanitized_text)

            if not extraction_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Extraction failed: {extraction_result.get('error', 'Unknown error')}",
                    "extraction_results": extraction_result,
                    "evaluation_results": None,
                    "processing_time": time.time() - start_time,
                }

            # Stage 2: Academic Evaluation
            logger.info("Starting Stage 2: Academic Evaluation")
            evaluation_result = self._evaluate_academically(
                sanitized_text, extraction_result["results"]
            )

            total_time = time.time() - start_time

            return {
                "success": True,
                "processing_time": total_time,
                "extraction_results": extraction_result,
                "evaluation_results": evaluation_result,
                "model_used": self.model_name,
                "stages_completed": {
                    "extraction": extraction_result.get("success", False),
                    "evaluation": evaluation_result.get("success", False),
                },
            }

        except Exception as e:
            logger.error(f"Error in LLM orchestration: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "extraction_results": None,
                "evaluation_results": None,
                "model_used": self.model_name,
            }

    def _sanitize_text_for_llm(self, text: str) -> str:
        """
        Sanitize and clean text before sending to LLM.

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned and sanitized text
        """
        if not text:
            return ""

        # Remove excessive whitespace and normalize
        import re

        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove excessive spaces
        text = re.sub(r" +", " ", text)

        # Remove leading/trailing whitespace from each line
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)

        # Join lines back together
        text = "\n".join(cleaned_lines)

        # Limit text length to prevent token overflow
        max_length = 8000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Text truncated due to length]"

        return text.strip()

    def _extract_information(self, text: str) -> Dict[str, Any]:
        """Stage 1: Extract basic information from the certificate."""
        stage_start = time.time()

        try:
            # Log text length for debugging
            logger.info(f"Processing text of length: {len(text)} characters")
            logger.info(f"Text parameter at start: {repr(text[:100])}")
            logger.debug(f"Text preview: {text[:200]}...")

            # Create extraction prompt
            logger.info("Creating extraction prompt...")
            prompt = EXTRACTION_PROMPT.format(document_text=text)
            logger.info(f"Prompt created successfully, length: {len(prompt)}")

            # Generate response
            logger.info("Generating LLM response...")
            response = self.model.generate_content(prompt)

            # Log raw response for debugging
            logger.info(f"Raw LLM response length: {len(response.text)} characters")
            logger.info(f"Raw LLM response: {repr(response.text)}")

            # Parse response
            results = self._parse_llm_response(response.text)

            stage_time = time.time() - stage_start

            return {
                "success": True,
                "processing_time": stage_time,
                "results": results,
                "raw_response": response.text,
            }

        except Exception as e:
            logger.error(f"Error in extraction stage: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Text parameter at error: {repr(text[:100])}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - stage_start,
                "results": None,
            }

    def _evaluate_academically(
        self, text: str, extracted_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: Evaluate the certificate for academic credits."""
        stage_start = time.time()

        try:
            # Format extracted info for the evaluation prompt
            extracted_info_str = json.dumps(
                extracted_info, indent=2, ensure_ascii=False
            )

            # Create evaluation prompt
            prompt = EVALUATION_PROMPT.format(
                extracted_info=extracted_info_str, document_text=text
            )

            # Generate response
            response = self.model.generate_content(prompt)

            # Parse response
            results = self._parse_llm_response(response.text)

            stage_time = time.time() - stage_start

            return {
                "success": True,
                "processing_time": stage_time,
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

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response and extract JSON."""
        try:
            # Clean the response text first
            response_text = response_text.strip()
            logger.info(f"Cleaned response text: {repr(response_text)}")

            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                logger.error(
                    f"No JSON brackets found in response: {response_text[:200]}..."
                )
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx]
            logger.info(f"Extracted JSON string: {repr(json_str)}")

            # Try to parse the JSON
            try:
                parsed_json = json.loads(json_str)
                logger.info(f"Successfully parsed JSON: {parsed_json}")
                return parsed_json
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Attempted to parse: {json_str}")

                # Try to fix common JSON issues
                fixed_json = self._fix_common_json_issues(json_str)
                logger.info(f"Fixed JSON: {repr(fixed_json)}")
                try:
                    parsed_json = json.loads(fixed_json)
                    logger.info(f"Successfully parsed fixed JSON: {parsed_json}")
                    return parsed_json
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to fix JSON: {e2}")
                    logger.error(f"Fixed JSON was: {fixed_json}")

                    # If we have a partial JSON like '\n    "employee_name"', try to construct a basic response
                    if '"employee_name"' in json_str:
                        logger.info(
                            "Detected partial JSON with employee_name, creating fallback response"
                        )
                        return self._create_fallback_response(json_str)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")

            # Return a structured error response
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response_text[:500],
                "confidence_level": "low",
                "employee_name": "Unknown",
                "position": "Unknown",
                "employer": "Unknown",
                "start_date": None,
                "end_date": None,
                "employment_period": "Unknown",
                "document_language": "en",
            }

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        import re

        # Remove any text before the first {
        json_str = json_str[json_str.find("{") :]

        # Remove any text after the last }
        json_str = json_str[: json_str.rfind("}") + 1]

        # Fix missing quotes around keys
        json_str = re.sub(r"(\s*)(\w+)(\s*):", r'\1"\2"\3:', json_str)

        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')

        # Fix trailing commas
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # Fix null values
        json_str = re.sub(r":\s*null\s*([,}])", r": null\1", json_str)

        return json_str

    def _create_fallback_response(self, partial_json: str) -> Dict[str, Any]:
        """Create a fallback response when we have partial JSON."""
        import re

        # Try to extract any available information from the partial JSON
        employee_name = "Unknown"
        position = "Unknown"
        employer = "Unknown"

        # Extract employee_name if present
        name_match = re.search(r'"employee_name"\s*:\s*"([^"]+)"', partial_json)
        if name_match:
            employee_name = name_match.group(1)

        # Extract position if present
        position_match = re.search(r'"position"\s*:\s*"([^"]+)"', partial_json)
        if position_match:
            position = position_match.group(1)

        # Extract employer if present
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
            "stages": ["extraction", "evaluation"],
        }

    def get_prompt_info(self) -> Dict[str, Any]:
        """Get information about the prompts being used."""
        return {
            "extraction_prompt_length": len(EXTRACTION_PROMPT),
            "evaluation_prompt_length": len(EVALUATION_PROMPT),
            "total_prompt_length": len(EXTRACTION_PROMPT) + len(EVALUATION_PROMPT),
        }
