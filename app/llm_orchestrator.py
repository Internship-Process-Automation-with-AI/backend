"""
LLM Orchestrator for Work Certificate Processing
Manages a two-stage process: extraction followed by evaluation.
"""

import json
import logging
import time
from typing import Any, Dict

import google.generativeai as genai

from app.config import settings
from app.prompts import EVALUATION_PROMPT, EXTRACTION_PROMPT

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

        start_time = time.time()

        try:
            # Stage 1: Information Extraction
            logger.info("Starting Stage 1: Information Extraction")
            extraction_result = self._extract_information(text)

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
                text, extraction_result["results"]
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

    def _extract_information(self, text: str) -> Dict[str, Any]:
        """Stage 1: Extract basic information from the certificate."""
        stage_start = time.time()

        try:
            # Create extraction prompt
            prompt = EXTRACTION_PROMPT.format(text=text)

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
            logger.error(f"Error in extraction stage: {e}")
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
                extracted_info=extracted_info_str, text=text
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
            # Try to find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response text: {response_text}")

            # Return a structured error response
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response_text,
                "confidence_level": "low",
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
