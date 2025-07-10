"""
Degree-specific evaluation logic for work certificate assessment.
This module handles different degree programs and their specific evaluation criteria.
"""

import logging
from typing import Any, Dict, List

from src.llm.degree_programs_data import DEGREE_PROGRAMS

logger = logging.getLogger(__name__)


class DegreeEvaluator:
    """Handles degree-specific evaluation of work certificates with bilingual support."""

    def __init__(self):
        """
        Initialize degree evaluator with bilingual data (Finnish + English).
        This ensures all degree programs are available and relevance scoring works
        regardless of certificate language.
        """
        # Always use the Finnish data which includes both Finnish and English keywords
        # This ensures bilingual matching for relevance scoring
        self.degree_programs = DEGREE_PROGRAMS

    def get_degree_info(self, degree_program: str) -> Dict[str, Any]:
        """
        Get degree program information.

        Args:
            degree_program: The degree program identifier

        Returns:
            Dictionary with degree program information
        """
        # First, try to find by exact name match
        for key, info in self.degree_programs.items():
            if degree_program.lower() == info["name"].lower():
                return info

        # Normalize the input: convert spaces to underscores and lowercase, remove special chars
        import re

        normalized_degree = re.sub(r"[^\w\s]", "", degree_program.lower()).replace(
            " ", "_"
        )

        # Try exact match with normalized key
        if normalized_degree in self.degree_programs:
            return self.degree_programs[normalized_degree]

        # Try exact match with original input (for backward compatibility)
        if degree_program.lower() in self.degree_programs:
            return self.degree_programs[degree_program.lower()]

        # Try partial name matching
        for key, info in self.degree_programs.items():
            if degree_program.lower() in info["name"].lower():
                return info

        # Default to general if no match found
        logger.warning(
            f"Unknown degree program: {degree_program}, using general criteria"
        )
        return self.degree_programs["general"]

    def get_degree_specific_guidelines(self, degree_program: str) -> str:
        """
        Get degree-specific evaluation guidelines for the LLM prompt.

        Args:
            degree_program: Student's degree program

        Returns:
            String with degree-specific guidelines
        """
        degree_info = self.get_degree_info(degree_program)

        guidelines = f"""
DEGREE-SPECIFIC EVALUATION GUIDELINES FOR {degree_info['name'].upper()}:

RELEVANT INDUSTRIES: {', '.join(degree_info['relevant_industries'])}
RELEVANT ROLES: {', '.join(degree_info['relevant_roles'])}

EVALUATION CRITERIA:
1. **Role Relevance**: How closely the job tasks align with {degree_info['name']} curriculum
2. **Industry Alignment**: Whether the work is in a relevant industry sector
3. **Skill Development**: Acquisition of skills directly applicable to the degree field
4. **Professional Growth**: Opportunities for career development in the field

PRACTICAL TRAINING REQUIREMENTS:
- Total practical training requirement: 30 ECTS credits
- Professional Training (degree-related): Minimum 20 ECTS credits required
- General Training (non-degree-related): Maximum 10 ECTS credits allowed

TRAINING CLASSIFICATION GUIDELINES:
- **Professional Training**: Work directly related to {degree_info['name']} with technical/specialized skills
- **General Training**: Basic work experience with general skills applicable to any field (ALWAYS receives credits)

IMPORTANT: General training provides valuable transferable skills (communication, teamwork, problem-solving, work ethic) and should ALWAYS receive academic credits, even when not directly related to the degree field.
"""

        return guidelines

    def validate_degree_program(self, degree_program: str) -> bool:
        """
        Validate if the degree program is supported.

        Args:
            degree_program: Degree program to validate

        Returns:
            True if supported, False otherwise
        """
        # Use the same normalization logic as get_degree_info
        normalized_degree = degree_program.lower().replace(" ", "_")

        # Try exact match with normalized key
        if normalized_degree in self.degree_programs:
            return True

        # Try exact match with original input (for backward compatibility)
        if degree_program.lower() in self.degree_programs:
            return True

        # Check if any degree program name contains the input
        for key, info in self.degree_programs.items():
            if degree_program.lower() in info["name"].lower():
                return True

        return False

    def get_supported_degree_programs(self) -> List[str]:
        """
        Get list of supported degree programs.

        Returns:
            List of supported degree program names
        """
        return [info["name"] for info in self.degree_programs.values()]
