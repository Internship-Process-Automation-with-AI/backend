"""
Degree-specific evaluation logic for work certificate assessment.
This module handles different degree programs and their specific evaluation criteria.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DegreeEvaluator:
    """Handles degree-specific evaluation of work certificates."""

    def __init__(self):
        # Define degree programs and their evaluation criteria
        self.degree_programs = {
            # 1. Informaatioteknologia (Information Technology)
            "information_technology": {
                "name": "Information Technology",
                "fields": [
                    "Bachelor of Engineering (BEng), Information Technology",
                    "Insinööri (AMK), tieto- ja viestintätekniikka",
                    "Tradenomi (AMK), tietojenkäsittely",
                ],
                "relevant_industries": [
                    "technology",
                    "software",
                    "consulting",
                    "finance",
                    "healthcare",
                    "education",
                    "telecommunications",
                ],
                "relevant_roles": [
                    "software development",
                    "programming",
                    "system administration",
                    "database management",
                    "network administration",
                    "cybersecurity",
                    "data analysis",
                    "web development",
                    "IT support",
                    "technical support",
                    "software engineering",
                    "data science",
                    "information systems",
                    "digital transformation",
                    "cloud computing",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.5,
                    "medium_relevance": 1.3,
                    "low_relevance": 1.0,
                },
            },
            # 2. Kulttuuri (Culture)
            "culture": {
                "name": "Culture",
                "fields": [
                    "Medianomi (AMK)",
                    "Musiikkipedagogi (AMK)",
                    "Tanssinopettaja (AMK)",
                ],
                "relevant_industries": [
                    "media",
                    "entertainment",
                    "education",
                    "arts",
                    "culture",
                    "broadcasting",
                    "publishing",
                ],
                "relevant_roles": [
                    "media production",
                    "content creation",
                    "teaching",
                    "pedagogy",
                    "music education",
                    "dance instruction",
                    "cultural management",
                    "arts administration",
                    "creative direction",
                    "multimedia design",
                    "audio production",
                    "visual arts",
                    "performing arts",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.4,
                    "medium_relevance": 1.2,
                    "low_relevance": 1.0,
                },
            },
            # 3. Luonnonvara-ala (Natural Resources)
            "natural_resources": {
                "name": "Natural Resources",
                "fields": ["Agrologi (AMK), maaseutuelinkeinot"],
                "relevant_industries": [
                    "agriculture",
                    "forestry",
                    "environmental",
                    "rural development",
                    "food production",
                    "sustainability",
                ],
                "relevant_roles": [
                    "agricultural consulting",
                    "rural development",
                    "environmental management",
                    "sustainable farming",
                    "food production",
                    "forestry management",
                    "land use planning",
                    "agricultural technology",
                    "environmental assessment",
                    "rural business development",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.4,
                    "medium_relevance": 1.2,
                    "low_relevance": 1.0,
                },
            },
            # 4. Liiketalous (Business Administration)
            "business_administration": {
                "name": "Business Administration",
                "fields": [
                    "Bachelor of Business Administration (BBA), International Business",
                    "Tradenomi (AMK), liiketalous",
                    "Tradenomi (AMK), liiketalous, verkkokoulutus",
                ],
                "relevant_industries": [
                    "finance",
                    "marketing",
                    "consulting",
                    "retail",
                    "manufacturing",
                    "technology",
                    "international trade",
                ],
                "relevant_roles": [
                    "marketing",
                    "sales",
                    "finance",
                    "accounting",
                    "management",
                    "consulting",
                    "business development",
                    "strategy",
                    "operations",
                    "human resources",
                    "international business",
                    "trade",
                    "logistics",
                    "supply chain management",
                    "entrepreneurship",
                    "business analysis",
                    "project management",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.4,
                    "medium_relevance": 1.2,
                    "low_relevance": 1.0,
                },
            },
            # 5. Sosiaali- ja terveysala (Social and Health Care)
            "healthcare": {
                "name": "Healthcare",
                "fields": [
                    "Bachelor of Health Care, Nursing",
                    "Bioanalyytikko (AMK)",
                    "Ensihoitaja (AMK)",
                    "Fysioterapeutti (AMK)",
                    "Kätilö (AMK)",
                    "Optometristi (AMK)",
                    "Röntgenhoitaja (AMK)",
                    "Sairaanhoitaja (AMK)",
                    "Sosionomi (AMK)",
                    "Suuhygienisti (AMK)",
                    "Terveydenhoitaja (AMK)",
                    "Toimintaterapeutti (AMK)",
                ],
                "relevant_industries": [
                    "healthcare",
                    "pharmaceuticals",
                    "medical devices",
                    "public health",
                    "social services",
                    "rehabilitation",
                ],
                "relevant_roles": [
                    "patient care",
                    "clinical work",
                    "health administration",
                    "research",
                    "public health",
                    "medical support",
                    "healthcare management",
                    "nursing",
                    "social work",
                    "rehabilitation",
                    "medical technology",
                    "health promotion",
                    "emergency care",
                    "maternal care",
                    "optical care",
                    "radiology",
                    "dental hygiene",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.4,
                    "medium_relevance": 1.2,
                    "low_relevance": 1.0,
                },
            },
            # 6. Tekniikka (Engineering)
            "engineering": {
                "name": "Engineering",
                "fields": [
                    "Bachelor of Engineering, Energy and Environmental Engineering",
                    "Bachelor of Engineering, Mechanical Engineering",
                    "Insinööri (AMK), energia- ja ympäristötekniikka",
                    "Insinööri (AMK), konetekniikka",
                    "Insinööri (AMK), sähkö- ja automaatiotekniikka",
                    "Insinööri (AMK), talotekniikka",
                    "Insinööri (AMK), rakennus- ja yhdyskuntatekniikka",
                    "Rakennusarkkitehti (AMK)",
                    "Rakennusmestari (AMK)",
                ],
                "relevant_industries": [
                    "manufacturing",
                    "construction",
                    "technology",
                    "automotive",
                    "aerospace",
                    "energy",
                    "environmental",
                    "building services",
                ],
                "relevant_roles": [
                    "design",
                    "development",
                    "testing",
                    "maintenance",
                    "project management",
                    "research",
                    "quality control",
                    "technical support",
                    "system administration",
                    "energy engineering",
                    "environmental engineering",
                    "mechanical engineering",
                    "electrical engineering",
                    "automation",
                    "building technology",
                    "construction management",
                    "architecture",
                    "sustainable design",
                    "energy efficiency",
                ],
                "quality_multipliers": {
                    "high_relevance": 1.5,
                    "medium_relevance": 1.3,
                    "low_relevance": 1.0,
                },
            },
            # General Studies (fallback)
            "general": {
                "name": "General Studies",
                "fields": [
                    "Liberal Arts",
                    "General Studies",
                    "Interdisciplinary Studies",
                ],
                "relevant_industries": ["any"],
                "relevant_roles": ["any"],
                "quality_multipliers": {
                    "high_relevance": 1.2,
                    "medium_relevance": 1.1,
                    "low_relevance": 1.0,
                },
            },
        }

    def get_degree_info(self, degree_program: str) -> Dict[str, Any]:
        """
        Get degree program information.

        Args:
            degree_program: The degree program identifier

        Returns:
            Dictionary with degree program information
        """
        degree_program_lower = degree_program.lower()

        # Try exact match first
        if degree_program_lower in self.degree_programs:
            return self.degree_programs[degree_program_lower]

        # Try partial matches
        for key, info in self.degree_programs.items():
            if key in degree_program_lower or degree_program_lower in key:
                return info

        # Default to general if no match found
        logger.warning(
            f"Unknown degree program: {degree_program}, using general criteria"
        )
        return self.degree_programs["general"]

    def calculate_relevance_score(
        self,
        degree_program: str,
        job_title: str,
        job_description: str,
        company_industry: str = "",
    ) -> Tuple[str, float]:
        """
        Calculate relevance score between work experience and degree program.

        Args:
            degree_program: Student's degree program
            job_title: Job title from certificate
            job_description: Job description/tasks from certificate
            company_industry: Company industry (optional)

        Returns:
            Tuple of (relevance_level, quality_multiplier)
        """
        degree_info = self.get_degree_info(degree_program)

        # Combine all text for analysis
        combined_text = f"{job_title} {job_description} {company_industry}".lower()

        # Count matches in relevant roles
        role_matches = 0
        for role in degree_info["relevant_roles"]:
            if role.lower() in combined_text:
                role_matches += 1

        # Count matches in relevant industries
        industry_matches = 0
        if company_industry:
            for industry in degree_info["relevant_industries"]:
                if industry.lower() in company_industry.lower():
                    industry_matches += 1

        # Calculate relevance score
        total_possible_roles = len(degree_info["relevant_roles"])
        role_score = role_matches / max(1, total_possible_roles)

        # Industry bonus
        industry_bonus = 0.2 if industry_matches > 0 else 0.0

        # Final relevance score (0.0 to 1.0)
        relevance_score = min(1.0, role_score + industry_bonus)

        # Determine relevance level and multiplier
        if relevance_score >= 0.6:
            relevance_level = "high_relevance"
        elif relevance_score >= 0.3:
            relevance_level = "medium_relevance"
        else:
            relevance_level = "low_relevance"

        quality_multiplier = degree_info["quality_multipliers"][relevance_level]

        logger.info(
            f"Relevance score: {relevance_score:.2f}, Level: {relevance_level}, Multiplier: {quality_multiplier}"
        )

        return relevance_level, quality_multiplier

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

QUALITY MULTIPLIERS:
- High Relevance (directly related to {degree_info['name']}): {degree_info['quality_multipliers']['high_relevance']}x
- Medium Relevance (somewhat related): {degree_info['quality_multipliers']['medium_relevance']}x  
- Low Relevance (general work experience): {degree_info['quality_multipliers']['low_relevance']}x

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
        degree_program_lower = degree_program.lower()

        # Check exact matches
        if degree_program_lower in self.degree_programs:
            return True

        # Check partial matches
        for key in self.degree_programs.keys():
            if key in degree_program_lower or degree_program_lower in key:
                return True

        return False

    def get_supported_degree_programs(self) -> List[str]:
        """
        Get list of supported degree programs.

        Returns:
            List of supported degree program names
        """
        return [info["name"] for info in self.degree_programs.values()]
