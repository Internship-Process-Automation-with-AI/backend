"""
Configuration for information extraction patterns and keywords.
"""

# Finnish keywords and patterns
FINNISH_KEYWORDS = {
    "document_types": {
        "work_certificate": [
            "työtodistus",
            "työsuhdetodistus",
            "työsuhteen todistus",
            "työsuhteen päättymistodistus",
        ],
        "employment_letter": [
            "työsopimus",
            "työsuhteen aloittaminen",
            "työpaikka",
            "työtehtävä",
        ],
    },
    "employee_keywords": ["työntekijä", "henkilö", "henkilökuntaan", "työsuhteessa"],
    "position_keywords": ["tehtävä", "toimi", "työtehtävä", "virka", "asema", "rooli"],
    "employer_keywords": ["työnantaja", "yritys", "firma", "yhtiö", "toimisto"],
    "date_keywords": [
        "alkoi",
        "aloitti",
        "päättyi",
        "lopetti",
        "työsuhteen alkoi",
        "työsuhteen päättyi",
        "työaika",
        "työskentelyaika",
    ],
    "company_suffixes": ["Oy", "Ab", "Ltd", "Ky", "Tmi", "Yhtiö"],
}

# English keywords and patterns
ENGLISH_KEYWORDS = {
    "document_types": {
        "work_certificate": [
            "work certificate",
            "employment certificate",
            "certificate of employment",
            "work reference",
            "employment reference",
            "experience certificate",
        ],
        "employment_letter": [
            "employment letter",
            "job offer",
            "employment contract",
            "work agreement",
        ],
    },
    "employee_keywords": [
        "employee",
        "worker",
        "staff member",
        "personnel",
        "team member",
    ],
    "position_keywords": [
        "position",
        "role",
        "job title",
        "duties",
        "responsibilities",
        "function",
    ],
    "employer_keywords": ["employer", "company", "organization", "firm", "corporation"],
    "date_keywords": [
        "started",
        "began",
        "ended",
        "terminated",
        "employment period",
        "work period",
        "from",
        "to",
    ],
    "company_suffixes": ["Ltd", "Inc", "Corp", "LLC", "Company", "Limited"],
}

# Date patterns for different formats
DATE_PATTERNS = {
    "finnish": [
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})",  # DD.MM.YYYY
        r"(\d{1,2})\.(\d{1,2})\.(\d{2})",  # DD.MM.YY
        r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
    ],
    "english": [
        r"(\d{1,2})/(\d{1,2})/(\d{4})",  # MM/DD/YYYY
        r"(\d{1,2})-(\d{1,2})-(\d{4})",  # MM-DD-YYYY
        r"(\d{4})-(\d{1,2})-(\d{1,2})",  # YYYY-MM-DD
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})",  # Month DD, YYYY
        r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})",  # DD Month YYYY
    ],
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {"high": 0.8, "medium": 0.6, "low": 0.4, "minimum": 0.2}

# Extraction settings
EXTRACTION_SETTINGS = {
    "max_text_length": 10000,  # Maximum text length to process
    "min_confidence": 0.2,  # Minimum confidence to include field
    "language_detection": True,
    "enable_fallback": True,
}
