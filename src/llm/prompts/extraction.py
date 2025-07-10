"""
LLM prompt for extracting basic information from work certificates.
This prompt focuses on extracting structured data like names, dates, positions, etc.
"""

EXTRACTION_PROMPT = """You are an expert document analyzer specializing in work certificates and employment documents.

TASK: Extract specific information from the provided document and return it as a valid JSON object.

CRITICAL REQUIREMENTS:
1. Respond with ONLY a valid JSON object
2. No text before or after the JSON
3. No explanations, no markdown formatting
4. Use double quotes for all strings
5. Include ALL required fields
6. Use null for missing entries

REQUIRED JSON FIELDS:
{{
    "employee_name": "Full name of the person employed",
    "employer": "Company or organization name (use null if not specified)",
    "certificate_issue_date": "Date when the certificate was issued (usually at the top of the document) in YYYY-MM-DD format ONLY (use null if not found)",
    "positions": [
        {{
            "title": "Job title or role",
            "employer": "Company or organization name for this specific position (use null if not specified)", 
            "start_date": "Start date in YYYY-MM-DD format (use null if not specified)",
            "end_date": "End date in YYYY-MM-DD format ONLY (use null if not specified)",
            "duration": "Duration description (e.g., '6 months', '1 year')",
            "responsibilities": "Key responsibilities and tasks for this role"
        }}
    ],
    "total_employment_period": "Total duration description (e.g., '2 years, 6 months')",
    "document_language": "en or fi",
    "confidence_level": "high(>75%), medium(50-75%), low(<50%), or null (if not specified)"
}}

IMPORTANT GUIDELINES:
- Extract ALL positions/roles and employers mentioned in the document
- Each position should have its own entry with dates, responsibilities, and employer
- If only one position is mentioned, still use the array format
- Focus on extracting specific responsibilities and tasks for each role
- If dates are missing for specific roles, use null
- If employer is missing for specific roles, use null
- If the end date is missing for a role, use the certificate issue date as the end date which is usually at the top of the document or below the document after the signature
- Always calculate duration if the start date and end date are present
- CRITICAL: All dates MUST be in YYYY-MM-DD format (e.g., "2009-11-27", not "27.11.2009")
- Convert any DD.MM.YYYY format dates to YYYY-MM-DD format

EXAMPLE DOCUMENT:
Työtodistus
27.11.2009
Insinööri Ari Tapani Valtamo (syntynyt 31.1. 1963) on toiminut yrityksessämme 13.2.1984 – 30.9.2009 välisenä aikana seuraavissa tehtävissä:
Varaosavastaava (5.2.2007 - 30.9.2009)
Varaosatietojen perustaminen, päivittäminen ja esittämistavan yhdenmukaistaminen
Kunnossapitovastaava (13.2.1984 - 4.2.2007)
Kunnossapitotoimenpiteiden suunnittelu ja toteutus

EXAMPLE JSON RESPONSE:
{{
    "employee_name": "Ari Tapani Valtamo",
    "employer": "Yritys",
    "certificate_issue_date": "2009-11-27",
    "positions": [
        {{
            "title": "Varaosavastaava",
            "employer": "Yritys",
            "start_date": "2007-02-05",
            "end_date": "2009-09-30",
            "duration": "2 years, 7 months",
            "responsibilities": "Varaosatietojen perustaminen, päivittäminen ja esittämistavan yhdenmukaistaminen"
        }},
        {{
            "title": "Kunnossapitovastaava",
            "employer": "Yritys",
            "start_date": "1984-02-13",
            "end_date": "2007-02-04",
            "duration": "23 years",
            "responsibilities": "Kunnossapitotoimenpiteiden suunnittelu ja toteutus"
        }}
    ],
    "total_employment_period": "25 years, 7 months",
    "document_language": "fi",
    "confidence_level": "high(>75%)"
}}

DOCUMENT TO ANALYZE:
{document_text}

RESPOND WITH ONLY THE JSON OBJECT:"""
