"""
LLM prompt for extracting basic information from work certificates.
This prompt focuses on extracting structured data like names, dates, positions, etc.
"""

EXTRACTION_PROMPT = """You are an expert document analyzer specializing in work certificates and employment documents.

TASK: Extract specific information from the provided document and return it as a valid JSON object.

CRITICAL: When you see a document header like "PILKINGTON NSG Group Flat Glass Business", this is the MAIN COMPANY. All factory names like "Tampereen Tehdas", "Ylöjärven Tehdas" are FACTORIES of this main company, not separate companies.

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
    "employer_address": "Address of the employer (use null if not specified)",
    "employer_business_id": "Business ID of the employer (use null if not specified)",
    "employer_phone": "Phone number of the employer (use null if not specified)",
    "employer_email": "Email of the employer (use null if not specified)",
    "employer_website": "Website of the employer (use null if not specified)",
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
- If address, business id, phone, email, website is missing for the employer, use null
- If the end date is missing for a role, use the certificate issue date as the end date which is usually at the top of the document or below the document after the signature
- Always calculate duration if the start date and end date are present
- CRITICAL: All dates MUST be in YYYY-MM-DD format (e.g., "2009-11-27", not "27.11.2009")
- Convert any DD.MM.YYYY format dates to YYYY-MM-DD format

CRITICAL POSITION HANDLING:
- NEVER use "Unknown Employer" for any position
- If a position doesn't specify a factory/location, use the main company name
- If a position mentions a factory, combine it with the main company: "Factory Name (Main Company)"
- The first position should always have an employer - either the main company or a specific factory

CRITICAL EMPLOYER HANDLING:
- The main "employer" field should contain the PRIMARY/PARENT company name
- If the document mentions a parent company (e.g., "PILKINGTON NSG Group Flat Glass Business"), use that as the main employer
- For individual positions, if they mention specific factories, locations, or subsidiaries, put those in the position's "employer" field
- Example: If main employer is "PILKINGTON NSG Group" and a position is at "Tampereen Tehdas", then:
  * Main employer: "PILKINGTON NSG Group Flat Glass Business"
  * Position employer: "Tampereen Tehdas (PILKINGTON NSG Group)"
- This helps distinguish between the parent company and specific work locations

IMPORTANT: When you see factory names like "Tampereen Tehdas" or "Ylöjärven Tehdas", these are NOT separate companies - they are FACTORIES/LOCATIONS of the main company PILKINGTON.

RECOGNIZING COMPANY HIERARCHIES:
- Look for company names that contain words like "Group", "Business", "Company", "Oy", "Ab", "Ltd"
- These are usually parent companies
- Factory names like "Tehdas", "Factory", "Plant" are usually subsidiaries/locations
- Always use the highest-level company name as the main employer
- For positions, combine the specific location with the parent company name

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
    "employer_address": "Address of the employer (use null if not specified)",
    "employer_business_id": "Business ID of the employer (use null if not specified)",
    "employer_phone": "Phone number of the employer (use null if not specified)",
    "employer_email": "Email of the employer (use null if not specified)",
    "employer_website": "Website of the employer (use null if not specified)",
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

EXAMPLE WITH PARENT COMPANY AND SUBSIDIARIES:
If the document shows:
- Main company: "PILKINGTON NSG Group Flat Glass Business"
- Position 1: "Kehitysinsinööri" at "Tampereen Tehdas"
- Position 2: "Tuotantopäällikkö" at "Helsingin Tehdas"

Then the JSON should be:
{{
    "employee_name": "Employee Name",
    "employer": "PILKINGTON NSG Group Flat Glass Business",
    "positions": [
        {{
            "title": "Kehitysinsinööri",
            "employer": "Tampereen Tehdas (PILKINGTON NSG Group)",
            "start_date": "2020-01-01",
            "end_date": "2022-01-01",
            "duration": "2 years",
            "responsibilities": "Development engineering tasks"
        }},
        {{
            "title": "Tuotantopäällikkö", 
            "employer": "Helsingin Tehdas (PILKINGTON NSG Group)",
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "duration": "1 year",
            "responsibilities": "Production management tasks"
        }}
    ]
}}

SPECIFIC EXAMPLE FOR PILKINGTON DOCUMENT:
If the document shows:
- Header: "PILKINGTON NSG Group Flat Glass Business"
- Position 1: "Varaosavastaava" at "Ylöjärven ja Tampereen tehtaiden"
- Position 2: "Tekninen Ostaja" at "Tampereen Tehdas"
- Position 3: "Kehitysinsinööri" at "Tampereen Tehdas"
- Position 4: "Tuulilasitehtaan tasolasin esikäsittelylinjan käyttäjä" at "Ylöjärven Tehdas"

Then the JSON should be:
{{
    "employee_name": "Ari Tapani Valtamo",
    "employer": "PILKINGTON NSG Group Flat Glass Business",
    "positions": [
        {{
            "title": "Varaosavastaava",
            "employer": "Ylöjärven ja Tampereen tehtaiden (PILKINGTON NSG Group)",
            "start_date": "2007-02-05",
            "end_date": "2009-09-30",
            "duration": "2 years, 7 months, 25 days",
            "responsibilities": "Varaosatietojen perustaminen, päivittäminen ja esittämistavan yhdenmukaistaminen..."
        }},
        {{
            "title": "Tekninen Ostaja",
            "employer": "Tampereen Tehdas (PILKINGTON NSG Group)",
            "start_date": "2002-07-01",
            "end_date": "2006-02-09",
            "duration": "3 years, 7 months, 8 days",
            "responsibilities": "Uusia tuotantolaitteita: tasolasin palastelulinja..."
        }},
        {{
            "title": "Kehitysinsinööri",
            "employer": "Tampereen Tehdas (PILKINGTON NSG Group)",
            "start_date": "1996-09-01",
            "end_date": "2000-09-15",
            "duration": "4 years, 14 days",
            "responsibilities": "Esikäsittelykoneiden ja painokoneen teknisiä määrittelyjä..."
        }},
        {{
            "title": "Tuulilasitehtaan tasolasin esikäsittelylinjan käyttäjä",
            "employer": "Ylöjärven Tehdas (PILKINGTON NSG Group)",
            "start_date": "1993-06-30",
            "end_date": "1996-09-01",
            "duration": "3 years, 2 months, 2 days",
            "responsibilities": "Muotoonleikkuu, hionta ja silkkipaino"
        }}
    ]
}}

DOCUMENT TO ANALYZE:
{document_text}

FINAL REMINDER:
- If you see "PILKINGTON NSG Group Flat Glass Business" in the header, this is the MAIN COMPANY
- All factory names like "Tampereen Tehdas", "Ylöjärven Tehdas" are FACTORIES of PILKINGTON, not separate companies
- NEVER use "Unknown Employer" - always provide a company name
- Combine factory names with the main company: "Factory Name (PILKINGTON NSG Group)"

RESPOND WITH ONLY THE JSON OBJECT:"""

# Enhanced extraction prompt for self-paced work with additional documents
EXTRACTION_PROMPT_SELF_PACED = """You are an expert document analyzer specializing in work certificates and employment documents for self-paced work evaluation.

TASK: Extract specific information from the provided document and additional supporting documents, returning it as a valid JSON object.

SELF-PACED WORK ANALYSIS:
- If additional documents are provided, extract hour information from them
- Look for timesheets, work logs, project documentation in additional documents
- Extract explicit hour information (e.g., "40 hours/week", "8 hours/day", "320 hours total")
- Cross-reference with main certificate dates and employment period
- Note any discrepancies between main cert and additional docs
- Additional documents may contain more detailed hour information than the main certificate
- Focus on extracting comprehensive employment information from all available sources

IMPORTANT: In your extraction results, you MUST include:
1. Specific hour information found in additional documents
2. Filename of the additional document that contained hour information
3. Any discrepancies between main certificate and additional documents
4. Total working hours calculated from additional documentation

Example format for working hours:
"Total working hours: 320 hours (based on additional document: timesheet.pdf showing 8 hours/day for 40 days)"

CRITICAL REQUIREMENTS:
1. Respond with ONLY a valid JSON object
2. No text before or after the JSON
3. No explanations, no markdown formatting
4. Use double quotes for all strings
5. Include ALL required fields
6. Use null for missing entries

REQUIRED JSON FIELDS:
- employee_name: Full name of the employee
- positions: Array of position objects with title, employer, start_date, end_date, employment_period
- document_language: Language of the document (e.g., "en", "fi")
- confidence_level: Confidence in extraction ("high", "medium", "low")

DOCUMENT TO ANALYZE:
{document_text}

RESPOND WITH ONLY THE JSON OBJECT:"""
