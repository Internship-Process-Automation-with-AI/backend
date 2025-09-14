"""
LLM prompt for extracting basic information from work certificates.
This prompt focuses on extracting structured data like names, dates, positions, etc.
"""

EXTRACTION_PROMPT = """You are an expert document analyzer specializing in work certificates and employment documents.

TASK: Extract specific information from the provided document and return it as a valid JSON object.

CRITICAL: When you see a document header like "PILKINGTON NSG Group Flat Glass Business", this is the MAIN COMPANY. All factory names like "Tampereen Tehdas", "Ylöjärven Tehdas" are FACTORIES of this main company, not separate companies.

CRITICAL EMPLOYEE NAME IDENTIFICATION:
1. **Look for phrases like**: "This is to certify that [NAME]", "hereby certify that [NAME]", "[NAME] has been employed", "Mr./Ms./Mrs. [NAME]"
2. **Finnish patterns**: "että [NAME] on työskennellyt", "Tämä todistaa, että [NAME]", "[NAME] on toiminut"
3. **Company headers are NOT employee names**: "HUS Helsinki University Hospital", "PILKINGTON NSG Group", "Digia Oy" are company names, not employee names
4. **Employee names typically appear**: After "certify that", after titles like "Mr./Ms./Mrs.", in employment statements, after "että" in Finnish
5. **Common patterns**:
   - "This is to certify that Mr. Bob Johnson has been employed..."
   - "We hereby certify that Anna Korhonen worked..."
   - "Tämä todistaa, että Eve Davis on työskennellyt..." (Finnish)
   - "että Pekka Virtanen on toiminut..." (Finnish)
   - "Employee: John Smith"
   - "Työntekijä: Maria Virtanen"
6. **Avoid company identifiers**: Skip text that contains "Oy", "Ab", "Ltd", "Inc", "Group", "Hospital", "University" when looking for employee names
7. **Look for personal titles**: Mr., Mrs., Ms., Herra, Rouva, Neiti often precede employee names
8. **Finnish company indicators**: "Oy", "Ab", "Oyj" are company suffixes, not part of personal names
9. **Context clues**: Employee names appear in the middle of sentences about employment, not at document headers

CRITICAL REQUIREMENTS:
1. Respond with ONLY a valid JSON object
2. No text before or after the JSON
3. No explanations, no markdown formatting
4. Use double quotes for all strings
5. Include ALL required fields
6. Use null for missing entries

REQUIRED JSON FIELDS:
{{
    "employee_name": "Full name of the person employed (NOT the company name)",
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
- Convert any Finnish date formats to YYYY-MM-DD format:
  * "2. 6. 2025" -> "2025-06-02"
  * "31. 8. 2025" -> "2025-08-31" 
  * "27.11.2009" -> "2009-11-27"
  * "13.2.1984" -> "1984-02-13"
- Always pad single digits with leading zeros (e.g., "2" becomes "02")

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
    "employer_address": null,
    "employer_business_id": null,
    "employer_phone": null,
    "employer_email": null,
    "employer_website": null,
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
    "confidence_level": "high"
}}

EXAMPLE WITH ENGLISH CERTIFICATE:
For a document like:
"HUS Helsinki University Hospital
Certificate of Employment
This is to certify that Mr. Bob Johnson has been employed at Helsinki City Hospital as a Nursing Assistant from March 1, 2022 to August 30, 2023."

The JSON should be:
{{
    "employee_name": "Bob Johnson",
    "employer": "Helsinki University Hospital",
    "employer_address": null,
    "employer_business_id": null,
    "employer_phone": null,
    "employer_email": null,
    "employer_website": null,
    "certificate_issue_date": "2023-09-05",
    "positions": [
        {{
            "title": "Nursing Assistant",
            "employer": "Helsinki City Hospital",
            "start_date": "2022-03-01",
            "end_date": "2023-08-30",
            "duration": "1 year, 5 months",
            "responsibilities": "Patient care, assisted in medical procedures, maintained professional nursing standards"
        }}
    ],
    "total_employment_period": "1 year, 5 months",
    "document_language": "en",
    "confidence_level": "high"
}}

EXAMPLE WITH FINNISH CERTIFICATE:
For a document like:
"Digia Oy
Kasarmintie 21, 90130 Oulu
Tämä todistaa, että Eve Davis on työskennellyt yrityksessä Digia Oy tehtävässä Ohjelmistokehittäjä harjoittelija ajalla 1.2.2022 - 31.7.2022."

The JSON should be:
{{
    "employee_name": "Eve Davis",
    "employer": "Digia Oy",
    "employer_address": "Kasarmintie 21, 90130 Oulu",
    "employer_business_id": null,
    "employer_phone": null,
    "employer_email": null,
    "employer_website": null,
    "certificate_issue_date": "2022-08-05",
    "positions": [
        {{
            "title": "Ohjelmistokehittäjä harjoittelija",
            "employer": "Digia Oy",
            "start_date": "2022-02-01",
            "end_date": "2022-07-31",
            "duration": "6 months",
            "responsibilities": "Ohjelmistojen kehitykseen, testaukseen sekä tietokantaylläpitoon"
        }}
    ],
    "total_employment_period": "6 months",
    "document_language": "fi",
    "confidence_level": "high"
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

CRITICAL NAME EXTRACTION RULES:
- **EMPLOYEE NAME IS NOT THE COMPANY NAME**: "HUS Helsinki" is a company, not an employee name
- **Look for certification phrases**: "This is to certify that [EMPLOYEE NAME]", "hereby certify that [NAME]"
- **Look for titles**: "Mr. Bob Johnson", "Ms. Anna Smith", "Herra Pekka Virtanen"
- **Employee names appear in employment statements**: "[NAME] has been employed", "[NAME] worked as"
- **Skip organizational headers**: The first few lines are usually company information, not employee names
- **Personal names vs organizational names**: 
  * Personal: "Bob Johnson", "Anna Virtanen", "Pekka Korhonen"
  * Organizational: "HUS Helsinki", "University Hospital", "NSG Group"

FINAL REMINDER:
- If you see "PILKINGTON NSG Group Flat Glass Business" in the header, this is the MAIN COMPANY
- All factory names like "Tampereen Tehdas", "Ylöjärven Tehdas" are FACTORIES of PILKINGTON, not separate companies
- NEVER use "Unknown Employer" - always provide a company name
- Combine factory names with the main company: "Factory Name (PILKINGTON NSG Group)"
- **MOST IMPORTANT**: The employee_name field should contain the PERSON'S name, not the company name

RESPOND WITH ONLY THE JSON OBJECT:"""
