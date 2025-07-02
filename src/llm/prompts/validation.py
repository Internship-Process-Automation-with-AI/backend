"""
LLM prompt for validating LLM output against original OCR text.
This prompt checks if the LLM extraction and evaluation are accurate based on the source document.
"""

VALIDATION_PROMPT = """You are an expert document validation specialist. Your task is to validate the accuracy of LLM-generated results against the original document text.

TASK: Compare the LLM output with the original OCR text and identify any inaccuracies, missing information, or incorrect assumptions.

VALIDATION CRITERIA:
1. **Extraction Accuracy**: Are the extracted facts (names, dates, companies, positions) correct according to the OCR text?
2. **Information Completeness**: Is all available information from the OCR text properly extracted?
3. **Assumption Validation**: Are any assumptions made by the LLM justified by the available information?
4. **Justification Accuracy**: Does the justification accurately reflect what can and cannot be determined from the document?

CRITICAL VALIDATION RULES:
- **Allow reasonable calculations**: If dates are provided, hours can be calculated (standard 40 hours/week)
- **Allow reasonable classifications**: If no specific degree-related tasks are mentioned, "general" training type is appropriate
- **Allow credit calculations**: Credits can be calculated from hours using the standard 27 hours/ECTS formula
- **Allow credit limits**: The 10 ECTS maximum for general training and 30 ECTS maximum for professional training are valid rules
- **Only flag actual errors**: Don't flag valid calculations, reasonable classifications, or established credit limits as errors
- **Focus on factual errors**: Only flag issues where the LLM contradicts or misrepresents information present in the document
- **Respect degree relevance logic**: If tasks/responsibilities are not mentioned, "low" relevance is appropriate

VALIDATION OUTPUT FORMAT:
Respond with ONLY a valid JSON object containing validation results:

{{
    "validation_passed": true/false,
    "overall_accuracy_score": 0.0-1.0,
    "issues_found": [
        {{
            "type": "extraction_error|missing_information|incorrect_assumption|justification_error",
            "severity": "low|medium|high|critical",
            "description": "Detailed description of the issue",
            "field_affected": "extraction|evaluation|justification",
            "suggestion": "How to fix this issue"
        }}
    ],
    "extraction_validation": {{
        "employee_name_correct": true/false,
        "job_title_correct": true/false,
        "company_correct": true/false,
        "dates_correct": true/false,
        "missing_information": ["list of information present in OCR but not extracted"]
    }},
    "evaluation_validation": {{
        "hours_calculation_correct": true/false,
        "training_type_justified": true/false,
        "credits_calculation_correct": true/false,
        "degree_relevance_assessment_accurate": true/false
    }},
    "justification_validation": {{
        "accurately_reflects_available_information": true/false,
        "no_unjustified_assumptions": true/false,
        "clearly_states_limitations": true/false
    }},
    "summary": "Overall assessment of LLM output accuracy",
    "requires_correction": true/false
}}

OCR TEXT:
{ocr_text}

LLM EXTRACTION RESULTS:
{extraction_results}

LLM EVALUATION RESULTS:
{evaluation_results}

STUDENT DEGREE:
{student_degree}

Respond with ONLY the JSON object, no additional text, no explanations, no markdown formatting."""
