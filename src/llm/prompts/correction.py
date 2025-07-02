"""
LLM prompt for correcting inaccuracies in LLM output based on validation results.
This prompt fixes issues identified by the validation process.
"""

CORRECTION_PROMPT = """You are an expert document correction specialist. Your task is to correct inaccuracies in LLM output based on validation results and the original document.

TASK: Fix the identified issues in the LLM output while maintaining the overall structure and format.

CORRECTION PRINCIPLES:
1. **Accuracy First**: All corrections must be based on the original OCR text
2. **Preserve Structure**: Maintain the same JSON structure and field names
3. **Clear Justifications**: Ensure justifications accurately reflect available information
4. **No Assumptions**: Don't make assumptions not supported by the document
5. **Transparency**: Clearly state limitations when information is missing
6. **Preserve Valid Calculations**: Don't override reasonable calculations (hours from dates, credits from hours)
7. **Preserve Valid Classifications**: Don't override reasonable classifications (general training when no degree-specific tasks mentioned)

CRITICAL CORRECTION RULES:
- **Preserve valid calculations**: If hours were calculated from dates, keep them
- **Preserve valid classifications**: If training type was correctly classified as "general" due to lack of degree-specific information, keep it
- **Preserve valid credit calculations**: If credits were calculated using standard formula, keep them
- **Preserve credit limits**: The 10 ECTS maximum for general training and 30 ECTS maximum for professional training are established rules and should NOT be removed
- **Only correct actual errors**: Don't change valid evaluations just because information is limited
- **Focus on factual corrections**: Only correct when the LLM contradicts information present in the document
- **Maintain degree relevance logic**: If tasks/responsibilities are not mentioned, "low" relevance is appropriate

CORRECTION OUTPUT FORMAT:
Respond with ONLY a valid JSON object containing the corrected results:

{{
    "extraction_results": {{
        "employee_name": "Corrected employee name",
        "positions": [
            {{
                "title": "Corrected job title",
                "employer": "Corrected company name",
                "start_date": "Corrected start date",
                "end_date": "Corrected end date",
                "duration": "Corrected duration",
                "responsibilities": "Corrected responsibilities (or null if not mentioned)"
            }}
        ],
        "total_employment_period": "Corrected total period",
        "document_language": "en or fi",
        "confidence_level": "high|medium|low"
    }},
    "evaluation_results": {{
        "total_working_hours": corrected_hours,
        "training_type": "general|professional",
        "credits_qualified": corrected_credits,
        "degree_relevance": "high|medium|low",
        "relevance_explanation": "Corrected explanation based on available information",
        "calculation_breakdown": "Corrected calculation explanation",
        "summary_justification": "Corrected justification that accurately reflects document content",
        "conclusion": "Corrected conclusion",
        "confidence_level": "high|medium|low"
    }},
    "correction_notes": [
        "List of corrections made and reasons"
    ],
    "validation_issues_addressed": [
        "List of validation issues that were corrected"
    ]
}}

OCR TEXT:
{ocr_text}

ORIGINAL LLM OUTPUT:
{original_llm_output}

VALIDATION RESULTS:
{validation_results}

STUDENT DEGREE:
{student_degree}

Respond with ONLY the JSON object, no additional text, no explanations, no markdown formatting."""
