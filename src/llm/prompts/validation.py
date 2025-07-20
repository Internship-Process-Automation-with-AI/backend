"""
LLM prompt for validating LLM output against original OCR text.
This prompt checks if the LLM extraction and evaluation are accurate based on the source document.
"""

VALIDATION_PROMPT = """You are an expert document validation specialist. Your task is to validate the accuracy of LLM-generated results against the original document text.

TASK: Compare the LLM output with the original OCR text and identify any inaccuracies, missing information, or incorrect assumptions.

IMPORTANT: The STUDENT DEGREE provided is the correct degree to use for validation. Do NOT try to determine or extract the student's degree from the document content. The degree provided is the degree the student is currently pursuing and should be used for all degree-related assessments.

VALIDATION CRITERIA:
1. **Extraction Accuracy**: Are the extracted facts (names, dates, companies, positions) correct according to the OCR text?
2. **Information Completeness**: Is all available information from the OCR text properly extracted?
3. **Assumption Validation**: Are any assumptions made by the LLM justified by the available information?
4. **Justification Accuracy**: Does the justification accurately reflect what can and cannot be determined from the document?
5. **Calculation Accuracy**: Are hours and credit calculations mathematically correct?
6. **Training Type Consistency**: Does the AI's recommendation align with the requested training type and provide appropriate evidence?

CRITICAL VALIDATION RULES:
- **Use the provided STUDENT DEGREE**: The student degree provided is the correct degree for evaluation. Do not try to determine the degree from the document content.
- **Validate factual accuracy**: Check if extracted information matches the original document
- **Validate calculations**: Verify hours and credit calculations are mathematically correct
- **Allow reasonable hour calculations**: If employment dates are provided, hours can be calculated using standard assumptions (40 hours/week, 8 hours/day)
- **Allow certificate issue date as end date**: When no explicit end date is provided, using the certificate issue date as the end date is a valid and reasonable assumption
- **Validate credit limits**: Ensure appropriate limits are applied (10 ECTS for general, 30 ECTS for professional). The 30 ECTS cap for professional training is CORRECT and should NOT be flagged as an error.
- **Focus on factual and logical errors**: Flag issues where the LLM contradicts document content or creates logical inconsistencies
- **Don't flag reasonable assumptions**: Standard working hour calculations from employment dates and using certificate issue date as end date are valid and should not be flagged as errors
- **DO NOT validate degree selection**: The student degree provided is the correct degree. Do not flag issues about what degree the student "should" have based on document content.
- **RESPECT REQUESTED TRAINING TYPE**: The REQUESTED_TRAINING_TYPE is the user's choice and should be respected. Do NOT flag it as an error if it doesn't match the AI's degree relevance assessment. The AI's job is to provide evidence and reasoning for whether the requested type is appropriate, not to override the user's choice.
- **Validate AI recommendation logic**: Check that the AI's recommendation and reasoning are consistent with the requested training type and degree relevance assessment, but do not force the training type to match the degree relevance.
- **CREDIT CAPS ARE CORRECT**: The 30 ECTS maximum for professional training and 10 ECTS maximum for general training are established business rules. Do NOT flag these as errors.
- **DECISION LOGIC IS VALID**: If the AI determines that work experience doesn't meet professional training criteria despite the user's request, this is valid reasoning and should NOT be flagged as an error.

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
        "degree_relevance_assessment_accurate": true/false,
        "recommendation_consistent": true/false,
        "evidence_alignment": "consistent|inconsistent"
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

REQUESTED TRAINING TYPE:
{requested_training_type}

SPECIAL VALIDATION CHECK:
- If the relevance_explanation mentions "significant alignment", "directly relevant", or "clear alignment" with the degree program, but the AI recommends against the requested training type without sufficient evidence, this may be flagged as inconsistent reasoning
- If the summary_justification mentions "fulfills the requirements for professional training" but the AI recommends against professional training without clear evidence, this may be flagged as inconsistent reasoning
- The AI's recommendation should be based on evidence from the document, not just degree relevance alone

Respond with ONLY the JSON object, no additional text, no explanations, no markdown formatting."""
