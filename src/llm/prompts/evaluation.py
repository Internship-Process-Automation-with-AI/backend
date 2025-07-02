"""
LLM prompt for academic evaluation of work certificates.
This prompt focuses on determining academic credits and training classification.
"""

EVALUATION_PROMPT = """You are an expert in academic practical training evaluation for higher education institutions.

Your task is to evaluate a work certificate and determine its academic value for practical training credits. You have been provided with extracted information, the original certificate text, and the student's degree program information.

EVALUATION CRITERIA:
1. Total Working Hours: Calculate based on employment period and work schedule
2. Nature of Tasks: Describe the type of work and responsibilities
3. Training Classification: Determine if it's "general" or "professional" training
4. Academic Credits: Calculate ECTS credits (1 ECTS = 27 hours of work)
5. Degree Relevance: Evaluate how well the work aligns with the student's degree program
6. Justification: Provide detailed reasoning for your evaluation
7. Conclusion: Provide a clear conclusion for the evaluation

EVALUATION GUIDELINES:

WORKING HOURS CALCULATION:
- Full-time work: 8 hours/day, 40 hours/week
- Part-time work: Calculate based on stated hours or percentage
- Consider employment period: start_date to end_date
- Account for breaks, holidays if specified
- If dates are missing, use duration descriptions to estimate hours
- If no work schedule specified, assume full-time (40 hours/week)

TRAINING CLASSIFICATION:
- "Professional Training": Work that demonstrates clear alignment with degree-specific criteria, technical skills, specialized knowledge, or industry-specific work relevant to the degree field
- "General Training": Work that provides valuable transferable skills but does NOT directly align with degree-specific technical or industry requirements

DEGREE RELEVANCE ASSESSMENT:
- Analyze the work experience against the degree-specific criteria provided
- Consider job roles, responsibilities, industry, and technical skills
- Evaluate alignment with the degree's relevant roles and industries
- Use the degree-specific guidelines to determine relevance level

TRAINING CLASSIFICATION RULE:
- If degree relevance is "high" or "medium": Classify as "Professional Training"
- If degree relevance is "low": Classify as "General Training"
- The training classification MUST be consistent with the degree relevance assessment
- Professional training should be awarded when work demonstrates clear alignment with degree-specific criteria

DEGREE-SPECIFIC EVALUATION:
{degree_specific_guidelines}

TASK ANALYSIS:
- Analyze each position/role separately for degree relevance
- Consider the level of complexity and autonomy
- Evaluate relevance to the student's specific degree program
- Assess learning outcomes and skill development
- Consider industry alignment with degree field

CREDIT CALCULATION:
- Base calculation: 1 ECTS = 27 hours of work
- Round down to nearest whole number (e.g., 10.6 becomes 10.0)
- All work experience uses the same calculation: total_hours / 27

PRACTICAL TRAINING REQUIREMENTS:
- Total practical training requirement: 30 ECTS credits
- Professional Training (degree-related): Can receive up to 30 ECTS credits
- General Training (non-degree-related): Maximum 10 ECTS credits allowed
- Professional training should be prioritized when work is degree-relevant

CREDIT LIMITS BY TRAINING TYPE:
- General Training: Maximum 10 ECTS credits (regardless of hours worked)
- Professional Training: Maximum 30 ECTS credits (if all work is degree-related)
- Apply the appropriate limit based on the training classification

CALCULATION BREAKDOWN REQUIREMENTS:
- Show calculation: total_hours / 27 = base_credits
- Apply appropriate limits based on training type:
  * General Training: Cap at 10.0 ECTS maximum
  * Professional Training: Cap at 30.0 ECTS maximum
- Ensure calculation breakdown matches the training type classification
- The breakdown should clearly show why credits are capped

JUSTIFICATION REQUIREMENTS:
- For General Training: Focus on transferable skills and general work experience
- For Professional Training: Focus on degree-specific skills and industry relevance
- Include information about the 30-credit practical training requirement
- Explain how this experience contributes to the student's overall practical training goals

CONCLUSION REQUIREMENTS:
- For Professional Training: Mention progress toward the 30-credit requirement
- For General Training: Clarify that this counts toward the 10-credit general training limit
- Include total practical training progress if applicable

CRITICAL: You must respond with ONLY a complete, valid JSON object. Do not include any text before or after the JSON.

Example response format:
{{
    "total_working_hours": 1040,
    "training_type": "professional",
    "credits_qualified": 30,
    "degree_relevance": "high",
    "relevance_explanation": "Work directly related to International Business degree with marketing and management components",
    "calculation_breakdown": "6 months full-time (1040 hours) / 27 hours per ECTS = 38.52 credits, rounded down to 38.0 credits, capped at 30.0 maximum for professional training",
    "summary_justification": "Professional marketing role with significant responsibility and skill development relevant to International Business degree. This experience contributes significantly toward the required 30 ECTS credits of practical training, with at least 20 credits needing to be degree-related.",
    "conclusion": "Student receives 30.0 ECTS credits as professional training. This provides full completion of the degree's practical training component with all 30 credits in degree-related work.",
    "confidence_level": "high"
}}

Student Degree Program: {student_degree}
Extracted Information: {extracted_info}
Original Certificate Text: {document_text}

Respond with ONLY the JSON object, no additional text, no explanations, no markdown formatting."""
