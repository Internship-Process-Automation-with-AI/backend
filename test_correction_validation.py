"""
Test script to verify structural validation after correction.
"""

import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.llm.models import validate_evaluation_results, validate_extraction_results


def test_correction_validation():
    """Test structural validation with corrected data that might have new issues."""

    # Simulated correction result that might introduce new issues
    correction_result = {
        "success": True,
        "results": {
            "extraction_results": {
                "employee_name": "Jonatan Farrell",
                "employer": "Your Company Name",
                "certificate_issue_date": "2024-11-27",  # Fixed date
                "positions": [
                    {
                        "title": "Software Developer",
                        "employer": "Your Company Name",
                        "start_date": "2020-01-15",  # Fixed date
                        "end_date": "2024-10-31",  # Fixed date
                        "duration": None,
                        "responsibilities": "integral part of our team and contributed significantly to the company's growth and success.",
                    },
                    {
                        "title": "Software Developer",
                        "employer": "Your Company Name",  # Fixed employer
                        "start_date": "2018-01-15",
                        "end_date": "2019-12-31",  # Fixed to not overlap
                        "duration": None,
                        "responsibilities": "Develop and maintain software applications\nWork with cross-functional teams to develop new features.\nTroubleshoot and debug software issues\nProvide training and support to junior developers",
                    },
                ],
                "total_employment_period": None,
                "document_language": "en",
                "confidence_level": "high",
            },
            "evaluation_results": {
                "total_working_hours": 28080,  # Corrected hours
                "training_type": "professional",
                "credits_qualified": 30.0,
                "degree_relevance": "high",
                "relevance_explanation": "The candidate worked as a Software Developer, which directly aligns with the BEng in Information Technology. Responsibilities include software development, feature development with cross-functional teams, troubleshooting, debugging, and providing training, all of which are highly relevant to the degree.",
                "calculation_breakdown": "28080 hours / 27 hours per ECTS = 1040.0 credits, capped at 30.0 maximum for professional training",
                "summary_justification": "The role of Software Developer is directly relevant to a Bachelor of Engineering in Information Technology. The responsibilities listed demonstrate development skills and the ability to work within a team. This experience significantly contributes toward the required 30 ECTS credits of practical training, with at least 20 credits needing to be degree-related.",
                "conclusion": "Student receives 30.0 ECTS credits as professional training, fulfilling the total practical training requirement for the degree.",
                "confidence_level": "high",
            },
            "correction_notes": [
                "Corrected the end date for the first 'Software Developer' position to '2024-10-31'.",
                "Corrected the employer for the second 'Software Developer' position to 'Your Company Name'.",
                "Recalculated the total working hours based on the employment dates and standard work week assumptions.",
            ],
            "validation_issues_addressed": [
                "extraction_error: end date for the first 'Software Developer' position",
                "extraction_error: employer for the second Software Developer position",
                "hours_calculation_correct",
            ],
        },
    }

    print("=== Testing Correction Validation ===\n")

    # Test validation of corrected extraction results
    print("1. Testing Corrected Extraction Validation:")
    print("-" * 50)
    try:
        corrected_extraction_validation = validate_extraction_results(
            correction_result["results"]["extraction_results"]
        )
        print(f"Validation passed: {corrected_extraction_validation.validation_passed}")
        print(f"Summary: {corrected_extraction_validation.summary}")
        print(f"Issues found: {len(corrected_extraction_validation.issues_found)}")

        for i, issue in enumerate(corrected_extraction_validation.issues_found, 1):
            print(f"\nIssue {i}:")
            print(f"  Type: {issue.type}")
            print(f"  Severity: {issue.severity}")
            print(f"  Description: {issue.description}")
            print(f"  Field: {issue.field_affected}")
            print(f"  Suggestion: {issue.suggestion}")

    except Exception as e:
        print(f"Error during corrected extraction validation: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test validation of corrected evaluation results
    print("2. Testing Corrected Evaluation Validation:")
    print("-" * 50)
    try:
        corrected_evaluation_validation = validate_evaluation_results(
            correction_result["results"]["evaluation_results"]
        )
        print(f"Validation passed: {corrected_evaluation_validation.validation_passed}")
        print(f"Summary: {corrected_evaluation_validation.summary}")
        print(f"Issues found: {len(corrected_evaluation_validation.issues_found)}")

        for i, issue in enumerate(corrected_evaluation_validation.issues_found, 1):
            print(f"\nIssue {i}:")
            print(f"  Type: {issue.type}")
            print(f"  Severity: {issue.severity}")
            print(f"  Description: {issue.description}")
            print(f"  Field: {issue.field_affected}")
            print(f"  Suggestion: {issue.suggestion}")

    except Exception as e:
        print(f"Error during corrected evaluation validation: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test with problematic correction (simulating LLM hallucination)
    print("3. Testing Problematic Correction (LLM Hallucination):")
    print("-" * 50)

    problematic_correction = {
        "success": True,
        "results": {
            "extraction_results": {
                "employee_name": "Jonatan Farrell",
                "employer": "Your Company Name",
                "certificate_issue_date": "2024-11-27",
                "positions": [
                    {
                        "title": "Software Developer",
                        "employer": "Your Company Name",
                        "start_date": "2020-01-15",
                        "end_date": "2024-10-31",
                        "duration": None,
                        "responsibilities": "integral part of our team and contributed significantly to the company's growth and success.",
                    },
                    {
                        "title": "Software Developer",
                        "employer": "Your Company Name",
                        "start_date": "2018-01-15",
                        "end_date": "2025-12-31",  # Future date (hallucination)
                        "duration": None,
                        "responsibilities": "Develop and maintain software applications\nWork with cross-functional teams to develop new features.\nTroubleshoot and debug software issues\nProvide training and support to junior developers",
                    },
                ],
                "total_employment_period": None,
                "document_language": "en",
                "confidence_level": "high",
            },
            "evaluation_results": {
                "total_working_hours": 50000,  # Unrealistic hours (hallucination)
                "training_type": "professional",
                "credits_qualified": 35.0,  # Exceeds limit (hallucination)
                "degree_relevance": "high",
                "relevance_explanation": "The candidate worked as a Software Developer...",
                "calculation_breakdown": "50000 hours / 27 hours per ECTS = 1851.85 credits, capped at 35.0 maximum for professional training",
                "summary_justification": "The role of Software Developer is directly relevant...",
                "conclusion": "Student receives 35.0 ECTS credits as professional training...",
                "confidence_level": "high",
            },
        },
    }

    try:
        problematic_extraction_validation = validate_extraction_results(
            problematic_correction["results"]["extraction_results"]
        )
        print(
            f"Problematic extraction validation passed: {problematic_extraction_validation.validation_passed}"
        )
        print(f"Summary: {problematic_extraction_validation.summary}")
        print(f"Issues found: {len(problematic_extraction_validation.issues_found)}")

        problematic_evaluation_validation = validate_evaluation_results(
            problematic_correction["results"]["evaluation_results"]
        )
        print(
            f"Problematic evaluation validation passed: {problematic_evaluation_validation.validation_passed}"
        )
        print(f"Summary: {problematic_evaluation_validation.summary}")
        print(f"Issues found: {len(problematic_evaluation_validation.issues_found)}")

    except Exception as e:
        print(f"Error during problematic correction validation: {e}")


if __name__ == "__main__":
    test_correction_validation()
