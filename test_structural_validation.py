"""
Test script to verify structural validation with problematic data.
"""

import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.llm.models import validate_evaluation_results, validate_extraction_results


def test_problematic_data():
    """Test structural validation with the problematic data from the JSON file."""

    # Problematic extraction data from the JSON file
    problematic_extraction = {
        "employee_name": "Jonatan Farrell",
        "employer": "Your Company Name",
        "certificate_issue_date": "2051-11-27",  # Future date
        "positions": [
            {
                "title": "Software Developer",
                "employer": "Your Company Name",
                "start_date": "2050-01-15",  # Future date
                "end_date": "2051-11-27",  # Future date
                "duration": None,
                "responsibilities": "integral part of our team and contributed significantly to the company's growth and success.",
            },
            {
                "title": "Software Developer",
                "employer": None,  # Missing employer
                "start_date": "2018-01-15",
                "end_date": "2024-10-31",
                "duration": None,
                "responsibilities": "Develop and maintain software applications\nWork with cross-functional teams to develop new features.\nTroubleshoot and debug software issues\nProvide training and support to junior developers",
            },
        ],
        "total_employment_period": None,
        "document_language": "en",
        "confidence_level": "high(>75%)",
    }

    # Problematic evaluation data
    problematic_evaluation = {
        "total_working_hours": 24992,
        "training_type": "professional",
        "credits_qualified": 30.0,
        "degree_relevance": "high",
        "relevance_explanation": "The candidate worked as a Software Developer, which directly aligns with the BEng in Information Technology. Responsibilities include software development, feature development with cross-functional teams, troubleshooting, debugging, and providing training, all of which are highly relevant to the degree.",
        "calculation_breakdown": "24992 hours / 27 hours per ECTS = 925.0 credits, capped at 30.0 maximum for professional training",
        "summary_justification": "The role of Software Developer is directly relevant to a Bachelor of Engineering in Information Technology. The responsibilities listed demonstrate development skills and the ability to work within a team. This experience significantly contributes toward the required 30 ECTS credits of practical training, with at least 20 credits needing to be degree-related.",
        "conclusion": "Student receives 30.0 ECTS credits as professional training, fulfilling the total practical training requirement for the degree.",
        "confidence_level": "high",
    }

    print("=== Testing Structural Validation ===\n")

    # Test extraction validation
    print("1. Testing Extraction Validation:")
    print("-" * 50)
    try:
        extraction_validation = validate_extraction_results(problematic_extraction)
        print(f"Validation passed: {extraction_validation.validation_passed}")
        print(f"Summary: {extraction_validation.summary}")
        print(f"Issues found: {len(extraction_validation.issues_found)}")

        for i, issue in enumerate(extraction_validation.issues_found, 1):
            print(f"\nIssue {i}:")
            print(f"  Type: {issue.type}")
            print(f"  Severity: {issue.severity}")
            print(f"  Description: {issue.description}")
            print(f"  Field: {issue.field_affected}")
            print(f"  Suggestion: {issue.suggestion}")

    except Exception as e:
        print(f"Error during extraction validation: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test evaluation validation
    print("2. Testing Evaluation Validation:")
    print("-" * 50)
    try:
        evaluation_validation = validate_evaluation_results(problematic_evaluation)
        print(f"Validation passed: {evaluation_validation.validation_passed}")
        print(f"Summary: {evaluation_validation.summary}")
        print(f"Issues found: {len(evaluation_validation.issues_found)}")

        for i, issue in enumerate(evaluation_validation.issues_found, 1):
            print(f"\nIssue {i}:")
            print(f"  Type: {issue.type}")
            print(f"  Severity: {issue.severity}")
            print(f"  Description: {issue.description}")
            print(f"  Field: {issue.field_affected}")
            print(f"  Suggestion: {issue.suggestion}")

    except Exception as e:
        print(f"Error during evaluation validation: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test with corrected data
    print("3. Testing with Corrected Data:")
    print("-" * 50)

    corrected_extraction = {
        "employee_name": "Jonatan Farrell",
        "employer": "Your Company Name",
        "certificate_issue_date": "2024-11-27",  # Corrected date
        "positions": [
            {
                "title": "Software Developer",
                "employer": "Your Company Name",
                "start_date": "2020-01-15",  # Corrected date
                "end_date": "2024-10-31",  # Corrected date
                "duration": None,
                "responsibilities": "integral part of our team and contributed significantly to the company's growth and success.",
            },
            {
                "title": "Software Developer",
                "employer": "Your Company Name",  # Added employer
                "start_date": "2018-01-15",
                "end_date": "2019-12-31",  # Corrected to not overlap
                "duration": None,
                "responsibilities": "Develop and maintain software applications\nWork with cross-functional teams to develop new features.\nTroubleshoot and debug software issues\nProvide training and support to junior developers",
            },
        ],
        "total_employment_period": None,
        "document_language": "en",
        "confidence_level": "high(>75%)",
    }

    try:
        corrected_validation = validate_extraction_results(corrected_extraction)
        print(
            f"Corrected data validation passed: {corrected_validation.validation_passed}"
        )
        print(f"Summary: {corrected_validation.summary}")
        print(f"Issues found: {len(corrected_validation.issues_found)}")

    except Exception as e:
        print(f"Error during corrected data validation: {e}")


if __name__ == "__main__":
    test_problematic_data()
