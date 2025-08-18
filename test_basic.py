#!/usr/bin/env python3
"""Simple test script to verify basic functionality."""

try:
    print("Testing basic functionality...")

    # Test 1: Import models
    print("1. Testing model imports...")
    from src.llm.models import EvaluationResults, ExtractionResults, Position

    print("   ✓ Models imported successfully")

    # Test 2: Create a simple Position
    print("2. Testing Position model...")
    position = Position(
        title="Software Developer",
        employer="Tech Corp",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    print(f"   ✓ Position created: {position.title} at {position.employer}")

    # Test 3: Create ExtractionResults
    print("3. Testing ExtractionResults model...")
    extraction = ExtractionResults(
        employee_name="John Doe", positions=[position], document_language="en"
    )
    print(f"   ✓ Extraction created: {extraction.employee_name}")

    # Test 4: Create EvaluationResults
    print("4. Testing EvaluationResults model...")
    evaluation = EvaluationResults(
        requested_training_type="professional",
        credits_qualified=20.0,
        degree_relevance="high",
        relevance_explanation="Work directly related to degree field",
        calculation_breakdown="20 credits for professional training",
        summary_justification="High relevance work experience",
        decision="ACCEPTED",
        justification="Work directly related to degree field",
    )
    print(f"   ✓ Evaluation created: {evaluation.decision}")

    print("\n🎉 All basic functionality tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
