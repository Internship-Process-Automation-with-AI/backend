#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the LLM orchestrator with degree-specific evaluation.
Allows testing of work certificate processing with different degree programs.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add the current directory (src) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from llm.cert_extractor import LLMOrchestrator
    from llm.degree_evaluator import DegreeEvaluator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Error type: {type(e)}")
    print(f"   Error details: {e.__class__.__name__}: {e}")
    import traceback

    traceback.print_exc()
    print("   Make sure you're running from the backend directory")
    sys.exit(1)


def clean_ocr_text(text: str) -> str:
    """Clean OCR text for LLM processing."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = [
        line.strip() for line in lines if line.strip() and len(line.strip()) > 2
    ]
    return "\n".join(cleaned_lines)


def list_ocr_outputs():
    """List all available OCR output files from the new organized structure."""
    # Go up one directory to find outputs folder
    outputs_dir = Path(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    files = []

    if outputs_dir.exists():
        # Look for OCR files in the new organized structure
        for sample_dir in outputs_dir.iterdir():
            if sample_dir.is_dir():
                # Look for OCRoutput_*.txt files in each sample directory
                for file_path in sample_dir.glob("OCRoutput_*.txt"):
                    if file_path.is_file():
                        files.append(str(file_path))

    # If no files found in new structure, try old structure for backward compatibility
    if not files:
        old_output_dir = Path(
            os.path.join(os.path.dirname(__file__), "..", "OCRoutput")
        )
        if old_output_dir.exists():
            for file_path in old_output_dir.glob("*.txt"):
                if file_path.is_file():
                    files.append(str(file_path))

    return sorted(files)


def load_ocr_text(file_path: str) -> str:
    """Load OCR text from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load OCR text from {file_path}: {e}")
        return ""


def save_results(results: Dict[str, Any], input_file: str) -> str:
    """Save orchestrator results to JSON file in the new organized structure."""
    try:
        # Extract base name from the OCR file path
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Remove "OCRoutput_" prefix if present to get the original sample name
        if base_name.startswith("OCRoutput_"):
            base_name = base_name[10:]  # Remove "OCRoutput_" prefix

        # Use the new organized output structure
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", base_name)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"LLMoutput_{base_name}_orchestrator_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return ""


def display_results(results: dict, filename: str):
    """Display orchestrator results in a formatted way."""
    print(f"\n{'=' * 80}")
    print("🤖 LLM ORCHESTRATOR RESULTS")
    print(f"📄 Document: {filename}")
    print(f"{'=' * 80}")

    # Overall status
    if not results.get("success", False):
        print("❌ Processing failed")
        print(f"   Error: {results.get('error', 'Unknown error')}")
        return

    print("✅ Processing completed successfully")
    print(f"⏱️  Processing time: {results.get('processing_time', 0):.2f} seconds")
    print(f"🤖 Model: {results.get('model_used', 'Unknown')}")
    print(f"🎓 Degree: {results.get('student_degree', 'Not specified')}")

    # Show stages completed
    stages_completed = results.get("stages_completed", {})
    if stages_completed:
        print("\n📊 STAGES COMPLETED:")
        stage_names = {
            "extraction": "📋 Extraction",
            "evaluation": "🎓 Evaluation",
            "validation": "🔍 Validation",
            "correction": "🔧 Correction",
        }
        for stage, completed in stages_completed.items():
            status = "✅" if completed else "❌"
            name = stage_names.get(stage, stage.title())
            print(f"   {status} {name}")

    # Extraction results
    extraction_results = results.get("extraction_results", {})
    if extraction_results.get("success"):
        data = extraction_results.get("results", {})
        print("\n📋 EXTRACTION:")
        print(f"   • Employee: {data.get('employee_name', 'N/A')}")

        # Handle the new nested positions structure
        positions = data.get("positions", [])
        if positions:
            position = positions[0]  # Show first position
            print(f"   • Job Title: {position.get('title', 'N/A')}")
            print(f"   • Company: {position.get('employer', 'N/A')}")
            print(
                f"   • Period: {position.get('start_date', 'N/A')} - {position.get('end_date', 'N/A')}"
            )
        else:
            # Fallback to old flat structure if positions not found
            print(f"   • Job Title: {data.get('job_title', 'N/A')}")
            print(f"   • Company: {data.get('company_name', 'N/A')}")
            print(
                f"   • Period: {data.get('start_date', 'N/A')} - {data.get('end_date', 'N/A')}"
            )

    # Evaluation results
    evaluation_results = results.get("evaluation_results", {})
    if evaluation_results.get("success"):
        data = evaluation_results.get("results", {})
        print("\n🎓 EVALUATION:")
        print(f"   • Hours: {data.get('total_working_hours', 'N/A')}")
        print(f"   • Type: {data.get('training_type', 'N/A')}")
        print(f"   • Credits: {data.get('credits_qualified', 'N/A')} ECTS")
        print(f"   • Relevance: {data.get('degree_relevance', 'N/A')}")

        # Show justification
        justification = data.get("summary_justification", "")
        if justification:
            print("\n📝 Justification:")
            print(f"   {justification}")

        # Show conclusion
        conclusion = data.get("conclusion", "")
        if conclusion:
            print("\n🎯 CONCLUSION:")
            print(f"   {conclusion}")

    # Validation results
    validation_results = results.get("validation_results", {})
    if validation_results.get("success"):
        data = validation_results.get("results", {})
        validation_passed = data.get("validation_passed", False)

        if not validation_passed:
            print("\n🔍 VALIDATION: ❌ (Correction needed)")
        else:
            print("\n🔍 VALIDATION: ✅ (Passed)")

        # Only show issues if validation failed
        if not validation_passed:
            issues = data.get("issues_found", [])
            if issues:
                print(f"   Issues: {len(issues)} found")

    # Correction results (if any)
    correction_results = results.get("correction_results", {})
    if correction_results and correction_results.get("success"):
        data = correction_results.get("results", {})
        print("\n🔧 CORRECTIONS: ✅ Applied")

        # Show corrected evaluation results
        corrected_evaluation = data.get("evaluation_results", {})
        if corrected_evaluation:
            print("\n✅ FINAL RESULTS (Corrected):")
            print(
                f"   • Hours: {corrected_evaluation.get('total_working_hours', 'N/A')}"
            )
            print(f"   • Type: {corrected_evaluation.get('training_type', 'N/A')}")
            print(
                f"   • Credits: {corrected_evaluation.get('credits_qualified', 'N/A')} ECTS"
            )
            print(
                f"   • Degree Relevance: {corrected_evaluation.get('degree_relevance', 'N/A')}"
            )

            # Show corrected justification
            corrected_justification = corrected_evaluation.get(
                "summary_justification", ""
            )
            if corrected_justification:
                print("\n📝 Justification:")
                print(f"   {corrected_justification}")

            # Show corrected conclusion
            corrected_conclusion = corrected_evaluation.get("conclusion", "")
            if corrected_conclusion:
                print("\n🎯 Conclusion:")
                print(f"   {corrected_conclusion}")

    print("=" * 80)


def select_degree_program() -> str:
    """Let user select a degree program."""
    degree_evaluator = DegreeEvaluator()
    supported_degrees = degree_evaluator.get_supported_degree_programs()

    print("\n🎓 SELECT DEGREE PROGRAM:")
    for i, degree in enumerate(supported_degrees, 1):
        print(f"   {i}. {degree}")
    print(f"   {len(supported_degrees) + 1}. Custom degree")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(supported_degrees) + 1}): ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(supported_degrees):
                selected_degree = supported_degrees[choice_num - 1]
                print(f"✅ Selected: {selected_degree}")
                return selected_degree
            elif choice_num == len(supported_degrees) + 1:
                custom_degree = input("Enter custom degree: ").strip()
                if custom_degree:
                    return custom_degree
                else:
                    print("❌ Please enter a valid degree name")
            else:
                print(f"❌ Please enter 1-{len(supported_degrees) + 1}")
        except ValueError:
            print("❌ Please enter a valid number")


def main():
    """Main function for testing the LLM orchestrator."""
    print("🤖 LLM ORCHESTRATOR TEST")
    print("=" * 50)

    # Initialize orchestrator
    orchestrator = LLMOrchestrator()
    if not orchestrator.is_available():
        print("❌ LLM orchestrator not available")
        print("   Make sure GEMINI_API_KEY is set in your environment")
        return

    print("✅ LLM orchestrator initialized")

    # List available OCR outputs
    ocr_files = list_ocr_outputs()
    if not ocr_files:
        print("❌ No OCR output files found")
        print(
            "   Please run test_ocr.py or mainpipeline.py first to generate OCR outputs"
        )
        return

    print(f"\n📄 Found {len(ocr_files)} OCR files:")
    for i, file_path in enumerate(ocr_files, 1):
        print(f"   {i}. {os.path.basename(file_path)}")

    # Get user selection
    while True:
        try:
            choice = input("\nEnter choice (number) or 'q' to quit: ").strip()
            if choice.lower() == "q":
                print("👋 Goodbye!")
                return

            choice_num = int(choice)
            if 1 <= choice_num <= len(ocr_files):
                selected_file = ocr_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter 1-{len(ocr_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

    # Select degree program
    student_degree = select_degree_program()

    print(f"\n{'=' * 50}")
    print(f"Processing: {os.path.basename(selected_file)}")
    print(f"Degree: {student_degree}")
    print(f"{'=' * 50}")

    # Load and clean OCR text
    ocr_text = load_ocr_text(selected_file)
    if not ocr_text:
        print("❌ No text found in OCR output file")
        return

    cleaned_text = clean_ocr_text(ocr_text)
    print(f"📄 Text length: {len(cleaned_text)} characters")

    # Show preview
    print("\n📄 TEXT PREVIEW:")
    print("-" * 40)
    preview = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
    print(preview)
    print("-" * 40)

    # Process with orchestrator
    print("\n🤖 Processing...")
    try:
        results = orchestrator.process_work_certificate(cleaned_text, student_degree)

        # Display results
        display_results(results, os.path.basename(selected_file))

        # Save results
        if results.get("success", False):
            output_path = save_results(results, selected_file)
            print(f"💾 Results saved to: {output_path}")

            # Show the organized output structure
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            if base_name.startswith("OCRoutput_"):
                base_name = base_name[10:]  # Remove "OCRoutput_" prefix
            output_base_dir = os.path.join(
                os.path.dirname(__file__), "..", "outputs", base_name
            )
            print(f"\n📁 Output organized in: {output_base_dir}")
            print(f"   ├── OCRoutput_{base_name}.txt     (OCR text)")
            print(
                f"   └── LLMoutput_{base_name}_orchestrator_*.json     (LLM evaluation results)"
            )

    except Exception as e:
        print(f"❌ Error during processing: {e}")


if __name__ == "__main__":
    main()
