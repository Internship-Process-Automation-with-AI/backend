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

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.llm.cert_extractor import LLMOrchestrator
    from src.llm.degree_evaluator import DegreeEvaluator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running from the backend directory")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)


def clean_ocr_text_conservative(text: str) -> str:
    """Clean OCR text conservatively for LLM processing."""
    if not text:
        return ""

    # Basic cleaning - preserve most content
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and len(line) > 2:  # Keep lines with meaningful content
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def list_ocr_outputs():
    """List all available OCR output files."""
    output_dir = Path("OCRoutput")
    files = []

    if output_dir.exists():
        for file_path in output_dir.glob("*.txt"):
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


def save_orchestrator_results(results: Dict[str, Any], input_file: str) -> str:
    """Save orchestrator results to JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = "LLMoutput"
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_orchestrator_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return ""


def display_orchestrator_results(results: dict, filename: str):
    """Display orchestrator results in a formatted way."""
    print(f"\n{'=' * 80}")
    print("🤖 LLM ORCHESTRATOR RESULTS")
    print(f"📄 Document: {filename}")
    print(f"{'=' * 80}")

    # Overall status
    success = results.get("success", False)
    if success:
        print("✅ Processing completed successfully")
    else:
        print("❌ Processing failed")
        error = results.get("error", "Unknown error")
        print(f"   Error: {error}")
        return

    # Processing time
    processing_time = results.get("processing_time", 0)
    print(f"⏱️  Total processing time: {processing_time:.2f} seconds")

    # Model used
    model_used = results.get("model_used", "Unknown")
    print(f"🤖 Model used: {model_used}")

    # Student degree
    student_degree = results.get("student_degree", "Not specified")
    print(f"🎓 Student Degree: {student_degree}")

    # Stages completed
    stages = results.get("stages_completed", {})
    print("📊 Stages completed:")
    print(f"   • Extraction: {'✅' if stages.get('extraction') else '❌'}")
    print(f"   • Evaluation: {'✅' if stages.get('evaluation') else '❌'}")

    # Extraction results
    extraction_results = results.get("extraction_results", {})
    if extraction_results.get("success"):
        extraction_data = extraction_results.get("results", {})
        print("\n📋 EXTRACTION RESULTS:")
        print(f"   • Employee Name: {extraction_data.get('employee_name', 'N/A')}")
        print(f"   • Job Title: {extraction_data.get('job_title', 'N/A')}")
        print(f"   • Company: {extraction_data.get('company_name', 'N/A')}")
        print(f"   • Start Date: {extraction_data.get('start_date', 'N/A')}")
        print(f"   • End Date: {extraction_data.get('end_date', 'N/A')}")
        print(
            f"   • Employment Period: {extraction_data.get('employment_period', 'N/A')}"
        )
        print(
            f"   • Document Language: {extraction_data.get('document_language', 'N/A')}"
        )

    # Evaluation results
    evaluation_results = results.get("evaluation_results", {})
    if evaluation_results.get("success"):
        evaluation_data = evaluation_results.get("results", {})
        print("\n🎓 EVALUATION RESULTS:")
        print(f"   • Total Hours: {evaluation_data.get('total_working_hours', 'N/A')}")
        print(f"   • Training Type: {evaluation_data.get('training_type', 'N/A')}")
        print(f"   • Credits: {evaluation_data.get('credits_qualified', 'N/A')} ECTS")
        print(
            f"   • Quality Multiplier: {evaluation_data.get('quality_multiplier', 'N/A')}"
        )
        print(
            f"   • Degree Relevance: {evaluation_data.get('degree_relevance', 'N/A')}"
        )
        print(f"   • Confidence: {evaluation_data.get('confidence_level', 'N/A')}")

        # Degree-specific information
        degree_relevance_level = evaluation_data.get("degree_relevance_level", "N/A")
        calculated_multiplier = evaluation_data.get(
            "calculated_quality_multiplier", "N/A"
        )
        print(f"   • Calculated Relevance Level: {degree_relevance_level}")
        print(f"   • Calculated Quality Multiplier: {calculated_multiplier}")

        print("\n🔧 NATURE OF TASKS:")
        tasks = evaluation_data.get("nature_of_tasks", "Not specified")
        if tasks:
            print(f"   {tasks}")

        print("\n📝 RELEVANCE EXPLANATION:")
        relevance_explanation = evaluation_data.get(
            "relevance_explanation", "No explanation provided"
        )
        if relevance_explanation:
            # Wrap long text
            words = relevance_explanation.split()
            lines = []
            current_line = "   "
            for word in words:
                if len(current_line + word) > 75:
                    lines.append(current_line)
                    current_line = f"   {word} "
                else:
                    current_line += word + " "
            lines.append(current_line)

            for line in lines:
                print(line.strip())

        print("\n📝 JUSTIFICATION:")
        justification = evaluation_data.get(
            "summary_justification", "No justification provided"
        )
        if justification:
            # Wrap long text
            words = justification.split()
            lines = []
            current_line = "   "
            for word in words:
                if len(current_line + word) > 75:
                    lines.append(current_line)
                    current_line = f"   {word} "
                else:
                    current_line += word + " "
            lines.append(current_line)

            for line in lines:
                print(line.strip())

    print("=" * 80)


def select_degree_program() -> str:
    """Let user select a degree program."""
    degree_evaluator = DegreeEvaluator()
    supported_degrees = degree_evaluator.get_supported_degree_programs()

    print("\n🎓 SELECT OAMK DEGREE PROGRAM:")
    print("Available OAMK degree programs:")

    # Group degrees by category for better display
    degree_categories = {
        "Information Technology": ["Information Technology"],
        "Culture": ["Culture"],
        "Natural Resources": ["Natural Resources"],
        "Business Administration": ["Business Administration"],
        "Healthcare": ["Healthcare"],
        "Engineering": ["Engineering"],
        "General Studies": ["General Studies"],
    }

    degree_list = []
    for category, degrees in degree_categories.items():
        for degree in degrees:
            if degree in supported_degrees:
                degree_list.append(degree)

    for i, degree in enumerate(degree_list, 1):
        print(f"   {i}. {degree}")

    print(f"   {len(degree_list) + 1}. Custom degree program")

    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(degree_list) + 1}): ").strip()
            choice_num = int(choice)

            if 1 <= choice_num <= len(degree_list):
                selected_degree = degree_list[choice_num - 1]
                print(f"✅ Selected: {selected_degree}")
                return selected_degree
            elif choice_num == len(degree_list) + 1:
                custom_degree = input("Enter custom degree program name: ").strip()
                if custom_degree:
                    print(f"✅ Selected: {custom_degree}")
                    return custom_degree
                else:
                    print("❌ Please enter a valid degree program name")
            else:
                print(f"❌ Please enter a number between 1 and {len(degree_list) + 1}")
        except ValueError:
            print("❌ Please enter a valid number")


def main():
    """Main function for testing the LLM orchestrator."""
    print("🤖 LLM ORCHESTRATOR TEST")
    print("=" * 50)

    # Initialize orchestrator
    print("🔧 Initializing LLM orchestrator...")
    orchestrator = LLMOrchestrator()

    if not orchestrator.is_available():
        print("❌ LLM orchestrator not available")
        print("   Make sure GEMINI_API_KEY is set in your environment")
        return

    print("✅ LLM orchestrator initialized successfully")

    # Get model stats
    stats = orchestrator.get_stats()
    print(f"📊 Model: {stats.get('model', 'Unknown')}")
    print(
        f"📊 API Status: {'✅ Available' if stats.get('available') else '❌ Unavailable'}"
    )

    # List available OCR outputs
    ocr_files = list_ocr_outputs()

    if not ocr_files:
        print("❌ No OCR output files found in 'OCRoutput' directory")
        print("   Please run test_ocr.py first to generate OCR outputs")
        return

    print(f"\n📄 Found {len(ocr_files)} OCR output files:")
    for i, file_path in enumerate(ocr_files, 1):
        print(f"   {i}. {os.path.basename(file_path)}")

    print("\n" + "=" * 50)

    # Get user selection
    while True:
        try:
            choice = input("Enter your choice (number) or 'q' to quit: ").strip()

            if choice.lower() == "q":
                print("👋 Goodbye!")
                return

            choice_num = int(choice)
            if 1 <= choice_num <= len(ocr_files):
                selected_file = ocr_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(ocr_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

    # Select degree program
    student_degree = select_degree_program()

    print(f"\n{'=' * 50}")
    print(f"Processing: {os.path.basename(selected_file)}")
    print(f"Degree Program: {student_degree}")
    print(f"{'=' * 50}")

    # Load and clean OCR text
    print("📄 Loading OCR text...")
    ocr_text = load_ocr_text(selected_file)

    if not ocr_text:
        print("❌ No text found in OCR output file")
        return

    print("📊 Original text length: {} characters".format(len(ocr_text)))
    print(f"🔍 DEBUG: ocr_text preview: {repr(ocr_text[:100])}")

    # Clean text for LLM
    print("🧹 Cleaning text for LLM processing...")
    cleaned_text = clean_ocr_text_conservative(ocr_text)
    print("📊 Cleaned text length: {} characters".format(len(cleaned_text)))
    print(f"🔍 DEBUG: cleaned_text preview: {repr(cleaned_text[:100])}")

    # Show preview of cleaned text
    print("\n📄 TEXT PREVIEW (first 300 characters):")
    print("-" * 40)
    preview = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
    print(preview)
    print("-" * 40)

    # Confirm with user
    proceed = input("\nProceed with LLM orchestration? (y/n): ").strip().lower()
    if proceed != "y":
        print("❌ Processing cancelled")
        return

    # Process with orchestrator
    print(
        "\n🤖 Starting LLM orchestration with {}...".format(
            stats.get("model", "Gemini")
        )
    )
    print("   Stage 1: Information Extraction")
    print("   Stage 2: Academic Evaluation (Degree-specific)")

    # Debug: Check what we're about to send
    print(f"\n🔍 DEBUG: cleaned_text type: {type(cleaned_text)}")
    print(f"🔍 DEBUG: cleaned_text length: {len(cleaned_text)}")
    print(f"🔍 DEBUG: cleaned_text preview: {repr(cleaned_text[:100])}")

    # Safety check: Make sure we're not passing corrupted text
    if not cleaned_text or len(cleaned_text) < 10:
        print("❌ ERROR: cleaned_text is too short or empty!")
        return

    if cleaned_text.startswith('"') or cleaned_text.startswith("{"):
        print("❌ ERROR: cleaned_text appears to be JSON, not document text!")
        print(f"❌ ERROR: cleaned_text = {repr(cleaned_text)}")
        return

    try:
        results = orchestrator.process_work_certificate(cleaned_text, student_degree)

        # Display results
        display_orchestrator_results(results, os.path.basename(selected_file))

        # Save results
        if results.get("success", False):
            output_path = save_orchestrator_results(results, selected_file)
            print(f"💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during LLM orchestration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
