#!/usr/bin/env python3
"""
Test script for the LLM Orchestrator
Tests the two-stage process: extraction followed by evaluation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

try:
    from llm.cert_extractor import LLMOrchestrator
    from utils.finnish_ocr_corrector import clean_ocr_text_conservative
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Python path: {sys.path}")
    sys.exit(1)


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


def save_orchestrator_results(
    results: dict, filename: str, output_dir: str = "LLMoutput"
):
    """Save orchestrator results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_orchestrator_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save results to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return output_path


def display_orchestrator_results(results: dict, filename: str):
    """Display orchestrator results in a formatted way."""
    print("=" * 80)
    print(f"LLM ORCHESTRATOR RESULTS: {filename}")
    print("=" * 80)

    if not results.get("success", False):
        print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        return

    print(f"⏱️  Total processing time: {results.get('processing_time', 0):.2f} seconds")
    print(f"🤖 Model: {results.get('model_used', 'Unknown')}")

    # Display stage completion
    stages = results.get("stages_completed", {})
    print("📊 Stages completed:")
    print("   • Extraction: {}".format("✅" if stages.get("extraction") else "❌"))
    print("   • Evaluation: {}".format("✅" if stages.get("evaluation") else "❌"))

    # Display extraction results
    extraction_results = results.get("extraction_results", {})
    if extraction_results.get("success"):
        extraction_data = extraction_results.get("results", {})
        print("\n📋 EXTRACTION RESULTS:")
        print(f"   • Employee: {extraction_data.get('employee_name', 'N/A')}")
        print(f"   • Position: {extraction_data.get('position', 'N/A')}")
        print(f"   • Employer: {extraction_data.get('employer', 'N/A')}")
        print(f"   • Start Date: {extraction_data.get('start_date', 'N/A')}")
        print(f"   • End Date: {extraction_data.get('end_date', 'N/A')}")
        print(f"   • Language: {extraction_data.get('document_language', 'N/A')}")
        print(f"   • Confidence: {extraction_data.get('confidence_level', 'N/A')}")

    # Display evaluation results
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
        print(f"   • Confidence: {evaluation_data.get('confidence_level', 'N/A')}")

        print("\n🔧 NATURE OF TASKS:")
        tasks = evaluation_data.get("nature_of_tasks", "Not specified")
        if tasks:
            print(f"   {tasks}")

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


def main():
    """Main test function."""
    print("🤖 LLM Orchestrator Test")
    print("=" * 50)

    # Initialize orchestrator
    print("🔄 Initializing LLM orchestrator...")
    try:
        orchestrator = LLMOrchestrator()
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return

    if not orchestrator.is_available():
        print("❌ LLM orchestrator not available. Please check your Gemini API key.")
        print("   Set GEMINI_API_KEY in your .env file or environment variables.")
        return

    # Get orchestrator stats
    stats = orchestrator.get_stats()
    prompt_info = orchestrator.get_prompt_info()
    print("✅ LLM orchestrator initialized successfully")
    print(f"🤖 Using model: {stats.get('model', 'Unknown')}")
    print(f"📊 Stages: {', '.join(stats.get('stages', []))}")
    print(
        f"📝 Prompt info: {prompt_info.get('total_prompt_length', 0)} total characters"
    )

    # List available OCR outputs
    ocr_files = list_ocr_outputs()

    if not ocr_files:
        print("❌ No OCR output files found in 'OCRoutput' directory")
        print("   Please run test_ocr.py first to generate OCR outputs")
        return

    print(f"\n📁 Found {len(ocr_files)} OCR output files:")
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

    print(f"\n{'=' * 50}")
    print(f"Processing: {os.path.basename(selected_file)}")
    print(f"{'=' * 50}")

    # Load and clean OCR text
    print("📄 Loading OCR text...")
    ocr_text = load_ocr_text(selected_file)

    if not ocr_text:
        print("❌ No text found in OCR output file")
        return

    print("📊 Original text length: {} characters".format(len(ocr_text)))

    # Clean text for LLM
    print("🧹 Cleaning text for LLM processing...")
    cleaned_text = clean_ocr_text_conservative(ocr_text)
    print("📊 Cleaned text length: {} characters".format(len(cleaned_text)))

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
    print("   Stage 2: Academic Evaluation")

    try:
        results = orchestrator.process_work_certificate(cleaned_text)

        # Display results
        display_orchestrator_results(results, os.path.basename(selected_file))

        # Save results
        if results.get("success", False):
            output_path = save_orchestrator_results(results, selected_file)
            print(f"💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during LLM orchestration: {e}")


if __name__ == "__main__":
    main()
