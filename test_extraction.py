#!/usr/bin/env python3
"""
Simple Information Extraction Test Script
Test the information extraction system with OCR output files.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add the app directory to the Python path and import
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    from app.information_extraction import InformationExtractor
except ImportError:
    logger.error("Failed to import InformationExtractor")
    sys.exit(1)


def list_output_files():
    """List all available output files from OCR processing."""
    output_dir = Path("output")
    files = []

    if output_dir.exists():
        for file_path in output_dir.glob("*.txt"):
            files.append(str(file_path))

    return sorted(files)


def read_text_file(file_path: str) -> str:
    """Read text content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def main():
    """Main test function."""
    print("🔍 Information Extraction Test Script")
    print("=" * 50)

    # List available output files
    output_files = list_output_files()

    if not output_files:
        print("❌ No output files found in 'output' directory")
        print("💡 First run the OCR test script to generate output files")
        return

    print(f"📁 Found {len(output_files)} output files:")
    for i, file_path in enumerate(output_files, 1):
        print(f"   {i}. {file_path}")

    print("\n" + "=" * 50)

    # Get user selection
    while True:
        try:
            choice = input("Enter your choice (number) or 'q' to quit: ").strip()

            if choice.lower() == "q":
                print("👋 Goodbye!")
                return

            choice_num = int(choice)
            if 1 <= choice_num <= len(output_files):
                selected_file = output_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(output_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

    print(f"\n{'=' * 60}")
    print(f"Testing: {selected_file}")
    print(f"{'=' * 60}")

    # Read the text content
    print("📄 Reading text content...")
    text_content = read_text_file(selected_file)

    if not text_content:
        print("❌ No text content found in file")
        return

    print(f"📊 Text length: {len(text_content)} characters")
    print(f"📝 First 200 characters: {text_content[:200]}...")

    # Initialize information extractor
    print("\n🔄 Initializing Information Extractor...")
    try:
        extractor = InformationExtractor()
    except Exception as e:
        print(f"❌ Failed to initialize Information Extractor: {e}")
        return

    # Extract information
    print("🔍 Extracting information...")
    try:
        result = extractor.extract_information(text_content)

        print("\n✅ Information extraction completed!")
        print("📊 Results:")
        print(f"   Success: {result.success}")
        print(f"   Overall Confidence: {result.overall_confidence:.2f}")
        print(f"   Processing Time: {result.processing_time:.2f} seconds")
        print(f"   Engine: {result.engine}")

        if result.errors:
            print(f"   Errors: {result.errors}")

        print("\n📋 Extracted Information:")
        print("=" * 40)

        data = result.extracted_data
        if data.document_type:
            print(f"📄 Document Type: {data.document_type}")
        if data.language:
            print(f"🌍 Language: {data.language}")
        if data.employee_name:
            print(f"👤 Employee: {data.employee_name}")
        if data.position:
            print(f"💼 Position: {data.position}")
        if data.employer:
            print(f"🏢 Employer: {data.employer}")
        if data.start_date:
            print(f"📅 Start Date: {data.start_date}")
        if data.end_date:
            print(f"📅 End Date: {data.end_date}")
        if data.work_period:
            print(f"⏱️  Work Period: {data.work_period}")
        if data.description:
            print(f"📝 Description: {data.description[:100]}...")

        print("\n🎯 Confidence Scores:")
        if data.confidence_scores:
            for field, score in data.confidence_scores.items():
                print(f"   {field}: {score:.2f}")

        print("=" * 40)

        # Save extraction results
        if result.success:
            output_filename = f"extraction_result_{Path(selected_file).stem}.json"
            output_path = Path("output") / output_filename

            # Ensure output directory exists
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"💾 Extraction results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during information extraction: {e}")


if __name__ == "__main__":
    main()
