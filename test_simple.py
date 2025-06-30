#!/usr/bin/env python3
"""
Simple test to isolate the issue with the LLM orchestrator.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "src"))

try:
    from llm.cert_extractor import LLMOrchestrator
    from utils.finnish_ocr_corrector import clean_ocr_text_conservative
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


def main():
    print("🧪 Simple Test")
    print("=" * 30)

    # Test 1: Text cleaning
    print("1. Testing text cleaning...")
    test_text = "This is a test document for Louis Peterson at Test Company."
    print(f"   Original: {repr(test_text)}")

    try:
        cleaned_text = clean_ocr_text_conservative(test_text)
        print(f"   Cleaned: {repr(cleaned_text)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Test 2: LLM Orchestrator initialization
    print("\n2. Testing LLM Orchestrator initialization...")
    try:
        orchestrator = LLMOrchestrator()
        print(f"   ✅ Initialized: {orchestrator.is_available()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Test 3: Process a simple document
    print("\n3. Testing document processing...")
    simple_doc = """October 7, 2012
To: Ms. Dominick
Dear Ms. Dominick,
We are in receipt of your request for employment verification for Louis Peterson. 
The internship will commence on May 28, 2013 and end on July 20, 2013.
Sincerely,
Jack Phillips
Global Partners Internship Coordinator"""

    print(f"   Document: {repr(simple_doc[:100])}...")

    try:
        results = orchestrator.process_work_certificate(simple_doc)
        print(f"   ✅ Results: {results.get('success', False)}")
        if results.get("success"):
            print(f"   📊 Processing time: {results.get('processing_time', 0):.2f}s")
        else:
            print(f"   ❌ Error: {results.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
