#!/usr/bin/env python3
"""
Simple OCR Test Script for Early Phase Testing
This script tests the OCR service with your sample files locally.
No API complexity - just pure document processing.
"""

import os
import sys
import logging
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from ocr_service import OCRService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def list_sample_files():
    """List all available sample files."""
    sample_dir = Path("file samples")
    files = []
    
    if sample_dir.exists():
        for file_path in sample_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                files.append(str(file_path))
    
    return sorted(files)

def save_results_to_file(text: str, filename: str, output_dir: str = "output"):
    """Save OCR results to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{base_name}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save text to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_path

def main():
    """Main test function."""
    print("🔍 OCR Service Test Script")
    print("=" * 50)
    
    # List available sample files
    sample_files = list_sample_files()
    
    if not sample_files:
        print("❌ No sample files found in 'file samples' directory")
        return
    
    print(f"📁 Found {len(sample_files)} sample files:")
    for i, file_path in enumerate(sample_files, 1):
        print(f"   {i}. {file_path}")
    
    print("\n" + "=" * 50)
    
    # Get user selection
    while True:
        try:
            choice = input("Enter your choice (number) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(sample_files):
                selected_file = sample_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(sample_files)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"\n{'=' * 60}")
    print(f"Testing: {selected_file}")
    print(f"{'=' * 60}")
    
    # Initialize OCR service
    print("🔄 Initializing OCR service...")
    try:
        ocr_service = OCRService()
    except Exception as e:
        print(f"❌ Failed to initialize OCR service: {e}")
        return
    
    # Process the file
    print("📄 Processing file...")
    try:
        result = ocr_service.extract_text_from_file(selected_file)
        
        print("\n✅ Processing completed!")
        print("📊 Results:")
        print(f"   Engine: {result.engine}")
        print(f"   Confidence: {result.confidence:.1f}%")
        print(f"   Time: {result.processing_time:.2f} seconds")
        print(f"   Success: {result.success}")
        print(f"   Text length: {len(result.text)} characters")
        
        print("\n📄 Extracted Text:")
        print("=" * 40)
        if result.text:
            print(result.text)
        else:
            print("(No text extracted)")
        print("=" * 40)
        
        # Save results to output directory
        if result.text:
            output_path = save_results_to_file(result.text, selected_file)
            print(f"💾 Results saved to: {output_path}")
        else:
            print("⚠️  No text to save")
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")

if __name__ == "__main__":
    main() 