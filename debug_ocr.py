import os
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from ocr_service import OCRService
from utils.pdf_converter import PDFConverter
from utils.image_preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tesseract_directly():
    """Test Tesseract OCR directly with a simple image."""
    print("🧪 Testing Tesseract directly...")
    
    # Create a simple test image with text
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255  # White background
    cv2.putText(img, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Test Tesseract
    try:
        text = pytesseract.image_to_string(img)
        print(f"✅ Tesseract test result: '{text.strip()}'")
        return True
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        return False

def test_pdf_conversion(file_path):
    """Test PDF to image conversion."""
    print(f"\n🔄 Testing PDF conversion: {file_path}")
    
    try:
        pdf_converter = PDFConverter()
        
        # Read PDF
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Convert to images
        images = pdf_converter.convert_pdf_to_images(pdf_bytes, dpi=300)
        
        print(f"✅ Converted {len(images)} pages")
        
        # Save first image for inspection
        if images:
            first_image = images[0]
            debug_image_path = "debug_converted_image.png"
            cv2.imwrite(debug_image_path, cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved debug image: {debug_image_path}")
            
            # Test OCR on converted image
            print("🔍 Testing OCR on converted image...")
            text = pytesseract.image_to_string(first_image)
            print(f"📄 Raw OCR result: '{text.strip()}'")
            
            return first_image
        
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        return None

def test_preprocessing(image):
    """Test image preprocessing."""
    print("\n🔄 Testing image preprocessing...")
    
    try:
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.preprocess_image(image)
        
        # Save processed image
        debug_processed_path = "debug_processed_image.png"
        cv2.imwrite(debug_processed_path, processed_image)
        print(f"💾 Saved processed image: {debug_processed_path}")
        
        # Test OCR on processed image
        print("🔍 Testing OCR on processed image...")
        text = pytesseract.image_to_string(processed_image)
        print(f"📄 Processed OCR result: '{text.strip()}'")
        
        return processed_image
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None

def test_full_pipeline(file_path):
    """Test the full OCR pipeline."""
    print(f"\n🚀 Testing full pipeline: {file_path}")
    
    try:
        ocr_service = OCRService()
        result = ocr_service.extract_text_from_file(file_path)
        
        print(f"📊 Full pipeline result:")
        print(f"   Engine: {result.engine}")
        print(f"   Confidence: {result.confidence:.1f}%")
        print(f"   Success: {result.success}")
        print(f"   Text: '{result.text.strip()}'")
        
        return result
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        return None

def main():
    """Main debug function."""
    print("🔍 OCR Debug Script")
    print("=" * 50)
    
    # Test 1: Direct Tesseract
    if not test_tesseract_directly():
        print("❌ Tesseract is not working properly. Please check installation.")
        return
    
    # Test 2: PDF conversion and OCR
    sample_file = "file samples/scanned/scanned/employment-letter-07.pdf"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file not found: {sample_file}")
        return
    
    # Test PDF conversion
    converted_image = test_pdf_conversion(sample_file)
    
    if converted_image is not None:
        # Test preprocessing
        processed_image = test_preprocessing(converted_image)
        
        # Test full pipeline
        full_result = test_full_pipeline(sample_file)
    
    print("\n" + "=" * 50)
    print("🔍 Debug Summary:")
    print("Check the generated debug images:")
    print("- debug_converted_image.png (raw PDF conversion)")
    print("- debug_processed_image.png (after preprocessing)")
    print("These will help identify where the issue is occurring.")

if __name__ == "__main__":
    main() 