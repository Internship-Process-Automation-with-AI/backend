ocr_service.py

The OCRService class is the core document processing engine that extracts text from various file formats (PDFs, images) using multiple strategies.
🔧 What It Does Step by Step
1. Smart Processing Strategy
File Input → Determine Type → Choose Best Method → Extract Text → Return Results
2. File Type Detection
PDFs: Uses PyMuPDF for fast text extraction, falls back to OCR if needed
Images: Uses OCR (Tesseract + Google Vision fallback)
Other formats: Returns error for unsupported types
3. Processing Methods
For PDFs:
First Try: PyMuPDF direct text extraction (fast, accurate)
Quality Check: If text quality is poor (< 30% confidence), fallback to OCR
OCR Fallback: Convert PDF to images, then use Tesseract/Google Vision
For Images:
Preprocessing: Apply image enhancement (grayscale, binarization, noise removal)
Primary OCR: Tesseract
Fallback OCR: Google Vision API (if available)
4. Key Features
Smart Decision Making:
    # Tries PyMuPDF first (fast)
if quality_score > 30.0 and len(extracted_text.strip()) > 50:
    return PyMuPDF_result
else:
    # Falls back to OCR (slower but more reliable)
    return OCR_result

Multiple OCR Engines:
Tesseract: Primary OCR (free, local)
Google Vision: Fallback OCR (cloud-based, more accurate)
Quality Assessment:
Calculates confidence scores
Determines if results are successful
Provides processing statistics
5. Output Format
Returns an OCRResult object containing:
Extracted text (string)
Confidence score (0-100%)
Engine used (pymupdf/tesseract/google_vision)
Processing time (seconds)
Success status (boolean)
