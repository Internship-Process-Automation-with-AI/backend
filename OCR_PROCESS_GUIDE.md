# OCR Process Guide

## 🚀 Quick Start - Running OCR Workflow

### Command Line Usage
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run OCR workflow on all documents in samples/ folder
python -m src.workflow.ocr_workflow

# Run with custom directories
python -m src.workflow.ocr_workflow --samples_dir "my_documents" --output_dir "my_results"
```

## What is OCR?

OCR (Optical Character Recognition) converts images, PDFs, and documents into readable text. Our system is specially designed for processing work certificates and academic documents.

## What Files Can We Process?

- **PDF files** (.pdf)
- **Word documents** (.docx, .doc) 
- **Images** (.png, .jpg, .jpeg, .bmp, .tiff)

## How It Works (Simple Version)

### Step 1: File Detection
The system looks at your file and decides how to process it:
- **Word documents**: Extract text directly (fastest, most accurate)
- **PDFs**: Convert to images, then extract text
- **Images**: Extract text directly

### Step 2: Language Detection
The system automatically detects if your document is:
- **Finnish** (looks for Finnish words and characters like ä, ö, å)
- **English** 

### Step 3: Text Extraction
- **Finnish documents**: Uses special Finnish-optimized settings
- **Other documents**: Uses standard settings
- **Word documents**: Gets text directly (95%+ accuracy)

### Step 4: Quality Check
The system checks how confident it is about the extracted text:
- **High confidence** (80%+): Text is likely very accurate
- **Medium confidence** (50-80%): Text is probably good
- **Low confidence** (<50%): Text might have errors

## Key Features

### 🎯 Smart Processing
- **Automatic language detection** - No need to specify language
- **File type optimization** - Each file type gets the best processing method
- **Quality scoring** - System picks the best result automatically

### 🇫🇮 Finnish Language Support
- **Finnish character recognition** (ä, ö, å)
- **Finnish word detection** (työtodistus, harjoittelu, etc.)
- **Finnish error correction** (fixes common OCR mistakes)

### 📊 Accuracy Levels
- **Word documents**: 95-99% accurate (native text)
- **Good quality PDFs**: 85-95% accurate
- **Scanned images**: 70-90% accurate (depends on scan quality)

## How to Use

### Simple Usage
```python
from src.ocr.cert_extractor import extract_certificate_text

# Extract text from any file
text = extract_certificate_text("my_document.pdf")
print(f"Extracted {len(text)} characters")
```

### Finnish Documents
```python
from src.ocr.cert_extractor import extract_finnish_certificate

# Special handling for Finnish documents
text = extract_finnish_certificate("finnish_document.pdf")
```

### Batch Processing
```python
from src.workflow.ocr_workflow import OCRWorkflow

# Process multiple documents
workflow = OCRWorkflow(samples_dir="documents", output_dir="results")
summary = workflow.process_all_documents()
print(f"Processed {summary['successful']} documents successfully")
```

## What You Get

### Output Format
```python
{
    "success": True,
    "text_length": 1250,           # Number of characters extracted
    "detected_language": "fin",     # Language detected
    "confidence": 84.5,            # Confidence score (0-100%)
    "processing_time": 3.2,        # Time taken in seconds
    "extracted_text": "Your extracted text here..."
}
```

### File Organization
Results are saved in organized folders:
```
processedData/
├── document1/
│   ├── ocr_output_document1.txt    # Extracted text
│   └── aiworkflow_output_*.json    # AI evaluation results
└── document2/
    ├── ocr_output_document2.txt
    └── aiworkflow_output_*.json
```

## Common Questions

### Q: How accurate is the OCR?
**A:** It depends on the file type:
- Word documents: 95-99% (very accurate)
- Good quality PDFs: 85-95% (quite accurate)
- Poor quality scans: 70-90% (may have some errors)

### Q: Does it work with Finnish documents?
**A:** Yes! The system is specially optimized for Finnish:
- Recognizes Finnish characters (ä, ö, å)
- Detects Finnish words automatically
- Uses Finnish-specific processing

### Q: What if the OCR fails?
**A:** The system has multiple fallback strategies:
- Tries different processing methods
- Uses different language settings
- Always returns something (even if confidence is low)

### Q: How fast is it?
**A:** Processing speed depends on file type:
- Word documents: Very fast (native text extraction)
- PDFs: Medium speed (conversion + OCR)
- Images: Medium speed (direct OCR)

## Tips for Best Results

### ✅ Do This
- Use good quality scans (300 DPI or higher)
- Ensure documents are well-lit and clear
- Use standard file formats (PDF, DOCX, PNG)

### ❌ Avoid This
- Very low quality scans
- Documents with heavy shadows or blur
- Unusual file formats

## Integration with AI Pipeline

The OCR system is part of a larger AI pipeline that:
1. **Extracts text** (OCR step)
2. **Analyzes content** (AI evaluation)
3. **Makes decisions** (ACCEPTED/REJECTED)
4. **Provides recommendations** (for rejected cases)

The OCR provides the text input that the AI system uses to evaluate work certificates and determine academic credits.

## Summary

Our OCR system is:
- **Smart**: Automatically detects languages and file types
- **Accurate**: Uses multiple strategies for best results
- **Fast**: Optimized for different file types
- **Finnish-friendly**: Special support for Finnish documents
- **Reliable**: Multiple fallback strategies

It's designed to work seamlessly with the AI evaluation pipeline to process work certificates and determine academic credits automatically. 