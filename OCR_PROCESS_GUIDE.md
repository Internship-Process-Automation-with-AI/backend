# OCR Process Guide

## 🚀 Quick Start - Running OCR Workflow

### Command Line Usage
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
# Windows:
python -m venv venv
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API Integration
```bash
# Start the FastAPI server
uvicorn src.API.main:app --reload

```

## What is OCR?

OCR (Optical Character Recognition) converts images, PDFs, and documents into readable text. Our system is specially designed for processing work certificates and academic documents.

## What Files Can We Process?

- **PDF files** (.pdf)
- **Word documents** (.docx, .doc) 
- **Images** (.png, .jpg, .jpeg, .bmp, .tiff)

## How It Works (Simple Version)

### Step 1: File Type Detection & Processing
The system automatically detects your file type and uses the most appropriate processing method:

#### **Word Documents (.docx, .doc)**
- **Processing Method**: Direct text extraction using python-docx library
- **Speed**: Fastest (1-3 seconds)
- **Accuracy**: 95-99% (extracts native text, no OCR needed)
- **What Happens**: Opens the document and reads the actual text content directly

#### **PDF Files (.pdf)**
- **Processing Method**: PDF → Image conversion → OCR processing
- **Speed**: Medium (5-15 seconds depending on page count)
- **Accuracy**: 85-95% (depends on PDF quality)
- **What Happens**: 
  1. Converts each PDF page to high-resolution image
  2. Runs OCR on each image
  3. Combines text from all pages

#### **Image Files (.png, .jpg, .jpeg, .bmp, .tiff)**
- **Processing Method**: Direct OCR processing
- **Speed**: Medium (3-10 seconds)
- **Accuracy**: 70-90% (depends on image quality and text clarity)
- **What Happens**: Runs OCR directly on the image without conversion

### Step 2: Language Detection & Optimization
The system automatically detects the document language and applies specialized processing:

#### **Finnish Documents**
- **Detection Method**: Scans for Finnish characters (ä, ö, å) and common Finnish words
- **OCR Engine**: Uses Tesseract with Finnish language pack (`fin`)
- **Special Features**:
  - Recognizes Finnish business terms (työtodistus, harjoittelu, työpaikka)
  - Handles Finnish date formats (1.1.2024, 1. tammikuuta)
  - Optimized for Finnish company names and addresses
- **Processing Time**: Same as other languages
- **Accuracy**: 5-10% better than generic OCR for Finnish text

#### **English Documents**
- **Detection Method**: Scans for English words and sentence structure
- **OCR Engine**: Uses Tesseract with English language pack (`eng`)
- **Special Features**:
  - Recognizes English business terminology
  - Handles English date formats (January 1, 2024, 01/01/2024)
  - Optimized for international company names
- **Processing Time**: Same as other languages
- **Accuracy**: Standard OCR accuracy for English text

#### **Mixed Language Documents**
- **Detection Method**: Identifies primary language, then processes accordingly
- **Processing**: Uses the detected primary language's optimization
- **Fallback**: If uncertain, defaults to English processing

### Step 3: Text Extraction & Processing
Based on the detected language and file type:

#### **For Finnish Documents:**
1. **Character Recognition**: Special handling for ä, ö, å characters
2. **Word Correction**: Finnish-specific spell checking and correction
3. **Business Term Recognition**: Identifies Finnish work certificate terminology
4. **Format Handling**: Recognizes Finnish business document formats

#### **For English Documents:**
1. **Standard OCR**: Uses English language optimization
2. **Business Recognition**: Identifies English business document patterns
3. **Format Handling**: Recognizes international document formats

### Step 4: Quality Assessment & Confidence Scoring
The system evaluates the quality of extracted text:

#### **Confidence Levels:**
- **High Confidence (80-100%)**: 
  - Text is very accurate
  - Minimal corrections needed
  - Suitable for immediate AI processing
  
- **Medium Confidence (50-79%)**: 
  - Text is generally good
  - Some corrections may be needed
  - Still suitable for AI processing with minor adjustments
  
- **Low Confidence (0-49%)**: 
  - Text may have significant errors
  - Manual review recommended
  - May need re-scanning with better quality

#### **Quality Factors:**
- **Image Resolution**: Higher DPI = better accuracy
- **Text Clarity**: Clear, dark text on light background
- **Document Condition**: Clean, unmarked documents
- **Language Complexity**: Technical terms may reduce confidence

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


## What You Get from OCR

### OCR Output Format
```python
{
    "success": True,
    "text_length": 1250,           # Number of characters extracted
    "detected_language": "fin",     # Language detected (fin/eng)
    "confidence": 84.5,            # OCR confidence score (0-100%)
    "processing_time": 3.2,        # Time taken for OCR processing
    "extracted_text": "Your extracted text here...",
    "file_type": "pdf",            # Original file type processed
    "ocr_engine": "tesseract"      # OCR engine used
}
```

### OCR Results Storage
When using the API, OCR results are stored in the database:

#### **Certificates Table** (OCR-specific data):
- **File content**: Original file stored as `BYTEA`
- **OCR output**: Extracted text stored as `TEXT` in `ocr_output` field
- **File metadata**: Filename, filetype, training_type, uploaded_at
- **Processing info**: When OCR was completed and with what settings

## OCR-Specific API Endpoints

### **OCR Processing:**
- **Process Certificate**: `POST /certificate/{certificate_id}/process` - Runs OCR on uploaded file
- **Get Status**: `GET /certificate/{certificate_id}/status` - Check OCR processing status
- **Preview Certificate**: `GET /certificate/{certificate_id}/preview` - View original file

### **File Upload (Required for OCR):**
- **Upload Certificate**: `POST /student/{student_id}/upload-certificate` - Upload file before OCR processing

## Common OCR Questions

### Q: How accurate is the OCR?
**A:** It depends on the file type:
- Word documents: 95-99% (very accurate - no OCR needed)
- Good quality PDFs: 85-95% (quite accurate)
- Poor quality scans: 70-90% (may have some errors)

### Q: Does it work with Finnish documents?
**A:** Yes! The system is specially optimized for Finnish:
- Recognizes Finnish characters (ä, ö, å)
- Detects Finnish words automatically
- Uses Finnish-specific OCR language pack
- 5-10% better accuracy for Finnish text

### Q: What if the OCR fails?
**A:** The system has multiple fallback strategies:
- Tries different processing methods
- Uses different language settings
- Always returns something (even if confidence is low)
- Logs errors for debugging

### Q: How fast is OCR processing?
**A:** Processing speed depends on file type:
- Word documents: Very fast (1-3 seconds, no OCR needed)
- PDFs: Medium speed (5-15 seconds, depends on page count)
- Images: Medium speed (3-10 seconds, direct OCR)

### Q: Can I process files through the API?
**A:** Yes! The system provides OCR-specific endpoints:
- Upload files: `/student/{student_id}/upload-certificate`
- Run OCR: `/certificate/{certificate_id}/process`
- Check status: `/certificate/{certificate_id}/status`

## Tips for Best OCR Results

### ✅ Do This
- Use good quality scans (300 DPI or higher)
- Ensure documents are well-lit and clear
- Use standard file formats (PDF, DOCX, PNG)
- Clean, unmarked documents
- High contrast text (dark text on light background)

### ❌ Avoid This
- Very low quality scans (below 150 DPI)
- Documents with heavy shadows or blur
- Unusual file formats
- Handwritten text (system designed for printed text)
- Documents with heavy annotations or stamps

## OCR's Role in the System

The OCR system is the **first step** in a larger AI pipeline:
1. **Extracts text** (OCR step) - This is what this guide covers
2. **Analyzes content** (AI evaluation) - Uses the extracted text
3. **Validates companies** (Company verification) - Based on extracted text
4. **Makes decisions** (ACCEPTED/REJECTED) - Based on AI analysis
5. **Provides recommendations** (for rejected cases)

**The OCR provides the text input** that the AI system uses to evaluate work certificates and determine academic credits. Without good OCR results, the AI cannot make accurate decisions.

## OCR Database Storage

### **OCR Data Storage:**

#### **Certificates Table** (where OCR results are stored):
- `certificate_id` (UUID, Primary Key)
- `file_content` (BYTEA) - Original uploaded file
- `ocr_output` (TEXT) - **Extracted text from OCR processing**
- `filename` (VARCHAR) - Original filename
- `filetype` (VARCHAR) - File extension (.pdf, .docx, .png, etc.)
- `uploaded_at` (TIMESTAMP) - When file was uploaded

### **OCR Processing Workflow:**
1. **Upload**: File is uploaded and stored in `certificates` table
2. **OCR Processing**: Text is extracted and stored in `ocr_output` field
3. **Result**: Extracted text is ready for AI processing

**Note**: The `decisions` table stores AI evaluation results, not OCR data. This guide focuses on OCR processing only.

## Summary

Our OCR system is:
- **Smart**: Automatically detects languages and file types
- **Accurate**: Uses multiple strategies for best results
- **Fast**: Optimized for different file types
- **Finnish-friendly**: Special support for Finnish documents with 5-10% better accuracy
- **Reliable**: Multiple fallback strategies if OCR fails
- **Database-integrated**: Stores OCR results in PostgreSQL
- **API-ready**: Provides OCR-specific FastAPI endpoints

**Purpose**: Extract text from work certificates and academic documents with high accuracy, especially optimized for Finnish language support. The extracted text is then used by the AI system for further evaluation and decision-making.

**This guide covers**: OCR processing only - how to extract text from documents. For information about AI evaluation, company validation, or the complete workflow, please refer to other system documentation. 