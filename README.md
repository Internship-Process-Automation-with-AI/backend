# OAMK Work Certificate Processing System

An advanced AI-powered system for processing and evaluating work certificates for academic credit assessment at OAMK University of Applied Sciences. The system combines Optical Character Recognition (OCR) with Large Language Model (LLM) processing to automatically extract, validate, and evaluate work experience for academic credit qualification.

## 🚀 Features

- 🔍 **Advanced OCR Processing**: Extract text from various document formats (PDF, DOCX, PNG, JPG, TIFF, BMP)
- 🤖 **LLM-Powered Analysis**: Intelligent extraction and evaluation using Google Gemini AI with automatic fallback models
- 🎓 **Academic Credit Assessment**: Automatic ECTS credit calculation and degree relevance evaluation
- ✅ **Multi-Stage Validation**: Comprehensive validation and correction pipeline
- 📊 **Organized Output**: Clean, structured output with organized file management and cleaned JSON
- 🌐 **Bilingual Support**: Finnish and English degree program support with automatic language detection
- 🛠️ **Production Ready**: Type-safe, well-documented, and thoroughly tested

## 📁 Project Structure

```
backend/
├── src/
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── ocr_model.py              # OCR service with Tesseract integration
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── cert_extractor.py         # LLM orchestrator for processing
│   │   ├── degree_evaluator.py       # Degree program management
│   │   ├── degree_programs_data.py   # Bilingual degree program definitions
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── correction.py         # Correction prompt templates
│   │       ├── evaluation.py         # Evaluation prompt templates
│   │       ├── extraction.py         # Extraction prompt templates
│   │       └── validation.py         # Validation prompt templates
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── docx_processor.py         # DOCX file processing
│   │   ├── finnish_ocr_corrector.py  # Finnish text correction
│   │   ├── image_preprocessing.py    # Image preprocessing utilities
│   │   ├── logger.py                 # Logging utilities
│   │   └── pdf_converter.py          # PDF processing utilities
│   ├── config.py                     # Application configuration
│   ├── mainpipeline.py               # Complete end-to-end processing pipeline
│   ├── test_extractor.py             # LLM processing test script
│   └── test_ocr.py                   # OCR processing test script
├── samples/                          # Sample work certificates for testing
│   ├── scanned/                      # Scanned document samples
│   ├── letter based format/          # Letter-based certificate samples
│   └── ...                          # Various sample formats
├── outputs/                          # Organized output directory
│   └── {sample_name}/                # Each sample gets its own directory
│       ├── OCRoutput_{sample}.txt    # OCR extracted text
│       └── LLMoutput_{sample}_*.json # LLM evaluation results
├── requirements.txt                  # Python dependencies
├── setup_precommit.py                # Pre-commit setup script
└── README.md                         # This file
```

## 🎯 Core Functionality

### 1. **OCR Processing**
- Multi-format document support (PDF, DOCX, images)
- Advanced text extraction with Tesseract OCR
- Automatic image preprocessing for better accuracy
- Finnish text correction and normalization

### 2. **LLM-Powered Analysis**
- **Extraction**: Extract employee information, job details, and employment periods
- **Evaluation**: Calculate working hours, determine training type, and assess degree relevance
- **Validation**: Validate results against original documents and identify issues
- **Correction**: Automatically correct identified problems

### 3. **Academic Credit Assessment**
- Automatic ECTS credit calculation (27 hours = 1 ECTS)
- Training type classification (General vs Professional)
- Degree relevance evaluation with bilingual support
- Credit limit enforcement (10 ECTS max for general training)
- Bilingual degree program matching (Finnish and English)

## 🛠️ Prerequisites

### 1. Install Tesseract OCR

**Windows:**
```powershell
# Option 1: Download from official site
# Visit: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install tesseract-ocr-w64-setup-v5.3.0.exe

# Option 2: Using Chocolatey
choco install tesseract

# Option 3: Using Scoop
scoop install tesseract
```

**macOS:**
```bash
# Using Homebrew
brew install tesseract

# Using MacPorts
sudo port install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install tesseract
```

### 2. Verify Tesseract Installation
```bash
tesseract --version
```

### 3. Set Up Google Gemini API
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 🚀 Setup Instructions

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```powershell
.\venv\Scripts\Activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Pre-commit Hooks (Optional)
```bash
python setup_precommit.py
```

## 📖 Usage

### Option 1: Complete Pipeline (Recommended)
Process documents from start to finish with OCR and LLM evaluation:

```bash
cd src
python mainpipeline.py
```

This will:
1. Show available sample files
2. Let you select a document
3. Let you choose a degree program
4. Process through OCR → LLM → Validation → Correction
5. Save organized results

### Option 2: Individual Components

**OCR Processing Only:**
```bash
cd src
python test_ocr.py
```

**LLM Processing Only (requires OCR output):**
```bash
cd src
python test_extractor.py
```

## 📊 Output Structure

The system creates organized output directories:

```
backend/outputs/
└── sample_certificate/
    ├── OCRoutput_sample_certificate.txt
    └── LLMoutput_sample_certificate_pipeline_20250702_120000.json
```

### Output Files:

1. **OCR Text File** (`OCRoutput_*.txt`):
   - Raw extracted text from the document
   - Cleaned and normalized

2. **LLM Results File** (`LLMoutput_*.json`):
   - Complete processing results (cleaned JSON format)
   - Extraction, evaluation, validation, and correction data
   - Academic credit assessment
   - Processing metadata
   - Simplified file paths and removed success fields for cleaner output

## 🎓 Supported Degree Programs

The system supports various OAMK degree programs including:
- Insinööri (AMK), tieto- ja viestintätekniikka
- Bachelor of Engineering (BEng), Information Technology
- Rakennusmestari (AMK)
- And more...

## 🔄 Recent Features

### JSON Output Cleaning
- **Removed Success Fields**: All `"success": true/false` fields are automatically removed from output JSON for cleaner, more readable results
- **Simplified File Paths**: File paths are simplified to show only `samples/filename` instead of full absolute paths
- **Cleaner Structure**: Output JSON is optimized for readability while preserving all essential data

### Gemini Model Fallback
- **Automatic Fallback**: If the primary Gemini model reaches quota limits, the system automatically switches to fallback models
- **Seamless Processing**: Fallback happens transparently without interrupting the processing pipeline
- **Multiple Models**: Supports fallback to `gemini-1.5-pro` when quota is reached

### Bilingual Support
- **Language Detection**: Automatically detects document language (Finnish/English)
- **Bilingual Matching**: Degree programs are matched using both Finnish and English keywords
- **Improved Accuracy**: Finnish certificates now receive correct relevance scoring and justifications

## ⚙️ Configuration

### Environment Variables
```bash
# Required for LLM processing
GEMINI_API_KEY=your-api-key-here

# Optional: Custom Tesseract path
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Degree Program Configuration
Edit `src/llm/degree_programs_data.py` to add or modify degree programs and their evaluation criteria. The system supports both Finnish and English degree programs with automatic language detection.

## 🔧 Development

### Code Quality Tools
```bash
# Install pre-commit hooks
python setup_precommit.py

# Run pre-commit on all files
pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit run
```

### Project Dependencies

**Core Processing:**
- `pytesseract`: OCR text extraction
- `opencv-python`: Image preprocessing
- `pillow`: Image handling
- `google-generativeai`: Google Gemini AI integration with fallback models

**Document Processing:**
- `pdf2image`: PDF to image conversion
- `python-docx`: DOCX file processing
- `pymupdf`: PDF text extraction

**Configuration & Development:**
- `pydantic`: Data validation
- `ruff`: Code linting
- `mypy`: Type checking
- `pre-commit`: Git hooks

## 🐛 Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Ensure Tesseract is installed and in PATH
   - Set `TESSERACT_CMD` environment variable
   - Check installation: `tesseract --version`

2. **"LLM orchestrator not available"**
   - Set `GEMINI_API_KEY` environment variable
   - Verify API key is valid
   - Check internet connection

3. **"No OCR output files found"**
   - Run `test_ocr.py` first to generate OCR outputs
   - Check that sample files exist in `samples/` directory

4. **Poor OCR accuracy**
   - Ensure documents are high-resolution
   - Images should have good contrast
   - Text should be clearly readable

5. **PDF processing fails**
   - Install `poppler-utils` (Linux) or `poppler` (macOS/Windows)
   - For Windows: Download from https://github.com/oschwartz10612/poppler-windows

### Logging
Check logs for detailed error information:
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```

## 📝 Sample Files

The `samples/` directory contains various test documents:
- Scanned PDFs
- Digital PDFs
- DOCX files
- Image files (PNG, JPG)
- Different certificate formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run pre-commit hooks: `pre-commit run --all-files`
5. Submit a pull request

## 📄 License

This project is developed for OAMK University of Applied Sciences.


