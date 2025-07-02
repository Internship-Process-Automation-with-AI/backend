# AI Internship Workflow Application

This project streamlines and automates the administrative workflow for student internships at OAMK. It features advanced Optical Character Recognition (OCR) using Tesseract to extract text from various document formats (PDF, DOCX, DOC, JPG, PNG, BMP, TIFF), making internship certificate processing efficient and automated.

## Features

- 🔍 **OCR Text Extraction**: Extract text from scanned certificates and documents
- 📄 **Multi-format Support**: PDF, DOCX, DOC, and all major image formats
- 🖼️ **Image Preprocessing**: Automatic grayscale conversion and binarization
- ⚙️ **Smart Configuration**: Auto-detection of Tesseract installation
- 📝 **Clean Output**: Normalized and formatted text extraction
- 🛠️ **Production Ready**: Type-safe, well-documented, and tested

## Project Structure

```
backend/
├── src/
│   ├── ocr/
│   │   ├── cert_extractor.py      # Certificate text extraction module
│   │   └── ocr.py                 # OCR processor with Tesseract integration
│   ├── utils/
│   │   ├── logger.py              # Logging utility
│   │   └── __init__.py
│   └── workflow/
│       └── automator.py           # Workflow automation (future)
├── config/
│   └── settings.py                # Application configuration with auto-detection
├── samples/                       # Sample files for testing
├── pyproject.toml                 # Project dependencies and configuration
├── .pre-commit-config.yaml        # Code quality hooks
├── test_cert_extraction.py        # Testing script
└── README.md                      # This file
```

## Prerequisites

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

## Setup Instructions

### 1. Clone the Repository
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
# Or install individual packages:
pip install pytesseract opencv-python pillow pdf2image python-docx docx2txt pydantic pydantic-settings
```

### 5. Configure Tesseract (Optional)

The application automatically detects Tesseract installation. If needed, create a `.env` file:

```env
# Only set if Tesseract is not in PATH or needs custom path
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Usage

### Basic Certificate Text Extraction

```python
from src.ocr.cert_extractor import extract_certificate_text

# Extract text from any supported format
text = extract_certificate_text("path/to/certificate.pdf")
text = extract_certificate_text("path/to/certificate.docx")
text = extract_certificate_text("path/to/certificate.jpg")

print(text)
```

### Testing the Setup

```bash
# Run the test script
python test_cert_extraction.py

# This will:
# 1. Create a sample certificate image
# 2. Test OCR extraction
# 3. Display results
```

### Adding Your Own Test Files

1. Place your certificate files in the `samples/` directory
2. Supported formats: `.pdf`, `.docx`, `.doc`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
3. Run the test script to process all files

## Configuration

The application uses smart configuration with automatic Tesseract detection:

- **Auto-detection**: Checks common installation paths
- **Environment variables**: Override with `TESSERACT_CMD`
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Logging**: Comprehensive logging for debugging

## Development

### Code Quality Tools

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Run pre-commit on staged files only
pre-commit run

```

### Project Dependencies

- **OCR & Image Processing**: `pytesseract`, `opencv-python`, `pillow`
- **Document Processing**: `pdf2image`, `python-docx`, `docx2txt`
- **Configuration**: `pydantic`, `pydantic-settings`
- **Development**: `ruff`, `mypy`, `pre-commit`

## Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Ensure Tesseract is installed and in PATH
   - Set `TESSERACT_CMD` environment variable
   - Check installation: `tesseract --version`

2. **PDF processing fails**
   - Install `poppler-utils` (Linux) or `poppler` (macOS/Windows)
   - For Windows: Download from https://github.com/oschwartz10612/poppler-windows

3. **Poor OCR accuracy**
   - Ensure documents are high-resolution
   - Images should have good contrast
   - Text should be clearly readable

### Logging

Check logs for detailed error information:
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
```


