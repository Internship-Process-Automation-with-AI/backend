# OAMK Internship Certificate OCR System

A smart OCR service for extracting text from internship certificates using PyMuPDF for fast text extraction with Tesseract OCR fallback.

## Quick Start

### 1. Setup Environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Install Tesseract OCR
- **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 3. Run Tests with locally saved test files
```bash
python test_ocr.py
```

### 4. Start API Server (NOT YET IMPLEMENTED)
```bash
python main.py
# Open http://localhost:8000/docs
```

## Features

- **Multi-format Support**: PDF, PNG, JPG, JPEG, TIFF, BMP
- **Smart Processing**: PyMuPDF text extraction + OCR fallback
- **Image Preprocessing**: Grayscale, noise removal, deskewing
- **Dual OCR Engines**: Tesseract + Google Vision API
- **RESTful API**: FastAPI endpoints for integration

## File Structure

```
backend/
├── app/
│   ├── config.py              # Configuration
│   ├── ocr_service.py         # Main OCR service
│   └── utils/
│       ├── image_preprocessing.py
│       └── pdf_converter.py
├── file samples/              # Test files
├── output/                    # OCR results
├── main.py                    # FastAPI server
├── test_ocr.py               # Test script
└── requirements.txt
```

## Configuration

TO BE IMPLEMENTED LATER

## Development

```bash
# Add new dependencies
pip install new-package
pip freeze > requirements.txt

# Run tests
python test_ocr.py

# Start development server
python main.py
```