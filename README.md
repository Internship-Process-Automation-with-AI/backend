# OAMK Internship Certificate OCR Processing System

A comprehensive OCR (Optical Character Recognition) service for extracting text from scanned internship certificates. This system supports multiple file formats, advanced image preprocessing, and uses **PyMuPDF for fast text extraction** with Tesseract OCR and Google Vision API as fallbacks.

## 🚀 **Key Features**

- **Multi-format Support**: PDF, PNG, JPG, JPEG, TIFF, BMP
- **Smart Processing**: PyMuPDF text extraction + OCR fallback
- **Advanced Preprocessing**: Grayscale conversion, binarization, noise removal, deskewing
- **Dual OCR Engines**: Tesseract (primary) + Google Vision API (fallback)
- **PDF Processing**: Fast text extraction with image conversion fallback
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Batch Processing**: Process multiple files simultaneously
- **Confidence Scoring**: Quality assessment of OCR results
- **Error Handling**: Robust error handling and logging

## 🎯 **Smart Processing Pipeline**

```
PDF Upload → PyMuPDF Text Extraction → Quality Check → 
    ↓ (if poor quality)
OCR Fallback → Tesseract → Google Vision
```

## 🐍 **Virtual Environment Setup (Recommended)**

**Using a virtual environment is highly recommended to avoid dependency conflicts and keep your system clean.**

### **Step 1: Create Virtual Environment**

**Windows:**
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### **Step 2: Verify Virtual Environment**

After activation, you should see `(venv)` at the beginning of your command prompt:

```bash
(venv) C:\Users\mello\OAMK PROJECT\backend>
```

### **Step 3: Install Dependencies in Virtual Environment**

```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

### **Step 4: Deactivate When Done**

```bash
# When you're finished working
deactivate
```

### **Virtual Environment Commands Reference**

| Command | Description |
|---------|-------------|
| `python -m venv venv` | Create virtual environment |
| `venv\Scripts\activate` (Windows) | Activate virtual environment |
| `source venv/bin/activate` (macOS/Linux) | Activate virtual environment |
| `deactivate` | Deactivate virtual environment |
| `pip list` | Show installed packages |
| `pip freeze > requirements.txt` | Save current dependencies |

### **IDE Integration**

**VS Code:**
1. Open the backend folder in VS Code
2. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (macOS)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from `./venv/Scripts/python.exe` (Windows) or `./venv/bin/python` (macOS/Linux)

**PyCharm:**
1. Go to File → Settings → Project → Python Interpreter
2. Click the gear icon → Add
3. Choose "Existing Environment"
4. Select the interpreter from your venv folder

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for OCR fallback)
3. **Google Cloud Vision API** (optional, for advanced fallback)

### Installation

1. **Clone the repository**
   ```bash
   cd backend
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**

   **Windows:**
   ```bash
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   # Add to PATH or set TESSERACT_CMD in .env
   ```

   **macOS:**
   ```bash
   brew install tesseract
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your settings
   ```

### Configuration

Create a `.env` file in the backend directory:

```env
# OCR Configuration
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows path
GOOGLE_CLOUD_CREDENTIALS=path/to/your/credentials.json

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760  # 10MB in bytes

# Processing Configuration
OCR_CONFIDENCE_THRESHOLD=60.0
IMAGE_PREPROCESSING_ENABLED=true
```

### Running the Service

1. **Activate virtual environment (if using)**
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Start the FastAPI server**
   ```bash
   python main.py
   ```

3. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Test the service**
   ```bash
   python test_ocr.py
   ```

## API Endpoints

### Extract Text from File
```http
POST /api/v1/ocr/extract-text
Content-Type: multipart/form-data

file: [uploaded file]
use_preprocessing: true (optional)
```

**Response:**
```json
{
  "filename": "certificate.pdf",
  "file_size": 1024000,
  "file_type": ".pdf",
  "extracted_text": "This is the extracted text...",
  "confidence": 85.5,
  "ocr_engine": "pymupdf",
  "processing_time": 0.34,
  "success": true,
  "preprocessing_applied": true
}
```

### Get Supported Formats
```http
GET /api/v1/ocr/supported-formats
```

### Get Service Status
```http
GET /api/v1/ocr/status
```

### Batch Processing
```http
POST /api/v1/ocr/batch
Content-Type: multipart/form-data

files: [multiple files]
use_preprocessing: true (optional)
```

## Usage Examples

### Python Client Example

```python
import requests

# Upload and process a file
with open('certificate.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/v1/ocr/extract-text', files=files)
    
result = response.json()
print(f"Extracted text: {result['extracted_text']}")
print(f"Engine used: {result['ocr_engine']}")
print(f"Confidence: {result['confidence']}%")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract-text" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@certificate.pdf" \
     -F "use_preprocessing=true"
```

## Architecture

```
backend/
├── venv/                    # Virtual environment (created by you)
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── services/
│   │   └── ocr_service.py     # Smart OCR service
│   └── utils/
│       ├── image_preprocessing.py  # Image enhancement
│       └── pdf_converter.py        # PyMuPDF text extraction
├── main.py                    # FastAPI application
├── test_ocr.py               # Test script
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Smart Processing Pipeline

1. **File Validation**: Check file type and size
2. **Smart PDF Processing**: 
   - Try PyMuPDF text extraction first (fast)
   - Fallback to OCR if text quality is poor
3. **Image Preprocessing**: 
   - Grayscale conversion
   - Noise reduction
   - Contrast enhancement
   - Binarization
   - Deskewing (if needed)
4. **OCR Extraction**: 
   - Try Tesseract first
   - Fallback to Google Vision API if needed
5. **Result Processing**: Combine text, calculate confidence
6. **Response**: Return structured JSON with results

## Benefits of PyMuPDF

### ✅ **Advantages:**
- **No External Dependencies**: No need for Poppler or other system libraries
- **Faster Processing**: Direct text extraction is much faster than OCR
- **Better Accuracy**: Native text extraction is more accurate than OCR
- **Lower Resource Usage**: Less memory and CPU intensive
- **Cross-Platform**: Works consistently across all platforms
- **Hybrid Approach**: Can still use OCR when needed

### 🔄 **Processing Strategy:**
- **Text-based PDFs**: Use PyMuPDF for instant, accurate extraction
- **Image-based PDFs**: Fallback to OCR with preprocessing
- **Mixed PDFs**: Combine both approaches for best results

## Testing

### Run Test Suite
```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run tests
python test_ocr.py
```

### Test with Sample Files
The test script will automatically test with files from the `file samples/` directory:
- Finnish certificates
- Form-based formats
- Letter-based formats
- Scanned documents

### Manual Testing
1. Activate virtual environment
2. Start the server: `python main.py`
3. Open http://localhost:8000/docs
4. Use the interactive API documentation to test endpoints

## Troubleshooting

### Common Issues

**1. Virtual Environment Issues**
```
Error: 'venv' is not recognized as an internal or external command
```
**Solution**: 
- Make sure you're in the backend directory
- Use the correct activation command for your OS
- Check that the venv folder exists

**2. Tesseract not found**
```
Error: TesseractNotFoundError
```
**Solution**: Install Tesseract and set `TESSERACT_CMD` in `.env`

**3. PyMuPDF installation issues**
```
Error: ImportError: No module named 'fitz'
```
**Solution**: 
- Make sure virtual environment is activated
- Reinstall PyMuPDF: `pip install --upgrade PyMuPDF`

**4. Google Vision API errors**
```
Error: google.cloud.vision error
```
**Solution**: Check credentials file path in `GOOGLE_CLOUD_CREDENTIALS`

**5. Low OCR confidence**
```
Warning: Low confidence OCR result
```
**Solution**: 
- Enable preprocessing: `use_preprocessing=true`
- Check image quality
- Try different preprocessing settings

### Performance Optimization

1. **Text-based PDFs**: Use PyMuPDF for instant processing
2. **Image-based PDFs**: Use OCR with preprocessing
3. **File Size**: Keep files under 10MB
4. **Batch Processing**: Use for multiple files

### Logging

The service uses structured logging. Check logs for:
- Processing times
- OCR confidence scores
- Engine selection (PyMuPDF vs OCR)
- Error details
- Service status

## Development

### Adding New Features

1. **New OCR Engine**: Extend `OCRService` class
2. **Preprocessing**: Add methods to `ImagePreprocessor`
3. **File Format**: Update `ALLOWED_EXTENSIONS` in config
4. **API Endpoints**: Add to `main.py`

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Include error handling
- Write tests for new features

### Development Workflow

1. **Activate virtual environment**
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install new dependencies**
   ```bash
   pip install new-package
   pip freeze > requirements.txt
   ```

3. **Run tests**
   ```bash
   python test_ocr.py
   ```

4. **Start development server**
   ```bash
   python main.py
   ```

## Deployment

### Production Considerations

1. **Environment Variables**: Set production values
2. **CORS**: Configure allowed origins
3. **File Storage**: Use cloud storage for uploads
4. **Monitoring**: Add health checks and metrics
5. **Security**: Implement authentication/authorization

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

## License

This project is part of the OAMK Internship Certificate Processing System.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the test logs
4. Contact the development team