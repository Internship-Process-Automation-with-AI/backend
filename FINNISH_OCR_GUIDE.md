# 🇫🇮 Finnish OCR Processing Guide

This guide shows you how to use the enhanced OCR pipeline with **Finnish language support** for processing Finnish internship certificates and documents.

## 📋 Prerequisites

### 1. Install Finnish Language Pack for Tesseract

**Windows:**
```powershell
# Method 1: Download additional language packs
# Visit: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install: tesseract-ocr-w64-setup-additional-languages-v5.3.0.exe

# Method 2: Manual installation (easy)
# Download fin.traineddata from: https://github.com/tesseract-ocr/tessdata
# Place in: C:\Program Files\Tesseract-OCR\tessdata\
```

**macOS:**
```bash
# Install with Homebrew
brew install tesseract-lang

# Or specifically for Finnish
brew install tesseract --with-all-languages
```

**Linux (Ubuntu/Debian):**
```bash
# Install Finnish language pack
sudo apt-get update
sudo apt-get install tesseract-ocr-fin

# Verify installation
tesseract --list-langs
# Should include 'fin' in the output
```

### 2. Verify Installation
```bash
# Check if Finnish language is available
tesseract --list-langs | grep fin

# Test Finnish OCR
echo "Testi tekstiä" | tesseract stdin stdout -l fin
```

## 🚀 Usage

### Command Line
```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

```

### API Integration
```bash
# Start FastAPI server
uvicorn src.API.main:app --reload

# Upload and process Finnish documents
# The system automatically detects Finnish language
```

### Python Code
```python
from src.ocr.cert_extractor import extract_finnish_certificate

# Extract text from Finnish document
result = extract_finnish_certificate("finnish_document.pdf")
print(f"Extracted {len(result.text)} characters")
print(f"Language detected: {result.detected_language}")
print(f"Confidence: {result.confidence}%")
```

## 🔍 Finnish Language Detection

### Automatic Detection
The system automatically detects Finnish documents by:
- **Character Analysis**: Looking for Finnish-specific characters (ä, ö, å)
- **Keyword Recognition**: Identifying Finnish words like:
  - `työtodistus` (work certificate)
  - `harjoittelu` (internship)
  - `työsuhteen` (employment relationship)
  - `vastuualueet` (areas of responsibility)
- **Context Analysis**: Analyzing document structure and content patterns

### Detection Examples
```python
# Finnish document indicators
finnish_indicators = [
    "työtodistus", "harjoittelu", "työsuhteen", "vastuualueet",
    "työnantaja", "työntekijä", "työaika", "työtehtävät",
    "sairaanhoitaja", "insinööri", "tradenomi", "medianomi"
]

# Language detection result
{
    "detected_language": "fin",
    "confidence": 95.2,
    "finnish_indicators_found": ["työtodistus", "harjoittelu"],
    "processing_method": "finnish_optimized"
}
```

## 📄 Finnish Document Types

### Common Finnish Work Certificates
- **Työtodistus**: Standard work certificate
- **Harjoittelutodistus**: Internship certificate
- **Työsuoritusvakuutus**: Work performance guarantee
- **Työsuhteen päättymistodistus**: Employment termination certificate

### Document Structure
```
TYÖTODISTUS

Työntekijän nimi: [Name]
Henkilötunnus: [Personal ID]
Työnantaja: [Employer]
Työsuhteen alkupäivä: [Start Date]
Työsuhteen päättymispäivä: [End Date]
Työtehtävät: [Job Duties]
Vastuualueet: [Areas of Responsibility]
```

## 🎯 Finnish OCR Optimization

### Language-Specific Settings
```python
# Finnish OCR configuration
finnish_config = {
    "language": "fin",
    "config": "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖabcdefghijklmnopqrstuvwxyzåäö0123456789.,;:!?()[]{}'\"-–— ",
    "preprocessing": "finnish_optimized"
}
```

### Preprocessing for Finnish Documents
1. **Character Normalization**: Handle Finnish characters (ä, ö, å)
2. **Layout Analysis**: Finnish documents often have specific layouts
3. **Text Cleaning**: Remove common OCR artifacts in Finnish text
4. **Confidence Scoring**: Finnish-specific confidence thresholds

### Common Finnish OCR Challenges
- **Character Recognition**: ä, ö, å can be confused with a, o, a
- **Compound Words**: Finnish has many long compound words
- **Technical Terms**: Industry-specific Finnish terminology
- **Date Formats**: Finnish date formats (DD.MM.YYYY)

## 📊 Performance Metrics

### Finnish OCR Accuracy
- **High Quality Scans**: 90-95% accuracy
- **Medium Quality Scans**: 80-90% accuracy
- **Low Quality Scans**: 70-80% accuracy
- **Handwritten Text**: 60-80% accuracy (depends on handwriting quality)

### Processing Speed
- **Finnish Language Pack**: Slightly slower than English (due to larger vocabulary)
- **Optimization**: Finnish-specific preprocessing adds minimal overhead
- **Batch Processing**: Multiple Finnish documents processed efficiently

## 🔧 Troubleshooting

### Common Issues

#### 1. Finnish Language Not Detected
```bash
# Check Tesseract installation
tesseract --list-langs

# Verify Finnish language pack
ls /usr/share/tessdata/fin.traineddata  # Linux
ls "C:\Program Files\Tesseract-OCR\tessdata\fin.traineddata"  # Windows
```

#### 2. Poor Finnish Text Recognition
```python
# Try different preprocessing
from src.ocr.cert_extractor import extract_certificate_text

# Force Finnish language
result = extract_certificate_text("document.pdf", language="fin")

# Use Finnish-specific preprocessing
result = extract_finnish_certificate("document.pdf")
```

#### 3. Character Recognition Issues
```python
# Check character encoding
import chardet

with open("ocr_output.txt", "rb") as f:
    raw_data = f.read()
    encoding = chardet.detect(raw_data)
    print(f"Detected encoding: {encoding}")
```

### Debug Mode
```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Run OCR with debug output
result = extract_finnish_certificate("document.pdf", debug=True)
```

## 🌐 Integration with AI Workflow

### Finnish Document Processing Pipeline
1. **Document Upload**: Finnish document uploaded to system
2. **Language Detection**: Automatic Finnish language detection
3. **OCR Processing**: Finnish-optimized text extraction
4. **AI Evaluation**: LLM processes Finnish text for credit evaluation
5. **Results**: Finnish-specific evaluation results

```

## 🚀 Best Practices

### For Finnish Documents
1. **Use High Quality Scans**: 300 DPI or higher for best results
2. **Ensure Good Lighting**: Clear, well-lit documents
3. **Check Character Quality**: Finnish characters should be clearly visible
4. **Use Standard Formats**: PDF or high-quality images
5. **Verify Language Pack**: Ensure Finnish Tesseract pack is installed

### Performance Optimization
2. **Language Detection**: Let system auto-detect Finnish
3. **Preprocessing**: Use Finnish-specific preprocessing when available
4. **Error Handling**: Implement fallback for language detection failures

## 📈 Future Enhancements

### Planned Finnish Support Improvements
- **Enhanced Character Recognition**: Better ä, ö, å recognition
- **Finnish Business Context**: Industry-specific terminology
- **Regional Variations**: Support for different Finnish dialects
- **Handwritten Finnish**: Improved handwritten text recognition
- **Finnish Document Templates**: Pre-defined document structures

---

**Note**: This Finnish OCR support is specifically designed for OAMK's academic environment and Finnish work certificate processing. The system automatically detects and optimizes for Finnish documents to provide the best possible text extraction results.

