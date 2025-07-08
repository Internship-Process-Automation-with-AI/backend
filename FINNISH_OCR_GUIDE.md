# 🇫🇮 Finnish OCR Processing Guide

This guide shows you how to use the enhanced OCR pipeline with **Finnish language support** for processing Finnish internship certificates and documents.

## 📋 Prerequisites

### 1. Install Finnish Language Pack for Tesseract

**Windows:**
```powershell
# Method 1: Download additional language packs
# Visit: https://github.com/UB-Mannheim/tesseract/wiki
# Download and install: tesseract-ocr-w64-setup-additional-languages-v5.3.0.exe

# Method 2: Manual installation
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

