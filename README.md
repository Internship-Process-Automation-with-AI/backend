# OAMK Internship Certificate OCR System

A smart OCR (Optical Character Recognition) service for extracting text from internship certificates using PyMuPDF for fast text extraction with Tesseract OCR fallback.

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
python tests/test_ocr.py
```

## Features

- **Multi-format Support**: PDF, PNG, JPG, JPEG, TIFF, BMP
- **Smart Processing**: PyMuPDF text extraction + OCR fallback
- **Image Preprocessing**: Grayscale, noise removal, deskewing
- **Dual OCR Engines**: Tesseract + Google Vision API (optional)
- **RESTful API**: FastAPI endpoints for integration (not yet implemented)

## File Structure

## Configuration

TO BE IMPLEMENTED LATER

## Development

```bash
# Add new dependencies
pip install new-package
pip freeze > requirements.txt

# Run tests
python tests/test_ocr.py


```

## Code Quality & Best Practices

This project uses **pre-commit hooks** to ensure code quality and consistency. Pre-commit automatically runs code formatting and quality checks before each commit.

### Setup Pre-commit (Required for Contributors)

```bash
# Quick setup (recommended)
python setup_precommit.py

# Or manual setup
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### What Pre-commit Does

- **Black**: Automatically formats Python code consistently
- **isort**: Sorts and organizes imports
- **Flake8**: Checks for code style and potential errors
- **Basic checks**: Removes trailing whitespace, validates YAML files

### Usage

1. **Normal workflow**: Just commit normally - pre-commit runs automatically
2. **Manual check**: `pre-commit run` (on staged files)
3. **Format all files**: `pre-commit run --all-files`

### Why Use Pre-commit?

- ✅ **Consistent code style** across the project
- ✅ **Catches errors early** before pushing to GitHub
- ✅ **Automated formatting** saves development time
- ✅ **Team collaboration** - everyone follows the same standards
- ✅ **Professional codebase** with clean, readable code

### Troubleshooting

If pre-commit fails:
1. Check the error message
2. Fix issues manually or let hooks auto-fix them
3. Stage fixed files: `git add .`
4. Commit again

For detailed instructions, see [PRE_COMMIT_README.md](PRE_COMMIT_README.md).

---

**Note**: All contributors should set up pre-commit to maintain code quality standards.