# OAMK Work Certificate Evaluation System

A comprehensive system for evaluating work certificates using OCR and LLM (Large Language Model) technology. The system extracts text from work certificates and uses Google's Gemini LLM to determine academic credits for practical training.

## 🚀 Quick Start

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

### 3. Configure API Keys
Create a `.env` file in the backend directory:
```env
# Required: Gemini API key for LLM evaluation
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Gemini model to use (default: gemini-1.5-flash)
GEMINI_MODEL=gemini-1.5-flash

# Optional: Google Cloud Vision API
GOOGLE_CLOUD_CREDENTIALS=path/to/credentials.json
```

### 4. Run the Pipeline
```bash
# Step 1: Extract text from documents
python test_ocr.py

# Step 2: Evaluate with LLM
python test_llm_evaluation.py
```

## 🎯 Features

- **Multi-format Support**: PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP
- **Smart OCR Processing**: PyMuPDF text extraction + Tesseract OCR fallback
- **Image Preprocessing**: Grayscale, noise removal, deskewing
- **Text Cleaning**: Automatic removal of OCR artifacts and gibberish
- **LLM Evaluation**: Google Gemini 1.5 Flash for intelligent analysis
- **Structured Output**: JSON results with evaluation metrics
- **Multi-language Support**: Finnish and English certificates
- **Academic Credit Calculation**: ECTS credits based on work hours

## 📊 Evaluation Criteria

The system evaluates work certificates for:

1. **Total Working Hours**: Calculated from employment period
2. **Nature of Tasks**: Description of work responsibilities  
3. **Training Type**: Classified as "general" or "professional"
4. **Credits Qualified**: ECTS credits (1 ECTS = 27 hours)
5. **Summary Justification**: Detailed explanation of evaluation

## 📁 File Structure

```
backend/
├── app/
│   ├── information_extraction/    # LLM evaluation logic
│   ├── ocr_model.py              # OCR processing
│   └── utils/                    # Utility functions
├── output/                       # OCR output files
├── llm_results/                  # LLM evaluation results
├── file samples/                 # Sample documents
├── test_ocr.py                   # OCR testing script
├── test_llm_evaluation.py        # LLM evaluation testing script
└── requirements.txt              # Python dependencies
```

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Gemini API key (required)
- `GEMINI_MODEL`: Gemini model to use (optional, default: gemini-1.5-flash)
- `GOOGLE_CLOUD_CREDENTIALS`: Path to Google Cloud credentials (optional)
- `TESSERACT_CMD`: Path to Tesseract executable (optional)
- `OCR_CONFIDENCE_THRESHOLD`: Minimum confidence for OCR (default: 50.0)

## 📈 Performance

- **OCR Processing**: 1-5 seconds per document
- **LLM Evaluation**: 2-10 seconds per document
- **Text Cleaning**: <1 second
- **Total Pipeline**: 3-15 seconds per document

## 🔍 Troubleshooting

### Common Issues

1. **Gemini API not available**
   - Check your `GEMINI_API_KEY` environment variable
   - Verify the API key is valid and has sufficient quota

2. **OCR quality issues**
   - Enable image preprocessing
   - Use Google Cloud Vision API for better results
   - Check document quality and resolution

3. **Text cleaning problems**
   - The system automatically cleans OCR artifacts
   - Finnish special characters are handled automatically
   - Check the cleaned text preview before LLM evaluation

## 📚 Documentation

- [LLM Evaluation Guide](LLM_EVALUATION_README.md) - Detailed LLM pipeline documentation
- [OCR Process Guide](OCR_PROCESS_GUIDE.md) - OCR processing details
- [Pre-commit Setup](PRE_COMMIT_README.md) - Code quality setup

## 🛠️ Development

```bash
# Add new dependencies
pip install new-package
pip freeze > requirements.txt

# Run tests
python test_ocr.py
python test_llm_evaluation.py
```

## 🔒 Security

- API keys are stored in environment variables
- No sensitive data is logged
- OCR processing is done locally (optional cloud fallback)
- LLM requests are sent securely to Google's servers

## 🤝 Contributing

This project uses **pre-commit hooks** to ensure code quality and consistency.

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

For detailed instructions, see [PRE_COMMIT_README.md](PRE_COMMIT_README.md).

---

**Note**: All contributors should set up pre-commit to maintain code quality standards.

