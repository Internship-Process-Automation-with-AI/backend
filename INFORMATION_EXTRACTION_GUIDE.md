# Information Extraction System Guide

## Overview

The Information Extraction system analyzes OCR text output to extract structured information from work certificates and employment documents. It supports both Finnish and English documents using rule-based extraction techniques.

## Architecture

```
information_extraction/
├── __init__.py              # Package initialization
├── models.py                # Data models (ExtractedData, ExtractionResult)
├── config.py                # Configuration (keywords, patterns, settings)
├── extractor.py             # Main extraction orchestrator
└── extractors/              # Specialized extractors
    ├── __init__.py
    ├── base_extractor.py    # Base class for all extractors
    ├── finnish_extractor.py # Finnish-specific extraction
    ├── english_extractor.py # English-specific extraction
    └── date_extractor.py    # Date parsing and validation
```

## Supported Information

### Document Information
- **Document Type**: work_certificate, employment_letter
- **Language**: finnish, english

### Employee Information
- **Employee Name**: Full name of the employee
- **Position/Job Title**: Role or position held
- **Job Description**: Responsibilities and duties

### Employer Information
- **Employer/Company Name**: Name of the employing organization
- **Company Details**: Additional company information

### Employment Dates
- **Start Date**: Employment start date
- **End Date**: Employment end date
- **Work Period**: Calculated duration (e.g., "2 years 3 months")

## Supported Languages

### Finnish (Primary)
- **Keywords**: työtodistus, työntekijä, työnantaja, tehtävä
- **Date Format**: DD.MM.YYYY (15.01.2023)
- **Company Suffixes**: Oy, Ab, Ltd, Ky, Tmi, Yhtiö

### English (Secondary)
- **Keywords**: work certificate, employee, employer, position
- **Date Format**: MM/DD/YYYY, YYYY-MM-DD
- **Company Suffixes**: Ltd, Inc, Corp, LLC, Company

## Usage

### Basic Usage
```python
from app.information_extraction import InformationExtractor

# Initialize extractor
extractor = InformationExtractor()

# Extract information from OCR text
result = extractor.extract_information(ocr_text)

# Check results
if result.success:
    data = result.extracted_data
    print(f"Employee: {data.employee_name}")
    print(f"Position: {data.position}")
    print(f"Employer: {data.employer}")
    print(f"Period: {data.work_period}")
    print(f"Confidence: {result.overall_confidence:.2f}")
```

### Integration with OCR System
```python
from app.ocr_model import OCRService
from app.information_extraction import InformationExtractor

# Process document with OCR
ocr_service = OCRService()
ocr_result = ocr_service.extract_text_from_file("document.pdf")

# Extract structured information
if ocr_result.success:
    extractor = InformationExtractor()
    extraction_result = extractor.extract_information(ocr_result.text)
    
    # Use extracted data
    if extraction_result.success:
        print("Extracted information:", extraction_result.extracted_data.to_dict())
```

## Configuration

### Keywords and Patterns
Edit `config.py` to customize:
- **Document type keywords** for different languages
- **Employee/employer patterns** for name extraction
- **Date patterns** for various date formats
- **Confidence thresholds** for quality control

### Settings
```python
EXTRACTION_SETTINGS = {
    "max_text_length": 10000,  # Maximum text to process
    "min_confidence": 0.2,     # Minimum confidence threshold
    "language_detection": True, # Enable language detection
    "enable_fallback": True    # Enable fallback mechanisms
}
```

## Output Format

### ExtractedData
```python
{
    "document_type": "work_certificate",
    "language": "finnish",
    "employee_name": "Matti Meikäläinen",
    "position": "Software Developer",
    "employer": "Tech Company Oy",
    "start_date": "2023-01-15",
    "end_date": "2023-12-31",
    "work_period": "11 months 16 days",
    "description": "Kehitti web-sovelluksia...",
    "confidence_scores": {
        "employee_name": 0.8,
        "position": 0.7,
        "employer": 0.9,
        "start_date": 0.8,
        "end_date": 0.8
    }
}
```

### ExtractionResult
```python
{
    "success": true,
    "extracted_data": {...},
    "overall_confidence": 0.8,
    "processing_time": 0.15,
    "engine": "information_extractor",
    "errors": []
}
```

## Testing

### Run Tests
```bash
# Run all information extraction tests
python -m pytest tests/test_information_extraction.py -v

# Run specific test
python -m pytest tests/test_information_extraction.py::TestInformationExtraction::test_finnish_work_certificate_extraction -v
```

### Test Coverage
- Finnish work certificate extraction
- English work certificate extraction
- Language detection
- Date parsing and validation
- Error handling
- Configuration validation

## Performance

### Typical Performance
- **Processing Time**: 0.1-0.5 seconds per document
- **Memory Usage**: Low (rule-based, no ML models)
- **Accuracy**: 70-90% depending on document quality
- **Confidence Scores**: 0.0-1.0 for each extracted field

### Optimization Tips
1. **Preprocess OCR text** for better extraction
2. **Adjust confidence thresholds** based on your needs
3. **Add custom patterns** for specific document formats
4. **Use language hints** when known

## Extending the System

### Adding New Languages
1. Create new extractor class inheriting from `BaseExtractor`
2. Add language keywords to `config.py`
3. Update language detection in main extractor
4. Add tests for new language

### Adding New Fields
1. Update `ExtractedData` model
2. Add extraction method to base extractor
3. Implement in language-specific extractors
4. Update confidence calculation

### Custom Patterns
1. Add patterns to `config.py`
2. Update extractor regex patterns
3. Test with sample documents
4. Adjust confidence scoring

## Troubleshooting

### Common Issues

**Low Confidence Scores**
- Check OCR text quality
- Verify language detection
- Review keyword patterns
- Adjust confidence thresholds

**Missing Information**
- Add missing keywords to config
- Improve regex patterns
- Check document format variations
- Enable fallback mechanisms

**Date Parsing Errors**
- Verify date format patterns
- Check for non-standard formats
- Update date parsing logic
- Add format-specific handlers

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run extraction with debug output
result = extractor.extract_information(text)
```

## Future Enhancements

### Planned Features
- **Machine Learning Integration**: NER models for better accuracy
- **More Languages**: Swedish, German, French support
- **Document Classification**: Automatic document type detection
- **Validation Rules**: Cross-field validation and consistency checks
- **API Integration**: REST API for external access

### AI Integration Path
1. **Current**: Rule-based extraction (fast, reliable)
2. **Phase 2**: Named Entity Recognition (spaCy)
3. **Phase 3**: Custom ML models (fine-tuned)
4. **Phase 4**: Hybrid approach (best of both)

## Best Practices

### For Development
1. **Test with real documents** from your domain
2. **Monitor confidence scores** for quality control
3. **Log extraction failures** for pattern improvement
4. **Validate extracted data** before using in production

### For Production
1. **Set appropriate confidence thresholds**
2. **Implement error handling** for edge cases
3. **Monitor performance metrics**
4. **Regular pattern updates** based on new documents 