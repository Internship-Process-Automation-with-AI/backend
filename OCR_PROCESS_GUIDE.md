# OCR Process Guide

## Overview

Our OCR system extracts text from PDFs and images using a smart, multi-engine approach with intelligent preprocessing and Finnish language optimization.

## Supported Formats
- **PDFs**: .pdf
- **DOCX Documents**: .docx
- **Images**: .png, .jpg, .jpeg, .tiff, .bmp

## Process Flow

### 1. File Input & Routing
```
File Upload → Check Extension → Route to PDF, DOCX, or Image Processor
```

### 2. PDF Processing (Two-Tier Approach)

#### Tier 1: Direct Text Extraction
- Uses **PyMuPDF** for fast text extraction
- Quality check: >30% confidence AND >25 characters
- **If successful**: Return immediately (fastest path)
- **If failed**: Move to Tier 2

#### Tier 2: OCR Processing
- Convert PDF pages to images
- Process each page with OCR
- Combine results from all pages

### 3. DOCX Processing (Direct Text Extraction)

#### Native Text Extraction
- Uses **python-docx** for direct text extraction
- Extracts text from paragraphs and tables
- **High confidence** (80%+) since DOCX contains native text
- **Fast processing** - no OCR required

**Extraction Process:**
1. Load DOCX document from bytes
2. Extract text from all paragraphs
3. Extract text from all table cells
4. Combine and return with high confidence score

### 4. Image Processing (Smart Preprocessing)

#### Raw Image First Strategy
```
Raw Image → Quality Check → Good? → Use Raw Result
                    ↓
                Not Good? → Preprocess → Compare → Pick Best
```

**Quality Thresholds:**
- Text length > 25 characters
- Confidence > 40%

**Why this approach:**
- Raw images often work perfectly for good scans
- Preprocessing can create artifacts
- Saves processing time

### 5. OCR Engine (Tesseract)

#### 14 Configuration Strategies (Finnish-Optimized):
1. `--oem 3 --psm 6 -l fin+eng` - Finnish + English (best for mixed content)
2. `--oem 3 --psm 6 -l fin` - Finnish language only
3. `--oem 3 --psm 3 -l fin+eng` - Finnish + English with auto segmentation
4. `--oem 3 --psm 4 -l fin+eng` - Finnish + English single column
5. `--oem 3 --psm 8 -l fin+eng` - Finnish + English single word
6. `--oem 3 --psm 13 -l fin+eng` - Finnish + English raw line
7. `--oem 3 --psm 6 -l eng` - English only
8. `--oem 3 --psm 3 -l eng` - English with auto segmentation
9. `--oem 3 --psm 4 -l eng` - English single column
10. `--oem 3 --psm 6` - Default: Assume uniform block of text
11. `--oem 3 --psm 3` - Fully automatic page segmentation
12. `--oem 3 --psm 4` - Assume single column of text
13. `--oem 3 --psm 8` - Single word
14. `--oem 3 --psm 13` - Raw line

**Process:** Try each config → Calculate confidence → Pick best result
**Finnish Boost:** Finnish language results get 20% confidence boost

### 6. Quality Scoring

```python
def text_quality_score(result):
    clean_text = result.text.strip().replace(" ", "").replace("\n", "")
    text_length = len(clean_text)
    
    # Prefer results with higher confidence
    confidence_bonus = result.confidence / 100.0
    
    # Penalize very low confidence results even if they have more text
    if result.confidence < 30.0:
        text_length *= 0.5  # Reduce score for low confidence
    
    # Calculate final score
    return text_length * (1 + confidence_bonus)
```

### 7. Finnish OCR Error Correction

**Post-Processing Features:**
- **Finnish Character Correction**: Fixes common OCR errors (ä→a6, ö→o6)
- **Work Certificate Terms**: Corrects employment document terminology
- **Context-Aware Corrections**: Applies corrections based on document context
- **Conservative Cleaning**: Preserves important information while removing artifacts

**Common Corrections:**
- `TYONANTAJA` → `TYÖNANTAJA`
- `TYONTEKIJA` → `TYÖNTEKIJÄ`
- `ty6ntekija` → `työntekijä`
- `a6` → `ä`, `o6` → `ö`

### 8. Google Vision Fallback

- If Tesseract fails or has low confidence
- Send to Google Vision API
- Return best available result

## Output Format

```python
OCRResult {
    text: str,              # Extracted text (Finnish-corrected)
    confidence: float,      # 0-100%
    engine: str,           # "pymupdf", "python-docx", "tesseract", "google_vision"
    processing_time: float, # Seconds
    success: bool          # Meets confidence threshold (default: 50%)
}
```

## Key Features

### Smart Decision Making
- **Raw image priority** - Fastest method first
- **Conditional preprocessing** - Only when needed
- **Quality-based selection** - Objective scoring
- **Multiple fallbacks** - Always returns something

### Processing Engines
1. **PyMuPDF** - Fast PDF text extraction
2. **python-docx** - Native DOCX text extraction
3. **Tesseract** - Primary OCR (local, free, Finnish-optimized)
4. **Google Vision** - Backup OCR (cloud, accurate)

### Finnish Language Optimization
- **Finnish-first Tesseract configs** - Prioritizes Finnish language detection
- **Confidence boosting** - Finnish results get 20% confidence boost
- **Error correction** - Post-processing fixes common Finnish OCR errors
- **Work certificate focus** - Optimized for employment documents

### Performance Benefits
- **50-70% faster** for good quality images
- **Avoids preprocessing artifacts**
- **Higher confidence scores**
- **Graceful degradation**

## Technical Implementation

### Main Entry Points
```python
# File processing
extract_text_from_file(file_path: str) -> OCRResult

# Bytes processing  
extract_text_from_bytes(file_bytes: bytes, extension: str) -> OCRResult
```

### Core Classes
- `OCRService` - Main processing engine
- `OCRResult` - Result container
- `ImagePreprocessor` - Preprocessing utilities
- `PDFConverter` - PDF handling with PyMuPDF
- `DOCXProcessor` - DOCX handling
- `FinnishOCRCorrector` - Finnish error correction

### Configuration
- `settings.OCR_CONFIDENCE_THRESHOLD` - Minimum confidence (default: 50.0)
- `settings.IMAGE_PREPROCESSING_ENABLED` - Enable/disable preprocessing
- `settings.TESSERACT_CMD` - Tesseract path
- `settings.GOOGLE_CLOUD_CREDENTIALS` - Google Vision credentials

## Usage Example

```python
# Initialize service
ocr_service = OCRService()

# Process file
result = ocr_service.extract_text_from_file("document.pdf")

# Check results
if result.success:
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}%")
    print(f"Engine: {result.engine}")
    print(f"Time: {result.processing_time}s")
```

## Best Practices

### For Development
1. **Monitor engine usage** - PyMuPDF for PDFs, Tesseract for images
2. **Check confidence scores** - 80%+ is usually good
3. **Review processing times** - Raw images should be fastest
4. **Enable logging** - Helps with debugging
5. **Test Finnish documents** - Verify error correction works

### For Production
1. **Set appropriate confidence thresholds** (default: 50%)
2. **Configure Google Vision credentials** for fallback
3. **Monitor performance metrics**
4. **Handle edge cases** (unsupported formats, errors)
5. **Verify Finnish language support** for target documents 