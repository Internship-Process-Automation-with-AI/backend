# AI Workflow Documentation

## Quick Start - Running the Pipeline

### Prerequisites
1. **Python Environment**: Python 3.8+ with virtual environment
2. **API Key**: Gemini API key for LLM processing
3. **Sample Documents**: Place test files in `samples/` directory

### Setup Steps
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables on .env file
export GEMINI_API_KEY="your_gemini_api_key_here"
# On Windows: set GEMINI_API_KEY=your_gemini_api_key_here

# 5. Run the main pipeline
python -m src.mainpipeline
```

### What Happens When You Run It
1. **File Selection**: Choose from available documents in `samples/`
2. **Degree Selection**: Pick your degree program from the list
3. **Training Type Selection**: Choose between "general" or "professional" training
4. **Email Entry**: Enter your OAMK student email (@students.oamk.fi)
5. **Processing**: Watch the 4-stage pipeline process your document
6. **Results**: View decision (ACCEPTED/REJECTED), justification, and recommendation (if rejected)

### Sample Output Location
Results are saved in `processedData/[document_name]/` directory:
- `ocr_output_[document].txt` - Extracted text
- `aiworkflow_output_[document]_[timestamp].json` - Complete results

---

## Overview

This document explains the AI-powered document processing workflow for evaluating work certificates and determining academic credits. The system uses a **4-stage LLM pipeline** combined with OCR processing to extract, evaluate, validate, and correct information from work certificates.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │    │   OCR Service   │    │  LLM Pipeline   │
│                 │    │                 │    │                 │
│ • Document File │───▶│ • Text Extraction│───▶│ • Extraction    │
│ • Degree Program│    │ • Image/PDF/DOCX│    │ • Evaluation    │
│ • Training Type │    │ • Confidence    │    │ • Validation    │
│ • Student Email │    │                 │    │ • Correction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Output JSON   │
                                              │                 │
                                              │ • Decision      │
                                              │ • Justification │
                                              │ • Recommendation│
                                              │ • Credits       │
                                              └─────────────────┘
```

## Main Pipeline Flow (`mainpipeline.py`)

### 1. **Initialization Phase**
```python
pipeline = DocumentPipeline()
pipeline.initialize_services()
```
- **OCR Service**: Initializes text extraction from documents
- **LLM Orchestrator**: Sets up Gemini AI model with fallback support
- **Degree Evaluator**: Loads degree-specific evaluation criteria

### 2. **User Input Collection**
```python
# File Selection
selected_file = pipeline.list_sample_files()  # Lists available documents

# Degree Selection  
student_degree = pipeline.select_degree_program()  # User chooses degree

# Training Type Selection
requested_training_type = pipeline.get_training_type()  # User chooses training type

# Email Collection
student_email = pipeline.get_student_email()  # OAMK student email
```

### 3. **Document Processing Pipeline**

#### **Step 1: OCR Processing**
```python
ocr_result = self.ocr_service.extract_text_from_file(file_path)
```
- **Input**: PDF, DOCX, or image files
- **Output**: Raw text with confidence scores
- **Engines**: python-docx, Tesseract OCR, or other OCR engines
- **Validation**: Ensures text was successfully extracted

#### **Step 2: Text Cleaning**
```python
cleaned_text = self.clean_ocr_text(ocr_result.text)
```
- Removes empty lines and whitespace
- Filters out very short lines (< 3 characters)
- Prepares text for LLM processing

#### **Step 3: LLM Processing (4-Stage Pipeline)**
```python
llm_results = self.orchestrator.process_work_certificate(
    cleaned_text, student_degree, requested_training_type
)
```

## LLM Pipeline Stages (`ai_workflow.py`)

### **Stage 1: Information Extraction**
**Purpose**: Extract structured data from unstructured text

**Input**: Raw document text
**Output**: Structured JSON with employee info, positions, dates, responsibilities

**Validation**: Pydantic models ensure data structure integrity
```json
{
  "employee_name": "John Doe",
  "employer": "Tech Corp",
  "positions": [
    {
      "title": "Software Developer",
      "start_date": "2023-01-15",
      "end_date": "2023-06-30",
      "responsibilities": "Developed web applications..."
    }
  ]
}
```

### **Stage 2: Academic Evaluation**
**Purpose**: Determine academic credits and training classification based on student's requested training type

**Input**: Extracted information + student degree + requested training type
**Output**: Credit calculation, decision (ACCEPTED/REJECTED), and justification

**Degree-Specific Guidelines**: Uses degree-specific criteria from `degree_programs_data.py`
```json
{
  "total_working_hours": 1040,
  "requested_training_type": "professional",
  "credits_calculated": 38.0,
  "credits_qualified": 30.0,
  "degree_relevance": "high",
  "relevance_explanation": "Work directly related to degree...",
  "decision": "ACCEPTED",
  "justification": "The work experience meets the criteria for professional training..."
}
```

### **Stage 3: Validation**
**Purpose**: Cross-check LLM output against original document

**Input**: Original text + extraction results + evaluation results
**Output**: Validation report with accuracy score and issues found

**Validation Checks**:
- Factual accuracy (names, dates, companies)
- Logical consistency (decision vs degree relevance)
- Calculation accuracy (hours, credits)
- Information completeness
- Decision justification quality

### **Stage 4: Correction**
**Purpose**: Fix identified issues automatically

**Input**: Validation results + original LLM output + student degree + requested training type
**Output**: Corrected results with explanation of changes

**Correction Types**:
- Fix factual errors (wrong company names, dates)
- Correct logical inconsistencies
- Adjust credit calculations
- Update decision justifications
- Refine recommendations (for rejected cases only)

## Data Validation

### **Structural Validation (Pydantic Models)**
Located in `models.py`, these models ensure data integrity:

```python
class ExtractionResults(BaseModel):
    employee_name: str
    positions: List[Position]
    # ... with validators for dates, business rules

class EvaluationResults(BaseModel):
    decision: str = Field(..., pattern="^(ACCEPTED|REJECTED)$")
    justification: str
    recommendation: Optional[str] = None
    credits_qualified: float
    # ... with validators for credit limits, consistency
```

### **Business Rules Validation**
- **Credit Limits**: General training max 10 ECTS, Professional max 30 ECTS
- **Decision Logic**: Based on evidence analysis and requested training type
- **Recommendation Rules**: Only provided for REJECTED decisions
- **Date Validation**: No future dates, end after start dates
- **Timeline Consistency**: No overlapping employment periods
- **Training Type Consistency**: Validates degree relevance vs training type consistency

## Decision and Recommendation System

### **Decision Logic**
The system evaluates work experience against the student's requested training type:

- **ACCEPTED**: Work experience meets criteria for the requested training type
- **REJECTED**: Work experience does not meet criteria for the requested training type

### **Recommendation System**
- **For ACCEPTED cases**: No recommendation needed (student's request is approved)
- **For REJECTED cases**: Provides actionable guidance to apply for general training instead

### **CLI Display Logic**
```bash
🎯 DECISION: ACCEPTED (Student receives 30.0 ECTS as PROFESSIONAL training)
📋 JUSTIFICATION: The work experience meets the criteria for professional training...

# OR for rejected cases:
🎯 DECISION: REJECTED
📋 JUSTIFICATION: The work experience does not meet the criteria for professional training...
💡 RECOMMENDATION: Apply this work experience as general training. The experience provides valuable transferable skills but does not meet the criteria for professional training in this degree program.
```

### **Frontend Display**
The frontend also shows the decision, justification, and recommendation (for rejected cases) in a user-friendly interface with:
- Document information and degree program
- Calculated credits and training type
- Decision status with color coding (green for accepted, red for rejected)
- Justification text
- Recommendation section (only for rejected cases)

## Degree-Specific Evaluation

### **Degree Programs Data** (`degree_programs_data.py`)
Each degree has specific evaluation criteria:

```python
"bachelor_of_health_care_nursing": {
    "name": "Bachelor of Health Care, Nursing",
    "relevant_industries": ["healthcare", "medical", "public health"],
    "relevant_roles": ["nurse", "patient care", "health promotion"]
}
```

### **Evaluation Guidelines**
Generated dynamically for each degree:
- Relevant industries and roles
- Decision criteria for requested training types
- Credit calculation rules
- Evidence analysis framework
- Recommendation guidelines for rejected cases

## Output Structure

### **Complete Pipeline Results**
```json
{
  "success": true,
  "file_path": "samples/document.pdf",
  "student_degree": "Bachelor of Health Care, Nursing",
  "student_email": "student@students.oamk.fi",
  "processing_time": 15.2,
  "ocr_results": {
    "success": true,
    "engine": "tesseract",
    "confidence": 95.5,
    "text_length": 1250
  },
  "llm_results": {
    "extraction_results": { /* structured employee data */ },
    "evaluation_results": { 
      "decision": "ACCEPTED|REJECTED",
      "justification": "Clear reasoning for decision",
      "recommendation": "Optional guidance for rejected cases",
      "credits_calculated": 38.0,
      "credits_qualified": 30.0
    },
    "validation_results": { /* validation report */ },
    "correction_results": { /* corrected data if needed */ },
    "structural_validation": { /* Pydantic validation results */ }
  }
}
```

## Key Components

### **Core Classes**
- **`DocumentPipeline`**: Main orchestration class with CLI display
- **`LLMOrchestrator`**: 4-stage LLM processing with decision logic
- **`DegreeEvaluator`**: Degree-specific evaluation logic
- **`OCRService`**: Text extraction from documents

### **Prompt System** (`prompts/`)
- **`extraction.py`**: Information extraction prompts
- **`evaluation.py`**: Academic evaluation with decision/justification prompts
- **`validation.py`**: Cross-checking prompts
- **`correction.py`**: Error correction prompts

### **Validation System** (`models.py`)
- **Pydantic Models**: Structural validation with decision/recommendation fields
- **Business Rules**: Credit limits, decision logic, recommendation rules
- **Data Integrity**: Date validation, timeline consistency

## Error Handling

### **Graceful Degradation**
- **OCR Failures**: Fallback to different engines
- **LLM Failures**: Model fallback (Gemini → fallback models)
- **Validation Failures**: Continue with warnings
- **Correction Failures**: Use original results

### **Logging and Monitoring**
- Detailed error messages
- Processing time tracking
- Confidence scores
- Validation issue reporting

## Usage Examples

### **Running the Pipeline**
```bash
cd backend
python src/mainpipeline.py
```

### **Testing Individual Components**
```bash
# Test OCR only
python src/test_ocr.py

# Test LLM processing only
python src/test_extractor.py
```

## Configuration

### **Environment Variables**
- `GEMINI_API_KEY`: Required for LLM processing
- `GEMINI_MODEL`: Primary model (default: gemini-2.0-flash)
- `GEMINI_FALLBACK_MODELS`: Backup models

### **Sample Files**
Place test documents in `samples/` directory:
- PDF files
- DOCX files
- Image files (PNG, JPG, etc.)

## Database Integration

The output JSON is structured for easy database storage:
- **Student Information**: Email, degree program
- **Document Information**: File path, processing metadata
- **Evaluation Results**: Decision, justification, recommendation, credits
- **Validation Status**: Accuracy scores, correction history

## Troubleshooting

### **Common Issues**
1. **OCR Failures**: Check file format and quality
2. **LLM API Errors**: Verify API key and quota
3. **Validation Errors**: Review business rules and data consistency
4. **Degree Not Found**: Check degree program spelling

### **Debug Mode**
Enable detailed logging to trace processing steps:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### **Planned Improvements**
- Multi-language support (Finnish/English)
- Additional document formats
- Real-time processing API
- Database integration
- User interface improvements

### **Extensibility**
The modular design allows easy addition of:
- New degree programs
- Additional validation rules
- Different LLM providers
- Custom evaluation criteria

---

**Note**: This system is designed for academic use at OAMK for evaluating work certificates and determining practical training credits. All evaluations follow Finnish higher education standards and ECTS credit system. 