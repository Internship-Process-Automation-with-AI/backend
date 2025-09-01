# AI Workflow Documentation

## Quick Start - Running the AI Workflow

### Prerequisites
1. **Python Environment**: Python 3.8+ with virtual environment
2. **API Key**: Gemini API key for LLM processing
3. **Database**: PostgreSQL database with initialized schema
4. **Sample Documents**: Place test files in `samples/` directory

### Setup Steps
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
export GEMINI_API_KEY="your_gemini_api_key_here"
# On Windows: set GEMINI_API_KEY=your_gemini_api_key_here

# 5. Set up database environment variables
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=oamk_certificates
export DATABASE_USER=your_username
export DATABASE_PASSWORD=your_password

# 6. Initialize database
python -m src.database.init_db

# 7. Start the FastAPI application (Primary Usage)
uvicorn src.API.main:app --reload

# 8. Alternative: Run legacy CLI pipeline for testing (Development Only)
python -m src.mainpipeline
```

---

## Overview

This document explains the AI-powered document processing workflow for evaluating work certificates and determining academic credits. The system uses a **4-stage LLM pipeline** combined with OCR processing to extract, evaluate, validate, and correct information from work certificates.

## Current System Architecture

The AI workflow system is now primarily accessed through the **FastAPI application**, not the CLI pipeline. Here's the current architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │  AI Workflow    │
│                 │    │   Application   │    │                 │
│ • Upload        │───▶│ • API Endpoints │───▶│ • OCR Service   │
│ • Process       │    │ • File Storage  │    │ • LLM Pipeline  │
│ • View Results  │    │ • Database      │    │ • 4-Stage Proc. │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   Database      │
                                              │                 │
                                              │ • File Storage  │
                                              │ • AI Results    │
                                              │ • Decisions     │
                                              └─────────────────┘
```

## Primary Usage: FastAPI Application

### **Main Application Flow**
1. **File Upload**: Frontend uploads certificate to `/student/{student_id}/upload-certificate`
2. **Processing Request**: Frontend calls `/certificate/{certificate_id}/process`
3. **AI Workflow Execution**: `ai_workflow.py` processes the document through 4-stage pipeline
4. **Database Storage**: Results stored in PostgreSQL database
5. **Frontend Display**: Results retrieved and displayed through API endpoints

### **API Endpoints Using AI Workflow**
- **`POST /certificate/{certificate_id}/process`**: Main endpoint that triggers the AI workflow
- **`GET /certificate/{certificate_id}/preview`**: View AI workflow results
- **`GET /student/{email}/applications`**: List all processed applications

## Legacy CLI Pipeline (Development/Testing Only)

The `mainpipeline.py` was used during development before the full application was built. It's now primarily useful for:

### **Development and Testing**
```bash
# Test the AI workflow with sample documents
python -m src.mainpipeline

# Test individual components
python -m src.workflow.ai_workflow
python -m src.workflow.ocr_workflow
```

### **What mainpipeline.py Does**
- **CLI Interface**: Command-line interface for testing
- **Sample File Processing**: Processes documents from `samples/` directory
- **Local Output**: Saves results to `processedData/` directory
- **Development Tool**: Useful for debugging and testing AI workflow components

### **When to Use mainpipeline.py**
- **Development**: Testing new AI workflow features
- **Debugging**: Troubleshooting OCR or LLM issues
- **Sample Processing**: Processing test documents locally
- **Component Testing**: Testing individual workflow components

## AI Workflow Core (`ai_workflow.py`)

### **Primary Workflow Class**
The `LLMOrchestrator` class in `ai_workflow.py` is the main engine that:

```python
class LLMOrchestrator:
    """Orchestrates LLM-based work certificate processing using a 4-stage approach."""
    
    def process_work_certificate(self, text, student_degree, requested_training_type):
        """Main method called by the FastAPI application."""
        # Stage 1: Information Extraction
        # Stage 2: Academic Evaluation  
        # Stage 3: Validation
        # Stage 4: Correction
```

### **Integration with FastAPI**
```python
# In api.py - the main API endpoint
@router.post("/certificate/{certificate_id}/process")
async def process_certificate(certificate_id: UUID):
    # 1. Get certificate from database
    # 2. Run OCR processing
    # 3. Call AI workflow
    llm_results = orchestrator.process_work_certificate(
        cleaned_text, student_degree, requested_training_type
    )
    # 4. Store results in database
    # 5. Return processing status
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

### **API Response Format**
```json
{
  "success": true,
  "processing_time": 15.2,
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

## Database Integration

### **Storage Strategy**
The system stores all data in PostgreSQL:
- **File Content**: Stored as `BYTEA` in the `certificates` table
- **AI Results**: Complete workflow JSON stored as `TEXT` in the `decisions` table
- **Student Information**: Stored in the `students` table with email validation
- **Reviewer Information**: Stored in the `reviewers` table

### **Key Database Tables**
```sql
-- Students table with email validation
CREATE TABLE students (
    student_id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL CHECK (email ~ '^[A-Za-z0-9._%+-]+@students\.oamk\.fi$'),
    degree VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255)
);

-- Certificates table with file storage
CREATE TABLE certificates (
    certificate_id UUID PRIMARY KEY,
    student_id UUID REFERENCES students(student_id),
    training_type training_type NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_content BYTEA, -- Actual file content stored in database
    ocr_output TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Decisions table with AI workflow results
CREATE TABLE decisions (
    decision_id UUID PRIMARY KEY,
    certificate_id UUID REFERENCES certificates(certificate_id),
    ai_decision decision_status NOT NULL,
    ai_justification TEXT NOT NULL,
    ai_workflow_json TEXT, -- Complete AI workflow output
    total_working_hours INTEGER,
    credits_awarded INTEGER,
    student_comment TEXT, -- Student comments for rejected cases
    reviewer_decision reviewer_decision, -- PASS/FAIL from human reviewer
    reviewer_comment TEXT
);
```

## Key Components

### **Core Classes**
- **`LLMOrchestrator`**: Main AI workflow class used by FastAPI application
- **`DocumentPipeline`**: Legacy CLI interface (development/testing only)
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

### **Primary Usage: FastAPI Application**
```bash
# Start the application
cd backend
uvicorn src.API.main:app --reload

# Use the API endpoints
# POST /certificate/{certificate_id}/process
# GET /certificate/{certificate_id}/preview
```

### **Development/Testing: Legacy CLI**
```bash
# Test the AI workflow locally
cd backend
python src/mainpipeline.py

# Test individual components
python -c "from src.workflow.ai_workflow import LLMOrchestrator; print('AI Workflow available')"
```

### **Testing Individual Components**
```bash
# Test OCR only
python -m src.workflow.ocr_workflow

# Test LLM processing only
python -c "from src.workflow.ai_workflow import LLMOrchestrator; o = LLMOrchestrator(); print('LLM Orchestrator initialized')"
```

## Configuration

### **Environment Variables**
- `GEMINI_API_KEY`: Required for LLM processing
- `GEMINI_MODEL`: Primary model (default: gemini-2.0-flash)
- `GEMINI_FALLBACK_MODELS`: Backup models
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD`: Database connection

### **Sample Files**
Place test documents in `samples/` directory:
- PDF files
- DOCX files
- Image files (PNG, JPG, etc.)

## API Integration

The system provides FastAPI endpoints for integration:
- **File Upload**: `/student/{student_id}/upload-certificate`
- **Processing**: `/certificate/{certificate_id}/process`
- **Results**: `/certificate/{certificate_id}/preview`
- **Student Applications**: `/student/{email}/applications`
- **Reviewer Decisions**: `/certificate/{certificate_id}/review`

## Troubleshooting

### **Common Issues**
1. **OCR Failures**: Check file format and quality
2. **LLM API Errors**: Verify API key and quota
3. **Validation Errors**: Review business rules and data consistency
4. **Degree Not Found**: Check degree program spelling
5. **Database Connection**: Verify database credentials and connection

### **Debug Mode**
Enable detailed logging to trace processing steps:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### **Planned Improvements**
- Enhanced multi-language support (Finnish/English)
- Additional document formats
- Real-time processing API improvements
- Advanced database analytics
- User interface improvements

### **Extensibility**
The modular design allows easy addition of:
- New degree programs
- Additional validation rules
- Different LLM providers
- Custom evaluation criteria

---

**Note**: This system is designed for academic use at OAMK for evaluating work certificates and determining practical training credits. All evaluations follow Finnish higher education standards and ECTS credit system.

**Important**: The primary workflow is now through the FastAPI application (`ai_workflow.py`). The `mainpipeline.py` is a legacy CLI tool used primarily for development and testing purposes. 