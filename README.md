# OAMK AI-Powered Academic Credit Evaluation Backend

## 🚀 Quick Start

1. **Clone & Setup**
   ```bash
   cd backend
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   # Required: Gemini API Key
   export GEMINI_API_KEY=your_gemini_api_key
   # Windows: set GEMINI_API_KEY=your_gemini_api_key
   
   # Database Configuration (defaults shown)
   export DATABASE_HOST=localhost
   export DATABASE_PORT=5432
   export DATABASE_NAME=oamk_certificates
   export DATABASE_USER=your_username
   export DATABASE_PASSWORD=your_password
   ```

3. **Initialize Database**
   ```bash
   python -m src.database.init_db
   ```

4. **Run FastAPI Server**
   ```bash
   uvicorn src.API.main:app --reload
   # API Documentation: http://127.0.0.1:8000/docs
   ```

5. **Frontend**
   See `../frontend/README.md` for UI setup and usage.

---

## 📝 What is this?
This backend processes work certificates for OAMK students, using OCR and AI to evaluate and assign ECTS credits for practical training. It exposes a FastAPI server and stores **all data in PostgreSQL** (including file content and AI workflow results).

---

## ✨ Key Features
- **Database-First Architecture**: All files and results stored in PostgreSQL
- **Multi-Language OCR**: Support for English and Finnish documents with auto-detection
- **4-Stage AI Pipeline**: Extraction → Evaluation → Validation → Correction
- **Company Validation**: AI-powered company legitimacy checking and verification
- **Degree-Specific Rules**: Tailored evaluation for different degree programs
- **Training Type Classification**: General (max 10 ECTS) vs Professional (max 30 ECTS)
- **Reviewer Workflow**: Complete review system with student comments and appeals
- **Cross-Device Access**: No file system dependencies
- **Real-time Processing**: Live status updates and progress tracking

---

## 🗂️ System Architecture

### **Data Flow:**
```
Frontend → API → Database Storage
    ↓
OCR Processing (Multi-language with Finnish optimization)
    ↓
LLM Evaluation (4-Stage Pipeline + Company Validation)
    ↓
Database Storage (Complete Results + Company Validation)
    ↓
Frontend Display (Rich UI with justification views)
```

### **Key Components:**
- **API Layer** (`src/API/api.py`): Pure data transport, no business logic
- **LLM Orchestrator** (`src/workflow/ai_workflow.py`): All decision-making logic
- **OCR Workflow** (`src/workflow/ocr_workflow.py`): Multi-language text extraction with Finnish optimization
- **Database** (`src/database/`): PostgreSQL with file content and AI results storage
- **Company Validation**: AI-powered business legitimacy checking

---

## 🔗 Complete API Endpoints

### **Student Operations:**
| Method | Path                                         | Description                       |
|--------|----------------------------------------------|-----------------------------------|
| POST   | `/student/{student_id}/upload-certificate`   | Upload certificate (stored in DB) |
| GET    | `/student/{email}/applications`              | List student's applications       |
| GET    | `/student/{email}`                           | Get student information           |

### **Certificate Processing:**
| Method | Path                                         | Description                       |
|--------|----------------------------------------------|-----------------------------------|
| POST   | `/certificate/{certificate_id}/process`      | Run OCR + AI evaluation           |
| GET    | `/certificate/{certificate_id}/status`       | Get processing status             |
| GET    | `/certificate/{certificate_id}/details`      | Get detailed results              |
| GET    | `/certificate/{certificate_id}/preview`      | Preview certificate from DB       |
| GET    | `/certificate/{certificate_id}`              | Download certificate from DB      |

### **Review & Feedback:**
| Method | Path                                         | Description                       |
|--------|----------------------------------------------|-----------------------------------|
| GET    | `/reviewers`                                 | List all reviewers                |
| POST   | `/certificate/{certificate_id}/review`       | Reviewer submits decision         |
| POST   | `/certificate/{certificate_id}/student-comment` | Student submits comment for rejected cases |
| POST   | `/certificate/{certificate_id}/feedback`     | Student feedback submission       |
| POST   | `/certificate/{certificate_id}/appeal`       | Student appeal submission         |
| POST   | `/certificate/{certificate_id}/appeal-review` | Appeal review decision            |

---

## 🗄️ Enhanced Database Schema

### **Key Tables:**
- **`students`**: Student information with email validation (@students.oamk.fi)
- **`certificates`**: File storage with `file_content BYTEA` and metadata
- **`decisions`**: AI evaluation results with complete workflow data
- **`reviewers`**: Reviewer information with position and department

### **Decision Table Fields:**
```sql
-- Core decision fields
ai_decision: ACCEPTED/REJECTED
ai_justification: Detailed reasoning for decision
total_working_hours: Calculated working hours
credits_awarded: ECTS credits assigned

-- Training details
training_duration: Duration of training
training_institution: Company/organization name
degree_relevance: Relevance to student's degree

-- Company validation (NEW)
company_validation_status: LEGITIMATE/UNVERIFIED/SUSPICIOUS
company_validation_justification: JSON with validation details

-- Evidence and recommendations
supporting_evidence: Evidence supporting the decision
challenging_evidence: Evidence challenging the decision
recommendation: Guidance for rejected cases

-- Complete workflow data
ai_workflow_json: Full AI pipeline results as JSON
```

### **Storage Strategy:**
- **File Content**: Stored as `BYTEA` in database (no file system)
- **AI Results**: Complete workflow JSON stored as `TEXT`
- **Company Validation**: Detailed validation results with evidence
- **Cross-Device**: All data accessible from any server/device

---

## 🤖 Enhanced AI Pipeline (4-Stage Process + Company Validation)

### **Stage 1: Information Extraction**
- Extract employee details, employment periods, responsibilities
- Structural validation of extracted data using Pydantic models
- Company information extraction for validation

### **Stage 2: Academic Evaluation**
- Calculate working hours and ECTS credits
- Evaluate degree relevance and training type
- Generate decision (ACCEPTED/REJECTED) with justification
- **Company Validation**: Check company legitimacy and business registration

### **Stage 3: Validation**
- Cross-check extraction and evaluation against original document
- Identify any inaccuracies or missing information
- Validate company information and business legitimacy

### **Stage 4: Correction (if needed)**
- Fix identified issues while preserving valid calculations
- Maintain business rules (credit limits, decision logic)
- Refine company validation results

### **Company Validation Features:**
- **Business Legitimacy Check**: Verify company existence and registration
- **Risk Assessment**: Evaluate potential risks and suspicious patterns
- **Evidence Collection**: Gather supporting evidence from multiple sources
- **Confidence Scoring**: Provide confidence levels for validation results

---

## 🌐 Enhanced Multi-Language Support

### **OCR Languages:**
- **English**: Default OCR processing with high accuracy
- **Finnish**: Auto-detection with Finnish language pack optimization
- **Auto-Detection**: Content-based language identification

### **Finnish Language Features:**
- **Character Recognition**: Optimized for Finnish characters (ä, ö, å)
- **Business Terminology**: Recognizes Finnish work certificate terms
- **Document Patterns**: Understands Finnish document layouts
- **Language Pack**: Uses Tesseract's Finnish language pack (`fin`)

### **Language Detection Process:**
- Finnish character detection (äöå)
- Finnish keyword recognition (työtodistus, harjoittelu, etc.)
- Document structure analysis
- Fallback to English if detection fails

---

## 🏢 Company Validation System

### **What It Does:**
- **Legitimacy Check**: Verifies if companies actually exist
- **Business Registration**: Checks business registry information
- **Risk Assessment**: Identifies potentially suspicious companies
- **Evidence Collection**: Gathers supporting documentation

### **Validation Results:**
```json
{
  "company_validation_status": "LEGITIMATE",
  "company_validation_justification": {
    "name": "Company Name",
    "status": "LEGITIMATE",
    "confidence": "high",
    "risk_level": "low",
    "justification": "Company verified through business registry...",
    "supporting_evidence": [
      "Business registry match found",
      "Website verification successful",
      "Industry information confirmed"
    ]
  }
}
```

### **Status Types:**
- **LEGITIMATE**: Company verified and legitimate
- **UNVERIFIED**: Company exists but limited verification
- **SUSPICIOUS**: Potential issues or risks identified

---

## 📊 Enhanced Decision Logic

### **Training Type Rules:**
- **General Training**: Maximum 10 ECTS credits
- **Professional Training**: Maximum 30 ECTS credits
- **Credit Calculation**: 1 ECTS = 27 hours of work

### **Decision Criteria:**
- **ACCEPTED**: Work experience meets requested training type criteria
- **REJECTED**: Work experience doesn't meet requested training type criteria
- **Justification**: Detailed reasoning for decision
- **Company Validation**: Company legitimacy affects decision confidence
- **Student Comments**: Students can provide comments for rejected cases

### **Company Validation Impact:**
- **LEGITIMATE companies**: Higher confidence in decisions
- **UNVERIFIED companies**: Standard confidence with verification notes
- **SUSPICIOUS companies**: Lower confidence, may require manual review

---

## 🔧 Development & Testing

### **Code Quality:**
- **Pre-commit hooks**: Ruff linting and formatting
- **Type hints**: Full type annotation coverage
- **Error handling**: Comprehensive exception management

### **Testing:**
- **Sample files**: Available in `samples/` directory
- **Database seeding**: Sample students and reviewers on init
- **API documentation**: Interactive Swagger UI at `/docs`
- **OCR testing**: Test with Finnish and English documents

### **Development Tools:**
```bash
# Test OCR workflow
python -m src.workflow.ocr_workflow

# Test AI workflow
python -m src.workflow.ai_workflow

# Run legacy CLI for testing (development only)
python -m src.mainpipeline
```

---

## 📄 More Information
- **Database Schema**: See `src/database/models.py` for current data models
- **API Documentation**: Interactive docs at `http://127.0.0.1:8000/docs`
- **Frontend**: See `../frontend/README.md`
- **AI Workflow**: See `AI_WORKFLOW_README.md`
- **OCR Process**: See `OCR_PROCESS_GUIDE.md`
- **Finnish OCR**: See `FINNISH_OCR_GUIDE.md`

---

**Note**: This system is designed for academic use at OAMK for evaluating work certificates and determining practical training credits. All evaluations follow Finnish higher education standards and ECTS credit system, with enhanced company validation for improved decision confidence.
