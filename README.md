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
- **Multi-Language OCR**: Support for English and Finnish documents
- **4-Stage AI Pipeline**: Extraction → Evaluation → Validation → Correction
- **Degree-Specific Rules**: Tailored evaluation for different degree programs
- **Training Type Classification**: General (max 10 ECTS) vs Professional (max 30 ECTS)
- **Reviewer Workflow**: Complete review system with appeals
- **Cross-Device Access**: No file system dependencies

---

## 🗂️ System Architecture

### **Data Flow:**
```
Frontend → API → Database Storage
    ↓
OCR Processing (Temporary Files)
    ↓
LLM Evaluation (4-Stage Pipeline)
    ↓
Database Storage (Complete Results)
    ↓
Frontend Display
```

### **Key Components:**
- **API Layer** (`src/API/api.py`): Pure data transport, no business logic
- **LLM Orchestrator** (`src/workflow/ai_workflow.py`): All decision-making logic
- **OCR Workflow** (`src/workflow/ocr_workflow.py`): Multi-language text extraction
- **Database** (`src/database/`): PostgreSQL with file content and AI results storage

---

## 🔗 Main API Endpoints
| Method | Path                                         | Description                       |
|--------|----------------------------------------------|-----------------------------------|
| POST   | `/student/{student_id}/upload-certificate`   | Upload certificate (stored in DB) |
| POST   | `/certificate/{certificate_id}/process`      | Run OCR + AI evaluation           |
| GET    | `/student/{email}/applications`              | List student's applications       |
| GET    | `/certificate/{certificate_id}`              | Download certificate from DB      |
| GET    | `/certificate/{certificate_id}/preview`      | Preview certificate from DB       |
| GET    | `/reviewers`                                 | List all reviewers                |
| POST   | `/certificate/{certificate_id}/review`       | Reviewer submits decision         |
| POST   | `/certificate/{certificate_id}/appeal`       | Student submits appeal            |

---

## 🗄️ Database Schema

### **Key Tables:**
- **`students`**: Student information with email validation
- **`certificates`**: File storage with `file_content BYTEA` and metadata
- **`decisions`**: AI evaluation results with `ai_workflow_json TEXT`
- **`reviewers`**: Reviewer information with position and department

### **Storage Strategy:**
- **File Content**: Stored as `BYTEA` in database (no file system)
- **AI Results**: Complete workflow JSON stored as `TEXT`
- **Cross-Device**: All data accessible from any server/device

---

## 🤖 AI Pipeline (4-Stage Process)

### **Stage 1: Information Extraction**
- Extract employee details, employment periods, responsibilities
- Structural validation of extracted data

### **Stage 2: Academic Evaluation**
- Calculate working hours and ECTS credits
- Evaluate degree relevance and training type
- Generate decision (ACCEPTED/REJECTED) with justification

### **Stage 3: Validation**
- Cross-check extraction and evaluation against original document
- Identify any inaccuracies or missing information

### **Stage 4: Correction (if needed)**
- Fix identified issues while preserving valid calculations
- Maintain business rules (credit limits, decision logic)

---

## 🌐 Multi-Language Support

### **OCR Languages:**
- **English**: Default OCR processing
- **Finnish**: Auto-detection with Finnish language pack
- **Auto-Detection**: Content-based language identification

### **Language Detection:**
- Finnish character detection (äöå)
- Finnish keyword recognition
- Fallback to English if detection fails

---

## 📊 Decision Logic

### **Training Type Rules:**
- **General Training**: Maximum 10 ECTS credits
- **Professional Training**: Maximum 30 ECTS credits
- **Credit Calculation**: 1 ECTS = 27 hours of work

### **Decision Criteria:**
- **ACCEPTED**: Work experience meets requested training type criteria
- **REJECTED**: Work experience doesn't meet requested training type criteria
- **Justification**: Detailed reasoning for decision
- **Recommendation**: Actionable guidance for rejected cases

---

## 🔧 Development

### **Code Quality:**
- **Pre-commit hooks**: Ruff linting and formatting
- **Type hints**: Full type annotation coverage
- **Error handling**: Comprehensive exception management

### **Testing:**
- **Sample files**: Available in `samples/` directory
- **Database seeding**: Sample students and reviewers on init
- **API documentation**: Interactive Swagger UI at `/docs`

---

## 📄 More Information
- **Database Schema**: See `src/database/schema.sql`
- **API Documentation**: Interactive docs at `http://127.0.0.1:8000/docs`
- **Frontend**: See `../frontend/README.md`
- **AI Workflow**: See `AI_WORKFLOW_README.md`
- **OCR Process**: See `OCR_PROCESS_GUIDE.md`
