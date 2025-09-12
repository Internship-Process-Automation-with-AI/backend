# OAMK AI-Powered Academic Credit Evaluation Backend

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   cd backend
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export GEMINI_API_KEY=your_gemini_api_key
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

4. **Run Server**
   ```bash
   uvicorn src.API.main:app --reload
   # API Documentation: http://127.0.0.1:8000/docs
   ```

---

## 📝 What is this?

This backend processes work certificates for OAMK students using AI to evaluate and assign ECTS credits. It uses OCR to extract text from documents, then AI to analyze the content and make decisions about credit awards.

**Key Features:**
- **OCR Processing**: Extracts text from PDFs, Word docs, and images
- **AI Evaluation**: 4-stage pipeline (Extract → Evaluate → Validate → Correct)
- **Multi-language Support**: English and Finnish with auto-detection
- **Database Storage**: All files and results stored in PostgreSQL
- **Validation Systems**: Company legitimacy and name verification
- **Reviewer Workflow**: Human review and approval process

---

## 🔗 API Endpoints

### **Student Operations:**
- `GET /student/{email}` - Get student information
- `GET /student/{email}/applications` - List student applications
- `POST /student/{student_id}/upload-certificate` - Upload certificate

### **Certificate Processing:**
- `POST /certificate/{certificate_id}/process` - Run OCR + AI evaluation
- `GET /certificate/{certificate_id}/status` - Get processing status
- `GET /certificate/{certificate_id}/preview` - Preview certificate
- `GET /certificate/{certificate_id}` - Download certificate
- `DELETE /certificate/{certificate_id}` - Delete certificate

### **Review & Feedback:**
- `GET /reviewers` - List all reviewers
- `GET /reviewer/{email}` - Get reviewer information
- `POST /certificate/{certificate_id}/review` - Submit reviewer decision
- `POST /certificate/{certificate_id}/feedback` - Submit student feedback
- `POST /certificate/{certificate_id}/appeal` - Submit student appeal

---

## 🗄️ Database Schema

### **Main Tables:**
- **`students`**: Student information (email validation required)
- **`certificates`**: File storage and metadata
- **`decisions`**: AI evaluation results
- **`reviewers`**: Reviewer information
- **`additional_documents`**: Supporting documents for self-paced work

### **Decision Fields:**
- `ai_decision`: ACCEPTED/REJECTED
- `ai_justification`: Detailed reasoning
- `total_working_hours`: Calculated working hours
- `credits_awarded`: ECTS credits assigned
- `company_validation_status`: Company legitimacy check
- `name_validation_match_result`: Name verification result
- `student_comment`: Student feedback for rejected cases
- `reviewer_decision`: Human reviewer decision (PASS/FAIL)

---

## 🤖 AI Processing Pipeline

### **Stage 1: Information Extraction**
- Extract employee details, dates, responsibilities
- Extract company information
- Extract employee name for verification

### **Stage 2: Academic Evaluation**
- Calculate working hours and ECTS credits (1 ECTS = 27 hours)
- Evaluate degree relevance
- Determine training type (General max 10 ECTS, Professional max 30 ECTS)
- Generate decision with justification

### **Stage 3: Validation**
- Verify company legitimacy
- Verify employee name matches student identity
- Cross-check results against original document

### **Stage 4: Correction**
- Fix any identified issues
- Maintain business rules and credit limits

---

## 🌐 Multi-Language Support

### **Supported Languages:**
- **English**: Default processing
- **Finnish**: Auto-detected with optimized OCR

### **Finnish Features:**
- Character recognition for ä, ö, å
- Business terminology recognition
- Document pattern understanding
- 5-10% better accuracy than generic OCR

---

## 🔧 Development

### **Testing:**
```bash
# Run tests
python -m pytest tests/
```

### **Code Quality:**
- Pre-commit hooks with Ruff linting
- Type hints throughout
- Comprehensive error handling

---

## 📄 More Information

- **API Documentation**: http://127.0.0.1:8000/docs
- **Frontend**: See `../frontend/README.md`
- **AI Workflow**: See `AI_WORKFLOW_README.md`
- **OCR Process**: See `OCR_PROCESS_GUIDE.md`
- **Finnish OCR**: See `FINNISH_OCR_GUIDE.md`

---

**Note**: This system is designed for OAMK students to evaluate work certificates and determine practical training credits following Finnish higher education standards.