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
2. **Set API Key**
   ```bash
   export GEMINI_API_KEY=your_gemini_api_key
   # Windows: set GEMINI_API_KEY=your_gemini_api_key
   ```
3. **Initialize Database**
   ```bash
   python -m src.database.init_db
   ```
4. **Run FastAPI Server**
   ```bash
   uvicorn src.API.main:app --reload
   # Docs: http://127.0.0.1:8000/docs
   ```
5. **Frontend**
   See `../frontend/README.md` for UI setup and usage.

---

## 📝 What is this?
This backend processes work certificates for OAMK students, using OCR and AI to evaluate and assign ECTS credits for practical training. It exposes a FastAPI server and stores all data in PostgreSQL.

---

## ✨ Key Features
- OCR for PDFs, images, DOCX
- 4-stage AI pipeline for credit evaluation
- Degree-specific rules and training type (general/professional)
- Reviewer workflow (with position & department)
- Justifications and recommendations for each decision
- REST API for frontend integration

---

## 🗂️ System Flow
1. **Student uploads certificate**
2. **OCR & AI pipeline** extract info, evaluate, and assign credits
3. **Decision stored** in DB with justification, credits_awarded, etc.
4. **Reviewer** (with position/department) can approve/reject/appeal
5. **Frontend** displays results, allows appeals, and reviewer actions

---

## 🔗 Main API Endpoints
| Method | Path                                         | Description                       |
|--------|----------------------------------------------|-----------------------------------|
| POST   | `/student/{student_id}/upload-certificate`   | Upload a certificate file         |
| POST   | `/certificate/{certificate_id}/process`      | Run OCR + AI evaluation           |
| GET    | `/student/{email}/applications`              | List student's applications       |
| GET    | `/certificate/{certificate_id}`              | Download certificate file         |
| GET    | `/reviewers`                                 | List all reviewers                |
| POST   | `/certificate/{certificate_id}/review`       | Reviewer submits decision/comment |
| POST   | `/certificate/{certificate_id}/appeal`       | Student submits appeal            |

---

## 🗄️ Database Schema (Key Tables)
- **students**: `student_id`, `email`, `degree`, ...
- **certificates**: `certificate_id`, `student_id`, `training_type`, `filename`, ...
- **decisions**: `decision_id`, `certificate_id`, `ai_decision`, `credits_awarded`, `justification`, ...
- **reviewers**: `reviewer_id`, `email`, `first_name`, `last_name`, `position`, `department`

**Notes:**
- `credits_awarded` is always set from the AI pipeline (max 30 for professional, 10 for general)
- Reviewer objects always include `position` and `department`

---

## 📄 More
- See `schema.sql` for full DB structure
- See `/docs` (Swagger UI) for all API routes
- Sample reviewers and students are seeded on DB init
- Frontend: see `../frontend/README.md`
