# Database Schema - UI-Relevant Data

## Overview

This document defines the database schema for storing work certificate evaluation results. Only the essential data needed for the student and teacher UI is saved, excluding technical processing details and raw LLM responses.

## Core Database Tables

### 1. **Students Table**
```sql
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    degree_program VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Data Source**: User input from pipeline
- `email`: Student's OAMK email (@students.oamk.fi)
- `degree_program`: Selected degree program

### 2. **Documents Table**
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(id),
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    document_type VARCHAR(50), -- 'pdf', 'docx', 'image'
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    processing_time DECIMAL(10,2), -- seconds
    error_message TEXT
);
```

**Data Source**: Pipeline metadata
- `original_filename`: Original uploaded file name
- `file_path`: Path to stored document
- `processing_status`: Current processing state
- `processing_time`: Total processing time in seconds

### 3. **Work Certificates Table**
```sql
CREATE TABLE work_certificates (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    student_id INTEGER REFERENCES students(id),
    
    -- Employee Information (from extraction)
    employee_name VARCHAR(255) NOT NULL,
    employer VARCHAR(255),
    certificate_issue_date DATE,
    
    -- Employment Details
    total_employment_period VARCHAR(100),
    document_language VARCHAR(10) DEFAULT 'en', -- 'en' or 'fi'
    
    -- Academic Evaluation Results
    total_working_hours INTEGER,
    training_type VARCHAR(20) NOT NULL, -- 'general' or 'professional'
    credits_qualified DECIMAL(5,2) NOT NULL,
    degree_relevance VARCHAR(10) NOT NULL, -- 'high', 'medium', 'low'
    
    -- Explanations for UI
    relevance_explanation TEXT,
    calculation_breakdown TEXT,
    summary_justification TEXT,
    conclusion TEXT,
    
    -- Processing Metadata
    confidence_level VARCHAR(20), -- 'high', 'medium', 'low'
    validation_passed BOOLEAN DEFAULT true,
    requires_correction BOOLEAN DEFAULT false,
    correction_applied BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Data Source**: LLM extraction and evaluation results

### 4. **Employment Positions Table**
```sql
CREATE TABLE employment_positions (
    id SERIAL PRIMARY KEY,
    work_certificate_id INTEGER REFERENCES work_certificates(id),
    
    -- Position Details
    title VARCHAR(255) NOT NULL,
    employer VARCHAR(255),
    start_date DATE,
    end_date DATE,
    duration VARCHAR(100),
    responsibilities TEXT,
    
    -- Calculated Fields
    duration_days INTEGER,
    duration_years DECIMAL(5,2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Data Source**: LLM extraction results (positions array)

### 5. **Validation Issues Table**
```sql
CREATE TABLE validation_issues (
    id SERIAL PRIMARY KEY,
    work_certificate_id INTEGER REFERENCES work_certificates(id),
    
    -- Issue Details
    issue_type VARCHAR(50) NOT NULL, -- 'extraction_error', 'missing_information', 'incorrect_assumption', 'justification_error'
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    description TEXT NOT NULL,
    field_affected VARCHAR(50), -- 'extraction', 'evaluation', 'justification'
    suggestion TEXT,
    
    -- Position-specific issues
    position_index INTEGER, -- NULL if not position-specific
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Data Source**: LLM validation results (only if validation_passed = false)

### 6. **Corrections Table**
```sql
CREATE TABLE corrections (
    id SERIAL PRIMARY KEY,
    work_certificate_id INTEGER REFERENCES work_certificates(id),
    
    -- Correction Details
    correction_type VARCHAR(50) NOT NULL, -- 'extraction', 'evaluation', 'both'
    original_value TEXT,
    corrected_value TEXT,
    correction_reason TEXT,
    validation_issue_id INTEGER REFERENCES validation_issues(id),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Data Source**: LLM correction results (only if correction_applied = true)

## Data Mapping from JSON Output

### **From `llm_results.extraction_results.results`:**
```json
{
  "employee_name": "John Doe",
  "employer": "Tech Corp",
  "certificate_issue_date": "2023-12-01",
  "positions": [
    {
      "title": "Software Developer",
      "employer": "Tech Corp",
      "start_date": "2023-01-15",
      "end_date": "2023-06-30",
      "duration": "5 months",
      "responsibilities": "Developed web applications..."
    }
  ],
  "total_employment_period": "5 months",
  "document_language": "en",
  "confidence_level": "high"
}
```

### **From `llm_results.evaluation_results.results`:**
```json
{
  "total_working_hours": 1040,
  "training_type": "professional",
  "credits_qualified": 30.0,
  "degree_relevance": "high",
  "relevance_explanation": "Work directly related to degree...",
  "calculation_breakdown": "1040 hours / 27 hours per ECTS = 38.5 credits, capped at 30.0",
  "summary_justification": "Professional role with significant responsibility...",
  "conclusion": "Student receives 30.0 ECTS credits as professional training.",
  "confidence_level": "high"
}
```

### **From `llm_results.validation_results.results`:**
```json
{
  "validation_passed": false,
  "issues_found": [
    {
      "type": "extraction_error",
      "severity": "medium",
      "description": "Wrong company name extracted",
      "field_affected": "extraction",
      "suggestion": "Correct employer name"
    }
  ]
}
```

## UI-Relevant Data Summary

### **What Students See:**
- ✅ **Personal Info**: Name, degree program, email
- ✅ **Document Status**: Processing status, upload date
- ✅ **Employment Details**: Company, position, dates, responsibilities
- ✅ **Academic Results**: Credits earned, training type, relevance
- ✅ **Explanations**: Why credits were awarded, calculation breakdown
- ✅ **Validation Status**: Whether results were validated/corrected

### **What Teachers See:**
- ✅ **Student Info**: Name, degree program, email
- ✅ **Document Details**: Original file, processing metadata
- ✅ **Employment Analysis**: Complete position history, responsibilities
- ✅ **Academic Evaluation**: Detailed credit calculation, relevance assessment
- ✅ **Quality Assurance**: Validation issues, corrections applied
- ✅ **Processing History**: Timestamps, confidence levels

### **What's NOT Saved (Technical Details):**
- ❌ Raw OCR text
- ❌ Raw LLM responses
- ❌ Processing logs
- ❌ Model names and versions
- ❌ Internal validation details
- ❌ Structural validation results
- ❌ Pipeline stage metadata

## Database Relationships

```
students (1) ──── (many) documents
documents (1) ──── (1) work_certificates
work_certificates (1) ──── (many) employment_positions
work_certificates (1) ──── (many) validation_issues
work_certificates (1) ──── (many) corrections
validation_issues (1) ──── (many) corrections
```

## Sample Queries for UI

### **Get Student's Work Certificates:**
```sql
SELECT 
    wc.*,
    d.original_filename,
    d.upload_date
FROM work_certificates wc
JOIN documents d ON wc.document_id = d.id
WHERE wc.student_id = $1
ORDER BY d.upload_date DESC;
```

### **Get Complete Certificate with Positions:**
```sql
SELECT 
    wc.*,
    ep.title, ep.employer, ep.start_date, ep.end_date, 
    ep.duration, ep.responsibilities
FROM work_certificates wc
LEFT JOIN employment_positions ep ON wc.id = ep.work_certificate_id
WHERE wc.id = $1
ORDER BY ep.start_date;
```

### **Get Validation Issues:**
```sql
SELECT * FROM validation_issues 
WHERE work_certificate_id = $1 
ORDER BY severity DESC, created_at;
```

### **Get Teacher Dashboard Data:**
```sql
SELECT 
    s.email, s.degree_program,
    wc.employee_name, wc.credits_qualified, wc.training_type,
    wc.degree_relevance, wc.validation_passed,
    d.upload_date, d.processing_time
FROM work_certificates wc
JOIN students s ON wc.student_id = s.id
JOIN documents d ON wc.document_id = d.id
WHERE d.processing_status = 'completed'
ORDER BY d.upload_date DESC;
```

## Data Integrity Rules

### **Business Rules:**
- `credits_qualified` must be ≤ 30 for professional training
- `credits_qualified` must be ≤ 10 for general training
- `training_type` must be 'general' or 'professional'
- `degree_relevance` must be 'high', 'medium', or 'low'
- `document_language` must be 'en' or 'fi'

### **Validation Rules:**
- `start_date` ≤ `end_date` for all positions
- `certificate_issue_date` cannot be in the future
- `total_working_hours` must be positive
- `confidence_level` must be 'high', 'medium', or 'low'

---

**Note**: This schema focuses on user-facing data while maintaining data integrity and providing comprehensive audit trails for quality assurance. 