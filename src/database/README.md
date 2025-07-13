# Database Setup for OAMK Work Certificate Processor

This directory contains the database configuration, models, and initialization scripts for the OAMK Work Certificate Processor PostgreSQL database using raw SQL operations.

## 🗄️ Database Schema

The database consists of three main tables following the schema diagram:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│       STUDENT       │    │    CERTIFICATE      │    │      DECISION       │
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ student_id (UUID)PK │◄───┤ certificate_id (PK) │◄───┤ decision_id (PK)    │
│ email (VARCHAR)     │    │ student_id (FK)     │    │ certificate_id (FK) │
│ degree (VARCHAR)    │    │ training_type (ENUM)│    │ ocr_output (TEXT)   │
│ created_at          │    │ filename (VARCHAR)  │    │ decision (ENUM)     │
│ updated_at          │    │ filetype (VARCHAR)  │    │ justification (TEXT)│
└─────────────────────┘    │ uploaded_at         │    │ created_at          │
                           └─────────────────────┘    │ assigned_reviewer   │
                                                      └─────────────────────┘
```

### Table Descriptions

#### Students Table
- **student_id**: UUID primary key
- **email**: Student's email (must be @students.oamk.fi)
- **degree**: Student's degree program
- **created_at/updated_at**: Timestamps

#### Certificates Table
- **certificate_id**: UUID primary key
- **student_id**: Foreign key to students table
- **training_type**: ENUM ('general', 'professional')
- **filename**: Original filename of uploaded certificate
- **filetype**: File extension/type
- **uploaded_at**: Upload timestamp

#### Decisions Table
- **decision_id**: UUID primary key
- **certificate_id**: Foreign key to certificates table
- **ocr_output**: Extracted text from OCR processing
- **decision**: ENUM ('accepted', 'rejected')
- **justification**: Explanation for the decision
- **created_at**: Decision timestamp
- **assigned_reviewer**: Optional reviewer identifier

## 🚀 Quick Setup

### Prerequisites
1. **PostgreSQL 12+** installed and running
2. **Python 3.10+** with virtual environment
3. **Dependencies** installed (`pip install -r requirements.txt`)

### Environment Variables
Create a `.env` file in the backend directory:

```bash
# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=oamk_certificates
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password_here
DATABASE_ECHO=false

# Other required variables
GEMINI_API_KEY=your_gemini_api_key_here
```

### Initialize Database

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Initialize database with tables
python -m src.database.init_db

# Or with sample data
python -m src.database.init_db --sample-data

# Reset database (WARNING: Deletes all data!)
python -m src.database.init_db --reset
```

## 📁 File Structure

```
src/database/
├── __init__.py           # Package initialization
├── models.py             # Data classes and enums
├── database.py           # Raw SQL database operations
├── init_db.py            # Database initialization script
├── schema.sql            # PostgreSQL schema file
└── README.md             # This file
```

## 🛠️ Usage Examples

### Basic Database Operations

```python
from src.database import (
    create_student, 
    get_students, 
    get_student_by_email,
    create_certificate,
    create_decision
)
from src.database.models import TrainingType, DecisionStatus

# Create new student
student = create_student(
    email="test@students.oamk.fi",
    degree="Information Technology"
)

# Get all students
students = get_students(skip=0, limit=100)

# Get student by email
student = get_student_by_email("test@students.oamk.fi")

# Create certificate
certificate = create_certificate(
    student_id=student.student_id,
    training_type=TrainingType.PROFESSIONAL,
    filename="certificate.pdf",
    filetype="pdf"
)

# Create decision
decision = create_decision(
    certificate_id=certificate.certificate_id,
    ocr_output="OCR text here...",
    decision=DecisionStatus.ACCEPTED,
    justification="Work experience meets requirements",
    assigned_reviewer="AI System"
)
```

### Advanced Operations

```python
# Get student with their certificates
student_with_certificates = get_student_with_certificates(student_id)

# Get certificates with pagination
certificates = get_certificates(skip=0, limit=100)

# Get decisions with pagination
decisions = get_decisions(skip=0, limit=100)

# Get system statistics
stats = get_statistics()
print(f"Total students: {stats['total_students']}")
print(f"Acceptance rate: {stats['decisions']['acceptance_rate']}%")

# Direct raw SQL if needed
from src.database.database import get_db_connection

with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM students WHERE degree = %s", ("Information Technology",))
        it_students = cur.fetchone()[0]
        print(f"IT students: {it_students}")
```

## 🔧 Database Management

### Manual SQL Setup

If you prefer to set up the database manually:

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# Create database
CREATE DATABASE oamk_certificates;
\c oamk_certificates;

# Run the schema file
\i src/database/schema.sql
```

### Health Check

```python
from src.database.database import check_database_health

health = check_database_health()
print(f"Database status: {health['status']}")
```

### Schema Information

```python
from src.database.init_db import get_database_schema_info

schema_info = get_database_schema_info()
for table in schema_info['tables']:
    print(f"Table: {table['name']}")
    for column in table['columns']:
        print(f"  {column['name']}: {column['type']}")
```

## 🔐 Security Considerations

1. **Email Validation**: Only @students.oamk.fi emails are accepted
2. **Foreign Key Constraints**: Ensure data integrity
3. **Connection Pooling**: Configured for production use
4. **SQL Injection Prevention**: Using parameterized queries
5. **Password Security**: Never log database passwords

## 🧪 Testing

### Unit Tests

```python
import pytest
from src.database import create_student, get_student_by_email

def test_create_student():
    # Create student
    student = create_student(
        email="test@students.oamk.fi", 
        degree="IT"
    )
    
    assert student.student_id is not None
    assert student.email == "test@students.oamk.fi"
    
    # Verify student was created
    retrieved_student = get_student_by_email("test@students.oamk.fi")
    assert retrieved_student is not None
    assert retrieved_student.student_id == student.student_id
```

### Integration Tests

```bash
# Run with test database
DATABASE_NAME=oamk_certificates_test python -m pytest tests/
```

## 📊 Performance Considerations

### Indexes
- Email index for fast student lookups
- Student ID index for certificate queries
- Certificate ID index for decision queries
- Timestamp indexes for date-based queries

### Connection Pooling
- Pool size: 20 connections
- Max overflow: 0
- Pool recycle: 3600 seconds (1 hour)

## 🔄 Schema Changes

For database schema changes, update the `schema.sql` file and reinitialize:

```bash
# Update schema.sql with your changes
# Then reinitialize database (WARNING: This will drop all data)
python -m src.database.init_db --reset

# Or for production, write migration scripts
python -c "
from src.database.database import get_db_connection
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('ALTER TABLE students ADD COLUMN new_field VARCHAR(255)')
        conn.commit()
"
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: could not connect to server: Connection refused
   ```
   - Check PostgreSQL service is running
   - Verify connection parameters in `.env`

2. **Database Does Not Exist**
   ```
   Error: database "oamk_certificates" does not exist
   ```
   - Run `python -m src.database.init_db`
   - Or create manually with `CREATE DATABASE oamk_certificates;`

3. **Permission Denied**
   ```
   Error: permission denied for database
   ```
   - Check database user permissions
   - Ensure user has CREATE, SELECT, INSERT, UPDATE, DELETE privileges

4. **Table Already Exists**
   ```
   Error: relation "students" already exists
   ```
   - Use `--drop` flag to drop existing tables
   - Or use `--reset` to reset entire database

### Debug Mode

Enable SQL logging for debugging:

```bash
DATABASE_ECHO=true python -m src.database.init_db
```

## 📚 Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Psycopg2 Documentation](https://www.psycopg.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Database Programming](https://realpython.com/python-sql-libraries/) 