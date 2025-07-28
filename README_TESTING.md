# Backend Testing Guide

This document provides comprehensive information about the backend test suite for the OAMK Work Certificate Processor.

## Test Suite Status ✅

**All tests are passing!** The test suite includes:

| Test File | Status | Tests |
|-----------|--------|-------|
| `test_ocr_processing.py` | ✅ All Passing | 15 tests |
| `test_ai_workflow.py` | ✅ All Passing | 12 tests |
| `test_api_endpoints.py` | ✅ All Passing | 15 tests |
| `test_file_upload_download.py` | ✅ All Passing | 12 tests |
| `test_database_models.py` | ✅ All Passing | 25 tests |
| `test_database_operations.py` | ✅ All Passing | 10 tests |
| `test_credits_calculation.py` | ✅ All Passing | 22 tests |
| **TOTAL** | **✅ 111 Tests Passing** | **111 tests** |

## Overview

The test suite covers all major components of the backend system:

- **Database Models**: Student, Certificate, Decision, Reviewer
- **API Endpoints**: Upload, process, applications, reviewers
- **OCR Processing**: Text extraction, language detection
- **AI Workflow**: LLM orchestrator, degree evaluator
- **Database Operations**: CRUD operations, connection management
- **File Upload/Download**: File validation, storage, retrieval
- **Main Pipeline**: Document processing, workflow orchestration
- **Credits Calculation**: Working hours conversion, validation

## Test Structure

```
backend/tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── test_database_models.py        # Database model tests
├── test_api_endpoints.py          # API endpoint tests
├── test_ocr_processing.py         # OCR functionality tests
├── test_ai_workflow.py            # AI workflow tests
├── test_database_operations.py    # Database operation tests
├── test_file_upload_download.py   # File handling tests
└── test_credits_calculation.py    # Credits calculation tests
```

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required system dependencies:
   - Tesseract OCR Engine
   - Poppler (for PDF processing)

### Basic Test Execution

Run all tests:
```bash
cd backend
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run tests with coverage report:
```bash
pytest --cov=src --cov-report=html
```

### Running Specific Test Categories

Run only unit tests:
```bash
pytest -m unit
```

Run only integration tests:
```bash
pytest -m integration
```

Run only API tests:
```bash
pytest -m api
```

Run only database tests:
```bash
pytest -m database
```

Run only OCR tests:
```bash
pytest -m ocr
```

Run only AI workflow tests:
```bash
pytest -m ai
```

Run only file handling tests:
```bash
pytest -m file
```



### Running Specific Test Files

Run a specific test file:
```bash
pytest tests/test_api_endpoints.py
```

Run a specific test class:
```bash
pytest tests/test_api_endpoints.py::TestStudentEndpoints
```

Run a specific test method:
```bash
pytest tests/test_api_endpoints.py::TestStudentEndpoints::test_student_lookup_success
```

### Test Configuration

The test configuration is defined in `pytest.ini`:

- **Test paths**: `tests/`
- **Test file pattern**: `test_*.py`
- **Test class pattern**: `Test*`
- **Test function pattern**: `test_*`
- **Coverage reporting**: HTML, XML, and terminal output
- **Warning filters**: Ignores deprecation warnings

## Test Fixtures

### Database Fixtures

- `temp_db`: Creates a temporary database for testing
- `sample_student`: Creates a sample student for testing
- `sample_reviewer`: Creates a sample reviewer for testing
- `sample_certificate`: Creates a sample certificate for testing
- `sample_decision`: Creates a sample decision for testing

### Mock Fixtures

- `mock_ocr_workflow`: Mocks OCR workflow functionality
- `mock_llm_orchestrator`: Mocks LLM orchestrator functionality

### File Fixtures

- `sample_pdf_file`: Creates a temporary PDF file for testing
- `sample_image_file`: Creates a temporary image file for testing
- `sample_docx_file`: Creates a temporary DOCX file for testing

### API Fixtures

- `test_client`: Creates a FastAPI test client

## Test Categories

### 1. Database Models (`test_database_models.py`)

Tests for all data models:
- Student model creation and validation
- Certificate model with file content
- Decision model with AI and reviewer decisions
- Reviewer model with department information
- Enum validation (TrainingType, DecisionStatus, etc.)

### 2. API Endpoints (`test_api_endpoints.py`)

Tests for all API endpoints:
- Student endpoints (lookup, applications, upload)
- Reviewer endpoints (lookup, certificates, review)
- File endpoints (download, preview)
- General endpoints (reviewers list)

### 3. OCR Processing (`test_ocr_processing.py`)

Tests for OCR functionality:
- Text extraction from images and PDFs
- Language detection (Finnish, English)
- Text cleaning and preprocessing
- File type validation
- Error handling

### 4. AI Workflow (`test_ai_workflow.py`)

Tests for AI workflow components:
- LLM orchestrator initialization and availability
- Document processing with AI
- Degree evaluator functionality
- Credits calculation logic
- Integration scenarios

### 5. Database Operations (`test_database_operations.py`)

Tests for database operations:
- CRUD operations for all models
- Connection management
- Error handling
- Transaction management

### 6. File Upload/Download (`test_file_upload_download.py`)

Tests for file handling:
- File upload validation
- File type checking
- File size validation
- File storage and retrieval
- Download functionality



### 7. Credits Calculation (`test_credits_calculation.py`)

Tests for credits calculation:
- Working hours to credits conversion
- Input validation
- Edge cases and error handling
- Performance testing

## Test Coverage

The test suite aims for comprehensive coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **API Tests**: End-to-end API testing
- **Error Handling**: Exception and error scenario testing
- **Edge Cases**: Boundary condition testing
- **Performance**: Load and performance testing

## Mocking Strategy

The test suite uses extensive mocking to:

1. **Isolate Components**: Test individual components without dependencies
2. **Control External Services**: Mock OCR and LLM services
3. **Database Isolation**: Use temporary databases for testing
4. **File System Isolation**: Use temporary files for testing

## Best Practices

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<scenario>_<expected_result>`

### Test Organization

- Group related tests in classes
- Use descriptive test method names
- Include both positive and negative test cases
- Test edge cases and error conditions

### Assertions

- Use specific assertions for better error messages
- Test both success and failure scenarios
- Validate return values and side effects
- Check error messages and status codes

### Fixtures

- Use fixtures for common test data
- Keep fixtures simple and focused
- Use dependency injection for complex setups
- Clean up resources in fixture teardown

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:

1. **Fast Execution**: Tests complete within reasonable time
2. **Isolation**: Tests don't depend on external services
3. **Reproducible**: Tests produce consistent results
4. **Coverage Reporting**: Generates coverage reports for CI

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` directory is in Python path
2. **Database Connection**: Check database configuration
3. **File Permissions**: Ensure test files can be created/deleted
4. **Mock Issues**: Verify mock setup and teardown

### Debugging Tests

Run tests with debug output:
```bash
pytest -v -s --tb=long
```

Run specific failing test:
```bash
pytest tests/test_specific.py::TestClass::test_method -v -s
```

### Coverage Analysis

Generate detailed coverage report:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

View coverage in browser:
```bash
open htmlcov/index.html
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate test markers
3. Include both positive and negative test cases
4. Update this documentation if needed
5. Ensure tests pass in CI environment

## Performance Considerations

- Tests should complete quickly (< 1 second per test)
- Use mocking to avoid slow external calls
- Clean up resources to prevent memory leaks
- Use temporary files and databases for isolation 