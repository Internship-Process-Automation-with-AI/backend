# Uploads Directory

This directory stores uploaded work certificate files and their processing results in organized folders.

## Structure

```
uploads/
├── 2024/                    # Year-based organization
│   ├── 01/                  # Month-based organization
│   │   ├── 15/              # Day-based organization
│   │   │   ├── original/    # Original uploaded files
│   │   │   │   ├── 20240115_143022_work_certificate.pdf
│   │   │   │   └── 20240115_143045_another_document.pdf
│   │   │   └── results/     # Processing results
│   │   │       ├── 20240115_143022_work_certificate_complete_results.json
│   │   │       ├── 20240115_143022_work_certificate_ocr_text.txt
│   │   │       ├── 20240115_143045_another_document_complete_results.json
│   │   │       └── 20240115_143045_another_document_ocr_text.txt
│   │   └── ...
│   └── ...
└── temp/                    # Temporary files during processing
```

## File Naming Convention

Files are renamed to avoid conflicts and maintain relationships:

### Original Files
- Original: `work_certificate.pdf`
- Stored as: `20240115_143022_work_certificate.pdf` (timestamp prefix)

### Processing Results
- Complete results: `20240115_143022_work_certificate_complete_results.json`
- OCR text: `20240115_143022_work_certificate_ocr_text.txt`

### File Relationships
All files related to a document share the same timestamp prefix, making it easy to trace:
- `20240115_143022_work_certificate.pdf` (original)
- `20240115_143022_work_certificate_complete_results.json` (results)
- `20240115_143022_work_certificate_ocr_text.txt` (OCR text)

## Security

- Files are organized by date for easy cleanup
- Original filenames are preserved in processing results
- Temporary files are automatically cleaned up
- No sensitive data is stored in filenames

## Database Integration

When database is implemented:
- File paths will be stored in database
- File metadata will be linked to processing results
- Cleanup policies can be implemented based on retention rules 