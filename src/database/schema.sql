-- PostgreSQL Database Schema for OAMK Work Certificate Processor
-- This script creates the database tables and constraints

-- Create database (run this separately if needed)
-- CREATE DATABASE oamk_certificates;
-- \c oamk_certificates;

-- Enable UUID extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types only if they don't already exist
DO $$ BEGIN
    CREATE TYPE training_type AS ENUM ('GENERAL', 'PROFESSIONAL');

EXCEPTION WHEN duplicate_object THEN NULL;

END $$;

DO $$ BEGIN
    CREATE TYPE decision_status AS ENUM ('ACCEPTED', 'REJECTED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE work_type AS ENUM ('REGULAR', 'SELF_PACED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE document_type AS ENUM ('HOUR_DOCUMENTATION', 'PROJECT_DETAILS');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Create reviewer_decision enum (PASS / FAIL)
DO $$ BEGIN
    CREATE TYPE reviewer_decision AS ENUM ('PASS','FAIL');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Create Reviewer table
CREATE TABLE IF NOT EXISTS reviewers (
    reviewer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    position VARCHAR(255),
    department VARCHAR(255)
);

-- Create Students table
CREATE TABLE IF NOT EXISTS students (
    student_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    degree VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),

-- Constraints
CONSTRAINT students_email_check CHECK (email ~ '^[A-Za-z0-9._%+-]+@students\.oamk\.fi$'),
    CONSTRAINT students_degree_check CHECK (LENGTH(degree) > 0)
);

-- Create Certificates table
CREATE TABLE IF NOT EXISTS certificates (
    certificate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    training_type training_type NOT NULL,
    work_type work_type NOT NULL DEFAULT 'REGULAR',
    filename VARCHAR(255) NOT NULL,
    filetype VARCHAR(50) NOT NULL,
    file_content BYTEA, -- Store the actual file content in the database
    ocr_output TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

-- Constraints
CONSTRAINT certificates_filename_check CHECK (LENGTH(filename) > 0),
    CONSTRAINT certificates_filetype_check CHECK (LENGTH(filetype) > 0)
);

-- Create Decisions table (simplified - no complex appeal workflow)
CREATE TABLE IF NOT EXISTS decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certificate_id UUID NOT NULL REFERENCES certificates(certificate_id) ON DELETE CASCADE,
    ai_justification TEXT NOT NULL,
    ai_decision decision_status NOT NULL, -- AI decision: ACCEPTED or REJECTED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    student_comment TEXT, -- Student's comment for rejected applications (replaces appeal_reason)
    reviewer_id UUID REFERENCES reviewers(reviewer_id),
    reviewer_decision reviewer_decision,  -- NULL = pending, PASS/FAIL
    reviewer_comment TEXT, -- Reviewer's comments (optional)
    reviewed_at TIMESTAMP WITH TIME ZONE, -- When the review was completed
    -- Evaluation details from AI analysis
    total_working_hours INTEGER, -- Total working hours from certificate
    credits_awarded INTEGER, -- Credits awarded (ECTS)
    training_duration TEXT, -- Duration of training (e.g., "3 months")
    training_institution TEXT, -- Institution where training was conducted
    degree_relevance TEXT, -- How relevant the training is to the degree
    supporting_evidence TEXT, -- Supporting evidence for the decision
    challenging_evidence TEXT, -- Challenging evidence against the decision
    recommendation TEXT, -- AI recommendation summary
    -- Complete AI workflow output
    ai_workflow_json TEXT, -- Complete AI workflow JSON output
    -- Company validation
    company_validation_status VARCHAR(20) DEFAULT 'UNVERIFIED' CHECK (company_validation_status IN ('LEGITIMATE', 'NOT_LEGITIMATE', 'PARTIALLY_LEGITIMATE', 'UNVERIFIED')),
    company_validation_justification TEXT, -- Detailed explanation of company validation results

-- Constraints
CONSTRAINT decisions_ai_justification_check CHECK (LENGTH(ai_justification) > 0)
);

-- Create Additional Documents table for self-paced work
CREATE TABLE IF NOT EXISTS additional_documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certificate_id UUID NOT NULL REFERENCES certificates(certificate_id) ON DELETE CASCADE,
    document_type document_type NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filetype VARCHAR(50) NOT NULL,
    file_content BYTEA, -- Store the actual file content in the database
    ocr_output TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

-- Constraints
CONSTRAINT additional_documents_filename_check CHECK (LENGTH(filename) > 0),
    CONSTRAINT additional_documents_filetype_check CHECK (LENGTH(filetype) > 0)
);

-- Create indexes for better performance (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_students_email ON students (email);

CREATE INDEX IF NOT EXISTS idx_students_degree ON students (degree);

CREATE INDEX IF NOT EXISTS idx_certificates_student_id ON certificates (student_id);

CREATE INDEX IF NOT EXISTS idx_certificates_training_type ON certificates (training_type);

CREATE INDEX IF NOT EXISTS idx_certificates_work_type ON certificates (work_type);

CREATE INDEX IF NOT EXISTS idx_certificates_uploaded_at ON certificates (uploaded_at);

CREATE INDEX IF NOT EXISTS idx_decisions_certificate_id ON decisions (certificate_id);

CREATE INDEX IF NOT EXISTS idx_decisions_ai_decision ON decisions (ai_decision);

CREATE INDEX IF NOT EXISTS idx_decisions_reviewer_decision ON decisions (reviewer_decision);

CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions (created_at);

CREATE INDEX IF NOT EXISTS idx_decisions_reviewed_at ON decisions (reviewed_at);

CREATE INDEX IF NOT EXISTS idx_decisions_student_comment ON decisions (student_comment);

CREATE INDEX IF NOT EXISTS idx_decisions_reviewer_id ON decisions (reviewer_id);

-- Company validation index
CREATE INDEX IF NOT EXISTS idx_decisions_company_validation_status ON decisions (company_validation_status);

-- Additional documents indexes
CREATE INDEX IF NOT EXISTS idx_additional_documents_certificate_id ON additional_documents (certificate_id);

CREATE INDEX IF NOT EXISTS idx_additional_documents_document_type ON additional_documents (document_type);

CREATE INDEX IF NOT EXISTS idx_additional_documents_uploaded_at ON additional_documents (uploaded_at);