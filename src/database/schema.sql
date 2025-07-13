-- PostgreSQL Database Schema for OAMK Work Certificate Processor
-- This script creates the database tables and constraints

-- Create database (run this separately if needed)
-- CREATE DATABASE oamk_certificates;
-- \c oamk_certificates;

-- Enable UUID extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types only if they don't already exist
DO $$ BEGIN
    CREATE TYPE training_type AS ENUM ('general', 'professional');

EXCEPTION WHEN duplicate_object THEN NULL;

END $$;

DO $$ BEGIN
    CREATE TYPE decision_status AS ENUM ('accepted', 'rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Create Students table
CREATE TABLE IF NOT EXISTS students (
    student_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    degree VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

-- Constraints
CONSTRAINT students_email_check CHECK (email ~ '^[A-Za-z0-9._%+-]+@students\.oamk\.fi$'),
    CONSTRAINT students_degree_check CHECK (LENGTH(degree) > 0)
);

-- Create Certificates table
CREATE TABLE IF NOT EXISTS certificates (
    certificate_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    training_type training_type NOT NULL,
    filename VARCHAR(255) NOT NULL,
    filetype VARCHAR(50) NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

-- Constraints
CONSTRAINT certificates_filename_check CHECK (LENGTH(filename) > 0),
    CONSTRAINT certificates_filetype_check CHECK (LENGTH(filetype) > 0)
);

-- Create Decisions table
CREATE TABLE IF NOT EXISTS decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certificate_id UUID NOT NULL REFERENCES certificates(certificate_id) ON DELETE CASCADE,
    ocr_output TEXT,
    decision decision_status NOT NULL,
    justification TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_reviewer VARCHAR(255),

-- Constraints
CONSTRAINT decisions_justification_check CHECK (LENGTH(justification) > 0)
);

-- Create indexes for better performance (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_students_email ON students (email);

CREATE INDEX IF NOT EXISTS idx_students_degree ON students (degree);

CREATE INDEX IF NOT EXISTS idx_certificates_student_id ON certificates (student_id);

CREATE INDEX IF NOT EXISTS idx_certificates_training_type ON certificates (training_type);

CREATE INDEX IF NOT EXISTS idx_certificates_uploaded_at ON certificates (uploaded_at);

CREATE INDEX IF NOT EXISTS idx_decisions_certificate_id ON decisions (certificate_id);

CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions (decision);

CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions (created_at);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at for students table (drop and recreate to avoid conflicts)
DROP TRIGGER IF EXISTS update_students_updated_at ON students;

CREATE TRIGGER update_students_updated_at 
    BEFORE UPDATE ON students 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();