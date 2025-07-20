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

-- Add/Update ENUM types

-- Create reviewer_decision enum (PASS / FAIL)
DO $$ BEGIN
    CREATE TYPE reviewer_decision AS ENUM ('PASS','FAIL');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Create appeal_status enum
DO $$ BEGIN
    CREATE TYPE appeal_status AS ENUM ('PENDING', 'APPROVED', 'REJECTED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Create Reviewer table
CREATE TABLE IF NOT EXISTS reviewers (
    reviewer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255)
);

-- Create Students table
CREATE TABLE IF NOT EXISTS students (
    student_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    degree VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255)

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
    filepath TEXT, -- Path or link to the uploaded file
    ocr_output TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

-- Constraints
CONSTRAINT certificates_filename_check CHECK (LENGTH(filename) > 0),
    CONSTRAINT certificates_filetype_check CHECK (LENGTH(filetype) > 0)
);

-- Create Decisions table
CREATE TABLE IF NOT EXISTS decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certificate_id UUID NOT NULL REFERENCES certificates(certificate_id) ON DELETE CASCADE,
    ai_justification TEXT NOT NULL,
    ai_decision decision_status NOT NULL, -- AI decision: ACCEPTED or REJECTED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    student_feedback TEXT, -- Student feedback for rejected applications
    reviewer_id UUID REFERENCES reviewers(reviewer_id),
    reviewer_decision reviewer_decision,  -- NULL = pending
    reviewer_comment TEXT, -- Reviewer's comments
    reviewed_at TIMESTAMP WITH TIME ZONE, -- When the review was completed

-- Constraints
CONSTRAINT decisions_ai_justification_check CHECK (LENGTH(ai_justification) > 0)
);

-- Create Appeals table
CREATE TABLE IF NOT EXISTS appeals (
    appeal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certificate_id UUID NOT NULL REFERENCES certificates(certificate_id) ON DELETE CASCADE,
    appeal_reason TEXT NOT NULL,
    appeal_status appeal_status DEFAULT 'PENDING',
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_by UUID REFERENCES reviewers(reviewer_id),
    review_comment TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT appeals_reason_check CHECK (LENGTH(appeal_reason) > 0)
);

-- Create indexes for better performance (only if they don't exist)
CREATE INDEX IF NOT EXISTS idx_students_email ON students (email);

CREATE INDEX IF NOT EXISTS idx_students_degree ON students (degree);

CREATE INDEX IF NOT EXISTS idx_certificates_student_id ON certificates (student_id);

CREATE INDEX IF NOT EXISTS idx_certificates_training_type ON certificates (training_type);

CREATE INDEX IF NOT EXISTS idx_certificates_uploaded_at ON certificates (uploaded_at);

CREATE INDEX IF NOT EXISTS idx_decisions_certificate_id ON decisions (certificate_id);

CREATE INDEX IF NOT EXISTS idx_decisions_ai_decision ON decisions (ai_decision);

CREATE INDEX IF NOT EXISTS idx_decisions_reviewer_decision ON decisions (reviewer_decision);

CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions (created_at);

CREATE INDEX IF NOT EXISTS idx_decisions_reviewed_at ON decisions (reviewed_at);

CREATE INDEX IF NOT EXISTS idx_appeals_certificate_id ON appeals (certificate_id);
CREATE INDEX IF NOT EXISTS idx_appeals_status ON appeals (appeal_status);
CREATE INDEX IF NOT EXISTS idx_appeals_submitted_at ON appeals (submitted_at);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at for students table (drop and recreate to avoid conflicts)
-- DROP TRIGGER IF EXISTS update_students_updated_at ON students;

-- CREATE TRIGGER update_students_updated_at
--     BEFORE UPDATE ON students
--     FOR EACH ROW
--     EXECUTE FUNCTION update_updated_at_column();

-- Migration script to update existing data if needed
-- Uncomment and run these if you have existing data that needs to be migrated

-- Add new columns to existing tables if they don't exist
-- DO $$ BEGIN
--     ALTER TABLE certificates ADD COLUMN IF NOT EXISTS filepath TEXT;
-- EXCEPTION WHEN duplicate_column THEN NULL;
-- END $$;

-- DO $$ BEGIN
--     ALTER TABLE decisions ADD COLUMN IF NOT EXISTS student_feedback TEXT;
-- EXCEPTION WHEN duplicate_column THEN NULL;
-- END $$;

-- DO $$ BEGIN
--     ALTER TABLE decisions ADD COLUMN IF NOT EXISTS review_status review_status DEFAULT 'PENDING';
-- EXCEPTION WHEN duplicate_column THEN NULL;
-- END $$;

-- DO $$ BEGIN
--     ALTER TABLE decisions ADD COLUMN IF NOT EXISTS reviewer_comment TEXT;
-- EXCEPTION WHEN duplicate_column THEN NULL;
-- END $$;

-- DO $$ BEGIN
--     ALTER TABLE decisions ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE;
-- EXCEPTION WHEN duplicate_column THEN NULL;
-- END $$;

-- Rename 'decision' column to 'ai_decision' if needed
-- DO $$ BEGIN
--     ALTER TABLE decisions RENAME COLUMN decision TO ai_decision;
-- EXCEPTION WHEN undefined_column THEN NULL;
-- END $$;