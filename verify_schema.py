#!/usr/bin/env python3
"""
Database Schema Verification Script

This script verifies that the database schema supports the new work_type and additional_documents functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import get_db_connection
from database.init_db import verify_database_schema


def verify_work_type_support():
    """Verify that the certificates table has the work_type column."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if work_type column exists in certificates table
                cur.execute("""
                    SELECT column_name, data_type, is_nullable, column_default 
                    FROM information_schema.columns 
                    WHERE table_name = 'certificates' AND column_name = 'work_type'
                """)

                result = cur.fetchone()
                if result:
                    print("✅ work_type column found in certificates table")
                    print(
                        f"   Type: {result[1]}, Nullable: {result[2]}, Default: {result[3]}"
                    )
                    return True
                else:
                    print("❌ work_type column NOT found in certificates table")
                    return False
    except Exception as e:
        print(f"❌ Error checking work_type column: {e}")
        return False


def verify_additional_documents_table():
    """Verify that the additional_documents table exists."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if additional_documents table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'additional_documents'
                    )
                """)

                exists = cur.fetchone()[0]
                if exists:
                    print("✅ additional_documents table found")

                    # Check table structure
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable 
                        FROM information_schema.columns 
                        WHERE table_name = 'additional_documents'
                        ORDER BY ordinal_position
                    """)

                    columns = cur.fetchall()
                    print("   Columns:")
                    for col in columns:
                        print(
                            f"     • {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})"
                        )

                    return True
                else:
                    print("❌ additional_documents table NOT found")
                    return False
    except Exception as e:
        print(f"❌ Error checking additional_documents table: {e}")
        return False


def verify_enum_types():
    """Verify that the required enum types exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check for work_type enum
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_type 
                        WHERE typname = 'work_type'
                    )
                """)
                work_type_exists = cur.fetchone()[0]

                # Check for document_type enum
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_type 
                        WHERE typname = 'document_type'
                    )
                """)
                document_type_exists = cur.fetchone()[0]

                if work_type_exists:
                    print("✅ work_type enum found")
                else:
                    print("❌ work_type enum NOT found")

                if document_type_exists:
                    print("✅ document_type enum found")
                else:
                    print("❌ document_type enum NOT found")

                return work_type_exists and document_type_exists
    except Exception as e:
        print(f"❌ Error checking enum types: {e}")
        return False


def main():
    """Main verification function."""
    print("🔍 Verifying Database Schema for Self-Paced Work Support")
    print("=" * 60)

    # Check basic schema
    print("\n📋 Checking basic schema...")
    schema_ok = verify_database_schema()

    # Check work_type support
    print("\n🏷️  Checking work_type support...")
    work_type_ok = verify_work_type_support()

    # Check additional_documents table
    print("\n📄 Checking additional_documents table...")
    additional_docs_ok = verify_additional_documents_table()

    # Check enum types
    print("\n🔤 Checking enum types...")
    enums_ok = verify_enum_types()

    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY:")
    print(f"   Basic Schema: {'✅ PASS' if schema_ok else '❌ FAIL'}")
    print(f"   Work Type Support: {'✅ PASS' if work_type_ok else '❌ FAIL'}")
    print(
        f"   Additional Documents Table: {'✅ PASS' if additional_docs_ok else '❌ FAIL'}"
    )
    print(f"   Enum Types: {'✅ PASS' if enums_ok else '❌ FAIL'}")

    all_ok = schema_ok and work_type_ok and additional_docs_ok and enums_ok
    print(
        f"\n🎯 Overall Result: {'✅ ALL CHECKS PASSED' if all_ok else '❌ SOME CHECKS FAILED'}"
    )

    if not all_ok:
        print("\n💡 To fix issues, run: python -m src.database.init_db")
        return False

    print("\n🎉 Database schema is ready for self-paced work functionality!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
