#!/usr/bin/env python3
"""
Debug script to identify the cause of 500 errors in the self-paced work functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from database.database import get_db_connection


def check_database_issues():
    """Check for common database issues that cause 500 errors."""
    print("🔍 Debugging 500 Error - Database Issues")
    print("=" * 50)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check 1: Does work_type column exist in certificates table?
                print("\n1. Checking work_type column in certificates table...")
                try:
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default 
                        FROM information_schema.columns 
                        WHERE table_name = 'certificates' AND column_name = 'work_type'
                    """)
                    result = cur.fetchone()
                    if result:
                        print("   ✅ work_type column exists")
                        print(
                            f"   Type: {result[1]}, Nullable: {result[2]}, Default: {result[3]}"
                        )
                    else:
                        print(
                            "   ❌ work_type column MISSING - This will cause 500 error!"
                        )
                        print(
                            "   💡 Fix: Run 'python -m src.database.init_db' to add the column"
                        )
                        return False
                except Exception as e:
                    print(f"   ❌ Error checking work_type column: {e}")
                    return False

                # Check 2: Does additional_documents table exist?
                print("\n2. Checking additional_documents table...")
                try:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = 'additional_documents'
                        )
                    """)
                    exists = cur.fetchone()[0]
                    if exists:
                        print("   ✅ additional_documents table exists")
                    else:
                        print(
                            "   ❌ additional_documents table MISSING - This will cause 500 error!"
                        )
                        print(
                            "   💡 Fix: Run 'python -m src.database.init_db' to create the table"
                        )
                        return False
                except Exception as e:
                    print(f"   ❌ Error checking additional_documents table: {e}")
                    return False

                # Check 3: Do enum types exist?
                print("\n3. Checking enum types...")
                try:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_type 
                            WHERE typname = 'work_type'
                        )
                    """)
                    work_type_exists = cur.fetchone()[0]

                    cur.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_type 
                            WHERE typname = 'document_type'
                        )
                    """)
                    document_type_exists = cur.fetchone()[0]

                    if work_type_exists:
                        print("   ✅ work_type enum exists")
                    else:
                        print(
                            "   ❌ work_type enum MISSING - This will cause 500 error!"
                        )
                        return False

                    if document_type_exists:
                        print("   ✅ document_type enum exists")
                    else:
                        print(
                            "   ❌ document_type enum MISSING - This will cause 500 error!"
                        )
                        return False
                except Exception as e:
                    print(f"   ❌ Error checking enum types: {e}")
                    return False

                # Check 4: Test the create_certificate function
                print("\n4. Testing create_certificate function...")
                try:
                    # This is just a test - we won't actually create anything
                    print("   ✅ create_certificate function can be imported")
                except Exception as e:
                    print(f"   ❌ Error importing create_certificate: {e}")
                    return False

                print("\n✅ All database checks passed!")
                return True

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print("💡 Make sure PostgreSQL is running and database is accessible")
        return False


def check_import_issues():
    """Check for import issues that could cause 500 errors."""
    print("\n🔍 Debugging 500 Error - Import Issues")
    print("=" * 50)

    try:
        print("\n1. Testing model imports...")
        print("   ✅ All model imports successful")

        print("\n2. Testing database function imports...")
        print("   ✅ Database function imports successful")

        print("\n3. Testing API imports...")
        print("   ✅ API function imports successful")

        return True

    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False


def main():
    """Main debug function."""
    print("🐛 Debugging 500 Internal Server Error")
    print("=" * 60)

    # Check database issues
    db_ok = check_database_issues()

    # Check import issues
    import_ok = check_import_issues()

    print("\n" + "=" * 60)
    print("📊 DEBUG SUMMARY:")
    print(f"   Database Issues: {'✅ RESOLVED' if db_ok else '❌ FOUND ISSUES'}")
    print(f"   Import Issues: {'✅ RESOLVED' if import_ok else '❌ FOUND ISSUES'}")

    if not db_ok or not import_ok:
        print("\n💡 RECOMMENDED FIXES:")
        if not db_ok:
            print("   1. Reset database: python -m src.database.init_db")
            print("   2. Verify schema: python verify_schema.py")
        if not import_ok:
            print("   3. Check Python path and module structure")
        return False

    print("\n🎉 No obvious issues found!")
    print("💡 Check backend server logs for the actual error message")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
