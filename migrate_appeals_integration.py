#!/usr/bin/env python3
"""
Migration script to integrate appeals into decisions table.
This script:
1. Adds appeal fields to the decisions table
2. Removes the separate appeals table
3. Updates indexes
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.database.database import get_db_connection


def migrate_appeals_integration():
    """Run the appeals integration migration."""
    print("Starting appeals integration migration...")

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Step 1: Add appeal fields to decisions table
                print("Adding appeal fields to decisions table...")

                # Add appeal_status enum if it doesn't exist
                cur.execute("""
                    DO $$ BEGIN
                        CREATE TYPE appeal_status AS ENUM ('PENDING', 'APPROVED', 'REJECTED');
                    EXCEPTION WHEN duplicate_object THEN NULL;
                    END $$;
                """)

                # Add appeal fields to decisions table
                appeal_fields = [
                    "appeal_reason TEXT",
                    "appeal_status appeal_status",
                    "appeal_submitted_at TIMESTAMP WITH TIME ZONE",
                    "appeal_reviewer_id UUID REFERENCES reviewers(reviewer_id)",
                    "appeal_review_comment TEXT",
                    "appeal_reviewed_at TIMESTAMP WITH TIME ZONE",
                ]

                for field in appeal_fields:
                    try:
                        field_name = field.split()[0]
                        cur.execute(f"ALTER TABLE decisions ADD COLUMN {field}")
                        print(f"  ✅ Added {field_name}")
                    except Exception as e:
                        if "duplicate_column" in str(e):
                            print(f"  ⚠️  Column {field_name} already exists")
                        else:
                            print(f"  ❌ Error adding {field_name}: {e}")

                # Step 2: Create new indexes for appeal fields
                print("Creating appeal indexes...")

                try:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_decisions_appeal_status ON decisions (appeal_status)"
                    )
                    print("  ✅ Created appeal_status index")
                except Exception as e:
                    print(f"  ⚠️  Appeal status index error: {e}")

                try:
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_decisions_appeal_submitted_at ON decisions (appeal_submitted_at)"
                    )
                    print("  ✅ Created appeal_submitted_at index")
                except Exception as e:
                    print(f"  ⚠️  Appeal submitted_at index error: {e}")

                # Step 3: Check if appeals table exists and remove it
                print("Checking for appeals table...")
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'appeals'
                """)

                if cur.fetchone():
                    print("  Found appeals table, removing it...")
                    cur.execute("DROP TABLE IF EXISTS appeals CASCADE")
                    print("  ✅ Removed appeals table")
                else:
                    print(
                        "  ⚠️  Appeals table not found (already removed or never existed)"
                    )

                # Step 4: Verify the migration
                print("Verifying migration...")

                # Check if appeal fields exist in decisions table
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'decisions' 
                    AND column_name LIKE 'appeal_%'
                    ORDER BY column_name
                """)

                appeal_columns = [row[0] for row in cur.fetchall()]
                expected_columns = [
                    "appeal_reason",
                    "appeal_status",
                    "appeal_submitted_at",
                    "appeal_reviewer_id",
                    "appeal_review_comment",
                    "appeal_reviewed_at",
                ]

                missing_columns = [
                    col for col in expected_columns if col not in appeal_columns
                ]

                if missing_columns:
                    print(f"  ❌ Missing columns: {missing_columns}")
                    return False
                else:
                    print(f"  ✅ All appeal columns found: {appeal_columns}")

                # Check if appeals table is gone
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'appeals'
                """)

                if cur.fetchone():
                    print("  ❌ Appeals table still exists")
                    return False
                else:
                    print("  ✅ Appeals table successfully removed")

                conn.commit()
                print("✅ Appeals integration migration completed successfully!")
                return True

    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = migrate_appeals_integration()
    sys.exit(0 if success else 1)
