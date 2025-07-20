#!/usr/bin/env python3
"""
Migration script to add evaluation detail columns to the decisions table.
This script adds columns for storing detailed evaluation results from the AI.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.database.database import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def migrate_evaluation_details():
    """Add evaluation detail columns to the decisions table."""

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if columns already exist
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'decisions' 
                    AND column_name IN (
                        'total_working_hours', 'credits_awarded', 'training_duration',
                        'training_institution', 'degree_relevance', 'supporting_evidence',
                        'challenging_evidence', 'recommendation'
                    )
                """)

                existing_columns = [row[0] for row in cur.fetchall()]

                # Add columns that don't exist
                columns_to_add = [
                    ("total_working_hours", "INTEGER"),
                    ("credits_awarded", "INTEGER"),
                    ("training_duration", "TEXT"),
                    ("training_institution", "TEXT"),
                    ("degree_relevance", "TEXT"),
                    ("supporting_evidence", "TEXT"),
                    ("challenging_evidence", "TEXT"),
                    ("recommendation", "TEXT"),
                ]

                for column_name, column_type in columns_to_add:
                    if column_name not in existing_columns:
                        logger.info(f"Adding column {column_name} to decisions table")
                        cur.execute(f"""
                            ALTER TABLE decisions 
                            ADD COLUMN {column_name} {column_type}
                        """)
                        logger.info(f"✓ Added column {column_name}")
                    else:
                        logger.info(f"Column {column_name} already exists, skipping")

                conn.commit()
                logger.info("✓ Migration completed successfully!")

                # Verify the columns were added
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'decisions' 
                    AND column_name IN (
                        'total_working_hours', 'credits_awarded', 'training_duration',
                        'training_institution', 'degree_relevance', 'supporting_evidence',
                        'challenging_evidence', 'recommendation'
                    )
                    ORDER BY column_name
                """)

                columns = cur.fetchall()
                logger.info("Verification - Current evaluation detail columns:")
                for column_name, data_type in columns:
                    logger.info(f"  - {column_name}: {data_type}")

                return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting evaluation details migration...")
    success = migrate_evaluation_details()

    if success:
        logger.info("Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("Migration failed!")
        sys.exit(1)
