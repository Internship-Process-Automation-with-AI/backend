"""
Database initialization script for OAMK Work Certificate Processor.

This script initializes the PostgreSQL database by creating tables
and setting up the schema using raw SQL operations.
"""

import logging
import sys
from pathlib import Path

import psycopg2

from .database import (
    create_database_if_not_exists,
    create_sample_reviewers,
    create_sample_students,
    get_db_connection,
    test_database_connection,
)

logger = logging.getLogger(__name__)


def init_database(drop_existing: bool = False) -> bool:
    """
    Initialize the database by creating all tables.

    Args:
        drop_existing: If True, drop existing tables before creating new ones

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # First, ensure the database exists (this must be done outside any transaction)
        if not create_database_if_not_exists():
            logger.error("Failed to create database")
            return False

        # Test database connection
        if not test_database_connection():
            logger.error("Database connection test failed")
            return False

        # Read and execute schema SQL
        schema_file = Path(__file__).parent / "schema.sql"
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Drop existing tables if requested
                if drop_existing:
                    logger.warning("Dropping existing tables...")
                    cur.execute("DROP TABLE IF EXISTS decisions CASCADE")
                    cur.execute("DROP TABLE IF EXISTS certificates CASCADE")
                    cur.execute("DROP TABLE IF EXISTS students CASCADE")
                    cur.execute("DROP TYPE IF EXISTS training_type CASCADE")
                    cur.execute("DROP TYPE IF EXISTS decision_status CASCADE")
                    logger.info("Existing tables dropped")

                # Read entire schema file
                logger.info("Creating database tables...")
                schema_sql = schema_file.read_text()

                # Execute the full script. Since we made everything idempotent with
                # IF NOT EXISTS, this should work even if objects already exist.
                try:
                    cur.execute(schema_sql)
                except psycopg2.Error as e:
                    logger.error(f"Error executing schema SQL: {e}")
                    raise

                conn.commit()
                logger.info("Database tables created successfully")

        # Verify tables were created
        if verify_database_schema():
            logger.info("Database schema verification successful")
            return True
        else:
            logger.error("Database schema verification failed")
            return False

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def verify_database_schema() -> bool:
    """
    Verify that all required tables exist in the database.

    Returns:
        bool: True if all tables exist, False otherwise
    """
    required_tables = ["students", "certificates", "decisions"]

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check if all required tables exist
                for table_name in required_tables:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                        """,
                        (table_name,),
                    )

                    if not cur.fetchone()[0]:
                        logger.error(f"Table '{table_name}' does not exist")
                        return False

                logger.info("All required tables exist")
                return True

    except psycopg2.Error as e:
        logger.error(f"Schema verification failed: {e}")
        return False


def get_database_schema_info() -> dict:
    """
    Get information about the database schema.

    Returns:
        dict: Schema information including tables and columns
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get table information
                cur.execute(
                    """
                    SELECT table_name, table_type 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                    """
                )

                tables = []
                for row in cur.fetchall():
                    table_name = row[0]
                    table_type = row[1]

                    # Get column information for each table
                    cur.execute(
                        """
                        SELECT column_name, data_type, is_nullable, column_default 
                        FROM information_schema.columns 
                        WHERE table_name = %s 
                        ORDER BY ordinal_position
                        """,
                        (table_name,),
                    )

                    columns = []
                    for col_row in cur.fetchall():
                        columns.append(
                            {
                                "name": col_row[0],
                                "type": col_row[1],
                                "nullable": col_row[2] == "YES",
                                "default": col_row[3],
                            }
                        )

                    tables.append(
                        {"name": table_name, "type": table_type, "columns": columns}
                    )

                return {"tables": tables, "total_tables": len(tables)}

    except psycopg2.Error as e:
        logger.error(f"Failed to get schema info: {e}")
        return {"error": str(e)}


def reset_database() -> bool:
    """
    Reset the database by dropping and recreating all tables.

    WARNING: This will delete all data in the database!

    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        logger.warning("Resetting database - all data will be lost!")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Drop all tables
                cur.execute("DROP TABLE IF EXISTS decisions CASCADE")
                cur.execute("DROP TABLE IF EXISTS certificates CASCADE")
                cur.execute("DROP TABLE IF EXISTS students CASCADE")
                cur.execute("DROP TYPE IF EXISTS training_type CASCADE")
                cur.execute("DROP TYPE IF EXISTS decision_status CASCADE")
                conn.commit()

                logger.info("All tables dropped")

        # Recreate tables
        if init_database():
            logger.info("All tables recreated")
            return True
        else:
            logger.error("Failed to recreate tables")
            return False

    except psycopg2.Error as e:
        logger.error(f"Database reset failed: {e}")
        return False


if __name__ == "__main__":
    """
    Script entry point for database initialization.
    Usage:
        python -m src.database.init_db
    """

    print("🔄 Resetting database...")
    if reset_database():
        print("✅ Database reset successful")
        print("👥 Adding sample students...")
        create_sample_students()
        print("✅ Sample students added!")
        create_sample_reviewers()
        print("✅ Sample reviewers added!")
    else:
        print("❌ Database reset failed")
        sys.exit(1)

    # Display schema information
    print("\n📊 Database Schema Information:")
    schema_info = get_database_schema_info()
    if "error" not in schema_info:
        for table in schema_info["tables"]:
            print(f"  📋 {table['name']} ({len(table['columns'])} columns)")
            for col in table["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                default = f", default: {col['default']}" if col["default"] else ""
                print(f"    • {col['name']}: {col['type']} {nullable}{default}")
    else:
        print(f"  ❌ Error: {schema_info['error']}")

    print("\n🎉 Database setup complete!")
