#!/usr/bin/env python3
"""
Script to add a student to the database.
"""

import sys
import uuid

from src.database.database import get_db_connection


def add_student(first_name, last_name, email, degree):
    """Add a student to the students table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Generate a new UUID for the student
                student_id = str(uuid.uuid4())

                # Insert the student
                cur.execute(
                    """
                    INSERT INTO students (student_id, email, first_name, last_name, degree)
                    VALUES (%s, %s, %s, %s, %s)
                """,
                    (student_id, email, first_name, last_name, degree),
                )

                conn.commit()
                print(f"✅ Student '{first_name} {last_name}' created successfully!")
                print(f"   Student ID: {student_id}")
                print(f"   Email: {email}")
                print(f"   Degree: {degree}")

    except Exception as e:
        print(f"❌ Error creating student: {e}")


def main():
    """Main function to handle command line arguments or interactive input."""
    if len(sys.argv) == 5:
        # Command line arguments provided
        first_name, last_name, email, degree = sys.argv[1:5]
        add_student(first_name, last_name, email, degree)
    else:
        # Interactive mode
        print("=== Add New Student ===")
        first_name = input("First name: ").strip()
        last_name = input("Last name: ").strip()
        email = input("Email: ").strip()
        degree = input("Degree: ").strip()

        if not all([first_name, last_name, email, degree]):
            print("❌ All fields are required!")
            return

        add_student(first_name, last_name, email, degree)


if __name__ == "__main__":
    main()
