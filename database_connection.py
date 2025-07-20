import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()


class PostgreSQLConnection:
    def __init__(self):
        self.connection_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "your_database"),
            "user": os.getenv("DB_USER", "your_username"),
            "password": os.getenv("DB_PASSWORD", "your_password"),
        }

    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            print("✅ Successfully connected to PostgreSQL!")
            return conn
        except psycopg2.Error as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            return None

    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        conn = self.connect()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)

                if query.strip().upper().startswith("SELECT"):
                    results = cursor.fetchall()
                    return [dict(row) for row in results]
                else:
                    conn.commit()
                    return {"message": "Query executed successfully"}

        except psycopg2.Error as e:
            print(f"❌ Error executing query: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def test_connection(self):
        """Test the database connection"""
        query = "SELECT version();"
        result = self.execute_query(query)
        if result:
            print(f"PostgreSQL Version: {result[0]['version']}")
        return result


# Example usage
if __name__ == "__main__":
    db = PostgreSQLConnection()

    # Test connection
    db.test_connection()

    # Example queries
    queries = [
        # Create table
        """
        CREATE TABLE IF NOT EXISTS test_users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # Insert data
        """
        INSERT INTO test_users (name, email) 
        VALUES (%s, %s) 
        ON CONFLICT (email) DO NOTHING;
        """,
        # Select data
        "SELECT * FROM test_users ORDER BY created_at DESC LIMIT 5;",
    ]

    # Execute queries
    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        if i == 1:  # Insert query with parameters
            result = db.execute_query(query, ("John Doe", "john@example.com"))
        else:
            result = db.execute_query(query)

        if result:
            print("Result:", result)
