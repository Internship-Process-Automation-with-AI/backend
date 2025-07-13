#!/usr/bin/env python3
"""
Database Setup Script for OAMK Work Certificate Processor

This script helps set up the PostgreSQL database with all required tables
and initial configuration.
"""

import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🗄️  OAMK Work Certificate Processor - Database Setup")
    print("=" * 60)
    print()


def check_postgresql():
    """Check if PostgreSQL is installed and running."""
    try:
        result = subprocess.run(
            ["psql", "--version"], capture_output=True, text=True, check=True
        )
        print(f"✅ PostgreSQL found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PostgreSQL not found. Please install PostgreSQL:")
        print("   Windows: https://www.postgresql.org/download/windows/")
        print("   macOS: brew install postgresql")
        print("   Linux: sudo apt-get install postgresql postgresql-contrib")
        return False


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        "psycopg2-binary",
        "fastapi",
        "uvicorn",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")

    if missing_packages:
        print("\n🔧 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def create_env_file():
    """Create .env file with database configuration."""
    env_file = Path(".env")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    print("📝 Creating .env file...")

    # Get database configuration from user
    db_host = input("Database host (default: localhost): ").strip() or "localhost"
    db_port = input("Database port (default: 5432): ").strip() or "5432"
    db_name = (
        input("Database name (default: oamk_certificates): ").strip()
        or "oamk_certificates"
    )
    db_user = input("Database user (default: postgres): ").strip() or "postgres"
    db_password = input("Database password: ").strip()

    gemini_api_key = input("Gemini API key (optional): ").strip()

    env_content = f"""# Database Configuration
DATABASE_HOST={db_host}
DATABASE_PORT={db_port}
DATABASE_NAME={db_name}
DATABASE_USER={db_user}
DATABASE_PASSWORD={db_password}
DATABASE_ECHO=false

# LLM Configuration
GEMINI_API_KEY={gemini_api_key}
"""

    env_file.write_text(env_content)
    print("✅ .env file created")
    return True


def test_database_connection():
    """Test database connection."""
    try:
        from src.database.database import test_database_connection

        if test_database_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


def initialize_database():
    """Initialize database with tables."""
    try:
        from src.database.init_db import init_database

        print("🔄 Initializing database...")

        if init_database():
            print("✅ Database initialized successfully")
            return True
        else:
            print("❌ Database initialization failed")
            return False
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    try:
        from src.database.init_db import create_sample_data

        create_samples = input("Create sample data for testing? (y/N): ").lower() == "y"

        if create_samples:
            print("📝 Creating sample data...")
            if create_sample_data():
                print("✅ Sample data created successfully")
            else:
                print("❌ Failed to create sample data")
        else:
            print("⏭️  Skipping sample data creation")

        return True
    except Exception as e:
        print(f"❌ Sample data creation error: {e}")
        return False


def display_next_steps():
    """Display next steps for the user."""
    print("\n🎉 Database setup complete!")
    print("\n📋 Next steps:")
    print("1. Start the API server:")
    print("   python -m src.api")
    print("\n2. Test the API:")
    print("   curl http://localhost:8000/api/health")
    print("\n3. View API documentation:")
    print("   http://localhost:8000/docs")
    print("\n4. Test database operations:")
    print(
        '   python -c "from src.database.init_db import get_database_schema_info; print(get_database_schema_info())"'
    )


def main():
    """Main setup function."""
    print_banner()

    # Check prerequisites
    print("🔍 Checking prerequisites...")
    if not check_postgresql():
        sys.exit(1)

    if not check_python_packages():
        print("\n💡 Install required packages first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    print("\n✅ All prerequisites met!")

    # Create environment file
    print("\n🔧 Setting up configuration...")
    if not create_env_file():
        sys.exit(1)

    # Test database connection
    print("\n🔗 Testing database connection...")
    if not test_database_connection():
        print("💡 Make sure PostgreSQL is running and credentials are correct")
        sys.exit(1)

    # Initialize database
    print("\n🗄️  Setting up database...")
    if not initialize_database():
        sys.exit(1)

    # Create sample data
    print("\n📊 Sample data setup...")
    create_sample_data()

    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
