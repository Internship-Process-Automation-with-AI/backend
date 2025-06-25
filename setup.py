#!/usr/bin/env python3
"""
Setup script for OAMK Internship Certificate OCR Processing System.
This script helps with installation and configuration.
Updated for PyMuPDF-based processing (no Poppler required).
"""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False


def check_tesseract():
    """Check if Tesseract is installed and accessible."""
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract found: {version}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Tesseract not found")
    return False


def install_tesseract():
    """Provide instructions for installing Tesseract."""
    system = platform.system().lower()
    
    print("\n📋 Tesseract Installation Instructions:")
    
    if system == "windows":
        print("1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Run the installer")
        print("3. Add Tesseract to your PATH or set TESSERACT_CMD in .env")
        print("4. Restart your terminal")
    elif system == "darwin":  # macOS
        print("1. Install Homebrew if not already installed:")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Install Tesseract:")
        print("   brew install tesseract")
    elif system == "linux":
        print("1. Ubuntu/Debian:")
        print("   sudo apt-get update && sudo apt-get install tesseract-ocr")
        print("2. CentOS/RHEL:")
        print("   sudo yum install tesseract")
        print("3. Arch Linux:")
        print("   sudo pacman -S tesseract")


def check_pymupdf():
    """Check if PyMuPDF is properly installed."""
    try:
        import fitz
        print("✅ PyMuPDF found and working")
        return True
    except ImportError:
        print("❌ PyMuPDF not found")
        return False


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    try:
        # Create a basic .env file
        env_content = """# OAMK Internship Certificate OCR Processing System
# Environment Configuration

# OCR Configuration
TESSERACT_CMD=
GOOGLE_CLOUD_CREDENTIALS=

# File Upload Configuration
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760

# Processing Configuration
OCR_CONFIDENCE_THRESHOLD=60.0
IMAGE_PREPROCESSING_ENABLED=true

# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=OAMK Internship Certificate Processor

# Development Configuration
LOG_LEVEL=INFO
DEBUG=false
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env file")
        print("📝 Please edit .env file with your configuration")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def create_upload_directory():
    """Create upload and output directories if they don't exist."""
    # Create uploads directory
    upload_dir = Path("uploads")
    if not upload_dir.exists():
        upload_dir.mkdir()
        print("✅ Created uploads directory")
    else:
        print("✅ Uploads directory already exists")
    
    # Create output directory
    output_dir = Path("output")
    if not output_dir.exists():
        output_dir.mkdir()
        print("✅ Created output directory")
    else:
        print("✅ Output directory already exists")


def run_tests():
    """Run the test suite."""
    try:
        print("\n🧪 Running tests...")
        subprocess.check_call([sys.executable, "test_ocr.py"])
        print("✅ Tests completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 OAMK Internship Certificate OCR Processing System Setup")
    print("📋 Updated for PyMuPDF-based processing (no Poppler required)")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Check PyMuPDF
    pymupdf_ok = check_pymupdf()
    if not pymupdf_ok:
        print("❌ PyMuPDF installation failed. Please check the installation.")
        sys.exit(1)
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    if not tesseract_ok:
        install_tesseract()
    
    # Create .env file
    create_env_file()
    
    # Create upload directory
    create_upload_directory()
    
    print("\n" + "=" * 60)
    print("📋 Setup Summary:")
    print(f"✅ Python dependencies: Installed")
    print(f"✅ PyMuPDF: {'Found' if pymupdf_ok else 'Not found'}")
    print(f"{'✅' if tesseract_ok else '❌'} Tesseract: {'Found' if tesseract_ok else 'Not found'}")
    print("✅ Configuration: .env file created")
    print("✅ Directories: uploads/ and output/ created")
    
    if not tesseract_ok:
        print("\n⚠️  IMPORTANT: Install Tesseract to use OCR fallback functionality")
    
    print("\n🎉 Setup completed!")
    print("\n📝 Next steps:")
    print("1. Install Tesseract (if not already installed) - for OCR fallback")
    print("2. Edit .env file with your configuration")
    print("3. Run: python test_ocr.py (to test OCR functionality)")
    print("4. Run: python main.py (to start the API server)")
    print("5. Open: http://localhost:8000/docs")
    print("\n💡 Smart OCR Features:")
    print("   - Raw image processing first (fastest)")
    print("   - Conditional preprocessing (only when needed)")
    print("   - Multiple Tesseract configurations")
    print("   - PyMuPDF for PDF text extraction")
    print("   - Google Vision API fallback")
    
    # Ask if user wants to run tests
    if tesseract_ok and pymupdf_ok:
        response = input("\n🧪 Run tests now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            run_tests()


if __name__ == "__main__":
    main() 