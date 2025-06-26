#!/usr/bin/env python3
"""
Pre-commit Setup Script
This script helps you set up pre-commit hooks for the project.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up pre-commit hooks...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists(".pre-commit-config.yaml"):
        print("❌ Error: .pre-commit-config.yaml not found!")
        print("Please run this script from the backend directory.")
        sys.exit(1)
    
    # Install pre-commit
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        sys.exit(1)
    
    # Install the git hook scripts
    if not run_command("pre-commit install", "Installing git hooks"):
        sys.exit(1)
    
    # Run pre-commit on all files (optional)
    print("\n🔄 Running pre-commit on all files...")
    if run_command("pre-commit run --all-files", "Running pre-commit checks"):
        print("✅ All files passed pre-commit checks!")
    else:
        print("⚠️  Some files need formatting. Run 'pre-commit run --all-files' to fix them.")
    
    print("\n" + "=" * 50)
    print("🎉 Pre-commit setup completed!")
    print("\n📋 How to use pre-commit:")
    print("1. Make your changes to the code")
    print("2. Stage your changes: git add .")
    print("3. Commit your changes: git commit -m 'your message'")
    print("4. Pre-commit will automatically run and format your code")
    print("5. If there are issues, fix them and commit again")
    print("\n💡 Manual commands:")
    print("- Run on staged files: pre-commit run")
    print("- Run on all files: pre-commit run --all-files")
    print("- Skip hooks (not recommended): git commit --no-verify")

if __name__ == "__main__":
    main() 