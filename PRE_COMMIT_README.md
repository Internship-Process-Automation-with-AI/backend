# Pre-commit Setup and Usage Guide

This project uses pre-commit hooks to ensure code quality and consistency before commits.

## What Pre-commit Does

Pre-commit automatically runs these checks before each commit:

1. **Ruff Lint** - Checks for code style and potential errors, automatically fixes issues
2. **Ruff Import Sorting** - Sorts and organizes your imports automatically
3. **Ruff Format** - Formats your Python code consistently
4. **Trailing whitespace** - Removes trailing whitespace
5. **End-of-file** - Ensures files end with a newline
6. **YAML validation** - Checks YAML files for syntax errors
7. **Large file check** - Prevents accidentally committing large files
8. **Type checking** - Validates type hints and annotations

## Quick Setup

### Option 1: Automatic Setup (Recommended)
```bash
# Make sure you're in the backend directory
cd backend

# Run the setup script
python setup_precommit.py
```

### Option 2: Manual Setup
```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run on all files to format existing code
pre-commit run --all-files
```

## How to Use

### Normal Workflow
1. Make your changes to the code
2. Stage your changes: `git add .`
3. Commit your changes: `git commit -m "your message"`
4. Pre-commit will automatically run and format your code
5. If there are issues, fix them and commit again

### Manual Commands

**Run on staged files only:**
```bash
pre-commit run
```

**Run on all files:**
```bash
pre-commit run --all-files
```

**Skip pre-commit (not recommended):**
```bash
git commit --no-verify
```

**Update pre-commit hooks:**
```bash
pre-commit autoupdate
```

## What Happens When You Commit

1. Pre-commit runs all configured hooks in this order:
   - **Ruff Lint**: Checks and fixes code style issues
   - **Ruff Import Sorting**: Sorts and organizes imports
   - **Ruff Format**: Formats code consistently
   - Other checks (whitespace, file endings, etc.)
2. If any hook fails, the commit is blocked
3. Ruff automatically fixes most issues
4. You may need to stage the fixed files and commit again

## Current Configuration

### Pre-commit Configuration (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
      - id: ruff-check-imports
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Ruff Configuration (`.ruff.toml`)
```toml
# Ruff configuration for consistent code formatting
line-length = 88
target-version = "py38"

[lint]
select = ["E", "F", "I", "N", "W", "B", "C4", "UP", "ARG", "PIE", "T20", "Q", "SIM", "TCH", "TID", "TCH", "INT", "PTH", "ERA", "PD", "PGH", "PL", "TRY", "NPY", "AIR", "PERF", "FURB", "LOG", "RUF"]
ignore = ["E501", "B008", "C901", "W191", "E203", "W503"]

[format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

## Troubleshooting

### "Hook failed" Error
If you get a hook failure:
1. Check the error message
2. Fix the issues manually or let the hooks fix them
3. Stage the fixed files: `git add .`
4. Try committing again

### Skip Hooks Temporarily
```bash
git commit --no-verify -m "your message"
```
⚠️ **Warning**: Only use this for emergency commits. Always fix issues properly.

### Reinstall Hooks
If hooks stop working:
```bash
pre-commit uninstall
pre-commit install
```

### Common Ruff Issues
```bash
# If Ruff fails to install
pip install --upgrade pip
pip install ruff

# If there are import sorting issues
pre-commit run ruff-check-imports --all-files

# If there are formatting issues
pre-commit run ruff-format --all-files
```

## Configuration

The pre-commit configuration is in `.pre-commit-config.yaml`. The current setup uses:

- **Ruff** for linting, import sorting, and formatting (replaces Black, isort, and Flake8)
- **Ruff Lint**: Checks for code style and potential errors
- **Ruff Import Sorting**: Automatically sorts imports
- **Ruff Format**: Formats code consistently

You can modify the configuration to:
- Add new hooks
- Remove hooks you don't need
- Change hook settings
- Adjust Ruff rules

## Benefits

- **Faster execution** - Ruff is significantly faster than traditional tools
- **Consistent code style** across the project
- **Catches errors early** before they reach the repository
- **Automated formatting** saves time
- **Team collaboration** - everyone uses the same standards
- **Professional codebase** with clean, readable code
- **Unified tool** - Ruff replaces multiple tools (Black, isort, Flake8)

## Integration with Development Workflow

### IDE Integration
- **VS Code**: Install Python and Ruff extensions
- **PyCharm**: Configure external tools for Ruff
- **Vim/Neovim**: Use ALE or similar for Ruff integration

### CI/CD Integration
Pre-commit hooks can be integrated into CI/CD pipelines:
```yaml
# GitHub Actions example
- name: Run pre-commit
  uses: pre-commit/action@v3.0.0
  with:
    extra_args: --all-files
```

### Team Workflow
1. **New developers**: Run `python setup_precommit.py` after cloning
2. **Existing developers**: Run `pre-commit install` to update hooks
3. **Code reviews**: Ensure pre-commit passes before merging
4. **CI checks**: Automated pre-commit validation in pull requests

## Files Added

- `.pre-commit-config.yaml` - Configuration file
- `setup_precommit.py` - Setup script
- `PRE_COMMIT_README.md` - This guide
- `.ruff.toml` - Ruff configuration
- Updated `requirements.txt` - Added pre-commit dependency

## Advanced Usage

### Custom Hooks
Add custom hooks to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: custom-check
      name: Custom Check
      entry: python scripts/custom_check.py
      language: python
      types: [python]
```

### Hook-Specific Commands
```bash
# Run only Ruff linting
pre-commit run ruff --all-files

# Run only formatting
pre-commit run ruff-format --all-files

# Run only import sorting
pre-commit run ruff-check-imports --all-files
```

### Exclude Files
Add to `.pre-commit-config.yaml`:
```yaml
default_language_version:
  python: python3
exclude: ^(docs/|tests/|migrations/)
```

---

**Note**: This pre-commit setup ensures consistent code quality across the OAMK project. All developers should use these hooks to maintain code standards and catch issues early in the development process. 