# Pre-commit Hooks Setup

This project uses pre-commit hooks to automatically check and format code before commits.

## What's Included

### Automatic Hooks (run on every commit):
- **Trailing whitespace removal**: Removes trailing spaces
- **End of file fixer**: Ensures files end with a newline
- **YAML checker**: Validates YAML syntax
- **Large file checker**: Prevents accidentally committing large files
- **Merge conflict checker**: Detects merge conflict markers
- **Debug statement checker**: Finds leftover debug statements
- **Black**: Python code formatter (88 character line length)
- **isort**: Import statement organizer
- **mypy**: Type checking (with relaxed settings)

### Manual Linting:
- **flake8**: Code style checker (run manually with `./lint.sh`)

## Installation

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

## Usage

### Automatic (on commit):
Pre-commit hooks run automatically when you commit. If any hook fails, the commit is blocked and you need to fix the issues.

### Manual runs:
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Run flake8 manually (not in pre-commit due to config issues)
./lint.sh
```

## Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hook configuration
- `setup.cfg`: Tool configurations (mypy, isort)
- `.flake8`: Flake8 configuration
- `requirements-dev.txt`: Development dependencies

## Troubleshooting

If pre-commit hooks fail:
1. Read the error messages carefully
2. Fix the reported issues
3. Stage the fixed files: `git add .`
4. Try committing again

To skip hooks (not recommended):
```bash
git commit --no-verify -m "commit message"
```

## Benefits

- **Consistent code style**: All code follows the same formatting rules
- **Catch issues early**: Problems are found before they reach the repository
- **Automated formatting**: No need to manually format code
- **Team collaboration**: Everyone uses the same code standards
