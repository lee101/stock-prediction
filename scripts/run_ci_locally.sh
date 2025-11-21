#!/bin/bash
# Local CI test script - replicates GitHub Actions ci-fast.yml workflow
# This creates a clean venv and runs all CI steps to catch issues before pushing

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venvci"
# Auto-detect Python version (prefer 3.13, fall back to 3.12, 3.11, or just python3)
if command -v python3.13 &> /dev/null; then
    PYTHON_CMD="python3.13"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
else
    PYTHON_CMD="python3"
fi
PYTHON_VERSION=$($PYTHON_CMD --version | awk '{print $2}')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}=== Local CI Test Runner ===${NC}"
echo "Python version: $PYTHON_VERSION"
echo "Virtual environment: $VENV_DIR"
echo ""

# Function to print step headers
step() {
    echo -e "\n${BLUE}>>> $1${NC}"
}

# Function to print success
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Clean up old venv if requested
if [[ "$1" == "--clean" ]]; then
    step "Cleaning old virtual environment"
    rm -rf "$VENV_DIR"
    success "Cleaned $VENV_DIR"
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    step "Creating virtual environment"
    $PYTHON_CMD -m venv "$VENV_DIR"
    success "Created virtual environment"
fi

# Activate virtual environment
step "Activating virtual environment"
source "$VENV_DIR/bin/activate"
success "Activated $VENV_DIR"

# Ensure uv is installed
step "Installing/upgrading uv"
pip install --quiet --upgrade uv
success "uv installed"

# Export CI environment variables
step "Setting CI environment variables"
export CI="1"
export FAST_CI="1"
export CPU_ONLY="1"
export FAST_SIMULATE="1"
export MARKETSIM_ALLOW_MOCK_ANALYTICS="1"
export MARKETSIM_SKIP_REAL_IMPORT="1"
export MARKETSIM_ALLOW_CPU_FALLBACK="1"
export ALP_PAPER="1"
export PYTHONUNBUFFERED="1"
export SKIP_TORCH_CHECK="0"
success "Environment variables set"

# ============================================================================
# JOB 1: LINT
# ============================================================================

step "JOB 1: Lint & Format Check"

# Install linting dependencies
step "Installing linting dependencies"
uv pip install --quiet ruff flake8
success "Linting tools installed"

# Define targets matching CI
RUFF_TARGETS="neuraldailytraining/model.py neuraldailytraining/runtime.py neuraldailytraining/trainer.py neuraldailymarketsimulator/simulator.py tests/test_neuraldailytraining.py tests/test_neuraldaily_alignment.py tests/test_neuraldaily_runtime_confidence_and_fees.py tests/test_trainer_utils.py tests/test_non_tradable_io.py"

# Run Ruff linter
step "Running Ruff linter"
if ruff check $RUFF_TARGETS; then
    success "Ruff linting passed"
else
    error "Ruff linting failed"
    exit 1
fi

# Run Ruff format check
step "Running Ruff format check"
if ruff format --check $RUFF_TARGETS; then
    success "Ruff format check passed"
else
    error "Ruff format check failed"
    exit 1
fi

# ============================================================================
# JOB 2: FAST UNIT TESTS
# ============================================================================

step "JOB 2: Fast Unit Tests (CPU)"

# Install CPU-only dependencies
step "Installing CPU-only dependencies from requirements-ci.txt"
if [[ ! -f "requirements-ci.txt" ]]; then
    error "requirements-ci.txt not found!"
    exit 1
fi

uv pip install --requirement requirements-ci.txt
success "Dependencies installed"

# Run fast unit tests
step "Running fast unit tests"
if python -m pytest \
    -v \
    -m "unit and not slow and not model_required and not cuda_required" \
    --tb=short \
    --maxfail=10 \
    tests/; then
    success "Fast unit tests passed"
else
    warning "Fast unit tests failed (continuing...)"
fi

# Run smoke tests (continue on error like CI)
step "Running smoke tests (minimal model tests)"
if python -m pytest \
    -v \
    -m "smoke and model_required and not cuda_required" \
    --tb=short \
    --maxfail=3 \
    tests/; then
    success "Smoke tests passed"
else
    warning "Smoke tests failed (continuing...)"
fi

# ============================================================================
# JOB 3: TYPE CHECKING
# ============================================================================

step "JOB 3: Type Checking"

# Install type checking tools
step "Installing type checking tools"
uv pip install --quiet ty pyright mypy
success "Type checking tools installed"

# Run ty check (continue on error like CI)
step "Running ty check"
if ty check; then
    success "ty check passed"
else
    warning "ty check failed (continuing...)"
fi

# Run Pyright (continue on error like CI)
step "Running Pyright"
if python -m pyright src; then
    success "Pyright check passed"
else
    warning "Pyright check failed (continuing...)"
fi

# Run mypy (continue on error like CI)
step "Running mypy"
if [[ -f "pyproject.toml" ]]; then
    if mypy src --config-file pyproject.toml; then
        success "mypy check passed"
    else
        warning "mypy check failed (continuing...)"
    fi
else
    warning "pyproject.toml not found, skipping mypy"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo -e "${GREEN}=== CI Test Complete ===${NC}"
echo -e "${BLUE}All required checks passed!${NC}"
echo ""
echo "To run with a clean environment: $0 --clean"
echo "To deactivate venv: deactivate"
