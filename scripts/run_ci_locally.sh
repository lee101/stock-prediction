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
RUN_LINT=0
RUN_TESTS=0
RUN_TYPECHECK=0
CLEAN_VENV=0
DRY_RUN=0
JOBS_SELECTED=0
PYTEST_BASETEMP=""

usage() {
    cat <<'EOF'
Usage: scripts/run_ci_locally.sh [options]

Runs the local CI workflow with the same required checks as .github/workflows/ci-fast.yml.
Tip: run `python scripts/dev_setup_status.py` first if you are not sure which local virtualenv is ready.

Options:
  --clean           Recreate the local CI virtual environment before running.
  --job NAME        Run only the selected job. Repeatable.
                    Valid jobs: lint, tests, typecheck
  --dry-run         Print the selected configuration and exit without running CI jobs.
  --help            Show this help text.
EOF
}

add_job() {
    case "$1" in
        lint)
            RUN_LINT=1
            JOBS_SELECTED=1
            ;;
        tests)
            RUN_TESTS=1
            JOBS_SELECTED=1
            ;;
        typecheck)
            RUN_TYPECHECK=1
            JOBS_SELECTED=1
            ;;
        *)
            echo "Unknown job: $1" >&2
            usage
            exit 2
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN_VENV=1
            shift
            ;;
        --job)
            if [[ $# -lt 2 ]]; then
                echo "--job requires a value" >&2
                usage
                exit 2
            fi
            add_job "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ $JOBS_SELECTED -eq 0 ]]; then
    RUN_LINT=1
    RUN_TESTS=1
    RUN_TYPECHECK=1
fi

SELECTED_JOBS=()
if [[ $RUN_LINT -eq 1 ]]; then
    SELECTED_JOBS+=("lint")
fi
if [[ $RUN_TESTS -eq 1 ]]; then
    SELECTED_JOBS+=("tests")
fi
if [[ $RUN_TYPECHECK -eq 1 ]]; then
    SELECTED_JOBS+=("typecheck")
fi

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
echo "Selected jobs: ${SELECTED_JOBS[*]}"
echo ""

declare -a REQUIRED_FAILURES=()
declare -a OPTIONAL_FAILURES=()

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

record_required_failure() {
    REQUIRED_FAILURES+=("$1")
    error "$1"
}

record_optional_failure() {
    OPTIONAL_FAILURES+=("$1")
    warning "$1"
}

create_pytest_basetemp() {
    local base_root="$PROJECT_ROOT/.pytest_tmp"
    mkdir -p "$base_root"
    mktemp -d "$base_root/local-ci-XXXXXX"
}

if [[ $DRY_RUN -eq 1 ]]; then
    if [[ $RUN_TESTS -eq 1 ]]; then
        if [[ -n "${LOCAL_CI_PYTEST_BASETEMP:-}" ]]; then
            echo "Pytest basetemp override: $LOCAL_CI_PYTEST_BASETEMP"
        else
            echo "Pytest basetemp: $PROJECT_ROOT/.pytest_tmp/local-ci-<auto>"
        fi
    fi
    warning "Dry run: no commands executed"
    exit 0
fi

# Clean up old venv if requested
if [[ $CLEAN_VENV -eq 1 ]]; then
    step "Cleaning old virtual environment"
    rm -rf "$VENV_DIR"
    success "Cleaned $VENV_DIR"
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    step "Creating virtual environment"
    if ! $PYTHON_CMD -m venv "$VENV_DIR"; then
        warning "stdlib venv creation failed; retrying with uv venv"
        rm -rf "$VENV_DIR"
        if ! command -v uv >/dev/null 2>&1; then
            $PYTHON_CMD -m pip install --quiet --user --break-system-packages --upgrade uv
        fi
        uv venv --python "$PYTHON_CMD" "$VENV_DIR"
    fi
    success "Created virtual environment"
fi

# Activate virtual environment
step "Activating virtual environment"
source "$VENV_DIR/bin/activate"
success "Activated $VENV_DIR"

# Choose installer inside the environment.
if python -m pip --version >/dev/null 2>&1; then
    PIP_INSTALL=(python -m pip install)
else
    if ! command -v uv >/dev/null 2>&1; then
        error "uv is required when the virtual environment does not provide pip"
        exit 1
    fi
    PIP_INSTALL=(uv pip install)
fi

# Ensure uv is available for later package installs.
step "Ensuring uv is available"
if ! command -v uv >/dev/null 2>&1; then
    "${PIP_INSTALL[@]}" --quiet --upgrade uv
fi
success "uv available"

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

if [[ $RUN_LINT -eq 1 ]]; then
    step "JOB 1: Lint & Format Check"

    # Install linting dependencies
    step "Installing linting dependencies"
    uv pip install --quiet ruff flake8
    success "Linting tools installed"

    # Define targets matching .github/workflows/ci-fast.yml
    RUFF_TARGETS="rl_trading_agent_binance/hybrid_prompt.py rl_trading_agent_binance/trade_binance_live.py evaluate_binance_lora_candidate.py scripts/evaluate_binance_lora_candidate.py trltraining tests/test_hybrid_allocation_prompt.py tests/test_hybrid_prompt_parsing.py tests/test_rl_only_fallback.py tests/test_evaluate_binance_lora_candidate.py tests/test_validate_hybrid_cycle_snapshots.py tests/test_trltraining.py"

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
fi

# ============================================================================
# JOB 2: FAST UNIT TESTS
# ============================================================================

if [[ $RUN_TESTS -eq 1 ]]; then
    step "JOB 2: Fast Unit Tests (CPU)"

    # Install CPU-only dependencies
    step "Installing CPU-only dependencies from requirements-ci.txt"
    if [[ ! -f "requirements-ci.txt" ]]; then
        error "requirements-ci.txt not found!"
        exit 1
    fi

    uv pip install --requirement requirements-ci.txt
    success "Dependencies installed"

    PYTEST_BASETEMP="${LOCAL_CI_PYTEST_BASETEMP:-$(create_pytest_basetemp)}"
    if [[ -z "$PYTEST_BASETEMP" ]]; then
        error "Failed to create pytest basetemp"
        exit 1
    fi
    success "Using pytest basetemp: $PYTEST_BASETEMP"

    # Run fast unit tests
    step "Running fast unit tests"
    if python -m pytest \
        -v \
        --basetemp "$PYTEST_BASETEMP" \
        -m "unit and not slow and not model_required and not cuda_required" \
        --tb=short \
        --maxfail=10 \
        tests/; then
        success "Fast unit tests passed"
    else
        record_required_failure "Fast unit tests failed"
    fi

    # Run smoke tests (continue on error like CI)
    step "Running smoke tests (minimal model tests)"
    set +e
    python -m pytest \
        -v \
        --basetemp "$PYTEST_BASETEMP" \
        -m "smoke and model_required and not cuda_required" \
        --tb=short \
        --maxfail=3 \
        tests/
    SMOKE_STATUS=$?
    set -e
    if [[ $SMOKE_STATUS -eq 0 ]]; then
        success "Smoke tests passed"
    elif [[ $SMOKE_STATUS -eq 5 ]]; then
        warning "No smoke tests matched the current markers"
    else
        record_optional_failure "Smoke tests failed"
    fi
fi

# ============================================================================
# JOB 3: TYPE CHECKING
# ============================================================================

if [[ $RUN_TYPECHECK -eq 1 ]]; then
    step "JOB 3: Type Checking"

    # Install type checking tools
    step "Installing type checking tools"
    uv pip install --quiet ty pyright
    success "Type checking tools installed"

    # Run ty check (required, matches ci-fast.yml)
    step "Running ty check"
    TY_TARGETS="rl_trading_agent_binance/hybrid_prompt.py rl_trading_agent_binance/trade_binance_live.py evaluate_binance_lora_candidate.py scripts/evaluate_binance_lora_candidate.py trltraining"
    if ty check $TY_TARGETS; then
        success "ty check passed"
    else
        record_required_failure "ty check failed"
    fi

    # Run Pyright (continue on error like CI)
    step "Running Pyright"
    if python -m pyright src; then
        success "Pyright check passed"
    else
        record_optional_failure "Pyright check failed"
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
if (( ${#REQUIRED_FAILURES[@]} > 0 )); then
    echo -e "${RED}=== CI Test Complete: Required Checks Failed ===${NC}"
    printf '%s\n' "${REQUIRED_FAILURES[@]}" | sed 's/^/  - /'
    echo ""
    echo "To run with a clean environment: $0 --clean"
    echo "To deactivate venv: deactivate"
    exit 1
fi

echo -e "${GREEN}=== CI Test Complete ===${NC}"
echo -e "${BLUE}All required checks passed!${NC}"
if (( ${#OPTIONAL_FAILURES[@]} > 0 )); then
    echo "Optional failures:"
    printf '%s\n' "${OPTIONAL_FAILURES[@]}" | sed 's/^/  - /'
fi
echo ""
echo "To run with a clean environment: $0 --clean"
echo "To deactivate venv: deactivate"
