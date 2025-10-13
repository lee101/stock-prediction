#!/bin/bash
"""
Convenience script to run Toto retraining system tests.
Provides simple commands for different test scenarios.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    
    # Check pip/uv
    if command -v uv &> /dev/null; then
        PIP_CMD="uv pip"
        print_success "Using uv for package management"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
        print_warning "Using pip (consider installing uv for faster package management)"
    else
        print_error "Neither uv nor pip found"
        exit 1
    fi
    
    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        print_warning "pytest not found, installing..."
        $PIP_CMD install pytest
    fi
    
    print_success "Dependencies check completed"
}

# Install test dependencies
install_deps() {
    print_header "Installing Test Dependencies"
    
    # Core testing packages
    $PIP_CMD install pytest pytest-mock pytest-timeout psutil
    
    # Optional testing packages (install if possible)
    echo "Installing optional packages..."
    $PIP_CMD install pytest-cov pytest-xdist pytest-json-report || print_warning "Some optional packages failed to install"
    
    # Core ML packages
    $PIP_CMD install torch numpy pandas scikit-learn || print_error "Failed to install core ML packages"
    
    print_success "Dependencies installed"
}

# Validate test setup
validate_setup() {
    print_header "Validating Test Setup"
    python3 test_runner.py validate
}

# Run different test suites
run_unit_tests() {
    print_header "Running Unit Tests"
    python3 test_runner.py unit
}

run_integration_tests() {
    print_header "Running Integration Tests"
    python3 test_runner.py integration
}

run_data_quality_tests() {
    print_header "Running Data Quality Tests"
    python3 test_runner.py data_quality
}

run_performance_tests() {
    print_header "Running Performance Tests"
    print_warning "Performance tests may take several minutes..."
    python3 test_runner.py performance
}

run_regression_tests() {
    print_header "Running Regression Tests"
    python3 test_runner.py regression
}

run_fast_tests() {
    print_header "Running Fast Tests (excluding slow ones)"
    python3 test_runner.py fast
}

run_all_tests() {
    print_header "Running All Tests"
    if [ "$1" = "--slow" ]; then
        print_warning "Including slow tests - this may take a while..."
        python3 test_runner.py all --slow
    else
        print_info "Excluding slow tests (use --slow to include them)"
        python3 test_runner.py all
    fi
}

# Run tests with coverage
run_coverage() {
    print_header "Running Tests with Coverage"
    python3 test_runner.py coverage
    
    if [ -d "htmlcov" ]; then
        print_success "Coverage report generated in htmlcov/"
        print_info "Open htmlcov/index.html in your browser to view the report"
    fi
}

# Quick smoke test
smoke_test() {
    print_header "Running Smoke Test"
    print_info "Running a few basic tests to verify everything works..."
    
    # Run dry run first
    python3 test_runner.py dry-run
    
    # Run a few unit tests
    python3 -m pytest test_toto_trainer.py::TestTotoOHLCConfig::test_config_initialization -v
    
    print_success "Smoke test completed"
}

# List available tests
list_tests() {
    print_header "Available Tests"
    python3 test_runner.py list
}

# Clean up test artifacts
cleanup() {
    print_header "Cleaning Up Test Artifacts"
    
    # Remove pytest cache
    rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
    
    # Remove coverage files
    rm -f .coverage htmlcov coverage.xml
    rm -rf htmlcov/
    
    # Remove test outputs
    rm -f test_report.json *.log
    rm -rf test_references/ logs/ checkpoints/ tensorboard_logs/ mlruns/
    
    print_success "Cleanup completed"
}

# CI/CD test suite
ci_tests() {
    print_header "Running CI/CD Test Suite"
    
    print_info "Step 1: Validation"
    validate_setup || exit 1
    
    print_info "Step 2: Unit tests"
    run_unit_tests || exit 1
    
    print_info "Step 3: Integration tests"
    run_integration_tests || exit 1
    
    print_info "Step 4: Data quality tests"
    run_data_quality_tests || exit 1
    
    print_info "Step 5: Regression tests"
    run_regression_tests || exit 1
    
    print_success "CI/CD test suite completed successfully"
}

# Development test suite (faster)
dev_tests() {
    print_header "Running Development Test Suite"
    
    print_info "Running fast tests for development..."
    run_fast_tests
    
    print_success "Development test suite completed"
}

# Show help
show_help() {
    cat << EOF
Toto Retraining System Test Runner

USAGE:
    ./run_tests.sh [COMMAND] [OPTIONS]

COMMANDS:
    help                Show this help message
    
    # Setup and validation
    deps                Install test dependencies
    validate            Validate test environment setup
    
    # Individual test suites
    unit                Run unit tests
    integration         Run integration tests  
    data-quality        Run data quality tests
    performance         Run performance tests (slow)
    regression          Run regression tests
    
    # Combined test suites
    fast                Run fast tests (excludes slow tests)
    all [--slow]        Run all tests (optionally include slow tests)
    ci                  Run CI/CD test suite
    dev                 Run development test suite (fast)
    
    # Coverage and reporting
    coverage            Run tests with coverage reporting
    smoke               Run quick smoke test
    list                List all available tests
    
    # Utilities
    cleanup             Clean up test artifacts
    
EXAMPLES:
    ./run_tests.sh deps              # Install dependencies
    ./run_tests.sh validate          # Check setup
    ./run_tests.sh unit              # Run unit tests
    ./run_tests.sh dev               # Quick development tests
    ./run_tests.sh all               # All tests except slow ones
    ./run_tests.sh all --slow        # All tests including slow ones
    ./run_tests.sh coverage          # Tests with coverage report
    ./run_tests.sh ci                # Full CI/CD suite

For more advanced options, use the Python test runner directly:
    python3 test_runner.py --help
EOF
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        help|--help|-h)
            show_help
            ;;
        deps|install-deps)
            check_dependencies
            install_deps
            ;;
        validate|check)
            check_dependencies
            validate_setup
            ;;
        unit)
            check_dependencies
            run_unit_tests
            ;;
        integration)
            check_dependencies
            run_integration_tests
            ;;
        data-quality|data_quality)
            check_dependencies
            run_data_quality_tests
            ;;
        performance|perf)
            check_dependencies
            run_performance_tests
            ;;
        regression)
            check_dependencies
            run_regression_tests
            ;;
        fast)
            check_dependencies
            run_fast_tests
            ;;
        all)
            check_dependencies
            run_all_tests "$2"
            ;;
        coverage|cov)
            check_dependencies
            run_coverage
            ;;
        smoke)
            check_dependencies
            smoke_test
            ;;
        list)
            check_dependencies
            list_tests
            ;;
        cleanup|clean)
            cleanup
            ;;
        ci|ci-cd)
            check_dependencies
            ci_tests
            ;;
        dev|development)
            check_dependencies
            dev_tests
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"