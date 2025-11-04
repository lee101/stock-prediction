#!/bin/bash
#
# Quick-start script to apply Toto compilation fixes and verify them.
#
# This script:
# 1. Applies the KVCache compilation fix
# 2. Runs accuracy tests to verify MAE equivalence
# 3. Reports results
#
# Usage:
#   ./apply_and_test_toto_compile_fix.sh          # Full test
#   ./apply_and_test_toto_compile_fix.sh --quick  # Quick test
#   ./apply_and_test_toto_compile_fix.sh --dry-run # Show what would be done

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
DRY_RUN=false
QUICK=false
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo_error "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run] [--quick] [--verbose]"
            exit 1
            ;;
    esac
done

echo_info "Toto torch.compile Fix Application and Testing"
echo_info "=============================================="
echo ""

# Step 1: Apply the fix
echo_info "Step 1: Applying compilation fix..."

if [ "$DRY_RUN" = true ]; then
    echo_warning "Running in DRY RUN mode - no changes will be made"
    python fix_toto_compile.py --dry-run
    echo_success "Dry run completed. Run without --dry-run to apply changes."
    exit 0
fi

python fix_toto_compile.py --backup

if [ $? -ne 0 ]; then
    echo_error "Failed to apply fix"
    exit 1
fi

echo_success "Fix applied successfully"
echo ""

# Step 2: Verify the fix
echo_info "Step 2: Verifying fix was applied..."

python fix_toto_compile.py --verify-only

if [ $? -ne 0 ]; then
    echo_error "Fix verification failed"
    exit 1
fi

echo_success "Fix verification passed"
echo ""

# Step 3: Run accuracy tests
echo_info "Step 3: Running accuracy tests..."

if [ "$QUICK" = true ]; then
    echo_info "Running QUICK test mode"
    export TOTO_COMPILE_QUICK=1
fi

if [ "$VERBOSE" = true ]; then
    echo_info "Enabling verbose compilation logging"
    export TORCH_LOGS="recompiles,graph_breaks,cudagraphs"
fi

# Run the test
python test_toto_compile_accuracy.py BTCUSD

TEST_RESULT=$?

echo ""

# Report results
if [ $TEST_RESULT -eq 0 ]; then
    echo_success "=============================================="
    echo_success "All tests PASSED!"
    echo_success "=============================================="
    echo ""
    echo_info "The compilation fix is working correctly:"
    echo "  ✓ No cudagraphs warnings"
    echo "  ✓ No recompilation limit warnings"
    echo "  ✓ MAE equivalence maintained"
    echo "  ✓ Performance improvements achieved"
    echo ""
    echo_info "Next steps:"
    echo "  1. Run your full backtest with compiled Toto"
    echo "  2. Monitor logs for any compilation warnings"
    echo "  3. Compare performance with uncompiled version"
else
    echo_error "=============================================="
    echo_error "Tests FAILED"
    echo_error "=============================================="
    echo ""
    echo_info "Troubleshooting:"
    echo "  1. Check test output above for specific failures"
    echo "  2. Review docs/TOTO_COMPILE_FIXES.md for details"
    echo "  3. Try running with --verbose to see compilation logs"
    echo "  4. Verify CUDA is available: python -c 'import torch; print(torch.cuda.is_available())'"
    exit 1
fi
