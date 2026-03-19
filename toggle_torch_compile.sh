#!/bin/bash
# Quick utility to toggle torch.compile on/off for production trading bot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env.compile"

show_status() {
    echo "================================"
    echo "Torch Compile Status"
    echo "================================"

    if [ -f "$ENV_FILE" ]; then
        echo "Configuration file: $ENV_FILE"
        cat "$ENV_FILE"
    else
        echo "No configuration file found"
        echo "Using default settings (compile enabled)"
    fi

    echo ""
    echo "Current environment:"
    echo "  TOTO_DISABLE_COMPILE=${TOTO_DISABLE_COMPILE:-not set}"
    echo "  TOTO_COMPILE_MODE=${TOTO_COMPILE_MODE:-max-autotune}"
    echo ""
}

enable_compile() {
    echo "Enabling torch.compile..."
    cat > "$ENV_FILE" << EOF
# Torch compile configuration
export TOTO_DISABLE_COMPILE=0
export TOTO_COMPILE_MODE=max-autotune
export TOTO_COMPILE_BACKEND=inductor
EOF
    echo "✅ Torch compile ENABLED"
    echo ""
    echo "To apply, run:"
    echo "  source $ENV_FILE"
    echo "  python trade_stock_e2e.py"
}

disable_compile() {
    echo "Disabling torch.compile..."
    cat > "$ENV_FILE" << EOF
# Torch compile configuration
export TOTO_DISABLE_COMPILE=1
EOF
    echo "✅ Torch compile DISABLED"
    echo ""
    echo "To apply, run:"
    echo "  source $ENV_FILE"
    echo "  python trade_stock_e2e.py"
}

set_mode() {
    local mode=$1
    echo "Setting torch.compile mode to: $mode"
    cat > "$ENV_FILE" << EOF
# Torch compile configuration
export TOTO_DISABLE_COMPILE=0
export TOTO_COMPILE_MODE=$mode
export TOTO_COMPILE_BACKEND=inductor
EOF
    echo "✅ Torch compile mode set to: $mode"
    echo ""
    echo "To apply, run:"
    echo "  source $ENV_FILE"
    echo "  python trade_stock_e2e.py"
}

run_test() {
    echo "Running compile stress test..."
    python "$PROJECT_ROOT/scripts/run_compile_stress_test.py" --mode production-check
}

# Parse command
case "${1:-status}" in
    enable)
        enable_compile
        show_status
        ;;
    disable)
        disable_compile
        show_status
        ;;
    status)
        show_status
        ;;
    mode)
        if [ -z "$2" ]; then
            echo "Error: Please specify a mode (default, reduce-overhead, max-autotune)"
            exit 1
        fi
        set_mode "$2"
        show_status
        ;;
    test)
        run_test
        ;;
    help|--help|-h)
        echo "Usage: $0 {enable|disable|status|mode|test}"
        echo ""
        echo "Commands:"
        echo "  enable          Enable torch.compile"
        echo "  disable         Disable torch.compile (safe mode)"
        echo "  status          Show current configuration"
        echo "  mode <MODE>     Set compile mode (default, reduce-overhead, max-autotune)"
        echo "  test            Run production readiness test"
        echo ""
        echo "Examples:"
        echo "  $0 disable                    # Disable for production"
        echo "  $0 mode reduce-overhead       # Use faster compile mode"
        echo "  $0 test                       # Test before deployment"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
