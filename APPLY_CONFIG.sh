#!/bin/bash
# Quick script to apply optimal configuration

echo "Applying optimal configuration for production trading..."
echo ""
echo "Configuration:"
echo "  - Toto: EAGER mode (torch.compile disabled)"
echo "  - Kronos: EAGER mode (default, no compile support)"
echo "  - ADD_LATEST: True (fetch real bid/ask from API)"
echo ""

# Apply configuration
source .env.compile

# Verify
echo "✓ Configuration applied:"
echo "  TOTO_DISABLE_COMPILE=${TOTO_DISABLE_COMPILE}"
echo "  ADD_LATEST=${ADD_LATEST}"
echo ""
echo "Ready to run:"
echo "  python trade_stock_e2e.py"
echo ""

# Export for current shell
export TOTO_DISABLE_COMPILE=1
export ADD_LATEST=1
