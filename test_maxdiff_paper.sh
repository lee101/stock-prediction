#!/bin/bash
# Test maxdiff trading on PAPER account with gates disabled to allow ETHUSD

echo "Testing MaxDiff trading with:"
echo "  - ETHUSD (blocked by consensus - bypassing)"
echo "  - UNIUSD (blocked by crypto sells - enabling)"
echo "  - SOLUSD (already in crypto list)"
echo ""

# Bypass consensus gate and enable crypto sells
PAPER=1 \
MARKETSIM_DISABLE_GATES=1 \
MARKETSIM_SYMBOL_SIDE_MAP="ETHUSD:buy,UNIUSD:both,SOLUSD:both,BTCUSD:both" \
python trade_stock_e2e.py
