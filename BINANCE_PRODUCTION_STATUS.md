# Binance Trading Bot - Production Status

**Date:** 2026-02-02  
**Status:** ✅ LIVE AND TRADING  
**Account Value:** $5,721.57 USDT

## Summary

The Binance trading bot is now fully operational and connected to a live account with real funds.

## Current Configuration

- **Symbol:** SOLUSD
- **Model:** binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt
- **Performance (backtested):** 17.7x return, Sortino 66.1
- **Intensity Scale:** 10.0 (conservative, backtested at 20.0)
- **Min Gap:** 0.0005 (0.05%)
- **Poll Interval:** 30 seconds
- **Order Expiry:** 90 minutes

## Account Holdings

| Asset | Quantity | Value (USD) |
|-------|----------|-------------|
| SOL   | 55.000000 | $5,668.85   |
| BTC   | 0.000685  | $53.11      |
| USDT  | 0.166018  | $0.17       |
| **Total** | | **$5,721.57** |

## Current Trading Plan

**Generated:** 2026-02-02 00:00:00 UTC

- **Buy Price:** $99.78 - No quantity (insufficient USDT)
- **Sell Price:** $102.20 - Quantity: 55.000000 SOL ✅ **Watcher active**

## Technical Details

### Data Status
- **Last Update:** 2026-02-02 00:00:00 UTC
- **Data Points:** 34,493 hourly bars (2021-01-01 to 2026-02-02)
- **Data File:** trainingdatahourly/crypto/SOLUSD.csv

### API Connection
- ✅ Binance API keys configured
- ✅ Account access verified
- ✅ Price fetching working
- ✅ Order placement enabled

### Bot Status
- **Process:** Running in tmux session 'binance-trader'
- **PID:** 3067328
- **Metrics Log:** strategy_state/binanceexp1-solusd-metrics.csv
- **Management Script:** scripts/manage_binance_trader.sh

## Monitoring Commands

```bash
# Watch bot live (Ctrl+B then D to detach)
./scripts/manage_binance_trader.sh view

# Check status
./scripts/manage_binance_trader.sh status

# Check account balance
./scripts/manage_binance_trader.sh balance

# View metrics log
tail -f strategy_state/binanceexp1-solusd-metrics.csv

# Check open orders
source .venv313/bin/activate
python -c "from src.binan import binance_wrapper; import json; \
orders = [o for o in binance_wrapper.get_all_orders('SOLUSDT') \
if o['status'] not in ['CANCELED', 'FILLED']]; \
print(json.dumps(orders, indent=2))"

# Stop bot (emergency)
./scripts/manage_binance_trader.sh stop
```

## Risk Management

### Current Setup
- **Position:** 99% SOL, <1% other assets
- **Liquidity:** Very low USDT (0.17), cannot execute buy orders
- **Exposure:** Fully exposed to SOL price movements
- **Strategy:** Waiting to sell SOL at higher prices

### Recommendations
1. **Monitor First 24-48 Hours:** Check metrics every 2-3 hours
2. **Watch for Fills:** Track when sell orders execute
3. **Balance Portfolio:** Consider converting some SOL to USDT for balanced trading
4. **Scale Up Gradually:** Current intensity (10.0) is conservative; can increase to 15.0-20.0 after verification

### Emergency Procedures
```bash
# Stop bot immediately
./scripts/manage_binance_trader.sh stop

# Cancel all open orders
source .venv313/bin/activate
python -c "from src.binan import binance_wrapper; binance_wrapper.cancel_all_orders()"

# Check account status
./scripts/manage_binance_trader.sh balance
```

## Files Changed/Created

1. `src/metrics_utils.py` - Performance metrics calculation
2. `scripts/manage_binance_trader.sh` - Bot management utility
3. `trainingdatahourly/crypto/SOLUSD.csv` - Updated to Feb 2026
4. `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt` - Model checkpoint
5. `strategy_state/binanceexp1-solusd-metrics.csv` - PnL tracking log

## Next Steps

1. **Monitor Performance:** Track PnL and Sortino ratio over first week
2. **Adjust Parameters:** Consider increasing intensity-scale after 24-48h of stable operation
3. **Add USDT:** Transfer more USDT to enable buy orders and balanced trading
4. **Review Orders:** Check if sell orders are filling and at what prices
5. **Scale Up:** Increase intensity from 10.0 → 15.0 → 20.0 as confidence builds

## Documentation

- **Full Runbook:** `binance_latest.md`
- **Model Training:** `binanceprogress.md`
- **API Reference:** `docs/api.md`

---

**⚠️ WARNING:** This bot trades with REAL money. Monitor closely during the first 24-48 hours. Emergency stop available via `./scripts/manage_binance_trader.sh stop`.
