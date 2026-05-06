# Trading System Health Check & Optimization

You are a production trading system monitor. Check all active systems, fix issues, and look for optimization opportunities.

## Context
- Read `/home/lee/code/stock/binanceprod.md` for current deployment details
- Read `/home/lee/code/stock/prod.md` for full production state
- Secrets live in `~/.secretbashrc`; source it before using commands that need local credentials.

## Health Check Checklist

### 1. Service Status
```bash
source ~/.secretbashrc
sudo -S supervisorctl status <<< "$SUDO_PASSWORD"
ps aux | grep -E "trade_live|trade_binance_live" | grep -v grep
```

### 2. Recent Errors (last 200 lines of each)
```bash
tail -200 /var/log/supervisor/binance-hybrid-spot-error.log
tail -200 /var/log/supervisor/binance-worksteal-daily-error.log
tail -200 /var/log/supervisor/unified-stock-trader.log
```
Look for: crashes, API errors, LOT_SIZE failures, rate limits, connection drops, stuck positions.

### 3. Position & PnL Review
```bash
python3 -m json.tool binance_worksteal/live_state.json
```
Check: positions held too long (>14d for worksteal, >6h for hybrid), unrealized losses, equity trend.

### 4. Market Simulator Validation
Run eval on deployed checkpoints to verify they're still performing:
```bash
source .venv/bin/activate
# Crypto15 s78 champion:
python -m pufferlib_market.evaluate_fast \
  --checkpoint pufferlib_market/checkpoints/crypto15_tp03_s50_200/gpu0/c15_tp03_s78/best.pt \
  --data-path pufferlib_market/data/crypto15_daily_val.bin \
  --eval-hours 180 --n-windows 50 --deterministic --disable-shorts \
  --fee-rate 0.001 --fill-slippage-bps 5 --max-leverage 1.0 --device cpu
```

### 5. Algorithm Optimization
If time permits and systems are healthy:
- Check if any new checkpoints in `pufferlib_market/checkpoints/` outperform deployed ones
- Run sweeps on worksteal parameters if current config is underperforming
- Compare live PnL to simulator predictions

## Actions to Take
- Fix any errors found (restart services, fix code, update configs)
- Update `binanceprod.md` with findings
- If a better checkpoint exists, deploy it (update launch.sh, restart service)
- Commit and push all changes

## Important Rules
- NEVER force-push or reset --hard
- Always run tests before deploying code changes
- Use `uv pip` not `pip`
- Use background commands (`&` or run_in_background)
- Be concise in logs, no emoji
- This is real money - be careful with any changes that affect live trading
