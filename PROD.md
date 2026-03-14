# Production Trading System

## Service: `unified-orchestrator.service`

```bash
# Status
sudo systemctl status unified-orchestrator.service

# Logs (live tail)
journalctl -u unified-orchestrator.service -f

# Logs (last N hours)
journalctl -u unified-orchestrator.service --since "6 hours ago" --no-pager

# Restart
sudo systemctl restart unified-orchestrator.service

# Stop
sudo systemctl stop unified-orchestrator.service
```

### Service file: `/etc/systemd/system/unified-orchestrator.service`

- **WorkingDirectory**: `/nvme0n1-disk/code/stock-prediction`
- **ExecStart**: `.venv313/bin/python -m unified_orchestrator.orchestrator --live`
- **Restart**: `always` (RestartSec=30)
- **User**: `administrator`
- **Env**: `GEMINI_API_KEY`, `PYTHONPATH`, `PYTHONUNBUFFERED=1`

## Account

- **Alpaca LIVE** (not paper) — uses `ALP_KEY_ID_PROD` / `ALP_SECRET_KEY_PROD` from `env_real.py`
- The default `alpaca_wrapper.py` connects to PAPER; orchestrator uses PROD keys directly
- Equity: ~$40k as of 2026-03-14

## What It Trades

### Crypto (24/7, CRYPTO_ONLY regime outside stock market hours)
- Symbols: BTCUSD, ETHUSD, SOLUSD, LTCUSD, AVAXUSD
- RL checkpoint: `pufferlib_market/checkpoints/autoresearch/longonly_forecast/best.pt`
- Gemini LLM overlay: `gemini-3.1-flash-lite-preview` refines RL signals with limit prices/confidence
- Max hold: 6 hours, trailing stop: 0.3-0.5%
- TP sell orders placed each cycle, canceled and replaced hourly

### Stocks (market hours only)
- Symbols: META, MSFT, NET, NVDA, PLTR (and others via meta-selector)
- Uses separate neural policy checkpoints

## Hourly Cycle

1. Build portfolio snapshot (Alpaca positions, equity, leverage)
2. Determine regime: CRYPTO_ONLY, STOCK_ONLY, or MIXED
3. For crypto: load RL model → get directional signal → call Gemini for limit prices & confidence
4. Place/adjust TP sell orders for existing positions
5. Execute new buy orders as limit orders
6. Sleep until next :01 UTC

## Known Issues

### Low Volume on Alpaca Crypto
- LTC has 4/18 zero-volume hours on Alpaca
- AVAX has 2/18 zero-volume hours
- BTC volume often <0.1 BTC/hour ($7k)
- Limit orders frequently expire unfilled (15-52min fill times observed)
- The bot's $3-10k orders are a significant share of Alpaca crypto volume
- **Impact**: many buy attempts cancel (04-05h UTC typical dead zone)

### Duplicate Processes
- If the systemctl service crashes (e.g., syntax error in code), it restarts in 30s
- Manual restarts via Claude/nohup can create duplicate instances
- **Always check**: `ps aux | grep orchestrator.*--live | grep -v grep`
- Kill duplicates: keep the systemctl one (check PID in `systemctl status`)

### Gemini Error: `allocation_pct`
- The `TradePlan` schema changed; older orchestrator code references `allocation_pct`
- Fixed in latest gemini_wrapper.py but can resurface after bad merges

### TP Sell `insufficient balance` Errors
- Happens when existing limit sell is pending and a new one tries to sell the same qty
- The cancel→place sequence has a race condition
- Harmless: next cycle will retry

## Monitoring Checklist

```bash
# Is it running?
systemctl status unified-orchestrator.service

# Any errors in last hour?
journalctl -u unified-orchestrator.service --since "1 hour ago" | grep -i error

# Duplicate processes?
ps aux | grep "orchestrator.*--live" | grep -v grep | wc -l
# Should be exactly 1

# Current positions (LIVE account)
python -c "
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from alpaca.trading.client import TradingClient
a = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)
for p in a.get_all_positions():
    print(f'{p.symbol:8s} qty={float(p.qty):12.6f} pnl=\${float(p.unrealized_pl):8.2f}')
print(f'Equity: \${float(a.get_account().equity):,.2f}')
"

# Recent fills
journalctl -u unified-orchestrator.service --since "6 hours ago" | grep filled

# Replay RL signals vs prod
python replay_recent_hours.py
```

## Files

| File | Purpose |
|------|---------|
| `unified_orchestrator/orchestrator.py` | Main hourly loop |
| `unified_orchestrator/rl_gemini_bridge.py` | RL+Gemini hybrid pipeline |
| `unified_orchestrator/position_tracker.py` | Entry times, peaks, trailing stops |
| `unified_orchestrator/alpaca_watcher.py` | Alpaca order/position mgmt |
| `unified_orchestrator/backout.py` | Deleverage / exit-only |
| `llm_hourly_trader/gemini_wrapper.py` | Gemini API wrapper + TradePlan schema |
| `llm_hourly_trader/providers.py` | LLM provider abstraction |
| `pufferlib_market/inference.py` | RL model inference |
| `env_real.py` | API keys (PROD + PAPER) |
| `replay_recent_hours.py` | Compare RL signals vs prod decisions |
