# Binance Production Systems

## CRITICAL (2026-04-07): Lag=0 Validation Is Untrustworthy
`robust_champion` was briefly deployed after a "+216% Sort=25.06" holdout — this was run at `decision_lag=0` which embeds lookahead bias (agent sees bar t features and trades at bar t close, impossible live). At realistic `decision_lag=2` on the SAME `mixed23_latest_val` data (50 windows x 30h):

| Checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| robust_champion (lag=0 claim: +12.4%/+3.35) | -2.8% | -0.50 | -14.0% |
| robust_reg_tp005_dd002 | +0.0% | +0.76 | -26.9% |
| robust_reg_tp005_ent | -5.5% | -2.09 | -21.1% |

**PROTOCOL**: Never trust a holdout unless run with `--decision-lag 2` AND a slippage sweep (0/5/10/20 bps). `evaluate_holdout.py` now has `--slippage-bps`. The C training env has a built-in `t_obs = t-1` single-bar lag, but production exhibits >=2 bars lag (forecast build + Gemini call + order placement). Training sim vs production gap is the ROOT CAUSE of why every deploy since 2026-03 looks great in holdout and drifts flat/negative live.

Launch.sh reverted to dd002 pending supervisor restart:
`sudo supervisorctl restart binance-hybrid-spot`

Artifacts: `analysis/robust_champion_validation/slip{0,5,10,20}.json`

## Active Services

### 1. binance-hybrid-spot
- **Supervisor**: `binance-hybrid-spot` (auto-restart)
- **Entry**: `rl_trading_agent_binance/trade_binance_live.py`
- **Launch**: `deployments/binance-hybrid-spot/launch.sh`
- **Config**: `deployments/binance-hybrid-spot/supervisor.conf`
- **Python**: `.venv313`
- **Symbols**: BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD + ADA XRP SUI (positions held)
- **Mode**: Cross-margin, 0.5x leverage, hourly cycles
- **RL model**: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- **Previous model**: `robust_reg_tp005_ent` (switched 2026-03-31, dd002 has better 120d: +54.6% Sort=2.85 vs +42.4% Sort=2.71)
- **LLM**: Gemini 3.1 Flash Lite (thinking=HIGH)
- **Equity**: ~$3,025 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-hybrid-spot-error.log`

**Cycle flow**:
1. Fetch portfolio + margin balances
2. Compute Chronos2 h24 forecasts
3. Run RL inference (pufferlib checkpoint)
4. Build prompt -> call Gemini for allocation
5. Place/cancel margin limit orders
6. Sleep 3600s

**Current behavior** (2026-03-31):
- Gemini very defensive: 65-77% cash allocation
- RL consistently signals LONG_XRP and LONG_ADA
- DOGE position being sold down via existing sell order
- Stale orders auto-cancelled after 2h
- **Model switch** (2026-03-31): `robust_reg_tp005_dd002` deployed
  - 120d backtest: +54.6% ret, Sort=2.85, DD=26.9%, 50 trades, 54% WR
  - Exec param sweep: slippage insensitive, fill buffer insensitive
  - Better than ent at all periods except 30d

### 2. binance-worksteal-daily
- **Supervisor**: `binance-worksteal-daily` (auto-restart)
- **Entry**: `binance_worksteal/trade_live.py`
- **Launch**: `deployments/binance-worksteal-daily/launch.sh`
- **Config**: `deployments/binance-worksteal-daily/supervisor.conf`
- **Python**: `.venv313`
- **Universe**: `binance_worksteal/universe_live.yaml` (15 symbols)
- **Mode**: Spot+margin, dip-buying with SMA-20 filter
- **Equity**: ~$3,044 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-worksteal-daily-error.log`
- **State**: `binance_worksteal/live_state.json`
- **Trades**: `binance_worksteal/trade_log.jsonl`

**Parameters** (from launch.sh):
- `--max-position-pct 0.10`
- `--rebalance-seeded-positions`
- dip_pct=0.20, tp=20%, sl=15%, trail=3%, maxpos=5, maxhold=14d, sma=20

**Current positions** (2026-03-31):
- ADAUSD: entry $0.2726 (Mar 25), target $0.27, stop $0.23
- ZECUSD: entry $238.05 (Mar 25), target $285.66, stop $202.34
- XRPUSD: entry $1.36 (Mar 26), target $1.42, stop $1.16

## Monitoring

```bash
# Check service status
sudo supervisorctl status

# View recent logs
tail -100 /var/log/supervisor/binance-hybrid-spot-error.log
tail -100 /var/log/supervisor/binance-worksteal-daily-error.log

# Check running processes
ps aux | grep trade | grep -v grep

# Restart services
sudo supervisorctl restart binance-hybrid-spot
sudo supervisorctl restart binance-worksteal-daily

# View worksteal positions
python3 -m json.tool binance_worksteal/live_state.json
```

## Deployment

```bash
# After code changes, restart to pick up new code:
sudo supervisorctl restart binance-worksteal-daily
sudo supervisorctl restart binance-hybrid-spot

# Or kill the process (supervisor auto-restarts):
kill -TERM $(pgrep -f "trade_live.*--live")
kill -TERM $(pgrep -f "trade_binance_live")
```

## Known Issues (2026-03-31)

### Fixed: LOT_SIZE sell failures (worksteal)
- **Symptom**: Positions stuck 6 days, all sell orders rejected by Binance
- **Root cause**: Running process (started Mar 26) predated `_prepare_limit_order` quantization
- **Fix**: Restarted process to pick up code with exchange filter quantization
- **Prevention**: `_prepare_limit_order` now quantizes qty to stepSize before submission

### Fixed: Invalid symbols in universe_v2.yaml
- HYPEUSDT, KASUSDT removed (not valid Binance margin symbols)
- Was causing APIError -1121 every scan cycle

### Fixed: Margin 15% limit-price spam
- ENJUSD deep-dip buy orders were being submitted every 4h and rejected (code -3064)
- Added pre-check in `_prepare_limit_order`: rejects orders >14% from index price

## Market Simulator Validation

```bash
# Validate pufferlib model with binary fills (ground truth):
python -m pufferlib_market.evaluate_fast \
  --checkpoint <path> \
  --data-path pufferlib_market/data/crypto15_daily_val.bin \
  --eval-hours 720 --n-windows 100 \
  --fee-rate 0.001 --fill-slippage-bps 5

# Validate worksteal strategy:
python -m binance_worksteal.backtest --days 30

# Multi-period eval:
python -m pufferlib_market.evaluate_multiperiod \
  --checkpoint <path> \
  --data-path <bin> \
  --periods 1d,7d,30d --json
```

## Realism Parameters
- fee=10bps, margin=6.25% annual, fill_buffer=5bps, max_hold=6h
- validation_use_binary_fills=True
- fill_temperature=0.01
- decision_lag>=2
- Test at slippage 0/5/10/20 bps before deploying
