# Production Trading Systems

## Active Deployments

### 1. Binance Hybrid Spot (`binance-hybrid-spot`) -- BROKEN -- Gemini key revoked + RL obs mismatch
- **Bot**: `rl-trading-agent-binance/trade_binance_live.py`
- **Launch**: `deployments/binance-hybrid-spot/launch.sh`
- **Model**: Pufferlib robust_reg_tp005_ent (h1024 MLP, PPO) + RL-only fallback when Gemini unavailable
- **RL Checkpoint**: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_ent/best.pt`
- **RL Marketsim**: +191.4% return, Sort=23.94, 59% WR, 80 trades/30d (C sim, binary fills, 5bps slippage)
- **Slippage robustness**: +181.6% @0bps, +191.4% @5bps, +252.8% @10bps, +194.3% @20bps, +168.1% @30bps
- **Config**: obs_norm, wd=0.05, slip=8bps, tp=0.005, ent_anneal 0.08->0.02
- **A40 sweep cost**: $0.53 for 50 trials
- **Symbols**: BTCUSD, ETHUSD, SOLUSD, DOGEUSD, AAVEUSD, LINKUSD
- **Mode**: Cross-margin, leverage reduced from 5x to 0.5x
- **Max hold**: 6h forced exit
- **Fees**: 10bps maker
- **Equity**: ~$3,056 (down from ~$3,333 on Mar 21)
- **Marketsim (30d Feb-Mar)**: +10.0%, Sort=8.74, 293 entries, 2.94% MaxDD
- **RL signal broken**: Always outputs LONG_UNI due to 23-sym model fed 6-sym obs. Fix: action masking for tradable symbols.
- **Eval command**:
  ```bash
  # Backtest is built into the bot's Chronos2 fallback path
  # For pufferlib models use:
  source .venv313/bin/activate
  python -m pufferlib_market.evaluate \
    --checkpoint <path> \
    --data-path pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin \
    --max-steps 720 --fee-rate 0.001 --fill-slippage-bps 5.0 \
    --num-episodes 20 --hidden-size 1024 --arch mlp --deterministic
  ```

### 2. Binance Worksteal Daily (`binance-worksteal-daily`) -- RUNNING
- **Bot**: `binance_worksteal/trade_live.py`
- **Launch**: `deployments/binance-worksteal-daily/launch.sh`
- **Strategy**: Rule-based dip-buying, SMA-20 filter, 30-symbol universe
- **Config**: dip=20%, tp=15%, sl=10%, trail=3%, max_positions=5, max_hold=14d
- **Equity**: ~$3,300
- **Marketsim (30d Feb-Mar)**: +9.13%, Sort=23.03, -1.5% MaxDD, 73% WR
- **Eval command**:
  ```bash
  source .venv313/bin/activate
  python binance_worksteal/backtest.py --days 30
  ```
- **Diagnostics**: `python binance_worksteal/trade_live.py --diagnose --symbols BTCUSD ETHUSD`

### 3. Alpaca Stock Trader (`unified-stock-trader`) -- STOPPED
- **Checkpoint**: `unified_hourly_experiment/checkpoints/wd_0.04/epoch_009.pt`
- **Architecture**: classic h512, 6L, 8 heads, seq=48
- **Symbols**: NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT
- **Marketsim (30d)**: Sort=2.68, +1.71%, 33 trades
- **Equity**: ~$46,467

## Best Candidate Models (Not Yet Deployed)

### Pufferlib PPO Models (C sim, binary fills, trustworthy)
| Model | Val Return | Sortino | WR | Slippage Test | Status |
|-------|-----------|---------|-----|---------------|--------|
| ent_anneal | +28.3% | 8.04 | 58% | +20.2% @10bps | obs mismatch with live bot |
| robust_champion | +1.80 | 4.67 | 65% | -- | obs mismatch |
| per_env_adv_smooth | +1.02% | 2.93 | 59% | -- | obs mismatch |

**Deployment blocker**: Pufferlib models use portfolio-level observations (obs_dim=396 for 23 symbols) while `rl_signal.py` expects single-symbol obs (obs_dim=73). Need to modify `rl-trading-agent-binance/rl_signal.py` to construct portfolio obs from all symbols simultaneously.

### Binanceneural Models (Soft fills -- LOOKAHEAD BIAS)
- crypto_portfolio_6sym: Sort=181 train, Sort=-5 at lag>=1 on binary sim
- **DO NOT DEPLOY** these models -- soft sigmoid fills leak information
- Fixed config: `validation_use_binary_fills=True`, `fill_temperature=0.01`, train with `decision_lag_bars=2`

## Marketsimulator Realism Checklist
- [x] Fee: 10bps maker (0.001)
- [x] Margin interest: 6.25% annual
- [x] Fill buffer: 5bps
- [x] Max hold: 6h forced exit
- [x] Decision lag: >=1 bar (lag=0 is cheating)
- [x] Binary fills for validation (not soft sigmoid)
- [x] No EOD close (GTC orders, hold overnight)
- [x] Slippage test: 0, 5, 10, 20 bps
- [ ] Multi-start-position evaluation (holdout windows)
- [ ] Cross-validation across time periods

## Deploy Command
```bash
# Deploy pufferlib model (once rl_signal.py fixed):
bash scripts/deploy_crypto_model.sh <checkpoint_path> BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD

# Revert to Gemini+Chronos2:
bash scripts/deploy_crypto_model.sh --remove-rl
```

## Security
- API keys must go in `.env.binance-hybrid` (gitignored), never in supervisor.conf or code
- Revoked key: `AIzaSyAHHu9-eq3YufxMcCFOWjpyff9pgrOXoX0` (already in git history, cannot be un-leaked)
- Generate new key at Google AI Studio and place in `.env.binance-hybrid`

## Monitoring
```bash
sudo supervisorctl status binance-hybrid-spot
sudo tail -50 /var/log/supervisor/binance-hybrid-spot.log
sudo tail -20 /var/log/supervisor/binance-hybrid-spot-error.log
```

## Incidents

### 2026-03-22 18:54 UTC -- Gemini API key revoked
- **What**: Gemini API key revoked/leaked. Bot received 403 PERMISSION_DENIED for 13+ hours. RL fallback also broken due to obs mismatch (always outputs LONG_UNI for non-configured symbol). Both primary and fallback signals broken simultaneously. No trades executed.
- **Impact**: Portfolio dropped from $3,333 (Mar 21) to $3,056 (Mar 23). Still holding 107.78 LINK (~$980).
- **Root cause**: API key was hardcoded in supervisor.conf and committed to git. Google detected it as leaked.
- **Remediation**: Move API key to gitignored `.env.binance-hybrid`, reduce leverage 5x->0.5x, add action masking to RL signal.
