# Current Algorithms — Monitor Ledger

**Authoritative source**: `alpacaprod.md` top section is always the canonical
live ledger. This file is a short pointer for the hourly monitor so it knows
which services should exist, where the best config lives, and what the bar
is to beat. Keep this file in sync with `alpacaprod.md` whenever we deploy.

Last synced: 2026-04-18 (off-market Saturday)

---

## 1. Daily RL Trader — LIVE

- **Process**: `trade_daily_stock_prod.py --daemon --live` under supervisor unit `daily-rl-trader` (launcher `deployments/daily-rl-trader/launch.sh`). The old systemd unit `daily-rl-trader.service` is intentionally inactive.
- **Broker boundary**: all writes go via `trading_server` on `127.0.0.1:8050` (supervisor unit `trading-server`). Account = `live_prod`. Singleton enforced by `src/alpaca_singleton.py`.
- **Ensemble**: v7 12-model (`src/daily_stock_defaults.py::DEFAULT_EXTRA_CHECKPOINTS`), primary `C_s7.pt`. Agreement gate `--min-agree-count 2` active since 2026-04-17 13:03Z.
- **Allocation**: `--allocation-pct 12.5`, leverage 1× (do NOT autonomously bump either).
- **Deploy bar** (screened32 realism gate, 263 windows, fb=5, binary fills, lag=2):
  - 1× med monthly ≥ +7.47% / p10 ≥ +3.18% / neg ≤ 10/263 / sortino ≥ 6.74
  - 1.5× knee med ≥ +10.33%, sortino ≥ 6.13
- **Signal cadence**: one `DAILY STOCK RL SIGNAL` line per trading day near 13:35 UTC (market open). Flat action at ~5-7% confidence is calibrated (val p90 ≈ 0.055–0.061). Only escalate if flat for >3 days AND conf > 0.10, or never-flat.
- **Equity baseline**: ≈$28k–$30k. Record exact value each run.

## 2. XGB Daily Trader — QUEUED (paper API blocker)

- **Process (when live)**: systemd unit `xgb-daily-trader.service` runs `python -m xgbnew.live_trader`. **DEAD since 2026-04-14** (paper REST 401 unauthorized). Live PROD keys fine; paper keys must be regenerated at the Alpaca dashboard and pasted into `env_real.py` (`ALP_KEY_ID_PAPER`, `ALP_SECRET_KEY_PAPER`). No other action needed — unit will restart fine once keys land.
- **Queued config** (ship after key fix):
  - `--model-path analysis/xgbnew_daily/live_model_v2_n400_d5_lr003_top1.pkl`
  - `--top-n 1 --allocation 1.0`
  - XGBoost `n_estimators=400, max_depth=5, learning_rate=0.03`, 846-symbol universe, seed-robust per 16-seed Bonferroni.
- **OOS evidence** (846-sym, 34 windows thru 2026-04-18, binary fills, lag=2, fee=0.278bps, fb=5bps):
  - 1.0× → med +32.2%/mo, p10 +20.3%, worst_dd 31.9%, **0/34 neg**, sortino 8.42
  - 1.25× → med +40.8%, p10 +25.1%, dd 39.0%, 0/34 neg
  - 1.5× → med +49.7%, p10 +29.6%, dd 45.8%, 0/34 neg
- **In-flight research**: DD-reduction sweep in `analysis/xgbnew_dd_sweep/` (SPY MA50 gate, vol-target sizing, seed variance). Ledger `xgboptimiztions.md`. If any round-2 cell beats baseline on (Δ sortino ≥ 0 AND Δ neg ≤ 0) at equal-or-better median, deploy that config instead of bare lev=1×.
- **Character**: buy-open / sell-close same session, flat overnight. `top_n=1 + allocation=1.0` → 100% single-stock concentration per session. No overnight margin hit.

## 3. Other services (present but not primary signal)

- `llm-stock-trader` — Gemini-driven picks YELP/NET/DBX/OPTX. Status drift; check supervisor, not urgent.
- `unified-orchestrator.service` — **DEAD** since 2026-03-29 (Gemini API exhausted). Do not attempt to revive without budget.
- Crypto RL (crypto12_ppo_v8) — training-snapshot only, **not OOS-validated**, do NOT deploy.

## 4. Realism invariants (marketsim truth vs prod truth)

- Ground truth for gate decisions = pufferlib C env with `validation_use_binary_fills=True`, `fill_buffer=5bps`, `fee=10bps` (XGB uses `fee=0.278bps` to match Alpaca real), `decision_lag=2`, `max_hold=6h`.
- Soft sigmoid fills have lookahead bias — never trust training sortino alone for deploy. Re-eval with binary fills at slippage 0/5/10/20 bps before any swap.
- Intrabar replay (`pufferlib_market/intrabar_replay.py`) is the bar-granularity visualizer; MP4/HTML artifacts under `models/artifacts/<run>/videos/`.

## 5. Where the hourly monitor's deploy authority starts and stops

**Auto-deploy authorized** (no human confirmation needed):
- Replace one v7 ensemble member with a passing 14th-candidate if all five 1× deploy bars clear AND Δp10 ≥ 0 AND Δneg ≤ 0 AND Δsortino ≥ 0 on a current screened32_realism_gate run.
- Restart a dead supervisor unit that owns a known-healthy config.
- Fix portfolio state with `health_check.py --fix`.

**Explicitly NOT auto-deploy** (human-only):
- Paper or live API key rotation.
- `--allocation-pct` change.
- `--leverage` change.
- Adding `--multi-position` > 1.
- Flipping XGB from paper to live.
- Any change to `alpaca_singleton.py` or the death-spiral guard.

See `monitoring/hourly_prod_check_prompt.md` for the full per-phase procedure.
