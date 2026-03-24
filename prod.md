# Production Trading Systems

## Active Deployments

### 1. Binance Hybrid Spot (`binance-hybrid-spot`) -- BROKEN -- Gemini key revoked + RL obs mismatch
- **Bot**: `rl-trading-agent-binance/trade_binance_live.py`
- **Launch**: `deployments/binance-hybrid-spot/launch.sh`
- **Model**: Pufferlib robust_champion (h1024 MLP, PPO) + RL-only fallback when Gemini unavailable
- **RL Checkpoint**: `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt`
- **50-window holdout** (seed=42, 30-bar windows, deterministic, no early stop):
  - Median return: +10.80%, Positive windows: 76%, Median Sortino: 3.28
  - 60-bar windows: 100% positive, full-span: +58.22% return, 26.58% max DD
  - Cross-seed (250 windows across 5 seeds): 76% positive, p5=-10.80%
- **Config**: obs_norm=True, wd=0.005, cosine LR, slip=5bps, tp=0.05, anneal_lr+ent
- **Previous**: robust_reg_tp005_ent had +191.4% single-window but only +3.03% median across 50 windows
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

### 3. Alpaca Stock Trader (`unified-stock-trader`) -- RUNNING (hourly meta-selector)
- **Bot**: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- **Architecture**: Chronos2 hourly, multiple models + meta-selector
- **Symbols**: NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT, AAPL, MSFT, META, TSLA, NET, BKNG, EBAY, EXPE, ITUB, BTG, ABEV
- **Equity**: ~$41,145 (2026-03-24; was $46,467 — dropped during crash-loop outage)
- **Strategies**: wd_0.06_s42:8 + wd_0.06_s1337:8 (2-strategy meta-selector)
- **NOTE (2026-03-24)**: Was crash-looping since ~Mar 19 — supervisor config referenced 5 missing checkpoints
  (wd_0.04, wd_0.05_s42, wd_0.08_s42, wd_0.03_s42, stock_sortino_robust_20260219b/c).
  Fixed by replacing with wd_0.06_s42:8 + wd_0.06_s1337:8 (only 2 strategies remain locally).
  Previous equity loss (~$5k) likely from pre-existing positions before outage, not model error.

### 4. Alpaca Daily PPO Trader (`trade_daily_stock_prod.py`) -- READY TO DEPLOY (tp05 ensemble)
- **Architecture**: h=1024 MLP PPO, stocks12 (AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,JPM,V,AMZN,PLTR)
- **Primary checkpoint**: `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt`
- **Ensemble member**: `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s15/best.pt`
  - tp05_s15 = seed=15, 128 envs, no bf16, 35M steps, best at update_000950
- **3-model ensemble (s123+s15+s36 softmax_avg) holdout eval** (50 windows, Sep2025-Mar2026, default_rng(42)):
  - Median: **+47.30%** / 90 days (+64% over 2-model s123+s15)
  - P10: **+29.93%** (+83% over 2-model s123+s15's 16.37%)
  - Worst window: **+20.97%** (+106% over 2-model s123+s15's 10.17%)
  - Negative windows: **0/50 (0%)** — ZERO negative windows
- **Slippage robustness** (3-model ensemble, 50-window, default_rng(42)):
  - @5bps:  med=47.30%, p10=29.93%, worst=20.97%, neg=0/50
  - @10bps: med=47.30%, p10=29.93%, worst=20.97%, neg=0/50 (identical)
  - @20bps: med=48.60%, p10=31.54%, worst=22.47%, neg=0/50
  - @50bps: med=37.92%, p10=22.79%, worst=14.94%, neg=0/50
- **tp05_s36**: seed=36, 128 envs, no bf16, 35M steps — 0/50 neg standalone, med=26.80%
- **tp05_s15**: seed=15, 128 envs, no bf16, 35M steps — 0/50 neg standalone, med=28.76%
- **Previous 2-model ensemble** s123+s15: med=28.76%, p10=16.37% (superseded 2026-03-24)
- **Previous 3-model ensemble** rmu2201+8597+5526: med=14.94%, p10=7.64% (kept for reference)
- **Config**: h=1024, lr=3e-4, ent=0.05, trade_penalty=0.05, dp=0.0, slip=0bps (training), anneal_lr=True
- **Launch**: `deployments/daily-stock-ppo/launch.sh`
- **Supervisor**: `deployments/daily-stock-ppo/supervisor.conf` (autostart=false — enable manually)
- **WARNING**: Symbol conflict with unified-stock-trader (both trade AAPL/MSFT/NVDA/GOOG/META/TSLA/PLTR) — coordinate before enabling both simultaneously
- **NOTE**: Previous default (stocks12_daily_tp05_longonly) scored -2.55% median, 32/50 negative
- **NOTE**: random_mut_2272 (prev deployed) scored -5.14% median, 29/50 negative
- **NOTE**: seed variance is extreme — seed=7 gives -13.6% median, seed=123 gives +16.5% median (same config)
- **Deploy command**:
  ```bash
  source .venv313/bin/activate
  python trade_daily_stock_prod.py --live
  # Uses 3-model ensemble (s123+s15+s36 softmax_avg) — DEFAULT_EXTRA_CHECKPOINTS in code
  # To restore 2-model: python trade_daily_stock_prod.py --live --extra-checkpoints <s15_path>
  # To restore standalone s123: python trade_daily_stock_prod.py --live --no-ensemble
  ```
- **Eval command**:
  ```bash
  source .venv313/bin/activate
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt \
    --data-path pufferlib_market/data/stocks12_daily_val.bin \
    --eval-hours 90 --n-windows 50 --seed 42 \
    --fee-rate 0.001 --fill-buffer-bps 5.0 --deterministic --no-early-stop
  # NOTE: lag=0 IS correct for daily models (1-bar obs lag built-in to C env)
  # Bot runs at 9:35 AM, drops today's incomplete bar, uses T-1 data → fills at T OPEN
  ```
- **Why ensemble works here**: unlike rmu models, s15 was trained with same tp05 config. Both models
  pick only high-conviction trades (~10/90d). softmax_avg of two similar architectures diversifies
  seed variance rather than diluting edge. Ensemble p10/worst strictly better than either alone.
- **Seed sweep results** (tp05 old-config: 128 envs, no bf16, 35M steps):
  - seed=15: **0/50 neg, med=39.14%, p10=17.72%** — DEPLOYED in ensemble
  - seed=33: 32/50 neg — bad
  - seed=3 (old-config): 1/50 neg best case, p10<5% — not deployable
  - Sweep continuing: seeds 16-32 (stream A), 34-50 (stream B)

## Confirmed 0/50-neg Models (50-window EnsembleTrader, default_rng(42))
All evaluated via batch 50-window test (deterministic, 5bps fill buffer, no early stop):
| Model | Checkpoint | med | p10 | worst | neg/50 | Notes |
|-------|-----------|-----|-----|-------|--------|-------|
| **ensemble s123+s15+s36** | s123+s15+`stocks12_seed_sweep/tp05_s36/best.pt` | **+47.30%** | **+29.93%** | **+20.97%** | **0** | **PRODUCTION — BEST** (2026-03-24) |
| ensemble s123+s15 | s123+`stocks12_seed_sweep/tp05_s15/best.pt` | +28.76% | +16.37% | +10.17% | 0 | Superseded by 3-model (2026-03-24) |
| **tp05_s36 standalone** | `stocks12_seed_sweep/tp05_s36/best.pt` | +26.80% | +10.08% | +3.43% | **0** | New discovery (2026-03-24) |
| **tp05_s15 standalone** | `stocks12_seed_sweep/tp05_s15/best.pt` | +28.76% | +14.50% | +8.42% | **0** | In 3-model ensemble (2026-03-24) |
| tp05_s123 standalone | `stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt` | +15.97% | +10.81% | +5.62% | 0 | Primary model (superseded by ensemble) |
| tp03/v2_sweep | `stocks12_v2_sweep/stock_trade_pen_03/best.pt` | +15.54% | +8.82% | +3.45% | 0 | Backup (tp=0.03, obs_norm=False) |
| rmu2201 | `autoresearch_stock/random_mut_2201/best.pt` | +12.31% | +3.55% | +0.38% | 0 | 3rd option; ensemble with s123 HURTS (p10→3.62%) |

**Full v2_sweep exhaustive eval (2026-03-24, 47 checkpoints):**
- No new 0/50-neg deployable models found beyond the 4 confirmed (tp05_s123, tp03/v2_sweep, reg_combo, rmu2201)
- Best near-misses: `h1024_a40` (4/50 neg, med=6.89%), `ent_005_a40` (6/50 neg, med=9.74%)
- `stock_ent_05` (med=29.37%!) has explosive median but 19/50 neg — too inconsistent
- tp05 default seed: 11/50 neg, med=7.73% — seed=123 is genuinely rare
- tp05 seed=7: 39/50 neg — confirms extreme seed variance
- A40-trained consistently beats H100-trained on same config (suggests A40 training setup is better for stocks12)
- Muon optimizer, obs_norm, high gamma, slip training, drawdown penalty: all bad for stocks12

**Failed experiments (2026-03-24):**
- tp03_s123 (tp=0.03, obs_norm=False, seed=123, 100M steps): **21/50 neg** — NOT deployable. High in-sample metric (188.7) but terrible OOS. The "lucky seed 123" effect is specific to tp=0.05 config.
- tp03_obs_norm_s123 (tp=0.03, obs_norm=True, seed=123, partial@update550): **28/50 neg** — NOT deployable. obs_norm caused training instability (peaked early then collapsed).

**Key findings:**
- softmax_avg ensemble of s123+tp03/v2_sweep HURTS (4/50 neg, p10=0.46%) — they don't complement via ensemble
- softmax_avg ensemble of s123+rmu2201 HURTS (p10→3.62%) — same issue
- Running each model STANDALONE is the best strategy; ensembling via softmax_avg dilutes strong signals
- 20-window holdout (seed=1337) is unreliable: tp03/v2_sweep scored -21.27 (rejected!) but is 0/50 neg; stock_drawdown_pen scored 0/20 neg but is 21/50 neg in 50-window
- **Only trust 50-window EnsembleTrader with default_rng(42) for deployment decisions**

## Model Search Noise Floor (2026-03-23)

### Active seed sweep (2026-03-24, ongoing)
- **Config**: tp05 (trade_penalty=0.05, h=1024, anneal_lr, no obs_norm, 35M steps, 128 envs, no bf16)
- **CSV**: `autoresearch_tp05_seeds_oldcfg_leaderboard.csv`
- **Streams**: A (seeds 15-32 sequential), B (seeds 33-50 sequential) running in parallel
- **Seeds tested** (final eval via standalone 50-window EnsembleTrader, default_rng(42)):
  - **seed=15: 0/50 neg, med=39.14% (CSV) / 28.76% standalone** — in 3-model ensemble (2026-03-24)
  - **seed=36: 0/50 neg, med=26.80% standalone** — ADDED to 3-model ensemble (2026-03-24)
  - seed=18: 14/50 neg — not deployable
  - seed=19: 30/50 neg — bad
  - seed=30: 44/50 neg — bad
  - seed=33: 32/50 neg — bad
  - seed=34: 45/50 neg — bad
  - seed=35: 24/50 neg — bad
  - seed=37: 39/50 neg — bad
  - seed=3 (old-config): 1/50 neg best case, p10<5% — not deployable
  - Seeds 16, 17: 44/50, 50/50 neg respectively — very bad
- **CRITICAL FINDING**: Training config determines which seeds produce trading models vs hold-cash:
  - 128 envs + bf16 + cuda-graph: ONLY seed=123 reliably escapes hold-cash
  - 128 envs + no bf16 + no cuda-graph + 35M steps: seed=15 gives 0/50 neg, seed=33 bad
  - The production tp05_s123 checkpoint was saved at update=44 (1.44M steps!) — very early in training
- **Deployment bar**: 0/50 neg via 50-window EnsembleTrader with default_rng(42)

### Seed variance characterization — how many seeds produce "real" results?

**rmu2201 base config** (h=256, ent=0.08, slip=12, dp=0.01, sp=0.005, wd=0.0, global advantage_norm):
- 24 seeds tested (30-window eval, seed=42)
- Positive median rate: ~7/24 = 29% of seeds produce positive median returns
- Deployment-quality rate: **0/24** (need ≤5% neg rate = ≤1.5/30 neg)
- Best seed: s14 (16.45% med, 6/30 neg → 50-win: 12.95%, 13/50 neg) — NOT deployable
- High-variance outlier: s23 (23.95% med, 22/50 neg = 44% neg) — high mean, terrible consistency
- **Conclusion**: rmu2201 base config produces ~0% deployment-quality rate. The original random_mut_2201 was an extremely lucky accident.

**per_env advantage_norm config** (h=256, ent=0.08, slip=12, dp=0.01, sp=0.005, wd=0.0, per_env advantage_norm):
- Discovered via autoresearch trial: random_mut_8597 (seed=1168) → 9.38% med, 5/50 neg (10%) — improvement!
- 3-seed variance characterization running (seeds 42, 123, 777) — see autoresearch_stocks12_per_env_sweep.csv
- **Deployment target**: ≤5% neg rate (≤2.5/50 windows). random_mut_2201 achieves 2%.
- **Working hypothesis**: per_env advantage normalization is more stable for multi-asset portfolios

### Interpretation guide
- Score = -135.64 → degenerate policy (actively loses money, identical trajectory across models)
- Score = -50.0 → hold-cash policy (no trades, 0% return every window)
- Score > -100 → escaped degenerate state (real signal, worth 50-window deep eval)
- Deployment quality threshold: 50-window med >5%, neg <5% (i.e., <3/50 neg windows)

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

### 2026-03-24 -- 20-window holdout (seed=1337) is unreliable for model ranking
- **What**: Comprehensive 50-window EnsembleTrader eval of all stocks12 checkpoints revealed that the autoresearch 20-window holdout (seed=1337) misranks models in both directions.
- **False negatives**: tp03/v2_sweep (score=-21.27, 20-window rejected) is actually 0/50 neg via 50-window test. Similarly, tp05_s123 scored poorly in the original stochastic autoresearch before being revalidated.
- **False positives**: stock_drawdown_pen (0/20 neg in 20-window) is 21/50 neg in 50-window. stock_trade_pen_05 default seed (1/20 neg) is 12/50 neg.
- **Root cause**: 20-window test with seed=1337 samples ~30% of all possible start windows. Models with sparse trading can be lucky/unlucky in which windows are sampled.
- **Fix**: Use 50-window EnsembleTrader with default_rng(42) as the ONLY reliable deployment metric. The 20-window holdout_robust_score is only a coarse filter (keep score > -40 for follow-up, but false negatives exist).

### 2026-03-24 -- Triton fused kernels broke training on PyTorch 2.9
- **What**: `fused_mlp_relu` Triton kernels in `TradingPolicy` don't support autograd backward pass. Training crashed with "element 0 of tensors does not require grad".
- **Root cause**: Triton fused kernels were used in both train and eval mode. They're inference-only optimizations (no custom backward).
- **Fix**: Guard all fused paths behind `not self.training` check. Also wrapped CUDA graph capture in try/except with autograd validation.
- **Impact**: All local training was broken. Fixed in commit `4e05306f`.

### 2026-03-24 -- Crypto29 daily autoresearch (local 3090 Ti)
- **Best**: `robust_reg_tp005_sds02` (holdout=-231.5, val_ret=-0.076)
- Config: wd=0.05, obs_norm, 8bps slip, tp=0.005, smooth_downside_penalty=0.2
- **Findings**: Smooth downside penalty helps. Trade penalty 0.005 > 0.01. h512 hurts. Entropy annealing bad on daily crypto.
- Holdout scores all negative (-231 to -283): 29-sym daily at 90 steps has limited edge
- **Next**: Need hourly data (34-sym, 720 steps) on larger GPU for meaningful differentiation

### 2026-03-23 -- Daily stock PPO: autoresearch leaderboard metric was wrong
- **What**: The autoresearch leaderboard ranked random_mut_2272 as best (robust_score=-5.15) and random_mut_2201 as worst (-110.76) on the holdout set. In reality the ranking is inverted.
- **Root cause**: Autoresearch used stochastic policy + `enable_drawdown_profit_early_exit=True` for holdout eval. Both inflate results for mediocre models. Deterministic + no-early-stop is the correct production proxy.
- **Impact**: random_mut_2272 was deployed (now known to be -5.14% median, 29/50 negative). random_mut_2201 (the actual best: +11.74% median, 1/50 negative) was ranked last and not deployed.
- **Fix**: Added `--no-early-stop` flag to `evaluate_holdout.py`. Updated DEFAULT_CHECKPOINT to random_mut_2201. Always use `--deterministic --no-early-stop` for final candidate selection.
- **Note**: random_mut_2201 uses h=256 (NOT h=1024) — shows smaller networks with right config can outperform.

### 2026-03-22 18:54 UTC -- Gemini API key revoked
- **What**: Gemini API key revoked/leaked. Bot received 403 PERMISSION_DENIED for 13+ hours. RL fallback also broken due to obs mismatch (always outputs LONG_UNI for non-configured symbol). Both primary and fallback signals broken simultaneously. No trades executed.
- **Impact**: Portfolio dropped from $3,333 (Mar 21) to $3,056 (Mar 23). Still holding 107.78 LINK (~$980).
- **Root cause**: API key was hardcoded in supervisor.conf and committed to git. Google detected it as leaked.
- **Remediation**: Move API key to gitignored `.env.binance-hybrid`, reduce leverage 5x->0.5x, add action masking to RL signal.
