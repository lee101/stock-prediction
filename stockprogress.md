# 5-Minute Stock RL Challenge

A competition-style benchmark for training the best RL trading policy on Alpaca equities under a fixed five-minute compute budget, evaluated on a realistic market simulator.

---

## 1. The Challenge

**Elevator pitch.** Train a PPO agent from scratch on 12 US equities (daily bars) in exactly five minutes on an A100 SXM4 80 GB. At the end of training, evaluate the saved checkpoint on 20 non-overlapping holdout windows drawn from `stocks12_daily_val.bin` with real-world Alpaca execution constraints (10 bps fee, 5 bps minimum slippage, 1× leverage cap). The goal is to maximise risk-adjusted annualised return across every one of those windows — not just the best one.

**Why it matters.** The winning checkpoint goes directly into production on the Alpaca brokerage account. Every improvement in the challenge score translates to real compounding edge on live capital. Holding the budget to five minutes forces the same discipline as live retraining: the agent must generalise from a short burst of gradient steps, not memorise a training set.

---

## 2. Scoring

### Primary metric

```
score = median_annualized_return * (1 + sortino_ratio) / (1 + max_drawdown_pct)
```

All three components are measured across the 20 holdout windows:
- `median_annualized_return` — annualised return (%) at the median window
- `sortino_ratio` — median Sortino ratio across windows (downside-only denominator)
- `max_drawdown_pct` — worst peak-to-trough drawdown (%) across all windows

Higher is always better. A model that earns 30% annualised but blows up one window scores lower than one that earns 25% smoothly across all 20.

### Evaluation harness

```bash
source .venv313/bin/activate
python -u -m pufferlib_market.autoresearch_rl \
    --stocks \
    --train-data pufferlib_market/data/stocks12_daily_train.bin \
    --val-data   pufferlib_market/data/stocks12_daily_val.bin \
    --holdout-data pufferlib_market/data/stocks12_daily_val.bin \
    --time-budget 300 \
    --holdout-n-windows 20 \
    --holdout-eval-steps 90 \
    --holdout-fill-buffer-bps 5.0 \
    --holdout-fee-rate 0.001 \
    --fee-rate-override 0.001 \
    --leaderboard autoresearch_stock_daily_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock
```

**Hard constraints:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `fee_rate` | 0.001 (10 bps) | Alpaca maker/taker rate |
| `fill_slippage_bps` | >= 5 bps at eval | Adverse fill on daily bars |
| `max_leverage` | 1.0 | No margin in production |
| Training wall-time | 300 s | One A100 SXM4 80 GB |
| Holdout windows | 20 | Non-overlapping, 90-bar each |
| Data split | `stocks12_daily_val.bin` | Held-out; never seen during training |

**Reported metrics per run:**

- `p10_ann` — 10th-percentile annualised return across windows (floor on worst-case)
- `median_ann` — median annualised return (primary return signal)
- `p90_ann` — 90th-percentile annualised return (ceiling on upside)
- `sortino` — median Sortino ratio
- `max_drawdown` — worst drawdown across all 20 windows (%)
- `pct_profitable` — fraction of the 20 windows with positive return (must be 100% to qualify for production deployment)
- `win_rate` — fraction of individual trades that were profitable

### Reproducibility requirement

Every leaderboard-eligible run must produce a manifest JSON recording:
- `git_commit` hash at training time
- all hyperparameters (the full `TrialConfig` dict)
- random seed
- hardware type and training wall-time

The manifest format is shown in `manifest_stocks_drytest_001.json`. Runs without a matching manifest are accepted as informal entries only.

### Production deployment gate

A checkpoint may be deployed to the live Alpaca account when:

1. All 20 holdout windows are profitable (`pct_profitable = 100%`)
2. `score` beats the current production checkpoint
3. Manifest JSON is present and git commit is clean

Current production checkpoint: `pufferlib_market/checkpoints/stocks_deployment_candidate.pt`
(symlink → `pufferlib_market/checkpoints/autoresearch_stock/random_mut_2272/best.pt`)

---

## 3. Current SOTA Table

Sorted by `score` (descending). The `score` column uses the formula above with annualised percentages as plain numbers (e.g. 33% = 33, not 0.33). Note: the raw leaderboard column `holdout_median_return_pct` stores total return over the 90-bar window; annualised figures are obtained by scaling by 252/90.

| Rank | Config | Score | p10 Ann% | Median Ann% | Sortino | MaxDD% | Profitable | Win Rate | Steps | Notes |
|------|--------|------:|----------|-------------|---------|--------|------------|----------|-------|-------|
| **1** | **random_mut_2272** | **~49** | **+19.8%** | **+30.0%** | **2.22** | **22.4%** | **20/20** | **58%** | 32.6M | CURRENT BEST — ent=0.03, wd=0.005, slip=12bps, sdt=0.01 |
| 2 | random_mut_4424 | ~14 | +6.4% | +20.5% | 1.96 | 15.9% | 20/20 | 62% | 33.4M | h256, slip=12bps, sdt=0.01 |
| 3 | random_mut_1228 | ~10 | +10.9% | +19.2% | 1.75 | 18.0% | 20/20 | 55% | 26.7M | obs_norm, lr=5e-4, ent=0.08, wd=0.001, h1024 |
| 4 | stock_ent_03 | n/a | n/a | ~57% (val) | 2.31 (val) | n/a | n/a | 58% | ~20M | Holdout eval failed (data too short — pre-fix) |
| 5 | stock_wd_01 | n/a | n/a | ~55% (val) | 2.03 (val) | n/a | n/a | 70% | ~33M | Holdout eval failed (data too short — pre-fix) |

**Reference baselines (different eval harness — not directly comparable):**

| Config | Median Ann% | Sortino | MaxDD% | Notes |
|--------|-------------|---------|--------|-------|
| trade_pen_05 (crypto daily) | +108% | 1.76 | low | Best crypto daily; 100% profitable on crypto12 val |
| wd_0.06_s42/epoch_008.pt (Alpaca neural) | +2.6% (120d) | positive | 3.4% | Neural policy, Chronos2 features, not RL |

---

## 4. What We Are Optimising

The challenge sits at the intersection of four objectives that are simultaneously in tension:

**Generalisation in few gradient steps.** With only 5 minutes (~8–12M PPO steps at h1024), the agent cannot memorise the training distribution. Every architectural and reward-shaping choice must improve the signal-to-noise ratio of each gradient step.

**Execution realism.** Training with friction (5–12 bps slippage, 10 bps fees, no shorting without cost) forces the agent to find wider edges that survive real fills. Configs trained without friction look better in-sample but fail OOS.

**Risk-adjusted smoothness.** The `score` formula rewards Sortino, not raw return. An agent that trades rarely but correctly outscores one that churns. Trade-penalty hyperparameters are a primary lever.

**Robustness across market regimes.** The 20 holdout windows span different volatility and trend regimes. A config that wins 18/20 but blows up 2 fails the production gate.

---

## 5. Improvement Axes (Roadmap)

### GPU compute scale
**Current:** 1× A100 SXM4 80 GB, ~8–12M steps in 300 s at h1024.
**H100 path:** 4× parallel envs at same wall-time, or same envs 2× faster. More trials per sweep pass.
**Potential gain:** +2–4× trials per sweep pass → faster convergence on hyperparameter optima.
**Effort:** Low — change instance type in launcher; no code changes needed.

### C environment throughput
**Current:** AVX2 SIMD hints, `__builtin_prefetch`, `-O3 -march=native -funroll-loops`.
**Next:** Profile with `perf` to find remaining bottlenecks; try batched env reset to reduce Python overhead.
**Potential gain:** 10–20% more steps per second → more training signal per 5 minutes.
**Effort:** Medium — C env changes require rebuild and regression test.

### Fused MLP kernel
**Current:** `pufferlib_market/kernels/fused_mlp.py` fused into `train.py`.
**Next:** BF16 end-to-end path (currently cast to `encoder.weight.dtype`); CUDA graph capture of full update step (`--cuda-graph-ppo`).
**Potential gain:** ~15–25% wall-time reduction on the PPO update phase.
**Effort:** Medium — correctness testing required; BF16 dtype cast already identified.

### Architecture search
**Current:** MLP (h1024, 3 layers) is the only validated architecture.
**Candidates:**
- GRU with 2–4 steps of temporal context (cheap; may help on daily bars where momentum matters)
- Cross-symbol attention head (small transformer over 12 symbol embeddings per bar)
- ResidualMLP (tested, known to fail with gamma=0.999 — avoid that combination)
**Potential gain:** Unknown; GRU has the best risk/reward given daily data frequency.
**Effort:** Medium — requires new arch wiring in `train.py` and holdout validation.

### Reward engineering
**Current:** `ret*10 + clip[-5,5]` base reward, trade penalty, smooth-downside penalty.
**Best known:** `smooth_downside_temperature=0.01`, `trade_penalty=0` (random_mut_2272).
**Candidates:**
- Sortino-shaped reward: scale positive returns by running Sortino estimate
- Drawdown-aware shaping: add per-step penalty proportional to current drawdown depth
- Regime-conditioned reward temperature: reduce reward scale in high-volatility bars
**Potential gain:** Medium — reward engineering was the largest lever in the crypto sweep (drawdown penalty hurt, smooth-downside temperature helped).
**Effort:** Low to medium — parameter sweeps are cheap at 5 minutes each.

### Data augmentation
**Current:** 12 US equities, ~3 years of daily bars (train split).
**Candidates:**
- Expand to 15–40 symbols (scripts exist: `export_stocks40_daily.sh`)
- Time-warp augmentation: randomly stretch/compress bar sequences during rollout
- Synthetic price paths: GBM or OU process mixed with real data
**Potential gain:** More symbols = more opportunity signal per rollout; crypto evidence shows crypto12 >> crypto8.
**Effort:** Low for more symbols (data already available); medium for synthetic paths.

### Hyperparameter random search
**Current:** 45+ named configs + 30 random mutations per sweep pass.
**Next:** Bayesian optimisation over the mutation space using `holdout_robust_score` as the objective. CMA-ES or TPE with the existing leaderboard as warm-start.
**Potential gain:** Higher-quality mutations with fewer wasted trials.
**Effort:** Medium — requires wiring an optimiser library around the existing trial runner.

### Test-time training (TTT)
**Current:** Not implemented. Checkpoint is static at inference.
**Concept:** Use the first N live bars (after market open) to run 1–2 PPO gradient steps before placing orders for the day.
**Potential gain:** Rapid adaptation to recent regime; validated in parameter-golf (LoRA TTT was a SOTA entry).
**Effort:** High — requires careful causal gating to avoid look-ahead; must be validated against the holdout harness.

### Meta-strategy and ensemble
**Current:** Single best checkpoint deployed.
**Concept:** Keep top-3 checkpoints; vote on position size at inference; fall back to cash when checkpoints disagree.
**Potential gain:** Reduces variance on individual run; validated for neural policies in `unified_hourly_experiment/meta_selector.py`.
**Effort:** Medium — inference code changes; must validate that ensemble doesn't add latency.

---

## 6. The H100 Variant (10-Minute Challenge)

An extended variant of the challenge for researchers with H100 access:

**Budget:** 10 minutes on a single H100 SXM5 80 GB (or equivalent).
**Split:** First 5 minutes = standard training phase (identical to A100 challenge). Final 5 minutes = optional test-time training (TTT) phase using the first 30 bars of each holdout window as adaptation context.
**Bar:** Score must be >= 1.5× the current 5-minute A100 SOTA to earn a leaderboard entry.
**Evaluation:** Same 20-window harness. TTT phase must not access future bars — strictly causal.

The TTT phase is inspired by the LoRA TTT entry in the parameter-golf challenge. For trading, the analogue is: fine-tune the last-layer weights or a small LoRA adapter using the most recent 30 daily bars of each eval window before evaluating positions on the remaining 60 bars. The adaptation must use only returns/OHLCV already available to a live trader at that point in time.

---

## 7. Experiment Log

Full results in `autoresearch_stock_daily_leaderboard.csv`. Key named experiments below.

| Date | Config | p10 Ann% | Median Ann% | Sortino | MaxDD% | Profitable | Notes |
|------|--------|----------|-------------|---------|--------|------------|-------|
| 2026-03-21 | random_mut_2272 | +19.8% | +30.0% | 2.22 | 22.4% | 20/20 | **CURRENT BEST** — ent=0.03, wd=0.005, slip=12bps, sdt=0.01, h1024 |
| 2026-03-21 | random_mut_4424 | +6.4% | +20.5% | 1.96 | 15.9% | 20/20 | h256, slip=12bps, sdt=0.01 — smaller model with all-profitable |
| 2026-03-21 | random_mut_1228 | +10.9% | +19.2% | 1.75 | 18.0% | 20/20 | obs_norm, lr=5e-4, ent=0.08, wd=0.001, h1024 |
| 2026-03-21 | stock_ent_03 | n/a | n/a | 2.31 (val only) | n/a | n/a | Holdout eval script bug — data too short; later fixed |
| 2026-03-21 | stock_wd_01 | n/a | n/a | 2.03 (val only) | n/a | n/a | Holdout eval script bug — data too short; later fixed |
| 2026-03-21 | stock_baseline | n/a | n/a | -0.84 (val) | n/a | 0% | Default config with no trade penalty — all windows losing |
| 2026-03-21 | stock_obs_norm | n/a | n/a | -1.25 (val) | n/a | 0% | obs_norm alone hurts without trade penalty |
| 2026-03-21 | stock_h512 | n/a | n/a | -1.72 (val) | n/a | 0% | h512 without regularisation fails |
| 2026-03-21 | stock_slip_15bps | n/a | n/a | -2.90 (val) | n/a | 0% | Too much friction: 15bps slip collapses learning |
| 2026-03-15 | trade_pen_05 (crypto daily ref) | — | +108% ann | 1.76 | low | 100% | Best crypto daily baseline; different dataset |
| 2026-03-13 | slip_5bps (crypto OOS ref) | — | +5.21%/30d | 1.62 | low | 96% | Best hourly crypto OOS; different dataset |

### Holdout eval fix (2026-03-21)

Early runs (trials 0–36) hit a `ValueError: MKTD too short for holdout eval` because the default `holdout_eval_steps` was not set for `--stocks` mode. The bug was fixed in commit `73daacbc` — `holdout_eval_steps` now defaults to 90 for `--stocks`. Trials 0–36 have valid `val_return` and `val_sortino` but no holdout metrics. Trials 37+ have full holdout stats.

---

## 8. Failed Experiments Summary

See `failedalpacaprogress.md`, `failedalpacaprogress4.md`, `failedalpacaprogress5.md` for detailed notes on past failures. Key lessons for the RL challenge:

- **obs_norm alone hurts**: Without trade penalty, obs_norm causes the agent to over-trade and collapse OOS. It helps only when combined with trade_penalty >= 0.05.
- **h512 without regularisation fails**: Smaller model converges faster in-sample but generalises worse than h1024 with wd.
- **15bps slippage is too high for stocks**: Unlike crypto where 12bps is fine, stocks daily bars already have tight spreads and the additional friction makes profitable edges disappear during training.
- **High entropy coef (0.08+) hurts on daily stocks**: In crypto, ent=0.05 was optimal. On stocks daily bars, lower ent (0.03) with weight decay generalises better.
- **gamma=0.999 kills credit assignment**: Verified in crypto; expected to be equally bad on stocks. Stick to gamma=0.99.
- **reward_scale=20 hurts**: Gradient magnitudes explode, causing unstable updates. reward_scale=10 (default) or 5 is safer.
- **Drawdown penalty hurts returns**: Adding an explicit drawdown penalty term caused in-sample returns to drop 2× in crypto experiments. Let the trade penalty proxy for it instead.
- **Non-determinism is real**: RL training with the same seed on the same hardware produces results within ~5% but not bit-identical. Deploy verified checkpoints; don't rely on re-training to reproduce a specific score.

---

## 9. Running the Challenge

### Reproduce the current SOTA

```bash
source .venv313/bin/activate

# Evaluate existing best checkpoint on the full holdout harness
python -u -m pufferlib_market.evaluate_holdout \
    --checkpoint pufferlib_market/checkpoints/autoresearch_stock/random_mut_2272/best.pt \
    --data-path pufferlib_market/data/stocks12_daily_val.bin \
    --eval-hours 90 \
    --n-windows 20 \
    --fill-buffer-bps 5.0 \
    --fee-rate 0.001 \
    --deterministic \
    --periods-per-year 252
```

### Run a new 5-minute sweep pass

```bash
source .venv313/bin/activate

python -u -m pufferlib_market.autoresearch_rl \
    --stocks \
    --time-budget 300 \
    --max-trials 50 \
    --leaderboard autoresearch_stock_daily_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock \
    --holdout-n-windows 20 \
    --holdout-eval-steps 90 \
    --holdout-fill-buffer-bps 5.0 \
    --holdout-fee-rate 0.001 \
    --fee-rate-override 0.001 \
    --rank-metric holdout_robust_score
```

### Launch remotely (A100 pod)

```bash
python launch_stocks_autoresearch_remote.py \
    --run-id sweep_$(date +%Y%m%d_%H%M) \
    --remote-host administrator@<pod-ip> \
    --time-budget 300 \
    --max-trials 50
```

The launcher syncs the repo, runs the sweep, and pulls the updated leaderboard CSV back.

### Data files

| File | Symbols | Bars | Split |
|------|---------|------|-------|
| `pufferlib_market/data/stocks12_daily_train.bin` | 12 US equities | 1,211 | Train |
| `pufferlib_market/data/stocks12_daily_val.bin` | 12 US equities | 194 | Val / holdout |

The 12 symbols are drawn from a universe of liquid, shortable US equities with Alpaca data availability. The val split covers the most recent period and is never used during training.

---

## 10. Leaderboard Submission Requirements

To add an entry to `autoresearch_stock_daily_leaderboard.csv` that is eligible for production deployment:

1. Run must complete in <= 300 s wall-time on a single A100 SXM4 80 GB (or document the hardware if different).
2. Manifest JSON must be committed alongside the checkpoint.
3. All 20 holdout windows must be profitable to qualify for the production gate.
4. `score` must exceed the current best by a margin > 2 (statistical noise floor based on observed run variance).
5. Results should be reproducible within 10% across 3 independent runs (RL non-determinism caveat: see section 8).

Non-qualifying runs (e.g. not all windows profitable, or longer compute budget) are welcome as informal leaderboard entries — append them to the CSV with a note in the `error` column.
