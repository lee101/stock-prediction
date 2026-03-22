# H100 Experiment Plan (Updated 2026-03-22)

## Executive Summary — v9 (lr=1e-4 CRITICAL FOR stocks11 DATA)

**Key findings since v8 (2026-03-22 afternoon experiments):**

### CRITICAL: lr=1e-4 (not 3e-4) is required for stocks11_2015 convergence
- All tp03_s777 configs (lr=3e-4, default) **collapse to hold-cash on stocks11 data**
  - Trained 45M steps on stocks11_2015, gave ZERO trades in evaluation
  - encoder.0.weight shape (1024, 192) confirms correct obs_size — it's purely a LR issue
- lr=1e-4 (no-anneal) converges on stocks11_2015 → the best result so far
- **The 3x robustness advantage of stocks11 is ONLY achieved with lr=1e-4 configs**
- Evidence from 450s sweep (16 trials):
  - stocks11 best: random_mut_9621 (lr=1e-4, no-anneal) → robust=-41.2, med=+4.7%
  - stocks12 best: random_mut_6320 (lr=3e-4, anneal) → robust=-46.6, med=+4.3%
- The SAME config (random_mut_9621, lr=1e-4) on stocks12 gives robust=-128.7 → confirms lr mismatch

### Why lr=1e-4 for stocks11, lr=3e-4 for stocks12?
- stocks11_2015 (2015-2025): 3895 steps, includes volatile regimes (2015 China crash,
  2018 correction, COVID crash). Higher gradient variance → lower LR needed to stabilize.
- stocks12_2019 (2019-2025): 1797 steps, mostly coherent bull market. Higher LR works.

### Fix applied: Added lr=1e-4 named configs to STOCK_EXPERIMENTS
New configs added (before random mutations, indices ~82-91):
- `lr1e4_s777`, `lr1e4_s42`, `lr1e4_s9621`, `lr1e4_s1137`
- `lr1e4_wd01_s777`, `lr1e4_wd005_s777`, `lr1e4_slip5_s777`, `lr1e4_anneal_s777`
- `lr1e4_h2048_s777` (256 envs, 4096 minibatch), `lr1e4_h2048_s42`

### H100 commands (v9 — UPDATED WITH lr=1e-4 FIX)
```bash
# PREFERRED: stocks11_2015 — now with lr=1e-4 configs in early pool
# These named configs (indices 82-91) ensure convergence in first 92 trials
# Random mutations from trial 92 also include 25% lr=1e-4 (in mutation space)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500
# Note: launch_stocks_autoresearch_remote.py now defaults to stocks11_2015 data

# ALTERNATIVE: stocks12 (lr=3e-4 works here, proven 4-5% hit rate)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500 \
  --train-data pufferlib_market/data/stocks12_daily_train_2019.bin \
  --val-data pufferlib_market/data/stocks12_daily_val.bin
```

### Architecture scaling: h2048 vs h1024 COMPLETED (2026-03-22)

**Arch comparison results (8 configs × 450s, 65-68M steps each):**

| Config | stocks11_2015 robust | stocks11_2012 robust | Winner |
|--------|---------------------|---------------------|--------|
| lr1e4_anneal_s777 | **-54.9** | **-40.3** | ← BEST on BOTH |
| lr1e4_s42 | -57.0 | -82.9 | h1024 wins |
| lr1e4_s9621 | -61.0 | -51.8 | h1024 wins |
| lr1e4_h2048_s777 | -79.6 | -122.7 | h1024 wins |
| lr1e4_h2048_s42 | -116.5 | -53.4 | **h1024 wins** |

**Conclusion: h2048 does NOT outperform h1024 at lr=1e-4. Use h1024.**

**stocks11_2012 (4840 days) > stocks11_2015 (3895 days):**
- lr1e4_anneal_s777: -40.3 (2012) vs -54.9 (2015) — **2012 wins**
- Added 5 lr=1e-4 H100-specific configs to STOCK_EXPERIMENTS (top of H100 section)
- Updated launch_stocks_autoresearch_remote.py default to stocks11_2012 data

---

## Executive Summary — v8 (LOCAL VALIDATION + STOCKS11 WINNER)

**Key findings since v7:**

### 1. Local RTX 5090 validation confirmed (2026-03-22)
- 450s budget gives 37-45M steps at 83-99k sps → above 33M convergence threshold
- Stocks11 extended shows **3x better holdout robustness** vs stocks12:
  - stocks11 trial 1: robust=-41.2, worst_ret=-8.2%, p25_ret=-0.05%
  - stocks12 trial 1: robust=-128.7, worst_ret=-20.6%, p25_ret=-16.5%
- Even the first random trial on stocks11 shows near-breakeven p25 performance
- **Recommendation: strongly prefer stocks11 data for H100 runs**

### 2. Mutation space improvements
- Added `trade_penalty=0.03` (KNOWN WINNER from tp03_s777 experiments)
- Added `anneal_ent` option for entropy scheduling
- Added `smooth_downside_penalty` [0.0, 0.1, 0.2, 0.5]

### 3. H100 commands (v8 — superseded by v9)
```bash
# FIRST: stocks11 extended (2x data, 3x better robustness from first principles)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500 \
  --train-data pufferlib_market/data/stocks11_daily_train_2015.bin \
  --val-data pufferlib_market/data/stocks11_daily_val_2015.bin

# SECOND: stocks12 (proven 4-5% hit rate, use as baseline)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500
```

### Proven winning models on 201-day hard val (stocks12)
| Model | median/90d | p10/90d | Steps | Config |
|-------|-----------|---------|-------|--------|
| random_mut_9497 | **+9.9%** | -2.3% | 33.8M | h=1024, wd=0.005, slip=12bps |
| random_mut_112 | **+6.8%** | -4.7% | 37.5M | h=256, standard |
| random_mut_7392 | +6.3% | -6.7% | 31.7M | h=1024, wd=0.005 |

### Expected outcomes from 500 H100 trials
- 82 named configs (fundamental diversity coverage, including h2048_h100, transformer_h100)
- 418 random mutations (improved space: tp=0.03, anneal_ent, smooth_downside_penalty)
- At 4-5% hit rate on stocks12: **~17-21 positive models**
- On stocks11 (3x better robustness): possibly **30-40+ positive models**
- Total time: 500 × 90s training + ~90s eval = **~25 hours**

---

## Executive Summary — v7 (POOL RESTRUCTURE + DATA EXTENSION)

**Key improvements since v6:**

### 1. Pool restructuring (2026-03-22)
- Moved 75 tp03/wd01 dense seed configs (ALL NEGATIVE at 300s on extended val) out of default pool
- Expanded random mutations: 30 → 450 slots
- Random mutations now at pool index 82 (was 157) → reachable in first 200 H100 trials
- Pool size: 532 entries (82 named + 450 random) → supports 500-trial H100 run

### 2. Extended stocks11 data: 2x training samples
- stocks11_daily_train_2015.bin: 11 symbols × 3895 days = **42,845 samples** (from 2015-01-02)
- vs stocks12_daily_train_2019.bin: 12 × 1797 = 21,564 samples (from 2020-09-30)
- Extra 2x data includes: **COVID crash (Mar 2020), Q4 2018 -20% correction, 2015 China crash**

### 3. H100 command (v7)
```bash
# Primary: stocks12 (proven, simpler, known 4-5% hit rate)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500

# Alternative: stocks11 extended (2x data, 3x better robustness — USE THIS)
python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 500 \
  --train-data pufferlib_market/data/stocks11_daily_train_2015.bin \
  --val-data pufferlib_market/data/stocks11_daily_val_2015.bin
```

### Proven winning models on 201-day hard val
| Model | median/90d | p10/90d | Steps | Config |
|-------|-----------|---------|-------|--------|
| random_mut_9497 | **+9.9%** | -2.3% | 33.8M | h=1024, wd=0.005, slip=12bps |
| random_mut_112 | **+6.8%** | -4.7% | 37.5M | h=256, standard |
| random_mut_7392 | +6.3% | -6.7% | 31.7M | h=1024, wd=0.005 |

### Expected outcomes from 500 H100 trials
- 82 named configs (fundamental diversity coverage)
- 418 random mutations
- At 4-5% hit rate: **~17-21 positive models**
- Total time: 500 × 90s training + ~90s eval = **~25 hours**



## Executive Summary — REVISED (v6: training duration breakthrough)

**ROOT CAUSE OF RECENT FAILURES FOUND (2026-03-22):**
All recent local sweeps used `--max-timesteps-per-sample 200` which caps training at
**4.3M steps**. But winning models need **33–37M steps** (~300s on A40). With 8x fewer
steps, models don't converge → 0 positive results in 34+ trials with tp03/wd01 variants.

| Setup | Steps | Time (A40) | Hard val results |
|-------|-------|------------|-----------------|
| cap=200 (recent sweeps) | ~4.3M | ~35s/trial | **0/34 positive** |
| no cap, 300s budget (original) | ~33-37M | 300s/trial | **~4-5% positive** |

**Proven winning models on extended 201-day hard val (binary fill, Sep 2025–Mar 2026):**
- `random_mut_9497` (h=1024, wd=0.005, slip=12bps, ent=0.05): **median=+9.9%, p10=-2.3%**
- `random_mut_112` (h=256, standard config): **median=+6.8%, p10=-4.7%**
- `random_mut_7392` (h=1024, wd=0.005, no slip): **median=+6.3%, p10=-6.7%**

**CORRECT H100 command (v6):**
```bash
python launch_stocks_autoresearch_remote.py \
  --gpu-type h100 --max-trials 500 \
  --time-budget 90 \
  --max-timesteps-per-sample 10000
```
- `--time-budget 90`: H100 at ~390k steps/sec × 90s = **~35M steps** ≈ A40 300s sweet spot
- `--max-timesteps-per-sample 10000`: effectively no cap (215M max >> 35M in 90s)
- `--stocks12` added automatically by the launcher (uses STOCK_EXPERIMENTS pool)
- Expected: 500 trials × 90s each = **~12.5 hours**; ~20-25 positive models (4-5% hit rate)

---

## Executive Summary — REVISED (v5: extended training data breakthrough)

**KEY FINDING (v5, 2026-03-22 local calibration):**
Extending training data from 2022–2025 (1302 days) to 2020–2025 (1797 days) dramatically improves
generalization on the hard extended val (Sep 2025 – Mar 2026). The extended val includes the
Nov 2025 – Feb 2026 bear market period where all models with old training data fail.

| Config | Old training (1302d) | New training (1797d) | Val tested |
|--------|---------------------|---------------------|------------|
| stock_trade_pen_03 | -102 (20% neg, worst=-14.6%) | **+3.1 (p25=+4%, worst=-0.15%)** | 201-day (hard) |
| stock_baseline | -87 (55% neg) | -89 (similar) | 201-day |

**With old training: 0/7 positive configs on hard val. With new training: 1/7 positive (trade_pen_03).**

**H100 strategy v5: Use stocks12_daily_train_2019.bin (1797 days, Oct 2020 – Aug 2025)**
- New step cap: 12 × 1797 × 200 = 4,312,800 (H100 hits in ~12s at 350k sps)
- Val: stocks12_daily_val.bin (201 days, Sep 2025 – Mar 2026, updated to include bear market)
- Expected hit rate: 5-15% positive configs (vs 0% with old 1302-day training)
- 1200 trials × ~40s each = ~13.3 hours on H100

---

## Executive Summary — REVISED (v4: diversity over targeting)

**The original plan targeting stocks20 was based on contaminated data.**

Original 73-config RTX 5090 sweep showed stocks20 winning (2 configs with positive holdout).
After fixing stock split adjustments (NFLX 10:1 Nov 2025, NVDA 10:1, GOOG 20:1, TSLA 3:1,
AMZN 20:1, AVGO 10:1), the "stocks20 positive holdout" result was entirely an artifact of
policies shorting the fake NFLX -93% price drop.

**Current state with clean data:**
- **Reliable SOTA baseline**: `random_mut_2272` (300s, 300M steps) — 0% neg, +10.5% med, +5.2% p10, Sortino=1.55
- **Lucky v2 SOTA (not reproducible)**: `stock_drawdown_pen` — 1/50 trials hit score=+24.9; 13-seed verification sweep ALL failed (scores -49 to -170)
- **Conclusion**: RL training has high variance. No single config reliably beats random_mut_2272 at 90s.

**Critical finding (2026-03-22 standalone drawpen verification):**
`stock_drawdown_pen` (+24.9, 0% neg) in the v2 50-trial sweep was a **lucky training run**.
Verified by running all 13 drawpen seed/param variants: every single one failed (scores -49 to -170).
RL is non-deterministic even with same seed — deploy verified checkpoints, never retrain expecting same.

**H100 strategy: DIVERSITY over depth — run 1000+ trials with 3.1M step cap**
- H100 at ~350k sps hits 3.1M step cap in ~9s → each trial runs ~35s total (train + eval)
- In 12 hours: ~1200+ diverse configs explored (vs 50 local, 500 original plan)
- With ~2% hit rate for positive scores → expect 24+ positive-score configs from 1200 trials
- Goal: find a reliable config that consistently beats random_mut_2272 across 20 holdout windows

---

## Data Quality Fixes Applied (2026-03-22)

All binaries have been re-exported with split-adjusted CSVs.

### Stock splits fixed (backups at `.pre_split_backup`)

| Symbol | Split  | Date       | Pre-split rows (daily) |
|--------|--------|------------|------------------------|
| NFLX   | 10:1   | 2025-11-15 | (previous session)     |
| NVDA   | 10:1   | 2024-06-10 | 587                    |
| GOOG   | 20:1   | 2022-07-18 | 110                    |
| TSLA   |  3:1   | 2022-08-25 | 138                    |
| AMZN   | 20:1   | 2022-06-06 | 82 daily + 3306 hourly |
| AVGO   | 10:1   | 2024-07-15 | 667 daily + 6969 hourly|

Remaining "big drops" are verified real market events:
- NFLX: -35% on 2022-04-20 (Q1 2022 subscriber loss earnings)
- META: -26% on 2022-02-03 (Q4 2021 earnings miss, stocks15 only)
- INTC: -26% on 2024-08-01 (Q2 2024 earnings disaster, stocks15 only)

### Re-exported binaries (all clean, verified)

| File                             | Symbols | Train days | Val days | Notes |
|----------------------------------|---------|-----------|---------|-------|
| stocks12_daily_train.bin         | 12      | 1302      | —       | 2022-02-07 to 2025-08-31 |
| stocks12_daily_train_2019.bin    | 12      | 1797      | —       | **2020-09-30 to 2025-08-31 (H100 target)** |
| stocks12_daily_val.bin           | 12      | —         | 201     | **2025-09-01 to 2026-03-20 (extended, hard)** |
| stocks15_daily_{train,val}.bin   | 15      | 1302      | 113     | |
| stocks20_daily_train.bin         | 20      | 1302      | —       | Can't extend (AMD/NET/NFLX from 2022) |
| stocks20_daily_val.bin           | 20      | —         | 201     | 2025-09-01 to 2026-03-20 |
| stocks20_cross_daily_{train,val} | 20      | 1302      | 158     | |

---

## Clean Data Sweep Results (2026-03-22, RTX 5090, 90s/trial)

### stocks12 (12 symbols)

| Config              | holdout | neg_rate | median% | p10%   |
|---------------------|---------|----------|---------|--------|
| h100_slip_10bps     | -22.84  | **10%**  | +5.64   | +0.92  |
| h100_ent_05         | -35.80  | **10%**  | +9.62   | +0.71  |
| h100_mut2272_slip5  | -44.56  | 20%      | +4.35   | -4.15  |
| h100_mut2272_wd01   | -78.54  | 50%      | +0.69   | -11.78 |
| h100_mut2272_style  | -102.73 | 75%      | -4.48   | -14.17 |

### stocks20 (20 symbols) — all FAILED with clean data

| Config              | holdout  | neg_rate | median% |
|---------------------|----------|----------|---------|
| h100_ent_05         | -77.11   | 50%      | +0.99   |
| h100_mut2272_style  | -87.69   | 55%      | -2.18   |
| h100_slip_5bps      | -134.29  | 85%      | -14.47  |
| h100_mut2272_slip5  | -138.80  | 65%      | -8.61   |
| h100_slip_10bps     | -139.15  | 75%      | -9.25   |

### Current SOTA baseline (random_mut_2272, 300s training, stocks12)

| Metric                | Old val (158d, easy) | New val (201d, hard) |
|-----------------------|----------------------|----------------------|
| holdout neg_rate      | **0%** (all 20 windows) | ~55% (fails on bear market) |
| holdout median        | +10.5%               | ~-1.4% to -7.6% (non-deterministic) |
| holdout p10           | +5.2%                | ~-15% |
| holdout median Sortino| 1.55                 | negative |

**rmu2272 fails on the hard 201-day val** — the Nov 2025–Feb 2026 bear market breaks it.
The H100 target model must beat rmu2272 on the **hard val** (201 days), not the easy one.

---

## V2 50-Trial Sweep Results (2026-03-22, stocks12_daily_train.bin, 90s/trial)

Full 50-trial sweep on stocks12. Top results:

| Config                  | score  | neg_rate | median%  | p10%    | worst%  | Sortino(med) | max_dd_worst% |
|-------------------------|--------|----------|----------|---------|---------|-------------|---------------|
| **stock_drawdown_pen**  | **+24.9** | **0%** | **+22.9** | **+4.8** | **+3.3** | **7.25** | **9.2%** |
| **stock_trade_pen_05_s123** | **+14.9** | **0%** | **+16.6** | **+7.7** | **+4.0** | **4.76** | **11.9%** |
| stock_trade_pen_05      | -3.5   | 5%      | +27.8    | +4.2    | -1.8    | 4.06        | 14.0%         |
| stock_trade_pen_03      | -21.3  | 20%     | +5.0     | -2.9    | -3.5    | 1.63        | 9.7%          |
| stock_ent_08            | -24.4  | 15%     | +5.8     | -3.1    | -7.3    | 1.72        | 10.7%         |

### stock_drawdown_pen config details
```
drawdown_penalty=0.05, trade_penalty=0.03
fill_slippage_bps=0.0 (NO training slippage — drawdown penalty is the regularizer)
hidden_size=1024, lr=3e-4, ent_coef=0.05, weight_decay=0.0, anneal_lr=True
train_steps=3,178,496 in 90s (~35k steps/sec — fewer steps than trade_pen due to penalty overhead)
```

### Key v2 sweep insight
Drawdown penalty approach with **zero slippage training** outperforms slip-training approach:
- `stock_drawdown_pen` (0% slip): score=+24.9 > `random_mut_2272` (12bps slip): estimated ~-5
- The `drawdown_penalty` term forces the policy to avoid large equity dips, which naturally prevents
  the reckless behavior that causes big holdout losses. More interpretable and stable than slip training.

### Why stock_drawdown_pen score is negative-proof
- `worst_ret=+3.3%` means even the worst of 20 windows was profitable
- `max_dd_worst=9.2%` is tight — worst-case drawdown capped below 10%
- `Sortino=7.25` median — extremely high risk-adjusted return

### Checkpoint
`pufferlib_market/checkpoints/stocks12_v2_sweep/stock_drawdown_pen/best.pt`

---

## Overfitting at 300s Budget (2026-03-22 Finding)

Local validation confirmed critical overfitting when training budget exceeds ~90s on stocks12:

| Config          | Budget | holdout neg_rate | holdout median | Notes         |
|-----------------|--------|------------------|----------------|---------------|
| h100_slip_10bps | 90s    | **10%**          | +5.64%         | Good          |
| h100_slip_10bps | 300s   | **50%**          | +0.20%         | Overfit       |

**Conclusion**: stocks12 optimal budget is ~90s (~4M steps on RTX 5090).
H100 at ~3.5x speedup reaches ~4M steps in 90s with better GPU utilization.
Use `time_budget=90` with `max_trials=500` for H100.

---

## Dataset Decision — UPDATED (2026-03-22 v5: extended training wins on hard val)

**CONCLUSION: Use stocks12_daily_train_2019.bin (1797 days, 2020-09-30 to 2025-08-31) for H100.**

### Why the previous "extended data is worse" finding was wrong

The previous finding (v4) used the **OLD 158-day val** (Sep–Feb 2026), which was mostly a bull
market (Sep–Nov 2025). The **new 201-day val** (Sep 2025 – Mar 2026) includes the Nov 2025 –
Feb 2026 bear market where all 2022-trained models fail.

| Experiment            | Training data | Val data | Best score | Notes |
|-----------------------|---------------|----------|-----------|-------|
| Previous v4 finding   | 1302 days (2022–2025) | 158 days (Sep–Feb 2026, easy) | -25.8 | Old easy val |
| Extended data (prev)  | 1797 days (2020–2025) | 158 days (same easy val)      | -47.6 | Worse on easy val |
| **New 50-trial (v5)** | 1302 days (2022–2025) | **201 days (Sep–Mar 2026, hard)** | **-44.97** (best: 0% loss windows) | 0/50 positive |
| **New train2019 (v5)**| **1797 days (2020–2025)** | **201 days (hard)** | **+3.10 (stock_trade_pen_03!)** | 1/7 positive |

**Root cause**: The 2020–2021 data (COVID recovery + 2021 bull market) teaches the model about
market cycles — specifically, recognizing when NOT to trade (bear markets). The 2022-only training
data lacks this: models only see one bear market (2022 tech crash) and one recovery. Adding the
2020–2021 data improves regime detection for the 2025–2026 bear market.

**Key data limits** (what restricts the start date):
- stocks12: PLTR IPO 2020-09-30 → effective start 2020-09-30 (1797 calendar days)
- stocks20: AMD/NET/NFLX only from 2022-02-07 → cannot extend stocks20 further back
- stocks11 (no PLTR): can extend to 2019-01-02 but loses PLTR; not recommended

### Extended training binary
- **File**: `pufferlib_market/data/stocks12_daily_train_2019.bin`
- **Range**: 2020-09-30 to 2025-08-31 (1797 calendar days, ~1240 trading days)
- **Samples**: 12 × 1797 = 21,564 per epoch
- **New step cap**: 21,564 × 200 = **4,312,800 steps** (H100 hits at ~12s, A40 hits at ~64s)

---

## H100 Configuration — FINAL

**Dataset**: stocks12 (12 symbols, 1302 train days, 158 val days)
**Binary files**: `pufferlib_market/data/stocks12_daily_{train,val}.bin`
**Symbols**: AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN

### Recommended H100 command

```bash
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

python -m pufferlib_market.autoresearch_rl \
    --stocks12 \
    --train-data pufferlib_market/data/stocks12_daily_train_2019.bin \
    --val-data pufferlib_market/data/stocks12_daily_val.bin \
    --time-budget 90 \
    --max-trials 1200 \
    --max-timesteps-per-sample 200 \
    --leaderboard autoresearch_stock_h100_leaderboard.csv \
    --checkpoint-root pufferlib_market/checkpoints/autoresearch_stock_h100
```

Notes:
- `--stocks12` uses combined pool (STOCK_EXPERIMENTS + H100_STOCK_EXPERIMENTS non-gpu), sets periods_per_year=252, fee_rate=0.001, holdout_eval_steps=90
- **`--max-timesteps-per-sample 200`**: caps each trial at 4.3M steps (12 syms × 1797 days × 200 = 4,312,800)
  - This is CRITICAL — without the cap, H100 settings overfits at 20M+ steps
  - **On H100 at ~350k sps**: 4.3M step cap hit in ~12s. `time_budget=90` is irrelevant. Total per trial: ~40-45s (12s train + 28-32s eval)
  - **Early rejection is irrelevant for H100**: training completes at cap before 25% time check fires.
  - 1200 trials × ~42s = ~14 hours on H100
  - On A40 (local): 4.3M step cap hit in ~64s, total ~90-120s/trial
- **DO NOT use `--h100-mode`**: that forces num_envs=256, minibatch_size=4096 → fewer PPO updates per step count → worse convergence
- **stocks12_daily_train_2019.bin**: 1797 days (Oct 2020–Aug 2025) — includes 2021 bull + 2022 bear; much better generalization on hard extended val (201 days through Mar 2026)
- **Val**: stocks12_daily_val.bin (201 days, 2025-09-01 to 2026-03-20) — includes the Nov 2025–Feb 2026 bear market regime
- **1200 trials** with 5-15% expected hit rate → 60-180 positive-score configs (vs 0/50 with old training)

### CRITICAL: Why NOT --h100-mode for the actual H100 run

The `--h100-mode` flag forces `num_envs=256, minibatch_size=4096` which causes overfitting:
- A40 with h100 settings: ~190k sps → 15.6M steps hit in 82s → OVERFIT (5x too many steps)
- Real H100 with h100 settings: ~500k sps → 15.6M steps hit in 31s → same overfitting
- stock_drawdown_pen discovered at A40 DEFAULT settings: ~35k sps → 3.2M steps in 90s ✓

With `--max-timesteps-per-sample 200`, training always caps at 3.1M steps regardless of GPU speed.
This replicates the non-overfitting regime where stock_drawdown_pen was discovered.

### eval_hours calibration (CRITICAL)

When using `evaluate_fast` (C env) on daily data, `eval_hours` counts **calendar days**, not trading days.
To evaluate ~90 trading days, use `eval_hours=130` (90 × 7/5 ≈ 126, rounded to 130).

The autoresearch `holdout_eval_steps=90` calls `evaluate_holdout.py` (Python sim) which counts 90 **trading bars**.
These two give different results:
- `evaluate_holdout --eval-hours 90` ≈ 90 trading days
- `evaluate_fast --eval-hours 90` ≈ 64 trading days → use `--eval-hours 130` for equivalent

Reference on random_mut_2272 (stocks12_daily_val.bin):
| Evaluator | eval_hours | median_return | p10_return |
|-----------|-----------|--------------|------------|
| evaluate_holdout | 90 | +4.9% | -11.6% |
| evaluate_fast | 90 | +0.3% | -12.2% |
| evaluate_fast | 130 | +8.6% | -18.9% |

---

## Key Findings

- **stocks20 results were invalid**: Original 73-config sweep showed stocks20 winning because
  NFLX had an unadjusted 10:1 forward split creating a fake -93% single-day drop in val data.
  Policies that shorted NFLX generated enormous artificial returns. After split-adjustment, all
  stocks20 configs fail in holdout at 90s budget.

- **stock_drawdown_pen was a LUCKY RUN** (2026-03-22 verification):
  Standalone sweep of all 13 drawpen seed/param variants (h100_drawpen_style through h100_drawpen_slip5)
  with early rejection disabled and correct step cap (200x):
  - ALL 13 variants: negative holdout scores (-49 to -170)
  - stock_drawdown_pen's +24.9 score in v2 sweep = 1 lucky run out of 50+ attempts (~2% hit rate)
  - RL training is non-deterministic; same config with same seed gives wildly different results
  - Don't target drawpen configs specifically; include them in the pool for coverage

- **random_mut_2272 is the reliable SOTA baseline**: 0% neg, +10.5% med, p10=+5.2%, Sortino=1.55.
  This was trained with 300s budget. At 90s, most configs fail; random_mut_2272 required 300s.
  For H100, the reliable baseline is the target to beat.

- **Early rejection is irrelevant for H100 runs**: With `--max-timesteps-per-sample 200` (3.1M
  step cap), H100 at ~350k sps hits the cap in ~9s. The 25%/50% time budget checks fire at 22.5s
  and 45s respectively. Since training completes at 9s (proc exits), the while loop exits before
  any check. `--early-reject-threshold` has no effect on H100 runs.

- **Early rejection has cross-family bias on A40**: If fast configs (rmu1228_slip5, val_ret=0.305)
  run before drawpen configs (val_ret≈0.1 at final), the threshold 0.305×0.8=0.244 causes
  early rejection of drawpen at 25% check. In v2 sweep this didn't happen because all scores
  were negative before trial 20. For A40 local sweeps mixing config families, use
  `--early-reject-threshold 0.3` or lower.

- **Optimal step cap is 3.1M (200x)**: stocks12 with 90s/trial, RL converges best at ~3.1M steps.
  More steps → overfitting (300s sweep: 50% negative windows). Less steps → undertraining.
  At 200x multiplier: cap = 12 × 1302 × 200 = 3,124,800 steps. This is the sweet spot.

- **2nd best v2 sweep: stock_trade_pen_05_s123** (0% neg, +16.6% med, +7.7% p10):
  `trade_penalty=0.05, seed=123` — also a lucky run (trial 1 in a sweep of 50). H100 pool
  has 8 trade_pen_05 variants. Include for coverage; expect ~2% positive hit rate.

- **stocks12 with slip_10bps was the best CONSISTENT config at 90s**: Before v2 sweep,
  h100_slip_10bps got 90% windows profitable (2/20 lose), median=+5.64%.
  This is more reproducible than drawpen/trade_pen lucky runs.

- **h2048 configs in H100 pool are stocks12-safe**: The h2048 variants require ~2.5x more memory
  but stocks12's smaller obs space makes them feasible even at H100 speeds.

---

## Expected Outcomes (REVISED v5)

**Val is now 201 days (hard: includes Nov 2025–Feb 2026 bear market). All old baselines FAIL.**
New H100 goal: find configs that are profitable even in the hard 201-day val period.

| Metric               | rmu2272 on hard val | local best (trade_pen_03, 1797d) | H100 Target (hard val) |
|----------------------|---------------------|----------------------------------|------------------------|
| holdout neg_rate     | ~55% (fails)        | ~0% (best worst=-0.15%)         | < 10%                  |
| holdout median       | ~-3% to -8%         | +15.9%                          | > +5%                  |
| holdout p10          | ~-15%               | +4.0% (p25)                     | > 0%                   |
| holdout worst window | ~-20%               | -0.15%                          | > -5%                  |
| robust score         | ~-90                | +3.1                            | > 0                    |

With 1200 trials and 5-15% hit rate (from local calibration): expect 60-180 positive-score configs.
The best-of-1200 should significantly exceed the local 1/7 result (trade_pen_03 at +3.1 robust score).

### tp03 variants sweep findings (2026-03-22, seed 1337, extended training)

16 trade_pen_03 variants tested with extended training data. Key results:

| Variant | Score | Worst | Median | Notes |
|---------|-------|-------|--------|-------|
| tp03_s2272 | **-33.8** | -0.6% | 0% | Seed 2272 is best explicit seed |
| tp03_wd01 | -39.4 | -14.7% | +5.6% | wd=0.01 helps; positive median! |
| tp03_h2048 | -50.0 | -9.0% | +6.0% | Larger net benefits from 5yr data |
| stock_trade_pen_03 | -71.5 | -15.5% | -0.6% | Baseline (seed 1337 is mediocre) |
| tp03_cosine | -72.1 | -8.2% | -1.0% | Cosine LR similar to default |
| tp03_ent03 | -76.7 | -10.9% | -3.2% | Lower entropy worse |
| tp03_h512 | -87.2 | -17.3% | -1.6% | Smaller net is worse |
| tp03_slip5 | -129.6 | -22.5% | -10.6% | **Slippage training HURTS** |
| tp03_slip10 | -110.4 | -26.6% | -0.6% | **Slippage training HURTS** |
| tp03_obs | -124.1 | -22.6% | -7.3% | obs_norm HURTS with trade_pen_03 |
| tp03_full_reg | -138.5 | -30.0% | -7.0% | Everything combined = worst |

**Rule: use trade_pen_03 WITHOUT slippage, WITHOUT obs_norm, WITH wd=0.01 or h2048**

Pool now includes best combinations: `tp03_s2272_wd01`, `tp03_h2048_wd01`, `tp03_s2272_h2048`

---

## H100 Experiment Pool Summary

`H100_STOCK_EXPERIMENTS` contains **162 configs** (used with `--h100-mode`):
- 62 structured configs (slip/ent/trade_pen/drawpen variants, seeds, h2048, rmu4424/rmu1228)
- 100 random mutations for exploration
- 4 h100-only configs (h2048 variants) that need actual H100 GPU

**New additions after v2 sweep (2026-03-22):**
- 13 `h100_drawpen_*` variants (drawdown_penalty=0.05 + trade_penalty=0.03, different seeds/params)
- 8 `h100_trade_pen_05_*` variants (different seeds + ent/wd/anneal_ent)
- 4 `h100_rmu4424_*` variants (h=256 small-network regularization)
- 3 `h100_rmu1228_*` variants (obs_norm=True + high ent + no slippage)

With stocks12 data, config families included for coverage (note: all have ~2% positive hit rate):
1. `h100_drawpen_*` — 13 variants (drawdown_pen was lucky v2 run; included for chance of repeat)
2. `h100_trade_pen_05_*` — 8 variants (trade_pen_05_s123 was lucky v2 run; included for coverage)
3. `h100_slip_10bps` — most CONSISTENT config at 90s (90% windows profitable, reproducible)
4. Random mutations — 100 slots for exploration of novel hyperparameter combinations

---

## How to Pull Results

```bash
# View leaderboard sorted by holdout_robust_score
python -c "
import csv
rows = list(csv.DictReader(open('autoresearch_stock_h100_leaderboard.csv')))
rows.sort(key=lambda r: float(r.get('holdout_robust_score') or '-inf'), reverse=True)
for r in rows[:10]:
    print(r['description'], r.get('holdout_robust_score'), r.get('holdout_negative_return_rate'))
"

# Or check the best checkpoint
ls -la pufferlib_market/checkpoints/autoresearch_stock_h100/
```

---

## Deployment Condition

If the best H100 checkpoint beats the current Alpaca deployment (random_mut_2272):

```bash
# Find best checkpoint (replace BEST_CONFIG with description from leaderboard)
BEST_CONFIG="h100_slip_10bps"  # or whatever wins

cp pufferlib_market/checkpoints/autoresearch_stock_h100/${BEST_CONFIG}/best.pt \
   pufferlib_market/checkpoints/stocks_deployment_candidate.pt

echo "Deployed ${BEST_CONFIG} as new stocks_deployment_candidate.pt"
```

Deployment conditions (must beat reliable SOTA random_mut_2272):
- holdout_negative_return_rate = 0% (ALL 20 windows profitable)
- holdout_p10 > +5.2% (random_mut_2272 bar)
- holdout_median_sortino > 1.55 (random_mut_2272 bar)
- holdout_median > +10.5% (random_mut_2272 bar)
- Validated at slippage 0/5/10/20 bps before deploying

---

## Data Files

```
pufferlib_market/data/stocks12_daily_train.bin  (12 symbols, 1302 days: 2022-02-07 to 2025-08-31)
pufferlib_market/data/stocks12_daily_val.bin    (12 symbols,  158 days: 2025-09-01 to 2026-02-05)
```

Symbols: AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, PLTR, JPM, V, AMZN

Data sources: trainingdata/train/{SYM}.csv (split-adjusted, backup at .pre_split_backup)
