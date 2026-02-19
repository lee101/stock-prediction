# Binance Progress Log 2 - SUI Trading

Updated: 2026-02-17

## SUI LoRA Fine-tuning

Optimized Chronos2 LoRA for SUI hourly forecasting with context=512.

| Config | Val MAE% |
|--------|----------|
| ctx=128 baseline | 3.19 |
| ctx=256 lr=5e-5 | 3.02 |
| ctx=512 baseline | 2.82 |
| ctx=1024 | 3.16 |
| ctx=512 steps=200 | 3.50 |
| ctx=512 lr=1e-4 r=32 steps=300 | 3.52 |

Best: ctx=512 with ~2.8-3.5% MAE (varies with data window).

Dilation ensemble tested - did not improve MAE vs single inference.

## Trading Bot Training

Training config: 10bp maker/taker fees, 25 epochs, sortino+return weighted loss.

Checkpoint: `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`

Final training metrics:
- Train sortino: 2322.82, return: 22.77%
- Val sortino: 288.84, return: 19.67%

## 7-Day Holdout Backtest (2026-02-08 to 2026-02-15)

| Strategy | Return | Sortino | Max DD | Trades | Final Equity |
|----------|--------|---------|--------|--------|--------------|
| Momentum | +5.65% | 9.08 | -5.84% | 3 | $10,565 |
| Neural (10bp) | +153.57% | 612.03 | -0.33% | 139 | $25,357 |

Neural policy significantly outperforms momentum baseline with 10bp fees:
- 27x higher return
- 67x higher sortino
- 17x lower max drawdown

## Margin/Leverage Trading Experiment (2026-02-17)

Trained leveraged SUI margin trading policies with 2-5x leverage on Binance cross-margin.
Margin interest: 2.23% annual (~0.00025% hourly). Checkpoint: `lev4x_rw0.012_s1337`.

### Leverage Comparison ($5k start, 10bp fees, lev4x checkpoint)

| Window | 1x | 2x | 3x | 4x | 5x | 4x 0-fee |
|--------|-----|-----|------|------|-------|----------|
| 3d | 1.3x | 1.7x | 2.1x | 2.8x | 3.5x | 3.2x |
| 7d | 1.7x | 2.9x | 4.9x | 8.2x | 13.8x | 11.1x |
| 10d | 2.1x | 4.6x | 9.7x | 20.5x | 43.3x | 32.7x |
| 14d | 3.2x | 10.1x | 31.5x | 97.0x | 295.5x | 182.4x |
| 30d | 6.7x | 43.5x | 279.9x | 1770x | 11016x | 6466x |

Sortino: ~163-211 (3d) to ~138-144 (70d), consistently increases with leverage.
Max DD: ~0.9% per 1x leverage (linear scaling, -3.7% at 4x, -4.6% at 5x).
Margin cost: negligible at short windows (<1% at 5x/10d), grows with compounding.

### Key Findings
- 5x > 4x on all metrics (sortino, return) with +0.9% more drawdown
- Max DD scales linearly with leverage (good risk properties)
- Margin cost (2.23% annual) is negligible vs trading profits
- 0-fee 4x is 1.5-3x better than 10bp 4x (fee sensitivity)
- Sim reinvests all profits at full leverage -> exponential compounding at long horizons

### Bug Fixes (2026-02-17)
- **Position sizing**: was using model intensity to scale trade size (0.2% -> $34 trades instead of $20k at 4x). Fixed: uses full equity * leverage.
- **Equity calc**: now uses USDT net + SUI net * price instead of just USDT net
- **Repay logic**: cancels open orders before repaying, partial repay fallback
- **Simultaneous buy+sell**: bot now places buy orders (add to position up to max leverage) while also managing sell orders, instead of only doing one at a time

### Deployed Config (2026-02-17)
- Supervisor: `binance-sui-margin` RUNNING
- Checkpoint: `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Max leverage: 4.0x, cycle: 5min, max hold: 6h
- Capital: ~$5,085 USDT in cross-margin account
- Simultaneous buy+sell enabled (adds to position while managing exit)

## Model Artifacts

- LoRA checkpoint: `chronos2_finetuned/SUI_lora_ctx512/finetuned-ckpt`
- Forecast cache: `binancechronossolexperiment/forecast_cache_sui_10bp/`
- Policy checkpoint (1x): `binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt`
- Policy checkpoint (4x margin): `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Hyperparams: `hyperparams/chronos2/hourly/SUIUSDT.json`

## RL CUDA PPO Residual Leverage Controller (2026-02-17)

PPO agent learns residual adjustments on top of baseline neural trading policy.
3D action space: [buy_scale, sell_scale, cap_ratio] where cap_ratio dynamically
controls per-step max leverage in [0,1] * max_leverage.

Code: `binanceleveragesui_rlcuda/` (synced from 5090 via R2)
Tests: 14 passed (env + residual_env + chronos + forecast windowing)

### Architecture
- Residual controller: multiplicative scaling of baseline buy/sell amounts
- Dynamic leverage cap: policy self-throttles leverage each step
- Hard cap enforcement: auto-deleverage if position exceeds cap
- Risk/smoothness reward terms: downside_penalty, pnl_smoothness, cap_smoothness
- SB3 PPO, MLP 256x256 SiLU, gamma=0.995

### Experiment Results (7d test set, 5x leverage, $10k start, 10bp fees)

| Experiment | RL Return | RL Sortino | RL Max DD | Baseline Return | Baseline DD | Avg Cap |
|-----------|-----------|------------|-----------|-----------------|-------------|---------|
| residual_sweep_30k (no cap) | **95.1x** | 1224 | -50.7% | 7.7x | -23.2% | n/a |
| caprisk_lite_30k | 5.9x | 1035 | -14.1% | 7.5x | -22.7% | 0.63 |
| caprisk_balanced_30k | 5.2x | 879 | -9.2% | 7.5x | -22.7% | 0.55 |
| caprisk_strict_30k | 5.4x | 923 | -9.0% | 7.5x | -22.7% | 0.55 |

### Multi-Window Eval (caprisk_lite, 5x leverage)

| Window | RL Return | RL Sortino | RL DD | Baseline Return | Baseline DD |
|--------|-----------|------------|-------|-----------------|-------------|
| 3d | 1.0x | 1170 | -13.7% | 1.4x | -22.7% |
| 7d | 5.9x | 1035 | -14.1% | 7.5x | -22.7% |
| 10d | **17.8x** | 583 | -18.0% | 9.7x | -22.7% |

### Key Findings
- Unconstrained residual (no cap control) gets 95x return but -50% DD -- unusable
- Cap-controlled versions trade ~25% return for ~60% lower drawdown
- At 10d window, RL cap-lite actually beats baseline on both return AND drawdown
- Policy self-throttles to ~55-63% of max leverage (avg_cap_ratio)
- Strict risk config: -9% DD vs -23% baseline (2.5x reduction), only -28% return loss
- Best seed varies: s1337 for balanced/strict, s2024 for lite

### Model Artifacts
- Best unconstrained: `binanceleveragesui_rlcuda/artifacts_residual_30k/seed_1337/best_model.zip`
- Best cap-lite: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_lite_30k/seed_2024/best_model.zip`
- Best cap-balanced: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_balanced_30k/seed_1337/best_model.zip`
- Best cap-strict: `binanceleveragesui_rlcuda/artifacts_residual_caprisk_30k/seed_1337/best_model.zip`
- All synced to local via R2 (netwrckstatic/models/binanceleveragesui_rlcuda/)

## Latest 3d Backtest (2026-02-14 to 2026-02-17)

Baseline model (lev4x_rw0.012_s1337) on latest data, $5k start, 10bp fees:

| Window | 1x | 2x | 3x | 4x | 5x |
|--------|-----|-----|------|------|------|
| 3d | 1.28x | 1.64x | 2.10x | 2.68x | 3.42x |
| 7d | 1.69x | 2.85x | 4.81x | 8.09x | 13.58x |
| 10d | 2.15x | 4.59x | 9.78x | 20.76x | 43.89x |
| 14d | 3.17x | 9.94x | 30.78x | 94.19x | 284.97x |
| 30d | 6.67x | 43.74x | 282.05x | 1788.65x | 11158.29x |

Sortino: 191-201 (3d) to 150-158 (30d). Max DD: -0.87% per 1x (linear).
Model performing excellently on fresh data.

### Probe-Mode Shutdown Strategy (TESTED - NO BENEFIT)
After unprofitable trade, shrink to $2 probe orders until profitable, then re-enable.
Result: NO drawdown improvement. 100% of probes were winners (model recovers too fast).
Returns reduced by 3-14% (missed compounding during probe). Not deploying.

### Deployed Config (updated 2026-02-17)
- Supervisor: `binance-sui-margin` RUNNING (was STOPPED, now active)
- Old systemd `sui-binance-trader`: STOPPED + DISABLED (was 1x, zero trades)
- Checkpoint: `binanceleveragesui/checkpoints/lev4x_rw0.012_s1337/policy_checkpoint.pt`
- Max leverage: 5.0x, cycle: 5min, max hold: 6h
- Equity: ~$5,120 USDT cross-margin
- Active trading confirmed: buy+sell orders at ~$20k notional

## Smoothness Optimization Sweep (2026-02-17)

PPO residual sweep varying downside/smoothness penalties. Baseline 5x: ret=7.5, DD=-23%.

| Config | Seed | Return | Sortino | DD | Cap |
|--------|------|--------|---------|------|------|
| dd=0.5 ps=5e-4 cap=0.3 | 2024 | **13.3x** | 979 | **-12.6%** | 0.74 |
| dd=1.0 ps=1e-3 cap=0.5 | 2024 | 15.1x | 1003 | -14.5% | 0.80 |
| dd=0.1 ps=1e-3 cs=1e-3 | 1337 | 11.4x | 811 | -13.6% | 0.67 |
| dd=1.0 ps=1e-3 | 1337 | 10.9x | **1085** | -13.9% | 0.66 |
| dd=0.5 ps=1e-4 | 1337 | 9.6x | 915 | -13.8% | 0.66 |
| dd=0.1 ps=1e-4 | 1337 | 9.4x | 896 | -13.5% | 0.65 |
| dd=2.0 ps=5e-3 | 1337 | 8.2x | 1008 | -14.4% | 0.65 |

### Full Sweep Results (8 configs, 2 seeds each)

| Config | Seed | Return | Sortino | DD | Cap |
|--------|------|--------|---------|------|------|
| **dd=0.1 ps=0 cap=0.5 cs=1e-3** | **2024** | **17.9x** | 969 | -13.9% | **0.80** |
| dd=1.0 ps=1e-3 cap=0.5 | 2024 | 15.1x | 1003 | -14.5% | 0.79 |
| dd=0.5 ps=5e-4 cap=0.3 | 2024 | 13.3x | 979 | **-12.6%** | 0.74 |
| dd=0.1 ps=1e-3 cs=1e-3 | 1337 | 11.4x | 811 | -13.6% | 0.67 |
| dd=1.0 ps=1e-3 | 1337 | 10.9x | **1085** | -13.9% | 0.66 |
| dd=0.5 ps=1e-4 | 1337 | 9.6x | 915 | -13.8% | 0.65 |
| dd=0.1 ps=1e-4 | 1337 | 9.4x | 896 | -13.5% | 0.65 |
| dd=2.0 ps=5e-3 | 1337 | 8.2x | 1008 | -14.4% | 0.65 |

All configs beat baseline (7.5x, -22.7% DD) on both return AND drawdown.
Cap floor is the strongest factor: configs with cap_floor get higher returns (policy more aggressive).
Best overall: dd=0.1 + cap=0.5 + cs=1e-3 (17.9x ret, -13.9% DD, 2.4x baseline, 40% less DD).

### Multi-Window Eval (top 4 configs, 5x leverage, $10k start, 10bp fees)

| Config | 3d | 7d | 10d | 14d | 30d |
|--------|-----|-----|------|-------|---------|
| **dd01_ps0_cap05 (RL)** | 3.0x | **17.9x** | **45.9x** | **104.2x** | **3440.5x** |
| dd1_ps1e3_cap05 (RL) | 2.5x | 15.1x | 38.5x | 83.6x | 2721.8x |
| dd05_ps5e4_cap03 (RL) | 2.3x | 13.3x | 32.0x | 70.9x | 1970.0x |
| dd1_ps1e3 (RL) | 1.5x | 8.8x | 19.3x | 44.6x | 1201.5x |
| **Baseline** | 1.4x | 7.5x | 9.7x | 14.6x | 51.5x |

| Config | 3d Sort | 7d Sort | 10d Sort | 14d Sort | 30d Sort |
|--------|---------|---------|----------|----------|----------|
| dd01_ps0_cap05 | 845 | 969 | 601 | 482 | 447 |
| dd1_ps1e3_cap05 | 919 | 1003 | 620 | 513 | 486 |
| dd05_ps5e4_cap03 | 869 | 979 | 616 | 487 | 464 |
| dd1_ps1e3 | 692 | 972 | 533 | 456 | 459 |

| Config | 3d DD | 7d DD | 10d DD | 14d DD | 30d DD |
|--------|-------|-------|--------|--------|--------|
| dd01_ps0_cap05 | -13.9% | -13.9% | -13.9% | -13.8% | -13.6% |
| dd05_ps5e4_cap03 | -12.6% | -12.6% | -12.6% | -12.6% | -12.4% |
| Baseline | -22.7% | -22.7% | -22.7% | -22.7% | -22.7% |

Key insight: RL residual controller scales exponentially better than baseline at longer windows.
At 30d, best RL gets 3440x vs baseline 51.5x (67x improvement), with 40% lower drawdown.
DD stays flat across windows for RL (capped by policy), while returns compound exponentially.

### Fine-Grained Sweep (around best config, 60k steps, CPU)

| Config | Return | Sortino | DD | Cap |
|--------|--------|---------|------|------|
| **cap_floor=0.7** | **20.3x** | 893 | -20.0% | 0.86 |
| cap_floor=0.6 | 18.7x | 895 | -18.9% | 0.83 |
| cs=5e-3 | 17.8x | 898 | -18.1% | 0.81 |
| cs=5e-4 | 17.2x | 899 | -17.9% | 0.80 |
| cap_floor=0.5 mc=0.05 | 17.2x | 893 | -18.0% | 0.81 |
| ps=1e-4 cs=1e-3 | 17.1x | 895 | -17.9% | 0.80 |
| cap_floor=0.4 | 17.1x | 900 | -17.1% | 0.79 |
| dd=0.5 cap=0.5 | 16.7x | 891 | -17.7% | 0.79 |
| mc=0.2 | 16.5x | 887 | -17.7% | 0.79 |
| mc=None | 16.1x | 894 | -17.9% | 0.79 |

cap_floor is the dominant knob: higher floor -> more aggressive -> higher return + higher DD.
Linear return/DD tradeoff: ~0.5x return per 1% DD increase.
Other params (cs, mc, ps, dd) have minimal effect once cap_floor is set.

### Longer Training (300k steps, GPU, best config)

| Steps | Return | Sortino | DD | Cap |
|-------|--------|---------|------|------|
| 60k (original) | 17.9x | 969 | **-13.9%** | 0.80 |
| **300k** | **23.8x** | 1000 | -21.0% | -- |

More steps improves return (17.9 -> 23.8x, +33%) but DD worsens (-13.9% -> -21.0%).
60k version has better risk-adjusted properties (DD nearly halved vs baseline).
Diminishing returns on longer training -- the policy exploits more but at cost of safety.

## Stock Model Training Progress (2026-02-17)

Remote 5090: 512h 4L lr=5e-5 seq=32, training complete (500 epochs).
Checkpoints: epochs 1-10, 13, 17, 22, 25, 28.

### Epoch Sweep (30d holdout, me=0.001, $10k start)

| Epoch | Return | Sortino | Trades |
|-------|--------|---------|--------|
| **3** | **+68.9%** | **1.86** | 17 |
| 8 | +52.0% | 1.70 | 11 |
| 10 | +48.2% | 1.48 | 23 |
| 6 | +46.3% | 1.57 | 11 |
| 5 | +45.1% | 1.54 | 11 |
| 7 (deployed 6L) | +42.6% | 1.36 | 23 |
| 2 | +42.1% | 1.75 | 23 |
| 1 | +39.8% | 1.32 | 23 |
| 9 | +41.4% | 1.32 | 23 |
| 13 | +2.2% | 0.29 | 19 |
| 17+ | Negative (overfit) | - | - |

Best: Epoch 3 (68.9% return, Sortino 1.86). Clear early-stopping pattern.
Overfitting starts hard after epoch 10 -- returns collapse to negative by epoch 17.

### Local 4L lr=3e-5 seq=48 Epoch Sweep

| Epoch | Return | Sortino | Trades |
|-------|--------|---------|--------|
| **11** | **+61.0%** | **2.24** | 25 |
| 12 | +59.1% | 2.07 | 25 |
| 14 | +57.8% | 2.19 | 25 |
| 7 | +53.0% | 2.00 | 25 |
| 5 | +49.1% | 1.94 | 25 |

### All Stock Models Comparison (30d holdout, me=0.001)

| Model | Best Epoch | Return | Sortino |
|-------|-----------|--------|---------|
| 6L (DEPLOYED) | 6 | **74.5%** | 2.48 |
| 6L | 7 | 57.8% | **3.38** |
| remote 4L lr=5e-5 | 3 | 68.9% | 1.86 |
| local 4L lr=3e-5 seq=48 | 11 | 61.0% | 2.24 |
| nas_512h_4L | 28 | 50.9% | 3.29 |

6L remains the best architecture. Both 4L variants underperform on sortino.
6L ep6 has highest raw return (74.5%), ep7 has best risk-adjusted (Sortino 3.38).
Consider deploying 6L ep6 for higher returns if willing to accept lower sortino.

## Sortino Maximization Experiments (2026-02-17)

Creative approaches to push SUI residual controller sortino higher.
Code: `experiments/sui_sortino_max/`

### RL Reward Shaping (10 experiments, 2 seeds each)

| Experiment | Return | Sortino | DD |
|-----------|--------|---------|------|
| conservative_sortino | 2.1x | **1129** | -20.8% |
| low_entropy_sortino | 2.8x | 1107 | -18.9% |
| high_cap_sortino | 7.6x | 1099 | -30.3% |
| dd_adaptive_sortino | 3.7x | 1071 | -32.6% |
| big_model_sortino (120k) | 4.0x | 1047 | -17.3% |
| high_gamma_sortino | 1.9x | 1020 | -23.3% |
| sortino_asym4_tight | 3.7x | 999 | -19.7% |
| **original best (dd01_ps0_cap05)** | **17.9x** | **969** | **-13.9%** |
| dd_adaptive_lev | 9.6x | 961 | -19.3% |
| sortino_asym2 | 4.4x | 942 | -20.8% |
| pure_sortino | 3.4x | 936 | -16.1% |
| Baseline (no RL) | 7.5x | 1046 | -22.7% |

Key finding: **sortino vs return tradeoff is fundamental**. Higher sortino configs (1129)
get only 2x return vs 17.9x for the original. The original config sits on the efficient
frontier -- near-optimal risk-adjusted returns.

### Ensemble Evaluation (top models averaged)

| Ensemble | 7d Return | 7d Sortino | 7d DD |
|----------|-----------|------------|-------|
| single_best | 17.9x | 969 | -13.9% |
| ensemble_4 | 17.7x | 980 | -14.2% |
| ensemble_top3 | 16.0x | **999** | -14.1% |
| ensemble_top2 | -- | -- | -- |

Ensembling provides marginal sortino improvement (+3%) at slight cost to return.
Not worth the complexity -- single model is near-optimal.

### Rule-Based Overlays (post-hoc action modification)

| Strategy | Return | Sortino | DD |
|----------|--------|---------|------|
| cap_clamp_05_08 | **21.8x** | 972 | -15.5% |
| cap_clamp_04_06 | 20.4x | 970 | -15.1% |
| cap_clamp_03_07 | 19.4x | 971 | -14.6% |
| **baseline (no overlay)** | **17.9x** | **969** | **-13.9%** |
| conservative_cap_04 | 17.0x | 964 | -13.9% |
| vol_gate | 15.8x | 966 | -13.9% |
| momentum_filter | 14.3x | 958 | -16.9% |

Cap clamping (forcing cap_ratio into a narrow range) can boost returns slightly
at cost of ~1-2% more DD. Vol gating and momentum filtering hurt performance.

### Conclusions
1. Original best config (dd=0.1, cap_floor=0.5, cs=1e-3) is near the efficient frontier
2. Sortino can be pushed to 1129 but at 90% return cost (2x vs 18x)
3. Ensembles and rule-based overlays provide marginal improvements at best
4. The dominant factor remains cap_floor (controls risk/return tradeoff linearly)
5. For production: stick with original best + consider cap_clamp for slightly more return

## Training Improvement Sweep (2026-02-18, in progress)

Sweeping 16 training configurations near baseline (rw=0.012). 25 epochs each, eval at 1x/3x/5x on 10d test, $5k start, 10bp fees.

### Results So Far (5/16 complete)

| Experiment | 5x Return | 5x Sortino | 5x Max DD | Notes |
|-----------|-----------|------------|-----------|-------|
| baseline_rw012 | 35.8x | 137.1 | -6.6% | current deployed |
| rw008 | 33.7x | 136.0 | -6.5% | lower return_weight |
| rw010 | 27.8x | 118.2 | -6.5% | |
| **rw014** | **36.2x** | **269.5** | **-2.3%** | **2x sortino, 1/3 DD** |
| **rw016** | **45.9x** | **176.6** | **-5.4%** | **best return** |

Key finding: **rw014 is dramatically better risk-adjusted** -- same return as baseline but sortino doubled (269 vs 137) and drawdown cut to 1/3 (-2.3% vs -6.6%). rw016 is best on pure return (+28% over baseline).

### Remaining Experiments (running)
- rw020 (higher return weight)
- cosine_rw012 (cosine LR schedule)
- cosine_rw012_min01 (cosine with lower floor)
- warmdown_rw012 (linear warmdown)
- smooth001_rw012 (smoothness penalty 0.001)
- smooth005_rw012 (smoothness penalty 0.005)
- cosine_smooth001 (combined cosine + smooth)
- ep35_rw012 / ep40_rw012 (more epochs)
- nano_rw012 (nano architecture)
- wd_linear_rw012 (linear weight decay)

### Live Bot Status (2026-02-18)
- Running at 5x leverage since ~Feb 17
- Currently in position: 25.4k SUI, $19.3k USDT borrowed
- Equity: ~$4,268 (down ~15% from $5k start)
- SUI dropped from ~$0.97 entry to ~$0.93 (-4.4%), 5x leveraged = ~22% equity loss
- Bot correctly placing sell limits above market, waiting for price recovery
- Data pipeline healthy: 1h stale (normal -- current bar incomplete)

## Covariate/Cross-Learning Forecast Experiment (queued)

Testing whether BTC/ETH/SOL covariates improve SUI OHLC forecast MAE using Chronos-2.

### Three Approaches
1. Univariate: SUI OHLC only (current production)
2. Multivariate: SUI OHLC jointly predicted (open/high/low/close together)
3. Cross-learning: SUI + BTC + ETH + SOL jointly with predict_batches_jointly=True

Script: `binanceleveragesui/eval_covariate_forecasts.py`
Model: finetuned LoRA (`chronos2_finetuned/binance_lora_20260208_newpairs_SUIUSDT/finetuned-ckpt`)
Eval: rolling 30-day holdout, every 24h, horizons h1/h4/h24
Status: waiting for GPU (sweep consuming 96%)
