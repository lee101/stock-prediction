# Apr 13 2026 — Strategic Thoughts & Plan

## Honest State Assessment

### What's live and its actual edge

**daily-rl-trader.service** — 32-model ensemble, stocks12, 12.5% allocation
- Feature mismatch: prod ensemble trained pre-RSI, compat-guard keeps it on legacy features
- 120d realistic single-offset replay: +0.21% total / 0.04%/month — effectively flat
- Root bottleneck even if it worked: 12.5% allocation caps actual portfolio return to 0.7-0.9%/month
- SPY 20d MA regime filter is good (bear-market guard, Mar cost -26%)

**LLM orchestrator** — Gemini 3.1-pro, YELP/NET/DBX/OPTX/PDYN/COIN/CRWD
- Event-driven/reasoning signal, no quantified backtest edge
- Worth running at small size as a diversifier
- Potentially interesting on high-news stocks (COIN, CRWD)

### Best proven result not yet deployed

**C_s31 + D_s29 u200 2-model ensemble** (stocks17 data):
- med=18.84% / p10=6.25% / worst=+4.04% / **0/50 neg** / Sortino=48.81
- Cross-variant diversity (AdamW + Muon) is the key — same-variant ensemble hurts
- Passed slippage stress 0/5/10/20 bps (all 0/50 neg)
- Eval: lag=2, binary fills, fee=10bps, 60d×50 windows — realistic

**Gap to close**: 18.84% median vs 27%/month target (crypto-calibrated, not relevant for stocks).
Realistic stock target: 5-8%/month at full allocation, we're at 18.84%/60d ≈ **9-10%/month**.
That's competitive. The blocker is allocation (12.5%) and needing screened32 to supersede stocks17.

---

## The Big Plan — End-to-End on Fast Remote GPU

The goal: train the best possible Chronos2 base → per-symbol LoRA → hyperparams/augmentation
sweep → RL trained on top of all these signals → beat 18.84% median on screened32 holdout.

### Step 1 — Deploy C_s31 + D_s29 now (quick win, <1h)

- Switch `daily-rl-trader.service` from 32-model prod ensemble to 2-model (C_s31 + D_s29)
- These are on stocks17 schema (RSI feature vector) — no compat-guard needed
- Increase allocation: 12.5% → 25% (2 models, split 12.5% each) or up to 50% with regime filter
- Rationale: 0/50 neg at 10bps fees, worst window +4%, this is the safest tested config

### Step 2 — Chronos2 base fine-tune on H100 (RunPod)

Current state:
- v2 (50k steps, 7796 series, Muon): MAE% 2.38% (vs baseline 2.45%)
- v3 (100k steps, stronger aug, running): awaiting result
- Chronos2 as direct signal: ZERO (r=-0.017, 50.1% directional accuracy)

The key insight: **Chronos2 cannot trade directly**. But it can:
1. Feed calibrated forecast features into the RL obs vector (replacing or augmenting Chronos2 raw signal)
2. Act as a pre-augmentation for RL training data (realistic synthetic price paths)
3. Screen stocks by learnability (forecast error correlation with future return)

Full RunPod training plan (H100, 80GB):
- **v4**: 200k steps, ctx=1024, batch=256 grad_accum=4 (eff batch=1024), Muon, amp_log_std=0.50
- Dataset: ALL trainingdata/ stocks (~2258 daily) + ALL binance_spot_hourly/ crypto (~205) + sliding-window variants + return variants = 10k+ series
- Use cutechronos compiled inference for eval loop during training (2-4x speedup vs HuggingFace)
- Rebuild cache with `--rebuild-cache` on the pod (330MB → ~500MB with more series)
- Expected: MAE% < 2.35% (each 50k steps gives ~0.03-0.05pp improvement so far)
- After training: calibrate buy/sell thresholds, upload to R2

### Step 3 — Per-symbol LoRA on top of v4 base

For each of the 32 screened stocks (screened32 universe):
- 500-1000 steps LoRA fine-tune (r=32, α=64)
- Sweep augmentation: amp_log_std ∈ {0.3, 0.45, 0.60}, sliding offsets, noise_frac
- Select best LoRA per symbol by MAE% on held-out recent data (Dec 2025–Apr 2026)
- Export hyperparams to `hyperparams/chronos2/<SYMBOL>.json`

### Step 4 — CuteChronos2 acceleration

`../cutedsl/cutechronos/` has fused Triton kernels (RoPE, RMSNorm-QKV, attention).
Training speedup path:
1. Use cutechronos for **inference-time** eval during RL obs building (already in `pipeline.py`)
2. For **training speedup**: wrap `chronos2_full_finetune.py` to use `torch.compile` on the encoder
   (Triton fused RMSNorm+QKV is the bottleneck at ctx=1024)
3. Benchmark with `cutechronos/benchmark.py` — target: 2-4x faster eval = 2-4x more steps in budget
4. For RunPod: compile once on the H100 with `torch.compile(mode="max-autotune")` and cache the compiled artifact

Key: the cutechronos kernels are drop-in replacements for the HF Chronos2 forward pass.
The model loads HF weights identically — just the forward is accelerated.

### Step 5 — RL on screened32 with all signals

Currently running: C/D variants (tp=0.02/0.05, adamw/muon, h=1024, seeds 1-20 each)
Results: awaiting first 0-neg seed on screened32

Next on RunPod — parallel sweep on 4× H100s:
- C variant seeds 1-50 + D variant seeds 1-50 simultaneously
- Each pod: 13 seeds, ~30min each, full 50-window holdout eval auto-triggered
- **Short-selling variant**: screened32_ext_shorts already has C/D dirs, test it properly
  - Bear market val data (Dec 2025–Apr 2026): this is the real test of robustness
- **Leverage variant**: screened32_leverage_sweep/C — only if 0-neg without leverage first
- Target: find 3rd model for ensemble (needs 0/50 neg on screened32_full_val.bin)
- Ensemble composition: C_s31 (stocks17) + D_s29 (stocks17) + best screened32 seed = 3-model

### Step 6 — Integration and deployment target

When we have a screened32 0-neg champion:
1. Run 3-model ensemble eval: C_s31 + D_s29 + screened32_best
2. Must beat 2-model: p10 > 6.25%, 0/50 neg, med > 18.84%
3. If pass: deploy to `daily-rl-trader.service` with allocation-pct 37.5% (3 × 12.5%)
4. Update alpacaprod.md, commit, restart service

---

## Technical Priorities for RunPod Task

### What goes on the pod

```
1. git clone / rsync stock-prediction repo
2. Build chronos2 data cache (all 2258+ daily + 205 crypto + variants = ~12k series)
3. Train Chronos2 v4 base: 200k steps, H100, torch.compile, cutechronos kernels
4. Per-symbol LoRA sweep: 32 stocks × sweep = ~96 runs × ~5min each
5. Calibrate all: chronos2_linear_calibration.py
6. Parallel: 4 pods × RL seed sweep, stocks17 + screened32
7. Pull all results, eval, report
```

### cutechronos compilation target

File: `../cutedsl/cutechronos/pipeline.py` wraps the full model with:
- Fused RMSNorm+QKV (Triton, `cutechronos/triton_kernels/`)  
- Grouped attention with flash-attention fallback
- `torch.inference_mode()` throughout
- torch.compile-compatible (no graph breaks in model.py)

To use in chronos2_full_finetune.py:
- Replace `AutoModelForSeq2SeqLM.from_pretrained` with `CuteChronos2Model.from_pretrained`
- Only for the eval loop (keep HF model for grad updates — Triton grads not stable)
- Or: compile the HF model's encoder forward with `torch.compile` — simpler, no kernel dep

### Data pipeline expansion

Currently: 7796 series (2258 daily + 205 crypto + 1435 sliding-daily + 3895 return variants)
Target: expand to ALL available data
- `trainingdata/`: ~2258 stocks (5yr daily bars)
- `binance_spot_hourly/`: ~205 crypto (hourly)
- Sliding-window: 7 offsets per series (already done)
- Return variants: log_diff, diff_norm, rank-normalized (already done)
- NEW: ETF holdings (SPY/QQQ constituents) from WRDS/Alpaca historical
- NEW: International ADRs in trainingdata/ if any
- NEW: Multi-horizon targets (3d, 5d, 10d forecasts, not just 1d)

---

## Key Rules to Not Break

1. **0/50 neg is the hard bar** — never add a model with any neg windows to the ensemble
2. **val_full holdout is ground truth** — s42 looked like a breakthrough until val_full showed -12.65% p50
3. **No obs-norm on daily stocks** — proven to collapse training; crypto only
4. **Never trust 5-offset augmented eval** — 1-decision/actual-day is production reality
5. **Binary fills lag=2 only** — soft sigmoid fills have lookahead bias, never trust training Sortino
6. **Regime filter stays** — SPY 20d MA blocks new entries in bear (not force-close)

---

## Monthly Return Reality Check (stocks)

| Allocation | Model edge (med/60d) | Est. monthly |
|---|---|---|
| 12.5% (current) | 18.84% (2-model) | ~1.2%/month portfolio |
| 25% | 18.84% | ~2.4%/month |
| 50% | 18.84% | ~4.7%/month |
| 50% | 27% (target) | ~6.7%/month |
| 100% | 27% (target) | ~13.5%/month |

The 27%/month hard rule is crypto-calibrated. For stocks with daily decisions:
**realistic target is 8-12%/month at ≥50% allocation** — achievable if screened32 finds
a champion that beats stocks17's 18.84% median.

The single biggest lever is **allocation** — not squeezing another 1-2% out of the model.
