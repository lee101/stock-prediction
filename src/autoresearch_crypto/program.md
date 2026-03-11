# autoresearch crypto

This is an experiment to have the LLM do its own research on crypto RL trading.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch-crypto/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch-crypto/<tag>` from current state.
3. **Read the in-scope files**:
   - `src/autoresearch_crypto/program.md` — this file.
   - `src/autoresearch_crypto/prepare.py` — fixed evaluation harness. Do not modify.
   - `src/autoresearch_crypto/train.py` — the file you modify.
   - `binanceneural/model.py` — model architectures (BinanceHourlyPolicy, BinanceHourlyPolicyNano).
   - `binanceneural/config.py` — PolicyConfig, TrainingConfig dataclasses.
   - `differentiable_loss_utils.py` — simulate_hourly_trades, compute_loss_by_type.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Confirm and go**: Confirm setup looks good.

## Architecture

The model outputs 4 values per timestep: buy_price, sell_price, buy_amount, sell_amount.
These are limit orders fed into a differentiable market simulator during training.
Evaluation uses binary fills (all-or-nothing when price touches limit).

Training data: DOGEUSD + AAVEUSD hourly bars with Chronos h1 forecasts.
Simulation: 10bps maker fees, 6.25% margin interest, 2x leverage, fill buffer 5bps, 1-bar decision lag.
Metric: `robust_score` aggregated across 3d/7d/14d/30d holdout windows per symbol (8 scenarios total).

## Experimentation

Each experiment runs on a single GPU. Training runs for a **fixed time budget of 5 minutes**. Launch:

```
.venv313/bin/python -m autoresearch_crypto.train > run.log 2>&1
```

**What you CAN do:**
- Modify `src/autoresearch_crypto/train.py` — everything is fair game: model architecture params, optimizer, hyperparameters, training loop, batch size, loss function, learning rate schedule, feature engineering, etc.

**What you CANNOT do:**
- Modify `src/autoresearch_crypto/prepare.py`. It is read-only. It contains the fixed evaluation.
- Modify `binanceneural/` or `differentiable_loss_utils.py`.
- Install new packages.

**The goal: get the highest robust_score.** Since time is fixed at 5 minutes, everything is fair game within train.py.

## Known findings (DO NOT re-discover)

- Muon optimizer does NOT work for small RL policy networks
- BF16 slightly hurts; use FP32
- wd=0.04 is optimal weight decay for AdamW
- h384 + cosine LR = best architecture combo
- Feature noise HURTS wider models
- Smoothness penalty + cosine LR together = bad
- Softmax allocation for 2-asset crypto FAILS
- Early epochs (1-3) tend optimal; later epochs overtrade

## Promising directions

- Loss functions: calmar, multiwindow, sortino_dd, log_wealth
- Curriculum: vary fill_buffer or fees across training
- Attention window tuning (local vs global)
- LR warmup strategies
- Gradient accumulation for larger effective batch
- Action gating / learned trade thresholds
- EMA model averaging
- Sequence length optimization (48 vs 72 vs 96)
- Model depth vs width tradeoffs
- Nano arch vs classic arch

## Output format

The script prints a summary:

```
---
robust_score:      -116.394282
val_loss:          0.235664
val_sortino:       -0.0091
val_return_pct:    -151.0178
training_seconds:  46.5
total_seconds:     55.3
peak_vram_mb:      403.8
scenario_count:    8
total_trade_count: 2035
num_steps:         116
num_epochs:        1
symbols:           DOGEUSD,AAVEUSD
```

Extract the key metric: `grep "^robust_score:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	robust_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. robust_score achieved — use 0.000000 for crashes
3. peak memory in GB (.1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description

## The experiment loop

LOOP FOREVER:

1. Look at current git state
2. Edit `src/autoresearch_crypto/train.py` with an experimental idea
3. git commit
4. Run: `.venv313/bin/python -m autoresearch_crypto.train > run.log 2>&1`
5. Parse: `grep "^robust_score:\|^peak_vram_mb:" run.log`
6. If grep empty = crash. Run `tail -n 50 run.log` to debug.
7. Record in results.tsv
8. If robust_score improved (higher), keep the commit
9. If robust_score is equal or worse, `git reset --hard HEAD~1`

**Timeout**: Each run ~5 minutes. Kill if >10 minutes.

**Crashes**: Fix typos/easy bugs. If idea is broken, discard and move on.

**NEVER STOP**: Do NOT pause to ask. The human may be asleep. Run indefinitely until manually stopped. If stuck, think harder — try combining ideas, try radical changes, reread code for angles.
