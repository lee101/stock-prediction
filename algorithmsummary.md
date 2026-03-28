# Algorithm Summary

## As of 2026-03-28

This is a working summary of the best algorithm families we have tried recently for daily multi-pair trading, with an emphasis on deployability rather than one-off backtest wins.

## Current ranking

| Rank | Family | Current read | Why it matters |
|---|---|---|---|
| 1 | Small specialized daily RL (`pufferlib_market`) with strong replay + holdout validation | Best path right now | Fast enough to iterate, close to execution semantics, can be pushed with C/CUDA and stricter validation |
| 2 | Conservative Sortino RL variants (`sortino_rc3`, `robust_reg`) | Best replay family so far, still unstable | Has produced the best replay-combo numbers, but still fails robust holdout often |
| 3 | Oracle-teacher / imitation-augmented RL | Most promising next algorithmic move | Short-horizon robust teacher exposes overfitting directly and can regularize RL |
| 4 | Forecast-first hybrids (Chronos / cache -> policy) | Still worth keeping alive | Good for richer priors, but validation has to stay strict and causal |
| 5 | Large LLM planners | Secondary research path, not the main production path | Useful for structured planning over tools, but not the first thing to optimize for edge right now |

## What looks best now

### 1. Specialized small RL model

This still looks like the right core bet.

- The task is narrow.
- The simulator matters more than broad world knowledge.
- Fast step throughput and honest validation matter more than giant parameter count.
- A `~100MB` class model with a better simulator and better regularization is more credible than a huge general model for direct signal extraction.

The right direction here is:

- keep pushing `pufferlib_market`
- use C / fused kernels / fast env stepping
- make validation harsher
- reject overfit winners earlier

### 2. `sortino_rc3` family

This is the current best replay family, but not yet production quality.

Recent evidence:

- local `sortino_rc3_tp08` proof:
  - replay-combo `+12.86`
  - holdout robust `-39.42`
- remote `sortino_rc3_tp08_slip8`:
  - replay-combo `+39.14`
  - holdout robust `-170.70`
- remote `sortino_rc3_tp09`:
  - replay-combo `+11.35`
  - holdout robust `-98.01`

Interpretation:

- this family can find replay edge
- it is still too path-sensitive
- the current winners are false positives until holdout robustness is materially better

### 3. Oracle teacher

This is now the most important algorithmic addition.

We added a short-horizon robust oracle in `pufferlib_market/oracle_plan.py` that:

- scores all legacy actions using the real fill-buffer / fee / slippage semantics
- evaluates them across multiple fill-buffer and slippage scenarios
- emits best and near-best actions
- compares model actions against the robust teacher

Why it matters:

- it gives a direct regret signal, not just final PnL
- it catches sparse “flat most of the time” policies that still get lucky replay wins
- it gives us a clean auxiliary supervised target for RL

Current result on local `sortino_rc3_tp08`:

- exact oracle match: `0.00`
- near-best oracle match: `0.00`
- mean regret: `22.03`

That strongly supports the idea that overfitting is a core issue right now.

### 4. Forecast-first hybrids

This branch remains alive, but it needs direct apples-to-apples comparisons
instead of isolated wins.

The best current shape for this family is:

- one shared Chronos2 upstream
- one hourly forecast-cache RL branch
- one daily RL branch using causal hourly forecast-context features from
  `pufferlib_market.export_data_daily_v4`
- same symbols
- same date window
- same downstream sweep family

That comparison pipeline is now wired and running remotely, which should tell us
whether Chronos helps more as hourly direct features or as compact daily context.

## What looks weak

### GSPO / group-relative branch

This branch is not where more compute should go right now.

Recent re-runs failed to reproduce prior promising shapes and came back materially negative on replay-combo or holdout.

Summary:

- good idea in principle
- poor stability in practice
- not the best current use of budget

### Giant LLM-first trading model

Nemotron-style tuning is interesting, but it is not the highest-probability mainline for our current edge search.

Why:

- we do not have huge amounts of clean finance-specific supervised data
- broad reasoning ability does not replace a realistic simulator
- finance edge is more likely to come from structure, tooling, constraints, and validation than raw model scale

If we use a large model, the best role is:

- structured planner over tools and features
- not raw market predictor
- likely PEFT only: LoRA / IA3 / DPO / light RL

That is a sidecar architecture, not the main alpha engine.

## Validation principles

These should drive the next round of research:

- Replay-combo alone is not enough.
- Holdout robustness must stay central.
- Oracle regret should become a first-class metric.
- Negative holdout families should be ranked down even if replay is good.
- Early exits should reject configs with obvious generalization gaps.

We now added a `generalization_score` into `pufferlib_market.autoresearch_rl` to do exactly that:

- blends replay-combo and holdout robust score
- penalizes train/val gap
- penalizes replay deterioration versus validation
- penalizes high holdout negative-return rate

This should make future leaderboards more honest.

## Best next experiments

### Track A: Mainline

- keep training small specialized RL policies
- use `generalization_score` or replay+holdout together when ranking
- continue remote short sweeps on the strongest conservative families

### Track B: New algorithmic move

- generate oracle labels on hard windows
- add an imitation / auxiliary loss against oracle best or near-best actions
- measure both replay-combo and oracle regret

### Track C: Systems

- inspect `PufferLib` 4.0 style fast stepping
- push more work into C / custom kernels / lower-overhead policy loops
- optimize for more useful gradient steps per wall-clock hour

### Track D: Optional LLM sidecar

- only after the core RL loop is stronger
- use a model like Nemotron as a planner over tools, not as the main signal model
- SFT -> DPO -> light cascaded RL, with PEFT only

## Bottom line

The best algorithm today is still not a giant foundation model. It is a fast, specialized, heavily regularized daily RL system with a realistic simulator, harsh validation, and now an oracle-style teacher to expose overfitting directly.
