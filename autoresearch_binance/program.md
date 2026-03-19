# autoresearch binance

Automated RL research on Binance crypto trading with PufferLib PPO + C market simulator.

## Setup

1. **Create branch**: `git checkout -b autoresearch-binance/<tag>`
2. **Read in-scope files**:
   - `src/autoresearch_binance/program.md` -- this file
   - `src/autoresearch_binance/prepare.py` -- FIXED evaluation harness. DO NOT MODIFY.
   - `src/autoresearch_binance/train.py` -- the file you modify
   - `pufferlib_market/train.py` -- reference PPO + policy architectures
   - `pufferlib_market/environment.py` -- C env wrapper (reference only, don't use TradingEnv directly)

## Architecture

- **Policy**: MLP (TradingPolicy) with 3-layer encoder + actor/critic heads
- **Environment**: C-compiled market simulator via `pufferlib_market.binding`
  - Uses `binding.shared()`, `binding.vec_init()`, `binding.vec_step()`, `binding.vec_reset()`, `binding.vec_close()`
  - 4 symbols (BTC, ETH, DOGE, AAVE), obs_size=73, num_actions=9
  - Shared numpy buffers for obs/act/rew/term/trunc
- **Action space**: Discrete 9 (flat + 4 long + 4 short)
- **Training**: PPO with GAE, parallel envs via C vectorization
- **Data**: Binary market data at `rl-trainingbinance/data/binance6_data.bin`

## Current best

- h1024, ~1.2M steps in 30s, ent_coef=0.05, reward_scale=10, reward_clip=5, cash_penalty=0.01
- robust_score=0.024 (baseline)

## Experimentation

Fixed time budget: **5 minutes** (env var `AUTORESEARCH_BINANCE_TIME_BUDGET_SECONDS`). Launch:
```
PYTHONPATH=$(pwd)/src:$PYTHONPATH .venv/bin/python -m autoresearch_binance.train > run.log 2>&1
```

**What you CAN do:**
- Modify `src/autoresearch_binance/train.py` -- everything: architecture, optimizer, hyperparams, training loop, batch size, reward shaping, feature engineering, learning rate schedule, regularization, etc.
- Create experiment modules under `src/autoresearch_binance/experiments/<slug>/`
- Low-level optimizations: torch.compile, custom kernels, mixed precision, data loading speedups

**What you CANNOT do:**
- Modify `src/autoresearch_binance/prepare.py`
- Modify `pufferlib_market/` source files
- Install new packages

**The goal: get the highest robust_score on holdout evaluation.**

## Known findings (DO NOT re-discover)

- anneal-lr is critical (nearly doubles returns)
- h1024 is the sweet spot for hidden size
- ent_coef=0.05 optimal
- reward_scale=10 + reward_clip=5 + cash_penalty=0.01 = best reward config
- 100M steps beats 300M steps (less overfit) on OOS
- ResidualTradingPolicy available but untested at scale
- Muon optimizer does NOT work for small RL policy networks
- BF16 slightly hurts small RL models

## Promising directions

- Architecture: ResidualMLP, attention over time steps, deeper/wider encoder
- Training: curriculum learning (vary fee/leverage over training), warm restarts
- Speed: torch.compile, fp16 inference during rollout, more envs for throughput
- Reward: multi-objective (return + sortino shaping), hindsight relabeling
- Regularization: weight decay, dropout in policy, spectral norm
- Data: augmentation (noise injection, time warping), bootstrap sampling
- PPO tuning: clip range schedule, dual clip, value clip, PPG
- Batch size / rollout length tradeoffs for 5min budget
- num_envs optimization (more envs = more throughput but more memory)
- Model EMA for evaluation

## Output format

The script prints:
```
---
robust_score:      0.023820
val_return_7d:     0.0424
val_sortino_7d:    10.42
val_return_14d:     0.0830
val_sortino_14d:    5.75
val_return_30d:     -0.1220
val_sortino_30d:    -2.66
training_seconds:  30.1
total_seconds:     30.7
peak_vram_mb:      121.6
total_timesteps:   1245184
num_updates:       76
```

## The experiment loop

LOOP FOREVER:
1. Read current code and recent history
2. Edit `src/autoresearch_binance/train.py` with one coherent improvement
3. git commit
4. Run: `PYTHONPATH=$(pwd)/src:$PYTHONPATH .venv/bin/python -m autoresearch_binance.train > run.log 2>&1`
5. Parse: `grep "^robust_score:\|^peak_vram_mb:" run.log`
6. If grep empty = crash. Run `tail -n 50 run.log` to debug.
7. Record in results.tsv
8. If robust_score improved, keep the commit
9. If worse, `git reset --hard HEAD~1`

**NEVER STOP. Run indefinitely.**
