# Autoresearch Stock Planner

This is an autonomous research setup for improving a stock trading planner under a fixed five-minute training budget.

## Scope

Read these files first:

1. `src/autoresearch_stock/README.md`
2. `src/autoresearch_stock/prompts.md`
3. `src/autoresearch_stock/prepare.py`
4. `src/autoresearch_stock/train.py`

The repo also contains the upstream inspiration in `external/autoresearch/`, but this experiment uses our own data and evaluation harness.

## Rules

What you CAN modify:

- `src/autoresearch_stock/train.py`
- `src/autoresearch_stock/experiments/<experiment_slug>/...` for isolated variants

What you CANNOT modify during the experiment loop:

- `src/autoresearch_stock/prepare.py`
- the evaluation metric
- the spread-loading rules
- dependencies

## Goal

Maximize `robust_score`.

Higher is better.

The evaluator uses:

- real repo OHLCV data,
- live spread history from `dashboards/metrics.db` when available,
- adverse next-bar entry pricing,
- per-symbol volume caps,
- multi-window robust scoring.

## Baseline run

Always begin with a baseline:

```bash
source .venv/bin/activate
python -m autoresearch_stock.train --frequency hourly > run.log 2>&1
grep "^robust_score:\\|^peak_vram_mb:\\|^training_seconds:" run.log
```

## Logging

Keep experiment notes deterministic and replayable.

Preferred locations:

- scheduler turn logs: `analysis/autoresearch_stock_agent/`
- deterministic replay bundle: `experiments/autoresearch_stock_agent/<turn_name>/`
- isolated importable experiment code: `src/autoresearch_stock/experiments/<experiment_slug>/`

If you add a new experiment directory under `src/autoresearch_stock/experiments/`, include a short `README.md` with:

- the hypothesis,
- the borrowed repo ideas,
- the exact benchmark command,
- why it should improve the realistic simulator.

You may also keep an untracked TSV called `src/autoresearch_stock/results.tsv` with columns:

```text
commit	robust_score	memory_gb	status	description
```

Use:

- `keep` when `robust_score` improves,
- `discard` when it does not,
- `crash` when the run fails.

## Experiment loop

1. Check the current git commit.
2. Prefer isolated code under `src/autoresearch_stock/experiments/<experiment_slug>/` if the idea needs more than a tiny edit.
3. Keep `src/autoresearch_stock/train.py` minimal when wiring an experiment in or out.
4. Run one five-minute experiment.
5. Extract `robust_score`.
6. Keep or discard based on real simulator results, not planner loss alone.
7. Record the result in `results.tsv` or in the turn bundle metadata.

## Guidance

- Simpler changes beat complicated ones when the score gain is small.
- Respect the five-minute budget. Optimize for learning efficiency, not long training.
- Prefer ideas that make the planner more robust across hourly and daily data, not just one recent slice.
- Borrow useful ideas from the repo without inheriting unrealistic simulation assumptions.
- Relevant inspiration includes:
  - `rl-trading/src/rl_trading_lab/strategy_language.py`
  - `rl-trading/src/rl_trading_lab/token_language.py`
  - `pufferlib_market/train.py`
  - `unified_hourly_experiment/meta_selector.py`
  - Chronos2 forecast reuse and normalization paths already present in the repo
