# Autoresearch Stock Planner

This is a stock-trading adaptation of Karpathy's `autoresearch` pattern.

The structure stays the same:

- `prepare.py` is fixed. It owns data loading, live spread profile lookup, and the evaluation harness.
- `train.py` is the main entrypoint the autonomous agent is meant to optimize.
- `src/autoresearch_stock/experiments/` is the preferred home for isolated multi-file experiment variants.
- `program.md` is the human-written instruction file for the agent loop.
- `prompts.md` and `prompts/` contain reusable manual prompt packs.

Unlike the original repo, the objective is not language-model loss. The objective here is a realistic trading eval under a fixed five-minute training budget.

## What is fixed

- Hourly or daily OHLCV data from this repo.
- Next-bar entry with adverse fill assumptions.
- Live spread history from `dashboards/metrics.db` when available.
- Conservative fallback spread costs when live history is missing.
- Volume-fraction caps to avoid unrealistic sizing.
- Multi-window robust scoring instead of one backtest slice.

## Metric

`robust_score` is the optimization target. Higher is better.

It is computed from multiple holdout windows and penalizes:

- negative returns,
- deep drawdowns,
- jagged equity curves,
- overly sparse or overly active trading.

## Setup

Activate an environment first, then inspect the fixed task:

```bash
source .venv/bin/activate
python -m autoresearch_stock.prepare --frequency hourly
python -m autoresearch_stock.prepare --frequency daily
```

Run one baseline experiment:

```bash
python -m autoresearch_stock.train --frequency hourly
python -m autoresearch_stock.train --frequency daily
```

Before a longer run, validate the resolved symbols and roots without touching CUDA:

```bash
python -m autoresearch_stock.train \
  --frequency hourly \
  --symbols NVDA,PLTR,GOOG \
  --data-root trainingdatahourly/stocks \
  --check-inputs-text
```

`--data-root` is optional when you use the repo's conventional layouts:
- hourly defaults to `trainingdatahourly/stocks`
- daily defaults to `trainingdata`

For machine-readable preflight output:

```bash
python -m autoresearch_stock.train \
  --frequency daily \
  --symbols AAPL,AMD,NVDA \
  --check-inputs
```

Useful overrides:

```bash
python -m autoresearch_stock.train \
  --frequency hourly \
  --symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH \
  --eval-windows 35,140,420 \
  --dashboard-db dashboards/metrics.db
```

Config notes:

- `--max-positions` must be at least `1`.
- `--max-volume-fraction` must be in `(0, 1]`.
- `--recent-overlay-bars`, slippage values, and `--min-edge-bps` must be non-negative.
- `--spread-lookback-days` must be at least `1`.
- `--annual-leverage-rate` must be non-negative and `--max-gross-leverage` must be at least `1.0`.
- `AUTORESEARCH_STOCK_INPUT_CHECK_WORKERS` optionally caps `--check-inputs` parallelism.

Current experimental flags:

```bash
python -m autoresearch_stock.train --frequency hourly --dynamic-score-floor
python -m autoresearch_stock.train --frequency hourly --soft-rank-sizing
python -m autoresearch_stock.train --frequency hourly --timestamp-budget-head
python -m autoresearch_stock.train --frequency hourly --budget-guided-keep-count
python -m autoresearch_stock.train --frequency hourly --continuous-budget-thresholds
python -m autoresearch_stock.train --frequency hourly --budget-entropy-confidence
python -m autoresearch_stock.train --frequency hourly --budget-consensus-dispersion
```

## Agent Loop

Run the autonomous 5-minute improvement loop:

```bash
python -m autoresearch_stock.agent_scheduler \
  --analysis-dir analysis/autoresearch_stock_agent \
  --experiment-bundle-root experiments/autoresearch_stock_agent \
  --repo-root . \
  --python .venv312/bin/python \
  --backends codex \
  --frequencies hourly,daily \
  --loop \
  --interval-seconds 300
```

Or use the wrapper script:

```bash
scripts/run_autoresearch_stock_agent_loop.sh
```

Artifacts land under `analysis/autoresearch_stock_agent/`:

- `backend_status.json` for backend availability
- `turns.jsonl` for per-turn records
- `leaderboard.tsv` for sortable summaries
- one directory per turn with prompt, backend stdout/stderr, result JSON, train log, and train diff

Deterministic replay bundles land under `experiments/autoresearch_stock_agent/`:

- `prompt.md`
- `benchmark_command.sh`
- `train.py.before`
- `train.py.after`
- copied backend logs and train logs when present
- copied prompt packs
- `metadata.json`

Right now Codex is usable on this machine. Claude is supported but disabled by default here because the current CLI auth/setup can hang during probing. Re-enable it only after fixing Claude auth:

```bash
python -m autoresearch_stock.agent_scheduler --backends codex,claude --allow-claude
```

## Prompt Packs

Prompt packs live under `src/autoresearch_stock/prompts/`, with an index at
`src/autoresearch_stock/prompts.md`.

Example manual hourly run:

```bash
python -m autoresearch_stock.agent_scheduler \
  --analysis-dir analysis/autoresearch_stock_agent \
  --experiment-bundle-root experiments/autoresearch_stock_agent \
  --repo-root . \
  --python .venv312/bin/python \
  --backends codex \
  --frequencies hourly \
  --max-turns 1 \
  --prompt-file src/autoresearch_stock/prompts/isolated_experiment_rules.md \
  --prompt-file src/autoresearch_stock/prompts/hourly_selectivity.md \
  --prompt-file src/autoresearch_stock/prompts/chronos_plan_language.md
```

## Notes

- The default time budget is 300 seconds via `AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS`.
- That env var exists mainly for tests and smoke runs. Normal experiments should leave it at five minutes.
- The live spread profile is optional. If `dashboards/metrics.db` is absent, the harness falls back to conservative symbol-level cost estimates.
