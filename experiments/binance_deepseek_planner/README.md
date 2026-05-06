# Binance DeepSeek Graph Planner

Direct DeepSeek API experiment for generating graph-policy configs, then
evaluating them offline with the shared sequential graph simulator. This is not
a live-trading path.

Offline smoke test:

```bash
python experiments/binance_deepseek_planner/deepseek_graph_planner.py \
  --offline \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 240 \
  --max-leverage 2 \
  --allow-short
```

Direct DeepSeek run:

```bash
python experiments/binance_deepseek_planner/deepseek_graph_planner.py \
  --load-env-real \
  --model deepseek-v4-pro \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 2160 \
  --eval-hours 720 \
  --max-leverage 2 \
  --allow-short
```

Notes:

- The script imports `env_real.py` only when `--load-env-real` is passed and
  never prints secret values.
- It calls DeepSeek directly through the OpenAI-compatible client at
  `DEEPSEEK_API_BASE` or `https://api.deepseek.com`.
- The default model for this experiment is `deepseek-v4-pro`.
- DeepSeek emits JSON candidate configs only; every hour of the market replay is
  walked sequentially by local code, so path dependence is preserved.

Iterative daily replanner:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_iterative_replanner.py \
  --load-env-real \
  --provider deepseek \
  --model deepseek-v4-pro \
  --prompt-mode minimal \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --calibration-days 4 \
  --candidates 1 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT
```

This asks DeepSeek for candidate daily templates, mutates them locally, scores
them only on trailing calibration days, then executes the selected template on
the next unseen 24-hour window. Use `--prod-inference` to flip the same
structured template interface to OpenAI `gpt-5.5` with `xhigh` reasoning for
production-time planning after a candidate passes the PnL/DD gate.

Prompt ablation knobs:

- `--prompt-mode minimal` suppresses analysis and explicitly forbids web search,
  tools, code, external facts, and explanatory text.
- `--feature-style line_pct` provides OHLC bars as bps moves from previous close.
- `--feature-style ohlc_mid` provides close return, OHLC offsets from each bar
  midpoint, and midpoint-normalized range as compact line tokens.
- `--disable-cash-guard` forces active template selection for research; leave it
  off for conservative gate-like behavior.

Energy refit agent:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_energy_refit_agent.py \
  --load-env-real \
  --model deepseek-v4-flash \
  --prompt-mode no_think \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --calibration-days 4 \
  --population 72 \
  --boost-pool 192 \
  --boost-take 16 \
  --neural-pool 192 \
  --neural-take 16 \
  --deepseek-refit \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT
```

This staged path seeds with DeepSeek JSON, mutates locally, fits an
XGBoost/CatBoost-style boosted regressor over simulated template scores
(`xgboost` if installed, otherwise sklearn HistGradientBoosting), optionally
asks DeepSeek for a small constrained refit, then fits a small Torch MLP scorer
for a lower-aggression mutation pass. Optional frontier stages include
`--gradient-steps` for direct gradient ascent through the Torch score surrogate
and `--cem-iterations` for cross-entropy/evolution-strategy RL refinement that
can change pair selection, amount, entry, take-profit, and stop dimensions.

Robustness gates:

- `--selection-mode robust` reranks candidates by score plus latest/worst
  calibration performance.
- `--min-positive-calibration-days` requires repeated calibration wins before
  active selection.
- `--require-latest-calibration-positive` rejects candidates failing the most
  recent calibration day.
- `--min-calibration-score` with `--include-cash-guard` forces cash if the best
  active edge is below the specified hurdle.

Rolling GPU energy trainer:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/rolling_energy_trainer.py \
  --train-days 240 \
  --val-days 45 \
  --eval-days 7 \
  --history-hours 240 \
  --templates-per-day 48 \
  --eval-candidates 8192 \
  --gradient-top 128 \
  --gradient-steps 60 \
  --epochs 60 \
  --batch-size 2048 \
  --device cuda \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT
```

This path trains a Torch scorer on many rolling simulated template outcomes
instead of refitting to only the most recent few calibration days. It uses the
same side/allocation/entry/take-profit/stop vector representation, scores large
candidate batches on GPU, gradient-refines the top candidates through the score
surrogate, and gates execution with `--threshold-mode validation_topday` by
default. The gate learns the active prediction hurdle from held-out validation
top-days, so weak candidate days go to cash without peeking at the eval week.

Useful research knobs:

- `--candidate-style legacy|frontier|mixed` keeps the prior simple generator as
  default while allowing volatility-aware/cross-sectional candidate priors.
- `--label-target score|total_return` switches the Torch target between
  risk-adjusted utility and raw next-day return.
- `--ensemble-size` and `--uncertainty-penalty` train multiple GPU scorers and
  rank by ensemble mean minus uncertainty.
- `--rerank-calibration-days` simulates the top predicted candidates on recent
  trailing days before choosing the next-day plan. This uses no future bars, but
  still needs multi-window validation before it should be trusted.
