# Binance OpenAI Structured Planner

OpenAI strict Structured Outputs experiment for graph-policy config generation.
This path is offline-only and shares the sequential graph simulator from
`experiments/binance_transition_graph/`.

Cheap structured-output smoke:

```bash
.venv313/bin/python experiments/binance_openai_planner/openai_graph_planner.py \
  --load-env-real \
  --model gpt-5.4-nano \
  --reasoning-effort low \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 240 \
  --max-leverage 2 \
  --allow-short \
  --candidates 1
```

Production-routing behavior:

- The planner always uses OpenAI Responses API Structured Outputs with
  `json_schema` and `strict: true`.
- The default cheap test model is `gpt-5.4-nano`.
- If a candidate passes `--monthly-target 0.27` and
  `--max-drawdown-target 0.20`, the result artifact records that production
  planning should route to `gpt-5.5` with reasoning effort `xhigh`.
- The script does not need to call the production model during research; the
  gate records the intended production model switch.

GPT-5.5/xhigh one-week replanner:

```bash
.venv313/bin/python experiments/binance_openai_planner/gpt55_week_replanner.py \
  --provider codex \
  --model gpt-5.5 \
  --reasoning-effort low \
  --prompt-mode minimal \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT
```

This asks for one complete daily portfolio plan, then executes the next 24
hourly Binance bars with 10 bps fees, 5 bps slippage, two-hour decision lag,
one entry per symbol per day, and daily forced flattening.

Prompt ablation knobs:

- `--reasoning-effort low` tests the hypothesis that less thinking beats xhigh
  for this short-horizon price-action task.
- `--prompt-mode minimal` tells the model to use only provided OHLC/features and
  avoid tools, web search, code, external facts, or explanations.
- `--feature-style line_pct` provides OHLC bars as bps moves from previous close.
- `--feature-style ohlc_mid` provides compact line tokens with close return,
  OHLC offsets from each bar midpoint, and midpoint-normalized range.
