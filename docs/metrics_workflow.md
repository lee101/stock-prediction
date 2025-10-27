# Metrics Collection Workflow

This document captures the workflow implemented so far for extracting
metrics from simulator runs without modifying the trading loop itself.

## 1. Run the simulator while capturing output

Use `tools/run_with_metrics.py` to wrap the simulator invocation and
persist both the raw log and a JSON summary built via
`tools.extract_metrics.extract_metrics`.

```bash
# Example: wrap the default run (replace flags with real ones)
python tools/run_with_metrics.py \
  --log runs/default.log \
  --summary runs/default_summary.json \
  -- --steps 5
```

The wrapper simply calls:

```text
python -m marketsimulator.run_trade_loop <passed arguments>
```

and then parses the generated log for numerical metrics
(`return`, `sharpe`, `pnl`, `balance`).

## 2. Summarise all captured runs

After collecting one or more runs, call
`tools/summarize_results.py` to rebuild `marketsimulatorresults.md`
automatically.

```bash
python tools/summarize_results.py \
  --log-glob 'runs/*.log' \
  --output marketsimulatorresults.md
```

The script scans all matching logs, extracts metrics via
`extract_metrics`, and regenerates `marketsimulatorresults.md` with a
section per run (timestamp + metrics table).

## 3. Notes & next steps

- The wrapper currently expects the CLI flags that `run_trade_loop`
  requires in production; add them after `--` in the example above.
- If you just need to test the tooling pipeline without running the
  real simulator, call `tools/mock_stub_run.py --log … --summary …`
  which emits deterministic-looking logs/metrics for smoke tests.
- `tools/metrics_to_csv.py` can ingest all `*_summary.json` files and
  produce a consolidated CSV table for downstream analysis.
- A fast “stub” execution mode (for smoke tests) is still pending
  until we can safely short-circuit configuration and data loading.
- Log parsing is intentionally simple (regex based). If we add new
  metrics to the simulator log, update
  `tools/extract_metrics.PATTERNS` accordingly.

## 7. Validate summaries

Use `tools/check_metrics.py` to ensure every summary JSON includes the required fields before sharing data:

```bash
python tools/check_metrics.py --glob 'runs/*_summary.json'
```

A JSON Schema lives at `schema/metrics_summary.schema.json` if you want to integrate with other tooling.
