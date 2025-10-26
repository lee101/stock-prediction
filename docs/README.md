# Metrics Tooling Overview

This folder collects everything needed to capture, validate, and analyse
metrics produced by the trading simulator.

## Quick Start

Follow `docs/metrics_quickstart.md` for the full command-by-command
walkthrough. The highlights:

1. Generate a stub or real run (`tools/mock_stub_run.py` or
   `tools/run_with_metrics.py`).
2. Summarise logs into `marketsimulatorresults.md`
   (`tools/summarize_results.py`).
3. Export the summaries to CSV (`tools/metrics_to_csv.py`).

## Core Utilities

| Script | Purpose |
| --- | --- |
| `tools/mock_stub_run.py` | Creates synthetic log/summary pairs for fast smoke tests. |
| `tools/run_with_metrics.py` | Wraps `python -m marketsimulator.run_trade_loop …` and captures both log and summary JSON. |
| `tools/summarize_results.py` | Sweeps matching logs and regenerates `marketsimulatorresults.md`. |
| `tools/metrics_to_csv.py` | Builds a CSV table from JSON summaries for downstream analysis. |
| `tools/check_metrics.py` | Validates summaries against `schema/metrics_summary.schema.json`. |
| `scripts/metrics_smoke.sh` | End-to-end CI smoke test (mock → summary → CSV). |

## Validation

To ensure every summary file is well-formed:

```bash
python tools/check_metrics.py --glob 'runs/*_summary.json'
```

The underlying schema lives at `schema/metrics_summary.schema.json`.

## Troubleshooting

- **No logs found** – Verify the run wrote `*.log` files to the directory
  you pass to the summariser. For mock runs, re-run
  `tools/mock_stub_run.py`.
- **Invalid JSON** – Run `tools/check_metrics.py` to pinpoint the field.
  Regenerate the summary with `run_with_metrics.py` if necessary.
- **CSV missing fields** – Ensure the summaries include the metrics you
  expect (`return`, `sharpe`, `pnl`, `balance`). The validator will warn
  if any required fields are absent.
- **CI smoke test failures** – Run
  `scripts/metrics_smoke.sh runs/local-smoke` locally to reproduce.

## Simulator Stub Status

The in-process stub mode inside `marketsimulator/run_trade_loop.py` is
still pending until we can safely short-circuit the simulator’s
configuration loading. All tooling above will continue to work with the
stub generator or with real simulator runs once available.

## Make targets

The repository now provides convenience targets:

```bash
make stub-run           # generate a stub log/summary
make summarize          # rebuild marketsimulatorresults.md
make metrics-csv        # export CSV from summaries
make metrics-check      # validate summaries
make smoke              # run the mock-based smoke test
```

Use the `RUN_DIR`/`SUMMARY_GLOB`/`LOG_GLOB` variables to customise locations, e.g. `make RUN_DIR=runs/experiment summarize`.

### Environment overrides

Most scripts honour the Make variables below. Override them on demand:

```bash
make RUN_DIR=runs/my-test summarize
make SUMMARY_GLOB='runs/my-test/*_summary.json' metrics-check
```

For more detailed failure scenarios, see `docs/metrics_troubleshooting.md`.
