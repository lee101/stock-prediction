# Metrics Pipeline Quickstart

This guide demonstrates the shortest path to generate metrics locally,
using the tooling we built while the simulator stub mode is pending.

## 1. Create a stub run (fast smoke test)

```bash
python tools/mock_stub_run.py \
  --log runs/stub.log \
  --summary runs/stub_summary.json \
  --seed 123
```

This produces a synthetic log and matching JSON summary in `runs/`.
The output structure mimics the real simulator, so downstream tools
behave identically.

## 2. (Optional) Capture a real simulator run

When you want genuine metrics, wrap the trading loop via
`tools/run_with_metrics.py` and pass whatever CLI flags the simulator
requires:

```bash
python tools/run_with_metrics.py \
  --log runs/live.log \
  --summary runs/live_summary.json \
  -- --steps 5  # add real flags after `--`
```

If the simulator run is long or depends on external configs, make sure
those assets exist before launching.

## 3. Summarise all runs into the main report

```bash
python tools/summarize_results.py \
  --log-glob 'runs/*.log' \
  --output marketsimulatorresults.md
```

The report includes every log encountered, so you can mix stub and live
runs during development.

## 4. Export summaries to CSV (for spreadsheets / BI)

```bash
python tools/metrics_to_csv.py \
  --input-glob 'runs/*_summary.json' \
  --output runs/metrics.csv
```

Each summary JSON becomes a row in `runs/metrics.csv`, containing all
numeric fields plus the originating file path.

## 5. Clean-up tip

The helper scripts never delete files. Periodically prune `runs/` to
avoid clutter:

```bash
rm runs/*.log runs/*_summary.json runs/*.csv
```

You can safely regenerate everything afterwards using the steps above.

## 6. CI smoke test

For automation, use `scripts/metrics_smoke.sh`. It runs the mock -> summary
-> CSV flow and exits non-zero on failure, which is ideal for CI pipelines:

```bash
scripts/metrics_smoke.sh runs/ci-smoke
```
