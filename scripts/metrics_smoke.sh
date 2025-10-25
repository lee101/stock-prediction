#!/usr/bin/env bash

# Simple smoke test for the metrics tooling. This avoids running the real
# simulator and instead drives the mock → summariser → CSV steps to confirm
# the pipeline stays healthy.
set -euo pipefail

RUN_DIR="${1:-runs/smoke}"
SEED="${SEED:-123}"

mkdir -p "$RUN_DIR"

python tools/mock_stub_run.py \
  --log "$RUN_DIR/stub.log" \
  --summary "$RUN_DIR/stub_summary.json" \
  --seed "$SEED"

python tools/summarize_results.py \
  --log-glob "$RUN_DIR/*.log" \
  --output "$RUN_DIR/marketsimulatorresults.md"

python tools/metrics_to_csv.py \
  --input-glob "$RUN_DIR/*_summary.json" \
  --output "$RUN_DIR/metrics.csv"

echo "Smoke test finished. Artefacts written to $RUN_DIR/"
