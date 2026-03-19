#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv312/bin/python}"

exec "$PYTHON_BIN" -m autoresearch_stock.agent_scheduler \
  --analysis-dir "$REPO_ROOT/analysis/autoresearch_stock_agent" \
  --experiment-bundle-root "$REPO_ROOT/experiments/autoresearch_stock_agent" \
  --repo-root "$REPO_ROOT" \
  --python "$PYTHON_BIN" \
  --backends "${BACKENDS:-codex}" \
  --frequencies "${FREQUENCIES:-hourly,daily}" \
  --loop \
  --max-turns "${MAX_TURNS:-1000000}" \
  --interval-seconds "${INTERVAL_SECONDS:-300}" \
  --codex-reasoning-effort "${CODEX_REASONING_EFFORT:-xhigh}" \
  "$@"
