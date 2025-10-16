#!/usr/bin/env bash
set -euo pipefail

# Lightweight coverage focusing on auto-generated tests.
# Skips strict torch check and measures only selected packages (default: src).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export SKIP_TORCH_CHECK=${SKIP_TORCH_CHECK:-1}
COVERAGE_PKGS=${COVERAGE_PKGS:-src}

pytest \
  -m auto_generated \
  tests/auto \
  $(printf ' --cov=%s' ${COVERAGE_PKGS}) \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=html:htmlcov \
  -q

echo "\nCoverage XML: coverage.xml"
echo "Coverage HTML: htmlcov/index.html"

