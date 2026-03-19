#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run_coverage.sh [pytest-args...]
# Produces terminal + XML + HTML coverage reports.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTEST_ARGS=(${@:-})

# Packages to measure; default to 'src' to avoid flooding report with non-target dirs.
COVERAGE_PKGS=${COVERAGE_PKGS:-src}

pytest \
  $(printf ' --cov=%s' ${COVERAGE_PKGS}) \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=html:htmlcov \
  -q ${PYTEST_ARGS[@]:-}

echo "\nCoverage XML: coverage.xml"
echo "Coverage HTML: htmlcov/index.html"
