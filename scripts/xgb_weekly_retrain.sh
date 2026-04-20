#!/bin/bash
# XGB weekly auto-retrain — re-fits the live 5-seed ensemble with
# train_end = today, health-checks the artifacts, snapshots the current
# live ensemble, atomically rotates the new one in, and restarts the
# xgb-daily-trader-live supervisor process.
#
# Motivation: project_xgb_recency_monotonic.md — recency is asymmetric-
# upside insurance. During regime shifts a 1-year-stale ensemble loses
# ~+86 goodness/year. A weekly retrain costs ~45s GPU.
#
# Safety:
#   - All training writes to a STAGING dir first.
#   - We only rotate if all 5 .pkl files exist, each >100KB, and the
#     manifest parses.
#   - Previous live dir is moved to ${LIVE_DIR}_prev_${TIMESTAMP}/ so
#     a rollback is `mv _prev_... alltrain_ensemble_gpu`.
#   - Backups older than 28 days are deleted.
#   - If the supervisor restart fails the script exits non-zero and
#     the rollback path is still trivially available.
#
# Flags:
#   DRY_RUN=1   — train to staging + verify artifacts, but DO NOT rotate
#                 the live dir and DO NOT restart the supervisor. Leaves
#                 the staging dir in place for inspection.

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

export HOME=/home/administrator
export USER=administrator

REPO=/nvme0n1-disk/code/stock-prediction
cd "${REPO}"

if [ -f "${HOME}/.secretbashrc" ]; then
  # shellcheck disable=SC1090
  set +e; set +u
  source "${HOME}/.secretbashrc" >/dev/null 2>&1
  set -euo pipefail
fi

source .venv/bin/activate
export PYTHONPATH="${REPO}"

LIVE_DIR="${REPO}/analysis/xgbnew_daily/alltrain_ensemble_gpu"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGE_DIR="${LIVE_DIR}_staging_${TIMESTAMP}"
BACKUP_DIR="${LIVE_DIR}_prev_${TIMESTAMP}"
TRAIN_END="$(date -u +%Y-%m-%d)"
LOG_DIR="${REPO}/analysis/xgbnew_daily/retrain_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/retrain_${TIMESTAMP}.log"

echo "[xgb-weekly-retrain] $(date -u +%Y-%m-%dT%H:%M:%SZ) start train_end=${TRAIN_END}" | tee -a "${LOG_FILE}"
echo "[xgb-weekly-retrain] live=${LIVE_DIR}"      | tee -a "${LOG_FILE}"
echo "[xgb-weekly-retrain] staging=${STAGE_DIR}"  | tee -a "${LOG_FILE}"

# ---------------------------------------------------------------- train
# Champion config: 400 trees / depth 5 / lr 0.03, seeds 0,7,42,73,197.
# GPU device. 846-symbol universe at 5M liquidity floor (training universe
# is broad; inference floor of 50M is applied by the live trader).
python -u -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 \
  --train-end "${TRAIN_END}" \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "${STAGE_DIR}" \
  --verbose \
  2>&1 | tee -a "${LOG_FILE}"

# --------------------------------------------------------------- verify
# Every seed must have produced a readable pkl ≥100 KB, and the manifest
# must parse and list all 5 seeds.
REQUIRED_SEEDS=(0 7 42 73 197)
MIN_PKL_BYTES=$((100 * 1024))

for seed in "${REQUIRED_SEEDS[@]}"; do
  pkl="${STAGE_DIR}/alltrain_seed${seed}.pkl"
  if [ ! -f "${pkl}" ]; then
    echo "[xgb-weekly-retrain] FAIL missing ${pkl}" | tee -a "${LOG_FILE}"
    exit 2
  fi
  size=$(stat -c%s "${pkl}")
  if [ "${size}" -lt "${MIN_PKL_BYTES}" ]; then
    echo "[xgb-weekly-retrain] FAIL ${pkl} is ${size} bytes (<${MIN_PKL_BYTES})" \
        | tee -a "${LOG_FILE}"
    exit 3
  fi
done

export STAGE_DIR
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json, os, sys
stage = os.environ["STAGE_DIR"]
m = json.loads(open(os.path.join(stage, "alltrain_ensemble.json")).read())
seeds = sorted(int(x) for x in m.get("seeds", []))
need = [0, 7, 42, 73, 197]
if seeds != need:
    print(f"[xgb-weekly-retrain] FAIL manifest seeds={seeds} expected={need}",
          file=sys.stderr)
    sys.exit(4)
print(f"[xgb-weekly-retrain] OK manifest trained_at={m.get('trained_at')} "
      f"train_end={m.get('train_end')} n_rows={m.get('n_rows')} "
      f"n_symbols={m.get('n_symbols_with_data')}")
PY

# ---------------------------------------------------------------- rotate
if [ "${DRY_RUN}" = "1" ]; then
  echo "[xgb-weekly-retrain] DRY_RUN=1 — skipping rotate + supervisor restart" \
      | tee -a "${LOG_FILE}"
  echo "[xgb-weekly-retrain] staging artifacts kept at ${STAGE_DIR}" \
      | tee -a "${LOG_FILE}"
  echo "[xgb-weekly-retrain] $(date -u +%Y-%m-%dT%H:%M:%SZ) done (dry-run)" \
      | tee -a "${LOG_FILE}"
  exit 0
fi

# Atomic directory swap. `mv` is atomic on ext4 same-fs.
if [ -d "${LIVE_DIR}" ]; then
  echo "[xgb-weekly-retrain] snapshotting ${LIVE_DIR} -> ${BACKUP_DIR}" \
      | tee -a "${LOG_FILE}"
  mv "${LIVE_DIR}" "${BACKUP_DIR}"
fi
mv "${STAGE_DIR}" "${LIVE_DIR}"
echo "[xgb-weekly-retrain] rotated staging -> live" | tee -a "${LOG_FILE}"

# ----------------------------------------------------------------- prune
# Keep last 28 days of backups; delete older.
find "${REPO}/analysis/xgbnew_daily" -maxdepth 1 -type d \
     -name "alltrain_ensemble_gpu_prev_*" -mtime +28 \
     -exec echo "[xgb-weekly-retrain] pruning {}" \; \
     -exec rm -rf {} + 2>&1 | tee -a "${LOG_FILE}" || true

# ---------------------------------------------------------------- restart
# sudo supervisorctl is the production control plane. The live process
# holds the Alpaca singleton lock so it must exit cleanly; autorestart in
# supervisor.conf brings it back immediately.
if command -v supervisorctl >/dev/null 2>&1; then
  echo "[xgb-weekly-retrain] restarting xgb-daily-trader-live" \
      | tee -a "${LOG_FILE}"
  sudo -n supervisorctl restart xgb-daily-trader-live 2>&1 | tee -a "${LOG_FILE}"
else
  echo "[xgb-weekly-retrain] WARN supervisorctl missing; live NOT restarted" \
      | tee -a "${LOG_FILE}"
fi

# --------------------------------------------- in-sample upper-bound report
# Honest-naming: this generates a ``full_fit_insample_overfit_YYYYMMDD/
# insample_OVERFIT_UPPER_BOUND.json`` — the numbers inside are IN-SAMPLE
# by construction (they use the last 8 weeks of the training window). It
# is the ceiling against which to read live PnL. Failure here is
# non-fatal; the retrain is still considered successful.
set +e
python scripts/xgb_full_fit_insample_report.py --weeks 8 2>&1 | tee -a "${LOG_FILE}"
report_rc=$?
set -e
if [ "${report_rc}" -ne 0 ]; then
  echo "[xgb-weekly-retrain] WARN in-sample upper-bound report rc=${report_rc} (non-fatal)" \
      | tee -a "${LOG_FILE}"
fi

echo "[xgb-weekly-retrain] $(date -u +%Y-%m-%dT%H:%M:%SZ) done" | tee -a "${LOG_FILE}"
