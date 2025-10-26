#!/usr/bin/env bash
# Synchronise experiment inputs and outputs with the R2 object store.

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: sync_experiment_assets.sh [--run "python run_trade_loop.py"]

Environment variables:
  R2_ENDPOINT            Required Cloudflare R2 endpoint URL.
  R2_BUCKET              Required bucket name (e.g. my-bucket).
  R2_ACCESS_KEY_ID       Required access key id for the bucket.
  R2_SECRET_ACCESS_KEY   Required secret key for the bucket.
  EXPERIMENT_ID          Optional subfolder name (default: marketsim-YYYYmmdd-HHMMSS).
  LOCAL_ROOT             Optional workspace root (default: /workspace).
  LOCAL_LOG_DIR          Optional log directory to upload (default: marketsimulator/run_logs).
  EXTRA_PULL_DIRS        Optional additional comma-separated remote subdirs to pull (relative to models/stock).

The script will:
  1. Sync core datasets (trainingdata*, compiled_models, hyperparams) down from
     s3://$R2_BUCKET/models/stock/ using the provided endpoint.
  2. Optionally run the supplied command.
  3. Sync the logs directory back to
     s3://$R2_BUCKET/models/stock/marketsimulator/logs/$EXPERIMENT_ID/.
USAGE
}

RUN_COMMAND=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --run)
            shift || { echo "--run expects a command" >&2; exit 1; }
            RUN_COMMAND="${1:-}"
            [[ -n "$RUN_COMMAND" ]] || { echo "--run expects a command" >&2; exit 1; }
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift || break
done

: "${R2_ENDPOINT:?Set R2_ENDPOINT to your R2 endpoint}" 
: "${R2_BUCKET:?Set R2_BUCKET to your R2 bucket name}" 
: "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID for AWS CLI}" 
: "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY for AWS CLI}" 

LOCAL_ROOT="${LOCAL_ROOT:-/workspace}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-marketsimulator/run_logs}"
EXPERIMENT_ID="${EXPERIMENT_ID:-marketsim-$(date -u +%Y%m%d-%H%M%S)}"
EXTRA_PULL_DIRS="${EXTRA_PULL_DIRS:-}"

if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI is required but not installed" >&2
    exit 1
fi

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="auto"

REMOTE_ROOT="s3://${R2_BUCKET}/models/stock"
LOGS_REMOTE="${REMOTE_ROOT}/marketsimulator/logs/${EXPERIMENT_ID}"

sync_down() {
    declare -a REMOTES=(
        "trainingdata"
        "trainingdatadaily"
        "trainingdatahourly"
        "compiled_models"
        "hyperparams"
    )

    IFS=',' read -r -a EXTRA <<<"$EXTRA_PULL_DIRS"
    for extra_dir in "${EXTRA[@]}"; do
        [[ -z "$extra_dir" ]] && continue
        REMOTES+=("$extra_dir")
    done

    for remote in "${REMOTES[@]}"; do
        local_target="${LOCAL_ROOT}/${remote}"
        mkdir -p "$local_target"
        echo "Syncing ${REMOTE_ROOT}/${remote}/ to ${local_target}/"
        aws s3 sync "${REMOTE_ROOT}/${remote}/" "${local_target}/" --endpoint-url "$R2_ENDPOINT" --no-progress || true
    done
}

sync_up_logs() {
    local logs_path="${LOCAL_ROOT}/${LOCAL_LOG_DIR}"
    if [[ ! -d "$logs_path" ]]; then
        echo "Log directory ${logs_path} does not exist; skipping upload" >&2
        return 0
    fi
    echo "Uploading logs from ${logs_path} to ${LOGS_REMOTE}/"
    aws s3 sync "${logs_path}/" "${LOGS_REMOTE}/" --endpoint-url "$R2_ENDPOINT" --no-progress
}

run_command() {
    if [[ -z "$RUN_COMMAND" ]]; then
        return 0
    fi
    echo "Running experiment command: $RUN_COMMAND"
    bash -lc "$RUN_COMMAND"
}

main() {
    sync_down

    run_status=0
    if ! run_command; then
        run_status=$?
        echo "Experiment command failed with status ${run_status}" >&2
    fi

    sync_up_logs || true

    exit "$run_status"
}

main "$@"
