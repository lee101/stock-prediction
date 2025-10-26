#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: launch_fal_training.sh [OPTIONS] [ENDPOINT]

Trigger a fal synchronous endpoint (training or simulator) with sensible defaults.

Options:
  --endpoint URL           Explicit endpoint base URL (overrides positional argument).
  --endpoint-path PATH     Endpoint path appended when missing (default: /api/train).
  --mode MODE              One of train|simulate|auto (default: auto).
  --payload-file PATH      Use raw JSON payload from file.
  --payload-json JSON      Use raw JSON payload from string.
  --symbols CSV            Comma-separated symbol list for simulate mode.
  --steps N                Simulation steps (default: 10).
  --step-size N            Simulation step size (default: 6).
  --initial-cash AMOUNT    Simulation starting cash (default: 100000).
  --top-k N                Simulation top-k picks (default: 4).
  --kronos-only            Force Kronos-only simulation (default: off).
  --no-compact-logs        Disable compact logging for simulation payloads.
  --trainer NAME           Trainer identifier for training payloads (default: hf).
  --epochs N               Training epochs (default: 2).
  --parallel-trials N      Parallel trials for sweeps (default: 2).
  --run-name NAME          Override generated training run name.
  --trainingdata-prefix P  Training data prefix (default: trainingdata).
  --output-root PATH       Training output root (default: /data/experiments).
  --val-days N             Validation horizon in days (default: 30).
  --no-sweeps              Disable sweeps for training payloads.
  --seed N                 Provide explicit seed for training payloads.
  --auth-token TOKEN       Authorization header value.
  --header "Name: Value"    Additional header (repeatable).
  --dry-run                Print the request without sending it.
  -h, --help               Show this message and exit.

Examples:
  launch_fal_training.sh https://fal.run/app-instance
  launch_fal_training.sh --endpoint https://fal.run/app --endpoint-path /api/simulate --dry-run
  launch_fal_training.sh --endpoint https://fal.run/app --payload-file request.json
USAGE
}

append_path_default="/api/train"
endpoint=""
append_path=""
mode="auto"
symbols_csv="AAPL,MSFT,NVDA,BTCUSD,ETHUSD"
steps=10
step_size=6
initial_cash=100000
top_k=4
kronos_only=false
compact_logs=true
trainer="hf"
epochs=2
parallel_trials=2
run_name=""
trainingdata_prefix="trainingdata"
output_root="/data/experiments"
val_days=30
do_sweeps=true
seed=""
auth_token=""
declare -a headers=()
payload_file=""
payload_json=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)
      endpoint=${2-}
      shift 2
      ;;
    --endpoint-path)
      append_path=${2-}
      shift 2
      ;;
    --mode)
      mode=${2-}
      shift 2
      ;;
    --payload-file)
      payload_file=${2-}
      shift 2
      ;;
    --payload-json)
      payload_json=${2-}
      shift 2
      ;;
    --symbols)
      symbols_csv=${2-}
      shift 2
      ;;
    --steps)
      steps=${2-}
      shift 2
      ;;
    --step-size)
      step_size=${2-}
      shift 2
      ;;
    --initial-cash)
      initial_cash=${2-}
      shift 2
      ;;
    --top-k)
      top_k=${2-}
      shift 2
      ;;
    --kronos-only)
      kronos_only=true
      shift
      ;;
    --no-compact-logs)
      compact_logs=false
      shift
      ;;
    --trainer)
      trainer=${2-}
      shift 2
      ;;
    --epochs)
      epochs=${2-}
      shift 2
      ;;
    --parallel-trials)
      parallel_trials=${2-}
      shift 2
      ;;
    --run-name)
      run_name=${2-}
      shift 2
      ;;
    --trainingdata-prefix)
      trainingdata_prefix=${2-}
      shift 2
      ;;
    --output-root)
      output_root=${2-}
      shift 2
      ;;
    --val-days)
      val_days=${2-}
      shift 2
      ;;
    --no-sweeps)
      do_sweeps=false
      shift
      ;;
    --seed)
      seed=${2-}
      shift 2
      ;;
    --auth-token)
      auth_token=${2-}
      shift 2
      ;;
    --header)
      headers+=("$2")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      if [[ -z "$endpoint" && "$1" == http*://* ]]; then
        endpoint=$1
        shift
      else
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$endpoint" ]]; then
  echo "Error: endpoint URL is required." >&2
  usage >&2
  exit 2
fi

if [[ -n "$payload_file" && -n "$payload_json" ]]; then
  echo "Error: use either --payload-file or --payload-json, not both." >&2
  exit 2
fi

if [[ -z "$append_path" ]]; then
  append_path=$append_path_default
fi

append_endpoint() {
  local base=$1
  local path=$2
  if [[ -z "$path" ]]; then
    printf '%s' "$base"
    return
  fi
  local trimmed_base=${base%/}
  local trimmed_path=${path#/}
  if [[ $trimmed_base == *"/$trimmed_path" ]]; then
    printf '%s' "$base"
    return
  fi
  printf '%s/%s' "$trimmed_base" "$trimmed_path"
}

effective_endpoint=$(append_endpoint "$endpoint" "$append_path")

resolved_mode=$mode
if [[ $resolved_mode == auto ]]; then
  if [[ $effective_endpoint == *"/api/simulate" ]]; then
    resolved_mode=simulate
  else
    resolved_mode=train
  fi
fi

if [[ $resolved_mode != train && $resolved_mode != simulate ]]; then
  echo "Error: --mode must be train, simulate, or auto." >&2
  exit 2
fi

if [[ -z "$run_name" && $resolved_mode == train ]]; then
  run_name=$(date -u +"faltrain_%Y%m%d_%H%M%S")
fi

if [[ -z "$payload_json" && -n "$payload_file" ]]; then
  payload_json=$(<"$payload_file")
fi

if [[ -z "$payload_json" ]]; then
  if [[ $resolved_mode == simulate ]]; then
    export LAUNCH_SYMBOLS="$symbols_csv"
    export LAUNCH_STEPS="$steps"
    export LAUNCH_STEP_SIZE="$step_size"
    export LAUNCH_INITIAL_CASH="$initial_cash"
    export LAUNCH_TOP_K="$top_k"
    export LAUNCH_KRONOS_ONLY="$kronos_only"
    export LAUNCH_COMPACT_LOGS="$compact_logs"
    payload_json=$(python - <<'PY'
import json, os, sys
symbols = [s.strip().upper() for s in os.environ["LAUNCH_SYMBOLS"].split(",") if s.strip()]
if not symbols:
    print("No valid symbols parsed from --symbols", file=sys.stderr)
    sys.exit(2)
payload = {
    "symbols": symbols,
    "steps": int(os.environ["LAUNCH_STEPS"]),
    "step_size": int(os.environ["LAUNCH_STEP_SIZE"]),
    "initial_cash": float(os.environ["LAUNCH_INITIAL_CASH"]),
    "top_k": int(os.environ["LAUNCH_TOP_K"]),
    "kronos_only": os.environ["LAUNCH_KRONOS_ONLY"].lower() == "true",
    "compact_logs": os.environ["LAUNCH_COMPACT_LOGS"].lower() == "true",
}
print(json.dumps(payload))
PY
)
  else
    export LAUNCH_RUN_NAME="$run_name"
    export LAUNCH_TRAINER="$trainer"
    export LAUNCH_DO_SWEEPS="$do_sweeps"
    export LAUNCH_PARALLEL_TRIALS="$parallel_trials"
    export LAUNCH_SYMBOLS="$symbols_csv"
    export LAUNCH_EPOCHS="$epochs"
    export LAUNCH_TRAININGDATA_PREFIX="$trainingdata_prefix"
    export LAUNCH_OUTPUT_ROOT="$output_root"
    export LAUNCH_VAL_DAYS="$val_days"
    export LAUNCH_SEED="$seed"
    payload_json=$(python - <<'PY'
import json, os
payload = {
    "run_name": os.environ["LAUNCH_RUN_NAME"],
    "trainer": os.environ["LAUNCH_TRAINER"],
    "do_sweeps": os.environ["LAUNCH_DO_SWEEPS"].lower() == "true",
    "sweeps": {"parallel_trials": int(os.environ["LAUNCH_PARALLEL_TRIALS"])},
    "symbols": [s.strip().upper() for s in os.environ["LAUNCH_SYMBOLS"].split(",") if s.strip()],
    "epochs": int(os.environ["LAUNCH_EPOCHS"]),
    "trainingdata_prefix": os.environ["LAUNCH_TRAININGDATA_PREFIX"],
    "output_root": os.environ["LAUNCH_OUTPUT_ROOT"],
    "val_days": int(os.environ["LAUNCH_VAL_DAYS"]),
    "parallel_trials": int(os.environ["LAUNCH_PARALLEL_TRIALS"]),
}
seed = os.environ.get("LAUNCH_SEED")
if seed:
    payload["seed"] = int(seed)
print(json.dumps(payload))
PY
)
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required to run this script." >&2
  exit 127
fi

echo "POST $effective_endpoint"

echo "$payload_json"

if [[ $dry_run -eq 1 ]]; then
  exit 0
fi

curl_args=(-X POST "$effective_endpoint" -H "Content-Type: application/json")

if [[ -n "$auth_token" ]]; then
  curl_args+=(-H "Authorization: $auth_token")
fi

for header in "${headers[@]}"; do
  curl_args+=(-H "$header")
done

curl_args+=(--data "$payload_json")

curl "${curl_args[@]}"
