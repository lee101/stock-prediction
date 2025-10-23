#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: launch_falmarket_simulation.sh [OPTIONS] [ENDPOINT]

Trigger the fal market simulator endpoint with sensible defaults.

Options:
  --endpoint URL           Explicit endpoint base URL (overrides positional argument).
  --endpoint-path PATH     Endpoint path appended when missing (default: /api/simulate).
  --payload-file PATH      Use raw JSON payload from file.
  --payload-json JSON      Use raw JSON payload from string.
  --symbols CSV            Comma-separated symbol list (default: AAPL,MSFT,NVDA).
  --steps N                Simulation steps (default: 32).
  --step-size N            Simulation step size (default: 1).
  --initial-cash AMOUNT    Simulation starting cash (default: 100000).
  --top-k N                Simulation top-k picks (default: 4).
  --kronos-only            Force Kronos-only mode (default: off).
  --no-compact-logs        Disable compact logging.
  --auth-token TOKEN       Authorization header value.
  --header "Name: Value"   Additional header (repeatable).
  --dry-run                Print the request without sending it.
  -h, --help               Show this message and exit.

Examples:
  launch_falmarket_simulation.sh https://fal.run/app-instance
  launch_falmarket_simulation.sh --endpoint https://fal.run/app --dry-run
USAGE
}

append_path_default="/api/simulate"
endpoint=""
append_path=""
symbols_csv="AAPL,MSFT,NVDA"
steps=32
step_size=1
initial_cash=100000
top_k=4
kronos_only=false
compact_logs=true
payload_file=""
payload_json=""
auth_token=""
declare -a headers=()
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

if [[ -z "$payload_json" && -n "$payload_file" ]]; then
  payload_json=$(<"$payload_file")
fi

if [[ -z "$payload_json" ]]; then
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
