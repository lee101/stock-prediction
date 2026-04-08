#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH="${LAUNCH:-/home/lee/code/stock/deployments/binance-hybrid-spot/launch.sh}"
LOG="${LOG:-/var/log/supervisor/binance-hybrid-spot.log}"
SUPERVISOR_PROG="${SUPERVISOR_PROG:-binance-hybrid-spot}"
PYTHON_BIN="${PYTHON_BIN:-/home/lee/code/stock/.venv313/bin/python}"
TRACE_DIR="${TRACE_DIR:-/home/lee/code/stock/strategy_state/hybrid_trade_cycles}"

usage() {
    cat <<'USAGE'
Usage: deploy_crypto_model.sh [--dry-run] [--remove-rl] [--force] [--manifest-path PATH] [--symbols "BTCUSD ETHUSD"] [--leverage 2.0] [--wait-for-live-cycle-seconds 120] <checkpoint_path> [symbols...]
       deploy_crypto_model.sh [--dry-run] [--remove-rl] [--force] [--manifest-path PATH] [--symbols "BTCUSD ETHUSD"] [--leverage 2.0] [--wait-for-live-cycle-seconds 120] [--min-healthy-live-cycles 2] <checkpoint_path> [symbols...]
       deploy_crypto_model.sh [--dry-run] [--force] [--manifest-path PATH] [--symbols "BTCUSD ETHUSD"] [--leverage 2.0] [--wait-for-live-cycle-seconds 120] [--min-healthy-live-cycles 2] --manifest-best
       deploy_crypto_model.sh [--dry-run] [--force] [--wait-for-live-cycle-seconds 120] [--min-healthy-live-cycles 2] --grid-best PATH

Deploy an RL checkpoint to the Binance hybrid spot bot.

  checkpoint_path   Path to .pt checkpoint file (absolute or relative)
  symbols           Override symbols (default: keep existing). Use either positional symbols or --symbols.

Options:
  --dry-run         Show what would change without modifying anything
  --remove-rl       Remove --rl-checkpoint from launch.sh (revert to LLM-only)
  --force           Override the production-eval deploy gate
  --manifest-path   Explicit prod eval manifest to use for the deploy gate
  --manifest-best   Resolve and deploy the best production-evaluated checkpoint from the manifest
  --grid-best       Resolve and deploy the best gated checkpoint/config from a config-grid output dir or best_by_config.csv
  --symbols         Override symbols as a quoted space- or comma-separated string
  --leverage        Override launch leverage for deploy/eval compatibility
  --wait-for-live-cycle-seconds
                    Wait up to N seconds for a healthy post-deploy live cycle snapshot
  --min-healthy-live-cycles
                    Require at least N healthy live cycle snapshots after deploy

Examples:
  deploy_crypto_model.sh rl-trainingbinance/checkpoints/best.pt
  deploy_crypto_model.sh --dry-run binanceneural/checkpoints/epoch_003.pt
  deploy_crypto_model.sh --dry-run --manifest-path analysis/current_binance_prod_eval_20260409/prod_launch_eval_manifest.json --manifest-best
  deploy_crypto_model.sh --dry-run --manifest-best --symbols "BTCUSD ETHUSD"
  deploy_crypto_model.sh --dry-run --manifest-best --leverage 2.0
  deploy_crypto_model.sh --dry-run --manifest-best --wait-for-live-cycle-seconds 300 --min-healthy-live-cycles 2
  deploy_crypto_model.sh --manifest-path analysis/current_binance_prod_eval_20260409/prod_launch_eval_manifest.json --wait-for-live-cycle-seconds 120 --manifest-best
  deploy_crypto_model.sh --dry-run --grid-best analysis/binance_prod_config_grid_20260409_120000
  deploy_crypto_model.sh --remove-rl
USAGE
    exit 0
}

DRY_RUN=0
REMOVE_RL=0
FORCE=0
MANIFEST_PATH=""
MANIFEST_BEST=0
GRID_BEST=""
CHECKPOINT=""
SYMBOLS=""
SYMBOLS_OPTION=""
backup=""
PREVIOUS_CHECKPOINT=""
PREVIOUS_SYMBOLS=""
EXPECTED_SYMBOLS=""
EXPECTED_SYMBOLS_CSV=""
LEVERAGE_OVERRIDE=""
PREVIOUS_LEVERAGE=""
EXPECTED_LEVERAGE=""
WAIT_FOR_LIVE_CYCLE_SECONDS=""
MIN_HEALTHY_LIVE_CYCLES=""

run_python() {
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
}

normalize_symbols() {
    local raw="$1"
    printf '%s\n' "$raw" | tr ',\t\r\n' '    ' | xargs
}

normalize_leverage() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        return 0
    fi
    run_python - "$raw" <<'PY'
from src.binance_hybrid_launch import parse_leverage_override
import sys

value = parse_leverage_override(sys.argv[1])
if value is None:
    raise SystemExit(1)
print(f"{value:g}")
PY
}

normalize_nonnegative_float() {
    local raw="$1"
    run_python - "$raw" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0.0:
    raise SystemExit(1)
print(f"{value:g}")
PY
}

normalize_nonnegative_int() {
    local raw="$1"
    run_python - "$raw" <<'PY'
import sys

value = int(sys.argv[1])
if value < 0:
    raise SystemExit(1)
print(value)
PY
}

apply_symbols_override() {
    local symbols="$1"
    if [[ -z "$symbols" ]]; then
        return 0
    fi
    if grep -q '\-\-symbols' "$LAUNCH"; then
        sed -i "s|--symbols [^\\\\]*\\\\|--symbols $symbols \\\\|" "$LAUNCH"
        echo "Updated --symbols to: $symbols"
    else
        sed -i "/--model /a\\  --symbols $symbols \\\\" "$LAUNCH"
        echo "Added --symbols to launch.sh: $symbols"
    fi
}

apply_leverage_override() {
    local leverage="$1"
    if [[ -z "$leverage" ]]; then
        return 0
    fi
    if grep -q '\-\-leverage' "$LAUNCH"; then
        sed -i "s|--leverage [^ \\\\]*|--leverage $leverage|" "$LAUNCH"
        echo "Updated --leverage to: $leverage"
    else
        sed -i "/--execution-mode /a\\  --leverage $leverage \\\\" "$LAUNCH"
        echo "Added --leverage to launch.sh: $leverage"
    fi
}

restart_supervisor() {
    echo "ilu" | sudo -S supervisorctl restart "$SUPERVISOR_PROG" 2>/dev/null
}

show_recent_logs() {
    if [[ -f "$LOG" ]]; then
        echo ""
        echo "--- Recent logs ---"
        tail -20 "$LOG"
    fi
}

verify_launch_state() {
    local expected_checkpoint="$1"
    local expected_symbols="$2"
    local expected_leverage="$3"
    local deployed_after="$4"
    local wait_timeout="${5:-}"
    local min_healthy_cycles="${6:-}"
    local -a verify_args=(
        -m src.binance_hybrid_prod_verify
        --launch-script "$LAUNCH"
        --expected-checkpoint "$expected_checkpoint"
        --log-path "$LOG"
        --trace-dir "$TRACE_DIR"
        --deployed-after "$deployed_after"
    )
    if [[ -n "$expected_symbols" ]]; then
        verify_args+=( --expected-symbols "$expected_symbols" )
    fi
    if [[ -n "$expected_leverage" ]]; then
        verify_args+=( --expected-leverage "$expected_leverage" )
    fi
    if [[ -n "$wait_timeout" ]]; then
        verify_args+=( --require-live-snapshot --wait-timeout-seconds "$wait_timeout" )
    fi
    if [[ -n "$min_healthy_cycles" ]]; then
        verify_args+=( --min-healthy-live-cycles "$min_healthy_cycles" )
    fi
    run_python "${verify_args[@]}"
}

verify_launch_without_rl() {
    local expected_symbols="$1"
    local expected_leverage="$2"
    local deployed_after="$3"
    local wait_timeout="${4:-}"
    local min_healthy_cycles="${5:-}"
    local -a verify_args=(
        -m src.binance_hybrid_prod_verify
        --launch-script "$LAUNCH"
        --expect-no-rl-checkpoint
        --log-path "$LOG"
        --trace-dir "$TRACE_DIR"
        --deployed-after "$deployed_after"
    )
    if [[ -n "$expected_symbols" ]]; then
        verify_args+=( --expected-symbols "$expected_symbols" )
    fi
    if [[ -n "$expected_leverage" ]]; then
        verify_args+=( --expected-leverage "$expected_leverage" )
    fi
    if [[ -n "$wait_timeout" ]]; then
        verify_args+=( --require-live-snapshot --wait-timeout-seconds "$wait_timeout" )
    fi
    if [[ -n "$min_healthy_cycles" ]]; then
        verify_args+=( --min-healthy-live-cycles "$min_healthy_cycles" )
    fi
    run_python "${verify_args[@]}"
}

resolve_best_checkpoint_from_manifest() {
    local manifest_arg="$1"
    local symbols_override="$2"
    local leverage_override="$3"
    run_python - "$LAUNCH" "$manifest_arg" "$symbols_override" "$leverage_override" <<'PY'
from src.binance_deploy_gate import resolve_best_deploy_candidate
import sys

launch_script = sys.argv[1]
manifest_path = sys.argv[2] or None
symbols_override = sys.argv[3] or None
leverage_override = sys.argv[4] or None
result = resolve_best_deploy_candidate(
    launch_script=launch_script,
    manifest_path=manifest_path,
    symbols_override=symbols_override,
    leverage_override=leverage_override,
)
if not result.checkpoint:
    if result.metric_name is not None:
        sys.stderr.write(
            f"{result.reason}: {result.metric_name} "
            f"current={result.current_metric!r} candidate={result.candidate_metric!r}\n"
        )
    else:
        sys.stderr.write(f"{result.reason}\n")
    raise SystemExit(1)
print(result.checkpoint)
PY
}


resolve_best_grid_target() {
    local grid_arg="$1"
    run_python - "$LAUNCH" "$grid_arg" <<'PYSUB'
from src.binance_deploy_gate import resolve_best_grid_deploy_candidate
import sys

launch_script = sys.argv[1]
grid_output_path = sys.argv[2]
result = resolve_best_grid_deploy_candidate(
    launch_script=launch_script,
    grid_output_path=grid_output_path,
)
if not result.checkpoint:
    if result.metric_name is not None:
        sys.stderr.write(
            f"{result.reason}: {result.metric_name} "
            f"current={result.current_metric!r} candidate={result.candidate_metric!r}\n"
        )
    else:
        sys.stderr.write(f"{result.reason}\n")
    raise SystemExit(1)
print(result.checkpoint)
print(result.symbols_override)
print("" if result.leverage_override is None else f"{result.leverage_override:g}")
print(result.manifest_path or "")
PYSUB
}

rollback_launch() {
    local failure_reason="$1"
    if [[ -z "$backup" || ! -f "$backup" ]]; then
        echo "Rollback unavailable: missing launch backup."
        return 1
    fi

    echo ""
    echo "Deployment failed: $failure_reason"
    echo "Rolling back launch.sh from $backup"
    cp "$backup" "$LAUNCH"
    echo "Restored launch.sh from backup"
    echo ""
    echo "Restarting $SUPERVISOR_PROG with previous config..."
    local rollback_started_at
    rollback_started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    if ! restart_supervisor; then
        echo "Rollback restart failed."
        return 1
    fi
    sleep 5
    show_recent_logs
    echo ""
    echo "Verifying rollback..."
    if [[ -n "$PREVIOUS_CHECKPOINT" ]]; then
        verify_launch_state "$PREVIOUS_CHECKPOINT" "$PREVIOUS_SYMBOLS" "$PREVIOUS_LEVERAGE" "$rollback_started_at"
    else
        verify_launch_without_rl "$PREVIOUS_SYMBOLS" "$PREVIOUS_LEVERAGE" "$rollback_started_at"
    fi
}

if [[ ! -f "$LAUNCH" ]]; then
    echo "ERROR: Launch script not found: $LAUNCH"
    exit 1
fi

mapfile -t _launch_info < <(
    run_python - "$LAUNCH" <<'PY'
from src.binance_hybrid_launch import parse_launch_script
import sys

cfg = parse_launch_script(sys.argv[1], require_rl_checkpoint=False)
print(cfg.rl_checkpoint or "")
print(" ".join(cfg.symbols))
print(cfg.leverage)
PY
)
PREVIOUS_CHECKPOINT="${_launch_info[0]:-}"
PREVIOUS_SYMBOLS="${_launch_info[1]:-}"
PREVIOUS_LEVERAGE="${_launch_info[2]:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage ;;
        --dry-run) DRY_RUN=1; shift ;;
        --remove-rl) REMOVE_RL=1; shift ;;
        --force) FORCE=1; shift ;;
        --manifest-best) MANIFEST_BEST=1; shift ;;
        --grid-best)
            [[ $# -lt 2 ]] && { echo "ERROR: --grid-best requires a value"; exit 1; }
            GRID_BEST="$2"
            shift 2
            ;;
        --manifest-path)
            [[ $# -lt 2 ]] && { echo "ERROR: --manifest-path requires a value"; exit 1; }
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --symbols)
            [[ $# -lt 2 ]] && { echo "ERROR: --symbols requires a value"; exit 1; }
            SYMBOLS_OPTION="$2"
            shift 2
            ;;
        --leverage)
            [[ $# -lt 2 ]] && { echo "ERROR: --leverage requires a value"; exit 1; }
            LEVERAGE_OVERRIDE="$2"
            shift 2
            ;;
        --wait-for-live-cycle-seconds)
            [[ $# -lt 2 ]] && { echo "ERROR: --wait-for-live-cycle-seconds requires a value"; exit 1; }
            WAIT_FOR_LIVE_CYCLE_SECONDS="$2"
            shift 2
            ;;
        --min-healthy-live-cycles)
            [[ $# -lt 2 ]] && { echo "ERROR: --min-healthy-live-cycles requires a value"; exit 1; }
            MIN_HEALTHY_LIVE_CYCLES="$2"
            shift 2
            ;;
        -*) echo "ERROR: Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$CHECKPOINT" ]]; then
                CHECKPOINT="$1"
            else
                SYMBOLS="${SYMBOLS:+$SYMBOLS }$1"
            fi
            shift
            ;;
    esac
done

if [[ "$REMOVE_RL" -eq 1 && "$MANIFEST_BEST" -eq 1 ]]; then
    echo "ERROR: --manifest-best cannot be used with --remove-rl"
    exit 1
fi

if [[ "$REMOVE_RL" -eq 1 && -n "$GRID_BEST" ]]; then
    echo "ERROR: --grid-best cannot be used with --remove-rl"
    exit 1
fi

if [[ "$MANIFEST_BEST" -eq 1 && -n "$CHECKPOINT" ]]; then
    echo "ERROR: provide either checkpoint_path or --manifest-best, not both"
    exit 1
fi

if [[ -n "$GRID_BEST" && -n "$CHECKPOINT" ]]; then
    echo "ERROR: provide either checkpoint_path or --grid-best, not both"
    exit 1
fi

if [[ "$MANIFEST_BEST" -eq 1 && -n "$GRID_BEST" ]]; then
    echo "ERROR: provide either --manifest-best or --grid-best, not both"
    exit 1
fi

if [[ -n "$SYMBOLS_OPTION" && -n "$SYMBOLS" ]]; then
    echo "ERROR: provide symbols via positional arguments or --symbols, not both"
    exit 1
fi

if [[ -n "$SYMBOLS_OPTION" ]]; then
    SYMBOLS="$SYMBOLS_OPTION"
fi

SYMBOLS="$(normalize_symbols "$SYMBOLS")"
if [[ -n "$LEVERAGE_OVERRIDE" ]]; then
    raw_leverage_override="$LEVERAGE_OVERRIDE"
    if ! LEVERAGE_OVERRIDE="$(normalize_leverage "$raw_leverage_override")"; then
        echo "ERROR: invalid leverage override: $raw_leverage_override"
        exit 1
    fi
fi
if [[ -n "$WAIT_FOR_LIVE_CYCLE_SECONDS" ]]; then
    raw_wait_timeout="$WAIT_FOR_LIVE_CYCLE_SECONDS"
    if ! WAIT_FOR_LIVE_CYCLE_SECONDS="$(normalize_nonnegative_float "$raw_wait_timeout")"; then
        echo "ERROR: invalid wait timeout: $raw_wait_timeout"
        exit 1
    fi
fi
if [[ -n "$MIN_HEALTHY_LIVE_CYCLES" ]]; then
    raw_min_healthy_cycles="$MIN_HEALTHY_LIVE_CYCLES"
    if ! MIN_HEALTHY_LIVE_CYCLES="$(normalize_nonnegative_int "$raw_min_healthy_cycles")"; then
        echo "ERROR: invalid minimum healthy live cycle count: $raw_min_healthy_cycles"
        exit 1
    fi
fi
if [[ -n "$MIN_HEALTHY_LIVE_CYCLES" && -z "$WAIT_FOR_LIVE_CYCLE_SECONDS" ]]; then
    echo "ERROR: --min-healthy-live-cycles requires --wait-for-live-cycle-seconds"
    exit 1
fi

if [[ -n "$GRID_BEST" ]]; then
    if [[ -n "$MANIFEST_PATH" ]]; then
        echo "ERROR: --grid-best cannot be combined with --manifest-path"
        exit 1
    fi
    if [[ -n "$SYMBOLS" ]]; then
        echo "ERROR: --grid-best cannot be combined with symbol overrides"
        exit 1
    fi
    if [[ -n "$LEVERAGE_OVERRIDE" ]]; then
        echo "ERROR: --grid-best cannot be combined with --leverage"
        exit 1
    fi
fi

EXPECTED_SYMBOLS="${SYMBOLS:-$PREVIOUS_SYMBOLS}"
EXPECTED_SYMBOLS="$(normalize_symbols "$EXPECTED_SYMBOLS")"
EXPECTED_SYMBOLS_CSV="${EXPECTED_SYMBOLS// /,}"
EXPECTED_LEVERAGE="${LEVERAGE_OVERRIDE:-$PREVIOUS_LEVERAGE}"

if [[ "$REMOVE_RL" -eq 1 ]]; then
    if [[ -z "$PREVIOUS_CHECKPOINT" ]]; then
        echo "launch.sh already has no --rl-checkpoint configured"
        exit 0
    fi
    echo "Removing --rl-checkpoint from $LAUNCH"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] Would remove --rl-checkpoint line from launch.sh"
        [[ -n "$SYMBOLS" ]] && echo "[dry-run] Would update --symbols to: $SYMBOLS"
        [[ -n "$LEVERAGE_OVERRIDE" ]] && echo "[dry-run] Would update --leverage to: $LEVERAGE_OVERRIDE"
        [[ -n "$WAIT_FOR_LIVE_CYCLE_SECONDS" ]] && echo "[dry-run] Would wait up to $WAIT_FOR_LIVE_CYCLE_SECONDS seconds for a healthy live cycle snapshot"
        [[ -n "$MIN_HEALTHY_LIVE_CYCLES" ]] && echo "[dry-run] Would require $MIN_HEALTHY_LIVE_CYCLES healthy live cycle snapshots after deploy"
        exit 0
    fi
    backup="${LAUNCH}.bak.$(date +%s)"
    cp "$LAUNCH" "$backup"
    sed -i '/--rl-checkpoint/d' "$LAUNCH"
    apply_symbols_override "$SYMBOLS"
    apply_leverage_override "$LEVERAGE_OVERRIDE"
    echo "Removed --rl-checkpoint from launch.sh"
    echo ""
    echo "Restarting $SUPERVISOR_PROG..."
    DEPLOYED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    if ! restart_supervisor; then
        if ! rollback_launch "supervisor restart failed after removing --rl-checkpoint"; then
            echo ""
            echo "Rollback failed."
        fi
        exit 1
    fi
    sleep 5
    show_recent_logs
    echo ""
    echo "Verifying deploy..."
    if ! verify_launch_without_rl "$EXPECTED_SYMBOLS" "$EXPECTED_LEVERAGE" "$DEPLOYED_AT" "$WAIT_FOR_LIVE_CYCLE_SECONDS" "$MIN_HEALTHY_LIVE_CYCLES"; then
        if ! rollback_launch "post-remove-rl verification failed"; then
            echo ""
            echo "Rollback failed."
        fi
        exit 1
    fi
    echo "Done."
    exit 0
fi

if [[ "$MANIFEST_BEST" -eq 1 ]]; then
    echo "Resolving best production-evaluated checkpoint from manifest..."
    if ! CHECKPOINT="$(resolve_best_checkpoint_from_manifest "$MANIFEST_PATH" "$EXPECTED_SYMBOLS_CSV" "$EXPECTED_LEVERAGE")"; then
        echo "ERROR: unable to resolve best checkpoint from production manifest"
        exit 1
    fi
    echo "Resolved manifest-best checkpoint: $CHECKPOINT"
fi

if [[ -n "$GRID_BEST" ]]; then
    echo "Resolving best gated checkpoint/config from grid search..."
    if ! mapfile -t _grid_info < <(resolve_best_grid_target "$GRID_BEST"); then
        echo "ERROR: unable to resolve best deploy target from config grid"
        exit 1
    fi
    CHECKPOINT="${_grid_info[0]:-}"
    SYMBOLS="$(normalize_symbols "${_grid_info[1]:-}")"
    LEVERAGE_OVERRIDE="${_grid_info[2]:-}"
    MANIFEST_PATH="${_grid_info[3]:-}"
    EXPECTED_SYMBOLS="$SYMBOLS"
    EXPECTED_SYMBOLS_CSV="${EXPECTED_SYMBOLS// /,}"
    EXPECTED_LEVERAGE="$LEVERAGE_OVERRIDE"
    echo "Resolved grid-best checkpoint: $CHECKPOINT"
    echo "Resolved grid-best symbols: $SYMBOLS"
    echo "Resolved grid-best leverage: $LEVERAGE_OVERRIDE"
    echo "Resolved grid-best manifest: $MANIFEST_PATH"
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint_path required (or use --remove-rl)"
    echo "Run with --help for usage."
    exit 1
fi

if [[ "$CHECKPOINT" != /* ]]; then
    CHECKPOINT="/home/lee/code/stock/$CHECKPOINT"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"

echo ""
echo "Checking deploy gate..."
GATE_ARGS=( -m src.binance_deploy_gate --launch-script "$LAUNCH" --candidate-checkpoint "$CHECKPOINT" )
if [[ -n "$MANIFEST_PATH" ]]; then
    GATE_ARGS+=( --manifest-path "$MANIFEST_PATH" )
fi
if [[ -n "$EXPECTED_SYMBOLS_CSV" ]]; then
    GATE_ARGS+=( --symbols-override "$EXPECTED_SYMBOLS_CSV" )
fi
if [[ -n "$EXPECTED_LEVERAGE" ]]; then
    GATE_ARGS+=( --leverage-override "$EXPECTED_LEVERAGE" )
fi
if ! run_python "${GATE_ARGS[@]}"; then
    if [[ "$FORCE" -ne 1 ]]; then
        echo ""
        echo "Deployment blocked by production-eval gate."
        echo "Run evaluate_binance_hybrid_prod.py with this candidate first, or re-run with --force."
        exit 1
    fi
    echo "WARNING: overriding production-eval gate with --force"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    if grep -q '\-\-rl-checkpoint' "$LAUNCH"; then
        echo "[dry-run] Would update existing --rl-checkpoint to: $CHECKPOINT"
    else
        echo "[dry-run] Would add --rl-checkpoint $CHECKPOINT to launch.sh"
    fi
    [[ -n "$SYMBOLS" ]] && echo "[dry-run] Would update --symbols to: $SYMBOLS"
    [[ -n "$LEVERAGE_OVERRIDE" ]] && echo "[dry-run] Would update --leverage to: $LEVERAGE_OVERRIDE"
    [[ -n "$WAIT_FOR_LIVE_CYCLE_SECONDS" ]] && echo "[dry-run] Would wait up to $WAIT_FOR_LIVE_CYCLE_SECONDS seconds for a healthy live cycle snapshot"
    [[ -n "$MIN_HEALTHY_LIVE_CYCLES" ]] && echo "[dry-run] Would require $MIN_HEALTHY_LIVE_CYCLES healthy live cycle snapshots after deploy"
    echo "[dry-run] No changes applied."
    exit 0
fi

backup="${LAUNCH}.bak.$(date +%s)"
cp "$LAUNCH" "$backup"
echo "Backup: $backup"

if grep -q '\-\-rl-checkpoint' "$LAUNCH"; then
    sed -i "s|--rl-checkpoint [^ \\\\]*|--rl-checkpoint $CHECKPOINT|" "$LAUNCH"
    echo "Updated --rl-checkpoint in launch.sh"
else
    sed -i "/--fallback-mode chronos2/a\\  --rl-checkpoint $CHECKPOINT \\\\" "$LAUNCH"
    echo "Added --rl-checkpoint to launch.sh"
fi

apply_symbols_override "$SYMBOLS"
apply_leverage_override "$LEVERAGE_OVERRIDE"

echo ""
echo "--- Updated launch.sh ---"
cat "$LAUNCH"
echo "--- end ---"

echo ""
echo "Restarting $SUPERVISOR_PROG..."
DEPLOYED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
if ! restart_supervisor; then
    if ! rollback_launch "supervisor restart failed after updating launch.sh"; then
        echo ""
        echo "Rollback failed."
    fi
    exit 1
fi
sleep 5
show_recent_logs

echo ""
echo "Verifying deploy..."
if ! verify_launch_state "$CHECKPOINT" "$EXPECTED_SYMBOLS" "$EXPECTED_LEVERAGE" "$DEPLOYED_AT" "$WAIT_FOR_LIVE_CYCLE_SECONDS" "$MIN_HEALTHY_LIVE_CYCLES"; then
    if ! rollback_launch "post-deploy verification failed"; then
        echo ""
        echo "Rollback failed."
    fi
    exit 1
fi

echo ""
echo "Deployment complete."
