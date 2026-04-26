#!/bin/bash
# Safely redeploy the single live-writer Alpaca bot.
#
# Usage:
#   scripts/deploy_live_trader.sh [--allow-dirty] [--allow-unmodeled-live-sidecars] \
#     [--allow-invalid-xgb-ensemble] <unit_name>
#
# Valid <unit_name>:
#   xgb-daily-trader-live    (current champion — 5-seed XGB daily)
#   trading-server           (broker-boundary server; other bots delegate here)
#   daily-rl-trader          (legacy RL ensemble trader)
#   none                     (special: stop ALL live writers)
#
# Why this script exists
# ----------------------
# HARD RULE #2 in CLAUDE.md: exactly ONE process is allowed to hold the
# Alpaca live-writer fcntl lock at
# ``strategy_state/account_locks/alpaca_live_writer.lock``. Two
# writers will corrupt state and the second one exits(42). Operators
# have to remember which supervisor units can claim the lock and to
# stop the others before starting a new one. This script makes that a
# single command, and it verifies the result.
#
# What the script does
# --------------------
# 1. Validates the requested unit is in the LIVE_WRITER_UNITS registry.
# 2. Runs scripts/alpaca_deploy_preflight.py for the requested unit.
# 3. Stops every OTHER registered live-writer supervisor unit that is
#    currently RUNNING (so the lock is free).
# 4. Waits for the lock file to disappear or lose its holder, then
#    starts the requested unit.
# 5. Polls /health (trading-server) AND the lock file and confirms the
#    holder PID matches the newly started process (or its child).
# 6. Appends a line to deployments/live_trader_history.log with
#    TIMESTAMP, operator, chosen unit, pid — an audit trail of who
#    was live at any given moment.
#
# Exit codes:
#   0  redeploy succeeded (or unit already running correctly)
#   2  usage / invalid unit
#   3  stop failed for a conflicting unit
#   4  start failed for the requested unit
#   5  lock-holder mismatch (started but something else owns the lock)
#   6  preflight failed
#   7  post-restart preflight still reports unsafe or restart-needed state

set -euo pipefail

REPO="${REPO:-/nvme0n1-disk/code/stock-prediction}"
LOCK_PATH="${LOCK_PATH:-${REPO}/strategy_state/account_locks/alpaca_live_writer.lock}"
HISTORY_LOG="${HISTORY_LOG:-${REPO}/deployments/live_trader_history.log}"
PREFLIGHT="${PREFLIGHT:-${REPO}/scripts/alpaca_deploy_preflight.py}"
PREFLIGHT_REPORT_DIR="${PREFLIGHT_REPORT_DIR:-${REPO}/deployments/preflight_reports}"
PREFLIGHT_REPORT_PATH=""
POST_PREFLIGHT_REPORT_PATH=""

# Registry of live-capable units. If a new bot is introduced that
# imports alpaca_wrapper in LIVE mode, it MUST be added here.
LIVE_WRITER_UNITS=(
  "xgb-daily-trader-live"
  "trading-server"
  "daily-rl-trader"
)

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

_append_history() {
  local status="$1"
  local supervisor_pid="${2:-}"
  local lock_pid="${3:-}"
  mkdir -p "$(dirname "$HISTORY_LOG")"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)  operator=${USER:-$(id -un)}  unit=$TARGET  status=$status  supervisor_pid=${supervisor_pid:-none}  lock_pid=${lock_pid:-none}  preflight_report=${PREFLIGHT_REPORT_PATH:-none}  post_preflight_report=${POST_PREFLIGHT_REPORT_PATH:-none}" \
    >> "$HISTORY_LOG"
}

_show_report_file() {
  local path="$1"
  [ -s "$path" ] && sed 's/^/  preflight> /' "$path"
  [ -s "${path}.stderr" ] && sed 's/^/  preflight> /' "${path}.stderr"
  [ -s "${path}.stderr" ] || rm -f "${path}.stderr"
}

_post_preflight_has_unresolved_findings() {
  local path="$1"
  local expected_service="$2"
  python3 - "$path" "$expected_service" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_service = sys.argv[2]
reports = payload.get("reports")
if not isinstance(reports, list) or not reports:
    print("preflight report missing non-empty reports list")
    raise SystemExit(1)
bad = []
seen_expected = False
for report in reports:
    if not isinstance(report, dict):
        bad.append(f"malformed report entry: {type(report).__name__}")
        continue
    service = report.get("service", "unknown")
    if service != expected_service:
        bad.append(f"unexpected service in preflight report: {service!r}")
        continue
    seen_expected = True
    blockers = report.get("apply_blockers") or []
    reasons = report.get("restart_reasons") or []
    safe_to_apply = report.get("safe_to_apply")
    if safe_to_apply is not True:
        bad.append(f"{service}: safe_to_apply={safe_to_apply!r}")
    if blockers or reasons:
        bad.append(
            f"{service}: apply_blockers={blockers or 'none'} "
            f"restart_reasons={reasons or 'none'}"
        )
if not seen_expected:
    bad.append(f"missing expected service in preflight report: {expected_service}")
if bad:
    print("\n".join(bad))
    raise SystemExit(1)
PY
}

_unit_in_registry() {
  local target="$1"
  for unit in "${LIVE_WRITER_UNITS[@]}"; do
    [ "$unit" = "$target" ] && return 0
  done
  return 1
}

_unit_state() {
  # Returns RUNNING / STOPPED / STARTING / FATAL / UNKNOWN
  # NOTE: supervisorctl exits non-zero for non-RUNNING units. Under
  # ``set -o pipefail`` we have to capture first, then awk — otherwise
  # the pipe exit status triggers the ``|| echo UNKNOWN`` fallback
  # *after* awk has already printed the real state.
  local unit="$1"
  local out
  out=$(sudo -n supervisorctl status "$unit" 2>/dev/null) || true
  if [ -z "$out" ]; then
    echo "UNKNOWN"
    return
  fi
  printf '%s\n' "$out" | awk 'NR==1 {print $2; exit}'
}

_unit_pid() {
  local unit="$1"
  local out
  out=$(sudo -n supervisorctl status "$unit" 2>/dev/null) || true
  [ -z "$out" ] && return 0
  printf '%s\n' "$out" | grep -oE 'pid [0-9]+' | awk 'NR==1 {print $2; exit}' || true
}

_lock_pid() {
  [ -f "$LOCK_PATH" ] || { echo ""; return 0; }
  python3 -c 'import json, sys; d=json.load(open("'"$LOCK_PATH"'")); print(d.get("pid",""))' 2>/dev/null || echo ""
}

_pid_is_alive() {
  local pid="$1"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

_wait_until() {
  # _wait_until <seconds> <command...>
  local deadline=$(( $(date +%s) + $1 ))
  shift
  until "$@"; do
    [ "$(date +%s)" -ge "$deadline" ] && return 1
    sleep 1
  done
  return 0
}

_usage() {
  cat >&2 <<USAGE
Usage: $0 [--allow-dirty] [--allow-unmodeled-live-sidecars] \
  [--allow-invalid-xgb-ensemble] <unit_name>

Valid units (one MUST win the Alpaca live-writer lock at a time):
USAGE
  for unit in "${LIVE_WRITER_UNITS[@]}"; do
    printf '  %s\n' "$unit" >&2
  done
  echo "  none                     (stop all live writers)" >&2
  cat >&2 <<USAGE

Options:
  --allow-dirty                    Pass through to alpaca_deploy_preflight.py.
  --allow-unmodeled-live-sidecars  Required for launch scripts that intentionally
                                   enable live sidecars not represented in the
                                   XGB sweep/evaluation artifact.
  --allow-invalid-xgb-ensemble     Break-glass only: allow restart even if the
                                   current XGB ensemble fails model-load
                                   validation.
USAGE
}

ALLOW_DIRTY=0
ALLOW_UNMODELED_LIVE_SIDECARS=0
ALLOW_INVALID_XGB_ENSEMBLE=0

while [ "${1:-}" != "" ]; do
  case "$1" in
    -h|--help)
      _usage
      exit 2
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --allow-unmodeled-live-sidecars)
      ALLOW_UNMODELED_LIVE_SIDECARS=1
      shift
      ;;
    --allow-invalid-xgb-ensemble)
      ALLOW_INVALID_XGB_ENSEMBLE=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      log "FAIL: unknown option '$1'"
      _usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [ "${1:-}" = "" ]; then
  _usage
  exit 2
fi

TARGET="$1"
shift

if [ "${1:-}" != "" ]; then
  log "FAIL: unexpected extra argument '$1'"
  _usage
  exit 2
fi

if [ "$TARGET" != "none" ] && ! _unit_in_registry "$TARGET"; then
  log "FAIL: '$TARGET' is not in LIVE_WRITER_UNITS registry"
  _usage
  exit 2
fi

# ---------------------------------------------------------------- pre-flight
log "deploy_live_trader: requested=$TARGET operator=${USER:-$(id -un)}"
log "registry: ${LIVE_WRITER_UNITS[*]}"

if [ "$TARGET" != "none" ]; then
  mkdir -p "$PREFLIGHT_REPORT_DIR"
  safe_target="${TARGET//[^A-Za-z0-9_.-]/_}"
  PREFLIGHT_REPORT_PATH="${PREFLIGHT_REPORT_DIR}/$(date -u +%Y%m%dT%H%M%SZ)_${safe_target}_$$.json"

  preflight_args=(python3 "$PREFLIGHT" --service "$TARGET" --fail-on-unsafe --json)
  if [ "$ALLOW_DIRTY" = "1" ]; then
    preflight_args+=(--allow-dirty)
  fi
  if [ "$ALLOW_UNMODELED_LIVE_SIDECARS" = "1" ]; then
    preflight_args+=(--allow-unmodeled-live-sidecars)
  fi
  if [ "$ALLOW_INVALID_XGB_ENSEMBLE" = "1" ]; then
    preflight_args+=(--allow-invalid-xgb-ensemble)
  fi

  log "running preflight: ${preflight_args[*]}"
  if ! "${preflight_args[@]}" > "$PREFLIGHT_REPORT_PATH" 2> "${PREFLIGHT_REPORT_PATH}.stderr"; then
    _show_report_file "$PREFLIGHT_REPORT_PATH"
    log "preflight report: $PREFLIGHT_REPORT_PATH"
    log "FAIL: alpaca deploy preflight rejected $TARGET"
    _append_history "preflight_failed" "$(_unit_pid "$TARGET")" "$(_lock_pid)"
    exit 6
  fi
  _show_report_file "$PREFLIGHT_REPORT_PATH"
  log "preflight report: $PREFLIGHT_REPORT_PATH"
fi

# Current state
for unit in "${LIVE_WRITER_UNITS[@]}"; do
  state=$(_unit_state "$unit")
  pid=$(_unit_pid "$unit")
  log "  pre: $unit  state=$state  pid=${pid:-none}"
done
lock_pid_before=$(_lock_pid)
log "  pre: lock holder pid=${lock_pid_before:-none}  alive=$(_pid_is_alive "$lock_pid_before" && echo yes || echo no)"

# --------------------------------------------------------------- stop others
for unit in "${LIVE_WRITER_UNITS[@]}"; do
  if [ "$unit" = "$TARGET" ]; then
    continue
  fi
  state=$(_unit_state "$unit")
  if [ "$state" = "RUNNING" ] || [ "$state" = "STARTING" ]; then
    log "STOP $unit (state=$state)"
    if ! sudo -n supervisorctl stop "$unit" 2>&1 | sed 's/^/  supervisorctl> /'; then
      log "FAIL: could not stop $unit"
      _append_history "stop_failed:$unit" "$(_unit_pid "$TARGET")" "$(_lock_pid)"
      exit 3
    fi
  fi
done

# Wait for the lock to be released if it was held by a stopped unit.
# We only wait if the holder PID is no longer alive (clean exit) OR
# if the lock file is gone.
if [ -n "$lock_pid_before" ]; then
  log "waiting up to 15s for lock to clear (holder pid=$lock_pid_before)..."
  if ! _wait_until 15 bash -c '
    pid="'"$lock_pid_before"'"
    [ ! -f "'"$LOCK_PATH"'" ] && exit 0
    # Lock file still there — check if holder died.
    if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi
    exit 1
  '; then
    log "WARN: lock still held by pid=$lock_pid_before after 15s"
    log "      (it may self-release on restart; proceeding)"
  fi
fi

# ---------------------------------------------------------------- none-path
if [ "$TARGET" = "none" ]; then
  log "TARGET=none — all live writers stopped."
  _append_history "stopped_all" "-" "$(_lock_pid)"
  exit 0
fi

# ---------------------------------------------------------------- start target
state=$(_unit_state "$TARGET")
if [ "$state" = "RUNNING" ]; then
  log "$TARGET already RUNNING (pid=$(_unit_pid "$TARGET")) — restart to pick up new config"
  if ! sudo -n supervisorctl restart "$TARGET" 2>&1 | sed 's/^/  supervisorctl> /'; then
    log "FAIL: could not restart $TARGET"
    _append_history "restart_failed" "$(_unit_pid "$TARGET")" "$(_lock_pid)"
    exit 4
  fi
else
  log "START $TARGET (prev state=$state)"
  if ! sudo -n supervisorctl start "$TARGET" 2>&1 | sed 's/^/  supervisorctl> /'; then
    log "FAIL: could not start $TARGET"
    _append_history "start_failed" "$(_unit_pid "$TARGET")" "$(_lock_pid)"
    exit 4
  fi
fi

# ---------------------------------------------------------------- verify
log "waiting up to 30s for $TARGET to claim the Alpaca live-writer lock..."
if ! _wait_until 30 bash -c '
  sup_pid=$(sudo -n supervisorctl status "'"$TARGET"'" 2>/dev/null | grep -oE "pid [0-9]+" | awk "{print \$2}")
  [ -z "$sup_pid" ] && exit 1
  [ ! -f "'"$LOCK_PATH"'" ] && exit 1
  lp=$(python3 -c "import json,sys; print(json.load(open(\"'"$LOCK_PATH"'\")).get(\"pid\",\"\"))" 2>/dev/null)
  # Lock may be held by the supervisor pid OR by a child (python process).
  # Accept either the supervisor-reported pid or any descendant.
  [ "$lp" = "$sup_pid" ] && exit 0
  # descendant check
  for pid in $(pgrep -P "$sup_pid" 2>/dev/null); do
    [ "$lp" = "$pid" ] && exit 0
  done
  exit 1
'; then
  log "FAIL: $TARGET did not claim the Alpaca live-writer lock within 30s"
  log "      lock_holder_now=$(_lock_pid)  supervisor_pid=$(_unit_pid "$TARGET")"
  _append_history "lock_mismatch" "$(_unit_pid "$TARGET")" "$(_lock_pid)"
  exit 5
fi

# ---------------------------------------------------------------- post-report
final_pid=$(_unit_pid "$TARGET")
final_lock_pid=$(_lock_pid)
log "OK: $TARGET RUNNING  supervisor_pid=$final_pid  lock_holder_pid=$final_lock_pid"

POST_PREFLIGHT_REPORT_PATH="${PREFLIGHT_REPORT_PATH%.json}_post.json"
log "running post-restart preflight: ${preflight_args[*]}"
if ! "${preflight_args[@]}" > "$POST_PREFLIGHT_REPORT_PATH" 2> "${POST_PREFLIGHT_REPORT_PATH}.stderr"; then
  _show_report_file "$POST_PREFLIGHT_REPORT_PATH"
  log "post-restart preflight report: $POST_PREFLIGHT_REPORT_PATH"
  log "FAIL: post-restart preflight rejected $TARGET"
  _append_history "post_preflight_failed" "$final_pid" "$final_lock_pid"
  exit 7
fi
_show_report_file "$POST_PREFLIGHT_REPORT_PATH"
log "post-restart preflight report: $POST_PREFLIGHT_REPORT_PATH"
if ! unresolved="$(_post_preflight_has_unresolved_findings "$POST_PREFLIGHT_REPORT_PATH" "$TARGET")"; then
  printf '%s\n' "$unresolved" | sed 's/^/  preflight> /'
  log "FAIL: post-restart preflight still reports unresolved findings"
  _append_history "post_preflight_unresolved" "$final_pid" "$final_lock_pid"
  exit 7
fi

for unit in "${LIVE_WRITER_UNITS[@]}"; do
  if [ "$unit" = "$TARGET" ]; then continue; fi
  log "  $unit state=$(_unit_state "$unit")"
done

_append_history "ok" "$final_pid" "$final_lock_pid"

exit 0
