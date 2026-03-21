# Production Trade Listener Audit

**Date:** 2026-03-21
**Account status:** $46,460 (was $55,935)
**Files audited:** `trade_execution_listener.py`, `src/trade_execution_monitor.py`, `scripts/trade_execution_listener.py`

---

## Issues Found

### Critical

#### 1. No staleness check on incoming signals (FIXED)
**File:** `trade_execution_listener.py`
**Description:** There was no check on whether a trade event's timestamp was recent. If the listener restarted after a crash, or if an event was replayed from a file or stdin buffer, stale signals from hours ago could be processed as live trades.
**Fix:** Added `_is_stale()` check — any signal older than 5 minutes is rejected and logged. Configurable via `--signal-max-age`.

#### 2. No error handling in Alpaca stream callback (FIXED)
**File:** `trade_execution_listener.py` (the `@stream.on_trade_updates` callback)
**Description:** Any exception raised inside the async trade update callback would propagate up and crash the stream connection silently. The stream used `stream._run_forever()` (a private method) with no reconnection logic, meaning a single bad message or momentary exception would kill the entire listener process.
**Fix:** Wrapped the callback body in try/except with structured logging. Added exponential-backoff reconnect loop (up to 10 attempts, max 120s delay).

#### 3. No reconnection on stream disconnect (FIXED)
**File:** `trade_execution_listener.py`
**Description:** A single network error or Alpaca WebSocket disconnect would exit the listener permanently. On the production machine this would silently stop all trade recording until someone manually restarted it.
**Fix:** Added retry loop with exponential backoff (base 2s, doubles per attempt, capped at 120s, max 10 retries).

#### 4. NYT exclusion — not enforced in listener (FIXED)
**File:** `trade_execution_listener.py`
**Description:** The listener had no symbol exclusion list. NYT is classified SHORT by the model but rallied +51.6% — it must never be traded. While `src/trade_directions.py` marks NYT as `short_only`, the execution listener itself was not enforcing any exclusion.
**Fix:** Added `EXCLUDED_SYMBOLS = frozenset({"NYT"})` at module level. All event paths (stdin, file, alpaca stream) check this before processing. `--check-config` validates NYT is present.

#### 5. Silent exception swallow in state persistence (FIXED)
**File:** `src/trade_execution_monitor.py`, `TradeHistoryWriter._ensure_loaded()`
**Description:** The `except Exception: pass` on state file loading silently discarded any load failure. If the state file was corrupted, locked, or missing, the monitor would start with empty state and silently lose all historical PnL tracking context — without any log message.
**Fix:** Changed to `except Exception as exc: _log.warning(...)` with the file path and error message.

### Moderate

#### 6. No JSON logging of any kind (FIXED)
**File:** `trade_execution_listener.py`
**Description:** The entire listener had zero log output — no signal received, no order processed, no error. It was impossible to diagnose why trades were not being recorded by reviewing logs.
**Fix:** Added structured JSON logging to every key path: `signal_received`, `signal_rejected_stale`, `signal_excluded`, `dry_run_would_process`, `position_closed`, `process_event_failed`, `json_parse_failed`, `alpaca_stream_connecting`, `alpaca_stream_error`, `heartbeat`.

#### 7. No dry-run mode (FIXED)
**File:** `trade_execution_listener.py`
**Description:** No way to test the listener's signal path without actually modifying state. Added `--dry-run` flag that logs everything but writes nothing.

#### 8. stdin handler crashes on malformed JSON line (FIXED)
**File:** `trade_execution_listener.py`, `_run_stdin()`
**Description:** `json.loads(line)` in the stdin loop had no try/except — a single malformed line would crash the entire stdin loop, stopping all subsequent event processing.
**Fix:** Added per-line try/except with `json_parse_failed` log event; the loop continues after bad lines.

#### 9. No config validation flag (FIXED)
**File:** `trade_execution_listener.py`
**Description:** No way to quickly check that API keys are present and not placeholder defaults before starting.
**Fix:** Added `--check-config` flag. Validates API keys are non-empty and non-placeholder, and that `EXCLUDED_SYMBOLS` contains NYT. Exits 0 on success, 1 on failure.

#### 10. No heartbeat logging (FIXED)
**File:** `trade_execution_listener.py`
**Description:** No periodic liveness signal, making it impossible for a supervisor to confirm the listener was still running.
**Fix:** Added `--heartbeat-interval` (default 60s). Emits `{"event": "heartbeat", ...}` log line at that cadence.

---

## Fixes Applied

| # | File | Change |
|---|------|--------|
| 1 | `trade_execution_listener.py` | Added `_is_stale()` + `SIGNAL_MAX_AGE_SECONDS = 300` — all signals checked before processing |
| 2 | `trade_execution_listener.py` | `@stream.on_trade_updates` callback wrapped in try/except |
| 3 | `trade_execution_listener.py` | Reconnect loop with exponential backoff in `_run_alpaca()` |
| 4 | `trade_execution_listener.py` | `EXCLUDED_SYMBOLS = frozenset({"NYT"})` enforced on all paths |
| 5 | `src/trade_execution_monitor.py` | `except Exception: pass` → `except Exception as exc: _log.warning(...)` |
| 6 | `trade_execution_listener.py` | Full structured JSON logging via `_emit()` helper |
| 7 | `trade_execution_listener.py` | `--dry-run` flag |
| 8 | `trade_execution_listener.py` | Per-line try/except in `_run_stdin()` |
| 9 | `trade_execution_listener.py` | `--check-config` flag with API key + exclusion validation |
| 10 | `trade_execution_listener.py` | `--heartbeat-interval` with periodic `heartbeat` log events |

**Changes explicitly NOT made** (trading logic untouched):
- Position sizing formula
- Which signals to trade
- Hold duration config (hold=5 is in the model checkpoint, not this file)
- PnL calculation in `PositionLots.apply_trade()`

---

## What This Listener Does (and Doesn't Do)

This file (`trade_execution_listener.py`) is a **PnL state recorder**, not the order submission component. It:
- Listens for fill confirmations from Alpaca
- Tracks open lot FIFO positions
- Records realized PnL to a JSON state file

The actual order submission happens in `neural_trade_stock_live.py` via `spawn_open_position_at_maxdiff_takeprofit()` and related watcher processes. That file has its own logging via `loguru` and error handling.

---

## Remaining Risks

1. **Alpaca stream `_run_forever()` is a private API.** If `alpaca-trade-api` changes internals, this breaks silently. Consider migrating to the official `alpaca-py` SDK which has a stable streaming interface.

2. **No deduplication of fill events.** If the stream reconnects and Alpaca replays recent fills, the same fill could be processed twice, recording duplicate PnL entries. The staleness check helps but does not fully prevent this — a fill replayed within the 5-minute window would be double-counted.

3. **`hold=5` is a model training concern, not this file.** Confirm the production checkpoint was trained with `hold=5` and not a different value.

4. **NYT is in `DEFAULT_ALPACA_CORE_SHORT_STOCKS` in `src/trade_directions.py`.** This means the neural model may still classify NYT as SHORT and generate signals for it. The exclusion in this listener stops PnL recording for NYT fills but does NOT stop the order submission loop (`neural_trade_stock_live.py`) from dispatching NYT entries. A separate exclusion should be enforced upstream in the order submission path — confirm `--symbols` passed to the live trader does not include NYT.

5. **`env_real.py` contains placeholder API key fallbacks.** If the environment variables are not set at startup, the listener falls back to hardcoded placeholder keys and silently proceeds. `--check-config` now catches this.

6. **No maximum fill size check.** An unexpectedly large fill (e.g., from a partial fill followed by a full fill) would be recorded as-is. Low risk but worth monitoring.

---

## Recommended Monitoring

```bash
# Validate config before starting (run on production machine):
python trade_execution_listener.py --check-config

# Dry-run against a replay file to confirm signal flow:
python trade_execution_listener.py --mode file --events-file /path/to/fills.jsonl --dry-run

# Run with heartbeat every 60s and pipe JSON logs to a structured log file:
python trade_execution_listener.py --mode alpaca \
  --heartbeat-interval 60 \
  2>&1 | tee -a /var/log/trade_listener.jsonl

# Parse heartbeats from log to confirm liveness:
grep '"event": "heartbeat"' /var/log/trade_listener.jsonl | tail -5

# Watch for rejected signals:
grep '"event": "signal_rejected_stale"\|signal_excluded\|process_event_failed' /var/log/trade_listener.jsonl
```

### Supervisor recommendation
Run under `systemd` or `supervisord` with `RestartSec=5` so a crash restarts within seconds. The reconnect loop handles WebSocket drops internally, but a process-level crash still needs an external supervisor.
