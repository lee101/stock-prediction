from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import socket
import subprocess
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
ARTIFACT_GLOBS = (
    "analysis/bitbankgo_stock_bridge/*.json",
    "analysis/xgbnew_daily/**/*.json",
    "analysis/xgbnew_ensemble/**/*.json",
    "analysis/xgbnew_multiwindow/**/*.json",
    "analysis/gpu*/*.json",
)
LIVE_LOG_DIR = REPO / "analysis/xgb_live_trade_log"
STATE_DIR = REPO / "strategy_state"
MAX_BARS = 1500
MAX_TRADES = 5000
MAX_REPLAY_TRADES = 10000
MAX_REPLAY_SYMBOLS = 80


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def safe_repo_path(raw: str) -> Path | None:
    if not raw:
        return None
    candidate = (REPO / raw).resolve()
    try:
        candidate.relative_to(REPO)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_tail(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def artifact_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    worst = payload.get("worst_slippage_cell") or {}
    if not isinstance(worst, dict):
        worst = {}
    summaries = payload.get("summaries") or []
    selected = payload.get("selected_configs") or []
    trades = payload.get("candidate_trades") or []
    split_rows = payload.get("split_rows") or {}
    stat = path.stat()
    return {
        "path": repo_rel(path),
        "name": path.stem,
        "strategy": payload.get("strategy") or path.parent.name,
        "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "passes_27pct_gate": bool(payload.get("passes_27pct_gate")),
        "worst_slippage": worst,
        "summary_count": len(summaries) if isinstance(summaries, list) else 0,
        "selected_count": len(selected) if isinstance(selected, list) else 0,
        "candidate_trade_count": len(trades) if isinstance(trades, list) else 0,
        "split_rows": split_rows if isinstance(split_rows, dict) else {},
    }


def list_artifacts() -> list[dict[str, Any]]:
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for pattern in ARTIFACT_GLOBS:
        for path in REPO.glob(pattern):
            if path in seen or path.name.startswith("."):
                continue
            seen.add(path)
            summary = artifact_summary(path)
            if summary is not None:
                rows.append(summary)
    rows.sort(key=lambda row: str(row.get("mtime", "")), reverse=True)
    return rows[:120]


def latest_live_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(LIVE_LOG_DIR.glob("*.jsonl"))[-7:]:
        events.extend(read_jsonl_tail(path, limit=120))
    events.sort(key=lambda row: str(row.get("ts", "")))
    return events[-300:]


def live_status() -> dict[str, Any]:
    lock_path = STATE_DIR / "account_locks/alpaca_live_writer.lock"
    lock = None
    if lock_path.exists():
        try:
            lock = read_json(lock_path)
        except Exception:
            lock = {"error": "could not read lock"}
    supervisor = []
    try:
        out = subprocess.run(
            ["sudo", "-n", "supervisorctl", "status", "xgb-daily-trader-live", "daily-rl-trader", "trading-server"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        supervisor = [line for line in out.stdout.splitlines() if line.strip()]
    except Exception as exc:
        supervisor = [f"supervisor status unavailable: {exc}"]
    events = latest_live_events()
    equity_points = [
        {"ts": e.get("ts"), "equity": e.get("equity_pre")}
        for e in events
        if e.get("event") == "session_start" and isinstance(e.get("equity_pre"), (int, float))
    ]
    tick_status = [e for e in events if e.get("event") == "tick_status"]
    latest_tick = tick_status[-1] if tick_status else None
    return {
        "generated_at": utc_now(),
        "hostname": socket.gethostname(),
        "lock": lock,
        "supervisor": supervisor,
        "equity_points": equity_points[-120:],
        "latest_tick": latest_tick,
        "recent_events": events[-80:],
    }


def load_bars(symbol: str) -> list[dict[str, Any]]:
    symbol = symbol.upper().strip()
    if not symbol:
        return []
    candidates = [
        REPO / "trainingdatahourly/stocks" / f"{symbol}.csv",
        REPO / "trainingdata/stocks" / f"{symbol}.csv",
        REPO / "trainingdata" / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = row.get("timestamp") or row.get("date") or row.get("datetime") or row.get("time")
            try:
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(row.get("open") or row.get("Open") or 0.0),
                        "high": float(row.get("high") or row.get("High") or 0.0),
                        "low": float(row.get("low") or row.get("Low") or 0.0),
                        "close": float(row.get("close") or row.get("Close") or 0.0),
                        "volume": float(row.get("volume") or row.get("Volume") or 0.0),
                    }
                )
            except (TypeError, ValueError):
                continue
    return rows[-MAX_BARS:]


def artifact_payload(raw_path: str) -> dict[str, Any]:
    path = safe_repo_path(raw_path)
    if path is None:
        return {"error": "artifact not found"}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {"error": "artifact is not an object"}
    return payload


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_trade(raw: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(raw.get("symbol", "")).upper().strip()
    entry_ts = raw.get("entry_ts")
    exit_ts = raw.get("exit_ts")
    if not symbol or not entry_ts or not exit_ts:
        return None
    side = parse_int(raw.get("side"), 0)
    if side == 0:
        return None
    entry_open = parse_float(raw.get("entry_open"))
    exit_close = parse_float(raw.get("exit_close"))
    return {
        "symbol": symbol,
        "side": 1 if side > 0 else -1,
        "horizon": parse_int(raw.get("horizon")),
        "threshold": parse_float(raw.get("threshold")),
        "signal_ts": raw.get("signal_ts"),
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "edge": parse_float(raw.get("edge")),
        "entry_open": entry_open,
        "exit_close": exit_close,
        "return_pct": ((exit_close - entry_open) / entry_open) * (1 if side > 0 else -1)
        if entry_open
        else 0.0,
    }


def trade_return_after_costs(trade: dict[str, Any], args: dict[str, Any]) -> float:
    entry = parse_float(trade.get("entry_open"))
    exit_price = parse_float(trade.get("exit_close"))
    if entry <= 0.0 or exit_price <= 0.0:
        return 0.0
    side = parse_int(trade.get("side"), 1)
    raw_return = (exit_price - entry) / entry
    if side < 0:
        raw_return = -raw_return * parse_float(args.get("short_exposure_scale"), 1.0)
    fill_buffer = parse_float(args.get("fill_buffer_bps"), 0.0) / 10000.0
    slippage = parse_float(args.get("selection_slippage_bps"), parse_float(args.get("slippage_bps"), 0.0)) / 10000.0
    fee = parse_float(args.get("fee_rate"), 0.0)
    leverage = parse_float(args.get("leverage"), 1.0)
    return leverage * (raw_return - 2.0 * fee - 2.0 * fill_buffer - 2.0 * slippage)


def portfolio_payload(raw_artifact: str, max_symbols: int = MAX_REPLAY_SYMBOLS) -> dict[str, Any]:
    payload = artifact_payload(raw_artifact) if raw_artifact else {}
    if not isinstance(payload, dict) or payload.get("error"):
        return {"error": "artifact not found or unsupported", "artifact": raw_artifact, "symbols": [], "trades": []}
    raw_trades = payload.get("candidate_trades") or []
    if not isinstance(raw_trades, list):
        raw_trades = []
    trades = [trade for raw in raw_trades if isinstance(raw, dict) for trade in [normalize_trade(raw)] if trade]
    symbol_counts: dict[str, dict[str, Any]] = {}
    for trade in trades:
        row = symbol_counts.setdefault(
            str(trade["symbol"]),
            {"symbol": trade["symbol"], "trade_count": 0, "long_count": 0, "short_count": 0, "first_entry": None, "last_exit": None},
        )
        row["trade_count"] += 1
        row["long_count"] += int(parse_int(trade["side"]) > 0)
        row["short_count"] += int(parse_int(trade["side"]) < 0)
        entry_ts = str(trade["entry_ts"])
        exit_ts = str(trade["exit_ts"])
        row["first_entry"] = entry_ts if row["first_entry"] is None else min(str(row["first_entry"]), entry_ts)
        row["last_exit"] = exit_ts if row["last_exit"] is None else max(str(row["last_exit"]), exit_ts)
    symbols = sorted(
        symbol_counts.values(),
        key=lambda row: (-parse_int(row.get("trade_count")), str(row.get("symbol"))),
    )[: max(1, min(max_symbols, MAX_REPLAY_SYMBOLS))]
    allowed = {str(row["symbol"]) for row in symbols}
    trades = [trade for trade in trades if str(trade["symbol"]) in allowed]
    trades.sort(key=lambda trade: (str(trade["entry_ts"]), str(trade["symbol"]), parse_int(trade["side"])))
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    max_positions = max(1, parse_int(args.get("max_positions"), 1))
    equity = 1.0
    equity_points = [{"ts": trades[0]["entry_ts"], "equity": equity}] if trades else []
    for trade in sorted(trades, key=lambda row: (str(row["exit_ts"]), str(row["symbol"]))):
        equity *= 1.0 + trade_return_after_costs(trade, args) / max_positions
        equity_points.append({"ts": trade["exit_ts"], "equity": equity})
    times = sorted({str(t["entry_ts"]) for t in trades} | {str(t["exit_ts"]) for t in trades})
    return {
        "artifact": raw_artifact,
        "strategy": payload.get("strategy"),
        "symbols": symbols,
        "trades": trades[:MAX_REPLAY_TRADES],
        "times": times,
        "equity_points": equity_points[-MAX_REPLAY_TRADES:],
        "max_positions": max_positions,
        "leverage": parse_float(args.get("leverage"), 1.0),
        "selection_slippage_bps": parse_float(args.get("selection_slippage_bps"), 0.0),
    }


def symbol_payload(symbol: str, raw_artifact: str) -> dict[str, Any]:
    symbol = symbol.upper().strip()
    payload = artifact_payload(raw_artifact) if raw_artifact else {}
    trades = []
    if isinstance(payload, dict):
        raw_trades = payload.get("candidate_trades") or []
        if isinstance(raw_trades, list):
            for trade in raw_trades:
                if not isinstance(trade, dict):
                    continue
                if str(trade.get("symbol", "")).upper() == symbol:
                    trades.append(trade)
    return {
        "symbol": symbol,
        "bars": load_bars(symbol),
        "trades": trades[:MAX_TRADES],
        "artifact": raw_artifact,
    }


def overview_payload() -> dict[str, Any]:
    artifacts = list_artifacts()
    return {
        "generated_at": utc_now(),
        "artifacts": artifacts,
        "live": live_status(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "StockDashboard/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_static(self, rel: str) -> None:
        if rel in ("", "/"):
            rel = "index.html"
        rel = rel.lstrip("/")
        path = (STATIC_DIR / rel).resolve()
        try:
            path.relative_to(STATIC_DIR)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/api/overview":
            self.send_json(overview_payload())
        elif parsed.path == "/api/artifact":
            self.send_json(artifact_payload(qs.get("path", [""])[0]))
        elif parsed.path == "/api/symbol":
            self.send_json(symbol_payload(qs.get("symbol", [""])[0], qs.get("artifact", [""])[0]))
        elif parsed.path == "/api/portfolio":
            self.send_json(portfolio_payload(qs.get("artifact", [""])[0]))
        elif parsed.path == "/api/live":
            self.send_json(live_status())
        elif parsed.path == "/healthz":
            self.send_json({"ok": True, "generated_at": utc_now()})
        else:
            self.send_static(parsed.path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only stock experiment Plotly dashboard.")
    parser.add_argument("--host", default=os.environ.get("STOCK_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("STOCK_DASHBOARD_PORT", "8899")))
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"stock_dashboard listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
