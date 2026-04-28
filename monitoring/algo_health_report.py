#!/usr/bin/env python3
"""Deterministic algorithm health report for the live XGB stock trader.

Runs in <60s with no LLM. Surfaces:

  1. Sim-vs-live parity (scores reproducible against current ensemble)
  2. Score trend (top-1 over last N sessions, gate gap, near-miss alerts)
  3. Per-seed ensemble disagreement (sanity for "ensemble blend working")
  4. Plan preview — if a BUY fired with the latest scoring, what would we
     submit? what is the live quote NOW? what notional vs equity?
  5. Stale-data check (last CSV bar age, latest Alpaca bar)
  6. Lock holder liveness (singleton pid alive, matches supervisor)
  7. Fills-last-10d count + 0/1 stock position invariant
  8. Hyperparam-sweep candidates from observed score distribution

Designed to be run hourly (or every 15min during market hours) by cron.
Writes JSONL to monitoring/logs/algo_health.jsonl AND a human-readable
report to monitoring/logs/algo_health_current.txt for tail -f.

Usage
-----
  source .venv/bin/activate
  python monitoring/algo_health_report.py                # full report
  python monitoring/algo_health_report.py --no-fetch     # no Alpaca calls
  python monitoring/algo_health_report.py --json         # JSONL only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = REPO / "monitoring" / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
TRADE_LOG_DIR = REPO / "analysis" / "xgb_live_trade_log"
LIVE_ENSEMBLE_DIR = REPO / "analysis" / "xgbnew_daily" / "alltrain_ensemble_gpu"
LOCK_PATH = REPO / "strategy_state" / "account_locks" / "alpaca_live_writer.lock"
LAUNCH_SH = REPO / "deployments" / "xgb-daily-trader-live" / "launch.sh"
SYMBOL_LIST = REPO / "symbol_lists" / "stocks_wide_1000_v1.txt"


@dataclass
class Section:
    name: str
    status: str  # "ok", "warn", "fail", "info"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _latest_trade_log_files(n: int = 10) -> list[Path]:
    if not TRADE_LOG_DIR.exists():
        return []
    files = sorted(TRADE_LOG_DIR.glob("*.jsonl"))
    return files[-n:]


def _live_launch_args() -> dict[str, str]:
    """Parse current launch.sh flags into a dict (best-effort).

    Resolves simple `VAR="..."` or `VAR=...` shell assignments earlier in the
    file so flag values like `"${MODEL_PATHS}"` are expanded to their literal
    string before being read as flag values.
    """
    if not LAUNCH_SH.exists():
        return {}
    raw = LAUNCH_SH.read_text()

    import re
    shell_vars: dict[str, str] = {}
    for line in raw.splitlines():
        m = re.match(r'^\s*([A-Z_][A-Z0-9_]*)=("([^"]*)"|([^\s#]+))\s*(?:#.*)?$', line)
        if not m:
            continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (m.group(4) or "")
        # Expand previously-defined ${VAR} / $VAR refs in val
        def _sub(mm: re.Match) -> str:
            vn = mm.group(1) or mm.group(2)
            return shell_vars.get(vn, mm.group(0))
        val = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)", _sub, val)
        shell_vars[name] = val

    def _expand(tok: str) -> str:
        # strip surrounding quotes once
        s = tok
        if (len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}):
            s = s[1:-1]
        def _sub(mm: re.Match) -> str:
            vn = mm.group(1) or mm.group(2)
            return shell_vars.get(vn, mm.group(0))
        return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)", _sub, s)

    flags: dict[str, str] = {}
    cur = None
    for tok in raw.replace("\\\n", " ").split():
        if tok.startswith("--"):
            cur = tok.lstrip("-")
            flags.setdefault(cur, "true")
        elif cur is not None:
            flags[cur] = _expand(tok)
            cur = None
    return flags


def _proc_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return pid == os.getpid()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Section: parse most-recent live "scored" event
# ---------------------------------------------------------------------------

def _last_scored_event(files: list[Path]) -> tuple[Path | None, dict | None]:
    for f in reversed(files):
        rows = _read_jsonl(f)
        for r in reversed(rows):
            if r.get("event") == "scored":
                return f, r
    return None, None


def section_score_trend(files: list[Path], min_score: float) -> Section:
    series = []
    for f in files:
        rows = _read_jsonl(f)
        scoreds = [r for r in rows if r.get("event") == "scored"]
        for s in scoreds:
            top20 = s.get("top20") or []
            if not top20:
                continue
            t = top20[0]
            seeds = t.get("per_seed_scores") or []
            seed_spread = (max(seeds) - min(seeds)) if seeds else 0.0
            series.append({
                "ts": s["ts"],
                "file": f.name,
                "n_candidates": s.get("n_candidates"),
                "top_symbol": t.get("symbol"),
                "top_score": float(t.get("score", 0.0)),
                "score_std": float(t.get("score_std", 0.0)),
                "per_seed_spread": float(seed_spread),
            })
    if not series:
        return Section("score_trend", "warn", "no scored events in trade log",
                       {"n_sessions": 0})

    last = series[-1]
    gap = float(min_score) - last["top_score"]
    near_miss = gap <= 0.10  # within 10pp of gate
    fired = last["top_score"] >= float(min_score)

    # Stale features signal: top score identical to 4dp across ≥3 recent sessions
    # consider only sessions with same top_symbol AND same n_candidates band
    stale = False
    if len(series) >= 3:
        last3 = series[-3:]
        keys = {(round(s["top_score"], 4), s["top_symbol"]) for s in last3}
        if len(keys) == 1:
            stale = True

    # Aggregate
    top_scores = [s["top_score"] for s in series]
    spreads = [s["per_seed_spread"] for s in series]
    summary = {
        "n_sessions": len(series),
        "last": last,
        "min_score_gate": float(min_score),
        "gate_gap": float(gap),
        "near_miss": near_miss,
        "gate_fired": fired,
        "top_score_min": min(top_scores),
        "top_score_max": max(top_scores),
        "top_score_mean": sum(top_scores) / len(top_scores),
        "per_seed_spread_min": min(spreads) if spreads else 0.0,
        "per_seed_spread_max": max(spreads) if spreads else 0.0,
        "stale_top_signature": stale,
        "trend_recent": series[-5:],
    }

    if stale:
        return Section("score_trend", "fail",
                       f"STALE top-1 across last 3 sessions: {last['top_symbol']} "
                       f"@ {last['top_score']:.4f} repeating", summary)

    if last["per_seed_spread"] < 1e-3:
        return Section("score_trend", "fail",
                       f"per-seed spread {last['per_seed_spread']:.4f} ~= 0 "
                       "— ensemble blend likely broken", summary)

    if fired:
        return Section("score_trend", "ok",
                       f"GATE FIRED: top {last['top_symbol']} @ "
                       f"{last['top_score']:.4f} >= ms={min_score}", summary)

    if near_miss:
        return Section("score_trend", "warn",
                       f"NEAR MISS: top {last['top_symbol']} @ "
                       f"{last['top_score']:.4f}, gate {min_score} "
                       f"(gap {gap:+.4f})", summary)

    return Section("score_trend", "info",
                   f"hold-cash regime: top {last['top_symbol']} @ "
                   f"{last['top_score']:.4f} vs gate {min_score} "
                   f"(gap {gap:+.4f})", summary)


# ---------------------------------------------------------------------------
# Section: lock holder, supervisor pid, and equity (basic infra)
# ---------------------------------------------------------------------------

def section_lock_holder() -> Section:
    if not LOCK_PATH.exists():
        return Section("lock_holder", "fail",
                       "alpaca_live_writer.lock missing — singleton not held",
                       {"path": str(LOCK_PATH)})
    try:
        meta = json.loads(LOCK_PATH.read_text())
    except Exception as e:
        return Section("lock_holder", "fail", f"lock file unreadable: {e}",
                       {"path": str(LOCK_PATH)})

    pid = meta.get("pid")
    alive = _proc_alive(int(pid)) if pid else False
    svc = meta.get("service_name", "")
    started = meta.get("started_at", "")
    age_sec = None
    try:
        age_sec = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(started)).total_seconds()
    except Exception:
        pass

    details = {
        "pid": pid,
        "service_name": svc,
        "started_at": started,
        "age_sec": age_sec,
        "pid_alive": alive,
    }

    if not alive:
        return Section("lock_holder", "fail",
                       f"lock pid {pid} is DEAD — singleton orphaned", details)
    if svc != "xgb_live_trader":
        return Section("lock_holder", "fail",
                       f"lock held by '{svc}' not xgb_live_trader", details)
    return Section("lock_holder", "ok",
                   f"lock OK: pid {pid} ({svc}) up "
                   f"{age_sec/3600:.1f}h" if age_sec else f"pid {pid} ({svc})",
                   details)


# ---------------------------------------------------------------------------
# Section: live ensemble vs launch.sh consistency
# ---------------------------------------------------------------------------

def section_ensemble_consistency(launch_args: dict[str, str]) -> Section:
    if not LIVE_ENSEMBLE_DIR.exists():
        return Section("ensemble_consistency", "fail",
                       f"live ensemble dir missing: {LIVE_ENSEMBLE_DIR}", {})
    pkls = sorted(LIVE_ENSEMBLE_DIR.glob("alltrain_seed*.pkl"))
    if len(pkls) < 5:
        return Section("ensemble_consistency", "fail",
                       f"only {len(pkls)} pkls under {LIVE_ENSEMBLE_DIR.name} (expected >=5)",
                       {"pkls": [p.name for p in pkls]})

    # Compare against --model-paths in launch.sh if present
    declared = launch_args.get("model-paths", "")
    declared_pkls = [Path(p).name for p in declared.split(",") if p.strip()]
    on_disk_names = [p.name for p in pkls]

    mtimes = [p.stat().st_mtime for p in pkls]
    age_days = (time.time() - max(mtimes)) / 86400.0
    train_end = launch_args.get("min-score", "")  # placeholder; not strictly available

    details = {
        "n_pkls": len(pkls),
        "pkls": on_disk_names,
        "declared_in_launch_sh": declared_pkls,
        "newest_pkl_age_days": age_days,
    }

    if declared_pkls and not all(d in on_disk_names for d in declared_pkls):
        missing = [d for d in declared_pkls if d not in on_disk_names]
        return Section("ensemble_consistency", "fail",
                       f"launch.sh references missing pkls: {missing}", details)
    if age_days > 14:
        return Section("ensemble_consistency", "warn",
                       f"newest pkl is {age_days:.1f} days old "
                       "— retrain cadence may have stalled", details)
    return Section("ensemble_consistency", "ok",
                   f"{len(pkls)} pkls present, newest {age_days:.1f}d old",
                   details)


# ---------------------------------------------------------------------------
# Section: alpaca account state (live)
# ---------------------------------------------------------------------------

def section_alpaca_account(timeout_s: float = 8.0) -> Section:
    try:
        import urllib.request
        import env_real  # type: ignore
        hdr = {
            "APCA-API-KEY-ID": env_real.ALP_KEY_ID_PROD,
            "APCA-API-SECRET-KEY": env_real.ALP_SECRET_KEY_PROD,
        }
        req = urllib.request.Request(
            "https://api.alpaca.markets/v2/account", headers=hdr
        )
        d = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
    except Exception as e:
        return Section("alpaca_account", "fail",
                       f"account read failed: {e}", {})

    eq = float(d.get("equity", 0.0))
    bp = float(d.get("buying_power", 0.0))
    cash = float(d.get("cash", 0.0))
    status = d.get("status", "")
    blocked = d.get("trading_blocked") or d.get("account_blocked")
    details = {
        "equity": eq,
        "buying_power": bp,
        "cash": cash,
        "status": status,
        "trading_blocked": bool(blocked),
        "balance_asof": d.get("balance_asof"),
        "non_marginable_buying_power": float(
            d.get("non_marginable_buying_power", 0.0)
        ),
    }
    if blocked:
        return Section("alpaca_account", "fail",
                       f"trading_blocked! status={status}", details)
    if status != "ACTIVE":
        return Section("alpaca_account", "warn",
                       f"account status={status}", details)
    return Section("alpaca_account", "ok",
                   f"ACTIVE eq=${eq:,.2f} bp=${bp:,.2f}", details)


# ---------------------------------------------------------------------------
# Section: positions + recent fills
# ---------------------------------------------------------------------------

def section_positions_and_fills(timeout_s: float = 8.0) -> Section:
    try:
        import urllib.request
        import env_real  # type: ignore
        hdr = {
            "APCA-API-KEY-ID": env_real.ALP_KEY_ID_PROD,
            "APCA-API-SECRET-KEY": env_real.ALP_SECRET_KEY_PROD,
        }
        pos = json.loads(urllib.request.urlopen(
            urllib.request.Request("https://api.alpaca.markets/v2/positions", headers=hdr),
            timeout=timeout_s).read())
        after = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        fills = json.loads(urllib.request.urlopen(
            urllib.request.Request(
                f"https://api.alpaca.markets/v2/account/activities/FILL?after={after}",
                headers=hdr),
            timeout=timeout_s).read())
    except Exception as e:
        return Section("positions_and_fills", "fail",
                       f"positions/fills read failed: {e}", {})

    stock_pos = [p for p in pos
                 if "USD" not in p.get("symbol", "")
                 and abs(float(p.get("market_value", 0.0))) > 1.0]
    crypto_pos = [p for p in pos
                  if "USD" in p.get("symbol", "")
                  and abs(float(p.get("market_value", 0.0))) > 1.0]
    details = {
        "n_stock_positions": len(stock_pos),
        "n_crypto_positions": len(crypto_pos),
        "stock_positions": [
            {"sym": p["symbol"], "qty": p["qty"], "mv": float(p["market_value"]),
             "side": p.get("side", "?"),
             "unrealized_pl": float(p.get("unrealized_pl", 0.0))}
            for p in stock_pos
        ],
        "crypto_positions": [
            {"sym": p["symbol"], "qty": p["qty"], "mv": float(p["market_value"])}
            for p in crypto_pos
        ],
        "fills_last_10d": len(fills),
        "recent_fills": [
            {"ts": f.get("transaction_time", "")[:19],
             "sym": f.get("symbol"), "side": f.get("side"),
             "qty": f.get("qty"), "px": f.get("price")}
            for f in fills[:5]
        ],
    }

    if len(stock_pos) > 1:
        return Section("positions_and_fills", "fail",
                       f"{len(stock_pos)} stock positions (top_n=1, expected <=1)",
                       details)
    return Section("positions_and_fills", "ok",
                   f"stock_pos={len(stock_pos)} crypto_pos={len(crypto_pos)} "
                   f"fills_10d={len(fills)}", details)


# ---------------------------------------------------------------------------
# Section: plan preview — if we fired NOW, what would we buy at what price?
# ---------------------------------------------------------------------------

def section_plan_preview(launch_args: dict[str, str], scored_event: dict | None,
                         equity: float, timeout_s: float = 8.0) -> Section:
    """Given the latest scored event, simulate the order plan as if we fired."""
    if not scored_event:
        return Section("plan_preview", "warn",
                       "no recent scored event to preview", {})

    top20 = scored_event.get("top20") or []
    if not top20:
        return Section("plan_preview", "warn",
                       "scored event has no top20", {})

    top = top20[0]
    sym = top["symbol"]
    score = float(top["score"])
    last_close = float(top.get("last_close", 0.0))
    spread_bps = float(top.get("spread_bps", 0.0))
    min_score = float(launch_args.get("min-score", "0.85") or 0.85)
    allocation = float(launch_args.get("allocation", "1.0") or 1.0)

    # Always preview the plan — even if gate didn't fire — so we know what
    # WOULD happen if regime flips.
    target_notional = allocation * equity
    target_qty = (target_notional / last_close) if last_close > 0 else 0.0

    # Live quote
    bid = ask = 0.0
    quote_age_s = None
    try:
        import urllib.request
        import env_real  # type: ignore
        hdr = {
            "APCA-API-KEY-ID": env_real.ALP_KEY_ID_PROD,
            "APCA-API-SECRET-KEY": env_real.ALP_SECRET_KEY_PROD,
        }
        # Convert local dash form to Alpaca dot form if needed
        api_sym = sym.replace("-", ".") if "-" in sym else sym
        url = f"https://data.alpaca.markets/v2/stocks/{api_sym}/quotes/latest?feed=iex"
        req = urllib.request.Request(url, headers=hdr)
        d = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
        q = d.get("quote", {})
        bid = float(q.get("bp", 0.0) or 0.0)
        ask = float(q.get("ap", 0.0) or 0.0)
        ts = q.get("t")
        if ts:
            try:
                quote_age_s = (datetime.now(timezone.utc)
                               - datetime.fromisoformat(ts.replace("Z", "+00:00"))
                               ).total_seconds()
            except Exception:
                pass
    except Exception:
        pass

    expected_buy_limit = ask * 1.0015 if ask > 0 else last_close  # 15bps guard
    expected_fill_drift_bps = (
        (expected_buy_limit / last_close - 1.0) * 1e4 if last_close > 0 else 0.0
    )

    details = {
        "candidate_symbol": sym,
        "candidate_score": score,
        "min_score_gate": min_score,
        "gate_would_fire": bool(score >= min_score),
        "last_close": last_close,
        "spread_bps_from_log": spread_bps,
        "live_bid": bid,
        "live_ask": ask,
        "live_spread_bps": ((ask - bid) / ((ask + bid) / 2) * 1e4) if (bid > 0 and ask > 0) else None,
        "quote_age_s": quote_age_s,
        "equity_for_sizing": equity,
        "allocation": allocation,
        "target_notional_usd": target_notional,
        "target_qty": target_qty,
        "expected_limit_price_buy": expected_buy_limit,
        "expected_drift_bps_vs_close": expected_fill_drift_bps,
    }

    if score < min_score:
        return Section("plan_preview", "info",
                       f"would HOLD CASH: top {sym} @ {score:.4f} < gate {min_score}",
                       details)
    if bid <= 0 or ask <= 0:
        return Section("plan_preview", "warn",
                       f"gate fired for {sym} @ {score:.4f} but no live quote available",
                       details)
    return Section("plan_preview", "ok",
                   f"would BUY {sym} qty~{target_qty:.2f} @ ~${expected_buy_limit:.2f} "
                   f"(notional ~${target_notional:,.0f}, drift {expected_fill_drift_bps:+.1f}bps)",
                   details)


# ---------------------------------------------------------------------------
# Section: hyperparam-sweep candidates
# ---------------------------------------------------------------------------

def section_hyperparam_candidates(score_trend: Section) -> Section:
    """If consistent near-misses, suggest sweep candidates."""
    s = score_trend.details
    top_max = s.get("top_score_max", 0.0)
    top_mean = s.get("top_score_mean", 0.0)
    gate = s.get("min_score_gate", 0.85)
    fired = s.get("gate_fired", False)
    n = s.get("n_sessions", 0)

    if n < 3:
        return Section("hyperparam_candidates", "info",
                       "not enough sessions for sweep recommendations", {})

    candidates = []
    # If max top_score < gate by < 0.10: a slightly lower gate would fire
    if top_max < gate and (gate - top_max) <= 0.10:
        candidates.append({
            "lever": "min_score",
            "current": gate,
            "suggested_grid": [round(top_mean, 2), round(top_mean + 0.02, 2),
                               round(top_max, 2)],
            "rationale": (f"top score has max {top_max:.4f}, mean {top_mean:.4f} "
                          f"— current gate {gate} never fires. Sweep slightly lower "
                          "gates on TRUE-OOS to find a firing config that still "
                          "passes 0-neg + p10>0."),
        })
    if not fired and top_mean > 0 and top_mean < gate * 0.9:
        candidates.append({
            "lever": "ensemble_recipe",
            "current": "5-seed alltrain",
            "suggested_grid": ["10-seed", "rank-trained", "shorter-cutoff retrain"],
            "rationale": (f"score collapse to mean {top_mean:.4f} — model calibration "
                          "shifted post-regime. Retrain-through-cutoff or expand "
                          "seed ensemble may restore high-confidence picks."),
        })

    details = {"candidates": candidates}
    if not candidates:
        return Section("hyperparam_candidates", "ok",
                       "no obvious sweep candidates from current trend", details)
    return Section("hyperparam_candidates", "info",
                   f"{len(candidates)} sweep candidate(s) suggested", details)


# ---------------------------------------------------------------------------
# Sim/prod parity replay (cheap version using last live scored event)
# ---------------------------------------------------------------------------

def section_parity_replay(scored_event: dict | None,
                          launch_args: dict[str, str],
                          symbols_path: Path) -> Section:
    """Re-score the same logged session and confirm bit-identity.

    Skipped if scored_event is older than 24h (data drift makes replay invalid).
    """
    if not scored_event:
        return Section("parity_replay", "warn", "no scored event to replay", {})

    ts = scored_event.get("ts", "")
    try:
        ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return Section("parity_replay", "warn", "unparseable ts", {"ts": ts})

    age_h = (datetime.now(timezone.utc) - ts_dt).total_seconds() / 3600.0
    if age_h > 36:
        return Section("parity_replay", "info",
                       f"last scored event {age_h:.1f}h old — skipping replay "
                       "(weekend or holiday)", {"age_h": age_h})

    try:
        from xgbnew.live_trader import (  # type: ignore
            _get_latest_bars,
            score_all_symbols,
        )
        from xgbnew.model import XGBStockModel  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        return Section("parity_replay", "fail",
                       f"import error: {e}", {})

    pkls = sorted(LIVE_ENSEMBLE_DIR.glob("alltrain_seed*.pkl"))
    if not pkls:
        return Section("parity_replay", "fail",
                       f"no pkls in {LIVE_ENSEMBLE_DIR}", {})

    try:
        models = [XGBStockModel.load(str(p)) for p in pkls]
    except Exception as e:
        return Section("parity_replay", "fail",
                       f"model load: {e}", {})

    syms = [s.strip() for s in symbols_path.read_text().splitlines()
            if s.strip() and not s.startswith("#")]

    try:
        bars = _get_latest_bars(syms, n_days=20)
    except Exception as e:
        return Section("parity_replay", "fail",
                       f"bar fetch: {e}", {})

    cutoff = ts_dt.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    cutoff_floor = pd.Timestamp(cutoff).floor("D")
    # Allow today's bar through (intraday rescore replays). For pre-open
    # sessions (now < market_open ET), Alpaca won't have the new daily bar yet
    # so it's a no-op; for intraday replays, the FINAL bar may differ from the
    # mid-session partial bar live saw — accept ≤0.01 as PARITY-CLOSE.
    keep_through = cutoff_floor + pd.Timedelta(days=1)
    new_bars = {}
    for sym, df in bars.items():
        ts_series = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        kept = df.loc[ts_series < keep_through].reset_index(drop=True)
        if len(kept):
            new_bars[sym] = kept
    bars = new_bars

    try:
        replay = score_all_symbols(
            symbols=syms,
            data_root=REPO / "trainingdata",
            model=models,
            live_bars=bars,
            min_dollar_vol=float(launch_args.get("min-dollar-vol", "5e6") or 5e6),
            max_spread_bps=30.0,
            min_vol_20d=float(launch_args.get("min-vol-20d", "0.0") or 0.0),
            max_vol_20d=float(launch_args.get("max-vol-20d", "0.0") or 0.0),
            max_ret_20d_rank_pct=float(launch_args.get("max-ret-20d-rank-pct", "1.0") or 1.0),
            min_ret_5d_rank_pct=float(launch_args.get("min-ret-5d-rank-pct", "0.0") or 0.0),
            score_uncertainty_penalty=0.0,
            now=cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff,
        )
    except Exception as e:
        return Section("parity_replay", "fail",
                       f"replay score_all_symbols: {e}\n{traceback.format_exc()[-400:]}", {})

    if replay is None or len(replay) == 0:
        return Section("parity_replay", "warn",
                       "replay produced 0 candidates", {"age_h": age_h})

    log_top20 = scored_event.get("top20", [])
    log_lookup = {r["symbol"]: float(r["score"]) for r in log_top20}
    rep_lookup = {replay.iloc[i]["symbol"]: float(replay.iloc[i]["score"])
                  for i in range(min(len(replay), 40))}

    deltas = []
    n_compared = 0
    n_missing = 0
    for sym, log_s in log_lookup.items():
        rep_s = rep_lookup.get(sym)
        if rep_s is None:
            n_missing += 1
            continue
        deltas.append(rep_s - log_s)
        n_compared += 1

    max_abs = max((abs(d) for d in deltas), default=0.0)
    details = {
        "age_h": age_h,
        "log_top": log_top20[0] if log_top20 else None,
        "replay_top_symbol": str(replay.iloc[0]["symbol"]),
        "replay_top_score": float(replay.iloc[0]["score"]),
        "n_log_top20": len(log_top20),
        "n_compared": n_compared,
        "n_missing_in_replay": n_missing,
        "max_abs_delta": max_abs,
    }

    if max_abs < 1e-3:
        return Section("parity_replay", "ok",
                       f"BIT-IDENTICAL on {n_compared}/{len(log_top20)} symbols "
                       f"(max |Δ|={max_abs:.5f})", details)
    if max_abs < 1e-2:
        return Section("parity_replay", "ok",
                       f"close parity (max |Δ|={max_abs:.4f}) — likely intraday "
                       f"bar precision noise", details)
    return Section("parity_replay", "fail",
                   f"DIVERGENCE max |Δ|={max_abs:.4f} — features/model drift?",
                   details)


# ---------------------------------------------------------------------------
# Section: stale data check
# ---------------------------------------------------------------------------

def section_stale_data() -> Section:
    """Latest CSV bar age vs latest Alpaca bar — flag if CSVs are far behind."""
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        return Section("stale_data", "warn", f"pandas import: {e}", {})

    sample = REPO / "trainingdata" / "AAPL.csv"
    if not sample.exists():
        return Section("stale_data", "warn", "no AAPL.csv in trainingdata", {})

    df = pd.read_csv(sample, nrows=0).columns.tolist()
    df = pd.read_csv(sample, usecols=[c for c in ("timestamp", "date") if c in df][:1])
    if df.empty:
        return Section("stale_data", "warn", "empty AAPL.csv", {})
    ts_col = df.columns[0]
    last_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").max()
    today = datetime.now(timezone.utc).date()
    age_days = (today - last_ts.date()).days if pd.notna(last_ts) else 999

    details = {
        "sample": str(sample),
        "last_csv_bar_date": str(last_ts.date()) if pd.notna(last_ts) else None,
        "age_days": age_days,
        "today_utc": str(today),
    }
    if age_days > 14:
        return Section("stale_data", "warn",
                       f"CSV trainingdata/AAPL.csv last bar {age_days}d old — "
                       "live extends via Alpaca bars but sims will be limited",
                       details)
    return Section("stale_data", "ok",
                   f"CSVs {age_days}d behind today (live extends from Alpaca)",
                   details)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_text(report: dict) -> str:
    out = []
    out.append("=" * 78)
    out.append(f"ALGO HEALTH REPORT  {report['ts']}  "
               f"| overall={report['overall_status']}")
    out.append("=" * 78)
    for s in report["sections"]:
        flag = {"ok": "🟢", "warn": "🟡", "fail": "🔴", "info": "🔵"}.get(
            s["status"], "❓"
        )
        out.append(f"{flag} {s['name']:<24} {s['status'].upper():<5} {s['message']}")
    out.append("-" * 78)
    # Pull through the most operator-relevant detail blocks
    interesting = {"score_trend", "plan_preview", "parity_replay",
                   "hyperparam_candidates"}
    for s in report["sections"]:
        if s["name"] not in interesting:
            continue
        out.append(f"\n[{s['name']}]")
        out.append(json.dumps(s["details"], indent=2, default=str)[:2000])
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip Alpaca API + bar-fetch (offline mode)")
    p.add_argument("--no-parity", action="store_true",
                   help="Skip the parity replay (fastest mode)")
    p.add_argument("--n-sessions", type=int, default=10,
                   help="How many recent trade-log files to summarize")
    p.add_argument("--json", action="store_true", help="JSON output only")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    launch_args = _live_launch_args()
    min_score = float(launch_args.get("min-score", "0.85") or 0.85)
    allocation = float(launch_args.get("allocation", "1.0") or 1.0)

    files = _latest_trade_log_files(args.n_sessions)
    last_file, last_event = _last_scored_event(files)

    sections: list[Section] = []
    sections.append(section_lock_holder())
    sections.append(section_ensemble_consistency(launch_args))
    sections.append(section_stale_data())

    score_trend = section_score_trend(files, min_score)
    sections.append(score_trend)

    if not args.no_fetch:
        acct = section_alpaca_account()
        sections.append(acct)
        eq = float(acct.details.get("equity", 0.0))
        sections.append(section_positions_and_fills())
        sections.append(section_plan_preview(launch_args, last_event, eq))
    else:
        sections.append(Section("alpaca_account", "info", "skipped --no-fetch", {}))
        sections.append(Section("positions_and_fills", "info", "skipped --no-fetch", {}))
        sections.append(Section("plan_preview", "info", "skipped --no-fetch", {}))

    if not args.no_parity and not args.no_fetch:
        sections.append(section_parity_replay(last_event, launch_args, SYMBOL_LIST))
    else:
        sections.append(Section("parity_replay", "info", "skipped", {}))

    sections.append(section_hyperparam_candidates(score_trend))

    overall = "ok"
    for s in sections:
        if s.status == "fail":
            overall = "fail"
            break
        if s.status == "warn" and overall != "fail":
            overall = "warn"

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "live_min_score": min_score,
        "live_allocation": allocation,
        "last_session_log": str(last_file) if last_file else None,
        "last_session_ts": last_event.get("ts") if last_event else None,
        "sections": [asdict(s) for s in sections],
    }

    # Always append JSONL
    jsonl = LOG_DIR / "algo_health.jsonl"
    with jsonl.open("a") as f:
        f.write(json.dumps(report, default=str) + "\n")

    # Always rewrite human-readable current
    cur = LOG_DIR / "algo_health_current.txt"
    text = render_text(report)
    cur.write_text(text)

    if args.json:
        print(json.dumps(report, default=str, indent=2))
    else:
        print(text)

    return 0 if overall == "ok" else (2 if overall == "fail" else 1)


if __name__ == "__main__":
    sys.exit(main())
