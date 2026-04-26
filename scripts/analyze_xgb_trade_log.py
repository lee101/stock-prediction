#!/usr/bin/env python3
"""Analyse xgb-daily-trader-live per-session trade logs.

Reads every ``analysis/xgb_live_trade_log/YYYY-MM-DD.jsonl`` and reports:

* per-session summary  (n_picks / n_fills / mean,p50,p95 slippage_bps /
  session PnL / fill rate)
* overall slippage distribution vs the sim's ``fill_buffer_bps``
  (5 bps by default — see ``BacktestConfig.fill_buffer_bps`` and
  ``marketsim/config.py``).

Interpretation rule (see ``project_xgb_prod_trade_log.md``):

    mean slippage_bps_vs_last_close ≈ fill_buffer_bps   → sim calibrated
    mean >> fill_buffer_bps                              → sim under-costs;
                                                           live PnL will
                                                           come in below sim
    mean << fill_buffer_bps                              → sim is
                                                           conservative

The output is informational — it does NOT gate deploy. Use it to decide
whether to bump ``fill_buffer_bps`` in the next sweep.
Use ``--fail-on-spy-provenance-warning`` in monitoring jobs that should fail
when SPY risk-control decision events cannot be tied to exact SPY data bytes.

Usage
-----
    python scripts/analyze_xgb_trade_log.py
    python scripts/analyze_xgb_trade_log.py --log-dir /path/to/logs
    python scripts/analyze_xgb_trade_log.py --json  # machine-readable
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO / "analysis" / "xgb_live_trade_log"
DEFAULT_FILL_BUFFER_BPS = 5.0
SPY_PROVENANCE_EVENTS = {"spy_vol_target", "spy_regime_gate"}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--fill-buffer-bps", type=float,
                   default=DEFAULT_FILL_BUFFER_BPS,
                   help="Sim's fill_buffer_bps assumption; default 5.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON to stdout instead of a "
                        "human-readable table.")
    p.add_argument("--min-sessions", type=int, default=1,
                   help="Suppress overall slippage verdict until at least "
                        "this many sessions have fills. Default 1.")
    p.add_argument("--fail-on-spy-provenance-warning", action="store_true",
                   help="Exit 3 if any session has SPY provenance warnings. "
                        "Default stays informational.")
    return p.parse_args()


def _load_events(path: Path) -> list[dict]:
    events = []
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                # append-only + fsync means we should never see partial
                # lines, but tolerate them for offline analysis.
                continue
    return events


def _percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _summarise_session(path: Path) -> dict:
    events = _load_events(path)
    summary = {
        "session_date": path.stem,
        "path": str(path),
        "n_events": len(events),
        "n_picks": 0,
        "n_buy_submitted": 0,
        "n_buy_filled": 0,
        "n_sell_submitted": 0,
        "n_buy_failed": 0,
        "n_sell_failed": 0,
        "slippages_bps": [],
        "mean_slip_bps": None,
        "p50_slip_bps": None,
        "p95_slip_bps": None,
        "max_slip_bps": None,
        "min_slip_bps": None,
        "equity_pre": None,
        "equity_post": None,
        "session_pnl_abs": None,
        "session_pnl_pct": None,
        "mode": None,
        "paper": None,
        "session_skipped": False,
        "skip_reason": None,
        "no_picks": False,
        "spy_csv": None,
        "spy_csv_sha256": None,
        "spy_decision_event_count": 0,
        "spy_decision_missing_csv_count": 0,
        "spy_decision_missing_sha256_count": 0,
        "spy_decision_csv_values": [],
        "spy_decision_sha256_values": [],
        "spy_provenance_warnings": [],
    }
    for ev in events:
        name = ev.get("event")
        if name == "session_start":
            summary["mode"] = ev.get("mode")
            summary["paper"] = ev.get("paper")
            summary["equity_pre"] = ev.get("equity_pre")
            summary["spy_csv"] = ev.get("spy_csv")
            summary["spy_csv_sha256"] = ev.get("spy_csv_sha256")
        elif name == "pick":
            summary["n_picks"] += 1
        elif name == "buy_submitted":
            summary["n_buy_submitted"] += 1
        elif name == "buy_filled":
            summary["n_buy_filled"] += 1
            slip = ev.get("slippage_bps_vs_last_close")
            if slip is not None:
                try:
                    summary["slippages_bps"].append(float(slip))
                except (TypeError, ValueError):
                    pass
        elif name == "sell_submitted":
            summary["n_sell_submitted"] += 1
        elif name == "buy_failed":
            summary["n_buy_failed"] += 1
        elif name == "sell_failed":
            summary["n_sell_failed"] += 1
        elif name == "session_end":
            summary["equity_post"] = ev.get("equity_post")
            summary["session_pnl_abs"] = ev.get("session_pnl_abs")
            summary["session_pnl_pct"] = ev.get("session_pnl_pct")
        elif name == "session_skipped":
            summary["session_skipped"] = True
            summary["skip_reason"] = ev.get("reason")
        elif name == "no_picks":
            summary["no_picks"] = True
        elif name in SPY_PROVENANCE_EVENTS:
            summary["spy_decision_event_count"] += 1
            spy_csv = ev.get("spy_csv")
            spy_hash = ev.get("spy_csv_sha256")
            if spy_csv:
                summary["spy_decision_csv_values"].append(str(spy_csv))
            else:
                summary["spy_decision_missing_csv_count"] += 1
            if spy_hash:
                summary["spy_decision_sha256_values"].append(str(spy_hash))
            else:
                summary["spy_decision_missing_sha256_count"] += 1
    slips = summary["slippages_bps"]
    if slips:
        summary["mean_slip_bps"] = statistics.fmean(slips)
        summary["p50_slip_bps"] = _percentile(slips, 0.50)
        summary["p95_slip_bps"] = _percentile(slips, 0.95)
        summary["max_slip_bps"] = max(slips)
        summary["min_slip_bps"] = min(slips)
    summary["spy_decision_csv_values"] = sorted(set(summary["spy_decision_csv_values"]))
    summary["spy_decision_sha256_values"] = sorted(set(summary["spy_decision_sha256_values"]))
    session_spy_hash = summary["spy_csv_sha256"]
    decision_hashes = summary["spy_decision_sha256_values"]
    if summary["spy_decision_event_count"] and not session_spy_hash:
        summary["spy_provenance_warnings"].append(
            "spy_session_hash_missing"
        )
    if summary["spy_decision_missing_csv_count"]:
        summary["spy_provenance_warnings"].append(
            "spy_decision_csv_missing"
        )
    if summary["spy_decision_missing_sha256_count"]:
        summary["spy_provenance_warnings"].append(
            "spy_decision_hash_missing"
        )
    if session_spy_hash is not None and decision_hashes:
        mismatches = [h for h in decision_hashes if h != session_spy_hash]
        if mismatches:
            summary["spy_provenance_warnings"].append(
                "spy_decision_hash_mismatch"
            )
    if len(decision_hashes) > 1:
        summary["spy_provenance_warnings"].append(
            "multiple_spy_decision_hashes"
        )
    return summary


def _overall(summaries: list[dict], fill_buffer_bps: float) -> dict:
    all_slips: list[float] = []
    total_fills = 0
    total_picks = 0
    total_failed = 0
    sessions_with_fills = 0
    pnl_pcts: list[float] = []
    spy_session_hashes: set[str] = set()
    spy_warning_sessions: list[str] = []
    for s in summaries:
        if s["slippages_bps"]:
            all_slips.extend(s["slippages_bps"])
            sessions_with_fills += 1
        total_fills += s["n_buy_filled"]
        total_picks += s["n_picks"]
        total_failed += s["n_buy_failed"] + s["n_sell_failed"]
        if s["session_pnl_pct"] is not None:
            try:
                pnl_pcts.append(float(s["session_pnl_pct"]))
            except (TypeError, ValueError):
                pass
        if s.get("spy_csv_sha256"):
            spy_session_hashes.add(str(s["spy_csv_sha256"]))
        if s.get("spy_provenance_warnings"):
            spy_warning_sessions.append(str(s["session_date"]))
    overall = {
        "n_sessions": len(summaries),
        "n_sessions_with_fills": sessions_with_fills,
        "total_picks": total_picks,
        "total_fills": total_fills,
        "total_failed": total_failed,
        "fill_buffer_bps_sim": fill_buffer_bps,
        "n_slippage_samples": len(all_slips),
        "mean_slip_bps": statistics.fmean(all_slips) if all_slips else None,
        "p50_slip_bps": _percentile(all_slips, 0.50),
        "p95_slip_bps": _percentile(all_slips, 0.95),
        "max_slip_bps": max(all_slips) if all_slips else None,
        "min_slip_bps": min(all_slips) if all_slips else None,
        "stdev_slip_bps": (statistics.pstdev(all_slips)
                           if len(all_slips) >= 2 else None),
        "mean_session_pnl_pct": (statistics.fmean(pnl_pcts)
                                 if pnl_pcts else None),
        "n_spy_session_hashes": len(spy_session_hashes),
        "spy_session_sha256_values": sorted(spy_session_hashes),
        "n_spy_provenance_warning_sessions": len(spy_warning_sessions),
        "spy_provenance_warning_sessions": spy_warning_sessions,
    }
    if all_slips and overall["mean_slip_bps"] is not None:
        delta = overall["mean_slip_bps"] - fill_buffer_bps
        overall["delta_vs_sim_bps"] = delta
        if abs(delta) < 2.0:
            verdict = "CALIBRATED (within 2 bps of sim)"
        elif delta > 0:
            verdict = (f"SIM UNDER-COSTS by {delta:+.2f} bps — live fills "
                       "more expensive than assumed; live PnL will come in "
                       "below sim")
        else:
            verdict = (f"SIM CONSERVATIVE by {delta:+.2f} bps — live fills "
                       "cheaper than assumed; live PnL likely > sim")
        overall["verdict"] = verdict
    else:
        overall["delta_vs_sim_bps"] = None
        overall["verdict"] = None
    return overall


def _render_human(summaries: list[dict], overall: dict,
                  fill_buffer_bps: float, min_sessions: int) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(" XGB live trade log — slippage residuals vs sim")
    lines.append("=" * 78)
    lines.append(f"  sim fill_buffer_bps = {fill_buffer_bps:.2f}")
    lines.append(f"  sessions found       = {overall['n_sessions']}")
    lines.append(f"    with fills         = {overall['n_sessions_with_fills']}")
    lines.append(f"  picks / fills / fail = "
                 f"{overall['total_picks']} / {overall['total_fills']} / "
                 f"{overall['total_failed']}")
    lines.append("")

    header = (f"{'date':12}  {'mode':5}  {'picks':>5}  {'fills':>5}  "
              f"{'mean':>8}  {'p50':>8}  {'p95':>8}  {'pnl%':>7}  {'notes'}")
    lines.append(header)
    lines.append("-" * len(header))
    for s in summaries:
        notes = []
        if s["session_skipped"]:
            notes.append(f"skipped({s.get('skip_reason') or '?'})")
        if s["no_picks"]:
            notes.append("no_picks")
        if s["n_buy_failed"] or s["n_sell_failed"]:
            notes.append(f"fail({s['n_buy_failed']}+{s['n_sell_failed']})")
        if s["spy_provenance_warnings"]:
            notes.append("spy_provenance_warning")

        def _fmt(x, w=8, p=2):
            return "     -- " if x is None else f"{x:>{w}.{p}f}"

        lines.append(
            f"{s['session_date']:12}  "
            f"{(s['mode'] or '-'):>5}  "
            f"{s['n_picks']:>5}  "
            f"{s['n_buy_filled']:>5}  "
            f"{_fmt(s['mean_slip_bps'])}  "
            f"{_fmt(s['p50_slip_bps'])}  "
            f"{_fmt(s['p95_slip_bps'])}  "
            f"{_fmt(s['session_pnl_pct'], 7)}  "
            f"{', '.join(notes)}"
        )
    lines.append("")

    if overall["n_slippage_samples"] >= 1 and \
            overall["n_sessions_with_fills"] >= min_sessions:
        lines.append("overall slippage (bps vs last_close):")
        lines.append(f"  samples  = {overall['n_slippage_samples']}")
        lines.append(f"  mean     = {overall['mean_slip_bps']:+.2f}")
        lines.append(f"  p50      = {overall['p50_slip_bps']:+.2f}")
        lines.append(f"  p95      = {overall['p95_slip_bps']:+.2f}")
        lines.append(f"  min/max  = {overall['min_slip_bps']:+.2f} / "
                     f"{overall['max_slip_bps']:+.2f}")
        if overall["stdev_slip_bps"] is not None:
            lines.append(f"  stdev    = {overall['stdev_slip_bps']:.2f}")
        lines.append(f"  Δ vs sim = {overall['delta_vs_sim_bps']:+.2f} bps")
        lines.append(f"  verdict  : {overall['verdict']}")
    else:
        lines.append("overall slippage: insufficient data "
                     f"(need ≥{min_sessions} session(s) with fills; "
                     f"have {overall['n_sessions_with_fills']})")
    if overall["mean_session_pnl_pct"] is not None:
        lines.append(f"  mean session pnl% = "
                     f"{overall['mean_session_pnl_pct']:+.3f}")
    if overall["n_spy_session_hashes"]:
        preview = ", ".join(
            h[:12] for h in overall["spy_session_sha256_values"][:3]
        )
        extra = "" if overall["n_spy_session_hashes"] <= 3 else ", ..."
        lines.append(
            "  SPY session hashes = "
            f"{overall['n_spy_session_hashes']} unique ({preview}{extra})"
        )
    if overall["n_spy_provenance_warning_sessions"]:
        lines.append(
            "  SPY provenance warnings = "
            f"{overall['n_spy_provenance_warning_sessions']} session(s): "
            f"{', '.join(overall['spy_provenance_warning_sessions'])}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    if not args.log_dir.exists():
        print(f"[analyze] log dir not found: {args.log_dir}", file=sys.stderr)
        return 2
    paths = sorted(args.log_dir.glob("*.jsonl"))
    if not paths:
        print(f"[analyze] no *.jsonl files in {args.log_dir}", file=sys.stderr)
        return 0  # nothing to report, not an error
    summaries = [_summarise_session(p) for p in paths]
    overall = _overall(summaries, args.fill_buffer_bps)
    if args.json:
        payload = {
            "overall": overall,
            "sessions": [
                {k: v for k, v in s.items() if k != "slippages_bps"}
                for s in summaries
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_human(summaries, overall,
                            args.fill_buffer_bps, args.min_sessions))
    if args.fail_on_spy_provenance_warning and \
            overall["n_spy_provenance_warning_sessions"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
