#!/usr/bin/env python3
"""Run the full ETF trend readiness pipeline sequentially.

This script orchestrates a single refresh cycle:

1. Fetch trend data (with provider fallbacks).
2. Regenerate readiness / momentum reports.
3. Probe forecast gates for the latest candidates.
4. Emit margin alerts when strategy-return shortfalls are small.

All commands run in-process via ``python`` so a cron/CI job can invoke a
single executable and inspect its exit code.  Each step stops the pipeline
on failure to avoid producing partially updated artefacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from provider_latency_status import evaluate


def run_step(label: str, argv: List[str]) -> None:
    print(f"[pipeline] {label}: {' '.join(argv)}", flush=True)
    result = subprocess.run(argv, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trend readiness pipeline.")
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=Path("marketsimulator/etf_watchlist.txt"),
        help="Watchlist to pass to fetch_etf_trends.py (default: marketsimulator/etf_watchlist.txt).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history to request for the trend fetch (default: 365).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window for trend metrics (default: 50).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["stooq", "yahoo"],
        choices=("stooq", "yahoo"),
        help="Ordered list of data providers to attempt (default: stooq yahoo).",
    )
    parser.add_argument(
        "--trend-summary",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Location for trend_summary.json (default: marketsimulator/run_logs/trend_summary.json).",
    )
    parser.add_argument(
        "--provider-log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage.csv"),
        help="CSV path for provider usage counts (default: marketsimulator/run_logs/provider_usage.csv).",
    )
    parser.add_argument(
        "--provider-switch-log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_switches.csv"),
        help="CSV path for provider switch events.",
    )
    parser.add_argument(
        "--provider-summary",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage_summary.txt"),
        help="Text file capturing provider usage summary for this run.",
    )
    parser.add_argument(
        "--provider-summary-window",
        type=int,
        default=20,
        help="Number of rows to include in provider usage timeline (0 = all).",
    )
    parser.add_argument(
        "--provider-sparkline",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage_sparkline.md"),
        help="Markdown file with provider usage sparkline.",
    )
    parser.add_argument(
        "--provider-sparkline-window",
        type=int,
        default=20,
        help="Number of runs to include in provider sparkline (0 = all).",
    )
    parser.add_argument(
        "--latency-log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency.csv"),
        help="CSV file capturing per-symbol latency observations.",
    )
    parser.add_argument(
        "--latency-summary",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_summary.txt"),
        help="Text file with aggregated latency statistics.",
    )
    parser.add_argument(
        "--latency-p95-threshold",
        type=float,
        default=500.0,
        help="Alert threshold (ms) for provider p95 latency.",
    )
    parser.add_argument(
        "--latency-rollup",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rollup.csv"),
        help="CSV file capturing per-run aggregated latency statistics.",
    )
    parser.add_argument(
        "--latency-rolling",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.md"),
        help="Markdown file summarising rolling latency averages.",
    )
    parser.add_argument(
        "--latency-rolling-window",
        type=int,
        default=5,
        help="Window size for rolling latency averages (number of runs).",
    )
    parser.add_argument(
        "--latency-rolling-json",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.json"),
        help="JSON file storing rolling latency stats for change detection.",
    )
    parser.add_argument(
        "--latency-rolling-history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling_history.jsonl"),
        help="JSONL file keeping rolling latency snapshots over time.",
    )
    parser.add_argument(
        "--latency-history-md",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.md"),
        help="Markdown file for long-horizon latency trends.",
    )
    parser.add_argument(
        "--latency-history-html",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.html"),
        help="HTML plot for latency history.",
    )
    parser.add_argument(
        "--latency-history-png",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.png"),
        help="PNG thumbnail for latency history.",
    )
    parser.add_argument(
        "--alert-notify",
        type=Path,
        default=Path("scripts/notify_latency_alert.py"),
        help="Optional notifier script to invoke when alerts fire (set to empty to disable).",
    )
    parser.add_argument(
        "--alert-log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alerts.log"),
        help="Log file passed to the notifier script.",
    )
    parser.add_argument(
        "--summary-webhook",
        type=str,
        default=None,
        help="Optional webhook URL to post the latency digest after pipeline completes.",
    )
    parser.add_argument(
        "--latency-delta-threshold",
        type=float,
        default=40.0,
        help="Trigger alert when rolling avg latency shifts more than this many ms.",
    )
    parser.add_argument(
        "--latency-warn-threshold",
        type=float,
        default=20.0,
        help="WARN threshold for provider latency status (default 20).",
    )
    parser.add_argument(
        "--halt-on-crit",
        action="store_true",
        help="Exit with code 2 when latency status is CRIT (after logging alerts).",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="Optional base URL for artefacts (e.g., https://example.com/logs). Used in alerts.",
    )
    parser.add_argument(
        "--readiness-md",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_readiness.md"),
        help="Candidate readiness markdown output path.",
    )
    parser.add_argument(
        "--readiness-history",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_readiness_history.csv"),
        help="History CSV for readiness snapshots.",
    )
    parser.add_argument(
        "--momentum-md",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_momentum.md"),
        help="Momentum summary markdown path.",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_forecast_gate_report.md"),
        help="Forecast gate report markdown path.",
    )
    parser.add_argument(
        "--gate-history",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_forecast_gate_history.csv"),
        help="Forecast gate history CSV path.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.003,
        help="Shortfall tolerance passed to forecast_margin_alert.py (default: 0.003).",
    )
    parser.add_argument(
        "--min-sma",
        type=float,
        default=200.0,
        help="Minimum SMA threshold for readiness / forecast probes (default: 200).",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.0,
        help="Minimum fractional percent change for readiness / forecast probes (default: 0).",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=1,
        help="Number of simulation steps for forecast probes (default: 1).",
    )
    parser.add_argument(
        "--probe-min-strategy-return",
        type=float,
        default=0.015,
        help="Strategy return gate for candidate probes (default: 0.015).",
    )
    parser.add_argument(
        "--probe-min-predicted-move",
        type=float,
        default=0.01,
        help="Predicted move gate for candidate probes (default: 0.01).",
    )
    args = parser.parse_args()

    previous_rolling: Dict[str, Dict[str, float]] = {}
    alert_messages: List[str] = []
    if args.latency_rolling_json.exists():
        try:
            previous_rolling = json.loads(args.latency_rolling_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_rolling = {}

    providers_arg = []
    for provider in args.providers:
        providers_arg.extend(["--providers", provider])

    run_step(
        "fetch_trends",
        [
            "python",
            "scripts/fetch_etf_trends.py",
            "--symbols-file",
            str(args.symbols_file),
            "--days",
            str(args.days),
            "--window",
            str(args.window),
            "--summary-path",
            str(args.trend_summary),
            "--provider-log",
            str(args.provider_log),
            "--provider-switch-log",
            str(args.provider_switch_log),
            "--latency-log",
            str(args.latency_log),
            *providers_arg,
        ],
    )

    run_step(
        "candidate_readiness",
        [
            "python",
            "scripts/generate_candidate_readiness.py",
            "--summary-path",
            str(args.trend_summary),
            "--output",
            str(args.readiness_md),
            "--csv-output",
            str(args.readiness_history),
            "--min-sma",
            str(args.min_sma),
            "--min-pct",
            str(args.min_pct),
        ],
    )

    run_step(
        "provider_latency_summary",
        [
            "python",
            "scripts/provider_latency_report.py",
            "--log",
            str(args.latency_log),
            "--output",
            str(args.latency_summary),
            "--p95-threshold",
            str(args.latency_p95_threshold),
            "--rollup-csv",
            str(args.latency_rollup),
        ],
    )

    run_step(
        "provider_latency_rolling",
        [
            "python",
            "scripts/provider_latency_rolling.py",
            "--rollup",
            str(args.latency_rollup),
            "--output",
            str(args.latency_rolling),
            "--window",
            str(args.latency_rolling_window),
            "--json-output",
            str(args.latency_rolling_json),
            "--history-jsonl",
            str(args.latency_rolling_history),
        ],
    )

    current_rolling: Dict[str, Dict[str, float]] = {}
    if args.latency_rolling_json.exists():
        try:
            current_rolling = json.loads(args.latency_rolling_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_rolling = {}

    if previous_rolling and current_rolling:
        pipeline_status = "OK"
        for provider, stats in current_rolling.items():
            prev_stats = previous_rolling.get(provider)
            if not prev_stats:
                continue
            shift = stats.get("avg_ms", 0.0) - prev_stats.get("avg_ms", 0.0)
            if abs(shift) >= args.latency_delta_threshold:
                message = (
                    f"Rolling latency for {provider} shifted {shift:+.2f} ms "
                    f"(threshold {args.latency_delta_threshold:.2f} ms)"
                )
                print(f"[alert] {message}")
                alert_messages.append(message)
                pipeline_status = "CRIT"
        if pipeline_status != "OK":
            print("[warn] Latency status = CRIT; downstream tasks should consider pausing onboarding.")

    status = "OK"
    status_details: Dict[str, Dict[str, float]] = {}
    if current_rolling:
        status, status_details = evaluate(
            current_rolling,
            warn_threshold=args.latency_warn_threshold,
            crit_threshold=args.latency_delta_threshold,
        )
        print(
            f"[info] Latency status {status} (warn={args.latency_warn_threshold}ms crit={args.latency_delta_threshold}ms)"
        )
        for provider, stats in sorted(status_details.items()):
            print(
                f"    {provider}: avg={stats['avg_ms']:.2f}ms Δavg={stats['delta_avg_ms']:.2f}ms "
                f"severity={stats['severity']}"
            )
        if status == "CRIT" and args.halt_on_crit:
            print("[error] Latency status CRIT and --halt-on-crit set; aborting pipeline.")
            raise SystemExit(2)

    run_step(
        "provider_latency_history",
        [
            "python",
            "scripts/provider_latency_history_report.py",
            "--history",
            str(args.latency_rolling_history),
            "--output",
            str(args.latency_history_md),
            "--window",
            str(max(args.latency_rolling_window * 2, 10)),
        ],
    )

    run_step(
        "provider_latency_history_plot",
        [
            "python",
            "scripts/provider_latency_history_plot.py",
            "--history",
            str(args.latency_rolling_history),
            "--output",
            str(args.latency_history_html),
            "--window",
            str(max(args.latency_rolling_window * 4, 20)),
        ],
    )

    run_step(
        "provider_latency_history_png",
        [
            "python",
            "scripts/provider_latency_history_png.py",
            "--history",
            str(args.latency_rolling_history),
            "--output",
            str(args.latency_history_png),
            "--window",
            str(max(args.latency_rolling_window * 4, 20)),
            "--warning-threshold",
            str(args.latency_delta_threshold),
        ],
    )

    run_step(
        "provider_latency_alert_digest",
        [
            "python",
            "scripts/provider_latency_alert_digest.py",
            "--log",
            str(args.alert_log),
            "--output",
            str(Path("marketsimulator/run_logs/provider_latency_alert_digest.md")),
            "--history",
            "marketsimulator/run_logs/provider_latency_alert_history.jsonl",
        ],
    )

    run_step(
        "provider_latency_leaderboard",
        [
            "python",
            "scripts/provider_latency_leaderboard.py",
            "--history",
            "marketsimulator/run_logs/provider_latency_alert_history.jsonl",
            "--output",
            "marketsimulator/run_logs/provider_latency_leaderboard.md",
        ],
    )

    run_step(
        "provider_latency_weekly_report",
        [
            "python",
            "scripts/provider_latency_weekly_report.py",
            "--history",
            "marketsimulator/run_logs/provider_latency_alert_history.jsonl",
            "--output",
            "marketsimulator/run_logs/provider_latency_weekly_trends.md",
        ],
    )

    run_step(
        "provider_latency_trend_gate",
        [
            sys.executable,
            "scripts/provider_latency_trend_gate.py",
            "--history",
            "marketsimulator/run_logs/provider_latency_alert_history.jsonl",
        ],
    )

    if args.summary_webhook:
        image_url_arg: List[str] = []
        if args.public_base_url and args.latency_history_png:
            try:
                rel_png = args.latency_history_png.resolve().relative_to(Path.cwd())
                image_url_arg = [
                    "--image-url",
                    f"{args.public_base_url.rstrip('/')}/{rel_png.as_posix()}",
                ]
            except ValueError:
                pass
        run_step(
            "notify_latency_summary",
            [
                sys.executable,
                "scripts/notify_latency_summary.py",
                "--digest",
                "marketsimulator/run_logs/provider_latency_alert_digest.md",
                "--webhook",
                args.summary_webhook,
                *image_url_arg,
            ],
        )

    if alert_messages and args.alert_notify:
        if args.alert_notify.exists():
            log_link = args.alert_log.resolve().as_uri() if args.alert_log else None
            plot_link = args.latency_history_png.resolve().as_uri() if args.latency_history_png else None
            if args.public_base_url:
                try:
                    rel_log = args.alert_log.resolve().relative_to(Path.cwd()) if args.alert_log else None
                    rel_png = (
                        args.latency_history_png.resolve().relative_to(Path.cwd())
                        if args.latency_history_png
                        else None
                    )
                    base = args.public_base_url.rstrip("/")
                    if rel_log:
                        log_link = f"{base}/{rel_log.as_posix()}"
                    if rel_png:
                        plot_link = f"{base}/{rel_png.as_posix()}"
                except ValueError:
                    # Artefact is outside cwd; keep file:// link
                    pass
            for message in alert_messages:
                cmd = [
                    sys.executable,
                    str(args.alert_notify),
                    "--message",
                    message,
                    "--log",
                    str(args.alert_log),
                ]
                if log_link:
                    cmd.extend(["--log-link", log_link])
                if plot_link:
                    cmd.extend(["--plot-link", plot_link])
                subprocess.run(cmd, check=False)
        else:
            print(f"[warn] Alert notifier not found: {args.alert_notify}")

    run_step(
        "candidate_momentum",
        [
            "python",
            "scripts/analyze_candidate_history.py",
            "--history",
            str(args.readiness_history),
            "--output",
            str(args.momentum_md),
        ],
    )

    run_step(
        "forecast_gate_probe",
        [
            "python",
            "scripts/check_candidate_forecasts.py",
            "--history",
            str(args.readiness_history),
            "--output",
            str(args.gate_report),
            "--csv-output",
            str(args.gate_history),
            "--min-sma",
            str(args.min_sma),
            "--min-pct",
            str(args.min_pct),
            "--steps",
            str(args.probe_steps),
            "--min-strategy-return",
            str(args.probe_min_strategy_return),
            "--min-predicted-move",
            str(args.probe_min_predicted_move),
        ],
    )

    run_step(
        "forecast_margin_alert",
        [
            "python",
            "scripts/forecast_margin_alert.py",
            "--report",
            str(args.gate_report),
            "--max-shortfall",
            str(args.margin_threshold),
        ],
    )

    run_step(
        "provider_usage_summary",
        [
            "python",
            "scripts/provider_usage_report.py",
            "--log",
            str(args.provider_log),
            "--output",
            str(args.provider_summary),
            "--timeline-window",
            str(args.provider_summary_window),
        ],
    )

    run_step(
        "provider_usage_sparkline",
        [
            "python",
            "scripts/provider_usage_sparkline.py",
            "--log",
            str(args.provider_log),
            "--output",
            str(args.provider_sparkline),
            "--window",
            str(args.provider_sparkline_window),
        ],
    )


if __name__ == "__main__":
    main()
    alert_messages: List[str] = []
