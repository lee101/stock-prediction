#!/usr/bin/env python3
"""Write latency status and digest preview to GitHub step summary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import json

from scripts.provider_latency_status import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit latency summary to GH step summary")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.json"),
    )
    parser.add_argument(
        "--digest",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_digest.md"),
    )
    args = parser.parse_args()

    if not args.snapshot.exists():
        raise FileNotFoundError(args.snapshot)

    snapshot = args.snapshot.read_text(encoding="utf-8")
    status, details = evaluate(
        json.loads(snapshot), warn_threshold=20.0, crit_threshold=40.0
    )

    digest_preview = ""
    if args.digest.exists():
        digest_lines = args.digest.read_text(encoding="utf-8").strip().splitlines()
        digest_preview = "\n".join(digest_lines[:20])

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not gh_summary:
        print("[warn] GITHUB_STEP_SUMMARY not set; skipping step summary output")
        return

    with open(gh_summary, "a", encoding="utf-8") as handle:
        handle.write("## Latency Health\n\n")
        handle.write(f"Status: **{status}**\n\n")
        handle.write("| Provider | Avg (ms) | ΔAvg (ms) | Severity |\n")
        handle.write("|----------|---------|-----------|----------|\n")
        for provider, stats in sorted(details.items()):
            handle.write(
                f"| {provider} | {stats['avg_ms']:.2f} | {stats['delta_avg_ms']:.2f} | {stats['severity']} |\n"
            )
        handle.write("\n")
        if digest_preview:
            handle.write("### Recent Alerts\n\n")
            handle.write("```.\n" + digest_preview + "\n```\n\n")


if __name__ == "__main__":
    import json

    main()
