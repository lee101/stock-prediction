#!/usr/bin/env python3
"""Simple helper to log latency alerts and optionally hit a webhook."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append latency alert message to a log file.")
    parser.add_argument("--message", required=True, help="Alert message to record.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alerts.log"),
        help="File to append the alert to.",
    )
    parser.add_argument(
        "--webhook",
        default=None,
        help="Optional webhook URL; overrides LATENCY_ALERT_WEBHOOK env if provided.",
    )
    parser.add_argument(
        "--format",
        choices=("slack", "teams", "raw"),
        default="slack",
        help="Webhook payload format (default slack).",
    )
    parser.add_argument("--channel", default=None, help="Optional Slack channel override (slack format).")
    parser.add_argument("--username", default="LatencyBot", help="Display name for slack/teams messages.")
    parser.add_argument("--title", default="Latency Alert", help="Title for structured payloads.")
    parser.add_argument("--log-link", default=None, help="URL to detailed alert/log view.")
    parser.add_argument("--plot-link", default=None, help="URL to latency plot/visualisation.")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {args.message}\n")
    print(f"[info] Logged alert to {args.log}")

    webhook = args.webhook or os.environ.get("LATENCY_ALERT_WEBHOOK")
    log_link = args.log_link or os.environ.get("LATENCY_ALERT_LOG_URL")
    plot_link = args.plot_link or os.environ.get("LATENCY_ALERT_PLOT_URL")
    if webhook:
        payload = build_payload(
            fmt=args.format,
            message=args.message,
            timestamp=timestamp,
            channel=args.channel,
            username=args.username,
            title=args.title,
            log_link=log_link,
            plot_link=plot_link,
        )
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                print(f"[info] Posted alert to webhook {webhook}")
        except urllib.error.URLError as exc:  # noqa: PERF203
            print(f"[warn] Failed to post webhook alert: {exc}")


if __name__ == "__main__":
    main()
def build_payload(
    fmt: str,
    message: str,
    timestamp: str,
    *,
    channel: str | None,
    username: str,
    title: str,
    log_link: str | None,
    plot_link: str | None,
) -> dict:
    if fmt == "slack":
        extra_lines = []
        if log_link:
            extra_lines.append(f"Log: {log_link}")
        if plot_link:
            extra_lines.append(f"Plot: {plot_link}")
        text = message
        if extra_lines:
            text = message + "\n" + "\n".join(extra_lines)
        payload = {"text": text, "username": username}
        if channel:
            payload["channel"] = channel
        return payload
    if fmt == "teams":
        facts = []
        if log_link:
            facts.append({"name": "Log", "value": f"[{log_link}]({log_link})"})
        if plot_link:
            facts.append({"name": "Plot", "value": f"[{plot_link}]({plot_link})"})
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": title,
            "title": title,
            "text": message,
            "sections": [{"facts": facts}] if facts else [],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Logs",
                    "targets": [{"os": "default", "uri": log_link or plot_link or ""}],
                }
            ],
            "themeColor": "0076D7",
        }
    # raw JSON payload
    payload = {"message": message, "timestamp": timestamp}
    if log_link:
        payload["log_link"] = log_link
    if plot_link:
        payload["plot_link"] = plot_link
    return payload
