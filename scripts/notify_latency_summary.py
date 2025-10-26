#!/usr/bin/env python3
"""Post the latency alert digest to a webhook (Slack/Teams/raw)."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from scripts.provider_latency_status import evaluate

def build_payload(
    fmt: str,
    text: str,
    *,
    username: str,
    title: str,
    image_url: str | None,
) -> dict:
    if fmt == "slack":
        payload = {"text": text, "username": username, "mrkdwn": True}
        if image_url:
            payload["attachments"] = [
                {
                    "fallback": title,
                    "title": title,
                    "image_url": image_url,
                }
            ]
        return payload
    if fmt == "teams":
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": title,
            "title": title,
            "text": text.replace("\n", "<br>")[:4000],
        }
        if image_url:
            payload["sections"] = [{"images": [{"image": image_url}], "text": text.replace("\n", "<br>")[:4000]}]
        else:
            payload["sections"] = []
        return payload
    payload = {"title": title, "text": text}
    if image_url:
        payload["image_url"] = image_url
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Send latency digest to webhook.")
    parser.add_argument(
        "--digest",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_digest.md"),
        help="Markdown digest to publish.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.json"),
        help="Latency snapshot JSON for headline status.",
    )
    parser.add_argument("--webhook", default=None, help="Webhook URL (overrides LATENCY_SUMMARY_WEBHOOK env).")
    parser.add_argument(
        "--format",
        choices=("slack", "teams", "raw"),
        default="slack",
        help="Payload format (default slack).",
    )
    parser.add_argument("--username", default="LatencyBot", help="Display name (slack/teams).")
    parser.add_argument("--title", default="Latency Alert Digest", help="Summary title.")
    parser.add_argument("--image-url", default=None, help="Optional image URL to embed in the message.")
    parser.add_argument("--image-path", default=None, help="Optional local image to upload to Slack webhooks.")
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_leaderboard.md"),
        help="Optional leaderboard markdown to include top offenders.",
    )
    parser.add_argument(
        "--weekly-report",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_weekly_trends.md"),
        help="Optional weekly trend markdown to embed.",
    )
    args = parser.parse_args()

    if not args.digest.exists():
        raise FileNotFoundError(f"Digest not found: {args.digest}")

    digest_text = args.digest.read_text(encoding="utf-8").strip()
    if not digest_text:
        print("[info] Digest is empty; nothing to send.")
        return

    status_block = ""
    if args.snapshot.exists():
        snapshot_data = json.loads(args.snapshot.read_text(encoding="utf-8"))
        status, details = evaluate(snapshot_data, warn_threshold=20.0, crit_threshold=40.0)
        lines = [f"*Latency Status:* *{status}*", "", "| Provider | Avg (ms) | ΔAvg (ms) | Severity |", "|----------|---------|-----------|----------|"]
        for provider, stats in sorted(details.items()):
            lines.append(
                f"| {provider} | {stats['avg_ms']:.2f} | {stats['delta_avg_ms']:.2f} | {stats['severity']} |")
        lines.append("")
        status_block = "\n".join(lines)
    else:
        print(f"[warn] Snapshot not found: {args.snapshot}")

    leaderboard_block = ""
    if args.leaderboard and args.leaderboard.exists():
        lines = args.leaderboard.read_text(encoding="utf-8").splitlines()
        header_idx = next((i for i, line in enumerate(lines) if line.startswith("| Provider")), None)
        if header_idx is not None:
            top_rows = lines[header_idx : header_idx + 5]
            leaderboard_block = "Top latency offenders:\n" + "\n".join(top_rows)

    weekly_block = ""
    if args.weekly_report and args.weekly_report.exists():
        weekly_lines = args.weekly_report.read_text(encoding="utf-8").strip().splitlines()
        if weekly_lines:
            weekly_block = "Weekly trend highlights:\n" + "\n".join(weekly_lines[:10])

    text_parts = [block for block in (status_block, leaderboard_block, weekly_block, digest_text) if block]
    text = "\n".join(part for part in text_parts if part)

    webhook = args.webhook or os.environ.get("LATENCY_SUMMARY_WEBHOOK")
    if not webhook:
        print("[warn] No webhook specified; skipping summary post.")
        return

    if args.format == "slack" and args.image_path:
        try:
            import requests

            with Path(args.image_path).open("rb") as fp:
                response = requests.post(
                    webhook,
                    params={
                        "filename": Path(args.image_path).name,
                        "filetype": "png",
                        "title": args.title,
                    },
                    files={"file": fp},
                )
            if response.status_code >= 400:
                print(f"[warn] Slack upload failed: {response.status_code} {response.text}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to upload image: {exc}")

    image_url = args.image_url or os.environ.get("LATENCY_SUMMARY_IMAGE")
    payload = build_payload(args.format, text, username=args.username, title=args.title, image_url=image_url)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            print(f"[info] Posted latency summary to {webhook}")
    except urllib.error.URLError as exc:
        print(f"[warn] Failed to post latency summary: {exc}")


if __name__ == "__main__":
    main()
