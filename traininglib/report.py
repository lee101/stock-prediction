from __future__ import annotations

import datetime
import json
from pathlib import Path

import torch


def write_report_markdown(
    out_path: str,
    title: str,
    args: dict,
    train_metrics: dict,
    eval_metrics: dict | None = None,
    notes: str | None = None,
):
    out = Path(out_path)
    if out.parent != Path('.'):
        out.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = f"CUDA x{torch.cuda.device_count()} | {torch.cuda.get_device_name(0)}"

    lines = [
        f"# {title}",
        "",
        f"*Generated:* {now}",
        f"*Device:* {device_info}",
        "",
        "## Args",
        "```json",
        json.dumps(args, indent=2, sort_keys=True),
        "```",
        "",
        "## Train Metrics",
        "```json",
        json.dumps(train_metrics, indent=2, sort_keys=True),
        "```",
    ]
    if eval_metrics:
        lines.extend(
            [
                "",
                "## Eval Metrics",
                "```json",
                json.dumps(eval_metrics, indent=2, sort_keys=True),
                "```",
            ]
        )
    if notes:
        lines.extend(["", "## Notes", notes])

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
