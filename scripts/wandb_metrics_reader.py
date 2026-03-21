#!/usr/bin/env python3
"""
WandB metrics reader — prints train/val metrics as LLM-friendly markdown.

Usage:
  python scripts/wandb_metrics_reader.py --project stock --last-n-runs 5
  python scripts/wandb_metrics_reader.py --project stock --run-id abc123xyz
  python scripts/wandb_metrics_reader.py --project stock --group my_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# WandB import guard
# ---------------------------------------------------------------------------


def _import_wandb():
    try:
        import wandb  # noqa: PLC0415

        return wandb
    except ImportError:
        print(
            "Error: wandb is not installed.\n"
            "Install it with:  uv pip install wandb\n"
            "Then set your API key:  wandb login\n"
            "  or export WANDB_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_api_key() -> None:
    """Exit with a helpful message if no WandB API key is available."""
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return
    # wandb also reads from ~/.netrc / the local config; only warn when blank
    netrc = Path.home() / ".netrc"
    wandb_dir = Path.home() / ".config" / "wandb" / "settings"
    if netrc.exists() or wandb_dir.exists():
        return  # key likely stored locally
    print(
        "Error: WANDB_API_KEY environment variable is not set and no local "
        "wandb credentials were found.\n"
        "Fix with one of:\n"
        "  export WANDB_API_KEY=<your-key>\n"
        "  wandb login",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_CONFIG_KEYS = ["hidden_size", "lr", "anneal_lr", "ent_coef", "seed"]


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _pick(d: dict, *keys: str, default=None):
    """Return the first found value among keys in dict d."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _sample_history(rows: list[dict], key: str, n_points: int = 10) -> list[float]:
    """Down-sample a metric history to at most n_points values."""
    values = [r[key] for r in rows if key in r and r[key] is not None]
    if not values:
        return []
    if len(values) <= n_points:
        return values
    step = max(1, len(values) // n_points)
    sampled = values[::step]
    # Always include the last value
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def _fmt_curve(values: list[float], precision: int = 3) -> str:
    if not values:
        return "—"
    return " → ".join(f"{v:.{precision}f}" for v in values)


def _extract_config(run_config: dict) -> dict:
    """Pull key hyperparams from a wandb run config dict."""
    result = {}
    for k in _CONFIG_KEYS:
        if k in run_config:
            val = run_config[k]
            # wandb stores config values as {"value": ..., "desc": ...} sometimes
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            result[k] = val
    return result


def _fmt_config(cfg: dict) -> str:
    """Format config dict as compact k=v string."""
    parts = []
    for k, v in cfg.items():
        short_k = {"hidden_size": "h", "anneal_lr": "ann", "ent_coef": "ent"}.get(k, k)
        if isinstance(v, float):
            parts.append(f"{short_k}={v:.0e}" if v < 0.01 else f"{short_k}={v}")
        else:
            parts.append(f"{short_k}={v}")
    return " ".join(parts) if parts else "—"


def _best_val_return(summary: dict) -> float | None:
    v = _pick(summary, "val/return", "val_return", "best_val_return")
    if v is not None:
        return float(v)
    return None


def _final_loss(summary: dict) -> float | None:
    v = _pick(summary, "train/policy_loss", "policy_loss", "train/loss", "loss")
    if v is not None:
        return float(v)
    return None


def _step_count(summary: dict) -> int | None:
    v = _pick(summary, "global_step", "_step", "train/global_step", "step")
    if v is not None:
        return int(v)
    return None


def fetch_runs(
    wandb,
    project: str,
    entity: str | None,
    run_id: str | None,
    group: str | None,
    last_n: int,
) -> list[dict[str, Any]]:
    """Fetch run metadata from WandB and return a list of run dicts."""
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    if run_id:
        run_path = f"{project_path}/{run_id}"
        try:
            raw = api.run(run_path)
            runs_raw = [raw]
        except Exception as exc:
            print(f"Error: could not fetch run {run_path!r}: {exc}", file=sys.stderr)
            return []
    else:
        filters: dict = {}
        if group:
            filters["group"] = group
        try:
            runs_raw = list(
                api.runs(project_path, filters=filters or None, order="-created_at")
            )
        except Exception as exc:
            print(f"Error: could not fetch runs for {project_path!r}: {exc}", file=sys.stderr)
            return []
        runs_raw = runs_raw[:last_n]

    result = []
    for r in runs_raw:
        summary = dict(r.summary) if r.summary else {}
        config = _extract_config(dict(r.config) if r.config else {})

        val_return = _best_val_return(summary)
        final_loss = _final_loss(summary)
        steps = _step_count(summary)
        duration = summary.get("_runtime") or None

        # Fetch history for loss curve + entropy (sampled)
        loss_curve: list[float] = []
        entropy_curve: list[float] = []
        try:
            history = list(r.history(keys=["train/policy_loss", "train/entropy"], samples=200))
            loss_curve = _sample_history(history, "train/policy_loss")
            entropy_curve = _sample_history(history, "train/entropy")
        except Exception:
            pass

        # Extra summary metrics
        sortino = _pick(summary, "val/sortino", "sortino")
        win_rate = _pick(summary, "val/win_rate", "win_rate")

        result.append(
            {
                "id": r.id,
                "name": r.name,
                "state": getattr(r, "state", "unknown"),
                "config": config,
                "val_return": val_return,
                "final_loss": final_loss,
                "steps": steps,
                "duration": duration,
                "loss_curve": loss_curve,
                "entropy_curve": entropy_curve,
                "sortino": sortino,
                "win_rate": win_rate,
            }
        )

    # Sort by primary metric descending (runs without the metric go last)
    def sort_key(run):
        v = run.get("val_return")
        return v if v is not None else float("-inf")

    result.sort(key=sort_key, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%" if abs(v) < 100 else f"{sign}{v:.1f}x"


def format_markdown(
    runs: list[dict],
    project: str,
    entity: str | None,
    primary_metric: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    proj_label = f"{entity}/{project}" if entity else project
    n = len(runs)

    lines: list[str] = []
    lines.append(f"## WandB Metrics: {proj_label} (last {n} run{'s' if n != 1 else ''})")
    lines.append(f"*Fetched: {now}*")
    lines.append("")

    if not runs:
        lines.append("*No runs found.*")
        return "\n".join(lines)

    # --- Summary table ---
    header = "| Run | Config | Best Val Return | Final Loss | Steps | Duration |"
    sep    = "|-----|--------|----------------|------------|-------|----------|"
    lines.append(header)
    lines.append(sep)

    val_returns = [r["val_return"] for r in runs]
    best_idx = next((i for i, v in enumerate(val_returns) if v is not None), None)

    for i, r in enumerate(runs):
        vr_str = _pct(r["val_return"])
        if i == best_idx:
            vr_str = f"**{vr_str}**"

        loss_str = f"{r['final_loss']:.3f}" if r["final_loss"] is not None else "—"
        steps_str = f"{r['steps'] / 1e6:.1f}M" if r["steps"] is not None else "—"
        dur_str = _fmt_duration(r["duration"])
        cfg_str = _fmt_config(r["config"])
        lines.append(
            f"| {r['name']} | {cfg_str} | {vr_str} | {loss_str} | {steps_str} | {dur_str} |"
        )

    lines.append("")

    # --- Best run detail ---
    if best_idx is not None:
        best = runs[best_idx]
        lines.append(f"### Best Run: {best['name']}")

        vr = _pct(best["val_return"])
        sortino_str = f" | Sortino: {best['sortino']:.2f}" if best["sortino"] is not None else ""
        wr_str = f" | Win rate: {best['win_rate'] * 100:.0f}%" if best["win_rate"] is not None else ""
        lines.append(f"- **Val return**: {vr}{sortino_str}{wr_str}")

        if best["loss_curve"]:
            lines.append(f"- **Train loss curve** (sampled): {_fmt_curve(best['loss_curve'])}")

        if best["entropy_curve"]:
            lines.append(f"- **Entropy**: {_fmt_curve(best['entropy_curve'])} (healthy decrease)")

        cfg = best["config"]
        if cfg:
            cfg_parts = []
            for k, v in cfg.items():
                cfg_parts.append(f"{k}={v}")
            lines.append(f"- **Config**: {', '.join(cfg_parts)}")

    return "\n".join(lines)


def format_json(runs: list[dict], project: str, entity: str | None) -> str:
    now = datetime.now(timezone.utc).isoformat()
    proj_label = f"{entity}/{project}" if entity else project
    payload = {
        "project": proj_label,
        "fetched_at": now,
        "runs": runs,
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query WandB API and print metrics as LLM-friendly markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity/username")
    parser.add_argument("--run-id", default=None, help="Specific run ID to fetch")
    parser.add_argument("--group", default=None, help="Filter by run group")
    parser.add_argument(
        "--last-n-runs", type=int, default=5, metavar="N",
        help="How many recent runs to show",
    )
    parser.add_argument(
        "--metric", default="val/return", dest="metric",
        help="Primary metric to sort/display",
    )
    parser.add_argument(
        "--format", default="markdown", choices=["markdown", "json"],
        dest="output_format",
        help="Output format",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    _check_api_key()
    wandb = _import_wandb()

    runs = fetch_runs(
        wandb=wandb,
        project=args.project,
        entity=args.entity,
        run_id=args.run_id,
        group=args.group,
        last_n=args.last_n_runs,
    )

    if args.output_format == "json":
        print(format_json(runs, args.project, args.entity))
    else:
        print(format_markdown(runs, args.project, args.entity, args.metric))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
