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
import importlib
import json
import math
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
    blocked_paths = {
        str(REPO.resolve()),
        str((REPO / "scripts").resolve()),
        str(Path.cwd().resolve()),
    }

    def _is_valid(module: Any) -> bool:
        return hasattr(module, "Api") and hasattr(module, "init")

    try:
        import wandb  # noqa: PLC0415

        if _is_valid(wandb):
            return wandb
    except ImportError:
        pass

    cached = sys.modules.pop("wandb", None)
    original_path = list(sys.path)
    try:
        sys.path = [
            path for path in sys.path
            if str(Path(path or ".").resolve()) not in blocked_paths
        ]
        importlib.invalidate_caches()
        import wandb  # noqa: PLC0415

        if _is_valid(wandb):
            return wandb
        raise ImportError("Imported wandb module does not expose the expected API client")
    except ImportError:
        if cached is not None:
            sys.modules["wandb"] = cached
        print(
            "Error: wandb is not installed.\n"
            "Install it with:  uv pip install wandb\n"
            "Then set your API key:  wandb login\n"
            "  or export WANDB_API_KEY=<your-key>",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        sys.path = original_path


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

_METRIC_ALIASES = {
    "val/return": "val_return",
    "val_return": "val_return",
    "best_val_return": "val_return",
    "val/sortino": "sortino",
    "sortino": "sortino",
    "val/win_rate": "win_rate",
    "win_rate": "win_rate",
    "train/final_return": "train_return",
    "train_return": "train_return",
    "val/max_drawdown": "max_drawdown",
    "max_drawdown": "max_drawdown",
    "smooth_score": "smooth_score",
    "stability_score": "stability_score",
    "generalization_gap": "generalization_gap",
    "overfit_risk": "overfit_risk",
}

_SUMMARY_PRIMARY_KEYS = (
    "val/return",
    "val_return",
    "best_val_return",
    "val/sortino",
    "sortino",
    "train/final_return",
    "train_return",
    "train/return",
    "train/policy_loss",
    "policy_loss",
    "smooth_score",
)


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


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    raw = getattr(obj, "_json_dict", None)
    if isinstance(raw, dict):
        return dict(raw)
    items = getattr(obj, "items", None)
    if callable(items):
        try:
            return dict(items())
        except Exception:
            pass
    try:
        return dict(obj)
    except Exception:
        return {}


def _summary_has_primary_metrics(summary: dict[str, Any]) -> bool:
    return any(_pick(summary, key) is not None for key in _SUMMARY_PRIMARY_KEYS)


def _load_local_wandb_summary(run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}

    search_roots: list[Path] = []
    for candidate in (REPO / "wandb", Path.cwd() / "wandb"):
        resolved = candidate.resolve()
        if resolved in search_roots or not resolved.exists():
            continue
        search_roots.append(resolved)

    patterns = (
        f"run-*-{run_id}/files/wandb-summary.json",
        f"offline-run-*-{run_id}/files/wandb-summary.json",
    )
    for root in search_roots:
        for pattern in patterns:
            matches = sorted(root.glob(pattern), reverse=True)
            for path in matches:
                try:
                    payload = json.loads(path.read_text())
                except Exception:
                    continue
                if isinstance(payload, dict):
                    return payload
    return {}


def _merge_local_summary(run_id: str | None, summary: dict[str, Any]) -> dict[str, Any]:
    if _summary_has_primary_metrics(summary):
        return summary
    local_summary = _load_local_wandb_summary(run_id)
    if not local_summary:
        return summary
    merged = dict(local_summary)
    merged.update(summary)
    return merged


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


def _sample_history_any(rows: list[dict], keys: list[str], n_points: int = 10) -> list[float]:
    for key in keys:
        sampled = _sample_history(rows, key, n_points=n_points)
        if sampled:
            return sampled
    return []


def _fmt_curve(values: list[float], precision: int = 3) -> str:
    if not values:
        return "—"
    return " → ".join(f"{v:.{precision}f}" for v in values)


def _fmt_metric_value(metric_name: str, value: float | None) -> str:
    if value is None:
        return "—"
    name = metric_name.lower()
    if any(token in name for token in ("return", "drawdown", "gap")):
        return _pct(value)
    if name.endswith("win_rate") or name.endswith("/win_rate") or name.endswith("profitable_pct"):
        return _pct(value)
    return f"{value:.3f}"


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


def _curve_direction_ratio(values: list[float], *, increasing: bool) -> float | None:
    if len(values) < 2:
        return None
    diffs = [curr - prev for prev, curr in zip(values, values[1:])]
    if not diffs:
        return None
    good_steps = 0
    for diff in diffs:
        if increasing and diff >= 0:
            good_steps += 1
        if not increasing and diff <= 0:
            good_steps += 1
    return good_steps / len(diffs)


def _curve_flip_rate(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    signs: list[int] = []
    for prev, curr in zip(values, values[1:]):
        diff = curr - prev
        if abs(diff) <= 1e-12:
            continue
        signs.append(1 if diff > 0 else -1)
    if len(signs) < 2:
        return 0.0
    flips = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return flips / (len(signs) - 1)


def _curve_noise_ratio(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    diffs = [curr - prev for prev, curr in zip(values, values[1:])]
    mean_abs_diff = sum(abs(diff) for diff in diffs) / len(diffs)
    mean_abs_value = sum(abs(value) for value in values) / len(values)
    scale = max(abs(values[0]), abs(values[-1]), mean_abs_value, 1e-8)
    return mean_abs_diff / scale


def _penalty_to_score(value: float | None, *, scale: float) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 1.0
    return 1.0 / (1.0 + (value / scale))


def compute_stability_metrics(run: dict[str, Any]) -> dict[str, float | None]:
    loss_curve = [float(v) for v in run.get("loss_curve") or []]
    return_curve = [float(v) for v in run.get("val_return_curve") or []]
    sortino_curve = [float(v) for v in run.get("val_sortino_curve") or []]

    loss_downhill_pct = _curve_direction_ratio(loss_curve, increasing=False)
    loss_flip_rate = _curve_flip_rate(loss_curve)
    loss_noise = _curve_noise_ratio(loss_curve)

    return_uphill_pct = _curve_direction_ratio(return_curve, increasing=True)
    return_flip_rate = _curve_flip_rate(return_curve)
    return_noise = _curve_noise_ratio(return_curve)

    sortino_uphill_pct = _curve_direction_ratio(sortino_curve, increasing=True)

    train_return = _coerce_float(run.get("train_return"))
    val_return = _coerce_float(run.get("val_return"))
    generalization_gap = None
    if train_return is not None and val_return is not None:
        generalization_gap = train_return - val_return

    max_drawdown = _coerce_float(run.get("max_drawdown"))
    drawdown_abs = abs(max_drawdown) if max_drawdown is not None else None
    smooth_score = _coerce_float(run.get("smooth_score"))

    stability_components: list[float] = []
    if loss_downhill_pct is not None:
        stability_components.append(loss_downhill_pct)
    if loss_flip_rate is not None:
        stability_components.append(1.0 - loss_flip_rate)
    if loss_noise is not None:
        stability_components.append(1.0 / (1.0 + 4.0 * loss_noise))
    if return_uphill_pct is not None:
        stability_components.append(return_uphill_pct)
    if return_flip_rate is not None:
        stability_components.append(1.0 - return_flip_rate)
    if return_noise is not None:
        stability_components.append(1.0 / (1.0 + 4.0 * return_noise))
    if sortino_uphill_pct is not None:
        stability_components.append(sortino_uphill_pct)

    gap_score = _penalty_to_score(max(generalization_gap, 0.0) if generalization_gap is not None else None, scale=0.5)
    if gap_score is not None:
        stability_components.append(gap_score)

    drawdown_score = _penalty_to_score(drawdown_abs, scale=0.15)
    if drawdown_score is not None:
        stability_components.append(drawdown_score)

    if smooth_score is not None:
        stability_components.append(0.5 + 0.5 * math.tanh(smooth_score / 2.0))

    stability_score = None
    if stability_components:
        stability_score = sum(stability_components) / len(stability_components)

    return {
        "loss_downhill_pct": loss_downhill_pct,
        "loss_flip_rate": loss_flip_rate,
        "loss_noise": loss_noise,
        "return_uphill_pct": return_uphill_pct,
        "return_flip_rate": return_flip_rate,
        "return_noise": return_noise,
        "sortino_uphill_pct": sortino_uphill_pct,
        "generalization_gap": generalization_gap,
        "overfit_risk": max(generalization_gap, 0.0) if generalization_gap is not None else None,
        "stability_score": stability_score,
    }


def _resolve_metric_value(run: dict[str, Any], metric_name: str) -> float | None:
    metric_key = (metric_name or "val/return").strip()
    run_key = _METRIC_ALIASES.get(metric_key, metric_key)

    direct_value = _coerce_float(run.get(run_key))
    if direct_value is not None:
        return direct_value

    summary = run.get("summary", {})
    if isinstance(summary, dict):
        summary_value = _coerce_float(_pick(summary, metric_key, metric_key.replace("/", "_"), metric_key.replace("/", ".")))
        if summary_value is not None:
            return summary_value

    return None


def _metric_sorts_desc(metric_name: str) -> bool:
    name = (metric_name or "").lower()
    if "drawdown" in name:
        return True
    lower_is_better_tokens = ("loss", "gap", "risk", "noise", "flip_rate")
    return not any(token in name for token in lower_is_better_tokens)


def fetch_runs(
    wandb,
    project: str,
    entity: str | None,
    run_id: str | None,
    group: str | None,
    last_n: int,
    primary_metric: str = "val/return",
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
        summary = _object_to_dict(getattr(r, "summary", None))
        summary = _merge_local_summary(getattr(r, "id", None), summary)
        config = _extract_config(_object_to_dict(getattr(r, "config", None)))

        val_return = _best_val_return(summary)
        final_loss = _final_loss(summary)
        steps = _step_count(summary)
        duration = summary.get("_runtime") or None
        train_return = _pick(summary, "train/final_return", "train_return", "train/return")
        max_drawdown = _pick(summary, "val/max_drawdown", "max_drawdown")
        smooth_score = _pick(summary, "smooth_score", "trial/smooth_score", "val/smooth_score")

        # Fetch history for loss/entropy and train/validation trajectory signals.
        loss_curve: list[float] = []
        entropy_curve: list[float] = []
        val_return_curve: list[float] = []
        val_sortino_curve: list[float] = []
        try:
            history = list(
                r.history(
                    keys=[
                        "train/policy_loss",
                        "train/entropy",
                        "val/return",
                        "val_return",
                        "eval_return",
                        "train/return",
                        "train/episode_return_mean",
                        "val/sortino",
                        "val_sortino",
                        "eval_sortino",
                    ],
                    samples=200,
                )
            )
            loss_curve = _sample_history(history, "train/policy_loss")
            entropy_curve = _sample_history(history, "train/entropy")
            val_return_curve = _sample_history_any(
                history,
                ["val/return", "val_return", "eval_return", "train/episode_return_mean", "train/return"],
            )
            val_sortino_curve = _sample_history_any(history, ["val/sortino", "val_sortino", "eval_sortino"])
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
                "summary": summary,
                "config": config,
                "val_return": val_return,
                "final_loss": final_loss,
                "steps": steps,
                "duration": duration,
                "train_return": _coerce_float(train_return),
                "max_drawdown": _coerce_float(max_drawdown),
                "smooth_score": _coerce_float(smooth_score),
                "loss_curve": loss_curve,
                "entropy_curve": entropy_curve,
                "val_return_curve": val_return_curve,
                "val_sortino_curve": val_sortino_curve,
                "sortino": sortino,
                "win_rate": win_rate,
            }
        )
        result[-1].update(compute_stability_metrics(result[-1]))

    sort_desc = _metric_sorts_desc(primary_metric)

    def sort_key(run: dict[str, Any]) -> float:
        value = _resolve_metric_value(run, primary_metric)
        if value is None:
            return float("-inf") if sort_desc else float("inf")
        return value

    result.sort(key=sort_key, reverse=sort_desc)
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

    metric_label = primary_metric or "val/return"
    metric_is_val_return = metric_label.strip().lower() in {"val/return", "val_return", "best_val_return"}

    # --- Summary table ---
    if metric_is_val_return:
        header = "| Run | Config | Best Val Return | Stability | Gap | Final Loss | Steps | Duration |"
        sep = "|-----|--------|----------------|-----------|-----|------------|-------|----------|"
    else:
        header = f"| Run | Config | {metric_label} | Val Return | Stability | Gap | Final Loss | Steps | Duration |"
        sep = "|-----|--------|----------------|------------|-----------|-----|------------|-------|----------|"
    lines.append(header)
    lines.append(sep)

    best_idx = next((i for i, _ in enumerate(runs)), None)

    for i, r in enumerate(runs):
        vr_str = _pct(r["val_return"])
        primary_value = _resolve_metric_value(r, metric_label)
        primary_str = _fmt_metric_value(metric_label, primary_value)
        stability_str = _fmt_metric_value("stability_score", _coerce_float(r.get("stability_score")))
        gap_str = _fmt_metric_value("generalization_gap", _coerce_float(r.get("generalization_gap")))
        if i == best_idx:
            primary_str = f"**{primary_str}**"
            if metric_is_val_return:
                vr_str = primary_str

        loss_str = f"{r['final_loss']:.3f}" if r["final_loss"] is not None else "—"
        steps_str = f"{r['steps'] / 1e6:.1f}M" if r["steps"] is not None else "—"
        dur_str = _fmt_duration(r["duration"])
        cfg_str = _fmt_config(r["config"])
        if metric_is_val_return:
            lines.append(
                f"| {r['name']} | {cfg_str} | {vr_str} | {stability_str} | {gap_str} | {loss_str} | {steps_str} | {dur_str} |"
            )
        else:
            lines.append(
                f"| {r['name']} | {cfg_str} | {primary_str} | {vr_str} | {stability_str} | {gap_str} | {loss_str} | {steps_str} | {dur_str} |"
            )

    lines.append("")

    # --- Best run detail ---
    if best_idx is not None:
        best = runs[best_idx]
        lines.append(f"### Best Run: {best['name']}")

        if not metric_is_val_return:
            primary_value = _resolve_metric_value(best, metric_label)
            lines.append(f"- **Primary metric ({metric_label})**: {_fmt_metric_value(metric_label, primary_value)}")

        vr = _pct(best["val_return"])
        sortino_str = f" | Sortino: {best['sortino']:.2f}" if best["sortino"] is not None else ""
        wr_str = f" | Win rate: {best['win_rate'] * 100:.0f}%" if best["win_rate"] is not None else ""
        lines.append(f"- **Val return**: {vr}{sortino_str}{wr_str}")
        if best.get("stability_score") is not None:
            lines.append(f"- **Stability score**: {best['stability_score']:.3f}")
        if best.get("generalization_gap") is not None:
            lines.append(f"- **Train/val gap**: {_pct(best['generalization_gap'])}")
        if best.get("smooth_score") is not None:
            lines.append(f"- **Market smooth score**: {best['smooth_score']:.3f}")

        if best["loss_curve"]:
            lines.append(f"- **Train loss curve** (sampled): {_fmt_curve(best['loss_curve'])}")

        if best.get("val_return_curve"):
            lines.append(f"- **Return curve** (sampled): {_fmt_curve(best['val_return_curve'])}")

        if best.get("val_sortino_curve"):
            lines.append(f"- **Sortino curve** (sampled): {_fmt_curve(best['val_sortino_curve'])}")

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
        primary_metric=args.metric,
    )

    if args.output_format == "json":
        print(format_json(runs, args.project, args.entity))
    else:
        print(format_markdown(runs, args.project, args.entity, args.metric))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
