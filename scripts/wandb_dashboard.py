#!/usr/bin/env python3
"""Print a rich comparison dashboard of RL trading experiments from WandB.

Extends wandb_metrics_reader.py with:
- Side-by-side comparison of best configs
- Training curve summary (loss, val_return over time)
- Architecture breakdown (mlp, transformer, gru, depth_recurrence)
- Optimizer comparison (adamw, muon)
- Data split analysis (mixed23, mixed32, crypto12, fdusd, etc.)

Usage:
  python scripts/wandb_dashboard.py --project stock --last-n-runs 20
  python scripts/wandb_dashboard.py --project stock --group autoresearch_20260321
  python scripts/wandb_dashboard.py --project stock --compare-archs
  python scripts/wandb_dashboard.py --project stock --compare-optimizers
  python scripts/wandb_dashboard.py --project stock --compare-datasets
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_scripts_dir = str(REPO / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Re-use shared helpers from wandb_metrics_reader to avoid duplication
import wandb_metrics_reader as _reader  # noqa: E402

_import_wandb = _reader._import_wandb
_check_api_key = _reader._check_api_key
_pick = _reader._pick
_sample_history = _reader._sample_history
_fmt_duration = _reader._fmt_duration
_fmt_curve = _reader._fmt_curve
_pct = _reader._pct


# ---------------------------------------------------------------------------
# Extended config keys and detection helpers
# ---------------------------------------------------------------------------

_CONFIG_KEYS = [
    "hidden_size", "lr", "anneal_lr", "ent_coef", "seed",
    "arch", "trade_penalty", "fill_slippage_bps", "fee_rate",
    "weight_decay", "obs_norm", "anneal_ent", "anneal_clip",
]

_ARCH_KEYWORDS = {
    "transformer": ["transformer", "attn", "attention"],
    "gru": ["gru", "recurrent", "rnn", "lstm"],
    "depth_recurrence": ["depth_recurrence", "depth_rec"],
    "resmlp": ["resmlp", "res_mlp", "residual"],
    "mlp": ["mlp"],  # fallback — checked last
}


def _detect_arch(run_dict: dict) -> str:
    cfg = run_dict.get("config", {})
    arch = cfg.get("arch", "").lower()
    if arch:
        return arch
    name = run_dict.get("name", "").lower()
    for arch_key, keywords in _ARCH_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return arch_key
    return "mlp"


def _detect_optimizer(run_dict: dict) -> str:
    cfg = run_dict.get("config", {})
    opt = str(cfg.get("optimizer", "")).lower()
    if opt:
        return opt
    if "muon" in run_dict.get("name", "").lower():
        return "muon"
    return "adamw"


def _detect_dataset(run_dict: dict) -> str:
    cfg = run_dict.get("config", {})
    data_path = str(cfg.get("data_path", "") or cfg.get("data-path", "")).lower()
    haystack = data_path + " " + run_dict.get("name", "").lower()
    for tag in ["fdusd", "mixed32", "mixed23", "crypto12", "crypto8", "crypto6", "crypto4"]:
        if tag in haystack:
            return tag
    if data_path:
        return Path(data_path).stem
    return "unknown"


def _extract_config(run_config: dict) -> dict:
    result = {}
    for k in _CONFIG_KEYS:
        if k in run_config:
            val = run_config[k]
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            result[k] = val
    return result


def _fmt_config(cfg: dict, keys: list[str] | None = None) -> str:
    short = {"hidden_size": "h", "anneal_lr": "ann", "ent_coef": "ent",
             "fill_slippage_bps": "slip", "trade_penalty": "tpen",
             "fee_rate": "fee", "weight_decay": "wd"}
    parts = []
    for k in keys or list(cfg.keys()):
        if k not in cfg:
            continue
        v = cfg[k]
        sk = short.get(k, k)
        parts.append(f"{sk}={v:.0e}" if isinstance(v, float) and v < 0.01 else f"{sk}={v}")
    return " ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Data fetching (extended over wandb_metrics_reader.fetch_runs)
# ---------------------------------------------------------------------------


def fetch_runs(
    wandb,
    project: str,
    entity: str | None,
    run_id: str | None,
    group: str | None,
    last_n: int,
) -> list[dict[str, Any]]:
    """Fetch run metadata from WandB, extended with arch/optimizer/dataset tags."""
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    if run_id:
        run_path = f"{project_path}/{run_id}"
        try:
            runs_raw = [api.run(run_path)]
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
        config_raw = dict(r.config) if r.config else {}
        config = _extract_config(config_raw)

        val_return_raw = _pick(summary, "val/return", "val_return", "best_val_return")
        val_return = float(val_return_raw) if val_return_raw is not None else None

        final_loss_raw = _pick(summary, "train/policy_loss", "policy_loss", "train/loss", "loss")
        final_loss = float(final_loss_raw) if final_loss_raw is not None else None

        steps_raw = _pick(summary, "global_step", "_step", "train/global_step", "step")
        steps = int(steps_raw) if steps_raw is not None else None

        loss_curve: list[float] = []
        val_return_curve: list[float] = []
        try:
            history = list(r.history(
                keys=["train/policy_loss", "train/return", "train/episode_return_mean"],
                samples=200,
            ))
            loss_curve = _sample_history(history, "train/policy_loss")
            val_return_curve = (
                _sample_history(history, "train/episode_return_mean")
                or _sample_history(history, "train/return")
            )
        except Exception:
            pass

        run_dict: dict[str, Any] = {
            "id": r.id,
            "name": r.name,
            "state": getattr(r, "state", "unknown"),
            "group": getattr(r, "group", None),
            "config": config,
            "config_raw": config_raw,
            "val_return": val_return,
            "final_loss": final_loss,
            "steps": steps,
            "duration": summary.get("_runtime") or None,
            "loss_curve": loss_curve,
            "val_return_curve": val_return_curve,
            "sortino": _pick(summary, "val/sortino", "sortino"),
            "win_rate": _pick(summary, "val/win_rate", "win_rate"),
            "max_drawdown": _pick(summary, "val/max_drawdown", "max_drawdown"),
            "num_trades": _pick(summary, "val/num_trades", "num_trades"),
        }
        run_dict["arch"] = _detect_arch(run_dict)
        run_dict["optimizer"] = _detect_optimizer(run_dict)
        run_dict["dataset"] = _detect_dataset(run_dict)
        result.append(run_dict)

    result.sort(
        key=lambda r: r["val_return"] if r["val_return"] is not None else float("-inf"),
        reverse=True,
    )
    return result


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _group_runs(runs: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        groups[r.get(key, "unknown")].append(r)
    return dict(groups)


def _best_in_group(group_runs: list[dict]) -> dict | None:
    filtered = [r for r in group_runs if r.get("val_return") is not None]
    if not filtered:
        return None
    return max(filtered, key=lambda r: r["val_return"])


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------


def format_summary_table(runs: list[dict]) -> str:
    header = ("| Run | Arch | Dataset | Config | Best Val Return"
              " | Sortino | Win Rate | Final Loss | Steps | Duration |")
    sep = ("|-----|------|---------|--------|----------------|"
           "---------|----------|------------|-------|----------|")
    lines = [header, sep]
    for r in runs:
        vr = _pct(r["val_return"])
        sr = f"{r['sortino']:.2f}" if r.get("sortino") is not None else "—"
        wr = f"{r['win_rate'] * 100:.0f}%" if r.get("win_rate") is not None else "—"
        loss = f"{r['final_loss']:.3f}" if r.get("final_loss") is not None else "—"
        steps = f"{r['steps'] / 1e6:.1f}M" if r.get("steps") is not None else "—"
        dur = _fmt_duration(r.get("duration"))
        cfg = _fmt_config(r["config"], keys=["hidden_size", "lr", "ent_coef", "fill_slippage_bps"])
        lines.append(
            f"| {r['name']} | {r.get('arch', '—')} | {r.get('dataset', '—')} "
            f"| {cfg} | {vr} | {sr} | {wr} | {loss} | {steps} | {dur} |"
        )
    return "\n".join(lines)


def format_group_comparison(runs: list[dict], group_key: str, group_label: str) -> str:
    groups = _group_runs(runs, group_key)
    if not groups:
        return f"*No runs to group by {group_label}.*"

    lines = [f"### Comparison by {group_label}", ""]
    header = (f"| {group_label} | Best Run | Val Return | Sortino"
              " | Win Rate | Convergence (train return curve) |")
    sep = (f"|{'-' * (len(group_label) + 2)}|----------|------------|---------|"
           "----------|----------------------------------|")
    lines += [header, sep]

    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (b := _best_in_group(kv[1])) and b["val_return"] or float("-inf"),
        reverse=True,
    )
    for group_name, grp_runs in sorted_groups:
        best = _best_in_group(grp_runs)
        if best is None:
            lines.append(f"| {group_name} | — | — | — | — | — |")
            continue
        vr = _pct(best["val_return"])
        sr = f"{best['sortino']:.2f}" if best.get("sortino") is not None else "—"
        wr = f"{best['win_rate'] * 100:.0f}%" if best.get("win_rate") is not None else "—"
        curve = _fmt_curve(best.get("val_return_curve") or [])
        lines.append(f"| {group_name} | {best['name']} | {vr} | {sr} | {wr} | {curve} |")

    lines.append("")
    lines.append(f"*{len(groups)} group(s), {len(runs)} total run(s)*")
    return "\n".join(lines)


def format_best_run_detail(best: dict) -> str:
    lines = [f"### Best Run: {best['name']}", ""]
    lines.append(f"- **Val return**: {_pct(best['val_return'])}")
    if best.get("sortino") is not None:
        lines.append(f"- **Sortino**: {best['sortino']:.2f}")
    if best.get("win_rate") is not None:
        lines.append(f"- **Win rate**: {best['win_rate'] * 100:.0f}%")
    if best.get("max_drawdown") is not None:
        lines.append(f"- **Max drawdown**: {_pct(best['max_drawdown'])}")
    if best.get("num_trades") is not None:
        lines.append(f"- **Avg trades/episode**: {best['num_trades']:.1f}")
    if best.get("loss_curve"):
        lines.append(f"- **Train loss curve** (sampled): {_fmt_curve(best['loss_curve'])}")
    if best.get("val_return_curve"):
        lines.append(f"- **Train return curve** (sampled): {_fmt_curve(best['val_return_curve'])}")
    cfg = best.get("config", {})
    if cfg:
        lines.append(f"- **Config**: {_fmt_config(cfg)}")
    lines.append(f"- **Architecture**: {best.get('arch', '—')}")
    lines.append(f"- **Optimizer**: {best.get('optimizer', '—')}")
    lines.append(f"- **Dataset**: {best.get('dataset', '—')}")
    if best.get("steps") is not None:
        lines.append(f"- **Total steps**: {best['steps'] / 1e6:.1f}M")
    if best.get("duration") is not None:
        lines.append(f"- **Duration**: {_fmt_duration(best['duration'])}")
    return "\n".join(lines)


def format_dashboard(
    runs: list[dict],
    project: str,
    entity: str | None,
    *,
    compare_archs: bool = False,
    compare_optimizers: bool = False,
    compare_datasets: bool = False,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    proj_label = f"{entity}/{project}" if entity else project
    n = len(runs)

    lines: list[str] = []
    lines.append(f"## WandB Trading Dashboard: {proj_label} ({n} run{'s' if n != 1 else ''})")
    lines.append(f"*Fetched: {now}*")
    lines.append("")

    if not runs:
        lines.append("*No runs found.*")
        return "\n".join(lines)

    runs = sorted(
        runs,
        key=lambda r: r["val_return"] if r.get("val_return") is not None else float("-inf"),
        reverse=True,
    )

    lines.append("### All Runs")
    lines.append("")
    lines.append(format_summary_table(runs))
    lines.append("")

    best_runs = [r for r in runs if r.get("val_return") is not None]
    if best_runs:
        lines.append(format_best_run_detail(best_runs[0]))
        lines.append("")

    if compare_archs:
        lines.append(format_group_comparison(runs, "arch", "Architecture"))
        lines.append("")

    if compare_optimizers:
        lines.append(format_group_comparison(runs, "optimizer", "Optimizer"))
        lines.append("")

    if compare_datasets:
        lines.append(format_group_comparison(runs, "dataset", "Dataset"))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a rich WandB trading dashboard as LLM-friendly markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity/username")
    parser.add_argument("--run-id", default=None, help="Specific run ID to fetch")
    parser.add_argument("--group", default=None, help="Filter by run group")
    parser.add_argument(
        "--last-n-runs", type=int, default=20, metavar="N",
        help="How many recent runs to fetch",
    )
    parser.add_argument("--compare-archs", action="store_true",
                        help="Add architecture breakdown section")
    parser.add_argument("--compare-optimizers", action="store_true",
                        help="Add optimizer comparison section")
    parser.add_argument("--compare-datasets", action="store_true",
                        help="Add dataset/data-split comparison section")
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

    print(format_dashboard(
        runs,
        args.project,
        args.entity,
        compare_archs=args.compare_archs,
        compare_optimizers=args.compare_optimizers,
        compare_datasets=args.compare_datasets,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
