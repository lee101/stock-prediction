#!/usr/bin/env python3
"""Build a stage-wise Binance trading pipeline scorecard from existing artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COMMON_QUOTES = ("FDUSD", "USDT", "USDC", "BUSD", "USD")


def canonical_asset(symbol: str) -> str:
    token = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
    if not token:
        return token
    for quote in COMMON_QUOTES:
        if token.endswith(quote) and len(token) > len(quote):
            return token[: -len(quote)]
    return token


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_forecast_artifacts(paths: list[Path], target_assets: set[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        metrics = data.get("metrics", []) if isinstance(data, dict) else []
        if not isinstance(metrics, list):
            raise ValueError(f"Forecast artifact must contain a metrics list: {path}")
        local_rows: list[dict[str, Any]] = []
        for item in metrics:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper()
            asset = canonical_asset(symbol)
            mae_percent = _safe_float(item.get("mae_percent"))
            if not symbol or mae_percent is None:
                continue
            row = {
                "symbol": symbol,
                "asset": asset,
                "horizon_hours": int(item.get("horizon_hours", 0)),
                "mae_percent": mae_percent,
                "count": int(item.get("count", 0)),
                "source_path": str(path),
            }
            rows.append(row)
            local_rows.append(row)
        local_assets = sorted({row["asset"] for row in local_rows})
        artifacts.append(
            {
                "path": str(path),
                "record_count": len(local_rows),
                "assets": local_assets,
                "mean_mae_percent": _mean([row["mae_percent"] for row in local_rows]),
            }
        )

    by_asset: dict[str, dict[str, Any]] = {}
    grouped_assets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_horizons: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_assets[row["asset"]].append(row)
        grouped_horizons[int(row["horizon_hours"])].append(row)

    for asset, asset_rows in sorted(grouped_assets.items()):
        by_asset[asset] = {
            "mean_mae_percent": _mean([row["mae_percent"] for row in asset_rows]),
            "best_mae_percent": min(row["mae_percent"] for row in asset_rows),
            "worst_mae_percent": max(row["mae_percent"] for row in asset_rows),
            "horizons": sorted({int(row["horizon_hours"]) for row in asset_rows}),
        }

    by_horizon = []
    for horizon, horizon_rows in sorted(grouped_horizons.items()):
        by_horizon.append(
            {
                "horizon_hours": int(horizon),
                "mean_mae_percent": _mean([row["mae_percent"] for row in horizon_rows]),
                "assets": sorted({row["asset"] for row in horizon_rows}),
            }
        )

    covered_assets = set(grouped_assets)
    missing_assets = sorted(target_assets - covered_assets)
    status = "missing"
    if rows:
        status = "partial" if missing_assets else "covered"

    return {
        "status": status,
        "artifacts": artifacts,
        "record_count": len(rows),
        "covered_assets": sorted(covered_assets),
        "missing_target_assets": missing_assets,
        "mean_mae_percent": _mean([row["mae_percent"] for row in rows]),
        "best_mae_percent": min((row["mae_percent"] for row in rows), default=None),
        "worst_mae_percent": max((row["mae_percent"] for row in rows), default=None),
        "by_asset": by_asset,
        "by_horizon": by_horizon,
    }


def _parse_policy_dict(path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    if {"sortino", "total_return"} <= set(data):
        total_return = _safe_float(data.get("total_return"))
        sortino = _safe_float(data.get("sortino"))
        if total_return is not None and sortino is not None:
            return {
                "kind": "selector_sweep_best",
                "path": str(path),
                "sortino": sortino,
                "total_return_pct": total_return * 100.0,
                "details": {
                    "intensity_scale": _safe_float(data.get("intensity_scale")),
                    "min_edge": _safe_float(data.get("min_edge")),
                    "risk_weight": _safe_float(data.get("risk_weight")),
                },
            }

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        test_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else metrics
        total_return = _safe_float(test_metrics.get("total_return"))
        sortino = _safe_float(test_metrics.get("sortino"))
        if total_return is not None or sortino is not None:
            return {
                "kind": "simulation_metrics",
                "path": str(path),
                "sortino": sortino,
                "total_return_pct": None if total_return is None else total_return * 100.0,
                "details": {"label": data.get("run_name") or path.name},
            }
    return None


def _parse_policy_list(path: Path, data: list[Any]) -> dict[str, Any] | None:
    rows = [row for row in data if isinstance(row, dict)]
    if not rows:
        return None
    first = rows[0]
    if {"min_sortino", "mean_sortino", "mean_return_pct"} <= set(first):
        return {
            "kind": "meta_search_best",
            "path": str(path),
            "min_sortino": _safe_float(first.get("min_sortino")),
            "mean_sortino": _safe_float(first.get("mean_sortino")),
            "mean_return_pct": _safe_float(first.get("mean_return_pct")),
            "mean_dd_pct": _safe_float(first.get("mean_dd_pct")),
            "beats": int(first.get("beats", 0)),
            "details": {
                "doge": first.get("doge"),
                "aave": first.get("aave"),
                "metric": first.get("metric"),
                "lookback_days": first.get("lookback_days"),
            },
        }
    return None


def parse_policy_artifacts(paths: list[Path]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    best_meta: dict[str, Any] | None = None
    best_selector: dict[str, Any] | None = None
    for path in paths:
        data = _load_json(path)
        parsed: dict[str, Any] | None = None
        if isinstance(data, dict):
            parsed = _parse_policy_dict(path, data)
        elif isinstance(data, list):
            parsed = _parse_policy_list(path, data)
        if parsed is None:
            raise ValueError(f"Unsupported policy artifact format: {path}")
        artifacts.append(parsed)
        if parsed["kind"] == "meta_search_best":
            if best_meta is None:
                best_meta = parsed
            else:
                left = (
                    _safe_float(parsed.get("min_sortino")) or float("-inf"),
                    int(parsed.get("beats", 0)),
                    _safe_float(parsed.get("mean_sortino")) or float("-inf"),
                    _safe_float(parsed.get("mean_return_pct")) or float("-inf"),
                )
                right = (
                    _safe_float(best_meta.get("min_sortino")) or float("-inf"),
                    int(best_meta.get("beats", 0)),
                    _safe_float(best_meta.get("mean_sortino")) or float("-inf"),
                    _safe_float(best_meta.get("mean_return_pct")) or float("-inf"),
                )
                if left > right:
                    best_meta = parsed
        else:
            if best_selector is None:
                best_selector = parsed
            else:
                left = (
                    _safe_float(parsed.get("sortino")) or float("-inf"),
                    _safe_float(parsed.get("total_return_pct")) or float("-inf"),
                )
                right = (
                    _safe_float(best_selector.get("sortino")) or float("-inf"),
                    _safe_float(best_selector.get("total_return_pct")) or float("-inf"),
                )
                if left > right:
                    best_selector = parsed

    status = "missing"
    if best_meta is not None:
        min_sortino = _safe_float(best_meta.get("min_sortino")) or 0.0
        beats = int(best_meta.get("beats", 0))
        status = "strong" if min_sortino >= 0.15 and beats >= 3 else "present"
    elif best_selector is not None:
        total_return_pct = _safe_float(best_selector.get("total_return_pct")) or 0.0
        sortino = _safe_float(best_selector.get("sortino")) or 0.0
        status = "strong" if total_return_pct > 0.0 and sortino > 1.0 else "present"

    return {
        "status": status,
        "artifacts": artifacts,
        "best_meta": best_meta,
        "best_selector": best_selector,
    }


def parse_parity_json_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for path in paths:
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Parity JSON artifact must be an object: {path}")
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        parsed.append(
            {
                "kind": "parity_json",
                "path": str(path),
                "exact_row_ratio": _safe_float(summary.get("exact_row_ratio")),
                "hourly_abs_count_delta_total": _safe_float(summary.get("hourly_abs_count_delta_total")),
                "live_filled_orders": _safe_float(summary.get("live_filled_orders")),
                "sim_trades": _safe_float(summary.get("sim_trades")),
            }
        )
    return parsed


def parse_parity_markdown_artifacts(paths: list[Path], target_assets: set[str]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    ratio_re = re.compile(r"(\d+)\s*/\s*(\d+)\s*\((\d+)%\)")
    header_re = re.compile(r"^###\s+(.+?)\s*$")
    for path in paths:
        current_asset: str | None = None
        per_asset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            header_match = header_re.match(line)
            if header_match:
                current_asset = canonical_asset(header_match.group(1))
                continue
            if current_asset is None:
                continue
            ratio_match = ratio_re.search(line)
            if not ratio_match:
                continue
            numerator = int(ratio_match.group(1))
            denominator = int(ratio_match.group(2))
            percent = int(ratio_match.group(3))
            per_asset[current_asset].append(
                {
                    "description": re.sub(r"\*+", "", line),
                    "matches": numerator,
                    "total": denominator,
                    "ratio": numerator / float(denominator) if denominator else None,
                    "percent": percent,
                }
            )
        parsed.append(
            {
                "kind": "parity_markdown",
                "path": str(path),
                "per_asset": {
                    asset: {
                        "runs": runs,
                        "best_ratio": max((run["ratio"] for run in runs if run["ratio"] is not None), default=None),
                        "latest_ratio": runs[-1]["ratio"] if runs else None,
                    }
                    for asset, runs in sorted(per_asset.items())
                },
                "missing_target_assets": sorted(target_assets - set(per_asset)),
            }
        )
    return parsed


def summarize_parity(
    json_artifacts: list[dict[str, Any]],
    markdown_artifacts: list[dict[str, Any]],
    target_assets: set[str],
) -> dict[str, Any]:
    best_ratio_by_asset: dict[str, float] = {}
    latest_ratio_by_asset: dict[str, float] = {}
    for artifact in markdown_artifacts:
        per_asset = artifact.get("per_asset", {})
        for asset, summary in per_asset.items():
            best_ratio = _safe_float(summary.get("best_ratio"))
            latest_ratio = _safe_float(summary.get("latest_ratio"))
            if best_ratio is not None:
                current = best_ratio_by_asset.get(asset)
                if current is None or best_ratio > current:
                    best_ratio_by_asset[asset] = best_ratio
            if latest_ratio is not None:
                latest_ratio_by_asset[asset] = latest_ratio

    exact_row_ratios = [
        artifact["exact_row_ratio"]
        for artifact in json_artifacts
        if _safe_float(artifact.get("exact_row_ratio")) is not None
    ]

    status = "missing"
    missing_targets = sorted(target_assets - set(best_ratio_by_asset))
    if best_ratio_by_asset or exact_row_ratios:
        status = "partial"
        if not missing_targets:
            lowest_best_ratio = min(best_ratio_by_asset.values()) if best_ratio_by_asset else None
            if lowest_best_ratio is not None and lowest_best_ratio >= 0.5:
                status = "good"
            else:
                status = "needs_work"
        else:
            status = "needs_work"

    return {
        "status": status,
        "json_artifacts": json_artifacts,
        "markdown_artifacts": markdown_artifacts,
        "best_ratio_by_asset": dict(sorted(best_ratio_by_asset.items())),
        "latest_ratio_by_asset": dict(sorted(latest_ratio_by_asset.items())),
        "mean_exact_row_ratio": _mean([ratio for ratio in exact_row_ratios if ratio is not None]),
        "missing_target_assets": missing_targets,
    }


def parse_log_artifacts(paths: list[Path]) -> dict[str, Any]:
    artifact_summaries: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    entry_skip_reasons: Counter[str] = Counter()
    latest_state_by_model: dict[str, dict[str, Any]] = {}
    model_event_counts: Counter[str] = Counter()
    negative_equity_states = 0
    max_leverage = 0.0

    for path in paths:
        local_event_counts: Counter[str] = Counter()
        local_entry_skip_reasons: Counter[str] = Counter()
        local_latest_state_by_model: dict[str, dict[str, Any]] = {}
        local_model_event_counts: Counter[str] = Counter()
        local_negative_equity_states = 0
        local_max_leverage = 0.0
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        for line in lines:
            row = json.loads(line)
            event = str(row.get("event", "unknown"))
            model = str(row.get("model") or row.get("symbol") or "unknown")
            local_event_counts[event] += 1
            event_counts[event] += 1
            local_model_event_counts[model] += 1
            model_event_counts[model] += 1
            if event == "entry_skip":
                reason = str(row.get("reason") or "unknown")
                local_entry_skip_reasons[reason] += 1
                entry_skip_reasons[reason] += 1
            if event == "state":
                equity = _safe_float(row.get("equity"))
                leverage = _safe_float(row.get("leverage")) or 0.0
                if equity is not None and equity < 0:
                    local_negative_equity_states += 1
                    negative_equity_states += 1
                local_max_leverage = max(local_max_leverage, leverage)
                max_leverage = max(max_leverage, leverage)
                local_latest_state_by_model[model] = {
                    "equity": equity,
                    "leverage": leverage,
                    "in_position": bool(row.get("in_position", False)),
                    "hours_held": _safe_float(row.get("hours_held")),
                }
                latest_state_by_model[model] = local_latest_state_by_model[model]
        artifact_summaries.append(
            {
                "path": str(path),
                "line_count": len(lines),
                "event_counts": dict(sorted(local_event_counts.items())),
                "model_event_counts": dict(sorted(local_model_event_counts.items())),
                "entry_skip_reasons": dict(sorted(local_entry_skip_reasons.items())),
                "negative_equity_states": local_negative_equity_states,
                "max_leverage": local_max_leverage,
                "latest_state_by_model": local_latest_state_by_model,
            }
        )

    status = "missing"
    if artifact_summaries:
        status = "healthy"
        if negative_equity_states > 0 or entry_skip_reasons.get("no_equity", 0) > 0:
            status = "blocked"
        elif event_counts.get("exit_attempt", 0) > 25:
            status = "monitor"

    return {
        "status": status,
        "artifacts": artifact_summaries,
        "event_counts": dict(sorted(event_counts.items())),
        "model_event_counts": dict(sorted(model_event_counts.items())),
        "entry_skip_reasons": dict(sorted(entry_skip_reasons.items())),
        "negative_equity_states": negative_equity_states,
        "max_leverage": max_leverage,
        "latest_state_by_model": dict(sorted(latest_state_by_model.items())),
    }


def build_recommendations(report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    forecast = report["stages"]["forecast"]
    policy = report["stages"]["policy"]
    parity = report["stages"]["parity"]
    execution = report["stages"]["execution"]

    missing_assets = forecast.get("missing_target_assets", [])
    if missing_assets:
        recommendations.append(
            "Produce symbol-level DOGE/AAVE Chronos2 MAE artifacts before the next policy/meta sweep so forecast changes can be attributed downstream."
        )
    if parity.get("status") == "needs_work":
        recommendations.append(
            "Prioritize meta-aware replay against logged signals/orders/fills; single-symbol replay is still underestimating the AAVE mismatch."
        )
    if execution.get("status") == "blocked":
        recommendations.append(
            "Treat account-state and equity-sync issues as production blockers before expanding AAVE allocation or leverage."
        )
    if policy.get("status") == "strong" and parity.get("status") in {"needs_work", "partial"}:
        recommendations.append(
            "Do not spend the next cycle only on new policy checkpoints; the current bottleneck is replay and simulator/live alignment."
        )
    if not recommendations and policy.get("status") in {"present", "strong"}:
        recommendations.append(
            "Use the current policy stack as the baseline and optimize the forecast layer on a latency/MAE frontier rather than purely chasing more model complexity."
        )
    return recommendations


def build_scorecard(
    *,
    name: str,
    target_symbols: list[str],
    forecast_paths: list[Path],
    policy_paths: list[Path],
    parity_json_paths: list[Path],
    parity_md_paths: list[Path],
    log_paths: list[Path],
) -> dict[str, Any]:
    target_assets = {canonical_asset(symbol) for symbol in target_symbols if canonical_asset(symbol)}
    forecast = parse_forecast_artifacts(forecast_paths, target_assets)
    policy = parse_policy_artifacts(policy_paths)
    parity = summarize_parity(
        parse_parity_json_artifacts(parity_json_paths),
        parse_parity_markdown_artifacts(parity_md_paths, target_assets),
        target_assets,
    )
    execution = parse_log_artifacts(log_paths)

    overall_status = "measurement_gap"
    if execution["status"] == "blocked":
        overall_status = "execution_blocker"
    elif parity["status"] == "needs_work":
        overall_status = "alignment_gap"
    elif forecast["status"] == "missing":
        overall_status = "measurement_gap"
    elif policy["status"] == "strong":
        overall_status = "ready_for_targeted_experiments"
    elif policy["status"] == "present":
        overall_status = "baseline_present"

    report = {
        "name": name,
        "targets": target_symbols,
        "target_assets": sorted(target_assets),
        "overall_status": overall_status,
        "stages": {
            "forecast": forecast,
            "policy": policy,
            "parity": parity,
            "execution": execution,
        },
    }
    report["recommendations"] = build_recommendations(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    forecast = report["stages"]["forecast"]
    policy = report["stages"]["policy"]
    parity = report["stages"]["parity"]
    execution = report["stages"]["execution"]
    lines = [
        f"# {report['name']}",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Targets: `{', '.join(report['targets'])}`",
        "",
        "## Forecast Stage",
        f"- Status: `{forecast['status']}`",
        f"- Covered target assets: `{', '.join(forecast['covered_assets']) if forecast['covered_assets'] else 'none'}`",
        f"- Missing target assets: `{', '.join(forecast['missing_target_assets']) if forecast['missing_target_assets'] else 'none'}`",
    ]
    if forecast.get("mean_mae_percent") is not None:
        lines.append(f"- Mean MAE % across provided artifacts: `{forecast['mean_mae_percent']:.4f}`")

    lines.extend(
        [
            "",
            "## Policy Stage",
            f"- Status: `{policy['status']}`",
        ]
    )
    if policy.get("best_meta"):
        best_meta = policy["best_meta"]
        lines.append(
            "- Best meta artifact: "
            f"`min_sortino={best_meta.get('min_sortino')}` "
            f"`mean_sortino={best_meta.get('mean_sortino')}` "
            f"`mean_return_pct={best_meta.get('mean_return_pct')}` "
            f"`beats={best_meta.get('beats')}`"
        )
    if policy.get("best_selector"):
        best_selector = policy["best_selector"]
        lines.append(
            "- Best selector artifact: "
            f"`sortino={best_selector.get('sortino')}` "
            f"`total_return_pct={best_selector.get('total_return_pct')}`"
        )

    lines.extend(
        [
            "",
            "## Parity Stage",
            f"- Status: `{parity['status']}`",
            f"- Missing target assets: `{', '.join(parity['missing_target_assets']) if parity['missing_target_assets'] else 'none'}`",
        ]
    )
    if parity.get("mean_exact_row_ratio") is not None:
        lines.append(f"- Mean exact row ratio: `{parity['mean_exact_row_ratio']:.4f}`")
    if parity.get("best_ratio_by_asset"):
        for asset, ratio in parity["best_ratio_by_asset"].items():
            lines.append(f"- Best replay match for `{asset}`: `{ratio:.2%}`")

    lines.extend(
        [
            "",
            "## Execution Stage",
            f"- Status: `{execution['status']}`",
            f"- Negative equity states: `{execution['negative_equity_states']}`",
            f"- Max leverage observed: `{execution['max_leverage']:.4f}`",
            f"- Model event counts: `{json.dumps(execution['model_event_counts'], sort_keys=True)}`",
            f"- Entry skip reasons: `{json.dumps(execution['entry_skip_reasons'], sort_keys=True)}`",
        ]
    )

    if report.get("recommendations"):
        lines.extend(["", "## Recommendations"])
        for recommendation in report["recommendations"]:
            lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Binance trading pipeline scorecard from existing artifacts.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-symbols", default="DOGEUSDT,AAVEUSDT")
    parser.add_argument("--forecast-json", nargs="*", default=[])
    parser.add_argument("--policy-json", nargs="*", default=[])
    parser.add_argument("--parity-json", nargs="*", default=[])
    parser.add_argument("--parity-md", nargs="*", default=[])
    parser.add_argument("--log-jsonl", nargs="*", default=[])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    target_symbols = [token.strip().upper() for token in str(args.target_symbols).split(",") if token.strip()]
    report = build_scorecard(
        name=args.name,
        target_symbols=target_symbols,
        forecast_paths=[Path(path) for path in args.forecast_json],
        policy_paths=[Path(path) for path in args.policy_json],
        parity_json_paths=[Path(path) for path in args.parity_json],
        parity_md_paths=[Path(path) for path in args.parity_md],
        log_paths=[Path(path) for path in args.log_jsonl],
    )

    json_output = json.dumps(report, indent=2)
    markdown_output = render_markdown(report)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json_output + "\n")
    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(markdown_output)

    if not args.output_json:
        print(json_output)
    if not args.output_md:
        print()
        print(markdown_output)


if __name__ == "__main__":
    main()
