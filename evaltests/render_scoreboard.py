"""
Render the latest RL scoreboard into a Markdown table for quick reporting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCOREBOARD_JSON = Path("evaltests/rl_benchmark_results.json")
OUTPUT_MD = Path("evaltests/scoreboard.md")
HISTORY_JSON = Path("evaltests/scoreboard_history.json")


def load_results() -> Mapping[str, Any]:
    if not SCOREBOARD_JSON.exists():
        raise FileNotFoundError(f"{SCOREBOARD_JSON} not found. Run rl_benchmark_runner first.")
    return json.loads(SCOREBOARD_JSON.read_text(encoding="utf-8"))


def load_history() -> list[Mapping[str, Any]]:
    if not HISTORY_JSON.exists():
        return []
    try:
        data = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def save_history(history: list[Mapping[str, Any]]) -> None:
    HISTORY_JSON.write_text(json.dumps(history, indent=2), encoding="utf-8")


def compute_deltas(current: Mapping[str, Any], previous: Mapping[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    if not isinstance(previous, Mapping):
        return deltas
    cur_score = current.get("score")
    prev_score = previous.get("score")
    if isinstance(cur_score, (int, float)) and isinstance(prev_score, (int, float)):
        deltas["score"] = cur_score - prev_score
    cur_spd = current.get("score_per_day")
    prev_spd = previous.get("score_per_day")
    if isinstance(cur_spd, (int, float)) and isinstance(prev_spd, (int, float)):
        deltas["score_per_day"] = cur_spd - prev_spd
    return deltas


def render_markdown(data: Mapping[str, Any], timestamp: datetime) -> str:
    scoreboard = data.get("scoreboard", [])
    baseline = data.get("baseline", {})
    baseline_pnl = None
    trade_history = baseline.get("trade_history")
    if isinstance(trade_history, Mapping):
        baseline_pnl = trade_history.get("total_realized_pnl")

    lines = [
        "# RL Scoreboard",
        "",
        f"Generated: {timestamp.isoformat()}",
        "",
    ]
    if baseline_pnl is not None:
        lines.append(f"- Baseline production realised PnL: {baseline_pnl:,.2f}")
        lines.append("")

    header = "| Rank | Name | Module | Score | Score/day | ΔScore | Δ/day | xBaseline | Notes |"
    sep = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    lines.extend([header, sep])
    history = load_history()
    prev = history[-1] if history else {}
    prev_map = {entry.get("name"): entry for entry in prev.get("scoreboard", [])} if isinstance(prev, Mapping) else {}
    for idx, entry in enumerate(scoreboard, start=1):
        name = entry.get("name", "unknown")
        module = entry.get("module", "unknown")
        score = entry.get("score")
        per_day = entry.get("score_per_day")
        rel = entry.get("relative_to_baseline")
        details = entry.get("details", {})
        note = ""
        if isinstance(details, Mapping):
            if module == "differentiable_market":
                note = f"report_sharpe={details.get('report_sharpe')}"
            elif module == "pufferlibtraining":
                note = f"best_pair={details.get('best_pair')}"
            elif module == "gymrl":
                note_parts = []
                adr = details.get("average_daily_return")
                if isinstance(adr, (int, float)):
                    note_parts.append(f"avg_daily_return={adr:.4f}")
                guard_neg = details.get("guard_negative_hit_rate")
                guard_turn = details.get("guard_turnover_hit_rate")
                guard_draw = details.get("guard_drawdown_hit_rate")
                guard_bits = []
                if isinstance(guard_neg, (int, float)):
                    guard_bits.append(f"neg={guard_neg:.2f}")
                if isinstance(guard_turn, (int, float)):
                    guard_bits.append(f"turn={guard_turn:.2f}")
                if isinstance(guard_draw, (int, float)):
                    guard_bits.append(f"draw={guard_draw:.2f}")
                if guard_bits:
                    note_parts.append("guard(" + ", ".join(guard_bits) + ")")
                note = "; ".join(note_parts)
        score_str = f"{score:,.4f}" if isinstance(score, (int, float)) else "-"
        per_day_str = f"{per_day:,.4f}" if isinstance(per_day, (int, float)) else "-"
        rel_str = f"{rel:,.4f}" if isinstance(rel, (int, float)) else "-"
        prev_entry = prev_map.get(name)
        deltas = compute_deltas(entry, prev_entry if isinstance(prev_entry, Mapping) else {})
        delta_score = deltas.get("score")
        delta_day = deltas.get("score_per_day")
        delta_score_str = f"{delta_score:+.4f}" if isinstance(delta_score, (int, float)) else "-"
        delta_day_str = f"{delta_day:+.4f}" if isinstance(delta_day, (int, float)) else "-"
        lines.append(f"| {idx} | {name} | {module} | {score_str} | {per_day_str} | {delta_score_str} | {delta_day_str} | {rel_str} | {note} |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load_results()
    timestamp = datetime.now(timezone.utc)
    OUTPUT_MD.write_text(render_markdown(data, timestamp), encoding="utf-8")
    history = load_history()
    history.append(
        {
            "timestamp": timestamp.isoformat(),
            "scoreboard": data.get("scoreboard", []),
        }
    )
    save_history(history[-20:])  # keep last 20 snapshots
    print(f"Scoreboard written to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
