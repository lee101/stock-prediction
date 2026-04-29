"""Build a side-by-side comparison JSON between the cross-attention transformer
and the XGB 15-seed baseline on the same OOS window.

Reads the transformer eval JSON + the XGB sweep JSON and emits a unified
summary table that can be embedded in the project memo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_cells(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(Path(path).read_text())
    return list(obj.get("cells", []))


def best_by_goodness(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cells:
        return None
    return max(
        cells,
        key=lambda c: c.get("goodness_score", -1e18),
    )


def find_cell(cells: list[dict[str, Any]], lev: float, ms: float | None = None,
              top_n: int = 1, fee_regime: str = "deploy") -> dict[str, Any] | None:
    for c in cells:
        if abs(float(c.get("leverage", -1)) - lev) > 1e-6:
            continue
        if int(c.get("top_n", -1)) != top_n:
            continue
        if str(c.get("fee_regime", "")) != fee_regime:
            continue
        if ms is not None and abs(float(c.get("min_score", -1)) - ms) > 1e-6:
            continue
        return c
    return None


def fmt_cell(c: dict[str, Any] | None) -> dict[str, Any]:
    if c is None:
        return {}
    return {
        "leverage": c.get("leverage"),
        "min_score": c.get("min_score"),
        "top_n": c.get("top_n"),
        "fee_regime": c.get("fee_regime"),
        "n_windows": c.get("n_windows"),
        "median_monthly_pct": round(float(c.get("median_monthly_pct", 0.0)), 3),
        "p10_monthly_pct": round(float(c.get("p10_monthly_pct", 0.0)), 3),
        "n_neg": c.get("n_neg"),
        "worst_dd_pct": round(float(c.get("worst_dd_pct", 0.0)), 3),
        "median_sortino": round(float(c.get("median_sortino", 0.0)), 3),
        "goodness_score": round(float(c.get("goodness_score", 0.0)), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cross-attn-json", required=True, type=Path)
    p.add_argument("--xgb-json", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    ca_cells = _load_cells(args.cross_attn_json)
    xgb_cells = _load_cells(args.xgb_json)

    ca_best = best_by_goodness(ca_cells)
    xgb_best = best_by_goodness(xgb_cells)

    summary = {
        "comparison": "cross_attn_transformer_v1_seed0 vs XGB 15-seed (oos2024_ensemble_gpu_fresh)",
        "oos_start": "2025-01-02",
        "oos_end": "2026-04-28",
        "window_days": 30,
        "stride_days": 7,
        "top_n": 1,
        "fee_regime": "deploy",
        "hold_through": True,
        "cross_attn": {
            "best_by_goodness": fmt_cell(ca_best),
            "lev1_ms0": fmt_cell(find_cell(ca_cells, 1.0, ms=0.0)),
            "lev1_ms045": fmt_cell(find_cell(ca_cells, 1.0, ms=0.45)),
            "lev2_ms0": fmt_cell(find_cell(ca_cells, 2.0, ms=0.0)),
        },
        "xgb_15seed": {
            "best_by_goodness": fmt_cell(xgb_best),
            "lev1_ms055": fmt_cell(find_cell(xgb_cells, 1.0, ms=0.55)),
            "lev1_ms060": fmt_cell(find_cell(xgb_cells, 1.0, ms=0.60)),
            "lev1_ms065": fmt_cell(find_cell(xgb_cells, 1.0, ms=0.65)),
            "lev1_ms070": fmt_cell(find_cell(xgb_cells, 1.0, ms=0.70)),
        },
        "verdict": {
            "ca_lev1_med_vs_xgb_lev1_med": (
                round(float((ca_best or {}).get("median_monthly_pct", 0.0)), 3)
                - round(float((xgb_best or {}).get("median_monthly_pct", 0.0)), 3)
            ),
            "ca_lev1_p10_vs_xgb_lev1_p10": (
                round(float((ca_best or {}).get("p10_monthly_pct", 0.0)), 3)
                - round(float((xgb_best or {}).get("p10_monthly_pct", 0.0)), 3)
            ),
        },
        "sources": {
            "cross_attn_json": str(args.cross_attn_json),
            "xgb_json": str(args.xgb_json),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
