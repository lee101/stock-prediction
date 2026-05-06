import csv
from pathlib import Path

from scripts.binance_head_to_head import load_leaderboard


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_daily_xgb_row_uses_monthly_gate(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance33_xgb_top_leverage_20260427.csv"
    _write_csv(
        csv_path,
        [
            {
                "experiment": "xgb08",
                "eval_days": 120,
                "slip_bps": 20,
                "median_monthly_pct": 31.0,
                "p10_monthly_pct": 2.0,
                "median_pct": 180.0,
                "p10_pct": 8.0,
                "neg_windows": 0,
                "windows": 45,
                "p90_dd_pct": 18.0,
                "median_trades": 21,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20)

    assert len(rows) == 1
    assert rows[0].family == "xgb_daily"
    assert rows[0].pass_27 == 1
    assert rows[0].pass_30 == 1


def test_easy_universe_row_keeps_variant_context(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance33_easy_universe_xgb_prune_aug_20260430.csv"
    _write_csv(
        csv_path,
        [
            {
                "experiment": "xgb08",
                "train_variant": "aug",
                "top_k": 5,
                "max_leverage": 1.25,
                "eval_days": 120,
                "slip_bps": 20,
                "median_monthly_pct": 31.0,
                "p10_monthly_pct": 2.0,
                "median_pct": 180.0,
                "p10_pct": 8.0,
                "neg_windows": 0,
                "windows": 45,
                "p90_dd_pct": 18.0,
                "median_trades": 21,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20)

    assert len(rows) == 1
    assert rows[0].family == "xgb_easy_universe"
    assert "top_k=5" in rows[0].config
    assert "train_variant=aug" in rows[0].config
    assert rows[0].pass_30 == 1


def test_rolling_summary_requires_nonnegative_worst_cell(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance_pack_rolling_validation_demo_summary.csv"
    _write_csv(
        csv_path,
        [
            {
                "candidate_id": "cfg_bad_tail",
                "cells": 3,
                "negative_cells": 0,
                "median_monthly_return_pct": 35.0,
                "worst_monthly_return_pct": -1.0,
                "worst_total_return_pct": -4.0,
                "worst_max_drawdown_pct": 10.0,
                "min_num_sells": 25,
                "rolling_score": 0.0,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20)

    assert len(rows) == 1
    assert rows[0].family == "hourly_pack_rolling"
    assert rows[0].pass_27 == 0
    assert "negative_tail" in rows[0].notes


def test_gate_requires_enough_windows(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance33_regime_gate_smoke.csv"
    _write_csv(
        csv_path,
        [
            {
                "config": "gate_smoke",
                "eval_days": 120,
                "slip_bps": 20,
                "median_monthly_pct": 35.0,
                "p10_monthly_pct": 20.0,
                "neg_windows": 0,
                "windows": 3,
                "p90_dd_pct": 10.0,
                "median_trades": 30,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20, min_windows=20)

    assert len(rows) == 1
    assert rows[0].pass_30 == 0
    assert "few_windows" in rows[0].notes


def test_linear_cvar_pack_family_and_context(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance33_linear_cvar_pack_rand0045.csv"
    _write_csv(
        csv_path,
        [
            {
                "scope": "val",
                "config": "rand0045",
                "gate": "mkt_ret20gt-0.1",
                "top_n": 3,
                "allocation_mode": "cvar",
                "max_gross": 1.305,
                "max_weight": 0.5,
                "lookback_days": 40,
                "cvar_alpha": 0.9,
                "risk_power": 1.0,
                "eval_days": 120,
                "slip_bps": 20,
                "median_monthly_pct": 31.0,
                "p10_monthly_pct": 2.0,
                "median_pct": 190.0,
                "p10_pct": 8.0,
                "neg_windows": 0,
                "windows": 45,
                "p90_dd_pct": 18.0,
                "median_trades": 30,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20, min_windows=20)

    assert len(rows) == 1
    assert rows[0].family == "linear_cvar_pack"
    assert "gate=mkt_ret20gt-0.1" in rows[0].config
    assert "max_gross=1.305" in rows[0].config
    assert rows[0].pass_30 == 1


def test_meta_anneal_family_and_context(tmp_path: Path) -> None:
    csv_path = tmp_path / "binance33_meta_anneal_demo.csv"
    _write_csv(
        csv_path,
        [
            {
                "phase": "val_exact_single",
                "candidate_id": "cand1",
                "book_mode": "single",
                "top_k": 1,
                "max_gross": 1.305,
                "threshold": 0.5,
                "market_gate": -0.1,
                "eval_days": 120,
                "slip_bps": 20,
                "median_monthly_pct": 31.0,
                "p10_monthly_pct": 2.0,
                "median_pct": 190.0,
                "p10_pct": 8.0,
                "neg_windows": 0,
                "windows": 45,
                "p90_dd_pct": 18.0,
                "median_trades": 30,
            }
        ],
    )

    rows = load_leaderboard([csv_path], min_trades=20, min_windows=20)

    assert len(rows) == 1
    assert rows[0].family == "meta_anneal"
    assert "book_mode=single" in rows[0].config
    assert "threshold=0.5" in rows[0].config
    assert rows[0].pass_30 == 1
