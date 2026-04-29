"""Tests for xgbnew.run_daily output contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from xgbnew import run_daily
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE


class _FakeModel:
    def __init__(self, **_kwargs):
        pass

    def fit(self, *_args, **_kwargs) -> None:
        return None

    def feature_importances(self) -> pd.Series:
        return pd.Series({"ret_1d": 1.0})

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.75, index=df.index)


def test_run_daily_records_fee_provenance(monkeypatch, tmp_path) -> None:
    train_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=1001, freq="min"),
            "symbol": ["AAA"] * 1001,
            "target_oc_up": [1] * 1001,
        }
    )
    val_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "symbol": ["AAA"],
            "target_oc_up": [1],
        }
    )
    test_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")],
            "symbol": ["AAA"],
            "target_oc_up": [1],
        }
    )
    seen_fee_rates: list[float] = []

    monkeypatch.setattr(run_daily, "load_chronos_cache", lambda _path: {})
    monkeypatch.setattr(
        run_daily,
        "build_daily_dataset",
        lambda **_kwargs: (train_df, val_df, test_df),
    )
    monkeypatch.setattr(run_daily, "XGBStockModel", _FakeModel)
    monkeypatch.setattr(run_daily, "print_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_daily.time, "strftime", lambda _fmt: "20260102_030405")

    def _fake_simulate(_test_df, _model, cfg):
        seen_fee_rates.append(float(cfg.fee_rate))
        trade = SimpleNamespace(
            symbol="AAA",
            score=0.75,
            leverage=cfg.leverage,
            actual_open=100.0,
            actual_close=101.0,
            gross_return_pct=1.0,
            spread_bps=2.0,
            commission_bps=cfg.commission_bps,
            fee_rate=cfg.fee_rate,
            fill_buffer_bps=cfg.fill_buffer_bps,
            net_return_pct=0.75,
        )
        return SimpleNamespace(
            total_return_pct=1.0,
            monthly_return_pct=2.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown_pct=3.0,
            win_rate_pct=100.0,
            directional_accuracy_pct=100.0,
            total_trades=1,
            avg_spread_bps=2.0,
            avg_fee_bps=10.0,
            day_results=[
                SimpleNamespace(
                    day=pd.Timestamp("2026-01-02").date(),
                    equity_end=10_100.0,
                    trades=[trade],
                )
            ],
        )

    monkeypatch.setattr(run_daily, "simulate", _fake_simulate)

    rc = run_daily.main(
        [
            "--symbols",
            "AAA",
            "--chronos-cache",
            str(tmp_path / "missing-cache"),
            "--output-dir",
            str(tmp_path / "out"),
            "--n-estimators",
            "1",
        ]
    )

    assert rc == 0
    assert seen_fee_rates
    assert set(seen_fee_rates) == {PRODUCTION_STOCK_FEE_RATE}
    summary = json.loads(
        (tmp_path / "out" / "summary_20260102_030405.json").read_text(encoding="utf-8")
    )
    assert summary["fee_rate"] == PRODUCTION_STOCK_FEE_RATE
    assert summary["fill_buffer_bps"] == 5.0
    assert summary["commission_bps"] == 0.0
    assert summary["results"][0]["avg_fee_bps"] == 10.0
    trades = pd.read_csv(tmp_path / "out" / "trades_top2_lev1.0_xw0.50.csv")
    assert trades.loc[0, "fee_rate"] == PRODUCTION_STOCK_FEE_RATE
    assert trades.loc[0, "fill_buffer_bps"] == 5.0
