"""Coverage for the leverage knob added to xgbnew.eval_multiwindow."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from xgbnew import eval_multiwindow
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE
from xgbnew.eval_multiwindow import SweepConfig, _config_grid, parse_args


def test_sweep_config_default_leverage_is_one():
    cfg = SweepConfig(
        n_estimators=400, max_depth=5, learning_rate=0.03, top_n=1, xgb_weight=1.0
    )
    assert cfg.leverage == 1.0


def test_config_grid_expands_leverage_grid():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
        "--xgb-weight-grid", "1.0",
        "--leverage-grid", "1.0,1.25,1.5,1.75,2.0",
    ])
    grid = _config_grid(args)
    assert len(grid) == 5
    levs = sorted(c.leverage for c in grid)
    assert levs == [1.0, 1.25, 1.5, 1.75, 2.0]
    for cfg in grid:
        assert cfg.n_estimators == 400
        assert cfg.max_depth == 5
        assert cfg.learning_rate == 0.03
        assert cfg.top_n == 1
        assert cfg.xgb_weight == 1.0


def test_config_grid_default_leverage_when_grid_empty():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
    ])
    grid = _config_grid(args)
    assert len(grid) == 1
    assert grid[0].leverage == 1.0


def test_eval_multiwindow_default_fee_is_production_stress_fee():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
    ])
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_config_grid_custom_single_leverage():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
        "--leverage", "1.5",
    ])
    grid = _config_grid(args)
    assert len(grid) == 1
    assert grid[0].leverage == 1.5


def test_sweep_config_default_random_state_is_forty_two():
    cfg = SweepConfig(
        n_estimators=400, max_depth=5, learning_rate=0.03, top_n=1, xgb_weight=1.0
    )
    assert cfg.random_state == 42


def test_config_grid_expands_random_state_grid():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
        "--random-state-grid", "0,1,2,3,4,5,6,7",
    ])
    grid = _config_grid(args)
    assert len(grid) == 8
    seeds = sorted(c.random_state for c in grid)
    assert seeds == [0, 1, 2, 3, 4, 5, 6, 7]
    for cfg in grid:
        assert cfg.n_estimators == 400
        assert cfg.max_depth == 5
        assert cfg.learning_rate == 0.03
        assert cfg.top_n == 1


def test_config_grid_default_random_state_when_grid_empty():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
    ])
    grid = _config_grid(args)
    assert len(grid) == 1
    assert grid[0].random_state == 42


def test_device_default_is_cuda():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
    ])
    assert args.device == "cuda"


def test_device_cuda_is_parsed():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
        "--device", "cuda",
    ])
    assert args.device == "cuda"


def test_config_grid_crosses_random_state_and_hyperparams():
    # Orthogonal sweep: 2 depths x 3 seeds = 6 cells.
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth-grid", "5,7",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
        "--random-state-grid", "42,7,11",
    ])
    grid = _config_grid(args)
    assert len(grid) == 6
    assert {(c.max_depth, c.random_state) for c in grid} == {
        (5, 42), (5, 7), (5, 11), (7, 42), (7, 7), (7, 11),
    }


def test_eval_multiwindow_model_save_uses_atomic_writer(monkeypatch, tmp_path):
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAA\nBBB\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    model_path = tmp_path / "saved.pkl"
    train_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-01").date()] * 1000,
            "symbol": ["AAA", "BBB"] * 500,
            "target_oc": [0.01, -0.01] * 500,
            "target_oc_up": [1, 0] * 500,
        }
    )
    oos_dates = [pd.Timestamp("2026-01-02").date() + pd.Timedelta(days=i).to_pytimedelta() for i in range(5)]
    oos_df = pd.DataFrame(
        {
            "date": [oos_dates[i % len(oos_dates)] for i in range(100)],
            "symbol": ["AAA", "BBB"] * 50,
            "target_oc": [0.01, -0.01] * 50,
            "target_oc_up": [1, 0] * 50,
        }
    )
    saved_paths = []

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def fit(self, *_args, **_kwargs) -> None:
            return None

        def predict_scores(self, df: pd.DataFrame) -> pd.Series:
            return pd.Series(0.75, index=df.index)

        def save(self, path) -> None:
            path.write_bytes(b"multiwindow-model")

    def save_model(model, path):
        saved_paths.append(path)
        model.save(path)

    def simulate(_df, _model, _cfg, **_kwargs):
        return SimpleNamespace(
            total_return_pct=1.0,
            sharpe_ratio=1.0,
            sortino_ratio=2.0,
            max_drawdown_pct=1.0,
            win_rate_pct=55.0,
            directional_accuracy_pct=55.0,
            total_trades=5,
            avg_fee_bps=10.0,
            avg_spread_bps=2.0,
            day_results=[object()],
        )

    monkeypatch.setattr(eval_multiwindow, "build_daily_dataset", lambda **_kwargs: (train_df, pd.DataFrame(), oos_df))
    monkeypatch.setattr(eval_multiwindow, "XGBStockModel", FakeModel)
    monkeypatch.setattr(eval_multiwindow, "save_model_atomic", save_model)
    monkeypatch.setattr(eval_multiwindow, "simulate", simulate)
    monkeypatch.setattr(eval_multiwindow.time, "strftime", lambda _fmt: "20260102_030405")

    rc = eval_multiwindow.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--chronos-cache",
            str(tmp_path / "missing-cache"),
            "--window-days",
            "5",
            "--stride-days",
            "5",
            "--model-save-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--n-estimators",
            "1",
            "--top-n-grid",
            "1",
        ]
    )

    assert rc == 0
    assert saved_paths == [model_path]
    assert model_path.read_bytes() == b"multiwindow-model"
    assert (output_dir / "multiwindow_20260102_030405.json").is_file()
