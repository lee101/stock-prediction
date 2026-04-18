"""Coverage for the leverage knob added to xgbnew.eval_multiwindow."""
from __future__ import annotations

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
