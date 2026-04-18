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


def test_device_default_is_none():
    args = parse_args([
        "--n-estimators", "400",
        "--max-depth", "5",
        "--learning-rate", "0.03",
        "--top-n-grid", "1",
    ])
    assert args.device is None


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
    # Orthogonal sweep: 2 depths × 3 seeds = 6 cells.
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
