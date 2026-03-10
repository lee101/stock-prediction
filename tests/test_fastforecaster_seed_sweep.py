from pathlib import Path

from FastForecaster.config import FastForecasterConfig
from FastForecaster.seed_sweep import _parse_int_list, _seed_config, _seed_run_name


def test_parse_int_list_unique_in_order():
    assert _parse_int_list("1337,1701,1337,2026") == (1337, 1701, 2026)


def test_parse_int_list_rejects_empty():
    try:
        _parse_int_list(" ,, ")
    except ValueError as exc:
        assert "at least one seed" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for empty seed list")


def test_seed_run_name_suffix():
    assert _seed_run_name("runx", 1701) == "runx_seed1701"


def test_seed_config_uses_seed_specific_output_dir(tmp_path: Path):
    base = FastForecasterConfig(
        output_dir=tmp_path / "root",
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
    )
    cfg = _seed_config(base, sweep_dir=tmp_path / "sweep", seed=1701)

    assert cfg.output_dir == tmp_path / "sweep" / "seed_1701"
    assert cfg.checkpoint_dir == cfg.output_dir / "checkpoints"
    assert cfg.metrics_dir == cfg.output_dir / "metrics"
