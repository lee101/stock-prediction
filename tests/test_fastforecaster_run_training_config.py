import sys

from FastForecaster.run_training import build_config, parse_args


def test_daily_dataset_uses_lower_min_rows_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_training", "--dataset", "daily", "--lookback", "128", "--horizon", "10"],
    )
    args = parse_args()
    cfg = build_config(args)
    assert cfg.min_rows_per_symbol == max(240, cfg.lookback + cfg.horizon + 64)


def test_daily_dataset_respects_explicit_min_rows(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_training",
            "--dataset",
            "daily",
            "--lookback",
            "128",
            "--horizon",
            "10",
            "--min-rows-per-symbol",
            "512",
        ],
    )
    args = parse_args()
    cfg = build_config(args)
    assert cfg.min_rows_per_symbol == 512
