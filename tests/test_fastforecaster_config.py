import pytest

from FastForecaster.config import FastForecasterConfig


def test_config_normalizes_symbols_and_tags():
    cfg = FastForecasterConfig(
        symbols=(" nvda", "GOOG", "nvda", ""),
        wandb_tags=["fast", "mae", "fast"],
        max_symbols=0,
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
    )
    assert cfg.symbols == ("GOOG", "NVDA")
    assert cfg.wandb_tags == ("fast", "mae")


def test_config_rejects_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        FastForecasterConfig(
            precision="int8",
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_invalid_lr_range():
    with pytest.raises(ValueError, match="min_learning_rate"):
        FastForecasterConfig(
            learning_rate=1e-4,
            min_learning_rate=1e-3,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_invalid_ema_decay():
    with pytest.raises(ValueError, match="ema_decay"):
        FastForecasterConfig(
            ema_decay=1.0,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )
