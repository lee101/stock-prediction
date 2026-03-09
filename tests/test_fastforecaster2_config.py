from pathlib import Path

import pytest

from fastforecaster2.config import FastForecaster2Config


def test_config_normalizes_symbols_and_tags():
    cfg = FastForecaster2Config(
        symbols=(" nvda", "GOOG", "nvda", ""),
        wandb_tags=["fast", "mae", "fast"],
        max_symbols=0,
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
    )
    assert cfg.symbols == ("GOOG", "NVDA")
    assert cfg.wandb_tags == ("fast", "mae")


def test_config_accepts_chronos_embedding_path():
    cfg = FastForecaster2Config(
        chronos_embeddings_path=Path("embeddings.json"),
        min_rows_per_symbol=400,
        lookback=128,
        horizon=16,
    )
    assert cfg.chronos_embeddings_path == Path("embeddings.json")


def test_config_rejects_invalid_chronos_blend():
    with pytest.raises(ValueError, match="chronos_embeddings_blend"):
        FastForecaster2Config(
            chronos_embeddings_blend=1.5,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_invalid_market_sim_top_k():
    with pytest.raises(ValueError, match="market_sim_top_k"):
        FastForecaster2Config(
            market_sim_top_k=0,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_invalid_market_sim_trade_intensity():
    with pytest.raises(ValueError, match="market_sim_max_trade_intensity"):
        FastForecaster2Config(
            market_sim_max_trade_intensity=101.0,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_invalid_market_sim_signal_ema_alpha():
    with pytest.raises(ValueError, match="market_sim_signal_ema_alpha"):
        FastForecaster2Config(
            market_sim_signal_ema_alpha=0.0,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_negative_market_sim_switch_score_gap():
    with pytest.raises(ValueError, match="market_sim_switch_score_gap"):
        FastForecaster2Config(
            market_sim_switch_score_gap=-1e-5,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )


def test_config_rejects_negative_market_sim_entry_score_threshold():
    with pytest.raises(ValueError, match="market_sim_entry_score_threshold"):
        FastForecaster2Config(
            market_sim_entry_score_threshold=-1e-5,
            min_rows_per_symbol=400,
            lookback=128,
            horizon=16,
        )
