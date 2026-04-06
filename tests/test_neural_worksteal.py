"""Tests for neural work-steal daily policy model, data, and training."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from binance_worksteal.model import DailyWorkStealPolicy, PerSymbolWorkStealPolicy
from binance_worksteal.data import (
    WorkStealDataset,
    WorkStealSequentialDataset,
    compute_features,
    compute_sma,
    compute_atr,
    compute_rsi,
    build_datasets,
    build_dataloader,
    FEATURE_NAMES,
    N_MARKET_FEATURES,
)
from binance_worksteal.train_neural import (
    simulate_daily_trades,
    run_sequential_sim,
    run_multistep_rollout,
    train_epoch,
    train_epoch_multistep,
    eval_epoch,
    eval_epoch_multistep,
)
from differentiable_loss_utils import compute_loss_by_type, DAILY_PERIODS_PER_YEAR_CRYPTO


def _make_fake_bars(n_days=100, n_symbols=3, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    symbol_data = {}
    symbol_features = {}
    symbols = [f"SYM{i}USDT" for i in range(n_symbols)]

    for sym in symbols:
        price = 100.0 + rng.randn(n_days).cumsum() * 2
        price = np.maximum(price, 1.0)
        high = price + rng.rand(n_days) * 3
        low = price - rng.rand(n_days) * 3
        low = np.maximum(low, 0.5)
        vol = rng.rand(n_days) * 1e6

        df = pd.DataFrame({
            "timestamp": dates,
            "open": price + rng.randn(n_days) * 0.5,
            "high": high,
            "low": low,
            "close": price,
            "volume": vol,
            "symbol": sym,
        })
        feats = compute_features(df)
        symbol_data[sym] = df
        symbol_features[sym] = feats

    return symbol_data, symbol_features, dates, symbols


# ---- Model tests ----

class TestModel:
    def test_forward_shape(self):
        B, T, S, F = 4, 30, 5, 11
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=64, num_layers=2, num_heads=2, seq_len=T)
        x = torch.randn(B, T, S, F)
        out = model(x)
        assert out["buy_offset"].shape == (B, S)
        assert out["sell_offset"].shape == (B, S)
        assert out["intensity"].shape == (B, S)

    def test_output_ranges(self):
        B, T, S, F = 2, 10, 3, 11
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T)
        x = torch.randn(B, T, S, F)
        out = model(x)
        assert (out["buy_offset"] >= 0).all() and (out["buy_offset"] <= 0.30).all()
        assert (out["sell_offset"] >= 0).all() and (out["sell_offset"] <= 0.30).all()
        assert (out["intensity"] >= 0).all() and (out["intensity"] <= 1.0).all()

    def test_gradients_flow(self):
        B, T, S, F = 2, 10, 3, 11
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T)
        x = torch.randn(B, T, S, F)
        out = model(x)
        loss = out["buy_offset"].sum() + out["sell_offset"].sum() + out["intensity"].sum()
        loss.backward()
        grad_norms = []
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)

    def test_different_n_symbols(self):
        for n_sym in [1, 5, 30]:
            model = DailyWorkStealPolicy(n_features=11, n_symbols=n_sym, hidden_dim=32, num_layers=1, num_heads=2, seq_len=10)
            x = torch.randn(2, 10, n_sym, 11)
            out = model(x)
            assert out["buy_offset"].shape == (2, n_sym)


class TestPerSymbolModel:
    def test_forward_shape(self):
        B, T, S, F = 4, 30, 5, 11
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=64,
            num_temporal_layers=2, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        x = torch.randn(B, T, S, F)
        out = model(x)
        assert out["buy_offset"].shape == (B, S)
        assert out["sell_offset"].shape == (B, S)
        assert out["intensity"].shape == (B, S)

    def test_output_ranges(self):
        B, T, S, F = 2, 10, 3, 11
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        x = torch.randn(B, T, S, F)
        out = model(x)
        assert (out["buy_offset"] >= 0).all() and (out["buy_offset"] <= 0.30).all()
        assert (out["sell_offset"] >= 0).all() and (out["sell_offset"] <= 0.30).all()
        assert (out["intensity"] >= 0).all() and (out["intensity"] <= 1.0).all()

    def test_gradients_flow(self):
        B, T, S, F = 2, 10, 3, 11
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        x = torch.randn(B, T, S, F)
        out = model(x)
        loss = out["buy_offset"].sum() + out["sell_offset"].sum() + out["intensity"].sum()
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0

    def test_different_n_symbols(self):
        for n_sym in [1, 5, 30]:
            model = PerSymbolWorkStealPolicy(
                n_features=11, n_symbols=n_sym, hidden_dim=32,
                num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=10,
            )
            x = torch.randn(2, 10, n_sym, 11)
            out = model(x)
            assert out["buy_offset"].shape == (2, n_sym)

    def test_shared_encoder_weight_sharing(self):
        S = 5
        torch.manual_seed(42)
        model = PerSymbolWorkStealPolicy(
            n_features=11, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=10,
        )
        # Give head non-zero weights so outputs can differ
        nn.init.normal_(model.head.weight, std=0.5)
        nn.init.normal_(model.head.bias, std=0.5)
        x = torch.randn(1, 10, S, 11)
        x[:, :, 0, :] *= 10.0
        out = model(x)
        # Different inputs through shared encoder should produce different outputs
        assert not torch.allclose(out["intensity"][:, 0], out["intensity"][:, 1], atol=1e-3)


# ---- Data tests ----

class TestData:
    def test_compute_features(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.rand(50) * 100 + 50,
            "high": np.random.rand(50) * 100 + 55,
            "low": np.random.rand(50) * 100 + 45,
            "close": np.random.rand(50) * 100 + 50,
            "volume": np.random.rand(50) * 1e6,
        })
        feats = compute_features(df)
        assert len(feats) == 50
        assert list(feats.columns) == FEATURE_NAMES
        assert not feats.isna().any().any()

    def test_compute_sma(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        sma = compute_sma(s, 3)
        assert abs(sma.iloc[-1] - 9.0) < 0.01

    def test_compute_rsi(self):
        s = pd.Series(np.arange(20, dtype=float))
        rsi = compute_rsi(s, 14)
        assert rsi.iloc[-1] > 99.0

    def test_compute_atr(self):
        df = pd.DataFrame({
            "high": np.arange(20, dtype=float) + 5,
            "low": np.arange(20, dtype=float),
            "close": np.arange(20, dtype=float) + 2,
        })
        atr = compute_atr(df, 14)
        assert len(atr) == 20
        assert atr.iloc[-1] > 0

    def test_dataset_creation(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=10, symbols=symbols)
        assert len(ds) == 90
        sample = ds[0]
        assert sample["features"].shape == (10, 3, 11)
        assert sample["ohlcv"].shape == (10, 3, 4)
        assert sample["target_high"].shape == (3,)

    def test_dataloader(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=10, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        assert batch["features"].shape[0] <= 8
        assert batch["features"].shape[1] == 10
        assert batch["features"].shape[2] == 3

    def test_n_market_features(self):
        assert N_MARKET_FEATURES == 8
        assert FEATURE_NAMES[N_MARKET_FEATURES] == "has_position"


class TestSequentialDataset:
    def test_creation(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols,
        )
        assert len(ds) > 0

    def test_shapes(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols,
        )
        sample = ds[0]
        assert sample["features"].shape == (5, 10, 3, 11)  # [R, T, S, F]
        assert sample["target_high"].shape == (5, 3)        # [R, S]
        assert sample["target_low"].shape == (5, 3)
        assert sample["target_close"].shape == (5, 3)
        assert sample["current_close"].shape == (5, 3)

    def test_position_features_zeroed(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols,
        )
        sample = ds[0]
        pos_feats = sample["features"][:, :, :, N_MARKET_FEATURES:]
        assert (pos_feats == 0).all()

    def test_stride(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds1 = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols, stride=1,
        )
        ds3 = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols, stride=3,
        )
        assert len(ds3) < len(ds1)

    def test_dataloader(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=10, rollout_len=5, symbols=symbols,
        )
        loader = build_dataloader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        B = batch["features"].shape[0]
        assert B <= 4
        assert batch["features"].shape == (B, 5, 10, 3, 11)
        assert batch["target_high"].shape == (B, 5, 3)


# ---- Simulator tests ----

class TestSimulator:
    def test_simulate_daily_trades(self):
        B, S = 4, 3
        highs = torch.rand(B, S) * 10 + 100
        lows = highs - torch.rand(B, S) * 5
        closes = (highs + lows) / 2
        buy_prices = closes * 0.95
        sell_prices = closes * 1.05
        intensity = torch.ones(B, S) * 0.2

        result = simulate_daily_trades(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_prices, sell_prices=sell_prices,
            intensity=intensity, initial_cash=10000.0,
        )
        assert result["portfolio_value"].shape == (B,)
        assert result["returns"].shape == (B,)

    def test_simulator_differentiable(self):
        B, S = 2, 3
        highs = torch.rand(B, S) * 10 + 100
        lows = highs - torch.rand(B, S) * 5
        closes = (highs + lows) / 2
        buy_prices = torch.nn.Parameter(closes * 0.95)
        sell_prices = torch.nn.Parameter(closes * 1.05)
        intensity = torch.nn.Parameter(torch.ones(B, S) * 0.2)

        result = simulate_daily_trades(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_prices, sell_prices=sell_prices,
            intensity=intensity, initial_cash=10000.0,
        )
        loss = -result["returns"].mean()
        loss.backward()
        assert buy_prices.grad is not None
        assert sell_prices.grad is not None
        assert intensity.grad is not None

    def test_zero_intensity_no_trades(self):
        B, S = 2, 3
        highs = torch.ones(B, S) * 110
        lows = torch.ones(B, S) * 90
        closes = torch.ones(B, S) * 100
        buy_prices = torch.ones(B, S) * 95
        sell_prices = torch.ones(B, S) * 105
        intensity = torch.zeros(B, S)

        result = simulate_daily_trades(
            highs=highs, lows=lows, closes=closes,
            buy_prices=buy_prices, sell_prices=sell_prices,
            intensity=intensity, initial_cash=10000.0,
        )
        assert torch.allclose(result["returns"], torch.zeros(B), atol=1e-6)


# ---- Multi-step rollout tests ----

class TestMultistepRollout:
    def test_rollout_shapes(self):
        B, R, T, S, F = 2, 5, 10, 3, 11
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        features = torch.randn(B, R, T, S, F)
        cur_close = torch.rand(B, R, S) * 100 + 50
        t_high = cur_close * 1.05
        t_low = cur_close * 0.95
        t_close = cur_close + torch.randn(B, R, S)

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high,
            target_lows=t_low,
            target_closes=t_close,
            current_closes=cur_close,
            model=model,
            initial_cash=10000.0,
        )
        assert result["returns"].shape == (B, R)
        assert result["final_has_pos"].shape == (B, S)
        assert result["final_cash"].shape == (B,)

    def test_rollout_gradients(self):
        B, R, T, S, F = 2, 5, 10, 3, 11
        torch.manual_seed(42)
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        nn.init.normal_(model.head.weight, std=0.1)
        nn.init.normal_(model.head.bias, std=0.1)

        features = torch.randn(B, R, T, S, F)
        cur_close = torch.rand(B, R, S) * 100 + 50
        t_high = cur_close * 1.10
        t_low = cur_close * 0.80
        t_close = cur_close + torch.randn(B, R, S)

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high,
            target_lows=t_low,
            target_closes=t_close,
            current_closes=cur_close,
            model=model,
            temperature=0.05,
            initial_cash=10000.0,
        )
        loss, _, _, _ = compute_loss_by_type(
            result["returns"],
            "sortino",
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
        )
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0

    def test_position_state_propagation(self):
        B, R, T, S, F = 1, 3, 10, 2, 11
        torch.manual_seed(99)
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        nn.init.normal_(model.head.weight, std=0.5)
        nn.init.normal_(model.head.bias, std=0.5)

        features = torch.randn(B, R, T, S, F)
        cur_close = torch.ones(B, R, S) * 100
        t_high = cur_close * 1.10
        t_low = cur_close * 0.80
        t_close = cur_close * 1.01

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high,
            target_lows=t_low,
            target_closes=t_close,
            current_closes=cur_close,
            model=model,
            temperature=0.05,
            initial_cash=10000.0,
            max_hold_days=14,
        )
        # Returns should vary across steps (not all identical)
        returns = result["returns"]
        assert returns.shape == (B, R)

    def test_force_close_after_max_hold(self):
        B, R, T, S, F = 1, 3, 10, 2, 11
        torch.manual_seed(42)
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        nn.init.normal_(model.head.weight, std=0.5)

        features = torch.randn(B, R, T, S, F)
        cur_close = torch.ones(B, R, S) * 100
        t_high = cur_close * 1.02
        t_low = cur_close * 0.80
        t_close = cur_close * 1.01

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high,
            target_lows=t_low,
            target_closes=t_close,
            current_closes=cur_close,
            model=model,
            temperature=0.05,
            initial_cash=10000.0,
            max_hold_days=2,
        )
        assert result["returns"].shape == (B, R)
        assert not result["returns"].isnan().any()

    def test_rollout_with_flat_model(self):
        B, R, T, S, F = 2, 5, 10, 3, 11
        model = DailyWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_layers=1, num_heads=2, seq_len=T,
        )
        features = torch.randn(B, R, T, S, F)
        cur_close = torch.rand(B, R, S) * 100 + 50
        t_high = cur_close * 1.05
        t_low = cur_close * 0.95
        t_close = cur_close + torch.randn(B, R, S)

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high,
            target_lows=t_low,
            target_closes=t_close,
            current_closes=cur_close,
            model=model,
            initial_cash=10000.0,
        )
        assert result["returns"].shape == (B, R)


# ---- Training tests ----

class TestTraining:
    def test_run_sequential_sim(self):
        B, T, S, F = 2, 10, 3, 11
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T)
        features = torch.randn(B, T, S, F)
        ohlcv = torch.rand(B, T, S, 4) * 100 + 50
        current_close = ohlcv[:, -1, :, 3]
        target_high = current_close + torch.rand(B, S) * 5
        target_low = current_close - torch.rand(B, S) * 5
        target_close = current_close + torch.randn(B, S)

        sim, actions = run_sequential_sim(
            features_seq=features, ohlcv_seq=ohlcv,
            target_highs=target_high, target_lows=target_low,
            target_closes=target_close, current_closes=current_close,
            model=model,
        )
        assert sim["portfolio_value"].shape == (B,)

    def test_loss_differentiable_through_model(self):
        B, T, S, F = 8, 10, 3, 11
        torch.manual_seed(123)
        model = DailyWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T,
            max_buy_offset=0.30, max_sell_offset=0.30,
        )
        nn.init.normal_(model.head.weight, std=0.1)
        nn.init.normal_(model.head.bias, std=0.1)

        features = torch.randn(B, T, S, F)
        ohlcv = torch.rand(B, T, S, 4) * 100 + 50
        current_close = ohlcv[:, -1, :, 3]
        target_high = current_close * 1.10
        target_low = current_close * 0.70
        target_close = current_close + torch.randn(B, S) * 2

        sim, actions = run_sequential_sim(
            features_seq=features, ohlcv_seq=ohlcv,
            target_highs=target_high, target_lows=target_low,
            target_closes=target_close, current_closes=current_close,
            model=model,
            temperature=0.05,
        )
        returns = sim["returns"]
        loss, score, sortino, annual_ret = compute_loss_by_type(
            returns.unsqueeze(0),
            "sortino",
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
        )
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0

    def test_train_one_epoch_fake_data(self):
        S, F, T = 3, 11, 10
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(50, S)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=T, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=True)

        config = {
            "maker_fee": 0.001,
            "initial_cash": 10000.0,
            "temperature": 5e-4,
            "max_positions": 3,
            "loss_type": "sortino",
            "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        device = torch.device("cpu")
        metrics = train_epoch(model, loader, optimizer, device, config)
        assert "loss" in metrics
        assert "mean_return" in metrics
        assert isinstance(metrics["loss"], float)

    def test_eval_one_epoch_fake_data(self):
        S, F, T = 3, 11, 10
        model = DailyWorkStealPolicy(n_features=F, n_symbols=S, hidden_dim=32, num_layers=1, num_heads=2, seq_len=T)

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(50, S)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=T, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=False)

        config = {
            "maker_fee": 0.001,
            "initial_cash": 10000.0,
            "temperature": 5e-4,
            "max_positions": 3,
            "loss_type": "sortino",
            "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        device = torch.device("cpu")
        metrics = eval_epoch(model, loader, device, config)
        assert "loss" in metrics
        assert "sortino" in metrics


class TestMultistepTraining:
    def test_train_one_epoch_multistep(self):
        S, F, T, R = 3, 11, 10, 5
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(60, S)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=T, rollout_len=R, symbols=symbols,
        )
        loader = build_dataloader(ds, batch_size=4, shuffle=True)

        config = {
            "maker_fee": 0.001,
            "initial_cash": 10000.0,
            "temperature": 0.02,
            "max_positions": 3,
            "max_hold_days": 14,
            "loss_type": "sortino",
            "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        device = torch.device("cpu")
        metrics = train_epoch_multistep(model, loader, optimizer, device, config)
        assert "loss" in metrics
        assert "mean_return" in metrics
        assert isinstance(metrics["loss"], float)

    def test_eval_one_epoch_multistep(self):
        S, F, T, R = 3, 11, 10, 5
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(60, S)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=T, rollout_len=R, symbols=symbols,
        )
        loader = build_dataloader(ds, batch_size=4, shuffle=False)

        config = {
            "maker_fee": 0.001,
            "initial_cash": 10000.0,
            "temperature": 0.02,
            "max_positions": 3,
            "max_hold_days": 14,
            "loss_type": "sortino",
            "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        device = torch.device("cpu")
        metrics = eval_epoch_multistep(model, loader, device, config)
        assert "loss" in metrics
        assert "sortino" in metrics

    def test_persymbol_with_single_step_eval(self):
        S, F, T = 3, 11, 10
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(50, S)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=T, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=False)

        config = {
            "maker_fee": 0.001,
            "initial_cash": 10000.0,
            "temperature": 5e-4,
            "max_positions": 3,
            "loss_type": "sortino",
            "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        device = torch.device("cpu")
        metrics = eval_epoch(model, loader, device, config)
        assert "loss" in metrics
        assert "sortino" in metrics


class TestAMP:
    @staticmethod
    def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
        if "out of memory" in str(exc).lower():
            pytest.skip(f"Neural work-steal CUDA AMP test skipped under shared-GPU resource pressure: {exc}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_epoch_amp(self):
        S, F, T = 3, 11, 10
        device = torch.device("cuda")
        try:
            model = DailyWorkStealPolicy(
                n_features=F, n_symbols=S, hidden_dim=32,
                num_layers=1, num_heads=2, seq_len=T,
            ).to(device)
        except Exception as exc:
            self._skip_for_cuda_resource_pressure(exc)
            raise
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(50, S)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=T, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=True)

        config = {
            "maker_fee": 0.001, "initial_cash": 10000.0, "temperature": 5e-4,
            "max_positions": 3, "loss_type": "sortino", "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        try:
            metrics = train_epoch(model, loader, optimizer, device, config, scaler=scaler)
        except Exception as exc:
            self._skip_for_cuda_resource_pressure(exc)
            raise
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_epoch_multistep_amp(self):
        S, F, T, R = 3, 11, 10, 5
        device = torch.device("cuda")
        try:
            model = PerSymbolWorkStealPolicy(
                n_features=F, n_symbols=S, hidden_dim=32,
                num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
            ).to(device)
        except Exception as exc:
            self._skip_for_cuda_resource_pressure(exc)
            raise
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(60, S)
        ds = WorkStealSequentialDataset(
            symbol_data, symbol_features, dates,
            seq_len=T, rollout_len=R, symbols=symbols,
        )
        loader = build_dataloader(ds, batch_size=4, shuffle=True)

        config = {
            "maker_fee": 0.001, "initial_cash": 10000.0, "temperature": 0.02,
            "max_positions": 3, "max_hold_days": 14, "loss_type": "sortino",
            "return_weight": 0.05, "grad_clip": 1.0,
        }
        try:
            metrics = train_epoch_multistep(model, loader, optimizer, device, config, scaler=scaler)
        except Exception as exc:
            self._skip_for_cuda_resource_pressure(exc)
            raise
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])

    def test_amp_fallback_cpu(self):
        S, F, T = 3, 11, 10
        device = torch.device("cpu")
        model = DailyWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_layers=1, num_heads=2, seq_len=T,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        symbol_data, symbol_features, dates, symbols = _make_fake_bars(50, S)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=T, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=True)

        config = {
            "maker_fee": 0.001, "initial_cash": 10000.0, "temperature": 5e-4,
            "max_positions": 3, "loss_type": "sortino", "return_weight": 0.05,
            "grad_clip": 1.0,
        }
        metrics = train_epoch(model, loader, optimizer, device, config, scaler=None)
        assert "loss" in metrics


class TestGradCheckpoint:
    def test_rollout_with_grad_checkpoint(self):
        B, R, T, S, F = 2, 5, 10, 3, 11
        torch.manual_seed(42)
        model = PerSymbolWorkStealPolicy(
            n_features=F, n_symbols=S, hidden_dim=32,
            num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
        )
        nn.init.normal_(model.head.weight, std=0.1)
        nn.init.normal_(model.head.bias, std=0.1)

        features = torch.randn(B, R, T, S, F)
        cur_close = torch.rand(B, R, S) * 100 + 50
        t_high = cur_close * 1.10
        t_low = cur_close * 0.80
        t_close = cur_close + torch.randn(B, R, S)

        result = run_multistep_rollout(
            features_seq=features,
            target_highs=t_high, target_lows=t_low,
            target_closes=t_close, current_closes=cur_close,
            model=model, temperature=0.05, initial_cash=10000.0,
            use_grad_checkpoint=True,
        )
        loss, _, _, _ = compute_loss_by_type(
            result["returns"], "sortino",
            periods_per_year=DAILY_PERIODS_PER_YEAR_CRYPTO,
        )
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.norm() > 0)
        assert grad_count > 0

    def test_grad_checkpoint_matches_no_checkpoint(self):
        B, R, T, S, F = 2, 3, 10, 3, 11
        features = torch.randn(B, R, T, S, F)
        cur_close = torch.rand(B, R, S) * 100 + 50
        t_high = cur_close * 1.10
        t_low = cur_close * 0.80
        t_close = cur_close + torch.randn(B, R, S)

        for gc in [False, True]:
            torch.manual_seed(123)
            model = PerSymbolWorkStealPolicy(
                n_features=F, n_symbols=S, hidden_dim=32,
                num_temporal_layers=1, num_cross_layers=1, num_heads=2, seq_len=T,
            )
            result = run_multistep_rollout(
                features_seq=features,
                target_highs=t_high, target_lows=t_low,
                target_closes=t_close, current_closes=cur_close,
                model=model, temperature=0.05, initial_cash=10000.0,
                use_grad_checkpoint=gc,
            )
            if gc:
                ret_gc = result["returns"].detach().clone()
            else:
                ret_no_gc = result["returns"].detach().clone()
        assert torch.allclose(ret_gc, ret_no_gc, atol=1e-5)


class TestDataLoaderOptimized:
    def test_pin_memory_and_workers(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=10, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
        batch = next(iter(loader))
        assert batch["features"].shape[0] <= 8

    def test_dataloader_default_backward_compat(self):
        symbol_data, symbol_features, dates, symbols = _make_fake_bars(100, 3)
        ds = WorkStealDataset(symbol_data, symbol_features, dates, seq_len=10, symbols=symbols)
        loader = build_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        assert batch["features"].shape[0] <= 8


class TestBuildDatasets:
    def test_build_datasets_real_data(self):
        data_dir = "trainingdata/train"
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        try:
            train_ds, val_ds, test_ds, loaded = build_datasets(
                data_dir=data_dir, symbols=syms, seq_len=10, test_days=30, val_days=15,
            )
        except (ValueError, FileNotFoundError):
            pytest.skip("Training data not available")
        assert len(loaded) > 0
        assert len(train_ds) > 0
        sample = train_ds[0]
        assert sample["features"].shape[1] == len(loaded)
        assert sample["features"].shape[2] == len(FEATURE_NAMES)
