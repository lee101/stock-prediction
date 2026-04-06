"""Tests for signal calibrator module and training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rl_trading_agent_binance"))

from signal_calibrator import CalibrationConfig, SignalCalibrator, load_calibrator, save_calibrator


class TestSignalCalibratorModule:
    def test_output_shape(self):
        cfg = CalibrationConfig(n_features=16, hidden=32)
        model = SignalCalibrator(cfg)
        features = torch.randn(100, 16)
        buy_off, sell_off, intensity = model(features)
        assert buy_off.shape == (100,)
        assert sell_off.shape == (100,)
        assert intensity.shape == (100,)

    def test_batched_output_shape(self):
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        features = torch.randn(4, 100, 16)
        buy_off, sell_off, intensity = model(features)
        assert buy_off.shape == (4, 100)
        assert sell_off.shape == (4, 100)
        assert intensity.shape == (4, 100)

    def test_zero_init_identity(self):
        cfg = CalibrationConfig(base_buy_offset=-0.001, base_sell_offset=0.008, base_intensity=0.5)
        model = SignalCalibrator(cfg)
        model.eval()
        features = torch.randn(50, 16)
        buy_off, sell_off, intensity = model(features)
        torch.testing.assert_close(buy_off, torch.full((50,), -0.001), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sell_off, torch.full((50,), 0.008), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(intensity, torch.full((50,), 0.5), atol=1e-6, rtol=1e-5)

    def test_price_adj_bounded(self):
        cfg = CalibrationConfig(max_price_adj_bps=25.0)
        model = SignalCalibrator(cfg)
        # Force large weights
        with torch.no_grad():
            model.net[-1].weight.fill_(100.0)
            model.net[-1].bias.fill_(100.0)
        features = torch.randn(200, 16)
        buy_off, sell_off, _intensity = model(features)
        max_adj = 25.0 / 10_000.0
        assert (buy_off - cfg.base_buy_offset).abs().max().item() <= max_adj + 1e-6
        assert (sell_off - cfg.base_sell_offset).abs().max().item() <= max_adj + 1e-6

    def test_amount_bounded(self):
        cfg = CalibrationConfig(max_amount_adj=0.30, base_intensity=0.5)
        model = SignalCalibrator(cfg)
        with torch.no_grad():
            model.net[-1].weight.fill_(100.0)
            model.net[-1].bias.fill_(100.0)
        features = torch.randn(200, 16)
        _, _, intensity = model(features)
        assert intensity.min().item() >= -1e-6
        assert intensity.max().item() <= 1.0 + 1e-6

    def test_to_prices(self):
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        model.eval()
        close = torch.tensor([100.0, 200.0, 50.0])
        features = torch.randn(3, 16)
        buy_p, sell_p, _inten = model.to_prices(features, close)
        # zero-init so buy_price = close * (1 + base_buy_offset)
        expected_buy = close * (1.0 + cfg.base_buy_offset)
        expected_sell = close * (1.0 + cfg.base_sell_offset)
        torch.testing.assert_close(buy_p, expected_buy, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(sell_p, expected_sell, atol=1e-3, rtol=1e-4)

    def test_gradient_flows(self):
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        # Perturb last layer so gradients flow through tanh
        with torch.no_grad():
            model.net[-1].weight.fill_(0.1)
        features = torch.randn(50, 16, requires_grad=False)
        close = torch.randn(50).abs() + 1.0
        buy_p, sell_p, inten = model.to_prices(features, close)
        loss = buy_p.mean() + sell_p.mean() + inten.mean()
        loss.backward()
        grads_nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert grads_nonzero >= 3

    def test_custom_base_signals(self):
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        model.eval()
        features = torch.randn(10, 16)
        base_buy = torch.full((10,), -0.002)
        base_sell = torch.full((10,), 0.010)
        base_int = torch.full((10,), 0.7)
        buy_off, sell_off, intensity = model(features, base_buy, base_sell, base_int)
        torch.testing.assert_close(buy_off, base_buy, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sell_off, base_sell, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(intensity, base_int, atol=1e-6, rtol=1e-5)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        cfg = CalibrationConfig(hidden=16, max_price_adj_bps=30.0)
        model = SignalCalibrator(cfg)
        with torch.no_grad():
            model.net[0].weight.fill_(0.1)
        path = tmp_path / "test_calibrator.pt"
        save_calibrator(model, path, cfg, metadata={"test": True})
        loaded_model, loaded_cfg = load_calibrator(path)
        assert loaded_cfg.hidden == 16
        assert loaded_cfg.max_price_adj_bps == 30.0
        features = torch.randn(10, 16)
        with torch.no_grad():
            orig = model(features)
            loaded = loaded_model(features)
        for o, lo in zip(orig, loaded):
            torch.testing.assert_close(o, lo)


class TestDifferentiableSimIntegration:
    def _make_market_data(self, n: int = 500):
        np.random.seed(42)
        price = 100.0
        opens, highs, lows, closes = [], [], [], []
        for _ in range(n):
            ret = np.random.normal(0, 0.01)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(np.random.normal(0, 0.005)))
            lo = min(o, c) * (1 - abs(np.random.normal(0, 0.005)))
            opens.append(o)
            highs.append(h)
            lows.append(lo)
            closes.append(c)
            price = c
        return (
            torch.tensor(opens, dtype=torch.float32),
            torch.tensor(highs, dtype=torch.float32),
            torch.tensor(lows, dtype=torch.float32),
            torch.tensor(closes, dtype=torch.float32),
        )

    def test_gradient_through_soft_sim(self):
        from differentiable_loss_utils import combined_sortino_pnl_loss, simulate_hourly_trades

        opens, highs, lows, closes = self._make_market_data(200)
        features = torch.randn(200, 16)
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        buy_p, sell_p, inten = model.to_prices(features, closes)
        result = simulate_hourly_trades(
            highs=highs.unsqueeze(0),
            lows=lows.unsqueeze(0),
            closes=closes.unsqueeze(0),
            opens=opens.unsqueeze(0),
            buy_prices=buy_p.unsqueeze(0),
            sell_prices=sell_p.unsqueeze(0),
            trade_intensity=inten.unsqueeze(0),
            maker_fee=0.001,
            temperature=0.01,
            decision_lag_bars=2,
        )
        loss = combined_sortino_pnl_loss(result.returns)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)

    def test_binary_sim_runs(self):
        from differentiable_loss_utils import simulate_hourly_trades_binary

        opens, highs, lows, closes = self._make_market_data(200)
        features = torch.randn(200, 16)
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        model.eval()
        with torch.no_grad():
            buy_p, sell_p, inten = model.to_prices(features, closes)
            result = simulate_hourly_trades_binary(
                highs=highs.unsqueeze(0),
                lows=lows.unsqueeze(0),
                closes=closes.unsqueeze(0),
                opens=opens.unsqueeze(0),
                buy_prices=buy_p.unsqueeze(0),
                sell_prices=sell_p.unsqueeze(0),
                trade_intensity=inten.unsqueeze(0),
                maker_fee=0.001,
                decision_lag_bars=2,
            )
        assert result.portfolio_values.shape == (1, 198)
        assert result.portfolio_values[0, -1].item() > 0

    def test_training_step_reduces_loss(self):
        from differentiable_loss_utils import combined_sortino_pnl_loss, simulate_hourly_trades

        opens, highs, lows, closes = self._make_market_data(300)
        features = torch.randn(300, 16)
        cfg = CalibrationConfig()
        model = SignalCalibrator(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            buy_p, sell_p, inten = model.to_prices(features, closes)
            result = simulate_hourly_trades(
                highs=highs.unsqueeze(0),
                lows=lows.unsqueeze(0),
                closes=closes.unsqueeze(0),
                opens=opens.unsqueeze(0),
                buy_prices=buy_p.unsqueeze(0),
                sell_prices=sell_p.unsqueeze(0),
                trade_intensity=inten.unsqueeze(0),
                maker_fee=0.001,
                temperature=0.01,
                decision_lag_bars=2,
            )
            loss = combined_sortino_pnl_loss(result.returns)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestTrainingPipeline:
    def test_time_split(self):
        from train_calibrator import time_split

        train_sl, val_sl, test_sl = time_split(1000)
        assert train_sl == slice(0, 700)
        assert val_sl == slice(700, 850)
        assert test_sl == slice(850, 1000)

    def test_prepare_symbol_tensors_btc(self):
        data_root = REPO / "trainingdatahourly" / "crypto"
        if not (data_root / "BTCUSD.csv").exists():
            pytest.skip("No training data available")
        from train_calibrator import prepare_symbol_tensors

        data = prepare_symbol_tensors("BTCUSD", data_root=data_root)
        assert data["features"].shape[1] == 16
        assert data["closes"].shape[0] == data["features"].shape[0]
        assert data["n_bars"] > 1000
