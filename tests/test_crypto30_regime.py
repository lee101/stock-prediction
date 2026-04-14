"""Tests for BTC MA regime filter in crypto30 daily bot."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def make_btc_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                          "close": closes, "volume": [1.0] * len(closes)}, index=dates)


class TestIsBullRegime:
    """Test Crypto30Ensemble.is_bull_regime without loading real checkpoints."""

    def _make_ensemble(self, regime_ma_period=15, min_confidence=0.0):
        from trade_crypto30_daily import Crypto30Ensemble
        ens = object.__new__(Crypto30Ensemble)
        ens.regime_ma_period = regime_ma_period
        ens.min_confidence = min_confidence
        ens.btc_symbol = "BTCUSD"
        return ens

    def test_disabled_filter(self):
        ens = self._make_ensemble(regime_ma_period=0)
        assert ens.is_bull_regime({}) is True

    def test_no_btc_data(self):
        ens = self._make_ensemble(15)
        assert ens.is_bull_regime({}) is True

    def test_insufficient_data(self):
        ens = self._make_ensemble(15)
        btc_df = make_btc_df([100.0] * 10)  # only 10 bars, need 15
        assert ens.is_bull_regime({"BTCUSD": btc_df}) is True

    def test_bull_regime(self):
        ens = self._make_ensemble(15)
        # 14 bars at 100, then current at 120 -> MA15 = (14*100+120)/15 = 101.33, current 120 > 101.33
        closes = [100.0] * 14 + [120.0]
        assert ens.is_bull_regime({"BTCUSD": make_btc_df(closes)}) is True

    def test_bear_regime(self):
        ens = self._make_ensemble(15)
        # 14 bars at 100, then current at 80 -> MA15 = (14*100+80)/15 = 98.67, current 80 < 98.67
        closes = [100.0] * 14 + [80.0]
        assert ens.is_bull_regime({"BTCUSD": make_btc_df(closes)}) is False

    def test_exact_equal_is_not_bull(self):
        ens = self._make_ensemble(15)
        # all 100 -> MA = 100, close = 100, not > so bear
        closes = [100.0] * 15
        assert ens.is_bull_regime({"BTCUSD": make_btc_df(closes)}) is False

    def test_longer_history_uses_last_n(self):
        ens = self._make_ensemble(15)
        # 50 bars trending up, last 15 avg should be high
        closes = list(range(50, 100))
        ma15 = np.mean(closes[-15:])
        expected = closes[-1] > ma15
        assert ens.is_bull_regime({"BTCUSD": make_btc_df(closes)}) == expected

    def test_regime_filter_forces_flat(self):
        """When bear regime, get_ensemble_signal returns flat without running model."""
        from trade_crypto30_daily import Crypto30Ensemble, TradingSignal
        ens = self._make_ensemble(15)
        ens.traders = [MagicMock()]
        ens.features_per_sym = 16
        ens.num_symbols = 30
        ens.num_actions = 61
        ens.device = MagicMock()

        # Bear regime
        closes = [100.0] * 14 + [80.0]
        daily_dfs = {"BTCUSD": make_btc_df(closes)}

        from trade_crypto30_daily import PortfolioState
        portfolio = PortfolioState()
        signal = ens.get_ensemble_signal(daily_dfs, {}, portfolio)
        assert signal.action == "flat"
        # Model should NOT have been called
        for t in ens.traders:
            t.policy.assert_not_called()


class TestConfidenceGate:
    def test_confidence_gate_rejects_low_conf(self):
        """Confidence gate forces flat when confidence < threshold."""
        import torch
        from trade_crypto30_daily import Crypto30Ensemble, PortfolioState

        ens = object.__new__(Crypto30Ensemble)
        ens.regime_ma_period = 0  # disable regime filter
        ens.min_confidence = 0.50  # high gate, will reject typical ~0.25 conf
        ens.btc_symbol = "BTCUSD"
        ens.symbols = ["BTCUSD"] * 30
        ens.num_symbols = 30
        ens.features_per_sym = 16
        ens.num_actions = 61
        ens.device = torch.device("cpu")

        # Mock trader that returns uniform-ish logits (low confidence)
        mock_trader = MagicMock()
        logits = torch.randn(1, 61)  # near-uniform -> low confidence
        mock_trader.policy.return_value = (logits, torch.tensor([0.0]))
        ens.traders = [mock_trader]

        closes = [100.0] * 20 + [120.0]
        daily_dfs = {"BTCUSD": make_btc_df(closes)}
        portfolio = PortfolioState()
        signal = ens.get_ensemble_signal(daily_dfs, {}, portfolio)
        # With uniform logits, softmax confidence for any action is ~1/61 ≈ 0.016
        # which is well below 0.50 threshold
        assert signal.action == "flat"

    def test_confidence_gate_allows_high_conf(self):
        """Confidence gate passes when confidence >= threshold."""
        import torch
        from trade_crypto30_daily import Crypto30Ensemble, PortfolioState

        ens = object.__new__(Crypto30Ensemble)
        ens.regime_ma_period = 0
        ens.min_confidence = 0.01  # very low gate
        ens.btc_symbol = "BTCUSD"
        ens.symbols = ["BTCUSD"] * 30
        ens.num_symbols = 30
        ens.features_per_sym = 16
        ens.num_actions = 61
        ens.device = torch.device("cpu")

        # Mock trader that returns very peaked logits (high confidence on action 1)
        mock_trader = MagicMock()
        logits = torch.full((1, 61), -100.0)
        logits[0, 1] = 10.0  # very confident on action 1 (long BTCUSD)
        mock_trader.policy.return_value = (logits, torch.tensor([1.0]))
        ens.traders = [mock_trader]

        closes = [100.0] * 20 + [120.0]
        daily_dfs = {"BTCUSD": make_btc_df(closes)}
        portfolio = PortfolioState()
        signal = ens.get_ensemble_signal(daily_dfs, {}, portfolio)
        assert signal.action != "flat"
        assert "long" in signal.action
