import types
from dataclasses import dataclass

import pandas as pd
import pytest
import torch
from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator
from neuraldailytraining.runtime import DailyTradingRuntime, TradingPlan


class _FakeConfidenceModel:
    """Minimal stand-in to verify runtime gating logic without loading checkpoints."""

    def __init__(self, trade: float, confidence: float) -> None:
        self.trade = float(trade)
        self.confidence = float(confidence)
        # Expose blocks so runtime won't fabricate a group mask
        self.blocks: list[object] = []

    def __call__(self, features: torch.Tensor, group_mask=None):
        batch, seq, _ = features.shape
        device, dtype = features.device, features.dtype
        logits = torch.zeros(batch, seq, 4, device=device, dtype=dtype)
        trade_logits = torch.full((batch, seq), self.trade, device=device, dtype=dtype).clamp(1e-4, 1 - 1e-4)
        conf_logits = torch.full((batch, seq), self.confidence, device=device, dtype=dtype).clamp(1e-4, 1 - 1e-4)
        logits[..., 2] = torch.logit(trade_logits)
        logits[..., 3] = torch.logit(conf_logits)
        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "trade_amount_logits": logits[..., 2:3],
            "confidence_logits": logits[..., 3:4],
        }

    def decode_actions(
        self,
        outputs,
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        asset_class: torch.Tensor,
        dynamic_offset_pct=None,
    ):
        buy = reference_close * 0.99
        sell = reference_close * 1.01
        trade_amount = torch.sigmoid(outputs["trade_amount_logits"]).squeeze(-1)
        confidence = torch.sigmoid(outputs["confidence_logits"]).squeeze(-1)
        return {
            "buy_price": buy,
            "sell_price": sell,
            "trade_amount": trade_amount,
            "confidence": confidence,
        }


def _build_runtime_for_confidence(trade: float, confidence: float, *, risk_threshold: float = 0.5):
    runtime = DailyTradingRuntime.__new__(DailyTradingRuntime)
    runtime.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime.model = _FakeConfidenceModel(trade, confidence)
    runtime.non_tradable = set()
    runtime.risk_threshold = risk_threshold
    runtime.symbol_to_group_id = {}

    def _prepare(self, symbol: str, *, as_of=None):
        seq_len = 3
        ref = torch.full((seq_len,), 10.0, device=self.device)
        return {
            "symbol": symbol,
            "features": torch.zeros(seq_len, 2, device=self.device),
            "reference": ref,
            "c_high": ref + 0.5,
            "c_low": ref - 0.5,
            "asset_flag": 0.0,
            "group_id": 0,
            "timestamp": "2025-11-20T00:00:00Z",
            "reference_close": float(ref[-1].item()),
        }

    runtime._prepare_symbol_window = types.MethodType(_prepare, runtime)
    return runtime


def test_plan_batch_applies_confidence_gate_and_risk_clamp():
    runtime = _build_runtime_for_confidence(trade=0.8, confidence=0.25, risk_threshold=0.5)

    plans = runtime.plan_batch(["AAPL"])

    assert len(plans) == 1
    # confidence (0.25) should gate the raw trade (0.8) before risk clamp (0.5)
    assert plans[0].trade_amount == pytest.approx(0.2, rel=1e-6)
    assert plans[0].sell_price > plans[0].buy_price


@dataclass
class _FixedPlan:
    symbol: str
    buy_price: float
    sell_price: float
    trade_amount: float


class _StubBuilder:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def build(self, _symbol: str) -> pd.DataFrame:
        return self._frame.copy()


class _StubRuntime:
    def __init__(self, frame: pd.DataFrame, plan: _FixedPlan) -> None:
        self._builder = _StubBuilder(frame)
        self.plan = plan
        self.non_tradable: set[str] = set()

    def plan_batch(self, symbols, *, as_of=None, non_tradable_override=None):
        return [
            TradingPlan(
                symbol=self.plan.symbol,
                timestamp=str(as_of),
                buy_price=self.plan.buy_price,
                sell_price=self.plan.sell_price,
                trade_amount=self.plan.trade_amount,
                reference_close=self.plan.buy_price,
            )
        ]


def test_market_simulator_uses_crypto_fee_default():
    date = pd.to_datetime("2025-11-19", utc=True)
    frame = pd.DataFrame(
        {
            "date": [date],
            "high": [12.0],
            "low": [10.0],
            "close": [11.0],
        }
    )
    plan = _FixedPlan(symbol="BTC-USD", buy_price=10.0, sell_price=12.0, trade_amount=0.5)
    runtime = _StubRuntime(frame, plan)

    simulator = NeuralDailyMarketSimulator(runtime, symbols=["BTC-USD"])
    _results, summary = simulator.run(days=1)

    fee = 0.0008  # default crypto fee in simulator
    initial_cash = 1.0
    qty = min(plan.trade_amount, initial_cash / (plan.buy_price * (1.0 + fee)))
    cash_after_buy = initial_cash - qty * plan.buy_price * (1.0 + fee)
    cash_after_sell = cash_after_buy + qty * plan.sell_price * (1.0 - fee)
    assert summary["final_equity"] == pytest.approx(cash_after_sell, rel=1e-6)
    assert summary["pnl"] > 0
