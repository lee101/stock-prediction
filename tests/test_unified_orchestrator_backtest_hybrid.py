from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator import backtest_hybrid as module


def _make_bars(symbol: str) -> pd.DataFrame:
    index = pd.date_range("2026-03-14 00:00:00+00:00", periods=48, freq="h", tz="UTC")
    close = [100.0 + i * 0.25 for i in range(len(index))]
    return pd.DataFrame(
        {
            "timestamp": index,
            "symbol": symbol,
            "open": [price - 0.1 for price in close],
            "high": [price + 0.4 for price in close],
            "low": [price - 0.5 for price in close],
            "close": close,
            "volume": [1_000.0 + i for i in range(len(index))],
        }
    )


def test_run_backtest_daily_cadence_reuses_one_plan_per_day(monkeypatch) -> None:
    bars = _make_bars("BTCUSD")
    llm_calls: list[dict[str, object]] = []

    monkeypatch.setattr(module, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module, "load_forecasts", lambda symbol, horizon: pd.DataFrame())
    monkeypatch.setattr(module, "build_prompt", lambda **kwargs: "prompt")

    def _fake_call_llm(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        review_thinking_level: str | None = None,
        reasoning_effort: str | None = None,
        reprompt_passes: int = 1,
        reprompt_policy: str = "always",
        review_max_confidence: float | None = None,
        review_model: str | None = None,
        review_cache_namespace: str | None = None,
        call_models: list[str] | None = None,
        provider_models: list[str] | None = None,
    ) -> TradePlan:
        if call_models is not None:
            call_models.extend([model, review_model or model][:reprompt_passes])
        if provider_models is not None:
            provider_models.extend([model, review_model or model][:reprompt_passes])
        llm_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "thinking_level": thinking_level,
                "review_thinking_level": review_thinking_level,
                "reasoning_effort": reasoning_effort,
                "reprompt_passes": reprompt_passes,
                "reprompt_policy": reprompt_policy,
                "review_max_confidence": review_max_confidence,
                "review_model": review_model,
                "review_cache_namespace": review_cache_namespace,
            }
        )
        return TradePlan("long", 99.5, 101.5, 0.75, f"call_{len(llm_calls)}")

    monkeypatch.setattr(module, "call_llm", _fake_call_llm)

    captured: dict[str, pd.DataFrame] = {}

    class _FakeSimulator:
        def __init__(self, config):
            self.config = config

        def run(self, bars_df: pd.DataFrame, actions_df: pd.DataFrame):
            captured["bars"] = bars_df.copy()
            captured["actions"] = actions_df.copy()
            return SimpleNamespace(
                equity_curve=pd.Series([10_000.0, 9_000.0, 10_250.0]),
                fills=[{"symbol": "BTCUSD"}],
                metrics={"sortino": 1.5},
            )

    monkeypatch.setattr(module, "HourlyTraderMarketSimulator", _FakeSimulator)

    results = module.run_backtest(
        symbols=["BTCUSD"],
        days=2,
        modes=["gemini_only"],
        decision_cadence="daily",
        reprompt_passes=2,
        reprompt_policy="actionable",
        review_max_confidence=0.6,
        review_model="gemini-2.5-flash",
        review_thinking_level="LOW",
        review_cache_namespace="review-low",
    )

    assert len(llm_calls) == 2
    assert all(call["reprompt_passes"] == 2 for call in llm_calls)
    assert all(call["reprompt_policy"] == "actionable" for call in llm_calls)
    assert all(call["review_max_confidence"] == 0.6 for call in llm_calls)
    assert all(call["review_model"] == "gemini-2.5-flash" for call in llm_calls)
    assert all(call["review_thinking_level"] == "LOW" for call in llm_calls)
    assert all(call["review_cache_namespace"] == "review-low" for call in llm_calls)
    assert results["gemini_only"]["api_calls"] == 4
    assert results["gemini_only"]["logical_calls"] == 4
    assert results["gemini_only"]["decision_cadence"] == "daily"
    assert results["gemini_only"]["reprompt_passes"] == 2
    assert results["gemini_only"]["reprompt_policy"] == "actionable"
    assert results["gemini_only"]["review_max_confidence"] == 0.6
    assert results["gemini_only"]["review_model"] == "gemini-2.5-flash"
    assert results["gemini_only"]["review_cache_namespace"] == "review-low"
    assert results["gemini_only"]["review_thinking_level"] == "LOW"
    assert results["gemini_only"]["max_drawdown"] == 10.0

    actions = captured["actions"]
    grouped = actions.groupby(actions["timestamp"].dt.floor("D"))
    assert [len(group) for _, group in grouped] == [24, 24]
    assert all(group["buy_price"].nunique() == 1 for _, group in grouped)
    assert all(group["sell_price"].nunique() == 1 for _, group in grouped)


def test_run_backtest_respects_explicit_time_bounds(monkeypatch) -> None:
    bars = _make_bars("BTCUSD")
    monkeypatch.setattr(module, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module, "load_forecasts", lambda symbol, horizon: pd.DataFrame())

    captured: dict[str, pd.DataFrame] = {}

    class _FakeSimulator:
        def __init__(self, config):
            self.config = config

        def run(self, bars_df: pd.DataFrame, actions_df: pd.DataFrame):
            captured["bars"] = bars_df.copy()
            captured["actions"] = actions_df.copy()
            return SimpleNamespace(
                equity_curve=pd.Series([10_000.0, 10_100.0]),
                fills=[],
                metrics={"sortino": 0.0},
            )

    monkeypatch.setattr(module, "HourlyTraderMarketSimulator", _FakeSimulator)

    results = module.run_backtest(
        symbols=["BTCUSD"],
        modes=["rl_only"],
        start_ts="2026-03-14T12:00:00Z",
        end_ts="2026-03-14T23:00:00Z",
    )

    assert results["rl_only"]["fills"] == 0
    bars_used = captured["bars"]
    assert len(bars_used) == 12
    assert bars_used["timestamp"].min() == pd.Timestamp("2026-03-14T12:00:00Z")
    assert bars_used["timestamp"].max() == pd.Timestamp("2026-03-14T23:00:00Z")


def test_run_backtest_actionable_policy_counts_real_calls(monkeypatch) -> None:
    bars = _make_bars("BTCUSD")

    monkeypatch.setattr(module, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module, "load_forecasts", lambda symbol, horizon: pd.DataFrame())
    monkeypatch.setattr(module, "build_prompt", lambda **kwargs: "prompt")

    def _fake_call_llm(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        review_thinking_level: str | None = None,
        reasoning_effort: str | None = None,
        reprompt_passes: int = 1,
        reprompt_policy: str = "always",
        review_max_confidence: float | None = None,
        review_model: str | None = None,
        review_cache_namespace: str | None = None,
        call_models: list[str] | None = None,
        provider_models: list[str] | None = None,
    ) -> TradePlan:
        if call_models is not None:
            call_models.append(model)
        if provider_models is not None:
            provider_models.append(model)
        return TradePlan("hold", 0.0, 0.0, 0.3, "flat hold")

    monkeypatch.setattr(module, "call_llm", _fake_call_llm)

    class _FakeSimulator:
        def __init__(self, config):
            self.config = config

        def run(self, bars_df: pd.DataFrame, actions_df: pd.DataFrame):
            return SimpleNamespace(
                equity_curve=pd.Series([10_000.0, 10_010.0]),
                fills=[],
                metrics={"sortino": 0.0},
            )

    monkeypatch.setattr(module, "HourlyTraderMarketSimulator", _FakeSimulator)

    results = module.run_backtest(
        symbols=["BTCUSD"],
        modes=["gemini_only"],
        decision_cadence="daily",
        reprompt_passes=2,
        reprompt_policy="actionable",
        start_ts="2026-03-14T00:00:00Z",
        end_ts="2026-03-14T23:00:00Z",
    )

    assert results["gemini_only"]["api_calls"] == 1
    assert results["gemini_only"]["logical_calls"] == 1


def test_run_backtest_entry_only_policy_skips_exit_only_review(monkeypatch) -> None:
    bars = _make_bars("BTCUSD")

    monkeypatch.setattr(module, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module, "load_forecasts", lambda symbol, horizon: pd.DataFrame())
    monkeypatch.setattr(module, "build_prompt", lambda **kwargs: "prompt")

    def _fake_call_llm(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        review_thinking_level: str | None = None,
        reasoning_effort: str | None = None,
        reprompt_passes: int = 1,
        reprompt_policy: str = "always",
        review_max_confidence: float | None = None,
        review_model: str | None = None,
        review_cache_namespace: str | None = None,
        call_models: list[str] | None = None,
        provider_models: list[str] | None = None,
    ) -> TradePlan:
        if call_models is not None:
            call_models.append(model)
        if provider_models is not None:
            provider_models.append(model)
        return TradePlan("hold", 0.0, 101.5, 0.6, "exit only")

    monkeypatch.setattr(module, "call_llm", _fake_call_llm)

    class _FakeSimulator:
        def __init__(self, config):
            self.config = config

        def run(self, bars_df: pd.DataFrame, actions_df: pd.DataFrame):
            return SimpleNamespace(
                equity_curve=pd.Series([10_000.0, 10_010.0]),
                fills=[],
                metrics={"sortino": 0.0},
            )

    monkeypatch.setattr(module, "HourlyTraderMarketSimulator", _FakeSimulator)

    results = module.run_backtest(
        symbols=["BTCUSD"],
        modes=["gemini_only"],
        decision_cadence="daily",
        reprompt_passes=2,
        reprompt_policy="entry_only",
        start_ts="2026-03-14T00:00:00Z",
        end_ts="2026-03-14T23:00:00Z",
    )

    assert results["gemini_only"]["api_calls"] == 1
    assert results["gemini_only"]["logical_calls"] == 1


def test_suppress_low_confidence_entry_converts_flat_long_to_hold() -> None:
    plan = TradePlan("long", 99.5, 101.5, 0.39, "weak entry")
    position = module.PositionState()

    gated, suppressed = module._suppress_low_confidence_entry(
        plan,
        position,
        min_plan_confidence=0.4,
    )

    assert suppressed is True
    assert gated.direction == "hold"
    assert gated.buy_price == 0.0
    assert gated.sell_price == 0.0
    assert gated.confidence == 0.39
    assert "suppressed_low_conf_entry(0.39<0.40)" in gated.reasoning


def test_suppress_low_confidence_entry_preserves_exit_for_open_position() -> None:
    plan = TradePlan("long", 99.5, 101.5, 0.2, "keep exit live")
    position = module.PositionState(direction="long", entry_price=99.0, qty=1.0)

    gated, suppressed = module._suppress_low_confidence_entry(
        plan,
        position,
        min_plan_confidence=0.4,
    )

    assert suppressed is False
    assert gated == plan


def test_run_backtest_counts_suppressed_low_confidence_entries(monkeypatch) -> None:
    bars = _make_bars("BTCUSD")

    monkeypatch.setattr(module, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module, "load_forecasts", lambda symbol, horizon: pd.DataFrame())
    monkeypatch.setattr(module, "build_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(
        module,
        "call_llm",
        lambda *args, **kwargs: TradePlan("long", 99.5, 101.5, 0.39, "weak entry"),
    )

    captured: dict[str, pd.DataFrame] = {}

    class _FakeSimulator:
        def __init__(self, config):
            self.config = config

        def run(self, bars_df: pd.DataFrame, actions_df: pd.DataFrame):
            captured["actions"] = actions_df.copy()
            return SimpleNamespace(
                equity_curve=pd.Series([10_000.0, 10_000.0]),
                fills=[],
                metrics={"sortino": 0.0},
            )

    monkeypatch.setattr(module, "HourlyTraderMarketSimulator", _FakeSimulator)

    results = module.run_backtest(
        symbols=["BTCUSD"],
        modes=["gemini_only"],
        decision_cadence="daily",
        min_plan_confidence=0.4,
        start_ts="2026-03-14T00:00:00Z",
        end_ts="2026-03-14T23:00:00Z",
    )

    assert results["gemini_only"]["suppressed_low_conf_entries"] == 1
    assert results["gemini_only"]["min_plan_confidence"] == 0.4
    assert captured["actions"]["buy_price"].eq(0.0).all()
