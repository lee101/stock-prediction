from __future__ import annotations

import math
from datetime import timezone
from typing import Dict
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd

from stockagent.agentsimulator import AccountSnapshot, TradingPlan
from stockagent2 import (
    AllocationPipeline,
    ForecastReturnSet,
    LLMViews,
    OptimizationConfig,
    PipelineConfig,
    TickerView,
)
from stockagent2.agentsimulator.plan_builder import PipelinePlanBuilder, PipelineSimulationConfig
from stockagent2.agentsimulator.plan_builder import _extract_history
from stockagent2.agentsimulator.runner import (
    RunnerConfig,
    _BarsMarketDataAdapter,
    _build_close_price_snapshots,
    _snapshot_from_positions,
    run_pipeline_simulation,
    run_pipeline_simulation_with_diagnostics,
)
from stockagent2.agentsimulator.forecast_adapter import SymbolForecast
from stockagent2.black_litterman import BlackLittermanFuser


def test_llm_views_expected_return_vector_weighting() -> None:
    views = LLMViews(
        asof="2025-10-15",
        universe=["AAPL", "MSFT"],
        views=[
            TickerView(
                ticker="AAPL",
                horizon_days=5,
                mu_bps=50,
                confidence=1.0,
                half_life_days=5,
            ),
            TickerView(
                ticker="AAPL",
                horizon_days=5,
                mu_bps=20,
                confidence=0.5,
                half_life_days=5,
            ),
        ],
    )
    universe = ["AAPL", "MSFT"]
    vector = views.expected_return_vector(universe)

    decay = math.exp(-math.log(2) * 4 / 5)
    daily_1 = (50 / 1e4) / 5
    daily_2 = (20 / 1e4) / 5
    weight_1 = 1.0 * decay
    weight_2 = 0.5 * decay
    expected = (daily_1 * weight_1 + daily_2 * weight_2) / (weight_1 + weight_2)

    assert np.isclose(vector[0], expected)
    assert vector[1] == 0.0


def test_black_litterman_blends_market_and_prior() -> None:
    mu_prior = np.array([0.001, 0.0005])
    sigma_prior = np.array([[0.0025, 0.0008], [0.0008, 0.0016]])
    market_weights = np.array([0.6, 0.4])

    views = LLMViews(
        asof="2025-10-15",
        universe=["AAA", "BBB"],
        views=[
            TickerView(
                ticker="AAA",
                horizon_days=5,
                mu_bps=40,
                confidence=0.9,
                half_life_days=5,
            )
        ],
    )

    fuser = BlackLittermanFuser(tau=0.05, market_prior_weight=0.4)
    result = fuser.fuse(
        mu_prior,
        sigma_prior,
        market_weights=market_weights,
        risk_aversion=3.0,
        views=views,
        universe=("AAA", "BBB"),
    )

    # Posterior mean should lie between the forecast prior and market equilibrium,
    # shifted in the direction of the discretionary view.
    assert result.mu_posterior.shape == mu_prior.shape
    assert result.sigma_posterior.shape == sigma_prior.shape
    assert result.market_weight == 0.4
    view_mean = views.expected_return_vector(("AAA", "BBB"))[0]
    lo = min(view_mean, result.mu_market_equilibrium[0])
    hi = max(view_mean, result.mu_market_equilibrium[0])
    assert lo <= result.mu_posterior[0] <= hi
    assert np.allclose(result.mu_prior, mu_prior)


def test_allocation_pipeline_end_to_end_feasible_weights() -> None:
    universe = ("AAPL", "MSFT", "TSLA")
    rng = np.random.default_rng(42)
    chronos_samples = rng.normal(
        loc=np.array([0.0006, 0.0003, 0.0001]),
        scale=0.0015,
        size=(512, len(universe)),
    )
    timesfm_samples = rng.normal(
        loc=np.array([0.0004, 0.0002, 0.0002]),
        scale=0.001,
        size=(400, len(universe)),
    )

    chronos = ForecastReturnSet(universe=universe, samples=chronos_samples)
    timesfm = ForecastReturnSet(universe=universe, samples=timesfm_samples)

    views = LLMViews(
        asof="2025-10-15",
        universe=list(universe),
        views=[
            TickerView(
                ticker="AAPL",
                horizon_days=5,
                mu_bps=45,
                confidence=0.7,
                half_life_days=10,
            ),
            TickerView(
                ticker="TSLA",
                horizon_days=5,
                mu_bps=-30,
                confidence=0.6,
                half_life_days=8,
            ),
        ],
    )

    optimisation_config = OptimizationConfig(
        net_exposure_target=1.0,
        gross_exposure_limit=1.3,
        long_cap=0.7,
        short_cap=0.1,
        min_weight=-0.2,
        max_weight=0.75,
        sector_exposure_limits={"TECH": 0.9, "AUTO": 0.5},
    )
    pipeline_config = PipelineConfig(
        tau=0.05,
        shrinkage=0.05,
        chronos_weight=0.7,
        timesfm_weight=0.3,
        risk_aversion=3.0,
        market_prior_weight=0.5,
    )
    pipeline = AllocationPipeline(
        optimisation_config=optimisation_config,
        pipeline_config=pipeline_config,
    )

    sector_map: Dict[str, str] = {"AAPL": "TECH", "MSFT": "TECH", "TSLA": "AUTO"}
    prev_weights = np.array([0.45, 0.35, 0.2])
    market_caps = {"AAPL": 3.0, "MSFT": 2.5, "TSLA": 0.8}

    result = pipeline.run(
        chronos=chronos,
        timesfm=timesfm,
        llm_views=views,
        previous_weights=prev_weights,
        sector_map=sector_map,
        market_caps=market_caps,
    )

    weights = result.weights
    assert np.isclose(weights.sum(), optimisation_config.net_exposure_target, atol=1e-6)
    assert np.sum(np.abs(weights)) <= optimisation_config.gross_exposure_limit + 1e-6
    assert np.all(weights <= optimisation_config.long_cap + 1e-6)
    assert np.all(weights >= -optimisation_config.short_cap - 1e-6)
    for sector, exposure in result.optimizer.sector_exposures.items():
        limit = optimisation_config.sector_exposure_limits[sector]
        assert abs(exposure) <= limit + 1e-6
    assert result.optimizer.status.lower().startswith("optimal") or result.optimizer.status == "SLSQP_success"
    assert result.diagnostics["llm_view_count"] == 2.0


class DummyForecastAdapter:
    def __init__(self, forecasts: Dict[str, SymbolForecast]) -> None:
        self._forecasts = forecasts

    def forecast(self, symbol: str, history: pd.DataFrame) -> SymbolForecast | None:
        return self._forecasts.get(symbol)


def _make_history(prices: Sequence[float], start: str = "2025-01-01") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame({"close": prices, "open": prices}, index=index)


def test_pipeline_plan_builder_generates_instructions() -> None:
    universe = ("AAPL", "MSFT")
    optimisation_config = OptimizationConfig(
        net_exposure_target=1.0,
        gross_exposure_limit=1.2,
        long_cap=0.8,
        short_cap=0.2,
        min_weight=-0.2,
        max_weight=0.8,
    )
    pipeline_config = PipelineConfig(
        tau=0.05,
        shrinkage=0.1,
        chronos_weight=0.6,
        timesfm_weight=0.4,
        market_prior_weight=0.4,
        annualisation_periods=40,
    )
    pipeline = AllocationPipeline(
        optimisation_config=optimisation_config,
        pipeline_config=pipeline_config,
    )

    forecasts = {
        "AAPL": SymbolForecast(
            symbol="AAPL",
            last_close=200.0,
            predicted_close=204.0,
            entry_price=201.0,
            average_price_mae=1.5,
        ),
        "MSFT": SymbolForecast(
            symbol="MSFT",
            last_close=300.0,
            predicted_close=297.0,
            entry_price=298.0,
            average_price_mae=1.2,
        ),
    }
    adapter = DummyForecastAdapter(forecasts)

    builder = PipelinePlanBuilder(
        pipeline=pipeline,
        forecast_adapter=adapter,
        pipeline_config=PipelineSimulationConfig(
            symbols=universe,
            sample_count=256,
            min_trade_value=10.0,
            min_volatility=0.001,
            llm_horizon_days=3,
        ),
        pipeline_params=pipeline_config,
    )

    market_frames = {
        "AAPL": _make_history(np.linspace(180, 200, 15)),
        "MSFT": _make_history(np.linspace(280, 300, 15)),
    }
    target_timestamp = market_frames["AAPL"].index[-1] + pd.Timedelta(days=1)
    snapshot = AccountSnapshot(
        equity=1_000_000.0,
        cash=1_000_000.0,
        buying_power=None,
        timestamp=pd.Timestamp.utcnow().to_pydatetime(),
        positions=[],
    )

    plan = builder.build_for_day(
        target_timestamp=target_timestamp,
        market_frames=market_frames,
        account_snapshot=snapshot,
    )

    assert plan is not None
    assert builder.last_allocation is not None
    assert len(plan.instructions) > 0


def test_pipeline_plan_builder_records_no_instruction_diagnostics() -> None:
    universe = ("AAPL",)
    optimisation_config = OptimizationConfig(
        net_exposure_target=1.0,
        gross_exposure_limit=1.2,
        long_cap=1.0,
        short_cap=0.0,
        min_weight=0.0,
        max_weight=1.0,
    )
    pipeline_config = PipelineConfig(
        tau=0.05,
        shrinkage=0.1,
        chronos_weight=0.6,
        timesfm_weight=0.4,
        market_prior_weight=0.4,
        annualisation_periods=40,
    )
    pipeline = AllocationPipeline(
        optimisation_config=optimisation_config,
        pipeline_config=pipeline_config,
    )
    adapter = DummyForecastAdapter(
        {
            "AAPL": SymbolForecast(
                symbol="AAPL",
                last_close=100.0,
                predicted_close=101.0,
                entry_price=100.5,
                average_price_mae=0.5,
            )
        }
    )
    builder = PipelinePlanBuilder(
        pipeline=pipeline,
        forecast_adapter=adapter,
        pipeline_config=PipelineSimulationConfig(
            symbols=universe,
            sample_count=64,
            min_trade_value=1_000_000.0,
            min_volatility=0.001,
            llm_horizon_days=3,
        ),
        pipeline_params=pipeline_config,
    )
    market_frames = {"AAPL": _make_history(np.linspace(90, 100, 15))}
    target_timestamp = market_frames["AAPL"].index[-1] + pd.Timedelta(days=1)
    snapshot = AccountSnapshot(
        equity=10_000.0,
        cash=10_000.0,
        buying_power=None,
        timestamp=pd.Timestamp.utcnow().to_pydatetime(),
        positions=[],
    )

    plan = builder.build_for_day(
        target_timestamp=target_timestamp,
        market_frames=market_frames,
        account_snapshot=snapshot,
    )

    assert plan is None
    assert builder.last_build_diagnostics is not None
    assert builder.last_build_diagnostics.status == "no_instructions"
    assert builder.last_build_diagnostics.skipped_min_trade_symbols == ("AAPL",)
    assert builder.last_build_diagnostics.forecasted_symbols == ("AAPL",)


def test_pipeline_plan_builder_respects_configured_history_requirement() -> None:
    pipeline = AllocationPipeline(
        optimisation_config=OptimizationConfig(),
        pipeline_config=PipelineConfig(annualisation_periods=40),
    )
    adapter = DummyForecastAdapter(
        {
            "AAPL": SymbolForecast(
                symbol="AAPL",
                last_close=100.0,
                predicted_close=101.0,
                entry_price=100.5,
                average_price_mae=0.5,
            )
        }
    )
    builder = PipelinePlanBuilder(
        pipeline=pipeline,
        forecast_adapter=adapter,
        pipeline_config=PipelineSimulationConfig(
            symbols=("AAPL",),
            history_min_period_divisor=2,
        ),
        pipeline_params=PipelineConfig(annualisation_periods=40),
    )

    plan = builder.build_for_day(
        target_timestamp=pd.Timestamp("2025-01-31", tz="UTC"),
        market_frames={"AAPL": _make_history(np.linspace(90, 100, 15))},
        account_snapshot=AccountSnapshot(
            equity=10_000.0,
            cash=10_000.0,
            buying_power=None,
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
            positions=[],
        ),
    )

    assert plan is None
    assert builder.last_build_diagnostics is not None
    assert builder.last_build_diagnostics.status == "no_history"
    assert builder.last_build_diagnostics.insufficient_history_symbols == ("AAPL",)


def test_pipeline_plan_builder_respects_configured_sampling_and_half_life_heuristics() -> None:
    captured: dict[str, object] = {}

    class CapturingPipeline:
        def run(self, *, chronos, timesfm, llm_views, previous_weights, **kwargs):
            captured["chronos"] = chronos
            captured["timesfm"] = timesfm
            captured["llm_views"] = llm_views
            return SimpleNamespace(weights=np.array([0.5]), diagnostics={})

    builder = PipelinePlanBuilder(
        pipeline=CapturingPipeline(),
        forecast_adapter=DummyForecastAdapter(
            {
                "AAPL": SymbolForecast(
                    symbol="AAPL",
                    last_close=100.0,
                    predicted_close=103.0,
                    entry_price=101.0,
                    average_price_mae=2.0,
                )
            }
        ),
        pipeline_config=PipelineSimulationConfig(
            symbols=("AAPL",),
            sample_count=512,
            min_trade_value=10.0,
            min_volatility=0.001,
            llm_horizon_days=40,
            secondary_sample_scale=2.0,
            sample_return_clip=0.5,
            min_view_half_life_days=7,
            max_view_half_life_days=9,
            rng_seed=7,
        ),
        pipeline_params=PipelineConfig(annualisation_periods=40),
    )

    plan = builder.build_for_day(
        target_timestamp=pd.Timestamp("2025-02-28", tz="UTC"),
        market_frames={"AAPL": _make_history(np.linspace(90, 100, 40))},
        account_snapshot=AccountSnapshot(
            equity=10_000.0,
            cash=10_000.0,
            buying_power=None,
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
            positions=[],
        ),
    )

    assert plan is not None
    chronos = captured["chronos"]
    timesfm = captured["timesfm"]
    llm_views = captured["llm_views"]
    assert np.std(timesfm.samples[:, 0]) > np.std(chronos.samples[:, 0]) * 1.5
    assert llm_views.views[0].half_life_days == 9


def test_pipeline_plan_builder_uses_stable_sampling_for_same_day_retries() -> None:
    captured_runs: list[tuple[np.ndarray, np.ndarray]] = []

    class CapturingPipeline:
        def run(self, *, chronos, timesfm, llm_views, previous_weights, **kwargs):
            captured_runs.append((chronos.samples.copy(), timesfm.samples.copy()))
            return SimpleNamespace(weights=np.array([0.5]), diagnostics={})

    builder = PipelinePlanBuilder(
        pipeline=CapturingPipeline(),
        forecast_adapter=DummyForecastAdapter(
            {
                "AAPL": SymbolForecast(
                    symbol="AAPL",
                    last_close=100.0,
                    predicted_close=103.0,
                    entry_price=101.0,
                    average_price_mae=2.0,
                )
            }
        ),
        pipeline_config=PipelineSimulationConfig(
            symbols=("AAPL",),
            sample_count=64,
            min_trade_value=10.0,
            rng_seed=11,
        ),
        pipeline_params=PipelineConfig(annualisation_periods=40),
    )

    common_kwargs = dict(
        target_timestamp=pd.Timestamp("2025-02-28", tz="UTC"),
        market_frames={"AAPL": _make_history(np.linspace(90, 100, 40))},
        account_snapshot=AccountSnapshot(
            equity=10_000.0,
            cash=10_000.0,
            buying_power=None,
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
            positions=[],
        ),
    )

    first_plan = builder.build_for_day(**common_kwargs)
    builder._previous_weights = {}
    second_plan = builder.build_for_day(**common_kwargs)

    assert first_plan is not None
    assert second_plan is not None
    assert len(captured_runs) == 2
    np.testing.assert_allclose(captured_runs[0][0], captured_runs[1][0])
    np.testing.assert_allclose(captured_runs[0][1], captured_runs[1][1])


def test_pipeline_simulation_config_validates_heuristic_bounds() -> None:
    with pytest.raises(ValueError, match="history_min_period_divisor"):
        PipelineSimulationConfig(history_min_period_divisor=0)
    with pytest.raises(ValueError, match="max_view_half_life_days"):
        PipelineSimulationConfig(min_view_half_life_days=5, max_view_half_life_days=4)


def test_extract_history_uses_strict_pre_target_window() -> None:
    frame = _make_history([100.0, 101.0, 102.0, 103.0])
    target_timestamp = frame.index[2]

    histories, latest_prices = _extract_history(
        market_frames={"AAPL": frame},
        target_timestamp=target_timestamp,
        min_length=2,
    )

    history = histories["AAPL"]
    assert list(history.index) == list(frame.index[:2])
    assert latest_prices == {"AAPL": 101.0}


def test_extract_history_skips_symbols_without_enough_pre_target_rows() -> None:
    frame = _make_history([100.0, 101.0, 102.0])

    histories, latest_prices = _extract_history(
        market_frames={"AAPL": frame},
        target_timestamp=frame.index[1],
        min_length=2,
    )

    assert histories == {}
    assert latest_prices == {}


def test_snapshot_from_positions_uses_aware_utc_timestamp() -> None:
    snapshot = _snapshot_from_positions(
        positions={"AAPL": 10.0},
        prices={"AAPL": 100.0},
        nav=2_000.0,
    )

    assert snapshot.timestamp.tzinfo == timezone.utc
    assert snapshot.timestamp.utcoffset() is not None


def test_build_close_price_snapshots_forward_fills_sparse_history() -> None:
    trading_days = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
    sparse_history = pd.DataFrame(
        {"close": [99.0, 100.0, 105.0]},
        index=pd.DatetimeIndex([trading_days[0], trading_days[0], trading_days[2]]),
    )

    snapshots = _build_close_price_snapshots(
        bars={"MSFT": sparse_history},
        symbols=("MSFT",),
        trading_days=trading_days,
    )

    assert snapshots[trading_days[0]] == {"MSFT": 100.0}
    assert snapshots[trading_days[1]] == {"MSFT": 100.0}
    assert snapshots[trading_days[2]] == {"MSFT": 105.0}


def test_build_close_price_snapshots_returns_empty_without_trading_days() -> None:
    snapshots = _build_close_price_snapshots(
        bars={
            "MSFT": pd.DataFrame(
                {"close": [100.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")]),
            )
        },
        symbols=("MSFT",),
        trading_days=(),
    )

    assert snapshots == {}


def test_bars_market_data_adapter_uppercases_symbols_and_returns_copy() -> None:
    frame = pd.DataFrame(
        {"close": [100.0], "open": [99.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")]),
    )
    adapter = _BarsMarketDataAdapter(bars={"MSFT": frame})

    result = adapter.get_symbol_bars("msft")

    assert result.equals(frame)
    assert result is not frame
    result.loc[:, "close"] = [200.0]
    assert frame["close"].iloc[0] == 100.0
    assert adapter.get_symbol_bars("AAPL").empty


def test_run_pipeline_simulation_respects_simulation_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_days = pd.date_range("2025-01-01", periods=2, freq="B", tz="UTC")
    frame = pd.DataFrame({"close": [100.0, 101.0], "open": [100.0, 101.0]}, index=trading_days)
    record: dict[str, object] = {}

    class DummyBundle:
        bars = {"MSFT": frame}

        def trading_days(self) -> list[pd.Timestamp]:
            return list(trading_days)

    def _fake_fetch_latest_ohlc(**kwargs):
        record["as_of"] = kwargs["as_of"]
        return DummyBundle()

    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.fetch_latest_ohlc",
        _fake_fetch_latest_ohlc,
    )
    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.CostAwareOptimizer",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.AllocationPipeline",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.CombinedForecastGenerator",
        lambda: object(),
    )

    class DummyBuilder:
        def __init__(self, *, pipeline, forecast_adapter, pipeline_config, pipeline_params):
            self.pipeline_config = pipeline_config
            self.pipeline_params = pipeline_params
            record["symbols"] = tuple(pipeline_config.symbols or ())
            self.last_allocation = SimpleNamespace(universe=("MSFT",), weights=np.array([1.0]))

        def build_for_day(self, *, target_timestamp, market_frames, account_snapshot):
            record["snapshot_timestamp"] = account_snapshot.timestamp
            return TradingPlan(target_date=target_timestamp.date(), instructions=[])

    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.PipelinePlanBuilder",
        DummyBuilder,
    )
    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.CombinedForecastAdapter",
        lambda generator: object(),
    )

    result = run_pipeline_simulation(
        runner_config=RunnerConfig(symbols=("AAPL", "MSFT"), lookback_days=20, simulation_days=1),
        optimisation_config=OptimizationConfig(),
        pipeline_config=PipelineConfig(),
        simulation_config=PipelineSimulationConfig(symbols=("MSFT",), sample_count=16),
    )

    assert result is not None
    assert len(result.plans) == 1
    assert result.simulation.starting_cash == RunnerConfig().starting_cash
    assert record["symbols"] == ("MSFT",)
    assert record["as_of"].tzinfo == timezone.utc
    assert record["snapshot_timestamp"].tzinfo == timezone.utc
    assert result.market_data_summary.symbols_requested == ("MSFT",)
    assert result.market_data_summary.loaded_symbols == ("MSFT",)
    assert result.market_data_summary.empty_symbols == ()
    assert result.market_data_summary.first_trading_day == trading_days[-1].date().isoformat()
    assert result.market_data_summary.last_trading_day == trading_days[-1].date().isoformat()
    assert isinstance(result.simulator.market_data, _BarsMarketDataAdapter)
    assert result.simulator.market_data.get_symbol_bars("MSFT").equals(frame)


def test_run_pipeline_simulation_returns_none_without_trading_days(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyBundle:
        bars: dict[str, pd.DataFrame] = {}

        def trading_days(self) -> list[pd.Timestamp]:
            return []

    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.fetch_latest_ohlc",
        lambda **_: DummyBundle(),
    )
    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.CostAwareOptimizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("optimizer should not be constructed")),
    )

    result = run_pipeline_simulation(
        runner_config=RunnerConfig(symbols=("MSFT",), lookback_days=20, simulation_days=1),
        optimisation_config=OptimizationConfig(),
        pipeline_config=PipelineConfig(),
    )

    assert result is None


def test_run_pipeline_simulation_with_diagnostics_reports_failure_context_without_trading_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyBundle:
        bars: dict[str, pd.DataFrame] = {}

        def trading_days(self) -> list[pd.Timestamp]:
            return []

    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.fetch_latest_ohlc",
        lambda **_: DummyBundle(),
    )

    attempt = run_pipeline_simulation_with_diagnostics(
        runner_config=RunnerConfig(symbols=("MSFT",), lookback_days=20, simulation_days=1),
        optimisation_config=OptimizationConfig(),
        pipeline_config=PipelineConfig(),
    )

    assert attempt.result is None
    assert attempt.failure_reason == "No trading days available for simulation."
    assert attempt.market_data_summary.symbols_requested == ("MSFT",)
    assert attempt.build_diagnostics == ()
