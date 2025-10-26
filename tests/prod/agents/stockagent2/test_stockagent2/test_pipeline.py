from __future__ import annotations

import math
from typing import Dict
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd

from stockagent.agentsimulator import AccountPosition, AccountSnapshot, TradingPlan
from stockagent2 import (
    AllocationPipeline,
    ForecastReturnSet,
    LLMViews,
    OptimizationConfig,
    PipelineConfig,
    TickerView,
)
from stockagent2.agentsimulator.plan_builder import PipelinePlanBuilder, PipelineSimulationConfig
from stockagent2.agentsimulator.runner import RunnerConfig, run_pipeline_simulation
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


def test_run_pipeline_simulation_respects_simulation_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    trading_days = pd.date_range("2025-01-01", periods=2, freq="B", tz="UTC")
    frame = pd.DataFrame({"close": [100.0, 101.0], "open": [100.0, 101.0]}, index=trading_days)

    class DummyBundle:
        bars = {"MSFT": frame}

        def trading_days(self) -> list[pd.Timestamp]:
            return list(trading_days)

    monkeypatch.setattr(
        "stockagent2.agentsimulator.runner.fetch_latest_ohlc",
        lambda **_: DummyBundle(),
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

    record: dict[str, object] = {}

    class DummyBuilder:
        def __init__(self, *, pipeline, forecast_adapter, pipeline_config, pipeline_params):
            self.pipeline_config = pipeline_config
            self.pipeline_params = pipeline_params
            record["symbols"] = tuple(pipeline_config.symbols or ())
            self.last_allocation = SimpleNamespace(universe=("MSFT",), weights=np.array([1.0]))

        def build_for_day(self, *, target_timestamp, market_frames, account_snapshot):
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
