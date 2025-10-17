# stockagent2 – Pipeline Simulation Results (2025-10-17)

- **Symbols:** AAPL, MSFT, NVDA, AMD  
- **Lookback / Horizon:** 200-day history, 5 trading days evaluated (3 produced allocations)  
- **Forecast generator:** Stub Toto/Kronos blend (`toto_scale=0.05`, `kronos_bump=0.06`) to avoid heavyweight model loads during smoke testing  
- **Plans generated:** 3  
- **Trades executed:** 6  
- **Ending equity:** \$1,014,886.82 (starting cash \$1,000,000; includes unrealised exposure)  
- **Realized PnL:** \$47.86  
- **Unrealized PnL:** \$15,661.06  
- **Total fees:** \$829.29  
- **Optimizer configuration:** Net exposure target 0.0, gross exposure limit 2.0, weight bounds [-0.8, 0.8], SCS solver

## Reproduction Command

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import numpy as np
import pandas as pd
from hyperparamstore.store import HyperparamStore
from stockagent.agentsimulator import AgentSimulator, fetch_latest_ohlc, AccountSnapshot
from stockagent2.agentsimulator.runner import RunnerConfig, _positions_from_weights, _snapshot_from_positions
from stockagent2.agentsimulator.plan_builder import PipelinePlanBuilder, PipelineSimulationConfig
from stockagent2.agentsimulator.forecast_adapter import CombinedForecastAdapter
from stockagent2.config import OptimizationConfig, PipelineConfig
from stockagent2.optimizer import CostAwareOptimizer
from stockagent2.pipeline import AllocationPipeline

os.environ.setdefault("FAST_TESTING", "1")
DATA_ROOT = Path("trainingdata")
HYPER_ROOT = Path("hyperparams")

class FakeTotoPipeline:
    def __init__(self, config):
        self.scale = 0.05
    def predict(self, *, context, prediction_length, num_samples, samples_per_batch, **kwargs):
        base = float(context[-1])
        samples = np.full((num_samples, prediction_length), base * 1.05, dtype=np.float32)
        return [SimpleNamespace(samples=samples)]

class FakeKronosWrapper:
    def __init__(self, config):
        self.bump = 0.06
        self.max_context = 128
        self.temperature = 0.6
        self.top_p = 0.85
        self.top_k = 0
        self.sample_count = 32
    def predict_series(self, *, data, timestamp_col, columns, pred_len, **kwargs):
        frame = pd.DataFrame(data)
        ts = pd.to_datetime(frame[timestamp_col], utc=True).iloc[-1]
        out = {}
        for column in columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            base = float(series.iloc[-1])
            predicted = base * 1.06
            out[column] = SimpleNamespace(
                absolute=np.array([predicted], dtype=float),
                percent=np.array([(predicted - base) / base], dtype=np.float32),
                timestamps=pd.Index([ts]),
            )
        return out

store = HyperparamStore(HYPER_ROOT)
generator = CombinedForecastAdapter(
    generator=CombinedForecastGenerator(
        data_root=DATA_ROOT,
        hyperparam_root=HYPER_ROOT,
        hyperparam_store=store,
        toto_factory=lambda cfg: FakeTotoPipeline(cfg),
        kronos_factory=lambda cfg: FakeKronosWrapper(cfg),
    )
)
symbols = ("AAPL", "MSFT", "NVDA", "AMD")
runner_cfg = RunnerConfig(symbols=symbols, lookback_days=200, simulation_days=5, starting_cash=1_000_000.0)
opt_cfg = OptimizationConfig(
    net_exposure_target=0.0,
    gross_exposure_limit=2.0,
    long_cap=0.8,
    short_cap=0.8,
    transaction_cost_bps=0.5,
    turnover_penalty_bps=0.3,
    min_weight=-0.8,
    max_weight=0.8,
)
pipe_cfg = PipelineConfig(risk_aversion=1.5, chronos_weight=0.6, timesfm_weight=0.4)
sim_cfg = PipelineSimulationConfig(symbols=symbols, lookback_days=runner_cfg.lookback_days, sample_count=256)

bundle = fetch_latest_ohlc(symbols=symbols, lookback_days=runner_cfg.lookback_days, as_of=datetime.now(timezone.utc))
trading_days = bundle.trading_days()[-runner_cfg.simulation_days:]

optimizer = CostAwareOptimizer(opt_cfg)
pipeline = AllocationPipeline(optimisation_config=opt_cfg, pipeline_config=pipe_cfg, optimizer=optimizer)
builder = PipelinePlanBuilder(pipeline=pipeline, forecast_adapter=generator, pipeline_config=sim_cfg, pipeline_params=pipe_cfg)

plans = []
positions = {}
nav = runner_cfg.starting_cash
for ts in trading_days:
    prices = {}
    for symbol, frame in bundle.bars.items():
        if symbol not in symbols:
            continue
        sliced = frame.loc[: ts]
        if sliced.empty:
            continue
        prices[symbol] = float(sliced.iloc[-1]["close"])
    if not prices:
        continue
    snapshot = _snapshot_from_positions(positions=positions, prices=prices, nav=nav)
    plan = builder.build_for_day(target_timestamp=ts, market_frames=bundle.bars, account_snapshot=snapshot)
    if plan is None or builder.last_allocation is None:
        continue
    plans.append(plan)
    positions = _positions_from_weights(
        weights={sym: w for sym, w in zip(builder.last_allocation.universe, builder.last_allocation.weights)},
        prices=prices,
        nav=nav,
    )

class Proxy:
    def __init__(self, bars):
        self._bars = bars
    def get_symbol_bars(self, symbol):
        return self._bars.get(symbol, pd.DataFrame())

sim = AgentSimulator(
    market_data=Proxy(bundle.bars),
    starting_cash=runner_cfg.starting_cash,
    account_snapshot=_snapshot_from_positions(positions={}, prices={}, nav=runner_cfg.starting_cash),
)
if plans:
    result = sim.simulate(plans)
    print(result.to_dict())
else:
    print({"status": "no_plans"})
PY
```

> **Heads-up:** This harness deliberately relaxes the optimiser bounds and uses synthetic Toto/Kronos forecasts so the pipeline converges quickly on CPU. Replace the stub factories with the real model loaders and tighten the optimisation limits before using the allocator in production.
