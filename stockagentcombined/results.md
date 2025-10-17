# stockagentcombined – Simulation Results (2025-10-17)

- **Symbols:** AAPL, MSFT, NVDA, AMD  
- **Lookback / Horizon:** 180-day history, 5 trading days simulated  
- **Forecast generator:** Stubbed Toto/Kronos blend (`toto_scale=0.05`, `kronos_bump=0.06`) for a fast smoke test without loading full model weights  
- **Plans generated:** 5 (one per trading day)  
- **Trades executed:** 20  
- **Ending equity:** \$999,940.62 (starting cash \$1,000,000)  
- **Realized PnL:** -\$45.36  
- **Unrealized PnL:** \$0.00  
- **Total fees:** \$28.03

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
from stockagent.agentsimulator import (
    AgentSimulator,
    AccountSnapshot,
    ProbeTradeStrategy,
    ProfitShutdownStrategy,
    fetch_latest_ohlc,
)
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombined.agentsimulator import CombinedPlanBuilder, SimulationConfig, build_daily_plans
from stockagentcombined.forecaster import CombinedForecastGenerator

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
generator = CombinedForecastGenerator(
    data_root=DATA_ROOT,
    hyperparam_root=HYPER_ROOT,
    hyperparam_store=store,
    toto_factory=lambda cfg: FakeTotoPipeline(cfg),
    kronos_factory=lambda cfg: FakeKronosWrapper(cfg),
)
symbols = ("AAPL", "MSFT", "NVDA", "AMD")
config = SimulationConfig(symbols=symbols, lookback_days=180, simulation_days=5, starting_cash=1_000_000.0)
bundle = fetch_latest_ohlc(symbols=config.symbols, lookback_days=config.lookback_days, as_of=datetime.now(timezone.utc))
plans = build_daily_plans(
    builder=CombinedPlanBuilder(generator=generator, config=config),
    market_frames={sym: bundle.bars[sym] for sym in symbols},
    trading_days=bundle.trading_days()[-config.simulation_days:],
)
bundle_for_sim = MarketDataBundle(
    bars={sym: bundle.bars[sym] for sym in symbols},
    lookback_days=config.lookback_days,
    as_of=bundle.as_of,
)
sim = AgentSimulator(
    market_data=bundle_for_sim,
    starting_cash=config.starting_cash,
    account_snapshot=AccountSnapshot(
        equity=config.starting_cash,
        cash=config.starting_cash,
        buying_power=None,
        timestamp=datetime.now(timezone.utc),
        positions=[],
    ),
)
result = sim.simulate(plans, strategies=[ProbeTradeStrategy(), ProfitShutdownStrategy()])
print(result.to_dict())
PY
```

> **Note:** The stubbed forecast adapters keep this smoke test fast and GPU-free. For production runs swap in the real Toto/Kronos loaders (see README guidance) so the agent consumes actual model forecasts.
