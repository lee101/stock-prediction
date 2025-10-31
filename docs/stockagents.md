# Stock Agent Simulator Shootout

Date: 2025-10-17  
Universe: `AAPL`, `MSFT`, `TSLA`  
Horizon: 3 recent trading days sampled from a shared synthetic market bundle (`open`, `high`, `low`, `close` smoothed trends with light cyclical noise).  
Broker costs: default per-agent fee settings (equity taker fee 5 bps, crypto fee unused).

| Agent | Realized P&L (USD) | Fees (USD) | Net P&L (USD) | Notes |
|-------|--------------------|-----------:|---------------:|-------|
| `stockagent` | 4 075.71 | 16.51 | **4 059.19** | Single-coach GPT planner proxy: buys 50 AAPL on day 1 open, rolls, exits day 3 close. Highest edge thanks to directional trend capture. |
| `stockagent2` | 1 909.91 | 1 954.53 | −44.62 | Pipeline allocator builds diversified weights but incurs higher turnover/fees; auto-close legs made the run slightly loss-making. |
| `stockagentcombined` | 38.59 | 27.17 | 11.43 | Toto/Kronos stub forecasts deliver modest one-day alpha; execution near breakeven after costs. |
| `stockagentindependant` | 23.18 | 4.11 | 19.06 | Stateless per-ticker loops stay light on fees but also light on gross alpha. |

## Methodology
- **Shared Market Data:** All agents saw identical three-day OHLC frames generated via `np.linspace` trends to remove data leakage between variants.
- **Execution Harness:** Each agent ran through its native `AgentSimulator`. For plan-driven agents (`stockagent`, `stockagentindependant`) we reused their test-proven instruction templates adjusted to the shared data.  
- **Combined Forecast Agent:** `CombinedPlanBuilder` paired with a stubbed Toto/Kronos forecaster produced open/close instructions automatically.  
- **Pipeline Agent:** `PipelinePlanBuilder` consumed dummy probabilistic forecasts → Black–Litterman fusion → convex allocator. We appended matching market-close exits to crystallize P&L.
- **Profit Metric:** Net P&L = realized P&L − fees. Unrealized P&L was flat after forced closes.

## Takeaways
1. **`stockagent` remains the profit leader** under identical data, delivering ~4.1 k net USD. Even with simple deterministic prompts it leverages directional conviction efficiently.  
2. **`stockagent2` needs fee-aware tuning.** The allocator found profitable views, but turnover made it net negative; reducing transaction-cost parameters or adding explicit exit horizons should help.  
3. **`stockagentcombined` and `stockagentindependant` stay roughly breakeven** in this regime. They are good baselines but trail the stateful planner by an order of magnitude.

## Next Steps
1. Feed `stockagent2` actual Chronos/TimesFM distributions and retune turnover penalties to claw back fees.  
2. Expand the shootout to a rolling 60-day walk-forward with realistic slippage to validate durability.  
3. Instrument the evaluation harness to dump per-day equity curves for Git-tracked regressions.  
4. Explore hybrid workflow: `stockagent` generates view scaffolds, `stockagent2` optimizes sizing, then execute via shared simulator to blend strengths.

---

# Ten-Day Multi-Asset Trial

- **Date Range:** 2025-06-02 → 2025-06-13 (10 trading days)  
- **Universe:** `COUR, GOOG, TSLA, NVDA, AAPL, ADSK, ADBE, COIN, META, AMZN, AMD, INTC, BTCUSD, ETHUSD, UNIUSD`  
- **Market Data:** shared synthetic OHLC bundle per symbol with deterministic trend + light noise  
- **Starting NAV:** \$1.5MM for each agent  
- **Execution:** native simulators with default fee schedules (equities 5 bps, crypto 15 bps)

| Agent | Realized P&L (USD) | Fees (USD) | Net P&L (USD) | Commentary |
|-------|-------------------:|-----------:|--------------:|------------|
| `stockagentindependant` | 2 761.63 | 50.88 | **2 710.76** | Stateless per-symbol trader quietly tops the table; lower fee drag thanks to lighter position sizing. |
| `stockagent` | 2 758.71 | 104.68 | 2 654.03 | Coordinated GPT planner remains competitive but pays ~2× the fees of the independent loop. |
| `stockagentcombined` | −9 799.18 | 15 200.10 | −24 999.28 | Stubbed Toto/Kronos views produced aggressive orders across all symbols, leading to outsized turnover and fee bleed; needs base-qty retune + cost-aware guardrails. |
| `stockagent2` | 0.00 | 7 546.80 | −7 546.80 | Pipeline allocator emitted negligible entry orders yet incurred forced-exit costs on the last day—highlighting that trade thresholds and closing logic must be reworked. |

## Optimization Targets
### `stockagentindependant`
- Moderate the min/max probe multipliers so late-day exits don’t overshoot target delta.
- Layer in sector/asset-class caps to avoid concentration if the universe grows beyond this list.

### `stockagent`
- Introduce cost-aware sizing (e.g., L2 turnover penalties) to capture the same alpha with fewer shares.
- Harmonize crypto lot sizing—current flat 20-share rule over-trades high-priced tokens.

### `stockagentcombined`
- Drop `base_quantity` from 10→3 and tie quantity to forecast confidence; current settings over-allocate even to flat views.  
- Integrate pipeline risk module or `stockagent2` allocator to enforce portfolio-level caps before placing simulator orders.

### `stockagent2`
- Lower `min_trade_value` ≤ \$100 and widen confidence decay so legitimate views produce entry orders.  
- Add explicit exit scheduling (e.g., day+2 close) instead of back-filling exits at simulation end, which currently burns fees without building exposure.  
- Once trades flow, revisit turnover/transaction cost weights so net P&L is positive.

## Next Workstream
1. Use `.venv313` + full requirements (with `qlib`) to run the real Kronos/Chronos pipelines, replacing today’s synthetic stubs.  
2. Wire a reusable benchmarking script that ingests a universe, horizon, and agent list, then emits CSV/Markdown summaries for regression tracking.  
3. After tuning the underperformers, re-run the 10-day benchmark to confirm closing the gap before moving to historical data.
