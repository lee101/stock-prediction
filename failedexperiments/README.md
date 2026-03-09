# Failed / Parked Experiments

Purpose: keep short, evidence-backed notes on approaches we should not promote until a specific validation gap is closed.

## As of 2026-03-09

### PufferLib PPO (`pufferlib_market/`)
- Status: parked
- Why: headline returns are dominated by a simulator that still lacks realistic slippage, queue priority, and market-impact constraints.
- Evidence: `stockalgorithms.md` notes the current 30-day `2,659x` result is not credible under real fills.
- Unblock condition: rerun with penetration/slippage/volume-aware fills and require the edge to survive 5-10 bp execution friction.

### LLM trading agents (`stockagentdeepseek*/`, `stockagentopus/`)
- Status: retired from promotion queue
- Why: live performance is materially negative and the strategy premise is weak.
- Evidence: `stockalgorithms.md` records roughly `-$8,662` live over 7 days versus trivial backtest gains.
- Unblock condition: none currently planned; only revisit if paired with a measurable non-LLM edge and hard execution controls.

### FastForecaster2 (`fastforecaster2/`)
- Status: parked for live deployment
- Why: forecast direction accuracy is near coin-flip, trading signals are sparse, and returns do not justify the drawdown.
- Evidence: `stockalgorithms.md` reports about `+0.94%` total return, `5.9%` win rate, and `26%` max drawdown.
- Unblock condition: improve signal quality, validate under live-style execution assumptions, and require materially better trade-rate-adjusted goodness scores.

### Limit-entry meta-selector recipes (`unified_hourly_experiment/`)
- Status: parked
- Why: current positive backtests do not survive passive-fill assumptions.
- Evidence: `alpacaprogress4.md` shows the live-style selector turns negative once limit-entry robustness is enforced with realistic execution buffers.
- Unblock condition: find a configuration that remains positive across multi-scenario bar-margin and TTL sweeps, not just market-order entry.

## Update rule

When adding an entry:
- include the repo path
- include the measured failure mode
- cite the latest artifact or writeup
- state the exact condition required before the experiment can re-enter consideration
