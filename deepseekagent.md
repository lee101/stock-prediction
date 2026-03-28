## DeepSeek Agent Benchmarks (offline)

Date generated: 2025-10-22  
Data source: `trainingdata/AAPL.csv` (final 30 trading days ending 2023-07-14 UTC)  
Command: `python scripts/deepseek_agent_benchmark.py`

### Methodology
- **Market data** – pulled from cached OHLC bars only; no live downloads or broker calls.  
- **Plans** – deterministic templates (per agent variant) crafted around the most recent trading day in the cache.  
  - *Baseline*: 8-unit buy at market open, close at same-day close.  
  - *Neural*: 5-unit buy with an extended (1% higher) target to mimic neural optimism.  
  - *Entry/Take-Profit*: 6-unit buy with exit at the session high to emulate a bracketed take-profit.  
  - *MaxDiff*: 5-unit limit entry one-third of the way between low/high with exit at the session high.  
  - *Replan*: sequential baseline plans across the last two trading days to capture compounding.  
- **Execution tooling** – `AgentSimulator`, `EntryTakeProfitSimulator`, and `MaxDiffSimulator` from the codebase, all using probe + profit shutdown risk strategies where applicable.  
- **Broker isolation** – `alpaca_wrapper` is stubbed, preventing any outbound API calls and keeping benchmarks offline.

### PnL Summary

| Scenario | Target Date | Realized PnL (USD) | Fees (USD) | Net PnL (USD) |
|----------|-------------|--------------------|-----------:|--------------:|
| Baseline | 2023-07-13  | −0.56              | 1.06       | **−1.62** |
| Neural   | 2023-07-13  | −0.35              | 0.66       | **−1.01** |
| Entry/Take-Profit | 2023-07-13 | 0.01 | 0.80 | **−0.79** |
| MaxDiff  | 2023-07-13  | 0.06               | 0.66       | **−0.61** |

All four single-day scenarios lose money after fees under the chosen parameters, underscoring how sensitive the simulators are to fee drag when trade sizes are modest.

### Replanning Pass (2 sessions)

- Window: 2023-07-13 → 2023-07-14  
- Total return: −0.0097%  
- Annualised: −1.21% (252-day basis)

The follow-up day reduces losses slightly but remains negative; the flat-to-down daily closes in the cached window simply do not offset transaction costs at the configured sizing.

### Reproduction

```bash
# JSON metrics
python scripts/deepseek_agent_benchmark.py --format json

# Console table (default)
python scripts/deepseek_agent_benchmark.py

# Alternative dataset or lookback
python scripts/deepseek_agent_benchmark.py --csv trainingdata/MSFT.csv --symbol MSFT --lookback 60
```

### Next Steps
1. Sweep quantities/exit rules to find regimes where net PnL turns positive; commit updated templates alongside results.  
2. Extend the script to ingest historical DeepSeek plan JSON (when available) so we can compare LLM-generated plans against the deterministic baselines.  
3. Introduce multi-symbol bundles (e.g., AAPL + NVDA) to quantify diversification and realistic fee drag in wider universes.
