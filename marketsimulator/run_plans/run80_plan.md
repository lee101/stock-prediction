# Run 80 – Controlled ETF Onboarding Plan

*Last updated: 2025-10-24T17:34Z (UTC)*

## Objective
Probe the highest-momentum ETF candidates under tighter risk controls to see whether the simulator can clear the strategy-return gate without eroding PnL. Focus is on minimal configuration edits so we can revert quickly if the run underperforms.

## Candidate Snapshot
Derived from the latest diagnostics (`candidate_momentum.md`, `candidate_forecast_gate_report.md`):

| Symbol | Trailing Avg % | Latest % | Latest SMA | Gate Status |
|--------|----------------|----------|------------|-------------|
| SMH | ~57.6% | 57.6% | 316.7 | Strategy return ≈ 0.006 (fails 0.015 floor) |
| QQQ | ~40.3% | 40.3% | 590.8 | Strategy return ≈ 0.0056 (fails 0.015 floor) |
| SOXX | ~34.7% | 34.7% | 266.1 | Strategy return ≈ 0.0071 (fails 0.015 floor) |
| XLY | ~34.0% | 34.0% | 235.8 | Strategy return ≈ 0.0063 (fails 0.015 floor) |

All four ETFs currently fail the entry gate only because strategy-return is a few bps below the default 0.015 threshold; no other guard (predicted move, drawdown, per-symbol max) is blocking entries.

## Proposed Run 80 Configuration
1. **Symbols:** `AAPL MSFT NVDA AMZN SOXX SMH`  
   (Keep MSFT/NVDA probes active so we have a baseline reference; swap QQQ for SMH to diversify sector exposure.)
2. **Gate Adjustments (temporary):**
   - `MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP`: `SMH:0.010`, `SOXX:0.015`, others unchanged.  
     *Rationale:* SMH is closest to passing; test with a 1.0% floor while keeping SOXX at default to compare behaviour.
   - `MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP`: `SMH:0.0075`, `SOXX:0.01`
3. **Risk Caps:**
   - `MARKETSIM_SYMBOL_KELLY_SCALE_MAP`: `SMH:0.10`, `SOXX:0.10` (reduced from 0.15 to limit leverage while gates are relaxed).
   - `MARKETSIM_SYMBOL_MAX_ENTRIES_MAP`: `SMH:4`, `SOXX:4`
   - `MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP`: `SMH:7200`, `SOXX:7200`
4. **Run Parameters:** `--steps 20 --step-size 1 --initial-cash 100000 --kronos-only --flatten-end --compact-logs`

## Pre-Run Checklist
1. Refresh trend data:  
   `python scripts/fetch_etf_trends.py --symbols-file marketsimulator/etf_watchlist.txt --days 365 --window 50`
2. Regenerate readiness & momentum reports + forecast diagnostics:  
   ```
   python scripts/generate_candidate_readiness.py --summary-path marketsimulator/run_logs/trend_summary.json \
      --output marketsimulator/run_logs/candidate_readiness.md \
      --min-sma 200 --min-pct 0.0 --csv-output marketsimulator/run_logs/candidate_readiness_history.csv
   python scripts/analyze_candidate_history.py --history marketsimulator/run_logs/candidate_readiness_history.csv \
      --output marketsimulator/run_logs/candidate_momentum.md --trailing 5
   python scripts/check_candidate_forecasts.py --history marketsimulator/run_logs/candidate_readiness_history.csv \
      --min-sma 200 --min-pct 0.3 --steps 1 --min-strategy-return 0.015 --min-predicted-move 0.01 \
      --output marketsimulator/run_logs/candidate_forecast_gate_report.md
   ```
   Proceed with Run 80 only if SMH (or another ETF) shows repeated strategy-return results ≥ 0.010 in the diagnostic report.

## Execution Command
```
MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP='AAPL:-0.03,AMZN:-0.02,SOXX:0.015,SMH:0.01' \
MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP='SMH:0.0075,SOXX:0.01' \
MARKETSIM_SYMBOL_KELLY_SCALE_MAP='AAPL:0.2,MSFT:0.25,NVDA:0.01,AMZN:0.15,SOXX:0.10,SMH:0.10' \
MARKETSIM_SYMBOL_MAX_ENTRIES_MAP='NVDA:1,MSFT:10,AAPL:10,AMZN:8,SOXX:4,SMH:4' \
MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP='AAPL:10800,MSFT:10800,NVDA:7200,AMZN:10800,SOXX:7200,SMH:7200' \
MARKETSIM_SYMBOL_SIDE_MAP='NVDA:sell' \
python scripts/run_sim_with_report.py --prefix 20251024-Run80 --max-fee-bps 25 --max-avg-slip 100 \
   --max-drawdown-pct 5 --min-final-pnl -2000 --max-worst-cash -40000 \
   --max-trades-map NVDA@ci_guard:2,MSFT@ci_guard:16,AAPL@ci_guard:20,SMH@ci_guard:8 \
   -- python marketsimulator/run_trade_loop.py \
   --symbols AAPL MSFT NVDA AMZN SOXX SMH \
   --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --flatten-end --compact-logs \
   --kronos-sharpe-cutoff -1.0
```

## Post-Run Evaluation
1. Inspect `run_logs/*_metrics.json`, `*_trades_summary.json`, and gate diagnostics to confirm whether SMH entries materialised and whether risk stayed within caps.
2. Update `marketsimulator/results.md` with outcomes (PnL, drawdown, entry counts).
3. If SMH performs acceptably, consider extending the test window (steps ≥ 40) or re-introducing QQQ with similar guardrails. If losses persist, revert thresholds and continue monitoring.
