# RL Experiment Expansion Plan (2025-10-25)

## Current Posture Snapshot
- `trade_stock_e2e.py` runs Kelly-lite sizing with spread/edge gating, but stores limited feature context alongside trades (`FlatShelf` stores only outcomes/learning). We lack rich state capture for RL replay.
- HF/PPO trainers (`training/production_ready_trainer.py`, `training/ppo_trainer.py`, `hftraining/portfolio_rl_trainer.py`) focus on price/technical tensors and synthetic environments; production-aligned signals such as bid/ask microstructure, realised risk budgets, and Alpaca execution metadata are absent.
- Differentiable-market pipelines (e.g., `differentiable_market/trainer.py`) can backpropagate through transaction costs yet rely on coarsely engineered features and static leverage limits.
- Forecast sources (Toto, Kronos, DeepSeek planners) provide diverse distributions, but their calibration metadata is scattered (`hyperparamstore/`, strategy logs) and not fused into a canonical feature store.

## Feature Collection Roadmap
- **Microstructure & Execution Tape**
  - Capture quote ladder snapshots per entry candidate to extend `compute_spread_bps` with order-book slope, quoted depth, imbalance, and fill probability.
  - Log realised slippage, partial fills, and quote persistence from Alpaca responses; backfill into RL datasets as cost targets.
  - Persist per-trade latency (signal→order, order→fill) to model execution risk.
- **Regime & Volatility Context**
  - Extend `record_portfolio_snapshot` pipeline to log realised intraday volatility, realised skew, drawdown velocity, and cross-asset correlation clusters.
  - Derive volatility regime labels (calm, choppy, crisis) using rolling HVaR and VIX term structure proxies; tag each experience tuple.
- **Forecast Diagnostics Layer**
  - Store Toto/Kronos sample distributions (mean, std, skew, calibration residuals), plus `edge_threshold_bps` decisions, per symbol/strategy.
  - Aggregate consensus stats from `evaluate_strategy_entry_gate` (pass/fail reason, fallback flags) and use as RL state inputs.
  - Ingest DeepSeek plan metadata (goal type, scenario assumptions) for hierarchical policy conditioning.
- **Alternative & Macro Signals**
  - Pull macroeconomic calendars, Treasury curve slopes, sector ETF momentum, and FX regime indicators; normalise into daily tensors.
  - Integrate on-chain metrics for crypto symbols (volume, active addresses) to reduce blind spots in BTC/ETH flows.
- **Risk Budget & Exposure Signals**
  - Persist `kelly_lite` outputs, leverage caps, portfolio concentration metrics, and drawdown trigger states into the replay buffer.
  - Track borrowed capital costs (from `get_leverage_settings`) and realised funding to let policies internalise financing drag.

## Algorithmic Experiment Tracks
1. **Distributionally Robust RL**
   - Train CVaR-optimised policies atop `hftraining/portfolio_rl_trainer.py` by augmenting loss with tail-risk penalties (`cvar_weight` sweep, bootstrap VaR inputs from stored forecast distributions).
   - Extend PPO agent to DistRL variant (quantile critics) so stochastic policies respect downside risk at execution time.
2. **Hierarchical Forecast-to-Action Pipelines**
   - Stage 1: Train a gating classifier that decides between probe, passive, or aggressive modes using consensus diagnostics; Stage 2: specialised actor per mode for position sizing and bracket selection.
   - Use DeepSeek plan descriptors as high-level intents (trend-follow, mean-revert, event-driven) to route through plan-specific sub-policies.
3. **Model-Based Scenario Rollouts**
   - Leverage differentiable-market environment to simulate stress scenarios seeded with collected macro/regime annotations; integrate latent risk budget dynamics and drawdown penalties.
   - Introduce learned scenario generators (diffusion or transformer-based) conditioned on macro factors to produce adversarial paths for distributional training.
4. **Graph & Factor-Aware Allocation**
   - Build graph neural network policies over asset correlation graphs to coordinate multi-asset trades; node features come from enriched feature store.
   - Combine pair-level RL (current portfolio trainer) with cross-sector factor exposures, enforcing leverage and concentration constraints through differentiable projections.
5. **Offline-to-Online RL Bridging**
   - Curate a `FlatShelf`-backed replay of historical trades with expanded features to pretrain via Conservative Q-Learning / IQL.
   - Deploy online fine-tuning with risk-aware exploration (epsilon schedule tied to drawdown state, probe symbols from `PROBE_SYMBOLS`).
6. **Meta-Calibration of Forecast Ensembles**
   - Train auxiliary networks to recalibrate Toto/Kronos outputs per regime; feed recalibrated distributions into RL states.
   - Explore residual-learning adapters (small MLPs) to correct systematic bias before RL consumption.

## Derisking & Forecast Vetting Enhancements
- **Dynamic Consensus Gating**: Replace static `CONSENSUS_MIN_MOVE_PCT` with adaptive threshold derived from realised spread costs, recent hit rates, and distribution skew.
- **Scenario-Based Kill Switches**: Use differentiable drawdown metrics and stored macro stress tags to trigger position unwinds or bracket tightening automatically.
- **Explainability Hooks**: Log Shapley-style attributions for policy actions (feature ablations within RL network) so risk desk can audit decisions before production rollout.
- **Execution Feedback Loop**: Integrate post-trade slippage residuals into policy updates, penalising actions that exceed projected cost envelopes.

## Multi-Asset Deployment Considerations
- Expand `trade_stock_e2e` to manage basket orders by batching `ramp_into_position` calls with correlation-aware throttles.
- Introduce continuous portfolio constraints (sector, factor, liquidity) enforced through projected gradient steps before order placement.
- Implement rolling hedges (index futures / sector ETFs) controlled by RL policies to neutralise beta when signal confidence drops.
- Automate pair selection via clustering on enriched features, feeding cluster IDs to portfolio trainer for diversified rollouts.

## Evaluation & Tooling
- Build a unified RL benchmark harness under `evaltests/rl_benchmark_runner.py` that replays enriched datasets across PPO, differentiable-market, and portfolio RL policies with identical cost assumptions.
- Add risk-focused metrics (CVaR, drawdown duration, tail hit rate) to `backtest_test3_inline.py` outputs to compare against production.
- Schedule nightly batch simulations (minute + daily) storing summaries in `marketsimulatorresults.md` with direct links to checkpoint configs.
- Instrument training scripts with nanochat-style run manifests (single JSON summarising data, config, FLOPs, best metrics) for reproducibility.

## Immediate Next Steps
- Stand up a feature logging scaffold in `trade_stock_e2e` to persist microstructure, forecast, and risk budget metadata into a unified parquet store.
- Prototype a distributionally-aware PPO critic using stored forecast quantiles; validate on historical Alpaca fills.
- Draft schema and ingestion jobs for macro/alt data feeds, ensuring timestamp alignment before adding to RL datasets.
- Design benchmark experiments comparing baseline PPO vs hierarchical gating pipeline on 2023–2025 data, including stress scenarios.

