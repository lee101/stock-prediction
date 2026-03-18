# Binance Progress Log 5 - Worksteal Robust Promotion

Updated: 2026-03-17

## Worksteal Robust Multi-Start Sweep

Window tested:
- `2026-01-13` to `2026-03-14`

Execution model:
- Daily signal generation
- Next-day hourly fill/exit replay
- `25bps` live entry gate
- BTC/ETH routed through FDUSD, alts through USDT
- Ranked across three starting states with equal starting equity:
  - `flat`
  - `BTCUSD`
  - `ETHUSD`

Code changes:
- `binance_worksteal/sweep.py`
  - added `--start-state`
  - added equal-equity seeded inventory scenarios
  - added robust aggregate metrics (`worst_*`, `mean_*`, per-start-state columns)
  - robust ranking now prioritizes worst-case return/sortino/drawdown instead of flat-only sortino
- `binance_worksteal/trade_live.py`
  - fixed live CLI/config mismatch so runtime can actually express `proximity_pct`, `lookback_days`, `ref_price_method`, `max_hold_days`, and `market_breadth_filter`
  - updated `DEFAULT_CONFIG` to the new robust winner
- `deployments/binance-worksteal-daily/launch.sh`
  - updated live supervisor launch flags to the promoted config
- `deployments/binance-worksteal-daily/supervisor.conf`
  - updated deployment notes/stats

Tests:
- `pytest tests/test_worksteal_trade_live.py tests/test_worksteal_sweep.py tests/test_worksteal_strategy.py tests/test_worksteal_gemini_prompt.py -q`
- Result: `29 passed`
- `python -m py_compile` passed on the touched Python files
- `bash -n deployments/binance-worksteal-daily/launch.sh` passed

## Result Artifacts

- Broad robust sweep:
  - `analysis/worksteal_hourly_20260317/hourly_gate25_multistart_spot_random220_latest60d.csv`
- Tight local refinement:
  - `analysis/worksteal_hourly_20260317/hourly_gate25_multistart_tight_random90_latest60d.csv`
- Direct comparisons:
  - `analysis/worksteal_hourly_20260317/multistart_candidate_comparison_latest60d.csv`
  - `analysis/worksteal_hourly_20260317/multistart_candidate_comparison_live30_latest60d.csv`
- Earlier flat-only references:
  - `analysis/worksteal_hourly_20260317/hourly_gate25_random220_latest60d.csv`
  - `analysis/worksteal_hourly_20260317/best_gate25_start_position_check.csv`
  - `analysis/worksteal_hourly_20260317/latest_60d_summary.csv`

## Best Robust Config

Promoted config:
- `dip_pct=0.10`
- `proximity_pct=0.005`
- `profit_target_pct=0.03`
- `stop_loss_pct=0.08`
- `max_positions=5`
- `max_hold_days=30`
- `lookback_days=20`
- `ref_price_method="sma"`
- `trailing_stop_pct=0.03`
- `sma_filter_period=0`
- `market_breadth_filter=0.0`
- `entry_proximity_bps=25`
- `max_leverage=1.0`
- `enable_shorts=False`

Broad 220-run robust winner:
- worst return: `+3.06%`
- mean return: `+3.95%`
- best return: `+5.62%`
- worst Sortino: `0.85`
- mean Sortino: `0.93`
- worst max drawdown: `-4.75%`
- mean max drawdown: `-4.16%`

Per start-state:

| Start | Return | Sortino | Max DD | Trades |
|---|---:|---:|---:|---:|
| flat | `+5.62%` | `1.05` | `-3.10%` | `26` |
| BTC | `+3.06%` | `0.85` | `-4.75%` | `15` |
| ETH | `+3.19%` | `0.90` | `-4.63%` | `15` |

## Comparison Versus Prior Candidates

Exact live 30-symbol universe comparison (`multistart_candidate_comparison_live30_latest60d.csv`):

| Candidate | Worst Ret | Mean Ret | Worst Sortino | Worst Max DD |
|---|---:|---:|---:|---:|
| current live default | `-1.90%` | `-1.22%` | very negative | `-1.90%` |
| old flat-only best | `-1.41%` | `-0.12%` | `-0.14` | `-6.86%` |
| **new robust best** | **`+3.06%`** | **`+3.95%`** | **`0.85`** | **`-4.75%`** |

Important read:
- The old live worksteal launch was effectively inert from flat cash and outright bad from seeded BTC/ETH starts.
- The old flat-only winner looked decent only from flat cash and failed robustness.
- The new robust config is the first worksteal candidate in this branch that stayed positive across all tested starting states while keeping drawdown under `5%`.

The tight 90-run refinement was useful but not better:
- best tight candidate worst return: `-0.26%`
- mean return: `+0.28%`
- so the broader 220-run search found the actual promotion candidate

## Deployment

Deployment target:
- supervisor program: `binance-worksteal-daily`
- launcher: `deployments/binance-worksteal-daily/launch.sh`

Applied:
- updated repo launch script to the promoted config
- synced `deployments/binance-worksteal-daily/supervisor.conf` into `/etc/supervisor/conf.d/binance-worksteal-daily.conf`
- restarted via `sudo supervisorctl restart binance-worksteal-daily`

Post-deploy checks:
- `supervisorctl status` shows `binance-worksteal-daily RUNNING`
- current PID after restart: `396094`
- current live command now includes:
  - `--dip-pct 0.10`
  - `--proximity-pct 0.005`
  - `--profit-target 0.03`
  - `--stop-loss 0.08`
  - `--max-hold-days 30`
  - `--lookback-days 20`
  - `--ref-method sma`
  - `--sma-filter 0`
  - `--market-breadth-filter 0.0`
  - `--trailing-stop 0.03`
  - `--entry-proximity-bps 25`
- latest supervisor error log lines show the restarted daemon waiting for the next run at `2026-03-18 00:05:00+00:00`

Operational note:
- Restart happened outside the UTC entry window, so it did not trigger an immediate trading cycle.
- The next live validation point is the next UTC daily cycle/log output.

## Next Work

- Watch the next `binance-worksteal-daily` run and confirm candidate selection/order placement behavior matches the new config.
- Extend the same robust-start-state ranking discipline to the other Binance daily strategy families before promoting any more configs.
- Do not add `XLMUSD`, `RNDRUSD`, or `TAOUSD` to the live worksteal launcher until the live symbol/pair aliases are validated end to end.

## Follow-On Multi-Window Check

Added reusable multi-window evaluation helpers:
- `binance_worksteal/robust_eval.py`
  - recent window builder
  - seeded-start scenario evaluation across windows
  - robust summary scoring via `src.robust_trading_metrics.summarize_scenario_results`

Added tests:
- `tests/test_worksteal_robust_eval.py`
- Updated verification set:
  - `pytest tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_trade_live.py tests/test_worksteal_strategy.py tests/test_worksteal_gemini_prompt.py -q`
  - Result: `31 passed`

Method:
- Exact live 30-symbol universe
- Three non-overlapping 60-day windows:
  - `2026-01-13` to `2026-03-14`
  - `2025-11-14` to `2026-01-13`
  - `2025-09-15` to `2025-11-14`
- Starting states:
  - `flat`
  - `BTCUSD`
  - `ETHUSD`

Artifacts:
- `analysis/worksteal_hourly_20260317/multiwindow_candidate_ladder_v1.csv`
- `analysis/worksteal_hourly_20260317/multiwindow_candidate_ladder_scenarios_v1.csv`
- `analysis/worksteal_hourly_20260317/multiwindow_candidate_ladder_riskoff_v1.csv`
- `analysis/worksteal_hourly_20260317/multiwindow_candidate_ladder_riskoff_scenarios_v1.csv`

Local neighborhood result:
- The currently deployed config remained best in its immediate parameter ladder:
  - robust score: `-39.22`
  - mean return across 9 scenarios: `+0.92%`
  - worst scenario return: `-3.19%`
  - worst drawdown: `4.75%`
- The main weakness is the middle 60-day window (`2025-11-14` to `2026-01-13`) where flat start was `-3.19%`.

Risk-off probe:
- Best risk-off neighbor was `high_ref_mb06`:
  - `ref_price_method="high"`
  - `market_breadth_filter=0.6`
  - robust score: `-33.91` (better aggregate)
- But latest-window performance got materially worse:
  - flat: `+1.33%`
  - BTC start: `-0.57%`
  - ETH start: `-0.45%`
- Because it degraded the most recent window and flipped seeded BTC/ETH recent performance negative, it was **not promoted**.

Deployment decision after the multi-window check:
- **No further live change**
- Keep the already deployed `ref=sma`, `market_breadth=0.0` config
- Continue searching, but require any replacement to beat the live config on both:
  - recent-window seeded-start performance
  - multi-window robustness

## 2026-03-18 Follow-Up

Live state check before more work:
- `binance-worksteal-daily` was still running
- latest completed live cycle in logs was still `2026-03-17 00:05 UTC`
- account state file remained flat:
  - `positions = {}`
  - `last_exit = {}`

Fix shipped:
- `binance_worksteal/strategy.py`
  - added `resolve_entry_config()` and risk-off trigger support
  - exposed shared `build_entry_candidates()` / `compute_market_breadth_skip()` helpers
  - backtests now use the same entry-config resolution path for static and dynamic regimes
- `binance_worksteal/trade_live.py`
  - live entries now use the shared strategy candidate builder instead of a forked local implementation
  - live CLI/runtime config now supports optional risk-off override parameters

Why this mattered:
- Before this pass, live entry generation still had its own copy of the dip-entry logic.
- That meant future regime logic and some entry-price details could diverge from the market simulator even if backtests looked good.
- After this change, live and sim share the same candidate construction path.

Tests after the fix:
- `pytest tests/test_worksteal_strategy.py tests/test_worksteal_trade_live.py tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_gemini_prompt.py -q`
- Result: `33 passed`

New artifacts:
- `analysis/worksteal_hourly_20260318/regime_switch_candidate_compare_v1.csv`
- `analysis/worksteal_hourly_20260318/regime_switch_candidate_compare_scenarios_v1.csv`
- `analysis/worksteal_hourly_20260318/regime_switch_latest60d_compare_v1.csv`

Regime experiment result:
- Static risk-off (`ref=high`, `market_breadth=0.6`) still had the better three-window aggregate score than the current live config:
  - robust score `-33.91` vs deployed `-39.22`
- But it remained unacceptable on the most recent window:
  - latest 60d flat: `+1.33%`
  - latest 60d BTC start: `-0.57%`
  - latest 60d ETH start: `-0.45%`
- Dynamic regime-switch attempt using BTC/ETH trend triggers was worse:
  - robust score `-48.62`
  - latest 60d BTC start `-1.97%`
  - latest 60d ETH start `-1.85%`

Decision:
- **No new parameter promotion on 2026-03-18**
- Keep the existing live parameters
- Ship only the live/backtest entry-path alignment fix

Deployment update:
- Restarted `binance-worksteal-daily` after the code fix with:
  - `sudo supervisorctl restart binance-worksteal-daily`
- Post-restart status:
  - `RUNNING`
  - PID `467211`
- Current live command remains the same promoted parameter set from the prior update
- Restart happened at `2026-03-18 00:21:31` local log time, still well before the next `00:05 UTC` run

## 2026-03-18 Allocation Refiner Experiment

Goal:
- Test whether a post-selection allocation refiner can improve work-steal order sizing without changing candidate ranking or exits.
- Promotion bar remains strict:
  - better recent PnL
  - better or at least not worse Sortino
  - smoother PnL
  - lower max drawdown

Code changes:
- `binance_worksteal/strategy.py`
  - added optional `allocation_scale_fn` hook to `run_worksteal_backtest()`
  - added `EntrySizingContext` so experiments can reuse the exact hourly simulator and only change per-entry size
  - default behavior is unchanged when no sizing callback is passed, so live execution is unaffected
- `binance_worksteal/allocation_refiner_eval.py`
  - new self-contained experimental allocator
  - collects offline entry/outcome examples from the promoted work-steal strategy
  - supports:
    - fixed sizing baseline
    - score-only heuristic sizing
    - Chronos-cache-aware heuristic sizing
    - learned residual sizing head
- `tests/test_worksteal_allocation_refiner_eval.py`
  - verifies baseline parity when the callback returns `1.0`
  - verifies scaled sizing changes entry quantity
  - verifies forecast-cache feature join and bounded heuristic outputs

Verification:
- `pytest tests/test_worksteal_strategy.py tests/test_worksteal_allocation_refiner_eval.py -q`
- Result: `30 passed`
- `python -m py_compile binance_worksteal/strategy.py binance_worksteal/allocation_refiner_eval.py` passed

Training/evaluation setup:
- Live 30-symbol work-steal universe
- Base runtime config remained the current promoted rule-only branch:
  - `max_position_pct=0.10`
  - `rebalance_seeded_positions=True`
  - same hourly execution model
- Chronos features came from `binanceneural/forecast_cache/h24` when available, with graceful fallback when missing/stale
- Training dataset:
  - `570` train rows
  - `143` validation rows
  - saved in `analysis/worksteal_hourly_20260318/allocation_refiner_v1/train_examples.csv`
- Learned head fit summary:
  - `hidden_size=64`
  - `epochs=250`
  - `min_scale=0.0`
  - `max_scale=1.35`
  - train loss `0.0196`
  - val loss `0.0539`
  - saved in `analysis/worksteal_hourly_20260318/allocation_refiner_v1/fit_summary.json`

Latest 60d screen (`flat`, `BTCUSD`, `ETHUSD`):
- Artifacts:
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/screen_summary.csv`
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/screen_scenarios.csv`
- Result:
  - `fixed`
    - return mean `+0.229%`
    - Sortino mean `0.204`
    - max DD mean `2.611%`
    - robust score `-1.960`
  - `heuristic_score`
    - return mean `-0.430%`
    - Sortino mean `-0.829`
    - max DD mean `1.068%`
    - robust score `-54.411`
  - `heuristic_chronos`
    - return mean `-0.455%`
    - Sortino mean `-0.984`
    - max DD mean `0.958%`
    - robust score `-54.735`
  - `learned_refiner`
    - return mean `+0.127%`
    - Sortino mean `0.177`
    - max DD mean `1.562%`
    - robust score `-1.060`

Read:
- The simple inference-time heuristics were clearly too conservative and were rejected immediately.
- The learned refiner was the only variant worth carrying forward.

Full robustness checks:
- Three windows x three starts:
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/full_summary_top2_3starts.csv`
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/full_scenarios_top2_3starts.csv`
- Three windows x all live start states (`flat` plus all 30 live symbols):
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/full_summary_all_live_states.csv`
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/full_scenarios_all_live_states.csv`
- Latest 60d x all live start states:
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/latest60d_summary_all_live_states.csv`
  - `analysis/worksteal_hourly_20260318/allocation_refiner_v1/latest60d_scenarios_all_live_states.csv`

Key numbers:
- Latest 60d, all live start states:
  - `fixed`
    - return mean `+0.229%`
    - Sortino mean `0.204`
    - max DD mean `2.613%`
    - robust score `-1.962`
  - `learned_refiner`
    - return mean `+0.127%`
    - Sortino mean `0.177`
    - max DD mean `1.563%`
    - robust score `-1.061`
- Three windows, all live start states:
  - `fixed`
    - return mean `-1.471%`
    - return p25 `-3.278%`
    - Sortino mean `-1.402`
    - max DD mean `2.773%`
    - max DD worst `3.606%`
    - robust score `-53.360`
  - `learned_refiner`
    - return mean `-0.842%`
    - return p25 `-1.849%`
    - Sortino mean `-1.361`
    - max DD mean `1.646%`
    - max DD worst `2.134%`
    - robust score `-47.989`

Decision:
- **No live promotion yet**
- The learned refiner is better on robustness and drawdown, but it is still worse than the current live fixed-size branch on the most recent 60-day window for both:
  - mean return
  - mean Sortino
- Because the current promotion rule requires recent PnL / Sortino improvement as well as smoother risk, the refiner stays experimental only for now.

Operational result:
- No supervisor change
- No live restart
- Live bot remains the current rule-only fixed-size configuration

Best current read:
- The residual sizing head is promising as a risk governor.
- It is not yet a clear leverage/order-size alpha source.
- The next useful move is not a live deploy; it is a second pass that trains directly for recent-regime utility so the head stops giving away too much upside while buying the drawdown reduction.

## Position Sizing Promotion

New CLI/runtime support:
- `binance_worksteal/trade_live.py`
  - added `--max-position-pct`
- Tests updated:
  - `tests/test_worksteal_trade_live.py`

Verification after the sizing work:
- `pytest tests/test_worksteal_trade_live.py tests/test_worksteal_strategy.py tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_gemini_prompt.py -q`
- Result: `33 passed`

Artifacts:
- `analysis/worksteal_hourly_20260318/position_size_ladder_v1.csv`
- `analysis/worksteal_hourly_20260318/position_size_ladder_latest60d_v1.csv`
- `analysis/worksteal_hourly_20260318/position_size_ladder_fine_v1.csv`
- `analysis/worksteal_hourly_20260318/position_size_ladder_fine_latest60d_v1.csv`

Finding:
- Lowering `max_position_pct` materially improved the 3-window robustness profile.
- The old live size (`0.25`) was too aggressive for the older windows:
  - robust score `-39.22`
  - worst scenario return `-3.19%`
  - worst drawdown `4.75%`
  - negative-return rate `55.6%`

Coarse ladder:
- `0.10` had the best pure robust score (`-23.25`) but gave up too much recent upside.
- `0.20` was the first strong balance candidate:
  - robust score `-26.49`
  - latest 60d:
    - flat `+4.49%`
    - BTC start `+2.07%`
    - ETH start `+2.19%`

Fine ladder:
- Best robust score in the refinement was `0.17`:
  - robust score `-25.52`
  - but latest 60d upside was cut more than needed
- Chosen live promotion: `max_position_pct=0.23`
  - robust score `-27.46`
  - worst scenario return `-2.93%`
  - worst drawdown `4.53%`
  - negative-return rate `33.3%`
  - latest 60d:
    - flat `+5.17%`
    - BTC start `+2.66%`
    - ETH start `+2.79%`

Why `0.23` won:
- It keeps a large share of the current-window upside versus `0.25`
- It still materially improves multi-window robustness and drawdown
- It avoids the sharper recent upside sacrifice from `0.17` to `0.20`

Deployment change:
- Updated:
  - `binance_worksteal/trade_live.py` default config
  - `deployments/binance-worksteal-daily/launch.sh`
  - `deployments/binance-worksteal-daily/supervisor.conf`
- New live flag:
  - `--max-position-pct 0.23`
- Restarted:
  - `sudo supervisorctl restart binance-worksteal-daily`
- Post-restart status:
  - `RUNNING`
  - PID `482147`

Live command after promotion:
- `... --max-positions 5 --max-position-pct 0.23 --max-hold-days 30 --lookback-days 20 --ref-method sma ...`

## Aptos Data Fix

Requested symbol:
- Aptos is tracked in this stack as `APTUSD` internally
- On Binance source data the backing files are `APTUSDT` / `APTFDUSD`

What I found:
- `APTUSD` was already present in:
  - the live work-steal symbol list
  - the work-steal research universe
- But local replay coverage was incomplete:
  - hourly local cache had only a 48-row stub `trainingdatahourly/crypto/APTUSD.csv`
  - daily local cache was missing `trainingdatadailybinance/APTUSDT.csv` and `trainingdatadailybinance/APTUSD.csv`
- That meant the backtests/sweeps were effectively undercounting the universe even though live could still fetch Aptos directly from Binance.

Fix applied:
- Rebuilt daily Aptos bars from full Binance hourly source:
  - `trainingdatadailybinance/APTUSDT.csv`
  - `trainingdatadailybinance/APTUSD.csv`
- Rebuilt full hourly Aptos proxy from Binance source:
  - copied `binance_spot_hourly/APTUSDT.csv` into `trainingdatahourly/crypto/APTUSDT.csv`
  - regenerated `trainingdatahourly/crypto/APTUSD.csv`

Post-fix validation:
- `load_daily_bars(..., ['APTUSD'])` now returns `1095` daily rows
- `load_hourly_bars(..., ['APTUSD'])` now returns `26278` hourly rows

Post-fix comparison artifacts:
- `analysis/worksteal_hourly_20260318/apt_included_compare_v1.csv`
- `analysis/worksteal_hourly_20260318/apt_included_compare_latest60d_v1.csv`

Result after actually including Aptos in replay:
- The promoted live config (`max_position_pct=0.23`) still beats the old `0.25` size:
  - robust score:
    - live `0.23`: `-45.27`
    - old `0.25`: `-46.96`
  - worst scenario return:
    - live `0.23`: `-4.82%`
    - old `0.25`: `-5.24%`
  - worst drawdown:
    - live `0.23`: `5.55%`
    - old `0.25`: `6.03%`

Latest 60d after Aptos is included:
- live `0.23`
  - flat `+5.17%`
  - BTC start `+2.66%`
  - ETH start `+2.79%`
- old `0.25`
  - flat `+5.62%`
  - BTC start `+3.06%`
  - ETH start `+3.19%`

Decision after the Aptos fix:
- **No further live parameter change**
- Keep `APTUSD` in the live universe
- Keep the current promoted `max_position_pct=0.23` config because it still has the better robustness tradeoff once Aptos is actually included

## Full Live-Universe Replay Repair and POL Swap

Problem found:
- The local work-steal replay tree had drifted away from the live 30-symbol launch universe.
- Some live symbols were missing locally (`ICPUSD`, `OPUSD`, `INJUSD`, `TIAUSD`, `SEIUSD`, `PEPEUSD`, `BNBUSD`).
- Several others existed but were badly stale:
  - `ALGOUSD`, `TRXUSD`, `MATICUSD` ended in 2023
  - `AVAXUSD`, `LTCUSD`, `UNIUSD`, `XRPUSD` ended on 2026-02-06
  - `APTUSD`, `BNBUSD` ended on 2026-02-24
  - `BCHUSD` only had 52 hourly rows

Replay data repair:
- Downloaded missing/raw-stale hourly Binance Vision data into `binance_spot_hourly/` for:
  - `ICPUSDT OPUSDT INJUSDT TIAUSDT SEIUSDT PEPEUSDT`
  - `ALGOUSDT AVAXUSDT LTCUSDT XRPUSDT UNIUSDT TRXUSDT MATICUSDT APTUSDT BNBUSDT BCHUSDT`
  - `POLUSDT POLFDUSD`
- Synced the refreshed source files into `trainingdatahourly/crypto/`
- Rebuilt USD proxies and rebuilt daily bars in `trainingdatadailybinance/`

Coverage validation after repair:
- Added `tests/test_worksteal_live_universe_coverage.py`
  - Fails if any live launch symbol is missing locally
  - Fails if any live launch symbol has less than 60 daily bars / 1440 hourly bars
  - Fails if any live launch symbol lags the freshest local live symbol by more than 14 days
- After the replay repair, the only remaining stale live symbol was `MATICUSD`
  - `MATICUSD` local hourly data still ended on `2023-06-23`
  - `POLUSDT` and `POLFDUSD` both fetched cleanly through `2026-03-15`

Decision:
- Replace `MATICUSD` with `POLUSD` in the work-steal live/backtest universe

Files updated for the symbol repair:
- `binance_worksteal/backtest.py`
- `deployments/binance-worksteal-daily/launch.sh`
- `binance_worksteal/trade_live.py`
- `deployments/binance-worksteal-daily/supervisor.conf`
- `scripts/build_binance_spot_hourly_usd_proxies.py`
- `tests/test_build_binance_spot_hourly_usd_proxies.py`
- `tests/test_worksteal_live_universe_coverage.py`

Verification after the replay repair:
- `pytest tests/test_build_binance_spot_hourly_usd_proxies.py tests/test_worksteal_live_universe_coverage.py tests/test_worksteal_trade_live.py tests/test_worksteal_strategy.py tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_gemini_prompt.py -q`
- Result: `35 passed`

## POL-Universe Sizing Re-Promotion

Artifacts:
- `analysis/worksteal_hourly_20260318/all_live_start_states_latest60d_key_sizes_pol_v1.csv`
- `analysis/worksteal_hourly_20260318/all_live_start_states_latest60d_key_sizes_pol_scenarios_v1.csv`
- `analysis/worksteal_hourly_20260318/all_live_start_states_multwindow_key_sizes_pol_v1.csv`
- `analysis/worksteal_hourly_20260318/all_live_start_states_multwindow_key_sizes_pol_scenarios_v1.csv`

Method:
- Re-ran the latest 60d window (`2026-01-13` to `2026-03-14`)
- Used the corrected 30-symbol live universe with `POLUSD` instead of stale `MATICUSD`
- Evaluated seeded starts from every live symbol plus `flat`
- Compared the current work-steal config at `max_position_pct` = `0.17`, `0.20`, `0.23`

Latest 60d all-live-start-state summary:
- `0.17`
  - robust score `-77.29`
  - mean return `-3.34%`
  - worst return `-7.70%`
  - worst drawdown `12.52%`
  - mean smoothness `0.003836`
- `0.20`
  - robust score `-77.95`
  - mean return `-3.29%`
  - worst return `-7.63%`
  - worst drawdown `13.31%`
  - mean smoothness `0.004399`
- `0.23`
  - robust score `-78.29`
  - mean return `-3.16%`
  - worst return `-7.46%`
  - worst drawdown `14.09%`
  - mean smoothness `0.004957`

Read:
- On the corrected live universe, `0.23` still buys a little extra upside in the friendliest starts
- But it clearly loses on drawdown and PnL smoothness across the full inherited-position set
- `0.17` is now the best score even on the corrected latest 60d window once all live seeded starts are included

3-window confirmation:
- I then reran a narrower corrected-universe 3-window comparison for just `0.17` vs `0.23`
- Result:
  - `0.17`
    - robust score `-66.50`
    - mean return `-2.26%`
    - return p25 `-2.82%`
    - worst return `-7.70%`
    - worst drawdown `12.52%`
    - negative-return rate `69.9%`
  - `0.23`
    - robust score `-85.26`
    - mean return `-3.13%`
    - return p25 `-4.64%`
    - worst return `-8.71%`
    - worst drawdown `14.09%`
    - negative-return rate `95.7%`
- This confirmed the redeploy direction: the corrected full-universe replay strongly prefers `0.17` over `0.23`

Representative latest-window starts:
- `flat`
  - `0.17`: `+0.39%`, max DD `4.44%`
  - `0.23`: `+0.53%`, max DD `6.00%`
- `POLUSD`
  - `0.17`: `+0.25%`, max DD `4.57%`
  - `0.23`: `+0.39%`, max DD `6.14%`
- `BTCUSD`
  - `0.17`: `-1.51%`, max DD `6.33%`
  - `0.23`: `-1.37%`, max DD `7.90%`

Decision:
- Promote `max_position_pct=0.17`
- Keep the rest of the live work-steal parameters unchanged
- Keep `POLUSD` in the live universe and remove stale `MATICUSD`

Deployment:
- Updated:
  - `binance_worksteal/trade_live.py`
  - `deployments/binance-worksteal-daily/launch.sh`
  - `deployments/binance-worksteal-daily/supervisor.conf`
- Restarted:
  - `sudo supervisorctl restart binance-worksteal-daily`
- Post-restart status:
  - `RUNNING`
  - PID `653744`

Current live command:
- `... --max-positions 5 --max-position-pct 0.17 --max-hold-days 30 --lookback-days 20 --ref-method sma ... --symbols ... PEPEUSD POLUSD`

## Legacy Rebalance Promotion

Artifacts:
- `analysis/worksteal_hourly_20260318/seeded_rebalance_compare_pol_v1.csv`
- `analysis/worksteal_hourly_20260318/seeded_rebalance_compare_pol_scenarios_v1.csv`
- `analysis/worksteal_hourly_20260318/seeded_rebalance_size_ladder_pol_v1.csv`
- `analysis/worksteal_hourly_20260318/seeded_rebalance_size_ladder_pol_scenarios_v1.csv`
- `analysis/worksteal_hourly_20260318/seeded_rebalance_size_ladder_pol_fine_low_v1.csv`
- `analysis/worksteal_hourly_20260318/seeded_rebalance_size_ladder_pol_fine_low_scenarios_v1.csv`

Implementation:
- Added seeded/inherited position provenance in the simulator via `Position.source`
- Added the simulator-side one-shot seeded rebalance option in `binance_worksteal/strategy.py`
- Added live-state normalization plus legacy-position rebalance logic in `binance_worksteal/trade_live.py`
  - Existing state positions without `source` are treated as `legacy`
  - Legacy positions that still match the current keep-set are adopted as `strategy`
  - Legacy positions outside the keep-set are exited with reason `legacy_rebalance`
  - New positions are persisted with `source="strategy"`
- Added focused live-path coverage in `tests/test_worksteal_trade_live.py`

Step 1: validate the rebalance idea before changing sizing
- Compared the corrected live-universe work-steal config at `max_position_pct=0.17` with seeded rebalance off vs on
- Same evaluation method as the recent live checks:
  - corrected 30-symbol universe ending `2026-03-14`
  - hourly execution
  - 3 recent 60-day windows
  - all live start states plus `flat`
- Result:
  - `0.17` with rebalance off
    - robust score `-66.50`
    - latest robust score `-77.29`
    - latest return mean `-3.34%`
    - latest worst drawdown `12.52%`
  - `0.17` with rebalance on
    - robust score `-61.54`
    - latest robust score `-3.63`
    - latest return mean `+0.39%`
    - latest worst drawdown `4.44%`
- Read:
  - The rebalance branch materially improved inherited-position handling
  - Flat-start behavior stayed effectively unchanged because there is nothing to rebalance from `flat`

Step 2: retune sizing on top of the rebalance branch
- Coarse ladder (`0.14`, `0.17`, `0.20`, `0.23`) showed the rebalance branch prefers smaller size
- Fine lower ladder (`0.10`, `0.12`, `0.14`, `0.16`) confirmed `0.10` as the best tested live cap
- Best tested config:
  - `max_position_pct=0.10`
  - `rebalance_seeded_positions=True`
- `0.10` summary:
  - robust score `-53.36`
  - latest robust score `-1.96`
  - return mean `-1.47%`
  - return p25 `-3.28%`
  - worst drawdown `3.61%`
  - mean smoothness `0.001285`
- Latest-window flat start at `0.10`:
  - return `+0.23%`
  - Sortino `0.20`
  - max drawdown `2.61%`
- Comparison against the current live `0.17` branch:
  - `0.10 + rebalance`
    - robust score `-53.36`
    - latest robust score `-1.96`
    - latest flat return `+0.23%`
    - latest flat max drawdown `2.61%`
  - `0.17` without rebalance
    - robust score `-66.50`
    - latest robust score `-77.29`
    - latest flat return `+0.39%`
    - latest flat max drawdown `4.44%`
- Decision:
  - Promote the rebalance branch
  - Lower live `max_position_pct` from `0.17` to `0.10`
  - Accept the smaller flat-start upside in exchange for much better mixed-start robustness and smoother PnL

Files updated for the promotion:
- `binance_worksteal/trade_live.py`
- `deployments/binance-worksteal-daily/launch.sh`
- `deployments/binance-worksteal-daily/supervisor.conf`
- `tests/test_worksteal_trade_live.py`

Verification:
- `pytest tests/test_build_binance_spot_hourly_usd_proxies.py tests/test_worksteal_live_universe_coverage.py tests/test_worksteal_trade_live.py tests/test_worksteal_strategy.py tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_gemini_prompt.py -q`
- Result: `40 passed`

Deployment:
- Updated supervisor launch config to:
  - `--max-position-pct 0.10`
  - `--rebalance-seeded-positions`
- Restarted:
  - `sudo supervisorctl restart binance-worksteal-daily`
- Post-restart status:
  - `RUNNING`
  - PID `735022`
- Current live command:
  - `... --max-position-pct 0.10 ... --entry-proximity-bps 25 --rebalance-seeded-positions --symbols ... PEPEUSD POLUSD`

## Startup Preview + Gemini Overlay Check

Files updated:
- `binance_worksteal/trade_live.py`
- `binance_worksteal/gemini_intraday_eval.py`
- `deployments/binance-worksteal-daily/launch.sh`
- `deployments/binance-worksteal-daily/supervisor.conf`
- `tests/test_worksteal_trade_live.py`
- `tests/test_worksteal_gemini_intraday_eval.py`

Operational fix:
- Added daemon startup preview support in `trade_live.py`
  - default daemon behavior now runs one startup preview cycle before waiting for the next scheduled UTC run
  - the startup preview uses live market/account context but does **not** place orders or mutate `live_state.json`
  - this gives an immediate plan/log on restart without turning every restart into an unscheduled live trading cycle

Gemini status check:
- The live work-steal path previously launched with `--gemini --gemini-model gemini-2.5-flash`
- Direct provider check showed:
  - `gemini-2.5-flash`: current API key is quota-exhausted (`429 RESOURCE_EXHAUSTED`)
  - `gemini-3.1-flash-lite-preview`: responds successfully
- I therefore evaluated `gemini-3.1-flash-lite-preview` instead of trying to promote `2.5`

Artifacts:
- `analysis/worksteal_hourly_20260318/gemini_intraday_compare_latest60d_31_v1_scenarios.csv`
- `analysis/worksteal_hourly_20260318/gemini_intraday_compare_latest60d_31_v1_summary.csv`

Method:
- Current promoted live config as baseline:
  - `max_position_pct=0.10`
  - `rebalance_seeded_positions=True`
  - hourly execution
- Latest 60-day window:
  - `2026-01-13` to `2026-03-14`
- Representative starts:
  - `flat`
  - `BTCUSD`
  - `ETHUSD`
- Compared:
  - `rule_only`
  - `gemini_overlay` with `gemini-3.1-flash-lite-preview`

Result:
- `rule_only`
  - mean return `+0.23%`
  - Sortino `0.20`
  - max drawdown `2.61%`
  - mean trade count `38.7`
- `gemini-3.1-flash-lite-preview`
  - mean return `-0.26%`
  - Sortino `-0.20`
  - max drawdown `0.56%`
  - mean trade count `4.7`
- Read:
  - Gemini 3.1 sharply reduced activity and drawdown, but it also reduced PnL too much and turned the latest-window result negative
  - It is not an improvement over the current rule-only branch on the tested current-config replay
  - I did **not** promote Gemini into the live work-steal branch

Deployment decision:
- Remove `--gemini` from the live launcher
- Keep the work-steal daemon on the promoted rule-only strategy
- Keep Gemini as an experimental overlay only until it beats the current rule-only branch on market-sim

Verification:
- `pytest tests/test_worksteal_trade_live.py tests/test_worksteal_gemini_intraday_eval.py tests/test_worksteal_strategy.py tests/test_worksteal_robust_eval.py tests/test_worksteal_sweep.py tests/test_worksteal_gemini_prompt.py tests/test_worksteal_live_universe_coverage.py -q`
- Result: `42 passed`

Deployment:
- Restarted `binance-worksteal-daily`
- Post-restart status:
  - `RUNNING`
  - PID `1269920`
- Current live command:
  - `... --daemon --max-position-pct 0.10 ... --entry-proximity-bps 25 --rebalance-seeded-positions --symbols ... PEPEUSD POLUSD`
- Startup preview confirmed in supervisor logs:
  - daemon now logs a preview cycle immediately on restart
  - the preview did not mutate `live_state.json`

## 2026-03-18 Per-Symbol Hybrid Spot Expansion

Goal:
- Systematically evaluate adding new pairs to the `binance-hybrid-spot` strategy
- Currently deployed: BTC, ETH, SOL (Gemini 3.1 Flash Lite + Chronos2 forecasts)
- Candidates: DOGE, AAVE, LINK, XRP
- Per pair: train hourly LoRA, find correlated assets for multivariate, build forecast caches, backtest with Gemini signals

### SOL Deposit
- 9.5 SOL deposited from spot to cross-margin
- Margin level improved from 2.0 to 2.37
- Total margin net: ~$3,730

### Multivariate Chronos2 Trainer Extension
- Added `--covariate-symbols` and `--covariate-cols` to `scripts/retrain_chronos2_hourly_loras.py`
- Added `_prepare_inputs_multivariate()` to `chronos2_trainer.py`
- Passes correlated assets as additional channels alongside target OHLC
- Tests: `tests/test_chronos2_multivariate.py` (8 passed)

### Correlation Analysis
| Target | Top Correlated | Correlation |
|--------|---------------|-------------|
| DOGE | PEPE | 0.831 |
| DOGE | ADA | 0.815 |
| DOGE | AVAX | 0.812 |
| AAVE | ETH | 0.814 |
| AAVE | AVAX | 0.776 |
| LINK | ETH | 0.835 |
| LINK | AVAX | 0.830 |

### DOGE LoRA Training (Remote 5090)
- Retrained on hourly data (44k rows) vs previous daily LoRA (1534 rows)
- Three configs tested:

| Config | val_mae% | test_mae% | Notes |
|--------|----------|-----------|-------|
| **ctx512 lr=1e-4** | **3.22** | **2.80** | **Best** |
| ctx1024 lr=5e-5 | 8.39 | 4.78 | Underfitting |
| ctx512+PEPE multivar | 3.35 | 2.87 | PEPE covariate marginal |

- Winner: `DOGEUSDT_lora_hourly_ctx512_lr1e4_20260318_040341`
- Multivariate PEPE covariate did NOT help vs baseline ctx512
- Key finding: ctx512 with lr=1e-4 is optimal for hourly crypto LoRA

### AAVE LoRA Training
- val_mae%=2.67, test_mae%=4.33
- Checkpoint: `AAVEUSDT_lora_hourly_ctx512_lr1e4_20260318_043102`

### LINK, XRP LoRA Training
- In progress on remote 5090

### Code Changes
- `rl-trading-agent-binance/trade_binance_live.py`: added LINK, XRP to TRADING_SYMBOLS
- `rl-trading-agent-binance/hybrid_prompt.py`: added SOL, LINK, XRP to SYMBOL_BINANCE_MAP
- `rl-trading-agent-binance/eval_new_symbol.py`: new reusable per-symbol evaluation script
- `hyperparams/chronos2/hourly/DOGEUSDT.json`: updated to point to best hourly LoRA

### LINK LoRA Training
- val_mae%=2.06, test_mae%=3.69
- Checkpoint: `LINKUSDT_lora_hourly_ctx512_lr1e4_20260318_043358`

### XRP LoRA Training
- First attempt failed (remote disk full)
- Cleaned old stock LoRA experiments (82GB recovered)
- Re-running on remote 5090

### LoRA Results Summary
| Symbol | val_mae% | test_mae% | Config |
|--------|----------|-----------|--------|
| **DOGE** | **3.22** | **2.80** | ctx512 lr=1e-4 1000 steps |
| AAVE | 2.67 | 4.33 | ctx512 lr=1e-4 1000 steps |
| LINK | 2.06 | 3.69 | ctx512 lr=1e-4 1000 steps |
| XRP | - | - | training in progress |

### Backtest (7-day quick screen, 5x leverage)
- BTC standalone: +7.87%, Sort=10.70, 31 trades (validates approach)
- DOGE standalone: -0.10%, Sort=0.00, 1 trade (9% entry rate - very conservative)
- AAVE, LINK: signal generation was slow (Gemini API rate limits), partial results only

Key insight: DOGE entry rate is very low (9%) vs BTC (~18%). This means adding DOGE is low risk - worst case it barely trades. The model is appropriately cautious.

### Deployment: DOGE Added to Live
- Updated `deployments/binance-hybrid-spot/launch.sh` to include DOGEUSD
- Restarted `binance-hybrid-spot`
- Status: RUNNING, PID 2411766
- Live symbols now: BTCUSD ETHUSD SOLUSD DOGEUSD
- DOGE max position: 10% (conservative)
- Forecast cache already covers DOGE (h1+h24 via binance-rl-cache-refresh)

### XRP LoRA Training
- val_mae%=6.55, test_mae%=20.20
- Very high error due to limited data (10,514 hourly rows vs 44k for DOGE)
- **Not deployed** - needs more hourly data collection first

### Final LoRA Results Summary
| Symbol | val_mae% | test_mae% | Hourly Rows | Deployed |
|--------|----------|-----------|-------------|----------|
| **DOGE** | **3.22** | **2.80** | 44,022 | Yes |
| **AAVE** | **2.67** | **4.33** | 44,024 | Yes |
| **LINK** | **2.06** | **3.69** | 45,604 | Yes |
| XRP | 6.55 | 20.20 | 10,514 | No |

### Deployment: 6-Symbol Hybrid Spot
- Updated `deployments/binance-hybrid-spot/launch.sh`:
  - `--symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD`
- Updated supervisor conf: `deployments/binance-hybrid-spot/supervisor.conf`
- Restarted `binance-hybrid-spot`
- Status: RUNNING, PID 2424338
- Live symbols: BTC, ETH, SOL, DOGE, AAVE, LINK (6 total, up from 3)
- Max positions: BTC 25%, ETH 20%, SOL 15%, DOGE 10%, AAVE 15%, LINK 10%
- Execution: cross-margin 5x leverage, hourly cycles
- Cache refresh already covers: BTC, ETH, SOL, DOGE, AAVE (LINK needs adding)

### Next Steps
- Add LINK to `binance-rl-cache-refresh` forecast coverage
- Collect more XRP hourly data before retraining/deploying
- Monitor new symbols' live entry rates and PnL over next 24-48h
- If DOGE/AAVE/LINK show positive PnL, keep; if negative, remove
- Consider prompt tuning to increase alt entry rates if they stay below 10%
- Run longer backtests (30d) when Gemini API rate allows
