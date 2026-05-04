# Binance Production Systems

## Hourly XGB Margin-Pack Paused After Fresh Revalidation (2026-05-04 NZ / 2026-05-04 UTC)

No new Binance XGB entries are currently running.

Operational state:

- `binance-hourly-xgb-margin-pack` is `STOPPED`.
- Binance live process audit reports no live writer processes.
- Existing actionable margin positions are still covered:
  `positions=6`, `covered=6`, `missing=0`, `partial=0`.
- The FET/ONDO entry orders from the prior live cycle were canceled at `0`
  fill before further research sweeps.

Fresh current-code revalidation:

- The deployed config on the current live non-stable universe replays at only
  `+5.80%/mo`, `+25.17%` total, `26.85%` max DD, Sortino `1.96`, `116` exits.
  This invalidates the old deploy assumption for the live universe.
- Broad short-only sweep best sampled row:
  `+26.06%/mo`, `+151.54%` total, `37.68%` max DD.
- Best smoother broad row:
  `+24.63%/mo`, `+140.35%` total, `21.48%` max DD, Sortino `3.65`.
- Exact leverage replay of that smoother row:
  `2.4x` -> `22.54%/mo`, `19.58%` DD;
  `2.8x` -> `24.63%/mo`, `21.48%` DD;
  `3.4x` -> `27.77%/mo`, `25.92%` DD.
- Stricter drawdown scaling did not fix the frontier:
  `3.4x` dropped to `15.51%/mo` while DD stayed `25.92%`.
- `side_mode=both` was worse in sampled testing; the current XGB signal remains
  a one-sided short edge, not a robust hedged long/short book.

Decision: hold paused. The exact `3.4x` replay crosses the monthly PnL target,
but drawdown remains above the desired cap and the strategy is still short-only.
Do not restart or promote this runner until a current-code candidate clears the
return gate with materially better drawdown behavior.

## Hourly XGB Margin-Pack Live Rollout (2026-05-04 NZ / 2026-05-04 07:20 UTC)

Historical rollout note, now superseded by the paused revalidation section
above. At the time, the live deployment used the single Binance writer surface:

- Supervisor program: `binance-hourly-xgb-margin-pack`
- Launch: `deployments/binance-hourly-xgb-margin-pack/launch.sh`
- Runner: `scripts/binance_hourly_xgb_margin_trader.py --execute --daemon --run-on-start --cycle-minutes 60 --refresh-data-before-cycle`
- Label: `h12_short_corr168_08_aggr2777_dd1998_20260504`
- Sim result from aggressive sweep: `27.77%/mo`, `+165.39%` total over about 120 days, `19.98%` max DD, Sortino `4.85`, `260` exits.
- Shape: hourly XGB, horizon `12`, short-only, max `2` active positions, max `4` pending entry watchers, direct `168h` correlation gate at max signed corr `0.8`, concentrated allocation, `2.55x` gross, drawdown scaler start/full/floor `4.5%/15%/32%`.

Production isolation:

- Stopped and made non-autostart/non-autorestart: `binance-hybrid-spot`, `binance-meta-margin`, `binance-worksteal-daily`, `binanceexp1-selector`, `binanceexp1-solusd`, and `crypto30-daily`.
- Current Binance writer audit passes with exactly one writer: `xgb_hourly_pack`.
- Cache/collector services such as `binance-5min-collector` and cache refreshers remain running.

Live safety behavior verified:

- Existing margin account coverage is clean: `6` positions, `6` covered, `0` missing, `0` partial.
- Pre-existing long positions are covered by open `SELL` exits: `LINK`, `ETH`, `AAVE`, `DOGE`, `BTC`, `SOL`.
- Live cycle at `2026-05-04T07:18:30Z` used data through `2026-05-04T07:00:00Z` and decision timestamp `2026-05-04T05:00:00Z`.
- It selected two candidate shorts: `FILUSDT` and `TONUSDT`, about `$2645` notional each.
- Both entry orders were submitted as cross-margin `SELL`/`AUTO_BORROW_REPAY`, filled `0`, and were canceled. No `BUY` exit was placed because no short position existed.
- Post-execute coverage after the settle gate remained clean: `6` positions, `6` covered, `0` missing.

Safety patch in this rollout:

- The runner now waits `--post-cycle-settle-seconds` after entry/cancel handling before the post-execute coverage gate. This avoids false missing-coverage failures from Binance's transient margin asset state after canceled borrow-style orders.
- Filled short deltas still get immediate `BUY`/`AUTO_REPAY` exits, and unfilled entries are canceled by default.
- Entry orders are now supervised for the full simulated entry TTL: `3600s`.
  The process polls fills during that window, places `BUY`/`AUTO_REPAY` exits
  for any filled short quantity, then cancels remaining unfilled entry quantity.
- New live entries are capped by current margin gross exposure: `equity * max_leverage * risk_scale - existing_gross`, so old covered positions do not let the runner over-gross the account.
- Daemon sleep is measured against cycle start, so a full-hour supervised entry
  window does not stretch the cadence into a two-hour loop.

Verification:

- Focused tests: `58 passed` for the XGB runner, Binance live process audit, margin exit coverage, and margin wrapper.
- Shell checks: supervisor launch script syntax passes; relevant diff whitespace check passes.
- Current artifact: `analysis/binance_hourly_xgb_margin_plan_latest.json`.
- Follow-up restart with `--entry-fill-wait-seconds 3600` placed `FETUSDT`
  and `ONDOUSDT` entry orders; both remained open as `NEW` after the old
  60-second cancel point, confirming the supervised 1-hour TTL is active.

## Hourly XGB Margin-Pack Dry-Run Deploy Blocked (2026-05-04 NZ / 2026-05-04 05:35 UTC)

No live deployment.

Candidate prepared for careful Binance cross-margin rollout:

- Label: `h12_short_corr168_08_2795_20260504`
- Sim result: `27.95%/mo`, `+168.01%` total over about 120 days, `29.66%` max DD, Sortino `4.09`, `246` exits.
- Shape: hourly XGB, horizon `12`, short-only, max `2` positions, direct `168h` correlation gate at max signed corr `0.8`, concentrated allocation, `2.55x` gross, drawdown scaler start/full/floor `4.5%/15%/32%`.
- Deploy runner: `scripts/binance_hourly_xgb_margin_trader.py`.

Safety work added:

- Runner is dry-run by default; live orders require `--execute`, `ALLOW_BINANCE_XGB_LIVE_TRADING=1`, fresh hourly data, clean Binance writer isolation, and existing margin exit coverage.
- Stable-base pairs such as `FDUSDUSDT`/`USDCUSDT` are filtered before liquidity top-N selection.
- Short entries are cross-margin limit `SELL` orders with `AUTO_BORROW_REPAY`.
- The execute path now watches entry fills, places matching `BUY`/`AUTO_REPAY` exits for filled short quantity, and cancels unfilled entry orders by default so an entry cannot fill later without the runner supervising exit coverage.
- `scripts/binance_margin_exit_coverage.py` now audits both long and short positions: longs require open `SELL` coverage, shorts require open `BUY` coverage.

Dry-run verification:

- Hourly data refresh wrote current bars through `2026-05-04T05:00:00Z`; downloader exited nonzero only because it reports old historical exchange gaps.
- Inference dry-run command: `.venv313/bin/python scripts/binance_hourly_xgb_margin_trader.py --json-out analysis/binance_hourly_xgb_margin_plan_20260504_dryrun.json`
- Dry-run result: `data_end=2026-05-04T05:00:00Z`, `decision_ts=2026-05-04T03:00:00Z`, data age about `0.57h`, margin equity about `$3059.47`, `risk_scale=1.000`, candidates `0`.
- Offline dry-run with live account state disabled also selected `0` candidates, so the no-trade result is the model signal at this decision point, not only active-account symbol skipping.
- Current margin exit coverage: `positions=3`, `covered=3`, `partial=0`, `missing=0` (`LINK`, `ETH`, `AAVE` are long positions covered by open `SELL` exits).
- Focused tests: `88 passed` across XGB runner, margin wrapper, exit coverage, snapshot coverage, live process audit, hourly pack, and simulator allocator tests.

Live blocker:

- Process audit still fails. Conflicting Binance writers are present: `binanceleveragesui.trade_margin_meta` PID `1943` and `binance_worksteal.trade_live` PID `3833`; hybrid margin PID `71548` is also active.
- Decision: do not run `--execute` on this machine until supervisor/root state is reconciled to a single allowed XGB margin-pack writer and the dry-run produces a real candidate on fresh data.

## Binance33 Year-Scale Robustness Pass (2026-05-02 NZ / 2026-05-01 UTC)

No production deployment.

Data refresh:

- Refreshed all 33 `trainingdatadailybinance/*.csv` proxy files through `2026-05-01`.
- Exported separate refreshed binaries under `analysis/binance33_refresh_20260502/` so the canonical `pufferlib_market/data/binance33_daily_*.bin` files were not overwritten.
- Refreshed val span is now `2025-10-01` through `2026-05-01` (`213` daily bars), versus the stale local cutoff around `2026-03-16`.

Current best recent candidate after adding April:

- Candidate: `allow_sui_strict_rb2_gross4p60_maxw0p34_sr1p04`.
- Old March-cutoff val: median monthly `+49.554%`, p10 `+42.627%`, p90 DD `19.708%`, worst DD `19.984%`, `0/45` negative windows.
- Refreshed val through `2026-05-01`: median monthly `+44.696%`, p10 `+28.429%`, worst monthly `+21.248%`, `0/93` negative windows, but p90 DD `48.446%` and worst DD `51.821%`.
- Latest refreshed 120d trace: `+184.43%`, DD `40.22%`, `207` trades.
- Viewer regenerated at `analysis/binance_backtest_space.html` against the refreshed validation window.

Additional robustness checks:

- Exact deterministic Binance33 rule sweep artifact: `analysis/binance33_rule_sweep_train_120_365_stride10_20260502.csv`.
- Sweep settings: train history, lag `2`, slippage `20bps`, max leverage `1.0`, 1440 generated long/short/regime configs, 120d and 365d rolling windows.
- Best active 120d train row was only about `+1.74%/mo` median with p10 `-14.61%/mo`, p90 DD `60.06%`, and `77/162` negative windows.
- Best active 365d train row was about `+3.61%/mo` median with p10 `-11.86%/mo`, p90 DD `80.54%`, and `59/137` negative windows.
- No active deterministic row had p90 DD `<=20%`, no active row had nonnegative p10, and no active row approached the `>=27%/mo` production gate.
- Shadow/profit-gate scout artifact: `analysis/binance33_meta_shadow_gate_scout_20260502.csv`. The gate reduced some recent drawdown but did not create a production row; high-PnL rows still had old-history p10 around `-8%/mo` and old-history p90 DD around `30-36%`, while stricter rows mostly went flat and still had extreme old-history worst DD.
- Old-history symbol failure analysis found `77/81` bad windows for the high-gross SUI-allowed candidate. Worst negative contributors were `SOLUSD`, `DOGEUSD`, `SHIBUSD`, `BTCUSD`, `DOTUSD`, `NEARUSD`, and `LINKUSD`. Excluding the first seven improved old-history median to roughly flat but left p10 around `-29%/mo` and p90 DD around `84%`.

Hedged/CVaR portfolio-packing pass:

- Added `scripts/sweep_binance33_hedged_cvar_pack.py` and `tests/test_sweep_binance33_hedged_cvar_pack.py` for small 2-3 position long/short books with binary fills, CVaR/vol sizing, per-symbol weight caps, gross caps, net caps, and separate long-signal support.
- Fixed the hedged packer so `max_net=0.0` means true neutral instead of disabling the net cap.
- Same-score hedged CVaR scout artifact: `analysis/binance33_hedged_cvar_pack_scout_20260502.csv`. The best smooth refreshed-val rows were only about `+0.5%` to `+2.5%/mo`; train rows were flat/negative. Hedging reduced drawdown but mostly removed the edge.
- Dual-alpha scout artifact: `analysis/binance33_dual_hedged_scout_20260502.csv`. Trend60 longs plus `rand0038` shorts produced attractive refreshed-val rows, e.g. `+5.19%/mo`, p10 `+1.80%/mo`, p90 DD `19.47%`, but the matching train row was `-2.25%/mo`, p10 `-11.30%/mo`, p90 DD `41.85%`.
- Corrected repeatable dual-signal scout artifact: `analysis/binance33_hedged_cvar_dual_signal_scout_20260502.csv`. After enforcing neutral books, the best train+val rows had low DD but were effectively flat (`0.00%/mo`, median trades `0`); the best refreshed-val rows around `+2.92%/mo` still had matching train median `-1.92%/mo`, p10 `-8.00%/mo`, and p90 DD `33.63%`.
- No hedged row passed the production gate; no paired row had nonnegative train p10.

Decision: hold. This is now blocked by both refreshed recent drawdown and old-history robustness. Hedged portfolio packing improves the shape but does not recover enough return under realistic lag-2 binary fills. The next useful research is a trained walk-forward regime/mixture model with an explicit cash expert and survival objective across `2021-2026`; fixed symbol pruning, simple market gates, deterministic rules, shadow PnL gates, and static CVaR packing are not enough.

Additional search pass:

- Added `scripts/sweep_binance33_meta_risk_overlay.py` to test causal overlays around meta-anneal candidates: always-on shadow-PnL gates, circuit breakers with cooldown, and book drawdown throttles. Initial `cand00426` overlay screens went either flat or low-return; the documented high-gross SUI candidate is not reproduced by raw `cand00426` JSON alone.
- Expanded `scripts/search_binance33_regime_gates.py` with cross-sectional dispersion/skew/range gates (`iqr_ret5`, `iqr_ret20`, `skew_ret5`, `skew_ret20`, `mkt_absret5`, `mkt_range1`).
- Dispersion gate scout artifact: `analysis/binance33_regime_gate_dispersion_rand0045_20260502.csv` (partial exact replay, stopped after `1197` rows because results were already stable). Best recent-val rows reached about `+9.47%/mo` with p10 `+6.96%/mo` and p90 DD `15.36%`, but their matching train rows were still around `-9%/mo` median, p10 `-28%/mo`, and p90 DD `75%+`. Best combined rows were near-flat with too few trades.
- Refreshed XGBoost targeted artifact: `analysis/binance33_xgb_refreshed_xgb08_11_leverage_20260502.csv`. `xgb08` at higher leverage reached up to `+15.93%/mo` median on refreshed val, but p10 was `-13.29%/mo`, `17/47` windows were negative, and p90 DD was `71.72%`.
- Refreshed XGBoost portfolio-pack artifact: `analysis/binance33_xgb_pack_refreshed_xgb08_11_20260502.csv`. Packing xgb08/xgb11 shorts across multiple symbols reduced upside to about `+5.24%/mo` and still had p10 `-5.11%/mo`, `14/47` negative windows, and p90 DD `42.48%`.
- No new row passed the `>=27-30%/mo`, nonnegative-p10, low-drawdown gate.

Decision remains hold/no deploy. The new evidence says the recent short edge is not just a bad static pack or a missing simple market-state gate; it is a regime-specific edge that breaks badly across older crypto regimes. More search should focus on a true walk-forward learner/objective that treats "cash" as a first-class expert and penalizes stale-regime survival failures during training, not post-hoc gating.

## Binance33 Meta-Anneal Deployment Review (2026-05-01 NZ / 2026-04-30 UTC)

No production deployment.

Best recent-validation candidate from the current pass:

- label: `allow_sui_strict_rb2_gross4p60_maxw0p34_sr1p04`
- source row: `cand00426` from `analysis/binance33_meta_anneal_linear_hand_aug_active_20260501.csv`
- excludes: `PEPEUSD,ICPUSD,TAOUSD,ATOMUSD,FILUSD,TIAUSD,ALGOUSD,BNBUSD,AAVEUSD`
- SUI is allowed; TAO remains excluded because exact allow-back sweeps were worse.
- book: short portfolio, `top_k=4`, rebalance every `2` days, `max_gross=4.60`, `max_weight=0.34`, `short_risk_mult=1.04`, `decision_lag=2`, binary fills.
- recent 120d validation at 20bps slippage: median monthly `+49.554%`, p10 monthly `+42.627%`, p90 DD `19.708%`, worst DD `19.984%`, `0` negative windows, median trades `116`.
- latest 120d trace: `+311.67%`, DD `19.71%`, `109` trades.
- viewer regenerated: `analysis/binance_backtest_space.html`.

Blocking robustness result:

- Same candidate on older Binance33 train history (`2021-01-01` through `2025-09-30`, 120d windows, stride 10, lag 2, 20bps) fails catastrophically: median monthly `-34.54%`, p10 monthly `-99.37%`, worst monthly `-99.44%`, p90 DD `100%`, worst DD `100%`, `128/162` negative windows.
- Simple BTC/market/breadth/volatility gates around the candidate did not fix this; the best gates preserved some recent PnL but still had 100% old-history drawdown in at least one window.
- Residual drawdown analysis on the recent validation worst case mostly pointed at `DOT`, `SOL`, `NEAR`, and `UNI`, but excluding those reduced robustness/objective quality versus the SUI-allowed winner.
- Local daily Binance data is stale around `2026-03-16`, so April 2026 is not included in the current local daily binary data.
- Current Binance live process state is still not a clean single-writer deployment surface for this strategy, and the existing `trade_crypto30_daily.py` path is a long-only spot/PPO-style daemon, not a margin short-portfolio executor for this candidate.

Decision: do not promote this to live production. It can be kept as a recent-regime research/shadow candidate only. Before any live margin rollout, refresh daily data through the current date, re-export train/validation binaries, rerun the 0/5/10/20bps binary-fill lag-2 grids including April 2026, build a dedicated dry-run/paper margin executor for the vector portfolio, and fix Binance writer isolation at supervisor/root level.

Research direction: more compute should optimize a walk-forward objective across `2021-2026`, including old-history survival and a cash/regime expert. More annealing on the recent slice alone is likely to improve the recent leaderboard while worsening year-scale reliability.

## Runtime Observation + Offline Head-to-Head (2026-04-30 NZ / 2026-04-30 09:30 UTC)

Current machine process scan shows the 2026-04-27 production note is stale:

- `binance-hybrid-spot` is still running the documented six-symbol margin launch (`robust_reg_tp005_dd002`, 0.5x).
- Retired/conflicting writers `binanceleveragesui.trade_margin_meta` PID `1943` and `binance_worksteal.trade_live` PID `3833` are still present but SIGSTOP'ed (`Tl`), not cleanly removed from supervisor.
- A new `binanceleveragesui.trade_margin_sui` process is running and is not covered by the 2026-04-27 ledger. Treat Binance live account state as writer-contaminated until this is reconciled at supervisor/root level.

Offline eval work added:

- `scripts/binance_head_to_head.py` normalizes PPO, rule, XGB, XGB-pack, and hourly-pack CSV artifacts into one gateable leaderboard.
- `scripts/sweep_binance33_rule_portfolio_pack.py` tests diversified deterministic daily rule portfolios.
- `scripts/search_binance33_easy_universe.py` tests calibration-ranked symbol pruning plus raw/augmented XGBoost retraining.
- `scripts/sweep_binance33_rule_risk_controls.py` and `scripts/search_binance33_linear_rules.py` test risk controls and composite linear rules around the recent short-reversion edge.
- `scripts/search_binance33_regime_gates.py` tests explicit cash/regime gates around the best linear rule.
- `scripts/sweep_binance33_lgbm.py` tests LightGBM `4.6.0`; `.venv312` XGBoost was upgraded from `3.1.1` to `3.2.0` for latest-library reruns.
- Latest combined artifact: `analysis/binance_head_to_head_20260430_after_latest_libs.csv`.

Result: the recent validation frontier improved. Linear rule `rand0045` at `1.305x` reaches `30.04%/mo`, p10 `19.54%/mo`, `0/45` negative windows, p90 drawdown `19.96%`, and 37 median trades on recent 120d/20bps validation. A simple market-regime gate (`mkt_ret20gt-0.1`) improves recent drawdown to `15.36%` while keeping `30.05%/mo`, but older-history stride-10 validation still fails (`-18.01%/mo`, `137/162` negative windows). XGBoost `3.2.0` and LightGBM targeted reruns did not beat this. This is not deployable, and live Binance account state is still writer-contaminated.

Decision: no Binance promotion/deploy. The next useful work is not another broad PPO swap; it is a regime-conditioned short-reversion learner with explicit out-of-regime cash gating, plus fixing live writer isolation before any margin experiment is considered production-capable.

## Runtime + Backtest Re-Audit (2026-04-27 NZ / 2026-04-27 10:50 UTC)

Current machine state:

- `binance-hybrid-spot` is now aligned with `deployments/binance-hybrid-spot/launch.sh`.
  - running PID: `71548`
  - checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
  - symbols: `BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD`
  - leverage: `0.5x`
- live-writer isolation is still not clean:
  - `binanceleveragesui.trade_margin_meta` PID `1943`
  - `binance_worksteal.trade_live` PID `3833`
  - `binanceexp1` selector/hourly processes are cache-only and not counted as writers.
- recent hybrid runtime snapshots are completing and exit coverage is present for all six meaningful long positions, but runtime audit still flags every managed symbol as oversized versus the current planned sleeve. Treat current live PnL as account-state contaminated until the extra writers are stopped persistently.

Offline live-like holdout rerun, same six-symbol launch mask, `decision_lag=2`, `slippage_bps=5`, `fill_buffer_bps=5`, `0.5x`, shorts disabled:

| Checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| launch `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |
| stale old `robust_champion` | `-4.48%` | `-6.17` | `-8.57%` | `5.17%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` | `6.31%` |
| `per_env_adv_smooth` | `-3.59%` | `-4.71` | `-10.75%` | `6.06%` |

Artifacts: `analysis/current_binance_prod_eval_20260427_runtime_compare/`

Current `binance-meta-margin` exact live-like replay after refreshing DOGE/AAVE 5m data through `2026-04-27T10:50Z`:

| Strategy | 30d return | MaxDD | Sortino | Trades |
|---|---:|---:|---:|---:|
| DOGE single model | `-0.20%` | `-0.33%` | `-1.56` | `137` |
| AAVE single model | `+0.41%` | `-0.42%` | `1.92` | `79` |
| deployed `winner_cash/omega` meta selector | `0.00%` | `0.00%` | n/a | `0` |

Artifact: `analysis/binance_meta_margin_current_live_like_30d_refreshed_20260427.json`

Production hint tested in the XGB portfolio packer:

- Narrowed universe to `DOGEUSDT,AAVEUSDT`, current 120d, `1x`, `20bps` fill buffer, strict entry/cash-gate-like settings.
- Stopped after 18 sampled configs because active variants were negative and strict variants went flat.
- Best non-flat row: `-0.43%` monthly, `-1.69%` total, `2.83%` max DD, 3 sells.
- Artifact: `analysis/binance_pack_current_doge_aave_prod_hint_long_20bps_120d_20260427.csv`

Decision:

- Do not infer that Binance production is secretly outperforming the simulator. The current hybrid launch is negative in live-like holdout; the meta-margin path is mostly a cash gate, not a high-PnL strategy; and the narrow DOGE/AAVE XGB transfer does not improve PnL.
- The useful transfer is the conservative `winner_cash` / live-like profit-gate pattern: new entries should require a recent strategy-level positive edge, not just a positive per-row model forecast.
- Next research direction should be an explicit rolling profit-gate / cash-state feature in the hourly packer, plus concentrated AAVE/ETH/BTC-style single/pair candidates. Do not spend more time on broad six-symbol RL checkpoint swaps unless lag-2 50-window holdout becomes positive.

## Runtime Repair (2026-04-27 NZ / 2026-04-26 21:52 UTC)

The live account had `5` margin closing sell orders but `6` meaningful long positions. `SOL` was the uncovered position (~$228); the remaining extra assets in the account were sub-dollar dust.

Actions taken:

- pushed `b63ae6b6` / `d743b9ea` to `origin/main`
- stopped conflicting live writer processes with `SIGSTOP`:
  - `binanceleveragesui.trade_margin_meta` PID `1943`
  - `binance_worksteal.trade_live` PID `3833`
- restarted the hybrid bot by terminating PID `1938`; supervisor relaunched `binance-hybrid-spot` as PID `71548`
- verified post-restart sell coverage for `BTC`, `ETH`, `SOL`, `DOGE`, `AAVE`, and `LINK`
- left one `SOLUSDT` buy ladder open below market; this is an entry order, not missing exit coverage

Important persistence note: direct `supervisorctl` was blocked by socket permissions, so the stopped conflicting processes are paused at the Unix process level. Local ignored supervisor templates now set retired Binance writers to `autostart=false` and `autorestart=false`, but `/etc/supervisor/conf.d/` still needs a root-level update for this to survive supervisor or host restarts.

## CRITICAL (2026-04-07): Lag=0 Validation Is Untrustworthy
`robust_champion` was briefly deployed after a "+216% Sort=25.06" holdout — this was run at `decision_lag=0` which embeds lookahead bias (agent sees bar t features and trades at bar t close, impossible live). At realistic `decision_lag=2` on the SAME `mixed23_latest_val` data (50 windows x 30h):

| Checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| robust_champion (lag=0 claim: +12.4%/+3.35) | -2.8% | -0.50 | -14.0% |
| robust_reg_tp005_dd002 | +0.0% | +0.76 | -26.9% |
| robust_reg_tp005_ent | -5.5% | -2.09 | -21.1% |

**PROTOCOL**: Never trust a holdout unless run with `--decision-lag 2` AND a slippage sweep (0/5/10/20 bps). `evaluate_holdout.py` now has `--slippage-bps`. The C training env has a built-in `t_obs = t-1` single-bar lag, but production exhibits >=2 bars lag (forecast build + Gemini call + order placement). Training sim vs production gap is the ROOT CAUSE of why every deploy since 2026-03 looks great in holdout and drifts flat/negative live.

Launch.sh reverted to dd002 and now pins live Gemini inference to `gemini-3.1-pro-preview` after supervisor restart on 2026-04-12:
`sudo supervisorctl restart binance-hybrid-spot`

Follow-up production isolation and strategy patch on 2026-04-12:

- stopped the conflicting Binance writers `binance-meta-margin`, `binance-worksteal-daily`, and `binance-sui-margin`
- restarted `binance-hybrid-spot`, now running as PID `1923453`
- live snapshots now record per-symbol forecast deltas/confidence plus `exit_order_coverage`
- hybrid allocation cycles now top up closing sell orders for held positions instead of only trimming oversized inventory
- negative margin base inventory now stays visible in the portfolio snapshot, and the hybrid buy pass can place explicit `short_cover` orders back toward zero

Follow-up pipeline audit hardening on 2026-04-12:

- hybrid prompt now shows the real live effective leverage instead of implying `5x`
- prompt now includes negative inventory in the portfolio section so Gemini can see accidental shorts / debt
- Chronos2 context now carries forecast issuance timestamps and lag so stale cache rows are visible instead of silently treated as fresh
- executable code defaults were moved off `gemini-2.5*`; active Binance runtime and eval paths now default to `gemini-3.1-*`

Artifacts: `analysis/robust_champion_validation/slip{0,5,10,20}.json`

## Audit (2026-04-09): Live-Like Six-Symbol Replay Is Still Not Deployable

Artifacts: `analysis/current_binance_prod_eval_20260409/`

### Reproduced current live config

Live source of truth remains:

- launch: `deployments/binance-hybrid-spot/launch.sh`
- symbols: `BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD`
- leverage: `0.5x`
- model: `gemini-3.1-pro-preview`
- RL checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`

### Runtime drift confirmed on machine

Follow-up machine audit on 2026-04-09 showed the launch file is **not** what the running hybrid PID is executing:

- launch target checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- running hybrid checkpoint: `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt`
- running hybrid PID: `1712424`

The process audit was also tightened in `src/binance_live_process_audit.py` so cache-only research helpers no longer count as live-writer conflicts.

After that fix, the remaining true live-writer conflicts on the machine are:

- `binanceleveragesui.trade_margin_meta`
- `binance_worksteal.trade_live`

So the current production risk is not just model quality; it is also checkpoint drift plus shared-account contamination from multiple real live writers.

Production-faithful replay settings:

- data: `pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin`
- `decision_lag=2`
- `slippage_bps=5`
- `fill_buffer_bps=5`
- `fee_rate=0.001`
- long-only
- tradable symbols masked to the six live Binance symbols

Reproduced 30-step / 50-window baseline:

| Checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| live `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |

### Candidate sweep under the same live constraints

Best six-symbol 30-step candidates tested on 2026-04-09:

| Checkpoint | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `gspo_like_drawdown_mix15_tp01` | `-1.26%` | `-1.48` | `-10.25%` |
| `gspo_like_drawdown_mix15` | `-1.38%` | `-2.47` | `-6.30%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` |
| live `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` |

None of the tested six-symbol configurations came out positive on the 30-step median.

### Full-span check with early-stop disabled

To avoid the holdout early-exit heuristic hiding losses, the top candidates were re-run across the full 179-step slice with `--no-early-stop`:

| Checkpoint | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `-5.75%` | `-0.18` | `18.84%` |
| `gspo_like_drawdown_mix15` | `-20.12%` | `-2.21` | `24.64%` |
| `gspo_like_drawdown_mix15_tp01` | `-21.04%` | `-1.83` | `28.61%` |
| live `robust_reg_tp005_dd002` | `-24.07%` | `-2.19` | `33.65%` |
| `reg_combo_2` | `-29.75%` | `-1.90` | `38.17%` |

`gspo_like_smooth_mix15` is the least bad checkpoint on the full span, but still not good enough to promote as-is.

### Subset search result

The six-symbol basket is hurting `gspo_like_smooth_mix15`. Searching all subsets of the six live symbols at the live `0.5x` leverage found:

| Tradable subset | med_ret | med_sort | p10_ret |
|---|---:|---:|---:|
| `ETHUSD` | `-0.06%` | `-3.46` | `-2.10%` |
| `BTCUSD,ETHUSD` | `-0.19%` | `-3.86` | `-2.53%` |
| `BTCUSD` | `-0.26%` | `-3.68` | `-1.64%` |

These are much better than the live six-symbol basket, but still slightly negative on median 30-step replay.

### Leverage simulator bug fixed

`pufferlib_market/hourly_replay.py` was silently disabling long leverage above `1.0x` for limit entries by clamping long notional back to available cash inside `_open_long_limit`.

Fix applied on 2026-04-09:

- leveraged long limit entries now allow negative cash, matching the existing equity accounting
- regression test added in `tests/test_pufferlib_market_hourly_replay.py`

Relevant test command:

```bash
source .venv313/bin/activate
pytest -q tests/test_pufferlib_market_hourly_replay.py tests/test_evaluate_holdout.py
```

Post-fix full-span result for the best concentrated candidate (`gspo_like_smooth_mix15`, `ETHUSD` only):

| leverage | total_ret | sortino | max_dd |
|---|---:|---:|---:|
| `0.5x` | `+2.87%` | `1.90` | `2.18%` |
| `1.0x` | `+5.61%` | `1.91` | `4.35%` |
| `2.0x` | `+10.68%` | `1.91` | `8.67%` |

But the same ETH-only setup remains negative on the 30-step median:

| leverage | med_ret | p10_ret |
|---|---:|---:|
| `0.5x` | `-0.06%` | `-2.10%` |
| `1.0x` | `-0.11%` | `-4.21%` |
| `2.0x` | `-0.23%` | `-8.41%` |

### Current recommendation

- Do **not** promote a new live checkpoint yet.
- The best paper candidate to keep evaluating is:
  - checkpoint: `pufferlib_market/checkpoints/mixed23_a40_sweep/gspo_like_smooth_mix15/best.pt`
  - tradable subset: `ETHUSD` only
  - leverage to study: `1.0x` to `2.0x`
- The next training loop should optimize directly for the concentrated `BTC/ETH` or `ETH`-only deployment profile, because the mixed 23-symbol models improve materially once masked down to that subset.

### Meta-selector crossover assessment (2026-04-14)

Tested PnL-momentum meta-selection (from stock meta-selector work) on crypto30 daily PPO models.

- 4-model softmax ensemble (current prod): +81.36% over 192d val data
- Meta-selector best (21m, lb=7 k=1): +16.16% with 48% MaxDD
- Meta-selector best (21m, lb=7 k=3): +7.61% with 33% MaxDD
- Individual models: ALL negative (-0.04% to -82.60%)
- **Conclusion**: meta-selection does NOT help for crypto30. The curated 4-model softmax ensemble is already optimal.
- Root cause: individual crypto models are too weak for momentum-based switching to find edge.
- For hourly crypto (binance-hybrid-spot): all models remain negative at lag=2 -- meta-selector cannot fix this.
- Artifact: `scripts/meta_strategy_crypto30_backtest.py`

### Follow-up candidate ranking (2026-04-09 later)

Artifacts:

- `analysis/binance_prod_live6_23sym_candidates_20260409/`
- `analysis/per_symbol_all23_holdout_20260409/`
- `analysis/live6_top_singletons_fullspan_20260409/`
- `analysis/per_symbol_all23_fullspan_2x_20260409/`

Fresh six-symbol 30-step / 50-window rerun across the main 23-symbol-compatible checkpoints:

| Checkpoint | med_ret | med_sort | p10_ret | med_dd |
|---|---:|---:|---:|---:|
| `per_env_adv_smooth` | `-2.08%` | `-2.39` | `-10.23%` | `5.92%` |
| `reg_combo_2` | `-2.36%` | `-2.54` | `-11.74%` | `6.85%` |
| `gspo_like_smooth_mix15` | `-2.38%` | `-2.05` | `-9.29%` | `6.31%` |
| launch `robust_reg_tp005_dd002` | `-2.90%` | `-3.84` | `-5.94%` | `4.63%` |
| actual running `robust_champion` | `-4.48%` | `-6.17` | `-8.57%` | `5.17%` |

Important read:

- `per_env_adv_smooth` was the least-bad six-symbol basket on median return, but it did **not** produce a deployable positive basket.
- the actual running checkpoint `robust_champion` is materially worse than the launch target on this production-shaped slice

Best meaningful concentrated full-span results at `2.0x` leverage:

| Checkpoint | subset | total_ret | sortino | max_dd |
|---|---|---:|---:|---:|
| `gspo_like_smooth_mix15` | `ETHUSD` | `+10.68%` | `1.91` | `8.67%` |
| `gspo_like_smooth_mix15` | `BTCUSD` | `+7.19%` | `1.03` | `15.66%` |
| `gspo_like_smooth_mix15` | `DOGEUSD` | `+5.76%` | `0.86` | `43.27%` |

Counterexamples from the same pass:

- launch `robust_reg_tp005_dd002` on `AAVEUSD`: `-10.36%`, Sortino `-1.14`, DD `17.65%` at `0.5x`
- `robust_champion` on `BTCUSD`: flat median on the short-window sweep was a no-trade artifact, and full-span `2.0x` still came out `-10.06%`

Bottom line:

- no existing model/subset combination found here is close to `27%` monthly PnL under the live-like lag/slippage rules
- the best current paper lane is still concentrated `gspo_like_smooth_mix15` on `ETHUSD`, studied at `1.0x` to `2.0x`, but it remains a research candidate rather than a live promotion

## Active Services

### 1. binance-hybrid-spot
- **Supervisor**: `binance-hybrid-spot` (auto-restart)
- **Entry**: `rl_trading_agent_binance/trade_binance_live.py`
- **Launch**: `deployments/binance-hybrid-spot/launch.sh`
- **Config**: `deployments/binance-hybrid-spot/supervisor.conf`
- **Python**: `.venv313`
- **Symbols**: BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD + ADA XRP SUI (positions held)
- **Mode**: Cross-margin, 0.5x leverage, hourly cycles
- **Launch target RL model**: `pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- **Actual running RL model on 2026-04-09**: `pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt`
- **Previous model**: `robust_reg_tp005_ent` (switched 2026-03-31, dd002 has better 120d: +54.6% Sort=2.85 vs +42.4% Sort=2.71)
- **LLM**: Gemini 3.1 Pro Preview
- **Equity**: ~$3,025 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-hybrid-spot-error.log`

**Cycle flow**:
1. Fetch portfolio + margin balances
2. Compute Chronos2 h24 forecasts
3. Run RL inference (pufferlib checkpoint)
4. Build prompt -> call Gemini for allocation
5. Place/cancel margin limit orders
6. Sleep 3600s

**Current behavior** (2026-03-31):
- Gemini very defensive: 65-77% cash allocation
- RL consistently signals LONG_XRP and LONG_ADA
- DOGE position being sold down via existing sell order
- Stale orders auto-cancelled after 2h
- **Model switch** (2026-03-31): `robust_reg_tp005_dd002` deployed
  - 120d backtest: +54.6% ret, Sort=2.85, DD=26.9%, 50 trades, 54% WR
  - Exec param sweep: slippage insensitive, fill buffer insensitive
  - Better than ent at all periods except 30d

### 2. binance-worksteal-daily
- **Supervisor**: `binance-worksteal-daily` (auto-restart)
- **Entry**: `binance_worksteal/trade_live.py`
- **Launch**: `deployments/binance-worksteal-daily/launch.sh`
- **Config**: `deployments/binance-worksteal-daily/supervisor.conf`
- **Python**: `.venv313`
- **Universe**: `binance_worksteal/universe_live.yaml` (15 symbols)
- **Mode**: Spot+margin, dip-buying with SMA-20 filter
- **Equity**: ~$3,044 (2026-03-31)
- **Logs**: `/var/log/supervisor/binance-worksteal-daily-error.log`
- **State**: `binance_worksteal/live_state.json`
- **Trades**: `binance_worksteal/trade_log.jsonl`

**Parameters** (from launch.sh):
- `--max-position-pct 0.10`
- `--rebalance-seeded-positions`
- dip_pct=0.20, tp=20%, sl=15%, trail=3%, maxpos=5, maxhold=14d, sma=20

**Current positions** (2026-03-31):
- ADAUSD: entry $0.2726 (Mar 25), target $0.27, stop $0.23
- ZECUSD: entry $238.05 (Mar 25), target $285.66, stop $202.34
- XRPUSD: entry $1.36 (Mar 26), target $1.42, stop $1.16

## Monitoring

```bash
# Check service status
sudo supervisorctl status

# View recent logs
tail -100 /var/log/supervisor/binance-hybrid-spot-error.log
tail -100 /var/log/supervisor/binance-worksteal-daily-error.log

# Check running processes
ps aux | grep trade | grep -v grep

# Restart services
sudo supervisorctl restart binance-hybrid-spot
sudo supervisorctl restart binance-worksteal-daily

# View worksteal positions
python3 -m json.tool binance_worksteal/live_state.json
```

## Deployment

```bash
# After code changes, restart to pick up new code:
sudo supervisorctl restart binance-worksteal-daily
sudo supervisorctl restart binance-hybrid-spot

# Or kill the process (supervisor auto-restarts):
kill -TERM $(pgrep -f "trade_live.*--live")
kill -TERM $(pgrep -f "trade_binance_live")
```

## Known Issues (2026-03-31)

### Fixed: LOT_SIZE sell failures (worksteal)
- **Symptom**: Positions stuck 6 days, all sell orders rejected by Binance
- **Root cause**: Running process (started Mar 26) predated `_prepare_limit_order` quantization
- **Fix**: Restarted process to pick up code with exchange filter quantization
- **Prevention**: `_prepare_limit_order` now quantizes qty to stepSize before submission

### Fixed: Invalid symbols in universe_v2.yaml
- HYPEUSDT, KASUSDT removed (not valid Binance margin symbols)
- Was causing APIError -1121 every scan cycle

### Fixed: Margin 15% limit-price spam
- ENJUSD deep-dip buy orders were being submitted every 4h and rejected (code -3064)
- Added pre-check in `_prepare_limit_order`: rejects orders >14% from index price

## Market Simulator Validation

```bash
# Validate pufferlib model with binary fills (ground truth):
python -m pufferlib_market.evaluate_fast \
  --checkpoint <path> \
  --data-path pufferlib_market/data/crypto15_daily_val.bin \
  --eval-hours 720 --n-windows 100 \
  --fee-rate 0.001 --fill-slippage-bps 5

# Validate worksteal strategy:
python -m binance_worksteal.backtest --days 30

# Multi-period eval:
python -m pufferlib_market.evaluate_multiperiod \
  --checkpoint <path> \
  --data-path <bin> \
  --periods 1d,7d,30d --json
```

## Realism Parameters
- fee=10bps, margin=6.25% annual, fill_buffer=5bps, max_hold=6h
- validation_use_binary_fills=True
- fill_temperature=0.01
- decision_lag>=2
- Test at slippage 0/5/10/20 bps before deploying

## Runtime Audit (2026-04-09)

The recent "market was good but hybrid still lost" question is not explainable from the hybrid model alone because the Binance margin account is not isolated.

Active live Binance traders on this machine at audit time:

- `rl_trading_agent_binance/trade_binance_live.py --live --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD --execution-mode margin --leverage 0.5`
- `python -m binanceleveragesui.trade_margin_meta ... --doge-checkpoint ... --aave-checkpoint ... --max-leverage 2.30`
- `python -m binanceexp1.trade_binance_selector --symbols BTCUSD,ETHUSD,SOLUSD ...`
- `python -m binanceexp1.trade_binance_hourly --symbols SOLUSD ...`

That means the hybrid bot is sharing a live Binance account with other active strategies, including a margin strategy that explicitly trades `AAVEUSDT` and `DOGEUSDT`.

### What the hybrid cycle snapshots showed

Window audited:

- `2026-04-08T00:00:00+00:00` to `2026-04-08T22:00:00+00:00`

Command:

```bash
source .venv313/bin/activate
python -m src.binance_hybrid_runtime_audit \
  --launch-script deployments/binance-hybrid-spot/launch.sh \
  --trace-dir strategy_state/hybrid_trade_cycles \
  --start 2026-04-08T00:00:00+00:00 \
  --end 2026-04-08T22:00:00+00:00 \
  --output-json analysis/current_binance_prod_eval_20260409/runtime_audit_20260408.json
```

Result:

- `22` allocation snapshots
- `12` snapshots with unexpected borrowed quote despite a `0.5x` launch
- `AAVEUSD` flagged as oversized in `6` managed-position snapshots
- `AAVEUSD` flagged in `15` oversized managed-order buckets
- foreign symbol/order activity also present: `DOTUSD`, `XRPUSDT`

The most concrete bad state was:

- `2026-04-08T19:09:05Z`
- account equity about `$3078`
- borrowed quote about `$6265`
- `AAVE` position `79.21747868`
- planned AAVE sleeve only about `$231`

So the account was not simply "missing an up day." It was carrying a massively stale / foreign AAVE exposure that the hybrid loop then tried to unwind with large sell orders.

### Why the recent PnL looked wrong

- the local market data over the last completed 24 hourly bars was broadly positive:
  - `BTC +2.96%`
  - `ETH +4.46%`
  - `AAVE +4.18%`
  - `LINK +2.98%`
  - `SOL +1.65%`
  - `DOGE +1.10%`
- but the hybrid account state was not the clean equal-weight basket assumed by the replay
- the live account was being mutated by other bots and by leftover margin debt / working orders
- therefore the "market simulator says it should be up" vs "production equity is down" comparison was contaminated at the account layer, not just at the model layer

### Code support added

`src/binance_hybrid_runtime_audit.py` now flags:

- borrowed quote exposure beyond the launch leverage
- managed positions far larger than their planned sleeve
- managed orders far larger than their planned sleeve

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_runtime_audit.py
```

Result: `9 passed`

### Practical conclusion

Do not use current live PnL as a clean readout of the hybrid strategy until the Binance account is isolated from:

- `binanceleveragesui.trade_margin_meta`
- `binanceexp1.trade_binance_selector`
- `binanceexp1.trade_binance_hourly`

If we want simulator-vs-production comparisons to mean anything, the next production step is account/process isolation first, not another checkpoint swap.

### Live fail-safe added

The highest-impact code change after the contamination audit was to harden the live hybrid loop itself.

`rl_trading_agent_binance/trade_binance_live.py` now evaluates the live account before any context fetch, Gemini call, or order placement and blocks live trading with status `account_guard_blocked` when it sees:

- borrowed quote inconsistent with the configured leverage
- foreign positions or working orders outside the configured launch symbol set
- single-symbol live sleeves or working orders larger than the hybrid gross budget implied by `effective_leverage`

This matters because the prior failure mode was not just "bad forecast quality"; it was the hybrid loop continuing to trade on top of a contaminated shared margin account. The new guard turns that into an explicit unhealthy runtime state instead of letting the bot compound the mismatch.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_hybrid_allocation_prompt.py tests/test_binance_hybrid_prod_verify.py
```

Result: `57 passed`

### Deploy preflight added

There was still a clear gap after the live cycle guard: the deploy gate and autoresearch auto-deploy path could approve a candidate even while the machine was already running other Binance live writers.

That is now hardened with a process-isolation preflight:

- new audit module: `src/binance_live_process_audit.py`
- detects the Binance writer processes we actually observed on this machine:
  - `rl_trading_agent_binance/trade_binance_live.py`
  - `binanceleveragesui.trade_margin_meta`
  - `binanceexp1.trade_binance_selector`
  - `binanceexp1.trade_binance_hourly`
  - `binance_worksteal.trade_live`
- `deploy_candidate_checkpoint()` now fails closed before touching supervisor if the process set is not isolated
- `gate_deploy_candidate()` supports `require_process_isolation=True`
- `binance_autoresearch_forever.py` now uses that isolation requirement on the real auto-deploy path

This matters because the prior deploy logic could still say "candidate passes deploy gate" even when the live account was already contaminated by concurrent Binance writers. That was the highest-value remaining production gap after the runtime guard.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_deploy_gate.py tests/test_binance_autoresearch_forever.py tests/test_search_binance_prod_config_grid.py
```

Result: `171 passed`

### Live-process drift audit

The next concrete gap I observed on the machine was launch/process drift:

- `deployments/binance-hybrid-spot/launch.sh` points at `mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt`
- the actual running hybrid PID was still `1712424`
- that PID was running `a100_scaleup/robust_champion/best.pt`

That means "launch file says X" and "actual live process is X" were not equivalent, which makes both production debugging and automated deployment decisions less trustworthy.

Implemented:

- extracted a shared `trade_binance_live.py` command parser into `src/binance_hybrid_launch.py`
- added `src/binance_hybrid_process_audit.py` to compare the running hybrid process command against the launch config
- `scripts/binance_autoresearch_forever.py` now blocks real deploy attempts when:
  - Binance live-writer isolation fails
  - or the currently running hybrid process does not match the current launch script

This closes the specific failure mode where an automated rollout could proceed while the machine was already in a stale live state.

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_process_audit.py tests/test_binance_autoresearch_forever.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `167 passed`

Machine sanity check:

```bash
source .venv313/bin/activate
python - <<'PY'
from src.binance_hybrid_process_audit import audit_running_hybrid_process_matches_launch
print(audit_running_hybrid_process_matches_launch('deployments/binance-hybrid-spot/launch.sh').reason)
PY
```

Observed result: `running hybrid process does not match launch config: rl_checkpoint`

### Manual deploy preflight alignment

Review checkpoint follow-up:

- `scripts/deploy_crypto_model.sh` still did not enforce the same live-process preflight as `scripts/binance_autoresearch_forever.py`
- that meant a manual deployment could still restart into the exact conflict/drift state the autoresearch path would reject

Fixed:

- tightened `src/binance_live_process_audit.py` so it only classifies actual Python writer processes instead of matching raw substrings in arbitrary commands
- added the same live-process preflight to `scripts/deploy_crypto_model.sh`
  - blocks when multiple Binance live writers are active
  - blocks when the currently running hybrid process command does not match `launch.sh`

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py tests/test_deploy_crypto_model.py
```

Result: `39 passed`

### Production eval now audits machine state

The next important gap was that `scripts/evaluate_binance_hybrid_prod.py` still treated the "current production baseline" as if runtime snapshots were the whole truth. That missed two machine-level failures we had already observed directly:

- multiple Binance live writers on the same account
- running hybrid process checkpoint drift versus `launch.sh`

Implemented:

- added `src/binance_hybrid_machine_audit.py`
- `scripts/evaluate_binance_hybrid_prod.py` now records:
  - `current_machine_audit`
  - `current_machine_audit_issues`
  - `current_machine_health_issues`
- for the default production launch, those machine issues are merged into the existing manifest baseline issue fields, so deploy gating treats them as part of the production baseline honesty check
- `src/binance_deploy_gate.py` also explicitly rejects manifests that already admit machine drift/health issues

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_machine_audit.py tests/test_evaluate_binance_hybrid_prod.py tests/test_binance_deploy_gate.py
```

Result: `64 passed`

Real machine dry-run:

```bash
source .venv313/bin/activate
python scripts/evaluate_binance_hybrid_prod.py --dry-run
```

Now reports both layers:

- runtime contamination from borrowed quote / foreign symbol activity
- machine contamination from `hourly`, `meta_margin`, `selector`, and `worksteal_daily`
- running hybrid process drift: `robust_champion` vs launch `robust_reg_tp005_dd002`

### Post-deploy verification now checks live machine state too

Another gap was that `src.binance_hybrid_prod_verify` only trusted:

- launch file contents
- supervisor log startup text
- recent live cycle snapshots

That was not enough for this machine. We had already observed that a restart can still leave the wrong hybrid process or extra Binance writers active, so deploy verification needed to check the actual process set after restart, not just the artifacts around it.

Implemented:

- `src.binance_hybrid_prod_verify.py` now supports `--require-machine-state-match`
- when enabled, deploy verification also runs the machine audit:
  - process isolation via `src.binance_live_process_audit.py`
  - running hybrid process vs `launch.sh` via `src.binance_hybrid_process_audit.py`
- `scripts/deploy_crypto_model.sh` now enables that flag for both RL deploys and `--remove-rl` deploys

This closes the remaining trust gap where preflight could pass but post-restart verification could still miss:

- extra Binance writers appearing after restart
- the wrong hybrid checkpoint continuing to run after restart

Validation:

```bash
source .venv313/bin/activate
pytest -q tests/test_binance_hybrid_prod_verify.py tests/test_deploy_crypto_model.py
pytest -q tests/test_binance_live_process_audit.py tests/test_binance_hybrid_process_audit.py
```

Results:

- `54 passed`
- `8 passed`

- refreshed open-order dedupe so live cycles replace stale BUY/SELL working orders when price drifts materially from the current target instead of only checking remaining size
- aligned hybrid rebalance sells with the same `_minimum_live_exit_price` floor used by the exit-coverage pass, avoiding duplicate flatten / coverage sells in one cycle

- account guard now inspects all nonzero margin assets, not just tracked strategy symbols; foreign holdings like `ZEC` now surface in snapshots and block live trading
- account-guard-blocked cycles now cancel tracked entry buys but still refresh/place closing sell coverage for tracked longs, so protective exits stay live even while new entries are frozen
- live cycle snapshots now record a `reentry_plan` / `next_entry_price` per symbol so every protected open long carries an explicit follow-on buy plan or an explicit `stay_flat`/`deferred_account_guard` state
- `scripts/eval_100d.py` now tolerates CUDA OOM in the pufferlib slippage sweep by retrying inference on CPU instead of dying mid-audit
- live exit pricing now reconstructs the active entry lot from recent fills instead of using the most recent buy blindly; snapshots carry `position_entry_time`, `hold_hours`, `exit_basis`, and the real live basis when it can be inferred
- account-guard-blocked cycles now cancel every non-cover BUY order and place closing sell coverage for any meaningful long on the account, including foreign holdings like `ZEC`/`XRP`, instead of only tracked strategy symbols
- `BINANCE_HYBRID_MAX_HOLD_HOURS` is now a configurable but disabled-by-default backstop; when enabled and a live long exceeds that age, the runner prices its closing order from the exchange midpoint and records `exit_basis=max_hold_midpoint` in the snapshot

## 2026-04-30 NVIDIA/CVaR and Symbol-Filter Research Pass

No production deployment.

New artifacts:
- `analysis/binance33_linear_cvar_pack_rand0045_targeted_20260430.csv`
- `analysis/binance33_linear_symbol_filter_rand0045_greedy_20260430.csv`
- `analysis/binance_head_to_head_20260430_after_nvidia_cvar_symbolfilter.csv`

Findings:
- NVIDIA-style CVaR portfolio packing around the current best linear short rule was not competitive. Best targeted
  validation row was only about `+5.18%/mo` with `2/45` negative windows.
- Train-history-only symbol filtering improved the bad historical regime but not enough: after excluding
  `AAVEUSD,BTCUSD,DOGEUSD,ETHUSD,SHIBUSD,SOLUSD`, train-history still had `61/162` negative windows and p90 DD
  around `80%`.
- Recent validation still contains the known borderline rows:
  - raw linear rule at leverage `1.305`: `+30.0387%/mo`, p10 `+19.5390%/mo`, p90 DD `19.9609%`, `0/45` negative windows.
  - `mkt_ret20gt-0.1` gate at leverage `1.305`: `+30.0539%/mo`, p10 `+17.8341%/mo`, p90 DD `15.358%`, `0/45`
    negative windows.

Production decision: hold. These are recent-validation candidates only; broader train-history robustness remains far
below the production bar, and the live Binance machine state previously showed writer/process contamination that must
stay resolved before any live flip.

## 2026-05-01 Meta-Anneal / XGBoost Blend Pass

No production deployment.

New artifacts:
- `analysis/binance33_meta_anneal_linear_hand_20260501.csv`
- `analysis/binance33_meta_anneal_single_linear_hand_20260501.csv`
- `analysis/binance33_meta_anneal_xgb_linear_20260501.csv`
- `analysis/binance33_meta_anneal_recent_tail_linear_hand_20260501.csv`
- `analysis/binance33_meta_channel_gate_grid_20260501.csv`
- `analysis/binance_head_to_head_20260501_after_meta_anneal.csv`

Findings:
- Temperature-cooled meta blending over linear and handcrafted channels mostly converged to low-return defensive books.
  Best full-history validation portfolio row was only about `+4.61%/mo`.
- XGBoost-channel blending was smoother but far below target: best exact single replay was about `+4.89%/mo` with tiny
  DD, but too few trades and nowhere near the PnL gate.
- Recent-tail annealing raised PnL by rediscovering 20-day winner shorts, but exact validation had p90 DD around `74%`.
- Focused exact channel/gate grid found no improvement over the known candidate. Best blend-normalized channel row was
  `+26.69%/mo`, p10 `+16.46%/mo`, p90 DD `19.96%`, below the 30%/mo target.

Production decision: hold. The meta/XGB blend pass did not produce a candidate better than the existing recent-only
rand0045/regime rows, and none solved old-history robustness.
