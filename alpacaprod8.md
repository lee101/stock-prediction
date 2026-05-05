# 2026-05-04 XGB Stock Optimization + Codex Monitor Test

## Scope

- Trading target is Alpaca stocks, not Binance.
- Production deploy gate remains median monthly return >= 27% on honest
  heldout/stress simulation with realistic fills and drawdown/tail controls.
- No live writer changes or order placement are allowed from this experiment
  unless a candidate clearly clears the gate.
- Existing dirty file `manifest_stocks_drytest_001.json` is unrelated and left
  untouched.

## 2026-05-04 Run Log

### Initial state

- Starting from commit `b7f9d01b` (`Guard aggressive XGB live buy sizing`).
- Active production remains documented in `alpacaprod.md`; no strategy
  promotion has happened in this run.
- Planned experiment tracks:
  - Stress-first hyperparameter sweeps around the top-1/top-N XGB stock family.
  - Correlation-aware and work-stealing allocation checks.
  - Codex scheduled production monitor wrapper smoke test.

### Sweep 1 partial: packing + work-stealing allocation

- Command family:
  `xgbnew.sweep_ensemble_grid` on `stocks_wide_1000_v1`,
  `heldout2025h2_xgb` 5-seed ensemble, OOS `2025-07-01 -> 2026-04-17`,
  `stress36x`, 10 bps fill buffer, hold-through, top-N `{1,2,3}`,
  leverage `{1.5,2.0,2.25}`, min score `{0.55,0.60}`, allocation modes
  `{equal,score_norm,softmax,worksteal}`, overnight cap `2.0`.
- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_pack_worksteal_20260504/sweep_20260504_115149.partial.json`
- Stopped after 200 completed cells because all tested cells had already
  fail-fast pruned; no cell survived the stress gate.
- Best partial median cell was `top_n=3`, `leverage=1.5`, `min_score=0.60`,
  `allocation=worksteal`, `max_vol_20d=1.5`: median `+11.22%/mo`,
  p10 `-10.65%`, worst monthly `-18.16%`, worst DD `25.95%`,
  failed after one negative window.
- Interpretation: naive packing is not enough. Work-stealing improves top-2/3
  versus equal sizing, but the tail window is still far below deployable.

### Sweep 2: correlation-aware work-stealing packing

- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_corr_worksteal_small_20260504/sweep_20260504_122103.json`
- Grid: top-N `{2,3}`, leverage `{1.5,1.75}`, work-stealing allocation,
  skew-gated regime (`regime_cs_skew_min=0.5`), `regime_cs_iqr_max=0.06`,
  and trailing correlation filters (`corr_window_days in {0,40}`,
  `corr_max_signed in {0.50,0.75}`).
- Result: `0/16` cells survived fail-fast.
- Best median cell: `top_n=3`, `leverage=1.75`, median `+10.37%/mo`,
  p10 `-5.57%` without the correlation filter and p10 `-3.51%` with
  40-day correlation filtering; worst DD about `14.2%`.
- Interpretation: correlation filtering reduced the bad tail on the packed
  top-3 case, but median remains far below the 27% target and still has a
  negative window.

### Sweep 3: opportunistic work-stealing entries

- Artifact:
  `analysis/xgbnew_daily/alpacaprod8_opportunistic_20260504/sweep_20260504_123042.json`
- Grid: top-N `{1,2}`, leverage `{1.75,2.0}`, watch-N `{3,5,10}`,
  entry discount `{20,40}` bps below open, work-stealing allocation,
  skew/IQR regime filters, `stress36x`, 10 bps fill buffer.
- Result: `0/24` cells survived fail-fast.
- Best median cell: `top_n=1`, `leverage=2.0`, watch-N `3/5/10`,
  20 bps entry discount, median `+16.90%/mo`, p10 `-5.27%`,
  worst monthly `-11.66%`, worst DD `17.40%`.
- Interpretation: opportunistic limit-style entry improved the median versus
  correlation-packed work-stealing while keeping drawdown reasonable, but it
  still failed the negative-window gate and remains below the deploy target.

### Deploy decision after sweeps

- No strategy candidate cleared the production gate.
- No XGB deployment or threshold/leverage promotion from these sweeps.
- Best direction for the next round is the opportunistic top-1 path, but it
  needs a tail-risk/regime fix before it is production material.

### 2026-05-04 continuation: heldout tail-regime search

- Goal: find a stock-only candidate that improves the high-median top-1 XGB
  family without hiding the September 2025 onward crash tail.
- Running/started artifacts:
  - `analysis/xgbnew_daily/alpacaprod8_tailregime_grid_20260504/`
  - `analysis/xgbnew_daily/alpacaprod8_tailregime_nofail_refine_20260504/`
  - `analysis/xgbnew_daily/alpacaprod8_mlp_muon_top1_20260504/`
  - `analysis/xgbnew_daily/alpacaprod8_baseline_attr_20260504/`
- Baseline attribution on the high-median top-1 XGB setup shows the core
  issue clearly: July/August were strongly positive, but the later four
  30-day windows were all negative. Worst contributors included `IONQ`,
  `IREN`, `QUBT`, `LITE`, `CRDO`, and `DDOG`, which points more toward a
  high-beta/regime transfer failure than a pure leverage/allocation bug.
- First tail-regime partial: the cross-sectional `ret_5d` IQR gate near
  `0.05` is the strongest live-realizable filter seen so far. Partial cells
  reached worst-cell median around `+38%` to `+40%/mo`, but still had one
  negative heldout window and/or drawdown above the deploy gate.
- MLP/Muon heldout partial is not competitive so far: low drawdown but
  negative/flat median. This does not replace the XGB selector.
- Completed lower-leverage no-fail refinement:
  `analysis/xgbnew_daily/alpacaprod8_tailregime_nofail_refine_20260504/sweep_20260504_140041.json`
  (`80` cells, `0` production passes). Once the candidate is evaluated across
  all six windows and the harsher `stress36x` + 10 bps fill-buffer cell, the
  attractive partial disappears. Best safety-ranked worst cell was only
  `+1.43%/mo` median, p10 `-5.45%`, DD `19.74%`, and `3/6` negative windows
  at `1.15x`. Higher leverage raised median slightly but increased p10 loss
  and/or drawdown, so this does not solve the tail.
- The broad XGB and MLP/Muon exploratory sweeps were stopped after useful
  partials to prioritize the completed no-fail XGB replay. Their partials are
  retained as rejected research artifacts, not deployment evidence.
- Tested an outside-the-box regime flip: when the cross-sectional regime gate
  fails, short the model's top-ranked high-beta favorite instead of sitting
  out. Added `BacktestConfig.regime_failed_top_side` for this experimental
  simulator path, with default `0` preserving existing behavior.
- Core stress check:
  `analysis/xgbnew_daily/alpacaprod8_regime_flip_topshort_core_20260504/core_flip_20260504_144734.json`.
  Result: `0/24` production passes under `stress36x` + 10 bps fill buffer.
  Sitting out the bad regime was best at every tested leverage; shorting the
  top-ranked name was worse at every leverage (`0.5x` already had median
  `-2.98%/mo`, p10 `-8.48%`, `5/6` negative windows; higher leverage worsened
  sharply). Conclusion: the top-ranked bad-regime names are volatile/noisy, but
  not reliably shortable by this simple flip rule.
- Probed a live-known extension filter using `price_vs_52w_high` and
  `price_vs_52w_range` to avoid already-extended high-beta names. The early
  stress cells were clearly non-competitive and the broad probe was stopped
  without promotion: examples at `1.5x`, `min_score=0.55` improved raw median
  from `-20.0%/mo` to roughly `-6%` to `-8%`, but p10 stayed around
  `-23%` to `-26%` and DD stayed `45%+`. This is a weak risk trim, not a
  deployable edge.
- Added an opt-in realized-loss cooldown to the simulator:
  `BacktestConfig.loss_cooldown_days` and
  `BacktestConfig.loss_cooldown_trigger_pct`. It is leak-free because it only
  reacts to completed daily PnL and clears hold-through state while paused.
  Diagnostic artifact:
  `analysis/xgbnew_daily/alpacaprod8_loss_cooldown_diag_20260504/cooldown_diag_20260504_151354.json`.
  Result: `0/32` production passes. Cooldown reduces damage but also cuts
  activity; best stress cell was `iqr_skew_opp3`, `2.0x`, `5`-day cooldown
  after a `4%` loss, median `+4.42%/mo`, p10 `-9.83%`, DD `24.16%`,
  `2/6` negative windows, median active days `21.7%`. Conclusion: useful as a
  safety primitive, but it cannot turn the current selector into a production
  candidate.
- Added an opt-in symbol-specific realized-loss cooldown:
  `BacktestConfig.symbol_loss_cooldown_days` and
  `BacktestConfig.symbol_loss_cooldown_trigger_pct`. Unlike portfolio
  cooldown, this only removes the losing symbol from the pick pool for N
  sessions and lets the model rotate to the next candidate. This directly
  targets repeated high-beta loser loops without forcing the account to cash.
- Before trusting that probe, reran the official one-cell heldout identity
  check on the previously promising top-1 family:
  `analysis/xgbnew_daily/alpacaprod8_identity_check_20260504/sweep_20260504_153335.json`.
  With current code/data, the exact `heldout2025h2_xgb`, top-1, `2.0x`,
  `min_score=0.55`, `stress36x`, 5 bps fill, uncertainty penalty `0.5`
  replayed as median `-25.65%/mo`, p10 `-40.02%`, DD `54.62%`, `4/6`
  negative windows. The older attractive heldout row had `n_windows=3` even
  though `expected_n_windows=6`; it was a fail-fast prefix, not full evidence.
  This means the older positive heldout artifact is not a deployment basis.
- Symbol cooldown diagnostic artifact:
  `analysis/xgbnew_daily/alpacaprod8_symbol_cooldown_diag_20260504/symbol_cooldown_diag_20260504_153237.json`.
  Result: `0/14` production passes on the worst-fill stress diagnostic. Best
  cell was still median `-22.15%/mo`, p10 `-33.13%`, DD `52.96%`, `4/6`
  negative windows. Conclusion: symbol cooldown is a reasonable research
  primitive, but it does not rescue the stale top-1 XGB candidate under the
  current replay.
- Ran a fast breadth-regime diagnostic to see whether leak-free day-level
  market breadth features separate the two good early windows from the bad
  later windows:
  `analysis/xgbnew_daily/alpacaprod8_fast_breadth_gate_diag_20260504/fast_breadth_gate_20260504_154857.json`.
  This used cached top-1 churn-day returns and OOS-derived thresholds, so it
  is explicitly research-only and not deploy evidence. Result: `0` useful
  passes. The best diagnostic gate was
  `ret20_pos_frac>=0.59701 & ret20_median>=0.023818`, median only
  `+6.15%/mo`, p10 `-17.69%`, DD `34.23%`, `2/6` negative windows. A few
  very sparse gates had zero negative windows but traded under `10%` of days
  and produced near-flat medians. Conclusion: simple breadth thresholding does
  not create the missing edge; it mostly avoids trading.
- Added a default-off online symbol realized-PnL rank overlay to the simulator:
  `BacktestConfig.symbol_pnl_half_life_days`,
  `BacktestConfig.symbol_loss_score_penalty`, and
  `BacktestConfig.symbol_pnl_score_cap`. The idea follows the same broad
  direction as recent decision-focused/adaptive portfolio work: do not only
  predict single-name returns independently; let the allocation policy react
  to its own realized failures. This implementation is intentionally
  live-realizable: it only uses completed trade PnL from earlier sessions,
  decays it by trading days, and subtracts a capped score penalty before the
  next ranking pass. Defaults preserve legacy behavior.
- Unit/regression result: focused simulator tests cover default identity and
  rotation after realized symbol loss. The first full heldout diagnostic grid
  was stopped because it was too broad for an interactive pass; after `100`
  full cells on the same `heldout2025h2_xgb` / `stocks_top200_v1` /
  `stress36x` / 5 bps fill / hold-through setup, the best observed partial
  was still negative: median `-17.41%/mo`, p10 `-27.19%`, DD `47.22%`,
  `4/6` negative windows. Conclusion: the online symbol-memory overlay is a
  useful simulator primitive, but early evidence says it does not rescue the
  stale top-1 family either.
- Added another default-off inference primitive:
  `BacktestConfig.min_top_score_gap`. It requires the marginal selected
  top-ranked score to beat the next unselected score by a configurable gap
  before allowing the long/top-ranked pack to trade. This is live-realizable
  because it only uses same-open model scores, and defaults to `0.0` for
  legacy identity. Rationale: if the model's top pick is barely separated
  from the rest of the day, that looks like a low-confidence cross-sectional
  ranking where aggressive leverage is not justified.
- Regression result: focused simulator tests verify that a crowded low-gap day
  is skipped while a clear top-pick day still trades. The first broad heldout
  score-gap diagnostic was stopped before artifact write because full
  top-200/stress simulation was too slow for an interactive pass. This knob is
  therefore implemented and tested as a research primitive, but it has no
  deployment evidence yet.
- Wired `min_top_score_gap` into the normal `xgbnew.sweep_ensemble_grid`
  checkpoint/resume path as `--min-top-score-gap-grid`, including stable cell
  keys, output rows, and sweep tests. This matters because future full
  top-200 runs can now checkpoint instead of losing work when a broad
  diagnostic is interrupted.
- Fast score-gap diagnostic artifact:
  `analysis/xgbnew_daily/alpacaprod8_fast_score_gap_diag_20260504/fast_score_gap_20260504_164333.json`.
  This is only a top-1 churn approximation and sweeps OOS-derived thresholds,
  so it is not deploy evidence. Result: `0/160` diagnostic passes. High gaps
  mostly killed trading; requiring at least `10%` median active days, the best
  cell was only median `-1.62%/mo`, p10 `-8.55%`, DD `13.57%`, `4/6`
  negative windows. Requiring at least `30%` active days made the best cell
  median `-7.59%/mo`, p10 `-18.10%`, DD `31.08%`, `5/6` negative windows.
  Conclusion: score separation is a sensible confidence feature, but by itself
  it does not uncover a deployable edge on the stale top-1 family.
- Deploy decision remains unchanged: no redeploy unless a completed worst-cell
  result clears the median/negative-window/drawdown gate.

### Monitoring prompt fix

- `monitoring/codex_prod_check_prompt.md` had stale assumptions from the
  older XGB-live layout. It now defers to `alpacaprod.md` and explicitly
  accepts the current manual `trading-server` + `daily-rl-trader` production
  mode, with XGB intentionally inert.
- Upgraded local Codex CLI from `0.56.0` to `0.128.0` so `gpt-5.5` is
  accepted.
- Updated `monitoring/codex_prod_check.sh` and `monitoring/monitor_agent.sh`
  for the current CLI flag:
  `--dangerously-bypass-approvals-and-sandbox` instead of obsolete `--yolo3`.
- Re-authenticated headless Codex with the machine `OPENAI_API_KEY` because
  the prior ChatGPT session returned 401 from the new CLI.
- Made both monitor wrappers source `~/.secretbashrc` under `set +e` so the
  profile does not break strict-mode wrapper execution.

### Real Codex monitor test

- Command: `CODEX_BIN=/home/administrator/.bun/bin/codex monitoring/codex_prod_check.sh`
- Successful wrapper artifact:
  `monitoring/logs/codex_prod_20260504T124713Z.log`
- Machine-readable current status:
  `monitoring/logs/codex_current.log` now reports `status=OK rc=0`.
- The scheduled Codex agent appended:
  `monitoring/logs/codex_prod_20260504.log`
- Agent verdict: `YELLOW`.
- Production checks from the agent:
  - `trading-server` pid `4130990` owns the singleton lock as
    `alpaca_wrapper_4130990`.
  - `daily-rl-trader` pid `4131404` is attached through `127.0.0.1:8050`.
  - Alpaca account active, `trading_blocked=false`, equity/cash `$19,799.53`,
    buying power `$39,599.06`.
  - `0` open orders and `0` material positions.
  - No services were started/stopped, no orders submitted, and no
    model/config/leverage/threshold changes made.
- Follow-up health check:
  `.venv313/bin/python monitoring/health_check.py --json` exits `0`;
  scheduled-audits is now OK for Codex. Remaining warnings are optional
  services and disk (`/` 94%, `/nvme0n1-disk` 86%).
