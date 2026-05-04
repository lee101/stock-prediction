# XGB Best Research Ledger

This file tracks promising XGB daily strategy variants and failed directions.
It is not the production ledger. Production state stays in `alpacaprod.md`,
and nothing here is deployable unless it clears the current project target:
median monthly return >= 27% on the worst realistic slippage/fee cell, with
acceptable drawdown, binary fills, and decision lag >= 2.

## 2026-05-02 - long/short packing and prod inactivity replay

### Best non-production research candidate so far

Artifact:
`analysis/xgbnew_daily/longshort_scaled_top200_100d_20260502/sweep_20260502_093507.json`

Best prod10bps cell from the 100d top-200 sweep:

- `top_n=1`
- `short_n=1`
- `max_short_score=0.45`
- `short_allocation_scale=0.25`
- `opportunistic_watch_n=5`
- `opportunistic_entry_discount_bps=30`
- `allocation_mode=softmax`
- median monthly return: `+17.74%`
- p10 monthly return: `+10.19%`
- worst drawdown: `19.65%`
- negative windows: `0/5`

Stress behavior was not production-grade:

- best stress36x median monthly return: `+6.79%`
- stress worst drawdown: `20.74%`

Decision: research-useful, not production. The small conditional short sleeve
improved the 100d shape, but it did not clear the 27% target or the stress cell.

### Current production stock config likely sits in cash

Artifact:
`analysis/xgbnew_daily/prod_replay_10mo_20260502/sweep_20260502_212350.json`

Replay config matched the live-style stock side:

- `top_n=1`
- `min_score=0.85`
- `leverage=2.0`
- `hold_through=True`
- `prod10bps` and `stress36x`
- 2025-07-01 through 2026-04-30, 30d windows, 15d stride

Result:

- median monthly return: `0.00%`
- p10 monthly return: `0.00%`
- worst drawdown: `0.00%`
- active day pct: `0.00%`
- negative windows: `0/12`

Interpretation: the stock gate is too conservative. The zero drawdown is not
a sign of safety; it means the stock sleeve did not trade in this replay.

### Aggressive packing probes

Partial artifacts:

- `analysis/xgbnew_daily/aggressive_pack_10mo_20260502/sweep_20260502_212956.partial.json`
- `analysis/xgbnew_daily/balanced_pack_10mo_20260502/sweep_20260502_213604.partial.json`
- `analysis/xgbnew_daily/long_pack_10mo_20260502/sweep_20260502_214247.partial.json`

Findings:

- Top-1 long-only has the current edge. At 2.0x leverage and 5 bps fill buffer,
  early-window median reached `+25.11%/mo` with p10 `+20.86%`, but breached the
  30% drawdown fail-fast after 3 windows.
- At 2.25x leverage, top-1 long-only reached `+27.78%/mo` with p10 `+22.81%`,
  but also breached the 30% drawdown fail-fast after 3 windows.
- Opportunistic top-1 entries can produce very high median values
  (`+47.99%/mo` seen in a partial prod10bps cell), but p10 was deeply negative
  and drawdown breached quickly. This is high variance, not yet robust.
- Forced long/short balance is currently bad. Best non-pruned partial cell with
  `top_n=2`, `short_n=1` was only `+2.87%/mo`, with negative p10. The bottom
  score is not reliable enough to short every day.
- Packing 2 or 3 long names diluted the signal sharply in the longer replay.
  Top-2/top-3 cells were weak or negative under the same filters.

Working conclusion:

- Reduce or replace the production `min_score=0.85` gate; it prevents stock
  trading.
- Keep the daily book concentrated around top-1 until the model can prove
  useful top-2/top-3 ranking.
- Use shorts only as a small conditional hedge for now, not as forced daily
  balance.
- Next high-value work is drawdown control for the top-1 edge: regime filters,
  per-symbol risk caps, stale-signal avoidance, and better validation on true
  unseen windows before any live deployment.

## 2026-05-02 - top-1 attribution and universe-selection probes

### Simulator fix found during attribution

The long/short side-aware hold-through state stores picks as `(symbol, side)`
pairs. One no-new-pick continuation path still treated that state as plain
symbols, so a held position could fail to carry forward when the next day had
no fresh pick. Fixed in `xgbnew/backtest.py` and covered by
`test_hold_through_carries_side_aware_pick_when_next_day_has_no_pick`.

### Full-window attribution of the promising top-1 setup

Artifact:
`analysis/xgbnew_daily/attribution_top1_225x_10mo_20260502.json`

Config:

- `top_n=1`
- `leverage=2.25`
- `min_score=0.0`
- `hold_through=True`
- `prod10bps`
- `fill_buffer_bps=5`
- `inference_min_dolvol=50M`
- `inference_min_vol=0.12`
- 2025-07-01 through 2026-04-30, 30d windows, 30d stride

Result after full six-window replay:

- median monthly return: `+19.97%`
- p10 monthly return: `-23.38%`
- worst drawdown: `62.93%`
- negative windows: `2/6`
- active day pct: `100%`

Interpretation: the earlier `+27.78%/mo` partial was a useful lead but not
stable across the full longer replay. The raw top-1 signal is real but has
unacceptable crash tails.

Top attribution contributors:

- `SNPS`: `+$5.0k`, 4 trades, 4 wins
- `CRWV`: `+$3.2k`, 25 trades, but worst trade `-37.12%`
- `RKLB`: `+$3.2k`, 6 trades, 6 wins
- `SNDK`: `+$2.1k`, 3 trades, 3 wins
- `TSLA`: `+$1.7k`, 5 trades, 5 wins

Worst attribution contributors:

- `APP`: `-$2.3k`, worst trade `-44.31%`, intraday DD contribution `45.22%`
- `QUBT`: `-$1.6k`, worst trade `-17.57%`
- `RDDT`: `-$1.5k`, worst trade `-14.26%`
- `SMCI`: `-$0.8k`, worst trade `-12.13%`
- `ALAB`: `-$0.8k`, worst trade `-12.68%`

### Hyperparameter probes around the top-1 edge

Partial artifact:
`analysis/xgbnew_daily/top1_volcap_sweep_10mo_20260502/sweep_20260502_225110.partial.json`

Max-vol caps reduced blowup risk but also removed most of the edge:

- `max_vol_20d=0.8`, `1.5x`, prod10bps, 5 bps buffer: median `+4.07%`, p10
  `+1.41%`, `0/6` negative windows.
- `max_vol_20d=1.0`, `1.5x`, prod10bps, 5 bps buffer: median `+11.12%`, p10
  `+0.59%`, `0/6` negative windows.

This is safer but too low-return. Looser caps reintroduced large negative
tails; tighter caps under-traded the useful high-vol winners.

Follow-up small risk sweep artifact:
`analysis/xgbnew_daily/top1_small_risk_sweep_20260502/sweep_20260502_230310.json`

Best risk-clean cells combined `max_vol_20d=0.8` with per-pick inverse-vol
sizing (`inv_vol_target_ann=0.25`):

- `1.25x`: median `+3.57%`, p10 `+1.63%`, DD `9.86%`, `0/6` negative windows.
- `1.50x`: median `+4.18%`, p10 `+1.82%`, DD `11.82%`, `0/6` negative windows.
- `2.00x`: median `+5.35%`, p10 `+2.09%`, DD `15.71%`, `0/6` negative windows.

This confirms the risk overlay can make the curve smooth, but it also cuts
the target return by too much. It is a useful fallback/safety sleeve idea, not
the desired aggressive champion.

Score gates were not a clear fix in the early cells of
`analysis/xgbnew_daily/top1_scoregate_sweep_10mo_20260502/`: `min_score=0.55`
still failed the 30% DD gate and reduced median. The current score scale is
not calibrated enough for a simple threshold to separate safe vs unsafe days.

### Universe-selection probes

First-half attribution artifact:
`analysis/xgbnew_daily/attribution_top1_2x_trainhalf_20260502.json`

First-half positive-only universe:
`symbol_lists/xgb_top1_positive_trainhalf_20260502.txt`

Later-half validation artifact:
`analysis/xgbnew_daily/positive_universe_validhalf_20260502/sweep_20260502_225951.json`

Result: failed badly. The first-half positive-only 16-symbol universe was
negative across the later-half validation. Best prod10bps cell shown was
`-17.63%/mo` at 1.5x, 5 bps buffer.

First-half loser-block universe:
`symbol_lists/xgb_top200_minus_trainhalf_losers_20260502.txt`

Later-half validation artifact:
`analysis/xgbnew_daily/minus_losers_validhalf_20260502/sweep_20260502_230257.json`

Result: also failed. Removing first-half losers from the broad top-200 universe
left 176 symbols but later-half validation remained negative. Best prod10bps
cell shown was `-6.24%/mo` at 1.5x, 5 bps buffer, with DD still over 30%.

Working conclusion:

- Static universe cherry-picking from a short attribution window is not stable.
- The production universe should be selected by a rolling/online rule, not by
  fixed winners/losers from the latest few months.
- The next promising direction is a learned or rule-based daily risk overlay
  that detects tail days before entry: recent gap/ATR shock, earnings/news
  proximity if available, abnormal premarket move, and rolling per-symbol
  post-signal loss state. Simple vol cap, score gate, and static symbol lists
  did not solve it.

## 2026-05-03 - BitbankGo-style worksteal allocation

Relevant BitbankGo learnings:

- `../bitbankgo/webapp/handlers/trading_bot.go` uses realistic limit fills:
  buy fills only when the next candle low penetrates the buy limit by the fill
  buffer; sell/short analog uses high penetration.
- Its mixed-pair allocator hard-codes the two-candidate case to `150% + 50%`
  gross allocation, matching the intended work-stealing behavior where the
  first executable idea gets most of the account and the second gets the
  residual gross exposure.
- Candidate scoring penalizes drawdown directly (`pnl - dd + sortino - trade
  penalty` style), which is useful for frontier ranking but did not by itself
  solve the stock XGB tail.

Implemented in the stock simulator:

- Added `allocation_mode="worksteal"` in `xgbnew/backtest.py`.
- With two filled trades it weights the first fill `75%` and the second fill
  `25%`. At `2x` leverage that is equivalent to `150% + 50%` gross exposure.
- With more than two fills the first still gets `75%` and the remaining fills
  split `25%`.
- Exposed the mode through `xgbnew/sweep_ensemble_grid.py`.
- Covered by `test_worksteal_allocation_front_loads_first_filled_trade` and
  the allocation-grid sweep test.

Main opportunistic sweep artifact:
`analysis/xgbnew_daily/worksteal_alloc_opp_only_10mo_20260503/sweep_20260503_030953.json`

Config family:

- `top_n=2`
- `opportunistic_watch_n in {4,8}`
- `opportunistic_entry_discount_bps in {20,30,50}`
- `allocation_mode in {equal,worksteal}`
- `leverage in {1.75,2.0}`
- `prod10bps`, fill buffer `{5,10}`
- 2025-07-01 through 2026-04-30, 30d windows, 30d stride

Result:

- Worksteal consistently lifted median return versus equal allocation in the
  same opportunistic cells.
- Best median cell in this sweep: `2.0x`, watch `4`, discount `30bps`, 5 bps
  fill buffer, worksteal allocation: median `+19.48%/mo`.
- It still failed badly on tail risk: p10 `-50.20%`, worst DD `52.53%`,
  `1/4` negative windows before fail-fast stopped it.

Risk-overlay follow-up artifact:
`analysis/xgbnew_daily/worksteal_risk_overlay_10mo_20260503/sweep_20260503_031442.json`

Config family:

- Same best worksteal/opportunistic shape: `top_n=2`, watch `4`, discount
  `30bps`, 5 bps fill buffer.
- Swept `leverage in {2.0,2.25}`, `inference_max_vol_20d in {0,0.8,1.0}`,
  and `inv_vol_target_ann in {0,0.20,0.25}`.

Result:

- Per-pick inverse-vol sizing made the curve much safer but too low-return:
  best non-fail-fast cells were around `+5%` to `+6.5%/mo`, p10 near `-3%`,
  DD roughly `18%` to `24%`.
- Raw max-vol caps sometimes had better p10, but fail-fast stopped them for
  DD around `35%+` or multiple negative windows.

Working conclusion:

- Worksteal allocation is a good simulator/control primitive and improves the
  median on the aggressive opportunity style.
- It does not fix the tail; it concentrates the portfolio into the same regime
  that causes the crash.
- Production should not move to this candidate.
- Next promising work is not more static sizing; it is a tail-day detector or
  fast intraday/5m simulator that can model chronological fill order and abort
  the second fill/rotation when the first fill is already moving against us.

## 2026-05-03 - Correlation-aware packing probe

Motivation:

- The crash tail can come from "different" symbols that are really the same
  market bet. This is especially bad with 2x worksteal allocation because the
  second fill can silently add exposure to the same factor.
- Older repo code had analogous correlation filters in
  `unified_orchestrator/backtest_improved.py`; no local `KuOpt` path was found
  by name in this workspace.

Implemented:

- Added leak-free trailing correlation support to `xgbnew/backtest.py`.
- `BacktestConfig.corr_window_days > 0` builds per-day trailing return
  correlation matrices using lagged `ret_1d` when available, otherwise shifted
  close-to-close returns.
- `corr_max_signed < 1.0` skips candidates whose signed correlation to any
  already selected same-day pick exceeds the threshold.
- Signed correlation is `side_i * side_j * corr(i,j)`, so it blocks long/long
  and short/short positive-correlation duplication, and also long/short pairs
  with strong negative correlation where both legs can lose together.
- Exposed the knobs in `xgbnew/sweep_ensemble_grid.py` as:
  `--corr-window-days-grid`, `--corr-min-periods`, and
  `--corr-max-signed-grid`.
- Added tests for same-side duplicate skipping and sweep grid plumbing.

Top-2 worksteal correlation sweep:
`analysis/xgbnew_daily/worksteal_corr_gate_10mo_20260503/sweep_20260503_035823.json`

Config family:

- `top_n=2`, `short_n=0`, worksteal allocation.
- `leverage in {2.0,2.25}`.
- Opportunistic watch `4`, entry discount `30bps`, 5 bps fill buffer.
- `corr_window_days in {0,20,60}`, `corr_max_signed in {1.0,0.8,0.6,0.4}`.
- `prod10bps`, 2025-07-01 through 2026-04-30, 30d windows.

Result:

- Correlation gating changed the fills and helped the 2x tail somewhat.
- Best 2x tail cell was `corr_window_days=60`, `corr_max_signed=0.4`:
  median `+20.75%/mo`, p10 `-38.74%`, worst DD `35.11%`.
- Baseline disabled correlation cell was median `+19.48%/mo`, p10 `-50.20%`,
  worst DD `52.53%`.
- This is a real improvement in drawdown, but it still fails the target and
  was fail-fast pruned.
- 2.25x remained too unstable: best-looking median cells around `+23.37%/mo`
  still had p10 near `-51%` and DD just over `35%`.

Top-3 correlation/allocator sweep:
`analysis/xgbnew_daily/top3_corr_pack_10mo_20260503/sweep_20260503_040236.json`

Result:

- Top-3 is rejected for now.
- Equal and score-normalized top-3 allocations went negative or nearly flat
  across multiple settings.
- Worksteal top-3 had low median (`~7%` to `10%/mo`) and very bad p10
  (`-48%` to `-57%`), worse than top-2.

Working conclusion:

- Correlation-aware packing is worth keeping as a simulator/prod knob because
  it reduces redundant same-factor exposure and gives a measurable DD
  improvement in the top-2 worksteal family.
- It is not sufficient as the main solution. The current high-return edge
  remains concentrated in a few fragile regimes; simply swapping correlated
  picks does not detect those regimes soon enough.
- Do not deploy the correlation-gated candidate yet.
- Next better direction: combine correlation with a per-day crash detector or
  chronological intraday worksteal sim, so the algorithm can reduce gross
  exposure when the first filled leg is already showing regime stress.

## 2026-05-03 - Minimum secondary allocation floor

Motivation:

- Top-2/top-3 can still behave like top-1 if learned allocation or score
  weighting concentrates too much in the highest-scored name.
- Add an explicit portfolio constraint: when at least two trades fill, cap the
  largest allocation weight so at least a fixed fraction remains outside the
  main pick. Example: `min_secondary_allocation=0.20` means the largest filled
  position can receive at most 80% of gross exposure.

Implemented:

- Added `BacktestConfig.min_secondary_allocation`.
- Added sweep knob `--min-secondary-allocation-grid`.
- Applies after equal/score_norm/softmax/worksteal weights and after
  long/short scaling, so it is a general concentration guard.
- Added tests for weight capping, sweep identity/plumbing, and preserved
  attribution support.

Validation:

- `.venv313/bin/python -m py_compile xgbnew/backtest.py xgbnew/sweep_ensemble_grid.py xgbnew/symbol_attribution.py`
- `.venv313/bin/pytest -q tests/test_xgbnew_backtest_packing.py tests/test_xgbnew_sweep_ensemble_grid.py tests/test_xgbnew_symbol_attribution.py`
- Result: `108 passed, 2 warnings`.

Focused alltrain worksteal/correlation sweep:
`analysis/xgbnew_daily/worksteal_secondary_floor_corr_20260503/sweep_20260503_201501.json`

- Config matched the prior worksteal setup: `top_n=2`, opportunistic watch `4`,
  entry discount `30bps`, `prod10bps`, `corr_window_days=60`,
  `corr_max_signed in {0.4,0.6}`.
- Best median remained the existing 2.25x worksteal/correlation cell:
  `+23.37%/mo`, p10 `-51.22%`, worst DD `35.08%`.
- `min_secondary_allocation=0.20` was identical for worksteal because
  worksteal already allocates `75%/25%` for two fills.
- `min_secondary_allocation=0.30` reduced median to about `+20.42%/mo` and
  did not repair the p10/DD tail.
- Softmax learned allocation stayed much lower median (`~8%` to `9%/mo`) with
  p10 near `-48%`.

Held-out 2025H2 replay:
`analysis/xgbnew_daily/heldout_worksteal_secondary_floor_corr_20260503/sweep_20260503_202022.json`

- Same worksteal/correlation family on models trained only through
  `2025-06-30` remained negative.
- Best-looking cells were still around `-2%/mo` median with p10 from about
  `-16%` to `-21%`, DD around `36%` to `40%`, and `2/3` negative windows before
  fail-fast pruning.

Conclusion:

- Keep `min_secondary_allocation` because it is the right structural guard for
  learned allocation and future RL policies.
- It does not solve the current XGB worksteal tail; the bottleneck is still
  regime detection / signal transfer, not just portfolio concentration.

## 2026-05-04 - Safety goodness metric

Why p10 can be worse than drawdown:

- `p10_monthly_pct` is a monthly-normalized endpoint return per rolling
  window.
- `worst_dd_pct` is the raw peak-to-trough equity drawdown inside a simulated
  window.
- When fail-fast stops a window early, a 35% raw loss over a short elapsed
  interval can monthly-normalize to a much larger endpoint loss, so `p10=-55`
  and `DD=35` can both be internally consistent.
- The optimizer should see the worse interval risk directly rather than
  requiring a human to compare those two different scales.

Implemented:

- Added per-window `window_interval_loss_pcts`:
  `max(endpoint_monthly_loss, close_to_close_drawdown, intraday_drawdown)`.
- Added cell-level `p05_monthly_pct`, `worst_monthly_pct`,
  `p90_interval_loss_pct`, `worst_interval_loss_pct`.
- Added `safety_goodness_score`, a harsher optimizer target:
  `p10 + 0.1*median - p90_interval_loss - 0.5*worst_interval_loss`
  minus negative-window, negative-magnitude, time-under-water, and ulcer
  penalties.
- Sweep output now prints `loss90`, `lossW`, and `safeG`, and sorts the main
  table by `safeG`.
- Friction-robust strategy summaries now carry worst safety score and rank by
  safety before pain-adjusted goodness.

Validation:

- `.venv313/bin/python -m py_compile xgbnew/sweep_ensemble_grid.py`
- `.venv313/bin/pytest -q tests/test_xgbnew_sweep_ensemble_grid.py`
- Result: `88 passed, 2 warnings`.

Focused alltrain no-fail sweep:
`analysis/xgbnew_daily/safety_goodness_worksteal_corr_nofail_20260503/sweep_20260504_000055.json`

- Same top-2 worksteal/correlation family, no fail-fast pruning, 6 full
  windows.
- Best safety-ranked cell was 2.0x rather than 2.25x:
  median `+11.43%/mo`, p10 `-9.80%`, `loss90=41.35%`, `lossW=52.63%`,
  `safeG=-122.43`.
- The tempting 2.25x high-median cells had worse interval loss
  (`loss90≈45%`, `lossW≈57%`) and lower safety scores (`safeG≈-133` to
  `-140`).

Focused held-out no-fail sweep:
`analysis/xgbnew_daily/safety_goodness_heldout_worksteal_corr_nofail_20260504/sweep_20260504_000423.json`

- Same family on models trained through `2025-06-30`.
- Best safety cell was still negative: median `-4.19%/mo`, p10 `-25.18%`,
  `loss90=50.18%`, `lossW=51.69%`, `4/6` negative windows,
  `safeG=-197.55`.

Conclusion:

- `safety_goodness_score` is a better optimizer objective for this project
  than median or plain p10/DD goodness because it measures interval pain on
  one comparable scale.
- The current worksteal XGB family is still rejected; the metric now makes
  that rejection automatic and visible in the sweep table.
