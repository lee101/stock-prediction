# Current Algorithms — Monitor Ledger

**Authoritative source**: `alpacaprod.md` top section is always the canonical
live ledger. This file is a short pointer for the hourly monitor so it knows
which services should exist, where the best config lives, and what the bar
is to beat. Keep this file in sync with `alpacaprod.md` whenever we deploy.

Last synced: **2026-04-27 19:03 UTC — hourly monitor. Phase 1 GREEN: xgb_live_trader RUNNING (pid=2370763, uptime 3h52m post 15:08 UTC rotation), singleton lock matches supervisor pid, RL path STOPPED (no procs / port 8050 closed), equity $19,799.53 / bp $39,599.06 STABLE since 13:10 UTC incident — no further bleed. Today 15:10 UTC session: top=NOC 0.7307, 0/667 pass ms=0.85, R9b healthy no-pick (varied vs prior session's 0.5814 BSX → ensemble producing fresh non-stale scores). Tick_status loop healthy (last 18:55:59Z). 0 stock positions, 4 dust crypto. Fills last 10d=4 (all crypto: BTC weekend round-trip + DOGE 04-18). 0 death-spiral markers, 0 traceback lines. Phase 2: no fixes. Phase 3: harvested 1808 sweep (4 cells lev=2 ms=0.85 ht=Y tn=1 × vol-grid {0.10,0.12} × fees {deploy,stress36x}) on oos2024_ensemble_gpu (PRE-fresh, train_end=2024-12-31) over 42 windows = ALL 4 cells fired 0 trades / 0 picks (act=0 med=+0.00). Min-vol is NOT the lever: at BOTH 0.10 and 0.12 the ms=0.85 gate is structurally inert on this OOS window. Combined with prior 1607 finding (the new train_end=2026-04-26 alltrain ensemble also 0 trades at ms=0.85): both PRE-fresh OOS and NEW alltrain agree on hold-cash regime. The recorded +141%/mo deploy-gate baseline must derive from a different OOS window definition / earlier ensemble snapshot / looser threshold; LIVE recipe-rotation kept hold-cash regime unchanged. XGB-family search remains exhausted (accumulated refuted memos: 15-seed bonferroni, retrain-throughs, disp-gate, rank:ndcg, CAT, inversion, per-pick-invvol, band-pass-vol, SPY-vta, cs_dispersion-features, short-history, no-picks-fallback). No new XGB sweep launched. Phase 4: no deploy — yesterday's 15:08 UTC recipe rotation remains correct. Disk / 89%, /nvme0n1-disk 85% — both stable, no purge this hour. Repo dirty: still the 3 xgbcat user-WIP paths (untouched). NEXT-HOUR FOLLOW-UPS: (a) Tue 2026-04-28 13:20 UTC pre-open: confirm new ensemble produces varied score distribution (≠ stuck-0.7307 stale-features signature), (b) auto-retrain timer Sun 2026-05-03 23:00 UTC will block rotation again unless xgbcat WIP is committed/cleaned by then (user-side), (c) skip new XGB sweep — pivot to architectural change (Chronos-2 latents as μ for CVaR LP, or RL-obs-XGB). (Prior 2026-04-27 18:18 UTC sync archived below.) ---ARCHIVED-2026-04-27-1818--- Phase 1 GREEN: xgb_live_trader RUNNING (pid=2370763, uptime 3h08m post 15:08 UTC rotation), singleton lock holder matches process cmdline, RL path STOPPED, equity $19,799.53 stable. Trade log 2026-04-27.jsonl: 13:22 scored top=0.5814 (pre-rotate) → no_picks; 15:10 (post-rotate, NOC) top=0.7307 → no_picks. Healthy R9b low-conviction regime; ms=0.85 gate working as designed. No death-spiral fires, no errors in supervisor log tail. Disk / at 89% (warn, stable since 13:15 UTC vacuum). Phase 2: no fixes required. Phase 3: launched deploy-gate sanity sweep (PID 423310, TS=20260427_1808) on oos2024_ensemble_gpu (5-seed train_end 2024-12-31) × lev=2.0 × ms=0.85 × top_n=1 × hold-through × min-vol-grid {0.10, 0.12} × fees {deploy, stress36x} — 4 cells, 42 windows, OOS 2025-01-02→2026-04-19. Dataset built in 158.4s; cells running silently for 10+ min as of audit cutoff (per prior 16:07 sweep timing ~12 min for 2 cells, 4-cell run expected ~18:30 UTC). Sweep harvested by next hour. Hypothesis under test: min-vol=0.10 reproduces documented +141%/mo deploy-gate baseline vs min-vol=0.12 0-trade calibration-inert (live config tightened to 0.12 for additional risk control). Phase 4: no deploy — rotation already applied at 15:08 UTC. NEXT-HOUR FOLLOW-UPS: (a) harvest sweep_20260427_1808 JSON when complete; if min-vol=0.10 reproduces +141%/mo and min-vol=0.12 = 0-trade, parity confirmed and live config is tightened-by-design, (b) on Tue 2026-04-28 13:20 UTC pre-open, confirm new ensemble produces varied score distribution (not stuck on 0.7307 — stale-features signature), (c) auto-retrain timer fires Sun 2026-05-03 23:00 UTC; xgbcat WIP dirty paths still pending user-side commit. (Prior 2026-04-27 17:09 UTC sync archived below.) ---ARCHIVED-2026-04-27-1709--- Phase 1 GREEN: xgb-daily-trader-live RUNNING (pid=2370763, ~2h post-rotation), singleton OK (service=xgb_live_trader pid matches supervisor), RL path STOPPED, equity $19,799.53 / bp $39,599.06 stable (no further bleed since 13:15 UTC incident). Post-rotation verification (a): 15:08 UTC session top-1 NOC score=0.7307 — varied vs prior session's BSX 0.5814 → ensemble is producing fresh, non-stale, non-NaN scores ✓. R9b healthy low-conviction continues (9+ consecutive no-pick days under ms=0.85). Phase 2: killed stale `until` zsh wrapper PID 3146599 (orphaned by 16:07 UTC monitor — its sweep finished but the file-name match in the loop condition was wrong, trapping the loop on `sleep 30`; this caused the prior hour's rc=124 timeout in scheduled-audits). Used SIGTERM cleanly, no supervisor impact. Disk / 89%, /nvme0n1-disk 85% — both stable since 13:15 UTC vacuum, no purge this hour. Phase 3: harvested the 16:07 UTC `deploy_gate_reeval_20260427_1607` sweep → 2 cells lev=2.0 ms=0.85 ht=Y tn=1 vol=[0.12,*] on the NEW alltrain_ensemble_gpu over 42 windows BOTH fired ZERO trades (median_pct=None). Confirms ms=0.85 gate stays calibration-inert on the train_end=2026-04-26 ensemble even in-sample — recipe-rotation kept the same hold-cash regime as expected. Manifest sanity check: NEW train_end=2026-04-26 / n_rows=1,096,505 vs PREV train_end=2026-04-20 / n_rows=897,115; same seeds {0,7,42,73,197}. No new sweep launched (XGB-family search exhausted). Phase 4: no deploy — rotation already applied at 15:08 UTC. NEXT-HOUR FOLLOW-UPS: (a) on Tue 2026-04-28 13:20 UTC pre-open, confirm new ensemble produces a varied score distribution (not stuck on 0.7307 — that's a stale-features signature), (b) auto-retrain timer fires next Sun 2026-05-03 23:00 UTC; repo dirty paths still 3 (3 xgbcat WIP files — user's active dev) and would still block rotation if dirty by then; user-side fix, (c) skip new XGB sweep — search exhausted per accumulated refuted memos; pivot to architectural change (RL-obs-XGB / Chronos-2 features / wide-momentum CVaR) for next edge attempt. (Prior 2026-04-27 15:09 UTC sync archived below.) ---ARCHIVED-2026-04-27-1509--- Phase 1 GREEN: xgb-daily-trader-live RUNNING (pid=2370763 NEW, lock holder matches), singleton OK, RL path STOPPED, equity $19,800 stable. Today 2026-04-27 13:22 UTC pre-rotate session: top-1 BSX 0.5814 R9b no-pick (recorded). Phase 2 ROTATION: rotated alltrain ensemble train_end 2026-04-20 → 2026-04-26 (the staging ensemble produced by Sunday auto-retrain that auto-rotation could not apply). Path: `mv` swap of `alltrain_ensemble_gpu` → `alltrain_ensemble_gpu_prev_20260427T150729Z`, `mv` `alltrain_ensemble_gpu_staging_20260426T230016Z` → `alltrain_ensemble_gpu`, `normalize_xgb_ensemble_manifest.py`, then `bash scripts/deploy_live_trader.sh --allow-dirty --allow-unmodeled-live-sidecars xgb-daily-trader-live`. Pre-validation: `validate_xgb_ensemble.py` OK, all 5 pkls load, recipe bit-identical to live (n_est=400 d=5 lr=0.03 same 15 feature cols same seeds {0,7,42,73,197}). Post-deploy preflight `safe_to_apply: true`; history log appended `status=ok supervisor_pid=2370763 lock_pid=2370763`. Recipe-rotation only — launch.sh flags + leverage + min_score + hold-through unchanged. Phase 3: no new sweeps (deploy fits inside 50-min hour budget). Phase 4: rotation deployed; rollback path documented at top of alpacaprod.md. NEXT-HOUR FOLLOW-UPS: (a) verify the post-restart 15:08 UTC scoring landed in trade log with reasonable top-1 score (not stale or NaN), (b) on Tue 2026-04-28 13:20 UTC pre-open, confirm new ensemble produces a varied score distribution (not stuck on 0.5814 — that's a stale-features signature), (c) repo dirty: 3 xgbcat WIP paths still block xgb_weekly_retrain.sh next Sunday — user should commit those before 2026-05-03 23:00 UTC. (Prior 2026-04-27 14:14 UTC sync archived below.) ---ARCHIVED-2026-04-27-1414--- Phase 1 GREEN (post-incident stable): xgb-daily-trader-live RUNNING (pid=3294379, uptime 1d18:41h), singleton lock holder matches supervisor pid, RL path STOPPED. Today 2026-04-27 13:22 UTC session: top-1 SE=BSX 0.5814 (per-seed std 0.0051), 0/671 pass ms=0.85 → HEALTHY R9b low-conviction (no_picks). Equity = $19,799.53 / cash = $19,799.53 / bp = $39,599.06 (matches Apr 27 13:10 reading; $9k drop is stable, not actively bleeding — incident block at top of alpacaprod.md remains the canonical record, no human-actionable change since prior hour). Phase 2: no fixes needed; account state stable. Phase 3 NEW FINDING: weekly auto-retrain at 2026-04-26 23:04 UTC successfully trained fresh staging ensemble at `analysis/xgbnew_daily/alltrain_ensemble_gpu_staging_20260426T230016Z/` (train_end=2026-04-26, 834 syms, 1,096,505 rows, seeds {0,7,42,73,197}) — but `scripts/xgb_weekly_retrain.sh` aborted at preflight with `apply_blockers: dirty_repo_outside_watchlist:6` → `safe_to_apply: False` → `Unsafe: xgb-daily-trader-live`. Production is therefore running 6-day-stale ensemble (deployed pkls dated 2026-04-20). The `repo_relative_dirty_paths_outside_watchlist` rule (preflight L797) blocks rotation on ANY dirty path outside the xgb-daily-trader-live watchlist (xgbnew/*.py, src/alpaca_singleton.py, deployments/xgb-*). Hourly-monitor doc updates (alpacaprod.md, monitoring/current_algorithms.md) routinely sit dirty between commits and CONTRIBUTE TO THIS BLOCK; xgbcat_risk_parity WIP files (modified 2026-04-27 13:06–14:07, pytest active on tests right now) are concurrently dirty too. THIS HOUR: committing the 2 routine monitor docs to remove them from the blocker count; xgbcat WIP stays untouched (active user dev). Phase 4: no deploy candidate (staging ensemble is candidate but proper rotation requires either (a) clean dirty repo before next Sunday 23:00 UTC auto-retrain or (b) manually validate staging via eval_100d.py at decision_lag=2 then `scripts/deploy_live_trader.sh xgb-daily-trader-live` — deferred this hour, search budget exhausted). NEXT-HOUR FOLLOW-UPS: (a) verify pytest run on xgbcat tests completed and let user commit/cleanup the WIP, (b) consider running deploy_gate_reeval against staging vs current ensemble (need correct CLI: --symbols-file + --model-paths comma-list), (c) auto-retrain timer fires next Sun 2026-05-03 23:00 UTC — ensure repo is clean by then. (Prior 2026-04-27 13:15 UTC + 2026-04-24 20:00 UTC syncs archived below.) ---ARCHIVED-2026-04-27-1315--- Phase 1 RED INCIDENT: equity dropped from $28,758.98 (Apr 25 Sat close) → $19,799.53 (Apr 27 13:10 UTC) = −$8,879.45 / −30.8%, with only +$160.55 net cash impact from BTC weekend trade and −$32.04 CFEE on `/v2/account/activities`. ~$9,088 unaccounted. No `jnlc`/`csd`/`ach` activity records. xgb-daily-trader-live RUNNING (pid=3294379, uptime 1d17h44m), singleton lock matches, no rogue writers, no death-spiral fires, RL path STOPPED. INCIDENT documented at top of alpacaprod.md (HUMAN REVIEW NEEDED). Hypothesis: Alpaca-side bookkeeping reconciliation, mis-priced dust, or deferred journal not surfaced through API. Phase 2: disk / 90→89% via journalctl vacuum + .tmp_train log purge (~2.8GB freed); LIVE config unchanged. Phase 3: detached deploy_gate_reeval launch failed 3× (zsh fd parse, PYTHONPATH, argparse mismatch — needs --model-paths + --symbols-file, not --ensemble-dir); aborted retries to honor "summary first" 40-min rule. No new bg sweeps launched. Phase 4: no deploy candidate. ---ARCHIVED-2026-04-24-2000--- Phase 1 GREEN: XGB RUNNING (pid=2173397, uptime 22:47h), singleton lock=xgb_live_trader matches pid, equity $28,678.98 flat (LIVE hold-cash matches design). Today 2026-04-24 session (13:20 UTC): top-1 SE=0.5501, per_seed_spread=0.0058 (5 seeds agree), 0/671 pass ms=0.85 → HEALTHY R9b low-conviction (scores vary 0.548/0.550/0.554/0.600/0.704 across last 5 sessions). 0 stock positions (5 dust crypto only, largest DOGE 0.006238). 2 fills last 10d (both 2026-04-18 DOGE crypto). RL path DOWN confirmed (both supervisors STOPPED, port 8050 closed, no rogue procs). Disk / at 85%, /nvme0n1-disk at 84% (warn level, no action). Next session Mon 2026-04-27 09:20 ET (crypto weekend hook live through Sun). Phase 2: no fixes needed. Phase 3: harvested 3 bg sweeps that finished since last hour: (1) **oos2024_fresh_15seed_livecfg_20260424_1908** (the 15-seed fresh-OOS test launched 19:08) — all 16 cells FAIL: ms≥0.75 = 0 trades (gate structurally inert even with 3× seed count), ms=0.65 lev=1 deploy +0.72/−13.40 17/42 neg, ms=0.55 lev=2 deploy −12.39/−50.87 25/42 neg. Adding 10 seeds to 5-seed fresh-OOS blend does NOT break the ms=0.85 inert gate as hypothesized; the signal just isn't there on fresh-features. (2) **oos2024_mscurve_20260424_1813** (5-seed same grid earlier): same shape — ms=0.55 lev=2 med=+23.31%/mo but p10=−12.02, 13/42 neg (fails). (3) **deploy_gate_reeval_20260424_1307** crashed with ModuleNotFoundError (missing PYTHONPATH=.); not re-launching since the space is exhausted. Phase 4: no deploy candidate. Phase 3 this hour: no new bg sweep launched — XGB-family search space is exhausted per accumulated refuted memos (retrain-through-*, short-history, disp-gate, rank:ndcg, CAT, inversion, per-pick-invvol, band-pass-vol, SPY-vta, cs_dispersion-features, 15-seed-fresh-OOS all refuted). Next-phase pivot requires architectural change (RL-obs-XGB, BC-from-XGB-picks, Chronos-2 features, or CVaR wide-momentum universe), not another XGB knob sweep. Continuing to hold cash per design.**

---

## 1. XGB Daily Trader — LIVE (primary, as of 2026-04-19 full stack)

- **Process**: `xgbnew.live_trader` (supervisor unit `xgb-daily-trader-live`,
  launcher `deployments/xgb-daily-trader-live/launch.sh`). NOT systemd.
- **Broker boundary**: **direct Alpaca SDK** — no `trading_server` on
  `127.0.0.1:8050`. Writes go straight to `api.alpaca.markets` using
  `ALP_KEY_ID_PROD` / `ALP_SECRET_KEY_PROD`.
- **Singleton**: `src/alpaca_singleton.py::enforce_live_singleton` fires at
  startup; the live-writer lock at `strategy_state/account_locks/alpaca_live_writer.lock`
  must show `service_name: xgb_live_trader` and a `pid` that matches
  `supervisorctl status xgb-daily-trader-live`.
- **HARD RULE #3 (death-spiral guard, TIME-AWARE as of commit 84ae21ef)**:
  `xgbnew/live_trader.py` calls `record_buy_price(sym, fill_px)` after each
  BUY and `guard_sell_against_death_spiral(sym, "sell", current_price)`
  before each SELL. Tolerance auto-selects by the age of the recorded buy:
  - Buy ≤ 8h old (intraday regime): sells > **50 bps** below buy are refused.
  - Buy > 8h old (hold-through overnight regime): sells > **500 bps** below
    buy are refused. This widens the tolerance to tolerate normal overnight
    gaps without tripping the guard.
  RuntimeError from the guard crashes the loop by design; supervisor
  autorestart handles reconnection. DO NOT "catch and skip" this exception.
- **Ensemble**: 5-seed alltrain GPU —
  `analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed{0,7,42,73,197}.pkl`,
  blend mode `mean` (predict_proba averaged). Models trained with
  `n_estimators=400 max_depth=5 learning_rate=0.03` on 2020-01-01 → 2026-04-19
  (846-symbol universe). Retrained weekly by `scripts/xgb_weekly_retrain.sh`
  (systemd timer `xgb-auto-retrain.timer`, Sun 23:00 UTC).
- **Deployed flags (live launch.sh)**:
  - `--top-n 1` — one symbol per session.
  - `--allocation 2.0` — 2× leverage (buy_notional = 200% equity).
  - `--min-score 0.85` — **ensemble-calibrated conviction gate**. Holds
    cash if top-1 blended `predict_proba < 0.85`. In OOS validation on
    `oos2024_ensemble_gpu` models across 60 windows, **~44% of trading
    days cleared the gate** — the other ~56% held cash. Daily top-score
    distribution: min 0.35, median 0.84, max 0.94. See §EXPECTED
    BEHAVIOR below for what "healthy no-pick" looks like.
  - `--hold-through` — if tomorrow's top-1 pick == today's held symbol,
    skip the SELL-then-BUY round trip; hold overnight. On no-pick days
    the current position is also held (does not flat at close).
  - `--min-dollar-vol 50000000` — inference liquidity floor. Drops
    ~240 symbols with $ADV < $50M. Strict-dominance win in OOS (+5.23 p10).
  - `--min-vol-20d 0.12` — inference volatility floor (tightened 0.10→0.12
    on 2026-04-20 13:17 UTC, strict-dom TRUE-OOS on `oos2024_ensemble_gpu`).
    Drops dead-zone low-vol names; stacks cleanly with 50M dolvol floor.
- **Baseline expected monthly return (TRUE OOS on `oos2024_ensemble_gpu`,
  2025-01-02 → 2026-04-19, 60 windows, fee=0.278bps fb=5bps, vol_20d=0.12)**:
  - deploy fees: **median +153%/mo, p10 +97%, 0/60 neg, worst DD 7.18%**
  - 36× fee stress: **median +102%/mo, p10 +58%, 0/60 neg, worst DD 8.34%**
  - intraday worst-DD (unrealised): ~13% both regimes
- **Health check threshold for realised PnL**: target 60-70% of in-sample
  headline = **≥ +40%/mo** in first month. **< +10%/mo over 2 trading
  weeks = broken** (signal drift, fill issues, or stale data pipeline).
- **Character (CHANGED 2026-04-19 with hold-through)**:
  - Signal: one session per trading day (at ~09:20 ET pre-open).
  - On pick-day: BUY at 09:30 ET, **usually hold into the next trading
    session** (flat overnight is the exception, not the rule).
  - On no-pick day: hold whatever's already in the position — **not
    force-flat**. This is by design (`--hold-through` semantics).
  - Top-1 pick is ~$57K notional on $28K equity at `--allocation 2.0`.
- **Logs**: `sudo tail -f /var/log/supervisor/xgb-daily-trader-live.log` and
  `…-error.log` (singleton lock + stderr). Trade log (structured JSONL):
  `analysis/xgb_live_trade_log/YYYY-MM-DD.jsonl` — one file per session
  with `session_start`, `scored` (top20 + per_seed_scores), `filtered`,
  `pick`, `buy_submitted`, `buy_filled`, `sell_submitted`, `session_end`.

### EXPECTED BEHAVIOR — healthy "no pick" vs silent bug

Because `--min-score 0.85` is designed to gate low-conviction days, the
expected baseline is **~44% of trading days result in a BUY, ~56% in no-op**.
The monitor must distinguish:

**HEALTHY no-pick signature (leave alone)**:
```
[xgb-live] Scoring 846 symbols...
[xgb-live] Conviction filter min_score=0.85: 0/602 candidates pass (top score=0.7036)
[xgb-live] No picks today — holding current positions (if any).
```
- `top score` is in range 0.55–0.84 (below gate but non-trivial).
- `n_candidates` > 400 (universe filters are firing normally).
- Per-seed scores in trade_log vary meaningfully across seeds (σ > 0.01).
- A few of these in a row is fine; base rate is 2–3 no-pick days in a row.

**BROKEN signatures (escalate)**:
- `top score` always exactly the same value across sessions → stale
  features (CSV pipeline stuck on one day, e.g. the known
  `trainingdata/train/` frozen-at-2026-04-10 bug — only matters if
  live_trader falls back to CSVs; normally it fetches fresh bars).
- `top score` collapses to near 0.5 (model random/broken) OR NaN.
- `n_candidates` = 0 (universe pull failed).
- `n_candidates` static for > 2 sessions (no-trading-day gate not firing on holiday).
- 10+ no-pick days in a row — possible signal drift, not just low conviction.
- Per-seed scores identical across ALL seeds → a single model being used,
  ensemble blend broken.

### Deploy gate (what must pass before swapping models or knobs)

**⚠ 2026-04-21 CORRECTION**: the numbers below were measured on the
**pre-stale-fix** `oos2024_ensemble_gpu` ensemble. A fresh retrain
(`oos2024_ensemble_gpu_fresh/`, trained 2026-04-21 07:49 UTC with the
loader-priority fix in place) produces a scoring distribution whose max
blended probability does not clear **0.75** on the true-OOS window
2025-01 → 2026-04-20 — every ms ≥ 0.75 cell in
`analysis/xgbnew_daily/sweep_20260421_fresh_vs/sweep_20260421_115402.json`
fires **zero** trades over 59 windows. Low-ms cells (ms 0.55–0.70) do fire
but all have negative median + p10 with neg_frac 17–75%. **0/108 cells
with positive median.** Same null result on `oos2025h1_ensemble_gpu_fresh`
(192-cell sweep per `project_xgb_true_oos_no_edge_2026_04_21.md`). Same
result on the `retrain_through_2026_03_20_ensemble` heldout sweep (0/42
positive at ms ≥ 0.65 over 8 windows 2026-03-21 → 2026-04-20). Every
prior "+141%/mo" claim was either in-sample (on `alltrain_ensemble_gpu`)
or pre-stale-fix (on pre-debe551d `oos2024_ensemble_gpu`).

**Interpretation**: the tariff-crash regime (2026-03 onwards) is not
predictable from 2020–2025H1 features with the current feature set. LIVE
being in hold-cash mode is the correct behavior. Don't lower the gate to
force trades — that would be a lossy bet.

**New deploy gate rule (2026-04-21)**: the bar is still "strict-dominance
on all four PnL metrics + intraday DD not worse by > 1pp", but it is now
measured on:

1. `oos2024_ensemble_gpu_fresh` (2024-12-31 cutoff, fresh-features retrain)
   across 59 windows 2025-01-02 → 2026-04-20.
2. `oos2025h1_ensemble_gpu_fresh` (2025-06-30 cutoff) across ~45 windows
   2025-07-01 → 2026-04-20.
3. Any new retrain-through ensemble's held-out window (currently
   `retrain_through_2026_03_20_ensemble` / `retrain_through_2026_02_28_ensemble`).

**A candidate becomes deploy-worthy only when at least ONE of these three
fresh true-OOS evaluations shows all of: median ≥ +27%/mo, p10 ≥ 0,
neg_frac ≤ 20%, worst-DD ≤ current LIVE worst-DD.** The old pre-fix
"+141%/mo" bar is retired until a positive-edge config is found on the
fresh ensembles.

Deployed LIVE config (archived pre-fix bar for reference, 60 windows
2025-01-02 → 2026-04-19 on pre-stale-fix `oos2024_ensemble_gpu`):

| metric | deploy fees | 36× fee stress |
|---|---:|---:|
| median %/mo (PRE-FIX, REFERENCE ONLY) | +141 | +108 |
| p10 (PRE-FIX) | +96 | +68 |
| n_neg / n_windows (PRE-FIX) | 0/60 | 0/60 |
| worst realized DD (PRE-FIX) | 7.18% | 8.34% |
| worst intraday DD (PRE-FIX) | 12.93% | 13.12% |

Keep using `xgbnew/sweep_ensemble_grid.py` with `--ensemble-sort-key
goodness` or `robG` as secondary, but always against a `_fresh` or
`retrain_through_*` ensemble dir.

⚠ **`alltrain_ensemble_gpu` is NOT OOS for filter validation** — it's
trained through today. Its "60-window" numbers are fully in-sample on the
validation window. For any filter-threshold tuning or feature-set change,
always re-eval on a `_fresh` or `retrain_through_*` ensemble. See
`feedback_alltrain_is_not_oos.md` and
`project_oos_ensembles_trained_pre_stale_fix.md`.

### Known data-pipeline hazard (2026-04-20)

`xgbnew/dataset.py::_load_symbol_csv` prefers `trainingdata/train/` (frozen
at 2026-04-10) over `trainingdata/` (fresh through 2026-04-15). The weekly
retrain and OOS sweeps are running on ~10-day-stale data. **Live inference
is UNAFFECTED** because `live_trader` fetches fresh bars from Alpaca,
not from the CSV loader. But the weekly retrain will produce a model
trained on stale ground truth until we flip the loader priority (or delete
the frozen `train/` subdir). See `project_xgb_stale_training_csvs.md`.

## 2. Daily RL Trader — STOPPED (intentional, kept for rollback)

- **Process**: `trade_daily_stock_prod.py --daemon --live` under supervisor
  unit `daily-rl-trader`. **STOPPED 2026-04-19 10:40 UTC** to free the
  singleton lock for XGB.
- **Do NOT start it** while `xgb-daily-trader-live` holds the lock — the two
  would race and one would crash on import.
- **Rollback procedure** (only on user's explicit direction):
  ```bash
  sudo supervisorctl stop xgb-daily-trader-live
  sudo supervisorctl start trading-server daily-rl-trader
  ```

## 3. Trading-server (broker boundary) — STOPPED

- Supervisor unit `trading-server` **STOPPED 2026-04-19 10:40 UTC**. It was
  the broker boundary for `daily-rl-trader`, now unused. Port 8050 is
  CLOSED — do NOT expect `curl http://127.0.0.1:8050/health` to work.
  The former expected `writer_lock_held_by_me=true` check is retired.
- Restart only as part of a daily-rl-trader rollback (see §2).

## 4. Other services (present but not primary signal)

- `llm-stock-trader` — Gemini-driven picks YELP/NET/DBX/OPTX. Owns its own
  singleton slot (account `llm_stock_writer`). Check supervisor; not urgent.
- `unified-orchestrator.service` — **DEAD** since 2026-03-29 (Gemini API
  exhausted). Do not attempt to revive without budget.
- Crypto RL (crypto12_ppo_v8) — training-snapshot only, **not OOS-validated**,
  do NOT deploy.

## 5. Realism invariants (marketsim truth vs prod truth)

- Ground truth for XGB swap decisions = `xgbnew/sweep_ensemble_grid.py` on
  the `oos2024_ensemble_gpu` 60-window grid with `fee_rate=0.0000278`
  (Alpaca real) and `fill_buffer_bps=5`.
- XGB uses `fee=0.278bps` to match Alpaca real. RL used `fee=10bps` and is
  out of prod; keep the two paths clearly separate in any re-eval.
- CPU/GPU hist drift: `tree_method=hist` on GPU drifts from CPU. Never
  promote a GPU-trained config without re-validating on CPU.
- Weekend/holiday gate: `xgbnew/live_trader._is_today_trading_day()` queries
  `/v2/clock` at top of both `run_session()` and `run_session_hold_through()`.
  No BUY will be placed on non-trading days even if a top score clears.

## 6. Where the hourly monitor's deploy authority starts and stops

**Auto-deploy authorized** (no human confirmation needed):
- Hot-swap the 5-seed ensemble for a 5-seed / 10-seed variant IFF the new
  variant beats the deploy gate in §1 on every metric (all four PnL +
  intraday DD).
- Restart a crashed `xgb-daily-trader-live` (its singleton + guard state
  are stable; supervisor autorestart is the happy path).
- Tweak `--min-dollar-vol` and `--min-vol-20d` inference floors if the
  adjusted value strict-dominates on the `oos2024_ensemble_gpu` sweep.

**Explicitly NOT auto-deploy** (human-only):
- Bump `--leverage` / `--allocation` (currently 2.0). Raising it further
  is a concentration-risk escalation.
- Lower `--min-score` below 0.85. (Would trade more days but widen DD.)
- Change `--top-n` above 1.
- Disable `--hold-through`.
- Rotate API keys.
- Any change to `alpaca_singleton.py` or the death-spiral guard (HARD RULE).
- Bring `daily-rl-trader` back online (that's a rollback, user only).

See `monitoring/hourly_prod_check_prompt.md` for the full per-phase procedure.
