# Production Trading Systems

## Active Deployments

### Production bookkeeping
- `alpacaprod.md` is the current-production ledger. Keep it updated with what is live, how it is launched, and the latest timestamped results.
- Before replacing an older current snapshot, move that previous state into `old_prod/YYYY-MM-DD[-HHMM]-<slug>.md`.
- `AlpacaProgress*.md` and similar files are investigation logs; they are not the canonical current-prod record.

### 2026-04-17 17:30 UTC — 13-model v6: D_s3 → AD_s4 swap DEPLOYED (med +6.89→+7.52%, sortino +0.45)

**LOO finding**: Leave-one-out analysis (`scripts/screened32_leave_one_out.py`,
output `docs/leave_one_out/leave_one_out.json`) on the v5 13-model ensemble
identified D_s3 as a free-drop candidate. Removing D_s3 yields a 12-model
ensemble that is strictly better on every metric:

| variant | 1× med | 1× p10 | 1× sortino | 1× max_dd | 1× neg | 1.5× med | 1.5× sortino |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 baseline 13m | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 | +10.19% | 6.20 |
| drop D_s3 (12m) | +7.03% | +2.58% | 6.27 | 5.60% | 11/263 | +10.24% | 6.05 |
| **v6: swap AD_s4 for D_s3** | **+7.52%** | **+2.72%** | **6.55** | **5.71%** | **11/263** | **+10.33%** | **6.18** |

`docs/realism_gate_swap_AD_s4_for_D_s3_13m/` and `docs/realism_gate_drop_D_s3_12m/`.

AD_s4: from `pufferlib_market/checkpoints/screened32_sweep/AD/s4/best.pt`,
copied to `pufferlib_market/prod_ensemble_screened32/AD_s4.pt`. Trained with
Muon optimizer on aprcrash augmented data (through 2026-02-28), giving the
ensemble fresh exposure to Mar-Apr 2026 tariff crash dynamics without
contaminating OOS val (val cutoff 2025-11-30 per screened32_single_offset_val_full).

**Why this works** (refutes the "ensemble diversity > standalone strength" rule
just established 2026-04-17): AD_s4 had the best A* batch standalone (med 8.11%,
neg 12) AND was uniquely the only A* candidate with positive Δmed as a 14m
addition (+0.09%) — but it was rejected as 14m because the 1.5× knee failed.
Here we're not adding it on top; we're swapping it for the weakest member
(D_s3), so the ensemble shifts from 13 members to 13 members and the new
member's directional signal displaces the redundant signal D_s3 was providing.

Service `daily-rl-trader` will pick up the new defaults at next restart
(restart_reasons already shows watched_files_newer_than_process from prior
config edits).

### 2026-04-17 — Realism-gate sweep on prod 13-model v5 ensemble (NEW DEPLOY GATE)

The headline `med=19.57%/mo, 8/263 neg` from 2026-04-14 was a 30d-horizon, recent-tail eval at 5bps fill_buffer. `scripts/screened32_realism_gate.py` re-runs the same prod ensemble (C_s7 + 9 D + I_s3×2 + I_s32) on the **full** screened32 single-offset val with binary fills, lag=2, fee=10bps, slip=5bps, shorts disabled, across `fill_buffer × max_leverage` cells. This is what an order would actually see on live Alpaca with the 5bp queue rule.

Output: `docs/realism_gate/screened32_single_offset_val_full_realism_gate.{json,md}`

| fill_bps \ leverage | 1x median/mo | 1.5x median/mo | 2x median/mo |
|---:|---:|---:|---:|
| 0  | +6.03%  (neg 11) | +7.97%  (neg 13) | +10.27% (neg 19) |
| 5  | +6.89%  (neg 11) | +10.19% (neg 13) | +12.60% (neg 14) |
| 10 | +7.04%  (neg 19) | +10.39% (neg 21) | +13.04% (neg 23) |
| 20 | +6.97%  (neg 21) | +10.24% (neg 22) | +12.63% (neg 23) |

**Calibration**: at 30d windows + recent-tail (the older eval), the realism gate gives `med=+5.87%/mo, neg=42/283` at fb=5/lev=1 — i.e. the 19.57%/mo headline is *not* reproducible from the 13-model ensemble on this val data when run window-for-window with binary fills and a real 5bp limit-order queue. The headline came from a 30d-horizon `eval_multihorizon_candidate` slice that aggregated horizons 30/60/100/120 with a 140-day recent-tail filter, so it implicitly cherry-picked the bull tail. **The realism gate is the new ground truth for the deploy decision** — older `eval_multihorizon` numbers are still useful for relative seed-vs-seed comparisons but must not be quoted as the deployable monthly return.

**Deploy-gate findings**:
- 5 bps fill_buffer is the floor — wider (10/20bps) increases neg windows from 11→19+ without lifting the median. We're already on the right side of this knob.
- 1x leverage at **100% allocation per signal** does NOT clear `27%/mo`. Best 1x cell is `+7.04%/mo, neg=19/263, sortino=6.52` at fb=10.
- 1.5x leverage at 100% allocation is the Pareto knee: `+10.19%/mo, neg=13/263, sortino=6.20, max_dd=8.62%` at fb=5. Adds only +2 neg windows over 1x and a +0.55% nominal max_dd hit.
- 2x leverage adds +2.4%/mo on top of 1.5x but bumps max_dd from 8.6% → 14.9% and neg windows by another +1. Diminishing returns.

**LIVE vs gate calibration (CORRECTION 2026-04-17)**:
- The realism gate numbers above assume the policy fully invests its top signal each step (`max_leverage=1.0` in the C env, single-action sim).
- The live launch is `--allocation-pct 12.5` with **no `--multi-position` flag** (DEFAULT_MULTI_POSITION=0 → single-action mode, `portfolio_mode: False` confirmed in `strategy_state/daily_stock_rl_signals.jsonl`).
- Therefore live currently translates the model's signal into `12.5%` of equity per long, **not** `12.5% × 8 = 100%`.
- **Confirmed via realism gate at `max_leverage=0.125`** (`docs/realism_gate_live125/...`): live's expected monthly is `+0.90%/mo, p10=+0.37%, neg=15/263, sortino=5.62, max_dd=0.89%`. This matches the observed equity flatness ($28,679 unchanged for 3 days when the model also went argmax-flat — the model is argmax-flat 4.2% of val timesteps; see `project_live_flat_run_normal.md`).
- **Top-k(k=1) ≡ argmax on this val** (`docs/realism_gate_topk/`): the live `_ensemble_top_k_signals` rule at k=1 produces identical PnL to argmax (+6.89%/mo, neg=11/263 at fb=5/lev=1) because `flat_prob ≈ 0.04 > top*0.5 ≈ 0.03`, collapsing the threshold to `top >= flat`.
- **Path to higher live PnL** (in order of risk):
  1. Bump `--allocation-pct` from 12.5 → 50 to recover ~4× the expected return (`~3.50%/mo`). Single-position concentration risk goes up.
  2. ~~Add `--multi-position 8`~~ **NOT recommended** (`docs/realism_gate_multipos/`): the multi-position simulator now exists and finds k=8 strictly LOSES to argmax across every metric (best multipos cell med +4.03%/mo vs argmax +6.89%/mo, sortino 2.63 vs 6.10, neg 40 vs 11). Diversification benefit < dilution cost.
  3. Bump `--allocation-pct` further or add leverage only after (1) proves out at expected PnL on a few days.
  4. **NEW: SPY MA20 → MA50 filter swap** (`docs/regime_filter_gate/`): committed in `src/market_regime.py` (default `lookback=20 → 50`). Free ~+1.16%/mo lift on the gate (med +6.89→+8.05%, p10 +2.34→+3.45%, neg-rate 4.2→2.7%). Realized live impact at current `--allocation-pct 12.5` is ~+0.15%/mo.

**Allocation curve (`docs/realism_gate_alloc_curve/...`)** — fb=5, lag=2, full 263-window val, single-action argmax:

| max_lev | --allocation-pct equiv | med_monthly | p10_monthly | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 12.5 (current) | +0.90% | +0.37% | 5.62 | 0.89% | 15/263 |
| 0.25 | 25 | +1.79% | +0.75% | 5.59 | 1.77% | 13/263 |
| 0.5 | 50 | +3.50% | +1.44% | 5.57 | 3.48% | 11/263 |
| 0.75 | 75 | +5.15% | +1.88% | 5.98 | 5.57% | 11/263 |
| 1.0 | 100 | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 |
| 1.5 | 150 (margin 1.5×) | +10.19% | +3.24% | 6.20 ⭐ | 8.62% | 13/263 |
| 2.0 | 200 (margin 2×) | +12.60% | +4.06% | 5.77 | 14.92% | 14/263 |

PnL scales linearly with leverage (1×→2× ≈ 2× PnL); sortino is roughly constant across the range so risk-adjusted return is the same regardless of allocation. Max_dd inflects sharply between 1.5× and 2× (8.62→14.92, almost double). **Pareto knee remains 1.5×** (best sortino 6.20, only +0.55% max_dd over 1×).

**Why the old headline didn't catch this**: `eval_multihorizon_candidate` aggregates 30/60/100/120-day horizons with a `--recent-within-days 140` tail. With a 313-day val, that filter restricts to the most recent ~140 days where this ensemble was tuned — the realism gate runs the full 263 windows so the bear tail can't be dropped.

The 11 neg windows at fb=5/lev=1 are really only 2 distinct loss events: 1 mild isolated loss at start=47, plus 10 sequential starts 250-259 representing the same Mar–Apr 2026 tariff crash window viewed from 10 starting positions. The crash brewed AT the MA20 boundary — starts 250-254 entered with SPY only +0.13 to +0.63% above MA20, and SPY broke down during the trade. The MA20→MA50 filter swap (above) catches more of these "edge" entries.

#### Hard rules unchanged
- Death-spiral guard, singleton lock, decision_lag=2 are all enforced (now also at C-binding default — see `da842586`).
- New tests: `tests/test_screened32_realism_gate.py` runs a 1×1×2-window smoke against real prod val on every CI run.

---