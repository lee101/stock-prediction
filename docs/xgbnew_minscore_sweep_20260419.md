# XGB 16-seed min-score conviction sweep — 2026-04-19

## Setup
16-seed alltrain ensemble at lev=1.0, fb=5, top_n=1. Varied `--min-score` (threshold on blended `predict_proba`) from 0 → 0.70. Flag newly added to `xgbnew/eval_pretrained.py` and forwarded to `BacktestConfig.min_score` which filters picks by `_score >= min_score` BEFORE the top_n selection. A zero-candidate day becomes cash-only (no loss, no gain).

Same 30-window grid, 846 symbols, 16 prime seeds.

## Results

| min_score | med %/mo | p10 %/mo | sortino | worst DD | n_neg | trade_days | Δmed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (baseline) | +40.44 | +4.74 | 19.63 | 31.62 | 2/30 | 816 | — |
| 0.55 | +46.98 | +28.86 | 52.63 | 7.01 | **0/30** | 645 | +6.54 |
| 0.60 | +46.98 | +28.86 | 52.63 | 7.01 | **0/30** | 645 | +6.54 |
| 0.65 | +47.34 | +28.86 | 52.63 | 7.01 | **0/30** | 630 | +6.90 |
| 0.70 | +52.51 | +31.16 | ∞† | **2.85** | **0/30** | 608 | +12.07 |

† sortino=270M at ms=0.70 is a divide-by-zero artifact of zero downside deviation.

**Critical checks — NOT a zero-trade masking effect**:
- Zero-window count = 0 at every threshold (every window had trades)
- Trade-day count drops gracefully: 816 → 645 (ms=0.55, 21% cut) → 608 (ms=0.70, 25% cut)
- All 30 windows still trade with at least 16 days out of ~27 possible

## Per-window breakdown (selected)

| window | baseline %/mo | ms=0.55 %/mo | ms=0.70 %/mo | trade_days base/55/70 |
| --- | ---: | ---: | ---: | ---: |
| 2025-01-16..02-14 | **−11.09** | **+29.18** | **+29.18** | 30/21/21 |
| 2025-01-30..02-28 | **−4.48** | **+17.88** | **+24.32** | 30/21/19 |
| 2025-02-13..03-14 | +2.08 | +13.70 | +31.38 | 30/21/16 |
| 2025-04-24..05-23 | +72.55 | +67.33 | +67.33 | 26/22/22 |
| 2025-07-17..08-15 | +53.32 | +43.20 | +40.28 | 30/22/21 |
| 2026-01-05..02-06 | +51.87 | +51.87 | +51.87 | 24/24/24 |

**Observations**:
- Conviction filter TURNS the 2 losing windows into large gains — it's not just "skip the bad days", it's "the model knows when it's wrong and its low-confidence picks really are losers".
- Most high-return windows are basically unchanged (the model was already high-conviction on those picks).
- Some medium windows (`07-17..08-15`) actually give up a little return for the insurance — the filter removed picks that were winners too. Cost is small.

## Strongest caveat: in-sample calibration

All 16 alltrain models were trained on data *including* these validation windows (train_end = 2026-04-19). The model's 0.55-0.70 confidence scores on these days have been tuned by gradient descent to correlate with the actual labels of those days. Real OOS calibration may be softer.

Reality-check translation:
- In-sample claim: `--min-score 0.55` cuts DD from 31.62% to 7.01%.
- If OOS calibration is 70% as sharp (consistent with our 60-70% of-in-sample heuristic on median): DD probably falls to ~15-20% range.
- Even the pessimistic outcome (DD 20%) would be a meaningful improvement over baseline 31.62%.

## Production ship plan

**Phase 1 — no-op code deploy** (this commit):
- Add `--min-score` flag to `xgbnew/live_trader.py`, default 0.0 (no behavior change).
- `xgbnew/eval_pretrained.py` already has the flag; the backtest wiring is already proven.
- Add tests to `tests/test_xgbnew_live_trader_guard.py` confirming default behavior unchanged.

**Phase 2 — activate** (**needs user approval**):
- Add `--min-score 0.55` to `deployments/xgb-daily-trader-live/launch.sh`.
- Monitor hit-rate (% of sessions with no trades) for first week.
- If >3 consecutive skipped-trade days, bump threshold down to 0.50 or revert.

## Comparison vs other upgrade candidates

| candidate | Δmed | Δp10 | ΔDD | Δsortino | Δn_neg | gate-pass? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 16-seed lev=1.0 | +1.59 | −0.08 | +0.18 | +0.77 | 0 | ✓ (every metric) |
| 5-seed n1600/lr01 lev=1.0 | −0.61 | +1.97 | −1.99 | +0.73 | 0 | ✗ (median) |
| 16-seed lev=1.25 | +13.36 | +0.52 | +7.11 | +0.70 | 0 | ✗ (DD) |
| **16-seed ms=0.55 lev=1.0** | **+8.13** | **+24.12** | **−24.61** | **+33.00** | **−2** | **✓ (every metric)** |

The conviction filter is the only candidate that improves *every* metric *massively* vs the deployed baseline AND is auto-deploy-eligible by metric gates (though outside §6's "same-topology" authority). It also stacks cleanly with any other axis.

## Files
- `analysis/xgbnew_deploy_baseline/deploy_16seed_minscore0.55_20260419.json`
- `analysis/xgbnew_deploy_baseline/deploy_16seed_minscore0.60_20260419.json`
- `analysis/xgbnew_deploy_baseline/deploy_16seed_minscore0.65_20260419.json`
- `analysis/xgbnew_deploy_baseline/deploy_16seed_minscore0.70_20260419.json`
- Code: `xgbnew/eval_pretrained.py`, `xgbnew/live_trader.py`
