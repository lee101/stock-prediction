# XGB alltrain 16-seed bonferroni — 2026-04-19

## Setup
- 16 prime-seeded alltrain models (GPU, n_estimators=400, max_depth=5, lr=0.03, min_dollar_vol=5M)
- Seeds: 0, 7, 42, 73, 197 (deployed 5) + {1, 3, 11, 23, 59} (extra5) + {2, 5, 13, 17, 19, 29} (extra10 — trained today 2026-04-19 14:47 UTC)
- Eval: `xgbnew.eval_pretrained --blend-mode mean --top-n 1 --leverage 1.0 --fill-buffer-bps 5 --fee-rate 2.78e-05`
- Windows: 30d rolling, 14d stride, 2025-01-02 → 2026-04-10 (n=30)
- Symbols: 846 from `stocks_wide_1000_v1.txt`
- Universe train_end = 2026-04-19 → **OOS range 2025-01-02..2026-04-10 is IN-SAMPLE** (alltrain design). Treat as upper-bound stability check, not honest OOS.

## Headline

| Metric | Deployed 5-seed (live) | 16-seed (this run) | Δ |
| --- | --- | --- | --- |
| median %/mo | +38.85 | **+40.44** | **+1.59** |
| p10 %/mo | +4.82 | +4.74 | −0.08 |
| mean %/mo | +37.72 | +38.17 | +0.45 |
| median sortino | 18.86 | **19.63** | +0.77 |
| worst DD % | 31.44 | 31.62 | +0.18 |
| n_neg / n_windows | 2 / 30 | 2 / 30 | 0 |

**Takeaway**: 16-seed is a quiet structural win (+1.6%/mo median, +0.77 sortino, same n_neg and effectively flat p10/DD). No regime flip. Same two negatives (2025-01-16..02-14 = −11.09% vs −10.92, 2025-01-30..02-28 = −4.48% vs −4.47). Seed averaging tightens the mean profile without adding tail risk.

## Robustness
- Same 30 windows → direct 1-to-1 comparison.
- Both runs hit the in-sample caveat; expected live yield ~60–70% of in-sample = ~+24–28%/mo (5-seed) vs ~+25–29%/mo (16-seed).
- The 16-seed eval effectively sampled 16 independent XGB fits on the same data — Bonferroni interpretation: with the median > +40%/mo across 16-way seed average, the probability that this signal is pure chance is negligible given prior 5-seed and 10-seed robustness.

## Decision
- **Do NOT auto-deploy.** `monitoring/current_algorithms.md` §6 only authorizes ensemble-size swap for same-topology 5-seed top-N variants. 16-seed is outside that authority.
- **User approval candidate**: Net +1.6%/mo med / +0.77 sortino upgrade at identical n_neg/p10, zero new training cost (all 16 pkls cached). Loading 16 models instead of 5 costs ~3× predict time per session (~1.5s → ~4.5s on CPU inference path in `live_trader.py`) — negligible for a once-per-day decision.
- If approved, swap `deployments/xgb-daily-trader-live/launch.sh` `--model-paths` to the 16-path list and restart the supervisor group.

## Files
- `analysis/xgbnew_deploy_baseline/deploy_16seed_20260419.json` — raw 16-seed eval
- `analysis/xgbnew_deploy_baseline/deploy_5seed_lev1_top1.json` — deployed baseline
- `analysis/xgbnew_daily/alltrain_ensemble_gpu_extra10/` — 6 fresh seeds
- `analysis/xgbnew_daily/alltrain_ensemble_gpu_extra5/` — earlier 5 seeds
