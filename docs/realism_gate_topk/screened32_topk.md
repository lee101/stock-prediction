# Top-K decision rule realism gate — `screened32_single_offset_val_full.bin`

`scripts/screened32_realism_gate_topk.py` runs the prod 13-model v5 ensemble
through `simulate_daily_policy` with the live `_ensemble_top_k_signals`
decision rule (`top_sym >= max(flat, top*ratio)`) instead of deterministic
argmax, sweeping `min_prob_ratio` ∈ {0.3, 0.5, 0.7, 1.0}. Same 263-window
sweep at lag=2, fb=5, lev=1, fee=10bps, slip=5bps, shorts disabled.

## Result — top-k(k=1) ≡ argmax on this val

| min_prob_ratio | median_monthly | p10_monthly | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|
| 0.30 | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 |
| 0.50 | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 |
| 0.70 | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 |
| 1.00 | +6.89% | +2.34% | 6.10 | 6.28% | 11/263 |

All ratios produce identical results, and they match the argmax baseline
exactly (`docs/realism_gate/.../fb=5,lev=1` row: `+6.89%/mo, neg=11/263`).

## Why all rows collapse

In this regime:
- `flat_prob` ≈ 0.04 (val median, see `project_live_flat_run_normal.md`)
- `top_prob` ≈ 0.06
- `top*ratio` ≤ `top*1.0` ≈ 0.06 — but `flat ≈ 0.04 > top*0.5 ≈ 0.03`

Therefore `threshold = max(flat, top*ratio) = flat` for every ratio < ~0.7,
collapsing the rule to `top_sym >= flat` — which is exactly `argmax(probs)
== top_sym_idx` (since per_sym=1, each symbol is one bin and argmax across
the 65 actions picks either flat or the single largest sym bin).

For `ratio=1.0`, the threshold becomes `max(flat, top)` = `top`, so the
condition `top >= top` is always true — and the rule still picks the top
symbol. Same outcome.

## Implication for live deploy

The `--multi-position N` flag's value is in `N>1` (true portfolio mode), not
in `k=1` with a different threshold. The `_ensemble_top_k_signals` rule at
k=1 is functionally identical to `_ensemble_softmax_signal` on this val —
both produce `+6.89%/mo` at 100% allocation, fb=5, lev=1.

**To beat argmax, we need true portfolio mode** (k>1) which splits equity
across the top-k signals via the trading server's portfolio rebalance path.
That bypasses the C env's single-action sim entirely, so we'd need a
separate Python multi-position simulator (or reuse the live trading-server
backtest path with synthetic prices) to estimate its PnL.

Until that's built: live's expected PnL at `--allocation-pct 12.5` (single
position) is `12.5% × 6.89%/mo ≈ 0.86%/mo`, which matches the observed
flat-equity behavior more closely than the realism gate's headline.
