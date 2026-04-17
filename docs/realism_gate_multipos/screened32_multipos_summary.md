# Multi-position k=8 portfolio realism gate — `screened32_single_offset_val_full.bin`

`scripts/screened32_realism_gate_multipos.py` runs the prod 13-model v5
ensemble through a top-K portfolio simulator: pick top-k symbols by
ensemble probability, equal-weight by normalized prob, rebalance daily at
close, decision_lag=2, fee=10bps + 5bps slip on rebalance deltas.

The policy obs uses `cash/scale = 1.0` and one-hot zeros (presents as
"no current single position", since the C env has no multi-position obs
analogue). Features at `t_obs = t-1` to match C env's 1-bar lag.

## Results — k=8, total_alloc=1.0, decision_lag=2

| min_prob_ratio | med_monthly | p10_monthly | sortino | max_dd | n_neg |
|---:|---:|---:|---:|---:|---:|
| 0.50 | +1.45% | −1.21% | 1.53 | 6.38% | 68/262 |
| 0.70 | +2.54% | −0.06% | 2.20 | 7.65% | 28/262 |
| 0.85 | +3.76% | −0.57% | 2.58 | 9.80% | 39/262 |
| 0.95 | +4.03% | −0.80% | 2.63 | 9.73% | 40/262 |
| **argmax (single-action baseline)** | **+6.89%** | **+2.34%** | **6.10** | **6.28%** | **11/263** |

## Conclusion: don't deploy `--multi-position 8`

Single-action argmax beats every multipos cell on every metric:
- median_monthly: argmax wins by 71%+ (+6.89% vs +4.03% best multipos)
- p10_monthly: argmax POSITIVE (+2.34%), every multipos NEGATIVE
- sortino: argmax 6.10 vs best multipos 2.63
- max_dd: argmax 6.28% vs best multipos 9.73%
- n_neg: argmax 11/263 vs best multipos 28/262

## Why concentration helps but doesn't close the gap

As `min_prob_ratio` increases (0.5 → 0.95), fewer symbols pass the
threshold, so the portfolio collapses toward 1-2 high-conviction names.
But multipos still rebalances daily across multiple symbols, paying
fees on every weight delta — and the diversification benefit (lower
variance) doesn't compensate for the dilution of the policy's strong
single-symbol signal.

Argmax fully concentrates equity on the single highest-prob symbol
every day. The policy was trained to emit one action per step, so its
top signal is highly informative; spreading equity across the 2nd-8th
ranked symbols dilutes that signal with weaker ones.

## What this means for live deploy

- **Keep single-position mode** (current `daily-rl-trader` config: no
  `--multi-position` flag → `_ensemble_softmax_signal` argmax path)
- **Path to higher PnL is `--allocation-pct`, not multi-position.**
  Live currently at `--allocation-pct 12.5` ≈ realism-gate `lev=0.125`
  cell = +0.90%/mo. Bumping toward `--allocation-pct 100` would give
  +6.89%/mo (8× current) at the cost of 8× position size on each trade.

## Caveats

- The simulator is approximate: real rebalance fees may differ from the
  bps-on-delta model used here, especially for limit-order fills.
- The policy obs has cash/scale=1.0 (clean cash, no positions) which
  doesn't match what the policy would see if we maintained 8 partial
  positions. This is a fundamental mismatch — the C env never trained
  the policy on multi-position observations — but it means the multipos
  results may be slightly biased upward (the policy thinks it's choosing
  fresh positions each day when in reality it has cumulative holdings).
- Even with these caveats, the directional conclusion is firm: **k=8
  dilution beats argmax concentration** is FALSE for this ensemble on
  this val.

## Reproduce

```bash
source .venv/bin/activate
for ratio in 0.5 0.7 0.85 0.95; do
    python scripts/screened32_realism_gate_multipos.py \
        --k 8 --total-alloc 1.0 --window-days 50 \
        --min-prob-ratio $ratio \
        --out-json docs/realism_gate_multipos/k8_alloc1_w50_r${ratio}.json
done
```
