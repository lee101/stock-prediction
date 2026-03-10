# PufferLib Best

## Current Best Stock Checkpoint

- Date: `2026-03-08`
- Checkpoint: `experiments/pufferlib4_push_pnl_20260308_shortcarry_followup/cont_lr1e4_same_reward_5m/best.pt`
- Manifest: `experiments/pufferlib4_push_pnl_20260308_shortcarry_followup/promotion_manifest.json`
- Training regime: resumed stock PPO from `update_003450.pt`, shorts enabled, `max_leverage=1.0`, `short_borrow_apr=0.0625`
- Evaluation regime: deterministic holdout, `720` market hours, `25` windows, annualized with `252 * 6.5 = 1638` stock market hours/year

### Holdout Summary

| Metric | Full Holdout | Recent Holdout |
| --- | ---: | ---: |
| p10 total return | `1.7997` | `1.3648` |
| median total return | `2.3163` | `2.6866` |
| p10 annualized return | `9.4036` | `6.0859` |
| median annualized return | `14.2932` | `18.4568` |
| median max drawdown | `0.0131` | `0.0132` |

## Previous Short-Enabled Reference

- Checkpoint: `experiments/pufferlib4_push_pnl_20260308_shortcarry/short1x_lr1e4_same_reward_10m/best.pt`
- Full holdout: `p10_total_return=1.6704`, `median_total_return=2.2361`
- Recent holdout: `p10_total_return=1.2835`, `median_total_return=2.5250`
- Full annualized: `p10=8.3427`, `median=13.4642`
- Recent annualized: `p10=5.5454`, `median=16.5709`
- Saved-checkpoint frontier winner inside that run: `update_003450.pt`

## Previous Long-Only Reference

- Checkpoint: `experiments/pufferlib4_push_pnl_20260308/resume_lr1e4_dp125_tp0005_10m/best.pt`
- Full holdout: `p10_total_return=1.2584`, `median_total_return=1.5727`
- Recent holdout: `p10_total_return=1.0105`, `median_total_return=1.8250`
- Full annualized: `p10=5.3814`, `median=7.5828`
- Recent annualized: `p10=3.8981`, `median=9.6183`

## Notes

- `pufferlib_market` now prints annualized return in training and holdout outputs.
- The stock simulator now models `short_borrow_apr` and no longer silently caps leveraged long entries back to cash in the C env.
- The strongest path in the latest pass was: re-score saved checkpoints, promote `update_003450.pt` as the true holdout leader, then continue that checkpoint for another `5M` steps with the same short-enabled reward settings.
