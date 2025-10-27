# GymRL Regime Filter Proposal

## Objective
Reduce exposure during regimes that historically produced the largest drawdowns or negative cumulative returns in hold-out evaluation, while keeping the current positive Sharpe configuration (loss-probe v13) intact.

## Data Inputs
- Hold-out summary (resampled 1H top-5): `evaltests/gymrl_holdout_summary_resampled_v11.json`, `evaltests/gymrl_holdout_summary_resampled_v11.md`
- Flagged regimes: `evaltests/gymrl_holdout_flags.csv`

## Suggested Filters
1. **Drawdown Guard**
   - Compute rolling max drawdown from the hold-out features.
   - Skip deployment (or reduce leverage by 50%) when expected drawdown exceeds the 90th percentile (~0.18 based on v11 windows).
2. **Negative Return Guard**
   - If cumulative return over the last 42 steps is negative (< 0), throttle turnover_penalty to 0.0075 and/or halve learning rate during live adaption.
3. **Turnover Spike Guard**
   - When expected turnover rises above the 90th percentile (~0.04), enforce stricter loss-shutdown (probe weight 0.0015).

## Implementation Notes
- Guards now live inside `PortfolioEnv` (via `RegimeGuard`) and can be toggled with `--regime-*` CLI flags; they are applied pre-trade so leverage scaling and turnover penalties propagate into reward shaping.
- For offline analysis, tag flagged windows in evaluation reports (see `evaltests/gymrl_holdout_flags.csv`).
- Consider simulating guard-enabled runs once pacing window allows new sweeps.

## Next Steps
1. Share guard proposal with stakeholders for feedback.
2. Schedule a short confirmation sweep (loss-probe v14) incorporating guard logic once cooldown expires.
