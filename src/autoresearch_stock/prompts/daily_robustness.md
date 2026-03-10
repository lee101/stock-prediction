Focus on daily robustness, not hourly churn tricks.

Priority:

- improve daily `robust_score` across many windows,
- preserve realistic execution assumptions,
- prefer fewer, higher-quality decisions.

What usually matters more on daily:

- stable checkpoint selection,
- better calibration of action confidence,
- regime robustness,
- smoother portfolio behavior,
- reducing drawdown and noisy flips.

Useful diagnostics:

- compare daily target asymmetry vs hourly target asymmetry,
- inspect trade counts and opportunity rates,
- check whether the current ambiguity threshold is too loose or too aggressive,
- look for overfitting to planner loss instead of trading outcome.

Good idea directions:

- stronger flat/no-trade behavior for weak daily edges,
- validation criteria closer to trading utility,
- smoother action targets,
- multi-horizon consistency penalties using existing feature blocks.
