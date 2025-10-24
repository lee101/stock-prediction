# GymRL Regime Guard A/B Results (Loss-Probe v11)

**Setup**
- Checkpoint: `gymrl/artifacts/sweep_20251025_lossprobe_v11/ppo_allocator_final_pnlpctp10.69_dailyp0.51_annualp185.79_logp0.0005.zip`
- Feature cache: `gymrl/cache/features_tototraining_resampled_1H_top5.npz`
- Window: trailing 42 steps (start index 3 781 of resampled 1H cube)

| Variant | Cumulative Return | Avg Turnover | Max Drawdown | Avg Turnover Penalty | Guard Hit Rates (drawdown / negative / turnover) |
| --- | --- | --- | --- | --- | --- |
| Baseline (guards off) | −4.16 % | 0.369 | 0.042 | 0.00690 | 0 / 0 / 0 |
| Guards – initial thresholds (18 %, ≤ 0, 0.50) | −4.86 % | 0.370 | 0.049 | 0.00749 | 0 / 97.6 % / 23.8 % |
| Guards – calibrated (3.6 %, ≤ −3 %, 0.55) | −4.21 % | 0.361 | 0.042 | 0.00710 | 19.0 % / 33.3 % / 9.5 % |

**Key Observations**
- The initial guard settings (drawdown 18 %, negative-return ≤ 0, turnover 0.50) over-fired: negative guard triggered on 97.6 % of steps with little benefit, deepening losses while raising the turnover penalty.
- Calibrated thresholds derived from the same window’s quantiles (drawdown ≈90th percentile at 3.6 %, trailing return 10th percentile ≈ −3 %, turnover 90th percentile ≈ 0.55) cut guard hit-rates to more selective bands (19 % / 33 % / 9.5 %) and reduced turnover by ~0.008 while keeping cumulative return within 5 bps of baseline.
- The calibrated guard trims average gross leverage to 0.48× (−3.3 p.p.), demonstrating actual leverage throttling; minimum leverage scale hit 0.6 while average scale stayed near 0.92.
- Cross-check on an earlier hold-out slice (start index 3 600) shows the calibrated guard behaving benignly: cumulative return stays flat (+1.01 % → +1.00 %), turnover impact is negligible (+0.0005), and only the turnover guard fires (16.7 % hit rate). This indicates the tuned thresholds avoid unnecessary throttling in benign regimes.
- A third slice (start index 3 300) shows similar behaviour to the neutral window: turnover/return deltas are sub-basis-point, with only the turnover guard activating (23.8 % hit rate) while leverage and turnover shifts remain at the fourth decimal place.

**Per-Window Summary (Baseline vs Calibrated Guard)**

| Window (start index) | Baseline Cum. Return | Guard Cum. Return | Baseline Avg Turnover | Guard Avg Turnover | Guard Hit Rates (drawdown / negative / turnover) |
| --- | --- | --- | --- | --- | --- |
| 3781 (stress) | −4.16 % | −4.21 % | 0.369 | 0.361 | 19.0 % / 33.3 % / 9.5 % |
| 3600 (neutral) | +1.01 % | +1.00 % | 0.356 | 0.357 | 0 / 0 / 16.7 % |
| 3300 (neutral) | −1.51 % | −1.52 % | 0.391 | 0.391 | 0 / 0 / 23.8 % |

### Latest Validation (guard confirmation)

- **Run**: `sweep_20251026_guard_confirm` (validation window)  
  - Cumulative return: +10.96 %  
  - Avg daily return: +0.00498  
  - Guard hit rates: turnover ~4.8 %, negative/drawdown 0 %  
  - Turnover vs v11: +0.0052 absolute (0.160 vs 0.155), leverage avg 0.68× (was 0.69×)

### Additional Hold-Out Windows (guard confirmation)

Sweep v12 evaluated across multiple 42-step windows on the resampled 1H top-5 cache:

| Start Index | Cum. Return | Avg Turnover | Guard Hits (neg / turn / draw) | Avg Leverage Scale |
| --- | --- | --- | --- | --- |
| 0 | −1.47 % | 0.389 | 0 % / 26.19 % / 16.67 % | 0.93 |
| 500 | −2.07 % | 0.360 | 7.14 % / 28.57 % / 0 % | 1.00 |
| 1000 | +5.08 % | 0.313 | 0 % / 19.05 % / 0 % | 1.00 |
| 1500 | +0.73 % | 0.324 | 0 % / 21.43 % / 0 % | 1.00 |
| 2000 | +1.49 % | 0.350 | 0 % / 21.43 % / 0 % | 1.00 |
| 2500 | +0.88 % | 0.409 | 0 % / 33.33 % / 0 % | 1.00 |
| 3000 | +8.57 % | 0.343 | 0 % / 19.05 % / 0 % | 1.00 |

**Next Steps**
1. Apply the calibrated thresholds in future GymRL sweeps (`--regime-drawdown-threshold 0.036 --regime-negative-return-threshold -0.03 --regime-turnover-threshold 0.55 --regime-turnover-probe-weight 0.002 --regime-leverage-scale 0.6`) and log guard hit rates inside `training_metadata.json`.
2. After the cooldown window, launch the PPO confirmation sweep with these guard settings and compare validation/hold-out deltas before promoting the policy.
