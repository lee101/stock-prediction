# 30-Day RL Market Validation

Date run: 2026-03-14 UTC

This replay uses `python -m unified_orchestrator.market_validation` with the hourly Alpaca-style market simulator. It is stricter than the pure PPO environment because it includes:

- hourly decision lag
- working-order persistence and cancel/replace delay
- partial fills on touch
- reserved cash / shared portfolio sizing
- crypto max-hold exits (`6h`)
- crypto trailing stops (`0.3%`)

## Crypto

Window: 2026-01-07 15:00 UTC to 2026-02-06 15:00 UTC
Symbols: `BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD`

| Checkpoint | Return | Sortino | Max DD | Trades | Win Rate | Goodness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `longonly_forecast` | `-55.79%` | `-35.06` | `56.39%` | `425` | `39.3%` | `-91.45` |
| `slip_5bps` | `-57.52%` | `-33.32` | `58.01%` | `583` | `37.2%` | `-92.43` |
| `ent_01` | `-62.21%` | `-36.18` | `62.72%` | `675` | `30.8%` | `-100.23` |
| `reg_combo_2` | `-63.82%` | `-35.83` | `64.21%` | `688` | `37.8%` | `-101.79` |

Interpretation:

- All current crypto RL candidates fail under the more realistic market replay.
- `longonly_forecast` remains the least bad candidate, but it is still nowhere near deployable as a standalone allocator.
- The gap versus earlier PPO/OOS comments means the deployable order-construction layer still matters a lot; RL direction alone does not survive realistic fills/cancels/risk exits.

## Stocks

Window: 2026-01-23 21:00 UTC to 2026-02-23 21:00 UTC
Symbols: `NVDA PLTR META MSFT NET`

Note: stock replay is conservative on leverage (`2x`) because the validator does not yet model the live intraday `4x -> 2x` deleveraging schedule.

| Checkpoint | Return | Sortino | Max DD | Trades | Win Rate | Goodness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stocks13_featlag1_fee5bps_longonly_run4` | `-0.23%` | `-0.38` | `0.66%` | `5` | `60.0%` | `-0.22` |
| `stocks13_issuedat_featlag1_fee5bps_longonly_run5` | `-1.54%` | `-3.28` | `1.73%` | `21` | `61.9%` | `-4.12` |

Interpretation:

- The stock RL path is far healthier than crypto under the same replay discipline.
- `stocks13_featlag1_fee5bps_longonly_run4` is the current stock leader on this live-like validation, but it is basically flat over the measured window.
- The stock issue is not runtime loading anymore; it is edge quality and low activity.

## Next Work

1. Calibrate the deterministic RL -> limit-price planner against actual live fills so crypto replay is not forced to rely on fixed heuristics.
2. Compare these results against pure PPO tail replay on the matching validation bins to isolate policy edge from execution-loss.
3. If crypto remains this weak after calibration, keep RL as a hint layer only and continue using the stock checkpoint as the stronger RL baseline.
