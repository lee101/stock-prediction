# Alpaca Migration Experiment 4 Progress

Tracking Alpaca crypto-only hourly experiments with CRYPTO_TRADING_FEE applied end-to-end.

## Current setup

- Training pipeline: `alpacamigrationexperiment4/`
- Default crypto fee: `loss_utils.CRYPTO_TRADING_FEE`
- Simulator supports intrabar round trips via `allow_intrabar_round_trips`
- Multi-asset selector available via `run_multiasset_selector.py`

## Results

| Timestamp (UTC) | Run | Symbol(s) | Eval window | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355 | SOLUSD | full val | 0.1019 | 2.3323 | 4 epochs, CRYPTO_TRADING_FEE, cache-only |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_eval20 | SOLUSD | 20d | -0.0004 | 0.1859 | baseline eval window |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar | SOLUSD | 20d | 0.0522 | 5.9735 | intrabar cycles (max=3) |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_best | SOLUSD | 20d | 0.0681 | 7.2754 | intrabar max=3, intensity=1.2, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_best_rt5 | SOLUSD | 20d | 0.0718 | 7.6836 | intrabar max=5, intensity=1.2, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_best_rt8 | SOLUSD | 20d | 0.0766 | 8.1433 | intrabar max=8, intensity=1.2, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_best_rt12 | SOLUSD | 20d | 0.0766 | 8.1433 | intrabar max=12, intensity=1.2, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_rt8_i14 | SOLUSD | 20d | 0.0795 | 8.4736 | intrabar max=8, intensity=1.4, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_rt8_i14_eval30 | SOLUSD | 30d | 0.0946 | 6.6566 | intrabar max=8, intensity=1.4, offset=0.0002 |
| 2026-02-04 11:23 | alpaca_mig4_solusd_20260204_112355_intrabar_rt12_i14_eval30 | SOLUSD | 30d | 0.0946 | 6.6566 | intrabar max=12, intensity=1.4, offset=0.0002 |
| 2026-02-04 11:42 | alpaca_mig4_solusd_6e_20260204_114240 | SOLUSD | full val | 0.2529 | 7.3452 | 6 epochs, CRYPTO_TRADING_FEE, cache-only |
| 2026-02-04 11:42 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt8_i14_eval20 | SOLUSD | 20d | 0.1170 | 20.6102 | intrabar max=8, intensity=1.4, offset=0.0002 |
| 2026-02-04 11:42 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt8_i14_eval30 | SOLUSD | 30d | 0.0985 | 6.7301 | intrabar max=8, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:12 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i14_eval20 | SOLUSD | 20d | 0.1185 | 20.7736 | intrabar max=10, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:12 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt16_i14_eval20 | SOLUSD | 20d | 0.1185 | 20.7736 | intrabar max=16, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:12 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i14_eval45 | SOLUSD | 45d | 0.1200 | 5.6112 | intrabar max=10, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:15 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i16_o0002_eval20 | SOLUSD | 20d | 0.0978 | 17.0318 | intrabar max=10, intensity=1.6, offset=0.0002 |
| 2026-02-04 12:15 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i16_o0004_eval20 | SOLUSD | 20d | 0.0522 | 9.4842 | intrabar max=10, intensity=1.6, offset=0.0004 |
| 2026-02-04 12:15 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i18_o0002_eval20 | SOLUSD | 20d | 0.0998 | 17.2041 | intrabar max=10, intensity=1.8, offset=0.0002 |
| 2026-02-04 12:15 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i14_o0004_eval20 | SOLUSD | 20d | 0.0275 | 5.1416 | intrabar max=10, intensity=1.4, offset=0.0004 |
| 2026-02-04 12:18 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i14_eval60 | SOLUSD | 60d | 0.2152 | 7.6422 | intrabar max=10, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:18 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt10_i14_eval90 | SOLUSD | 90d | 0.2491 | 7.5298 | intrabar max=10, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:18 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt20_i14_eval20 | SOLUSD | 20d | 0.1185 | 20.7736 | intrabar max=20, intensity=1.4, offset=0.0002 |
| 2026-02-04 12:18 | alpaca_mig4_solusd_6e_20260204_114240_intrabar_rt24_i14_eval20 | SOLUSD | 20d | 0.1185 | 20.7736 | intrabar max=24, intensity=1.4, offset=0.0002 |
| 2026-02-04 11:55 | alpaca_mig4_solusd_8e_20260204_115546 | SOLUSD | full val | 0.1403 | 2.4949 | 8 epochs, CRYPTO_TRADING_FEE, cache-only |
| 2026-02-04 11:55 | alpaca_mig4_solusd_8e_20260204_115546_intrabar_rt8_i14_eval20 | SOLUSD | 20d | 0.0810 | 8.3224 | intrabar max=8, intensity=1.4, offset=0.0002 |

### Sweep highlights (20d, intrabar max=3)

- Best return: intensity=1.20 offset=0.00020 → total_return=0.0681, sortino=7.2754
- Runner: `python -m alpacamigrationexperiment4.sweep --checkpoint binanceneural/checkpoints/alpaca_mig4_solusd_20260204_112355/epoch_002.pt --symbol SOLUSD --sequence-length 96 --cache-only --intensity 0.8 1.0 1.2 1.4 --offset 0.0 0.0002 --eval-days 20 --allow-intrabar-round-trips --max-round-trips-per-bar 3`

### Sweep highlights (20d, intensity/offset fixed, max round trips)

- Best return: max_round_trips_per_bar=8 (tie with 12) → total_return=0.0766, sortino=8.1433

### Sweep highlights (20d, max_round_trips=8, offset=0.0002)

- Best return: intensity=1.40 → total_return=0.0795, sortino=8.4736

### Sweep highlights (20d, 6-epoch checkpoint, max_round_trips=8, offset=0.0002)

- Best return: intensity=1.40 → total_return=0.1170, sortino=20.6102

### Sweep highlights (20d, 6-epoch checkpoint, intensity=1.4, offset=0.0002)

- Best return: max_round_trips_per_bar=10/16 → total_return=0.1185, sortino=20.7736

## Artifact sync

```bash
aws s3 sync alpacamigrationexperiment4/ s3://models/stock/models/alpaca/migrationexperiment4/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync binanceneural/checkpoints/alpaca_mig4_solusd_20260204_112355/ s3://models/stock/models/alpaca/migrationexperiment4/checkpoints/alpaca_mig4_solusd_20260204_112355/ --endpoint-url "$R2_ENDPOINT"
```
