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

### Sweep highlights (20d, intrabar max=3)

- Best return: intensity=1.20 offset=0.00020 → total_return=0.0681, sortino=7.2754
- Runner: `python -m alpacamigrationexperiment4.sweep --checkpoint binanceneural/checkpoints/alpaca_mig4_solusd_20260204_112355/epoch_002.pt --symbol SOLUSD --sequence-length 96 --cache-only --intensity 0.8 1.0 1.2 1.4 --offset 0.0 0.0002 --eval-days 20 --allow-intrabar-round-trips --max-round-trips-per-bar 3`

## Artifact sync

```bash
aws s3 sync alpacamigrationexperiment4/ s3://models/stock/models/alpaca/migrationexperiment4/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync binanceneural/checkpoints/alpaca_mig4_solusd_20260204_112355/ s3://models/stock/models/alpaca/migrationexperiment4/checkpoints/alpaca_mig4_solusd_20260204_112355/ --endpoint-url "$R2_ENDPOINT"
```
