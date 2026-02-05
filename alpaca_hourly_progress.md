# Alpaca Hourly Experiments Progress

Tracking Alpaca hourly experiments (Chronos2-driven) for crypto + stocks.

For detailed run-by-run metrics, see `newnanoalpacahourlyexp/progress.md`.

## Current setup

- Training pipeline: `newnanoalpacahourlyexp/`
- Simulator enforces stock market hours + EOD flattening
- Fees: `src/fees.get_fee_for_symbol` (TRADING_FEE for stocks, CRYPTO_TRADING_FEE for crypto)

## Latest results

| Symbol | Run | total_return | sortino | Notes |
| --- | --- | --- | --- | --- |
| (pending) | | | | |

## Next planned sweeps

- Sweep attention_window around best nano settings for Alpaca data.
- Sweep value_embedding_every and skip_scale_init.
- Compare crypto-only multi-symbol selector vs per-symbol independent allocations.

## Artifact sync

```bash
aws s3 sync newnanoalpacahourlyexp/outputs/ s3://models/stock/models/alpaca/hourly/outputs/ --endpoint-url "$R2_ENDPOINT"
```
