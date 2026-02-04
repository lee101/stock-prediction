# Alpaca Hourly Experiments Progress

Tracking Alpaca hourly experiments (Chronos2-driven) for crypto + stocks.

## Current setup

- Training pipeline: `newnanoalpacahourlyexp/`
- Simulator enforces stock market hours + EOD flattening
- Fees: `src/fees.get_fee_for_symbol` (TRADING_FEE for stocks, CRYPTO_TRADING_FEE for crypto)
- Evaluation window: `--eval-days` / `--eval-hours`

## Results

| Timestamp (UTC) | Run | Symbol(s) | Eval window | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-04 01:48 | alpaca_solusd_20260204_013044 | SOLUSD | 10d | 0.6308 | 337.4409 | Val score improved each epoch; best val score=928.7657 (epoch 6). |

### Run details

- `alpaca_solusd_20260204_013044` (SOLUSD, 6 epochs)
  - Epoch 1: train_score=403.2413, train_sortino=402.2524, train_return=12.3622 | val_score=411.4803, val_sortino=410.9042, val_return=7.2008
  - Epoch 2: train_score=674.0388, train_sortino=672.8838, train_return=14.4373 | val_score=593.4017, val_sortino=592.4183, val_return=12.2917
  - Epoch 3: train_score=981.8684, train_sortino=980.5699, train_return=16.2317 | val_score=736.1913, val_sortino=735.2188, val_return=12.1564
  - Epoch 4: train_score=1210.5519, train_sortino=1209.1305, train_return=17.7667 | val_score=764.4767, val_sortino=763.5307, val_return=11.8255
  - Epoch 5: train_score=1479.6653, train_sortino=1478.0927, train_return=19.6575 | val_score=875.7024, val_sortino=874.6393, val_return=13.2895
  - Epoch 6: train_score=1729.2419, train_sortino=1727.5392, train_return=21.2837 | val_score=928.7657, val_sortino=927.6492, val_return=13.9556

## Planned sweeps

- Sweep attention_window around best nano settings for Alpaca data.
- Sweep value_embedding_every and skip_scale_init.
- Compare crypto-only multi-symbol selector vs per-symbol independent allocations.
