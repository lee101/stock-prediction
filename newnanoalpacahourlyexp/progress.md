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
| 2026-02-04 22:51 | alpaca_solusd_eval_20260204_225119 | SOLUSD | 10d | 0.7208 | 180.0594 | GPU eval (cu130), cache-only forecasts, checkpoint=alpaca_solusd_20260204_013044/epoch_006.pt. |
| 2026-02-04 22:51 | alpaca_linkusd_eval_20260204_225142 | LINKUSD | 10d | 0.6760 | 68.3498 | GPU eval (cu130), cache-only forecasts, checkpoint=alpaca_linkusd_20260204_111049/epoch_006.pt. |
| 2026-02-04 22:52 | alpaca_btcusd_eval_20260204_225212 | BTCUSD | 10d | 0.0316 | 17.2238 | GPU eval (cu130), cache-only forecasts, checkpoint=alpaca_mig4_btcusd_6e_20260204_122215/epoch_004.pt. |
| 2026-02-04 22:52 | alpaca_ethusd_eval_20260204_225241 | ETHUSD | 10d | 0.1159 | 15.5802 | GPU eval (cu130), cache-only forecasts, checkpoint=alpaca_mig4_ethusd_6e_20260204_123651/epoch_006.pt. |
| 2026-02-04 22:53 | alpaca_selector_crypto_20260204_225355 | BTCUSD,ETHUSD,SOLUSD,LINKUSD | 10d | 0.1746 | 15.7671 | GPU selector (cu130), cache-only forecasts; final_cash=11737.5574, open_symbol=BTCUSD. |
| 2026-02-04 22:55 | alpaca_selector_sol_link_20260204_225548 | SOLUSD,LINKUSD | 10d | 0.2428 | 22.6131 | GPU selector (cu130), cache-only forecasts; final_cash=12000.2073, open_symbol=LINKUSD. |
| 2026-02-04 23:19 | alpaca_uniusd_20260204_231926 | UNIUSD | 10d | 0.2618 | 42.3321 | GPU train+eval (cu130), cache-only (h1), best checkpoint=epoch_004.pt. |
| 2026-02-04 23:35 | alpaca_nvda_20260204_233529 | NVDA | 10d | 0.0135 | 164.9478 | GPU train+eval (cu130), cache-only (h1), best checkpoint=epoch_003.pt. |
| 2026-02-04 23:50 | alpaca_nflx_20260204_235030 | NFLX | 10d | 0.0185 | 43.9566 | GPU train+eval (cu130), cache-only (h1), best checkpoint=epoch_004.pt. |
| 2026-02-05 00:02 | alpaca_aapl_20260205_000242 | AAPL | 10d | 0.0012 | 720.8533 | GPU train+eval (cu130), generated h1 forecasts, best checkpoint=epoch_006.pt. |
| 2026-02-05 00:20 | alpaca_selector_sol_link_uni_20260205_002031 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.2459 | 16.9464 | GPU selector (cu130), cache-only, h1-only features (SOL/LINK evaluated without h24 deltas); final_cash=12551.6965, open_symbol=UNIUSD. |
| 2026-02-05 00:20 | alpaca_nvda_eval_20260205_002048 | NVDA | 10d | 0.0135 | 165.1457 | GPU eval (cu130), cache-only (h1), checkpoint=alpaca_nvda_20260204_233529/epoch_003.pt. |
| 2026-02-05 00:21 | alpaca_nflx_eval_20260205_002114 | NFLX | 10d | 0.0185 | 43.9518 | GPU eval (cu130), cache-only (h1), checkpoint=alpaca_nflx_20260204_235030/epoch_004.pt. |
| 2026-02-05 00:21 | alpaca_aapl_eval_20260205_002137 | AAPL | 10d | 0.0012 | 721.1418 | GPU eval (cu130), cache-only (h1), checkpoint=alpaca_aapl_20260205_000242/epoch_006.pt. |

### Run details

- `alpaca_solusd_20260204_013044` (SOLUSD, 6 epochs)
  - Epoch 1: train_score=403.2413, train_sortino=402.2524, train_return=12.3622 | val_score=411.4803, val_sortino=410.9042, val_return=7.2008
  - Epoch 2: train_score=674.0388, train_sortino=672.8838, train_return=14.4373 | val_score=593.4017, val_sortino=592.4183, val_return=12.2917
  - Epoch 3: train_score=981.8684, train_sortino=980.5699, train_return=16.2317 | val_score=736.1913, val_sortino=735.2188, val_return=12.1564
  - Epoch 4: train_score=1210.5519, train_sortino=1209.1305, train_return=17.7667 | val_score=764.4767, val_sortino=763.5307, val_return=11.8255
  - Epoch 5: train_score=1479.6653, train_sortino=1478.0927, train_return=19.6575 | val_score=875.7024, val_sortino=874.6393, val_return=13.2895
  - Epoch 6: train_score=1729.2419, train_sortino=1727.5392, train_return=21.2837 | val_score=928.7657, val_sortino=927.6492, val_return=13.9556
- `alpaca_linkusd_20260204_111049` (LINKUSD, 6 epochs)
  - Epoch 1: train_score=518.7328, train_sortino=517.5977, train_return=14.1884 | val_score=451.9708, val_sortino=451.1180, val_return=10.6601
  - Epoch 2: train_score=947.0956, train_sortino=945.6250, train_return=18.3830 | val_score=758.2278, val_sortino=757.3846, val_return=10.5404
  - Epoch 3: train_score=1469.7466, train_sortino=1467.9972, train_return=21.8673 | val_score=1197.2281, val_sortino=1196.0094, val_return=15.2339
  - Epoch 4: train_score=1933.5210, train_sortino=1931.4456, train_return=25.9422 | val_score=1480.2184, val_sortino=1478.7138, val_return=18.8080
  - Epoch 5: train_score=2340.4449, train_sortino=2338.1093, train_return=29.1942 | val_score=1761.6944, val_sortino=1760.2263, val_return=18.3512
  - Epoch 6: train_score=2713.5914, train_sortino=2711.0394, train_return=31.9006 | val_score=1895.8920, val_sortino=1894.1832, val_return=21.3606

## Planned sweeps

- Sweep attention_window around best nano settings for Alpaca data.
- Sweep value_embedding_every and skip_scale_init.
- Compare crypto-only multi-symbol selector vs per-symbol independent allocations.
