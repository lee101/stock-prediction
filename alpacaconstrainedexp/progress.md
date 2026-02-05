# Alpaca Constrained Cross-Learning Progress

Tracking constrained Chronos2 multi-symbol fine-tunes, global policy, and selector results.

## Universe

- Longable tech: `NVDA`, `GOOG`, `MSFT`
- Longable crypto: `BTCUSD`, `ETHUSD`, `SOLUSD`
- Shortable stocks: `YELP`, `EBAY`, `TRIP`, `MTCH`, `KIND`, `ANGI`, `Z`, `EXPE`, `BKNG`, `NWSA`, `NYT`

## Symbol group ranking (MAE%)

| Date (UTC) | Report | Long symbols | Short symbols | Notes |
| --- | --- | --- | --- | --- |
| 2026-02-05 | `alpacaconstrainedexp/outputs/symbol_groups_20260205_022345.json` | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD | NWSA,BKNG,MTCH,EXPE,EBAY | Chronos2 base MAE% ranking (ctx=512, val=168h). |

## Chronos2 multi-symbol fine-tunes

| Date (UTC) | Run | Symbols | Steps | Context | LR | Preaug | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | chronos2_multi_20260205_022403 | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 1000 | 1024 | 1e-5 | percent_change | LoRA fine-tune. |

## Global policy training

| Date (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-05 | constrained_global_20260205_0230 | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 0.0608 | 73.1732 | MA windows=168, min_history=150. |

## Selector results (10d)

| Date (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| (running) | selector_20260205_10d | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (running) | (running) | 10d constrained selector (checkpoint `constrained_global_20260205_0230`). |

## Notes

- Forecast cache root: `alpacaconstrainedexp/forecast_cache`
- Chronos fine-tunes: `alpacaconstrainedexp/chronos_finetuned`
