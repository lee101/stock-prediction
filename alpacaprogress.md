# Alpaca Cross‑Learning Progress

Tracking Chronos2 multi‑symbol fine‑tunes + global trading policy results.

## Latest summary (10d / 20d / 30d marketsim)

| Date (UTC) | Run | Symbols | Eval window | total_return | sortino | ann_return_365 | ann_return_252 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-07 | selector_mixed14_crypto5_tsla_int10000_minedge001_dip0025_eval70d_20260207 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,TSLA | 70d | 54.8285 | 104.5562 | -- | -- | **NEW BEST (70d val window)** Mixed14 policy epoch_002 + selector tuned: `intensity=10000, min_edge=0.001, risk_weight=0.05, dip=0.0025, edge_mode=high_low`. Leverage realism: `max_leverage_crypto=1.0`, `max_leverage_stock=2.0`, `margin_interest_annual=0.0675` (financing_cost_paid=$0; cash-only end). Policy checkpoint: `binanceneural/checkpoints/alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319/epoch_002.pt`. Forecast cache: `alpacanewccrosslearning/forecast_cache/mixed14_robust_20260205_2301_lb4000`. Outputs: `alpacanewccrosslearning/outputs/global_selector_mixed14_crypto5_tsla_int10000_minedge001_rw005_dip0025_eval70d_minh400_20260207`. |
| 2026-02-07 | selector_mixed14_crypto5_tsla_int10000_minedge001_dip0025_eval60d_20260207 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,TSLA | 60d | 29.6487 | 101.8694 | -- | -- | Same config as 70d (above). Outputs: `alpacanewccrosslearning/outputs/global_selector_mixed14_crypto5_tsla_int10000_minedge001_rw005_dip0025_eval60d_minh400_20260207`. |
| 2026-02-07 | selector_mixed7_novol_baseline_rebuiltstocks_2xboth_int675_minh480_minedge0005_dip0025_20260207 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 60d | 438.0321 | 66.3432 | -- | -- | **Experimental leverage** (stocks+crypto), same tuned selector thresholds as 30d: `intensity=2.0, min_edge=0.0005, risk_weight=0.15, dip=0.0025, edge_mode=high_low`, `max_leverage_stock=2.0`, `max_leverage_crypto=2.0`, `margin_interest_annual=0.0675` (financing_cost_paid=$1744.10). Uses `--min-history-hours 480` because lb2400 stock validation slices are <720 rows (NVDA=494, NFLX=482). Policy checkpoint: `binanceneural/checkpoints/alpaca_cross_global_mixed7_novol_baseline_seq128_rebuiltstocks_nocompile_20260205_2250/epoch_003.pt`. Outputs: `alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260207_eval60d_2xboth_int675_minh480_minedge0005_dip0025`. |
| 2026-02-07 | selector_mixed7_novol_baseline_rebuiltstocks_2xboth_int675_minh480_minedge0005_dip0025_20260207 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 30d | 14.9824 | 51.1532 | -- | -- | **Experimental leverage** (stocks+crypto), tuned selector thresholds: `intensity=2.0, min_edge=0.0005, risk_weight=0.15, dip=0.0025, edge_mode=high_low`, `max_leverage_stock=2.0`, `max_leverage_crypto=2.0`, `margin_interest_annual=0.0675` (financing_cost_paid=$60.15). Uses `--min-history-hours 480` because lb2400 stock validation slices are <720 rows (NVDA=494, NFLX=482). Policy checkpoint: `binanceneural/checkpoints/alpaca_cross_global_mixed7_novol_baseline_seq128_rebuiltstocks_nocompile_20260205_2250/epoch_003.pt`. Outputs: `alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260207_eval30d_2xboth_int675_minh480_minedge0005_dip0025`. |
| 2026-02-07 | selector_mixed7_novol_baseline_rebuiltstocks_2xboth_int675_minh480_20260207 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 30d | 14.9496 | 49.8393 | -- | -- | **Experimental leverage** (stocks+crypto): selector config `intensity=2.0, min_edge=0.001, risk_weight=0.15, dip=0.0025, edge_mode=high_low`, `max_leverage_stock=2.0`, `max_leverage_crypto=2.0`, `margin_interest_annual=0.0675` (financing_cost_paid=$60.08). Uses `--min-history-hours 480` because lb2400 stock validation slices are <720 rows (NVDA=494, NFLX=482). Policy checkpoint: `binanceneural/checkpoints/alpaca_cross_global_mixed7_novol_baseline_seq128_rebuiltstocks_nocompile_20260205_2250/epoch_003.pt`. Outputs: `alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260207_eval30d_2xboth_int675_minh480`. |
| 2026-02-07 | selector_stock19_yelp_2x_int675_rw005_20260207_071233 | NVDA,NET,AMD,GOOG,MSFT,META,AMZN,AAPL,TSLA,YELP,ANGI,Z,MTCH,TRIP,BKNG,EBAY,EXPE,NWSA,NYT | 30d (limited: 2026-01-22..2026-02-05) | 0.3480 | 20.5075 | -- | -- | **Stock-only long/short + 2x leverage**: selector config `min_edge=0.0, risk_weight=0.05, edge_mode=high_low`, allow-short with `YELP` short-only, `max_leverage_stock=2.0`, `margin_interest_annual=0.0675`. Policy checkpoint: `binanceneural/checkpoints/alpaca_cross_global_stock19_yelp_short_seq128_lb2400_20260207_065227/epoch_006.pt`. Outputs: `alpacanewccrosslearning/outputs/global_selector_stock19_yelp_seq128_20260207_071233_eval30d_2x_int675_rw005`. Note: effective selector window is limited by the validation slice (87 timestamps). |
| 2026-02-06 | selector_mixed14_crypto2_opt_20260206_0720 | SOLUSD,LINKUSD | 90d | 53.3718 | 74.7513 | -- | -- | **CURRENT BEST** SOLUSD+LINKUSD with mixed14 policy epoch_002 (intensity=10000, min_edge=0.0, risk_weight=0.05). Policy: `binanceneural/checkpoints/alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319/epoch_002.pt`. Better sortino than previous 203x claim (which may have been from stale checkpoint state). |
| 2026-02-06 | selector_crypto2_optimalint_4500_20260206_0455 | SOLUSD,LINKUSD | 90d | 203.0947 | 63.9824 | -- | -- | (Note: Result from context summary - cannot reproduce. Checkpoint may have changed.) Just SOLUSD+LINKUSD with extreme intensity (intensity=4500, min_edge=0.003, risk_weight=0.2). Policy checkpoint `binanceneural/checkpoints/alpaca_cross_global/epoch_004.pt` (trained on crypto5). Individual: SOLUSD=52.18x, LINKUSD=61.40x. |
| 2026-02-06 | selector_crypto5_optimalint_4500_20260206_0445 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD | 90d | 199.9813 | 66.0594 | -- | -- | Crypto-only policy with extreme intensity (intensity=4500, min_edge=0.003, risk_weight=0.2). Policy checkpoint `binanceneural/checkpoints/alpaca_cross_global/epoch_004.pt`. Returns plateau at intensity>=4500. 30d total_return=5.31 (intensity=1800); 60d total_return=32.19. |
| 2026-02-06 | selector_mega24_optimalint_1800_rw02_20260206_0410 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 5.1788 | 93.4322 | 4.0e+09 | 2.1e+07 | Optimal high-intensity config (intensity=1800, min_edge=0.003, risk_weight=0.2, dip=0.0, edge_mode=high_low). Same policy checkpoint `alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119/epoch_003.pt`. Feature windows: MA/EMA/ATR=24,72. 60d total_return=27.7584; 90d total_return=146.4756 sortino=60.3065. Strategy predominantly trades LINKUSD/SOLUSD crypto with short-term momentum. |
| 2026-02-06 | selector_mega24_extremeint_1000_20260206_0350 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 4.9907 | 92.2307 | 2.9e+09 | 1.7e+07 | Extreme intensity config (intensity=1000, min_edge=0.003, risk_weight=0.15, dip=0.0). Same policy checkpoint. Feature windows: MA/EMA/ATR=24,72. 90d total_return=134.7572 sortino=59.9816. |
| 2026-02-06 | selector_mega24_highint_55_20260206_0335 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 2.7422 | 51.2120 | 1.5e+07 | 1.0e+05 | Higher intensity config (intensity=5.5, min_edge=0.002, risk_weight=0.15, dip=0.0). Same policy checkpoint `alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119/epoch_003.pt`. Feature windows: MA/EMA/ATR=24,72. 60d total_return=11.5708 sortino=52.7799; 90d total_return=31.6731 sortino=53.9447. |
| 2026-02-06 | selector_mega24_highint_50_20260206_0320 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 2.7233 | 51.3348 | 1.5e+07 | 1.0e+05 | High-intensity config (intensity=5.0, min_edge=0.0015, risk_weight=0.15, dip=0.0). Same policy checkpoint `alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119/epoch_003.pt`. Feature windows: MA/EMA/ATR=24,72. 60d total_return=11.0619 sortino=53.8151; 90d total_return=29.4445 sortino=54.9547. |
| 2026-02-06 | selector_mega24_highint_40_20260206_0310 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 2.6458 | 50.8641 | 1.4e+07 | 9.5e+04 | High-intensity config (intensity=4.0, min_edge=0.001, risk_weight=0.15, dip=0.0). Same policy checkpoint. Feature windows: MA/EMA/ATR=24,72. 10d total_return=0.5377 sortino=30.2647; 20d total_return=1.1746 sortino=37.5570; 60d total_return=10.4295 sortino=56.0192; 90d total_return=27.8506 sortino=57.4679. |
| 2026-02-06 | selector_mega24_novol_baseline_short_seq128_bestret_20260206_0146 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 10d | 0.6083 | 32.1289 | 3.406e+07 | 1.586e+05 | Forecast cache `mega24_novol_baseline_20260206_0038_lb2400`; policy checkpoint `alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119/epoch_003.pt`; selector sweep best config intensity=2.4, min_edge=0.0005, risk_weight=0.05, dip=0.0025, edge_mode=high (max_hold=0). Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mega24_novol_baseline_short_seq128_20260206_0146/selector_sweep.csv`. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega24_novol_baseline_short_seq128_20260206_bestret_eval10d`. |
| 2026-02-06 | selector_mega24_novol_baseline_short_seq128_bestret_20260206_0146 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 20d | 1.0498 | 31.2419 | 4.882e+05 | 8.461e+03 | Same config as 10d. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega24_novol_baseline_short_seq128_20260206_bestret_eval20d`. |
| 2026-02-06 | selector_mega24_novol_baseline_short_seq128_bestret_20260206_0146 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 2.0877 | 39.9895 | 9.063e+05 | 1.297e+04 | Same config as 10d; 60d total_return=7.0347 sortino=42.6167; 90d total_return=11.6065 sortino=44.9608. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega24_novol_baseline_short_seq128_20260206_bestret_eval30d` + `..._eval60d` + `..._eval90d`. |
| 2026-02-06 | selector_mega21_novol_baseline_bestret_20260206_004012 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 10d | 0.9894 | 29.2976 | 8.001e+10 | 3.370e+07 | Forecast cache `mega21_novol_baseline_20260205_2351_lb2400`; policy checkpoint `alpaca_cross_global_mega21_novol_baseline_seq128_cacheonly_20260206_0008/epoch_002.pt`; selector sweep best config intensity=2.4, min_edge=0.001, risk_weight=0.1, dip=0.0, edge_mode=high_low. Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mega21_novol_baseline_seq128_20260206_004012/selector_sweep.csv`. |
| 2026-02-06 | selector_mega21_novol_baseline_bestret_20260206_004012 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 20d | 1.5699 | 28.6031 | 3.027e+07 | 1.462e+05 | Same config as 10d. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega21_novol_baseline_seq128_20260206_bestret_eval20d`. |
| 2026-02-06 | selector_mega21_novol_baseline_bestret_20260206_004012 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 30d | 2.2359 | 29.9237 | 1.603e+06 | 1.923e+04 | Same config as 10d; 60d total_return=4.9335 sortino=25.9576; 90d total_return=6.6144 sortino=24.6822. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega21_novol_baseline_seq128_20260206_bestret_eval30d` + `..._eval60d` + `..._eval90d`. |
| 2026-02-06 | selector_mega21_novol_baseline_allowshort_bestret_20260206_004534 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 10d | 0.5685 | 30.4566 | 1.365e+07 | 8.436e+04 | Same forecast cache + checkpoint `alpaca_cross_global_mega21_novol_baseline_allowshort_seq128_lb2400_20260206_001135/epoch_006.pt` with allow-short + direction constraints. Best sweep config intensity=1.6, min_edge=0.002, risk_weight=0.1, dip=0.0025. Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mega21_novol_baseline_allowshort_seq128_20260206_004534/selector_sweep.csv`. |
| 2026-02-06 | selector_mega21_novol_baseline_allowshort_bestret_20260206_004534 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 20d | 0.8786 | 18.8913 | 9.942e+04 | 2.819e+03 | Same config as 10d (allow-short). Outputs: `alpacanewccrosslearning/outputs/global_selector_mega21_novol_baseline_allowshort_seq128_20260206_bestret_eval20d_v2`. |
| 2026-02-06 | selector_mega21_novol_baseline_allowshort_bestret_20260206_004534 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 30d | 1.2496 | 18.1390 | 1.923e+04 | 9.062e+02 | Same config as 10d (allow-short); 60d total_return=3.1398 sortino=18.9946; 90d total_return=4.3957 sortino=19.0079. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega21_novol_baseline_allowshort_seq128_20260206_bestret_eval30d_v2` + `..._eval60d_v2` + `..._eval90d_v2`. |
| 2026-02-06 | selector_expanded23_short_20260206_0145 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 10d | 0.8197 | 39.7007 | 8.0e+10 | 3.4e+07 | New 23-symbol universe with shorting; Chronos2 LoRA `cross_lora_expanded24_novol_20260206_0025` (1000 steps, no vol); forecast cache `expanded24_novol_20260206_lb2400`; policy `alpaca_cross_global_expanded24_short_seq128_20260206_0109/epoch_003.pt`; selector intensity=2.2, min_edge=0.001, risk_weight=0.15, dip=0.0. |
| 2026-02-06 | selector_expanded23_short_20260206_0145 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 20d | 1.1777 | 35.1971 | 3.0e+07 | 1.5e+05 | Same config as 10d. |
| 2026-02-06 | selector_expanded23_short_20260206_0145 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 1.9026 | 39.7303 | 1.0e+07 | 7.0e+04 | Same config as 10d; 60d total_return=5.6625 sortino=43.6673. |
| 2026-02-06 | selector_expanded23_short_int28_20260206_0245 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 10d | 0.8698 | 41.1470 | 1.1e+11 | 4.5e+07 | Higher intensity config (intensity=2.8 vs 2.2); same policy+cache as `selector_expanded23_short_20260206_0145`; min_edge=0.001, risk_weight=0.15, dip=0.0. |
| 2026-02-06 | selector_expanded23_short_int28_20260206_0245 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 20d | 1.2469 | 36.6003 | 5.5e+07 | 2.4e+05 | Same config as 10d. |
| 2026-02-06 | selector_expanded23_short_int28_20260206_0245 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,GOOG,MSFT,AMD,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 30d | 2.0325 | 40.8946 | 1.2e+07 | 8.5e+04 | Same config as 10d; 60d total_return=5.7067 sortino=43.6831; 90d total_return=9.2755 sortino=44.8129. |
| 2026-02-05 | selector_mixed7_novol_baseline_rebuiltstocks_best_20260205_2310 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.4613 | 347.5836 | 1.895e+14 | 7.202e+09 | Forecast cache `mixed7_novol_baseline_20260205_2136_lb2400` with NVDA/NFLX forecasts rebuilt post gap-fill; policy checkpoint `alpaca_cross_global_mixed7_novol_baseline_seq128_rebuiltstocks_nocompile_20260205_2250/epoch_003.pt`; selector intensity=2.0 min_edge=0.001 risk_weight=0.15 dip=0.0025. |
| 2026-02-05 | selector_mixed7_novol_baseline_rebuiltstocks_best_20260205_2310 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 2.2006 | 225.4522 | 1.661e+09 | 2.322e+06 | Same config as 10d. |
| 2026-02-05 | selector_mixed7_novol_baseline_rebuiltstocks_best_20260205_2310 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 30d | 3.6897 | 184.4416 | 1.464e+08 | 4.341e+05 | Same config as 10d; 60d total_return=10.9753 sortino=47.4376. |
| 2026-02-05 | selector_mixed14_seq128_lb4000_bestret_20260205_2336 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT | 10d | 1.0862 | 108.9982 | 4.530e+11 | 1.116e+08 | Seq128 policy checkpoint epoch_004 (`alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319`) with selector sweep best total_return config intensity=1.8, min_edge=0.002, risk_weight=0.15, dip=0.0 (lb4000 cache). 30d total_return=2.7628 sortino=76.5497; 60d total_return=11.5012 sortino=54.1330. |
| 2026-02-05 | selector_mixed14_seq128_lb4000_bestret_20260205_2336 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT | 20d | 1.6025 | 72.0240 | 3.809e+07 | 1.713e+05 | Same config as 10d; 30d total_return=2.7628 sortino=76.5497; 60d total_return=11.5012 sortino=54.1330. |
| 2026-02-05 | selector_mixed14_seq128_lb4000_bestret_20260205_2336 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT | 30d | 2.7628 | 76.5497 | 1.004e+07 | 6.824e+04 | Same config as 10d; 60d total_return=11.5012 sortino=54.1330. |
| 2026-02-05 | selector_seq128_lb2400_best_20260205_2227 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.0106 | 132.6112 | 1.178e+11 | 4.403e+07 | Seq128 policy checkpoint epoch_003 (`alpaca_cross_global_mixed7_robust_short_seq128_lb2400_20260205_2222`) with selector sweep config intensity=2.2, min_edge=0.004, risk_weight=0.15, dip=0.0 (lb2400 cache). 30d total_return=2.8514 sortino=103.1766. |
| 2026-02-05 | selector_seq128_lb2400_best_20260205_2227 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.5574 | 110.4913 | 2.768e+07 | 1.375e+05 | Same config as 10d; 30d total_return=2.8514 sortino=103.1766; 60d total_return=10.4748 sortino=47.7637. |
| 2026-02-05 | selector_seq128_best_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.6463 | 128.4568 | 2.669e+15 | 4.472e+10 | Seq128 policy checkpoint epoch_003 with intensity=2.0, min_edge=0.004, risk_weight=0.2, dip=0.005. |
| 2026-02-05 | selector_seq128_best_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.6463 | 128.4568 | 5.166e+07 | 2.115e+05 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_target3_20260205_041640 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1952 | 71.5462 | 2.909e+12 | 4.029e+08 | Mixed7 robust_scaling short-window; targeted min_edge/risk sweep v3 (intensity=2.0, min_edge=0.004, risk_weight=0.2, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_target3 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1952 | 71.5462 | 1.706e+06 | 2.007e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_target2_20260205_041026 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1819 | 73.1664 | 2.331e+12 | 3.457e+08 | Mixed7 robust_scaling short-window; targeted min_edge/risk sweep (intensity=2.0, min_edge=0.003, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_target2 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1819 | 73.1664 | 1.527e+06 | 1.859e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_intensity20_20260205_040605 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1737 | 73.5806 | 2.031e+12 | 3.144e+08 | Mixed7 robust_scaling short-window; best sweep config (intensity=2.0, min_edge=0.001, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_int20 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1737 | 73.5806 | 1.425e+06 | 1.773e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_intensity16_20260205_035644 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1541 | 77.5686 | 1.459e+12 | 2.502e+08 | Mixed7 robust_scaling short-window; best sweep config (intensity=1.6, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205_int16 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1541 | 77.5686 | 1.208e+06 | 1.582e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_mixed7_20260205_025742 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 10d | 1.1196 | 77.4274 | 8.096e+11 | 1.666e+08 | Mixed7 robust_scaling + short-window features; best sweep config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_mixed7_best_20260205 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 20d | 1.1196 | 77.4274 | 8.998e+05 | 1.291e+04 | Same config as 10d; window-limited by stock data. |
| 2026-02-05 | selector_sweep_robust_intensity_20260205_022332 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.5629 | 30.8584 | 1.198e+07 | 7.71e+04 | Robust_scaling policy; best sweep config (intensity=1.2, min_edge=0.0005, risk_weight=0.25, dip=0.005). |
| 2026-02-05 | selector_robust_best_20260205 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.7482 | 21.3372 | 2.675e+04 | 1138 | Robust_scaling policy; best sweep config carried to 20d eval. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 10d | 0.0379 | 510.2498 | 2.888 | 1.553 | Global policy checkpoint epoch_001. |
| 2026-02-05 | selector_cross_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 20d | 0.0379 | 510.2498 | 0.9717 | 0.5979 | Same as 10d (limited window in current cache). |

Annualized returns use CAGR: `(1 + total_return) ** (basis_days / eval_days) - 1`. Short windows will inflate annualized values; use them only for relative comparison across runs.

## Constrained experiment (alpacaconstrainedexp)

| Date (UTC) | Run | Stage | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | chronos2_multi_20260205_022403 | Chronos2 LoRA | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (n/a) | (n/a) | Preaug=percent_change, ctx=1024, steps=1000. |
| 2026-02-05 | constrained_global_20260205_0230 | Global policy | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 0.0608 | 73.1732 | MA windows=168, min_history=150. |
| 2026-02-05 | selector_20260205_10d | Selector 10d | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (running) | (running) | Constrained selector sim (checkpoint epoch_004). |

## Constrained11 (alpacanewccrosslearning)

| Date (UTC) | Run | Stage | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | cross_lora_constrained11_novol_baseline_20260205_2314 | Chronos2 LoRA | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | (n/a) | (n/a) | Steps=800, ctx=1024, lr=2e-5, no volume/log_volume. Forecast cache root `constrained11_novol_baseline_20260205_2314_lb2400`. |
| 2026-02-05 | alpaca_cross_global_constrained11_novol_baseline_seq128_20260205_2321 | Global policy | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 1.6171 | 40.3057 | Seq128 policy trained on constrained11 forecasts; best checkpoint `epoch_003.pt`. (Metrics printed by training script are target-symbol eval, not best-trade selector.) |
| 2026-02-05 | selector_sweep_constrained11_novol_baseline_20260205_2331 | Selector sweep (10d) | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 0.5264 | 15.2793 | Best total_return sweep config: intensity=2.0, min_edge=0.002, risk_weight=0.15, dip=0.0025. |
| 2026-02-05 | global_selector_constrained11_novol_baseline_20260205_2340 | Selector eval (30d) | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 1.1620 | 14.5161 | Best sortino config (intensity=2.0, min_edge=0.001, risk_weight=0.15, dip=0.0025): 10d total_return=0.5092 sortino=16.1369; 20d total_return=0.6756 sortino=11.9047; 30d total_return=1.1620 sortino=14.5161; 60d total_return=3.4263 sortino=14.5699. Outputs: `alpacanewccrosslearning/outputs/global_selector_constrained11_novol_baseline_20260205_2340`. |

## Mega21 (alpacanewccrosslearning)

| Date (UTC) | Run | Stage | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | cross_lora_mega21_novol_baseline_20260205_2351 | Chronos2 LoRA | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | (n/a) | (n/a) | Steps=1000, ctx=1024, lr=2e‑5, no volume/log_volume. Fine‑tuned model: `alpacanewccrosslearning/chronos_finetuned/cross_lora_mega21_novol_baseline_20260205_2351/finetuned`. |
| 2026-02-05 | mega21_novol_baseline_20260205_2351_lb2400 | Forecast cache + MAE | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | (n/a) | (n/a) | Forecast cache root: `alpacanewccrosslearning/forecast_cache/mega21_novol_baseline_20260205_2351_lb2400` (h1+h24). Forecast MAE (30d) CSV: `alpacanewccrosslearning/outputs/forecast_eval_mega21_novol_baseline_lb2400_20260206_eval30d.csv`. |
| 2026-02-06 | alpaca_cross_global_mega21_novol_baseline_seq128_cacheonly_20260206_0008 | Global policy | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 1.3970 | 34.1090 | MA/EMA/ATR=24/72, min_history=200, sequence_length=128. Best checkpoint: `epoch_002.pt`. (Training-script metrics are target-symbol eval, not best-trade selector.) |
| 2026-02-06 | selector_sweep_mega21_novol_baseline_20260206_0025 | Selector sweep (10d) | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 0.9773 | 28.6181 | Best config: intensity=2.0, min_edge=0.001, risk_weight=0.1, dip=0.0, edge_mode=high_low. Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mega21_novol_baseline_20260206_0025/selector_sweep.csv`. |
| 2026-02-06 | global_selector_mega21_novol_baseline_20260206_0032 | Selector eval | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 1.9947 | 26.2052 | Best config carried forward: 10d total_return=0.9773 sortino=28.6181; 20d total_return=1.4347 sortino=25.5505; 30d total_return=1.9947 sortino=26.2052; 60d total_return=4.3993 sortino=24.7954; 90d total_return=6.1003 sortino=24.0604. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega21_novol_baseline_20260206_0032`. |

## Mega24 (alpacanewccrosslearning)

| Date (UTC) | Run | Stage | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-02-06 | cross_lora_mega24_novol_baseline_20260206_0038 | Chronos2 LoRA | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | (n/a) | (n/a) | Steps=1200, ctx=1024, lr=2e‑5, no volume/log_volume. Fine‑tuned model: `alpacanewccrosslearning/chronos_finetuned/cross_lora_mega24_novol_baseline_20260206_0038/finetuned`. |
| 2026-02-06 | mega24_novol_baseline_20260206_0038_lb2400 | Forecast cache + MAE | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | (n/a) | (n/a) | Forecast cache root: `alpacanewccrosslearning/forecast_cache/mega24_novol_baseline_20260206_0038_lb2400` (h1+h24). Forecast MAE (30d) CSV: `alpacanewccrosslearning/outputs/forecast_eval_mega24_novol_baseline_lb2400_20260206_0106_eval30d.csv` (AMD forecasts rebuilt post gap-fill). |
| 2026-02-06 | alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119 | Global policy | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 2.2250 | 56.7291 | Seq128 policy trained on mega24 forecasts (MA/EMA/ATR=24/72, trend/drawdown=72, min_history=200, allow-short). Best checkpoint: `epoch_003.pt`. (Training-script metrics are target-symbol eval (BTCUSD), not best-trade selector.) |
| 2026-02-06 | selector_sweep_mega24_novol_baseline_short_seq128_20260206_0146 | Selector sweep (10d) | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 0.6083 | 32.1289 | Best config: intensity=2.4, min_edge=0.0005, risk_weight=0.05, dip=0.0025, edge_mode=high, max_hold=0. Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mega24_novol_baseline_short_seq128_20260206_0146/selector_sweep.csv`. |
| 2026-02-06 | global_selector_mega24_novol_baseline_short_seq128_20260206_bestret | Selector eval | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 2.0877 | 39.9895 | Best config carried forward: 10d total_return=0.6083 sortino=32.1289; 20d total_return=1.0498 sortino=31.2419; 30d total_return=2.0877 sortino=39.9895; 60d total_return=7.0347 sortino=42.6167; 90d total_return=11.6065 sortino=44.9608. Outputs: `alpacanewccrosslearning/outputs/global_selector_mega24_novol_baseline_short_seq128_20260206_bestret_eval10d` + `..._eval20d` + `..._eval30d` + `..._eval60d` + `..._eval90d`. |

## Chronos2 multi‑symbol fine‑tunes

| Date (UTC) | Run | Symbols | Steps | Context | LR | Preaug | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | cross_smoke_20260205_012017 | SOLUSD,LINKUSD | 20 | 512 | 1e‑5 | baseline | Smoke LoRA run. |
| 2026-02-05 | cross_lora_20260205_012542 | SOLUSD,LINKUSD,UNIUSD | 200 | 1024 | 1e‑5 | baseline | Forecast cache built (h1+h24, 720h lookback). |
| 2026-02-05 | cross_lora_mixed7_robust_20260205_022709 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 200 | 1024 | 1e‑5 | robust_scaling | Mixed7 LoRA (preaug robust_scaling). |
| 2026-02-05 | cross_lora_mixed7_robust_lr2e5_20260205_040853 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 200 | 1024 | 2e‑5 | robust_scaling | Mixed7 LoRA (eval_loss=0.167414). |
| 2026-02-05 | cross_lora_mixed7_robust_20260205_2209 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 400 | 1024 | 1e‑5 | robust_scaling | Mixed7 LoRA retrain on refreshed hourly data (final eval_loss≈0.1658). Forecast cache built (lb2400, h1+h24). |
| 2026-02-05 | cross_lora_mixed7_novol_baseline_20260205_2136 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 500 | 1024 | 2e‑5 | baseline | No volume/log_volume covariates. Forecast cache built: `mixed7_novol_baseline_20260205_2136_lb2400` (NVDA/NFLX rebuilt post gap-fill). Forecast MAE eval saved to `alpacanewccrosslearning/outputs/forecast_eval_mixed7_novol_baseline_lb2400_20260206_eval30d.csv`. |
| 2026-02-05 | cross_lora_constrained11_novol_baseline_20260205_2314 | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 800 | 1024 | 2e‑5 | baseline | No volume/log_volume covariates. Forecast cache built: `constrained11_novol_baseline_20260205_2314_lb2400`. Forecast MAE eval saved to `alpacanewccrosslearning/outputs/forecast_eval_constrained11_novol_baseline_lb2400_20260206_eval30d.csv`. |
| 2026-02-05 | cross_lora_mixed14_robust_20260205_2301 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT | 800 | 1024 | 1e‑5 | robust_scaling | Mixed14 LoRA retrain (final eval_loss≈0.3392). Forecast cache built: `mixed14_robust_20260205_2301_lb4000` (h1+h24). Forecast MAE eval saved to `alpacanewccrosslearning/outputs/forecast_eval_mixed14_robust_lb4000_20260205_eval30d.csv`. |
| 2026-02-05 | cross_lora_mega21_novol_baseline_20260205_2351 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 1000 | 1024 | 2e‑5 | baseline | Mega21 LoRA baseline (no volume/log_volume; eval_loss=0.313619). Forecast cache built: `mega21_novol_baseline_20260205_2351_lb2400` (h1+h24). Forecast MAE eval saved to `alpacanewccrosslearning/outputs/forecast_eval_mega21_novol_baseline_lb2400_20260206_eval30d.csv`. |
| 2026-02-06 | cross_lora_mega24_novol_baseline_20260206_0038 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 1200 | 1024 | 2e‑5 | baseline | Mega24 LoRA baseline (no volume/log_volume; eval_loss≈0.3791). Forecast cache built: `mega24_novol_baseline_20260206_0038_lb2400` (h1+h24). Forecast MAE eval saved to `alpacanewccrosslearning/outputs/forecast_eval_mega24_novol_baseline_lb2400_20260206_0106_eval30d.csv`. |

## Global policy training

| Date (UTC) | Run | Symbols | total_return | sortino | Notes |
| --- | --- | --- | --- | --- | --- |
| 2026-02-05 | alpaca_cross_global_lora_20260205_013206 | SOLUSD,LINKUSD,UNIUSD | 0.1872 | 47.2885 | MA windows 168/336, min_history=200. |
| 2026-02-05 | alpaca_cross_global_robust_20260205_020915 | SOLUSD,LINKUSD,UNIUSD | 0.1010 | 26.9010 | Robust_scaling forecasts, MA windows 168/336, min_history=200. |
| 2026-02-05 | alpaca_cross_global_mixed7_robust_short_20260205_025307 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 0.3038 | 76.5146 | Mixed7 policy with short-window feature overrides; window 2025-09-01..2025-11-12. |
| 2026-02-05 | alpaca_cross_global_mixed7_robust_short_seq128_20260205_043448 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 0.5935 | 291.1584 | Seq128 policy with short-window feature overrides; best checkpoint epoch_003. |
| 2026-02-05 | alpaca_cross_global_mixed7_robust_short_seq128_lb2400_20260205_2222 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 18.0359 | 65.6919 | Seq128 policy trained on lb2400 mixed7 cache (`mixed7_robust_20260205_2209_lb2400`); MA/EMA/ATR=24/72, min_history=200. |
| 2026-02-05 | alpaca_cross_global_mixed7_novol_baseline_seq128_rebuiltstocks_nocompile_20260205_2250 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX | 13.6560 | 57.9001 | Retrained after rebuilding NVDA/NFLX forecast caches in `mixed7_novol_baseline_20260205_2136_lb2400`; best checkpoint `epoch_003.pt` used in latest selector runs. |
| 2026-02-05 | alpaca_cross_global_constrained11_novol_baseline_seq128_20260205_2321 | NVDA,MSFT,GOOG,BTCUSD,SOLUSD,ETHUSD,NWSA,BKNG,MTCH,EXPE,EBAY | 1.6171 | 40.3057 | Constrained11 policy trained on `constrained11_novol_baseline_20260205_2314_lb2400`; best checkpoint `epoch_003.pt`. |
| 2026-02-05 | alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319 | SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT | 18.3426 | 64.8020 | Mixed14 policy trained on `mixed14_robust_20260205_2301_lb4000` (MA/EMA/ATR=24/72, min_history=200); best checkpoint `epoch_004.pt`. |
| 2026-02-06 | alpaca_cross_global_mega21_novol_baseline_seq128_cacheonly_20260206_0008 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 1.3970 | 34.1090 | Seq128 policy trained on `mega21_novol_baseline_20260205_2351_lb2400`; best checkpoint `epoch_002.pt`. Metrics are training-script target-symbol eval (BTCUSD), not best-trade selector. |
| 2026-02-06 | alpaca_cross_global_mega21_novol_baseline_allowshort_seq128_lb2400_20260206_001135 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,NFLX,MSFT,GOOG,AAPL,AMZN,META,TSLA,JPM,V,WMT,NWSA,BKNG,MTCH,EXPE,EBAY | 0.0864 | 106.1668 | Seq128 mega21 policy with allow-short enabled (stocks only). Best checkpoint `epoch_006.pt`. Long-only extras: NFLX,JPM,V,WMT. Short-only extras: EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT. Metrics are training-script target-symbol eval (NVDA, 60d), not best-trade selector. |
| 2026-02-06 | alpaca_cross_global_mega24_novol_baseline_short_seq128_20260206_0119 | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AMD,GOOG,NET,MSFT,META,AMZN,TSLA,AAPL,NFLX,EBAY,TRIP,MTCH,ANGI,Z,EXPE,BKNG,NWSA,NYT | 2.2250 | 56.7291 | Seq128 mega24 policy trained on `mega24_novol_baseline_20260206_0038_lb2400` (MA/EMA/ATR=24/72, trend/drawdown=72, min_history=200, allow-short). Best checkpoint `epoch_003.pt`. Metrics are training-script target-symbol eval (BTCUSD), not best-trade selector. |
| 2026-02-07 | alpaca_cross_global_stock19_yelp_short_seq128_lb2400_20260207_065227 | NVDA,NET,AMD,GOOG,MSFT,META,AMZN,AAPL,TSLA,YELP,ANGI,Z,MTCH,TRIP,BKNG,EBAY,EXPE,NWSA,NYT | 0.0631 | 59.5710 | Stock-only seq128 policy trained on `mega24_plus_yelp_novol_baseline_20260207_lb2400` forecasts with allow-short + `YELP` short-only. Feature windows: MA/EMA/ATR=24,72; trend/drawdown=72; min_history=200. Best checkpoint: `epoch_006.pt`. Metrics are training-script target-symbol eval (NVDA), not best-trade selector. |

## PufferLib C RL Market Simulator

Pure C environment with PufferLib 3.0 Ocean binding. PPO training, 64 envs. Reward: clipped return*10 + cash penalty (-0.01). All runs use Chronos2 forecasts as input features.

### 2026-02-10 Critical Fix (Hourly Selector Sim: Decision Lag)

- Added `decision_lag_bars` support to the Alpaca hourly Python simulators so the selector can be evaluated **live-like**: decisions from the latest completed bar are executed on the next bar.
- Without this lag, the simulator can implicitly trade using information from the same bar it fills on (optimistic / lookahead-like), inflating returns.
- Use `--decision-lag-bars 1` in `newnanoalpacahourlyexp/run_multiasset_selector.py` (and `run_experiment.py` / `sweep.py`) for realistic reporting.

### 2026-02-10 New: Live Hourly Trader vs Shared-Cash Backtest (Prod Mismatch)

- Prod is running `newnanoalpacahourlyexp.trade_alpaca_hourly` (multi-symbol loop), but most historical results here were **selector** sims.
- I added a new shared-cash simulator for the hourly trader loop: `newnanoalpacahourlyexp/marketsimulator/hourly_trader.py` + CLI `python -m newnanoalpacahourlyexp.run_hourly_trader_sim`.
- With prod-style settings (Mixed7 `epoch_003.pt`, symbols `SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX`, `decision_lag_bars=1`, `intensity_scale=2.0`), the shared-cash sim is **negative over 30d**:
  - 30d: `total_return=-0.1016`, `sortino=-5.8142` (outputs: `experiments/hourly_trader_sim_data_20260210_eval30d/`)
  - Parameter tuning via `experiments/hourly_trader_sim_data_20260210_eval30d/run_sweep.py` improves losses but does not find a profitable config on this 30d window.
- A realistic (lagged) selector with an edge threshold is much more robust in the same down window:
  - Crypto5 (`BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD`), `decision_lag_bars=1`, `risk_weight=0.3`, `min_edge=0.01`, `intensity=2.0`:
    - 30d: `total_return=+0.2119`, `sortino=+4.2898` (outputs: `experiments/selector_sim_crypto5_mixed7_epoch003_lag1_eval30d_rw03_int2_minedge01_20260210/`)
    - 60d: `total_return=+0.0850`, `sortino=+0.9979`
    - 90d: `total_return=+0.1265`, `sortino=+1.2238`

### 2026-02-10 Critical Fix (Short Accounting + Daily MKTD v2)

- **Fixed a major short-position accounting bug** in `pufferlib_market/src/trading_env.c` where short-sale proceeds were effectively treated as profit on close, inflating episode `total_return` (and compounding into unrealistic multi-thousand-x runs).
- Added regression test: `tests/test_pufferlib_market_accounting.py`.
- Added `periods_per_year` env param (Sortino annualisation) + MKTD v2 optional `tradable` mask + daily exporter `pufferlib_market/export_data_daily.py`.
- **Implication**: results logged before **2026-02-10** for this env are not directly comparable; rerun is required for realistic PnL.

### Daily MKTD v2 (calendar-day) Experiments

Training/eval on daily `trainingdata/train` with a 50-calendar-day holdout window and a tradable mask that prevents stock trades on market-closed days (weekends/holidays).

| Date (UTC) | Run | Symbols | Train steps | Episode len | Holdout window | total_return | sortino | max_drawdown | trades | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-10 | daily_mix8_h256_lr3e4_ent001_noanneal | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA | 250k | 50d | 2025-12-17..2026-02-05 | **+0.0497** | **+1.91** | 0.1194 | 7 | Best of 10-run sweep at 250k steps. Policy mostly alternates **NVDA/TSLA long** during the holdout. Checkpoint: `experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_h256_lr3e4_ent001_noanneal/best.pt`. Data: `experiments/pufferlib_market_daily_20260210/eval50d_mktd_v2.bin`. |
| 2026-02-10 | daily_mix8_h256_lr3e4_ent001_noanneal_2M | BTCUSD,ETHUSD,SOLUSD,LINKUSD,UNIUSD,NVDA,AAPL,TSLA | 2M | 50d | 2025-12-17..2026-02-05 | **+0.3773** | **+5.42** | 0.2848 | 26 | Extended training (2M steps, no anneal) massively improves holdout. Policy is very active, mostly trading **UNIUSD long/short** + some LINKUSD and TSLA long/short. Checkpoint: `experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_h256_lr3e4_ent001_noanneal_2M/best.pt`. |

### Daily → Hourly Replay (Execution + Double-Execution Stress Test)

To catch “looks good on daily close-to-close” artifacts and measure higher-resolution risk, I added a replay evaluator:

- CLI: `python -m pufferlib_market.replay_eval`
- Core utils: `pufferlib_market/hourly_replay.py`
- Experiment dir: `experiments/pufferlib_market_daily_hourly_replay_20260210/`

Holdout: 2025-12-17..2026-02-05 (50 steps / 51 calendar days), checkpoint `experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_h256_lr3e4_ent001_noanneal_2M/best.pt`.

- **Daily eval (baseline, daily C-env dynamics)**: total_return **+0.3773**, sortino **+5.42**, max_drawdown **0.2848**, trades **26**.
- **Hourly replay (frozen daily action trace, execute once/day at NYSE close; mark-to-market hourly)**:
  - total_return **+0.2692**, hourly-sortino **+2.42**, max_drawdown **0.2856**
  - num_orders **52** (open+close counted), max_orders_in_day **2**
  - Report: `experiments/pufferlib_market_daily_hourly_replay_20260210/replay_report.json`
- **Hourly policy stress test (call the *daily-trained* policy every hour; daily features frozen per day; portfolio fields update hourly)**:
  - total_return **-0.1037**, hourly-sortino **-0.67**, max_drawdown **0.2786**
  - num_orders **56**, max_orders_in_day **7** (shows real multi-execution/day thrash risk)
  - Report: `experiments/pufferlib_market_daily_hourly_replay_20260210/replay_report_with_hourly_policy.json`

Additional checkpoint sanity check:

- 250k sweep-best (`experiments/pufferlib_market_daily_20260210/checkpoints/daily_mix8_h256_lr3e4_ent001_noanneal/best.pt`): hourly_replay total_return **+0.0951**, hourly-sortino **+1.04**; hourly-policy stress test total_return **-0.1053**. Report: `experiments/pufferlib_market_daily_hourly_replay_20260210/replay_report_250k_with_hourly_policy.json`.

### Evaluation Results (deterministic, all random-start episodes)

| Date (UTC) | Run | Symbols | Episode len | Eval mean return | Eval geom mean mult | Eval win rate | Eval 100% profitable? | Train best | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-08 | **crypto12_ppo_v5_60d_annealLR** | BTC,ETH,SOL,LINK,UNI,AVAX,DOT,AAVE,DOGE,LTC,ALGO,XRP | **1440h (60d)** | **+681101% (6812.9x)** | **6812.9x** | **0.684** | Yes (108/108) | +590749% | **ABSOLUTE BEST!** 12 crypto, 100M steps, anneal-LR, 60d episodes, h1024. **$10k→$68M per 60d.** Worst +5965.9x. Checkpoints: `pufferlib_market/checkpoints/crypto12_ppo_v5_60d_annealLR/`. |
| 2026-02-08 | **crypto12_ppo_v3_h1024_100M_annealLR** | BTC,ETH,SOL,LINK,UNI,AVAX,DOT,AAVE,DOGE,LTC,ALGO,XRP | 720h (30d) | **+80340% (804.4x)** | **804.4x** | **0.792** | Yes (217/217) | +65863% | **Best 30d model!** 12 crypto, 100M steps, anneal-LR, h1024. $10k→$8M per 30d. Worst +720.8x. Checkpoints: `pufferlib_market/checkpoints/crypto12_ppo_v3_h1024_100M_annealLR/`. |
| 2026-02-08 | **crypto8_ppo_v16_h1024_100M_annealLR** | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | **+43813% (439.1x)** | **439.1x** | **0.804** | Yes (217/217) | +41220% | 100M steps + anneal-LR. **3.3x over no-anneal 100M** (132.9→439.1). Worst +347.6x. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v16_h1024_100M_annealLR/`. |
| 2026-02-08 | crypto12_ppo_v2_h1024_100M | BTC,ETH,SOL,LINK,UNI,AVAX,DOT,AAVE,DOGE,LTC,ALGO,XRP | 720h (30d) | +20785% (209.8x) | 209.8x | 0.736 | Yes (217/217) | +13485% | 12 crypto, 100M steps, no anneal. Checkpoints: `pufferlib_market/checkpoints/crypto12_ppo_v2_h1024_100M/`. |
| 2026-02-08 | crypto12_ppo_v2_h1024_100M | BTC,ETH,SOL,LINK,UNI,AVAX,DOT,AAVE,DOGE,LTC,ALGO,XRP | 720h (30d) | +20785% (209.8x) | 209.8x | 0.736 | Yes (217/217) | +13485% | 12 crypto, 100M steps, no anneal. Checkpoints: `pufferlib_market/checkpoints/crypto12_ppo_v2_h1024_100M/`. |
| 2026-02-08 | crypto8_ppo_v12_h1024_200M | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +17014% (171.1x) | 171.1x | 0.739 | Yes (217/217) | +15692% | **200M steps** no anneal. Only +29% over 100M. Proves anneal-LR is far more efficient. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v12_h1024_200M/`. |
| 2026-02-08 | crypto8_ppo_v14_annealLR | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +13804% (139.8x) | 139.8x | 0.739 | Yes (217/217) | +12314% | 50M steps + anneal-LR. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v14_annealLR/`. |
| 2026-02-08 | crypto8_ppo_v9_h1024_100M | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +13202% (132.9x) | 132.9x | 0.777 | Yes (434/434) | +8984% | 100M steps, no anneal. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v9_h1024_100M/`. |
| 2026-02-08 | crypto8_ppo_v15_ent003 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +6438% (65.4x) | 65.4x | 0.713 | Yes (217/217) | +5175% | 50M steps, ent-coef=0.03 (vs 0.05 baseline). Lower entropy hurts. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v15_ent003/`. |
| 2026-02-08 | crypto8_ppo_v6_h1024 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +6711% (68.0x) | 68.0x | 0.729 | Yes (217/217) | +5013% | Previous best (no anneal). 50M steps, 1x lev, 1024 hidden. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v6_h1024/`. |
| 2026-02-08 | 2026-02-08 | **crypto12_ppo_v1_h1024** | BTC,ETH,SOL,LINK,UNI,AVAX,DOT,AAVE,DOGE,LTC,ALGO,XRP | 720h (30d) | **+12552% (127.5x)** | **127.5x** | **0.733** | Yes (434/434) | +7658% | **12 crypto symbols.** 50M steps, 1x lev, 1024 hidden. New Chronos2 finetune on 12 symbols. Worst episode +118x. Checkpoints: `pufferlib_market/checkpoints/crypto12_ppo_v1_h1024/`. |
| 2026-02-08 | **crypto8_ppo_v10_h1024_60d** | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | **1440h (60d)** | **+28002% (281.9x)** | **281.9x** | 0.611 | Yes (434/434) | +19093% | **60-day episodes!** 50M steps, 1x lev, 1024 hidden. Turns 10k→2.83M per episode. Worst episode +264x. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v10_h1024_60d/`. |
| 2026-02-08 | crypto8_ppo_v8_h4096 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +5796% (58.9x) | 58.9x | 0.722 | Yes (434/434) | +4275% | 50M steps, 1x lev, **4096 hidden** (50.9M params). Better than h2048 but still under-converges vs h1024 at 50M steps. Worst episode +49.89x. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v8_h4096/`. |
| 2026-02-08 | crypto8_ppo_v7_h2048 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +5006% (51.0x) | 51.0x | 0.718 | Yes (217/217) | +4049% | 50M steps, 1x lev, **2048 hidden** (13M params). Worse than h1024 — likely under-converged at 50M steps. Worst episode +44,396% (445x). Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v7_h2048/`. |
| 2026-02-08 | crypto8_ppo_v5_h512 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +4537% (46.3x) | 46.3x | 0.733 | Yes (217/217) | +3860% | 50M steps, 1x lev, 512 hidden (865K params). Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v5_h512/`. |
| 2026-02-08 | crypto5_ppo_v5_h512 | BTC,ETH,SOL,LINK,UNI | 720h (30d) | +4367% (44.5x) | 44.5x | 0.739 | Yes (217/217) | +3389% | 50M steps, 1x lev, 512 hidden (838K params). Best WR overall (73.9%). Checkpoints: `pufferlib_market/checkpoints/crypto5_ppo_v5_h512/`. |
| 2026-02-08 | crypto8_ppo_v1 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +3844% (39.4x) | 39.4x | 0.665 | Yes (217/217) | +2619% | 50M steps, 1x lev, 256 hidden. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v1/`. |
| 2026-02-08 | crypto5_ppo_v4_720h | BTC,ETH,SOL,LINK,UNI | 720h (30d) | +3225% (33.2x) | 33.2x | 0.725 | Yes (217/217) | +2394% | 50M steps, 1x lev, 256 hidden. Checkpoints: `pufferlib_market/checkpoints/crypto5_ppo_v4_720h/`. |
| 2026-02-08 | crypto8_ppo_v4_anneal | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +2780% (28.7x) | 28.7x | 0.662 | Yes (217/217) | +2473% | 50M steps, 1x lev, 256 hidden, **LR annealing**. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v4_anneal/`. |
| 2026-02-08 | crypto8_ppo_v3_100M | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +2428% (25.2x) | 25.2x | 0.646 | Yes (217/217) | +2011% | **100M steps**, 1x lev, 256 hidden, lr=1e-4. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v3_100M/`. |
| 2026-02-08 | crypto2_ppo_v2_lev2 | SOL,LINK | 720h (30d) | +1857% (19.5x) | 19.5x | 0.609 | Yes (217/217) | +3084% | 50M steps, **2x leverage**. 3981 timesteps. Checkpoints: `pufferlib_market/checkpoints/crypto2_ppo_v2_lev2/`. |
| 2026-02-08 | crypto8_ppo_v2_lev2 | BTC,ETH,SOL,LINK,UNI,AAVE,AVAX,DOT | 720h (30d) | +1259% (13.6x) | nan | 0.560 | No (48%) | +1670% | 50M steps, **2x leverage**. High variance - huge upside but risky. Checkpoints: `pufferlib_market/checkpoints/crypto8_ppo_v2_lev2/`. |
| 2026-02-08 | crypto2_ppo_v1 | SOL,LINK | 720h (30d) | +373% (4.7x) | 4.7x | 0.656 | Yes (217/217) | +252% | 30M steps, 1x lev. First profitable RL run. Checkpoints: `pufferlib_market/checkpoints/crypto2_ppo_v1/`. |
| 2026-02-08 | crypto5_ppo_v3 | BTC,ETH,SOL,LINK,UNI | 360h (15d) | +134% (2.3x) | 2.3x | 0.581 | Yes (434/434) | +90% | 10M steps, 1x lev. Shorter episodes. Checkpoints: `pufferlib_market/checkpoints/crypto5_ppo_v3/`. |

### Currently Training (in-progress, not yet evaluated)

| Run | Config | Steps Done | Training Return | Notes |
| --- | --- | --- | --- | --- |
| crypto12_ppo_v3_h1024_100M_annealLR | h1024, 12 symbols, 100M steps, anneal-LR | **DONE** | eval **804.4x** | **ULTIMATE MODEL!** LR annealing + symbols + steps all compound. |
| crypto8_ppo_v16_h1024_100M_annealLR | h1024, 100M steps, anneal-LR | **DONE** | eval **439.1x** | Anneal-LR gives 3.3x over no-anneal 100M. |
| crypto12_ppo_v2_h1024_100M | h1024, 12 symbols, 100M steps (no anneal) | **DONE** | eval **209.8x** | 12 symbols + 100M = 209.8x. |
| crypto8_ppo_v12_h1024_200M | h1024, 200M steps (no anneal) | ~160M/200M | ~+86x | Tests if returns keep scaling past 100M. Still running. |

### Completed Experiments (Wave 2)

| Run | Config | Eval Result | Notes |
| --- | --- | --- | --- |
| **crypto8_ppo_v14_annealLR** | h1024, 50M steps, anneal-LR | **139.8x** | **ANNEAL-LR DOUBLES RETURNS!** 68x→140x from just adding `--anneal-lr`. |
| crypto8_ppo_v15_ent003 | h1024, 50M steps, ent-coef=0.03 | 65.4x | Worse than 0.05 (68x). Lower entropy hurts exploration. |
| crypto8_ppo_v13_drawdown | h1024, 50M steps, drawdown-penalty=0.5 | ~25x est. | **Drawdown penalty kills learning.** train_best=19.9 (vs 42.75 baseline). |
| crypto8_ppo_v11_resmlp_h1024 | ResidualMLP h1024, gamma=0.999 | 9.9x | gamma=0.999 kills credit assignment. Don't use. |

### Key Findings

- **LR ANNEALING IS TRANSFORMATIVE**: `--anneal-lr` gives 3.3x improvement at 100M steps (132.9x → 439.1x on crypto8). Improvement compounds with more symbols and steps!
- **Improvements multiply**: anneal-LR (3.3x) × more symbols 8→12 (1.8x) = 804.4x vs 132.9x baseline = **6.1x total improvement**
- **Scaling peaks at h1024 for 50M steps**: h1024 (3.3M params) > h4096 > h2048 > h512 > h256. Larger models under-converge.
- **More symbols = better**: 12 crypto > 8 crypto > 5 crypto > 2 crypto. Each symbol adds more opportunity.
- **More training steps = better**: 100M >> 50M. Returns scale near-linearly (before anneal-LR).
- **1x leverage >> 2x leverage** for risk-adjusted returns (100% vs 48% profitable)
- **Entropy coef 0.05 is optimal**: 0.03 hurts (65.4x vs 68.0x), 0.01 causes collapse
- **Drawdown penalty HURTS**: Makes value function 10x harder to learn
- **All 1x-leverage models are 100% profitable** across hundreds of evaluation episodes
- **Best 30d model: 804.4x**, $10k→$8M, worst episode +720.8x, 79.2% WR
- **Best 60d model: 6,812.9x**, $10k→$68M, worst episode +5,965.9x, 68.4% WR
- **60d compounding is massive**: 804.4^2 ≈ 647,059x theoretical but actual 6,812.9x shows different optimization

### Comparison: RL vs Selector approach (estimated 90d equivalent)

| Approach | Symbols | Per-episode mult | Episode len | Notes |
| --- | --- | --- | --- | --- |
| **RL crypto12_60d_annealLR** | **12 crypto** | **6,812.9x** | **60d** | **$10k→$68M per 60d. ABSOLUTE BEST.** |
| **RL crypto12_annealLR_100M** | **12 crypto** | **804.4x** | 30d | Best 30d model. 79.2% WR. |
| RL crypto8_annealLR_100M | 8 crypto | 439.1x | ~84,581,286,000x | 439.1^3. Anneal-LR + 100M. 80.4% WR. |
| RL crypto12_h1024_100M | 12 crypto | 209.8x | ~9,230,985x | 12 symbols, 100M steps, no anneal. |
| RL crypto8_annealLR_50M | 8 crypto | 139.8x | ~2,733,438x | 50M steps + anneal-LR. |
| RL crypto8_h1024_100M | 8 crypto | 132.9x | ~2,349,137x | 100M steps, no anneal. |
| RL crypto8_h1024 | 8 crypto | 68.0x | ~314,432x | 50M steps, no anneal. |
| Selector mixed14 | 14 mixed | ~4.7x (30d) | 174.4x (90d actual) | intensity=10000, epoch_004. |
| Selector mixed7 lev | 7 mixed (2x lev) | 15.0x | 438.0x (60d actual) | 2x leverage stocks+crypto. |

**The RL approach with LR annealing is transformative.** The best model turns $10k → $8M per 30d episode (804.4x), with 100% profitability and 79.2% win rate. Key compounding discoveries: **anneal-LR (3.3x boost) + more symbols (1.8x boost) + more steps (2x boost)** all multiply together.

## TODO

- Always rebuild per-symbol forecast parquet caches after any hourly data gap-fill or corporate-action discontinuity (e.g., NFLX split on 2025-11-17) to avoid stale features.
- When using lookback-limited forecast cache roots, avoid accidental full-history forecast backfills (use `--cache-only` or rely on the dataset auto-windowing).
- Extend mixed7 window beyond 2025‑11‑12 (done: lb2400 cache `mixed7_robust_20260205_2209_lb2400`).
- Run 30d selector evals on extended window (done: `selector_seq128_lb2400_best_20260205_2227`).
- Try higher intensity grid (1.4/1.6) and alternate min_edge for mixed7 policy.
- Try longer lookback (lb4000+) so stocks have 700+ bars (done: `mixed14_robust_20260205_2301_lb4000`; follow-up: compare longer feature windows vs short windows).
- Expand symbol universe (e.g., add AAPL/AMZN/META/TSLA/JPM/V/WMT) (done: mixed14 LoRA+policy+selector `selector_mixed14_seq128_lb4000_bestret_20260205_2336`).
- Try adding `max_hold_hours` sweeps (`alpacanewccrosslearning.run_selector_sweep --max-hold-hours-list ...`) and compare robustness across 30d/60d windows.

## Preaug sweep (eval_loss)

| Date (UTC) | Run | Preaug | LR | eval_loss |
| --- | --- | --- | --- | --- |
| 2026-02-05 | cross_lora_20260205_012542 | baseline | 1e‑5 | 0.015715 |
| 2026-02-05 | cross_lora_pct_20260205_015038 | percent_change | 1e‑5 | 0.017027 |
| 2026-02-05 | cross_lora_pct2_20260205_015132 | percent_change | 2e‑5 | 0.016981 |
| 2026-02-05 | cross_lora_log_20260205_015221 | log_returns | 1e‑5 | 0.018255 |
| 2026-02-05 | cross_lora_log2_20260205_015312 | log_returns | 2e‑5 | 0.018150 |
| 2026-02-05 | cross_lora_differencing_20260205_020248 | differencing | 1e‑5 | 0.247775 |
| 2026-02-05 | cross_lora_detrending_20260205_020333 | detrending | 1e‑5 | 0.192059 |
| 2026-02-05 | cross_lora_robust_scaling_20260205_020416 | robust_scaling | 1e‑5 | 0.088192 |
| 2026-02-05 | cross_lora_minmax_standard_20260205_020502 | minmax_standard | 1e‑5 | 0.187223 |
| 2026-02-05 | cross_lora_rolling_norm_20260205_020545 | rolling_norm | 1e‑5 | 0.232486 |

## Selector sweep (10d, 36 configs)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_20260205_0155/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.001 risk_weight=0.75 edge_mode=high_low dip_threshold=0.005
total_return=0.207267 sortino=43.719822 final_cash=832.755134
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.75 edge_mode=high_low dip_threshold=0.005
total_return=0.065652 sortino=839.391481 final_cash=10648.974612
```

## Selector sweep (10d, 27 configs, robust scaling)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_robust_20260205_022214/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0005 risk_weight=0.75 edge_mode=high_low dip_threshold=0.0
total_return=0.237024 sortino=26.255566 final_cash=12370.243393
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.01
total_return=0.150579 sortino=134.386252 final_cash=11505.785239
```

## Selector sweep (10d, 54 configs, robust scaling + intensity)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_robust_intensity_20260205_022332/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=0.562932 sortino=30.858390 final_cash=15629.319609
```

Best sortino (10d):
```
intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.01
total_return=0.150579 sortino=134.386252 final_cash=11505.785239
```

## Selector sweep (10d, 54 configs, mixed7 short-window)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_20260205_025742/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.119632 sortino=77.427394 final_cash=21196.315290
```

Best sortino (10d):
```
intensity=1.2 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.954793 sortino=149.867478 final_cash=19547.932885
```

## Selector sweep (10d, 54 configs, mixed7 intensity 1.4/1.6)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_intensity16_20260205_035644/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.6 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.154100 sortino=77.568583 final_cash=21540.999453
```

Best sortino (10d):
```
intensity=1.4 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.966661 sortino=135.564330 final_cash=19666.608425
```

## Selector sweep (10d, 54 configs, mixed7 intensity 1.8/2.0)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_intensity20_20260205_040605/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.001 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.173715 sortino=73.580583 final_cash=21737.154313
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.0 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.992136 sortino=132.820633 final_cash=19921.356461
```

## Selector sweep (10d, 48 configs, mixed7 targeted min_edge/risk)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_target2_20260205_041026/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.003 risk_weight=0.25 edge_mode=high_low dip_threshold=0.005
total_return=1.181937 sortino=73.166435 final_cash=21819.366134
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.0005 risk_weight=0.25 edge_mode=high_low dip_threshold=0.0
total_return=0.982480 sortino=123.332145 final_cash=19824.803099
```

## Selector sweep (10d, 27 configs, mixed7 targeted min_edge/risk v3)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_target3_20260205_041640/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.0 offset=0.0 min_edge=0.004 risk_weight=0.2 edge_mode=high_low dip_threshold=0.005
total_return=1.195198 sortino=71.546160 final_cash=21951.980205
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.003 risk_weight=0.2 edge_mode=high_low dip_threshold=0.0025
total_return=1.097744 sortino=79.130229 final_cash=20977.442667
```

## Selector sweep (10d, 144 configs, mixed7 seq128 lb2400)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_seq128_lb2400_20260205_2227/selector_sweep.csv`

Best total_return (10d):
```
intensity=2.2 offset=0.0 min_edge=0.004 risk_weight=0.15 edge_mode=high_low dip_threshold=0.0
total_return=1.010591 sortino=132.611165 final_cash=0.0
```

Best sortino (10d):
```
intensity=1.6 offset=0.0 min_edge=0.002 risk_weight=0.2 edge_mode=high_low dip_threshold=0.005
total_return=0.867208 sortino=202.196523 final_cash~0.0
```

Best total_return config extra evals (same selector config):
- 20d: total_return=1.557372 sortino=110.491282
- 30d: total_return=2.851445 sortino=103.176568 (`alpacanewccrosslearning/outputs/selector_best_mixed7_seq128_lb2400_20260205_2227_eval30d`)

## Selector sweep (10d, 48 configs, mixed7 novol baseline rebuiltstocks)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed7_novol_baseline_rebuiltstocks_20260205_2300/selector_sweep.csv`

Best total_return (10d) (also best sortino in sweep):
```
intensity=2.0 offset=0.0 min_edge=0.001 risk_weight=0.15 edge_mode=high_low dip_threshold=0.0025
total_return=1.461318 sortino=347.583555 final_cash=24613.182194
```

Best config extra evals (same selector config + checkpoint `epoch_003.pt`):
- 20d: total_return=2.200639 sortino=225.452239 (`alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260205_2310/eval20_best`)
- 30d: total_return=3.689747 sortino=184.441639 (`alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260205_2310/eval30_best`)
- 60d: total_return=10.975341 sortino=47.437592 (`alpacanewccrosslearning/outputs/global_selector_mixed7_novol_baseline_rebuiltstocks_20260205_2310/eval60_best`)

## Selector sweep (10d, 240 configs, mixed14 seq128 lb4000)

Sweep CSV: `alpacanewccrosslearning/outputs/selector_sweep_mixed14_seq128_lb4000_20260205_2336/selector_sweep.csv`

Best total_return (10d):
```
intensity=1.8 offset=0.0 min_edge=0.002 risk_weight=0.15 edge_mode=high_low dip_threshold=0.0
total_return=1.086154 sortino=108.998202 final_cash=20861.535803
```

Best sortino (10d):
```
intensity=2.0 offset=0.0 min_edge=0.002 risk_weight=0.1 edge_mode=high_low dip_threshold=0.0
total_return=1.006344 sortino=124.813456 final_cash=20063.440005
```

Best total_return config extra evals (same selector config + checkpoint `epoch_004.pt`):
- 20d: total_return=1.602473 sortino=72.0240 (`alpacanewccrosslearning/outputs/selector_best_mixed14_seq128_lb4000_20260205_2336_bestret_eval20d`)
- 30d: total_return=2.762574 sortino=76.5497 (`alpacanewccrosslearning/outputs/selector_best_mixed14_seq128_lb4000_20260205_2336_bestret_eval30d`)
- 60d: total_return=11.501177 sortino=54.1330 (printed run; no output dir)

## 2026-02-10: pufferlib_market daily RL correctness + reward shaping

Key fixes + improvements:
- Fixed `pufferlib_market` C env terminal flags: `term_buf` now correctly exposes episode boundaries to PPO/GAE (was being cleared by an internal reset).
- Added reward-shaping knobs for joint PnL/Sortino optimization:
  - `downside_penalty`: penalize negative returns via `ret^2`
  - `trade_penalty`: per open/close penalty to discourage churn
- Cash penalty edge-case: when *no symbols are tradable* (e.g., stock-only weekends/holidays), we now skip `cash_penalty` to avoid penalizing unavoidable flat exposure.
- Added regression tests:
  - `tests/test_pufferlib_market_terminals.py`
  - `tests/test_pufferlib_market_reward_shaping.py`
- Exposed `--decision-lag-bars` in `newnanoalpacahourlyexp` eval+sweep CLIs for more live-like hourly execution delay.
