# Binance Progress Log

Updated: 2026-02-06

## Notes / Fixes
- Binance hourly spot datasets are stored in `binance_spot_hourly/`; both `trainingdatahourlybinance/` and `binancetrainingdatahourly/` are symlinks to that directory (safety: avoids overwriting Alpaca CSVs via symlink aliasing).
- Fixed Chronos multi-step horizon alignment in `binanceneural/forecasts.py` (previously multi-horizon caches effectively used the 1-step prediction for every horizon). Any cached forecasts for horizons >1 generated before 2026-02-06 should be considered invalid and rebuilt before comparing multi-horizon PnL/MAE runs.
- Added stable-quote symbol utilities + tests, plus a builder script for `trainingdatahourlybinance/`.
  - New: `src/binance_symbol_utils.py`, `tests/test_binance_symbol_utils.py`, `scripts/build_trainingdatahourlybinance.py`.
- Fixed "U" stable-quote ambiguity so stock tickers like `MU`/`LULU`/`BIDU` are not misclassified as crypto or remapped into `*/U` pairs.
  - Updated: `src/stock_utils.py`, `src/chronos2_params.py`, `src/preaug/forecast_config.py`
  - New tests: `tests/test_stock_utils_remap_symbols_u_quote.py`, `tests/test_chronos2_params_crypto_detection_u_suffix.py`, `tests/test_forecast_config_asset_type_crypto_detection.py`
- `binancecrosslearning/run_global_selector.py` now supports `--frame-split {val,full}` and `--val-fraction/--validation-days` overrides. This matters for short-history symbols (e.g. U pairs): selector sims use only timestamps where actions exist, so `val` split windows can collapse to ~1 day when `sequence_length` is large relative to available history.
- Selector simulator now supports optional volume participation caps via `SelectionConfig.max_volume_fraction` (and CLI `--max-volume-fraction`). This is important for low-liquidity symbols like `BTCU` where bar volume can be far smaller than the notional implied by `initial_cash`.
- Fixed `refresh_daily_inputs.py` / `update_key_forecasts.py` to run forecast refresh under the active venv interpreter (was calling system `python`, breaking `chronos` imports) and corrected log formatting.
- Added `binancecrosslearning/` pipeline (multi-symbol Chronos2 fine-tune + global policy + selector) with Binance defaults.
- Extended crypto symbol detection + fee heuristics for stable-quote pairs (USDT/FDUSD/USDC/etc); added tests.
- Fixed `newnanoalpacahourlyexp.data.AlpacaHourlyDataModule` to compute rolling features on the full price history (forecasts are merged with `how="left"`) before applying lookback trimming.
- Windowed Chronos forecast generation for the SOL experiment data module when `max_history_days` is set (avoids full-history forecast rebuilds when using fresh cache roots).
  - Updated: `binancechronossolexperiment/data.py`, `binancechronossolexperiment/forecasts.py`
  - New test: `tests/test_binancechronossolexperiment_forecast_windowing.py`
- Fixed torch.compile CUDAGraph overwrite crash in nano sliding-window attention (no longer caches the mask on module state; trainer marks step boundaries when supported).
- Added `state_dict`-aware max_len inference when loading classic models so positional encoding buffers match checkpoint shapes.
  - Updated: `binanceneural/model.py`, `binancechronossolexperiment/inference.py`, `binanceneural/run_simulation.py`, `binanceneural/trade_binance_hourly.py`, `binanceneural/sweep.py`.
- Added max_len inference for binanceexp1 checkpoint reloads (prevents positional encoding size mismatch).
  - Updated: `binanceexp1/run_experiment.py`, `binanceexp1/sweep.py`.
- Added `value_embedding`-aware max_len inference for nano policy checkpoints (avoids size mismatch on reload).
- New PnL/RL utility library (`src/tradinglib/`) with tests (metrics, rewards, benchmarks, target gating).
- Added input-dim alignment for checkpoint embeddings when feature sets drift (pads/trims `embed.weight`).
  - Updated: `binanceneural/model.py`, `binanceneural/sweep.py`, `binanceneural/run_simulation.py`,
    `binanceexp1/sweep.py`, `binanceexp1/run_experiment.py`, `binanceexp1/run_multiasset_selector.py`.
- Added multi-asset best-trade selector simulator + CLI + tests.
  - New: `binanceneural/marketsimulator/selector.py`, `binanceexp1/run_multiasset_selector.py`,
    `tests/test_binance_multiasset_selector.py`.
- Binance data downloads now work in restricted locations by default:
  - `src/binan/binance_wrapper.py` retries with Binance.US when Binance Global returns HTTP 451 restricted-region.
  - `src/hourly_data_refresh.py` retries Binance.US for kline fallback when Binance Global is blocked.
- Binance data downloader now drops the current-hour in-progress kline from persisted CSVs (prevents leakage):
  - Updated: `binance_data_wrapper.py`
  - New tests: `tests/test_binance_data_wrapper_incomplete_hour.py`
- Binance hourly pair list updated: `MATIC/FDUSD` -> `POL/FDUSD` (Binance.US `MATICUSDT` is status BREAK / no fresh klines).
- As of 2026-02-06 (UTC), `AAVEFDUSD` and `MATICFDUSD` report status `BREAK` via `get_symbol_info`; treat them as inactive and exclude from active FDUSD training/trading lists.
- Fixed crypto symbol fixtures to classify `APT*`, `AVAX*`, `BCH*`, `POL*` (and `AEUR*`) stable-quote symbols as crypto. Any prior backtests/sims involving these symbols before 2026-02-06 should be considered suspect if market-hours/EOD logic was enabled.
- Chronos2 hourly trainer now prefers hourly preaug selections over legacy daily preaug entries (fixes ETHUSDT selecting old `rolling_norm`).
  - Updated: `chronos2_trainer.py`
  - New tests: `tests/test_chronos2_trainer_preaug_choice.py`
  - New helper: `scripts/retrain_chronos2_lora_binance_pairs.py`

## Binance FDUSD (zero-fee) hourly

### Data collection
- Script: `scripts/collect_binance_hourly_zero_fee_pairs.py`
- Output dir: `binance_spot_hourly/` (symlinks: `binancetrainingdatahourly/`, `trainingdatahourlybinance/`)
- Data source: Binance REST when available; falls back to Binance Vision (public dataset) for restricted regions / endpoint-missing symbols (FDUSD/U pairs when using Binance.US).
- Binance Vision note (UTC): as of 2026-02-06, FDUSD/U hourly bars are current through **2026-02-05 23:00** (Vision daily zip lag).
- Download summary (UTC, 1h bars):
  - AAVEFDUSD: 12283 bars, 2024-08-22 08:00 → 2026-01-16 02:00 (pair appears inactive/BREAK)
  - ADAFDUSD: 20344 bars, 2023-10-12 08:00 → 2026-02-05 23:00
  - APTFDUSD: 17800 bars, 2024-01-26 08:00 → 2026-02-05 23:00
  - ATOMFDUSD: 20176 bars, 2023-10-19 08:00 → 2026-02-05 23:00
  - AVAXFDUSD: 20176 bars, 2023-10-19 08:00 → 2026-02-05 23:00
  - BCHFDUSD: 20176 bars, 2023-10-19 08:00 → 2026-02-05 23:00
  - BNBFDUSD: 22212 bars, 2023-07-26 08:00 → 2026-02-05 23:00
  - BTCFDUSD: 22000 bars, 2023-08-04 08:00 → 2026-02-05 23:00
  - DOTFDUSD: 20008 bars, 2023-10-26 08:00 → 2026-02-05 23:00
  - ETHFDUSD: 22000 bars, 2023-08-04 08:00 → 2026-02-05 23:00
  - LINKFDUSD: 20008 bars, 2023-10-26 08:00 → 2026-02-05 23:00
  - LTCFDUSD: 20344 bars, 2023-10-12 08:00 → 2026-02-05 23:00
  - POLFDUSD: 12254 bars, 2024-09-13 10:00 → 2026-02-05 23:00
  - SOLFDUSD: 21184 bars, 2023-09-07 08:00 → 2026-02-05 23:00
  - UNIFDUSD: 16864 bars, 2024-03-05 08:00 → 2026-02-05 23:00
  - FDUSDUSDT: 22212 bars, 2023-07-26 08:00 → 2026-02-05 23:00
  - AEURUSDT: 19002 bars, 2023-12-04 10:00 → 2026-02-05 23:00

### Account conversion (USDT → FDUSD)
- Script: `scripts/convert_binance_usdt_to_fdusd.py` (dry-run by default; pass `--execute` for a live market order)
- Conversion symbol: `FDUSDUSDT`

### Chronos2 LoRA (FDUSD, per-symbol batch)
- Script: `scripts/retrain_chronos2_lora_binance_pairs.py`
- Run id: `20260206_1142_fdusd`
- Summary: `reports/chronos2_lora_binance/binance_lora_20260206_1142_fdusd_summary.md`
- Promoted configs: `hyperparams/chronos2/hourly/{BTCFDUSD,ETHFDUSD,SOLFDUSD,BNBFDUSD}.json`

| Symbol | Preaug | Val MAE% | Test MAE% | Val ret MAE | Test ret MAE | Output Dir |
|---|---|---:|---:|---:|---:|---|
| BTCFDUSD | differencing | 0.2689 | 0.5913 | 0.0027 | 0.0060 | `chronos2_finetuned/binance_lora_20260206_1142_fdusd_BTCFDUSD` |
| ETHFDUSD | differencing | 0.3863 | 0.8596 | 0.0039 | 0.0088 | `chronos2_finetuned/binance_lora_20260206_1142_fdusd_ETHFDUSD` |
| SOLFDUSD | differencing | 0.3930 | 0.8519 | 0.0040 | 0.0087 | `chronos2_finetuned/binance_lora_20260206_1142_fdusd_SOLFDUSD` |
| BNBFDUSD | differencing | 0.2423 | 0.6669 | 0.0024 | 0.0068 | `chronos2_finetuned/binance_lora_20260206_1142_fdusd_BNBFDUSD` |

### Chronos2 LoRA (FDUSD, multi-symbol pl=24)
- Fine-tune (LoRA, multi-symbol, prediction_length=24, steps=600, context=1024, lr=5e-5, batch=64, val=168h):
  - `binancecrosslearning/chronos_finetuned/chronos2_fdusd15_pl24_20260206_231800/finetuned`
  - Note: finetune symbol list included `AAVEFDUSD` + `MATICFDUSD` (both status BREAK). The forecast cache below is built for the active FDUSD list (excludes BREAK pairs, includes `POLFDUSD`).
- Forecast cache (prediction_length=24 finetune):
  - Root: `binancecrosslearning/forecast_cache_fdusd_pl24_20260206_231800/`
  - Horizons: h1/h4/h24
  - Lookback: 5000h (up to 2026-02-06 11:00 UTC)
- Forecast MAE% (close p50 vs realized close at the target horizon, over cache window):
  - BTCFDUSD: h1 0.2792, h4 0.5672, h24 1.4879
  - ETHFDUSD: h1 0.4931, h4 1.0040, h24 2.6177
  - SOLFDUSD: h1 0.5778, h4 1.1514, h24 3.1174
  - BNBFDUSD: h1 0.4446, h4 0.8582, h24 2.2679
  - LINKFDUSD: h1 0.6685, h4 1.3456, h24 3.3521
  - ADAFDUSD: h1 0.6533, h4 1.3112, h24 3.2017
  - APTFDUSD: h1 0.7288, h4 1.3411, h24 3.3980
  - ATOMFDUSD: h1 0.6011, h4 1.1229, h24 2.8131
  - AVAXFDUSD: h1 0.6700, h4 1.2975, h24 3.3929
  - BCHFDUSD: h1 0.5078, h4 0.9924, h24 2.5006
  - DOTFDUSD: h1 0.6738, h4 1.2675, h24 3.2597
  - LTCFDUSD: h1 0.5702, h4 1.0901, h24 2.7615
  - POLFDUSD: h1 0.6740, h4 1.2668, h24 3.1431
  - UNIFDUSD: h1 0.7671, h4 1.5197, h24 3.6638
- Summary MAE%: h1 mean 0.5935 median 0.6272; h4 mean 1.1525 median 1.2091; h24 mean 2.9270 median 3.1303.

### Chronos2 LoRA (per-symbol, Binance FDUSD)
- Script: `scripts/retrain_chronos2_hourly_loras.py`
- Fine-tune (LoRA, prediction_length=1, steps=2000, lr=1e-4, context=1024, val/test=168h):
  - BTCFDUSD: `chronos2_finetuned/BTCFDUSD_lora_20260206_220610/finetuned-ckpt` → val_mae%=0.2840, test_mae%=0.6335 (preaug=differencing)
  - ETHFDUSD: `chronos2_finetuned/ETHFDUSD_lora_20260206_221506/finetuned-ckpt` → val_mae%=0.3968, test_mae%=0.9089 (preaug=rolling_norm)
  - SOLFDUSD: `chronos2_finetuned/SOLFDUSD_lora_20260206_222436/finetuned-ckpt` → val_mae%=0.4512, test_mae%=0.9942 (preaug=detrending)
  - BNBFDUSD: `chronos2_finetuned/BNBFDUSD_lora_20260206_223329/finetuned-ckpt` → val_mae%=0.2783, test_mae%=0.8006 (preaug=baseline)
- Updated `hyperparams/chronos2/hourly/{BTCFDUSD,ETHFDUSD,SOLFDUSD,BNBFDUSD}.json` to point `model_id` at the finetuned checkpoints.
- Forecast cache (per-symbol LoRA models):
  - Root: `binancecrosslearning/forecast_cache_fdusd_pslora/`
  - Horizons: h1/h4/h24
  - Lookback: 5000h (up to 2026-02-06 09:00 UTC)
  - Built with `scripts/build_hourly_forecast_caches.py` (context=1024, batch=64, force_rebuild).
- Forecast MAE% (close p50 vs realized close at the target horizon, over cache window):
  - BTCFDUSD: h1 0.2784, h4 0.5665, h24 1.5039
  - ETHFDUSD: h1 0.4752, h4 1.0161, h24 2.6343
  - SOLFDUSD: h1 0.5845, h4 1.1497, h24 3.1117
  - BNBFDUSD: h1 0.4434, h4 0.8622, h24 2.2718
- Note: this per-symbol LoRA (trained with prediction_length=1) currently underperforms the multi-symbol FDUSD LoRA above on h4/h24; next step is retraining with prediction_length=24 (or multi-symbol covariate fine-tune) before using it for PnL.

### Per-symbol policy (binancechronossolexperiment2, FDUSD)
- Forecast cache root: `binancechronossolexperiment2/forecast_cache_fdusd_lora_20260206_1142` (h1/h4/h24; ~97d lookback)

| Symbol | Total Return | Sortino | Last 2d Return | Run |
|---|---:|---:|---:|---|
| BTCFDUSD | 0.276788 | 60.372765 | 0.029840 | `btcfdsu_lora1142_nano_v2_muon_90d_20260206_1209` |
| ETHFDUSD | 0.489481 | 72.549181 | 0.094878 | `ethfdsu_lora1142_nano_v2_muon_90d_20260206_1213` |
| SOLFDUSD | 0.360591 | 59.817919 | 0.069591 | `solfdsu_lora1142_nano_v2_muon_90d_20260206_1217` |
| BNBFDUSD | 0.251770 | 65.636306 | 0.011541 | `bnbfdsu_lora1142_nano_v2_muon_90d_20260206_1221` |

### Global policy (binancecrosslearning, FDUSD)
- Train run: `binance_cross_global_fdusd_lora1142_20260206_1247` (6 epochs, seq=72, nano, muon_mix, horizons 1/4/24, cache-only)
- Checkpoint: `binancecrosslearning/checkpoints/binance_cross_global_fdusd_lora1142_20260206_1247/epoch_005.pt`
- Train script eval (BTCFDUSD, last 30d): total_return=0.7290, sortino=53.4784
- Selector eval (shared cash, last 30d, 4 symbols):
  - Output dir: `binancecrosslearning/outputs/binance_cross_global_fdusd_lora1142_20260206_1247_selector30d`
  - total_return=0.0893, sortino=41.5856 (open_symbol=SOLFDUSD)
- Selector eval (shared cash, last 30d, 4 symbols, tuned threshold):
  - Checkpoint: `binancecrosslearning/checkpoints/binance_cross_global_fdusd_lora1142_20260206_1247/epoch_005.pt`
  - min_edge=0.0011, edge_mode=high_low, horizon=1
  - total_return=0.2243, sortino=202.7667 (final_cash=12242.8095, open_symbol=SOLFDUSD)
- Short-MA variant (to reuse the same feature set for U transfer tests; horizons 1/4, seq=48):
  - Train run: `binance_cross_global_fdusd_shortma_h14_seq48_20260206_1311`
  - Checkpoint: `binancecrosslearning/checkpoints/binance_cross_global_fdusd_shortma_h14_seq48_20260206_1311/epoch_006.pt`
  - Train script eval (BTCFDUSD, last 30d): total_return=0.5118, sortino=38.2844
  - Transfer selector eval on U (last 7d, BTCU/ETHU/SOLU/BNBU): total_return=-0.0023, sortino=-16.3569 (open_symbol=BTCU)

- Train run (8 epochs, seq=96, nano, muon_mix, horizons 1/4/24, cache-only, no-compile; forecast_cache_fdusd_pl24_20260206_231800):
  - `binance_cross_global_fdusd_pl24cache_20260206_231800_nocompile`
  - Checkpoint: `binancecrosslearning/checkpoints/binance_cross_global_fdusd_pl24cache_20260206_231800_nocompile/epoch_004.pt`
  - Train script eval (BTCFDUSD, last 30d): total_return=0.9125, sortino=169.3030
- Selector per-symbol eval (last 30d, initial_cash=10_000, maker_fee=0.0; intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.5):
  - ETHFDUSD: total_return=0.2540, sortino=62.7863
  - POLFDUSD: total_return=0.2479, sortino=75.1713
  - SOLFDUSD: total_return=0.1841, sortino=172.4909
  - LINKFDUSD: total_return=0.1806, sortino=34.3439
  - ADAFDUSD: total_return=0.1698, sortino=93.8072
  - DOTFDUSD: total_return=0.1513, sortino=27.5322
  - LTCFDUSD: total_return=0.1501, sortino=92.7471
  - BTCFDUSD: total_return=0.1453, sortino=195.7072
  - BCHFDUSD: total_return=0.1065, sortino=187.6229
  - AVAXFDUSD: total_return=0.0993, sortino=24.9872
  - BNBFDUSD: total_return=0.0947, sortino=38.5976
  - APTFDUSD: total_return=0.0503, sortino=4.3109
  - ATOMFDUSD: total_return=0.0435, sortino=14.9628
  - UNIFDUSD: total_return=0.0299, sortino=6.5480
- Selector eval (shared cash, last 30d, 14 symbols, initial_cash=10_000, maker_fee=0.0; intensity=1.0 offset=0.0 min_edge=0.0 risk_weight=0.5):
  - total_return=0.3305, sortino=51.1465, final_cash=6091.7925, open_symbol=SOLFDUSD
- Selector sweep highlight (same 14 symbols, last 30d, maker_fee=0.0; min_edge=0.0 risk_weight=0.5):
  - Best found: intensity=1.2 offset=0.0 → total_return=3.8566, sortino=67.4564, final_cash=16975.0014, open_symbol=SOLFDUSD
- Selector eval (shared cash, last 90d, 14 symbols; validation_days=120, maker_fee=0.0; min_edge=0.0 risk_weight=0.5):
  - intensity=1.0 offset=0.0 → total_return=0.2598, sortino=124.3081, final_cash=12598.1507, open_symbol=POLFDUSD
  - intensity=1.2 offset=0.0 → total_return=1.8496, sortino=147.2874, final_cash=6587.8356, open_symbol=SOLFDUSD

## Binance U (zero-fee) hourly

### Data collection
- Script: `scripts/collect_binance_hourly_zero_fee_pairs.py`
- Output dir: `binance_spot_hourly/` (symlinks: `binancetrainingdatahourly/`, `trainingdatahourlybinance/`)
- Binance Vision note (UTC): as of 2026-02-06, U hourly bars are current through **2026-02-05 23:00**.
- Local refresh (UTC): `python scripts/binance_auto_pipeline.py --pair-list u --update-data` pulled bars through **2026-02-06 14:00**.
- Download summary (UTC, 1h bars):
  - BTCU: 415 bars, 2026-01-20 08:00 → 2026-02-06 14:00
  - ETHU: 247 bars, 2026-01-27 08:00 → 2026-02-06 14:00
  - SOLU: 247 bars, 2026-01-27 08:00 → 2026-02-06 14:00
  - BNBU: 247 bars, 2026-01-27 08:00 → 2026-02-06 14:00
  - UUSDT: 583 bars, 2026-01-13 08:00 → 2026-02-06 14:00

### Account conversion (USDT → U)
- Script: `scripts/convert_binance_usdt_to_u.py` (dry-run by default; pass `--execute` for a live market order)
- Conversion symbol: `UUSDT`

### Chronos2 LoRA (U, per-symbol batch)
- Script: `scripts/retrain_chronos2_lora_binance_pairs.py`
- Run id: `20260206_1150_u`
- Summary: `reports/chronos2_lora_binance/binance_lora_20260206_1150_u_summary.md`
- Promoted configs: `hyperparams/chronos2/hourly/{BTCU,ETHU,SOLU,BNBU,UUSDT}.json`

| Symbol | Preaug | Val MAE% | Test MAE% | Val ret MAE | Test ret MAE | Output Dir |
|---|---|---:|---:|---:|---:|---|
| BTCU | detrending | 0.5294 | 1.0014 | 0.0053 | 0.0101 | `chronos2_finetuned/binance_lora_20260206_1150_u_BTCU` |
| ETHU | differencing | 0.8110 | 1.1172 | 0.0082 | 0.0112 | `chronos2_finetuned/binance_lora_20260206_1150_u_ETHU` |
| SOLU | detrending | 0.9711 | 2.0238 | 0.0098 | 0.0207 | `chronos2_finetuned/binance_lora_20260206_1150_u_SOLU` |
| BNBU | baseline | 0.7638 | 1.5529 | 0.0076 | 0.0158 | `chronos2_finetuned/binance_lora_20260206_1150_u_BNBU` |
| UUSDT | rolling_norm | 0.0118 | 0.0119 | 0.0001 | 0.0001 | `chronos2_finetuned/binance_lora_20260206_1150_u_UUSDT` |

### Per-symbol policy (binancechronossolexperiment2, U)
- Forecast cache roots:
  - h1/h4/h24: `binancechronossolexperiment2/forecast_cache_u_lora_20260206_1150` (BTCU/ETHU)
  - h1/h4: `binancechronossolexperiment2/forecast_cache_u_lora_20260206_1150_h14` (ETHU/SOLU/BNBU)

| Symbol | Total Return | Sortino | Run |
|---|---:|---:|---|
| BTCU | -0.035896 | -9.532768 | `btcu_lora1150_nano_v2_muon_short2_20260206_1226` |
| ETHU | -0.006701 | -5.722986 | `ethu_lora1150_nano_v2_muon_short3_20260206_1232` |
| SOLU | -0.053301 | -28.874740 | `solu_lora1150_nano_v2_muon_short3_20260206_1236` |
| BNBU | -0.065375 | -32.911935 | `bnbu_lora1150_nano_v2_muon_short3_20260206_1244` |

### Global policy (binancecrosslearning, U)
- Prebuilt/filled forecast cache for h1/h4 now includes BTCU/ETHU/SOLU/BNBU:
  - `binancechronossolexperiment2/forecast_cache_u_lora_20260206_1150_h14`
- Train run: `binance_cross_global_u_shortma_h14_seq48_20260206_1314` (30 epochs, seq=48, MA windows 24/72/168, horizons 1/4, cache-only)
  - Checkpoint: `binancecrosslearning/checkpoints/binance_cross_global_u_shortma_h14_seq48_20260206_1314/epoch_024.pt`
  - Train script eval (BTCU, last 7d): total_return=-0.0269, sortino=-13.0422
  - Selector eval (shared cash, last 7d, BTCU/ETHU/SOLU/BNBU): total_return=-0.0484, sortino=-17.1613 (open_symbol=SOLU)

### BTCU selector tuning (transfer policy)
- Policy checkpoint (trained on FDUSD symbols, feature-matched to U):
  - `binancecrosslearning/checkpoints/binance_cross_global_fdusd_shortma_h14_seq48_20260206_1311/epoch_006.pt`
- Forecast cache root (U LoRA, h1/h4): `binancechronossolexperiment2/forecast_cache_u_lora_20260206_1150_h14`

**Best BTCU sim so far (holdout-style val split):**
- Run selector with: `--frame-split val --val-fraction 0.5 --edge-mode close --min-edge 0.0064 --intensity-scale 20 --price-offset-pct 0.00025`
- BTCU (2026-02-02 00:00 → 2026-02-05 23:00, 78 hours): total_return=0.0757, sortino=22.8950, max_drawdown=-0.0060
- Volume-capped variant (more realistic fills): add `--max-volume-fraction 0.2` → total_return=0.0462, sortino=12.2726 (same window; ends holding inventory; portfolio_end=10462.06)

**Full-frame backtest (train+val):**
- Run selector with: `--frame-split full --edge-mode close --min-edge 0.0064 --intensity-scale 20 --price-offset-pct 0.00025`
- BTCU (2026-01-31 15:00 → 2026-02-05 23:00, 106 hours): total_return=0.0986, sortino=11.3014, max_drawdown=-0.0159

### U selector sweep (global U policy, base Chronos2 cache; last 7d)
- Data refresh: `python scripts/binance_auto_pipeline.py --pair-list u --update-data` (U bars through 2026-02-06 14:00 UTC).
- Forecast cache root (h1/h4; built from base `amazon/chronos-2` due to missing U hourly configs in this workspace):
  - `binancecrosslearning/forecast_cache_u_pslora_20260206_151313_h14`
  - MAE%: BTCU h1 0.3837 h4 0.8761; ETHU h1 1.0750 h4 2.5006; SOLU h1 0.9778 h4 1.9014; BNBU h1 0.7092 h4 1.5787
- Policy checkpoint:
  - `binancecrosslearning/checkpoints/binance_cross_global_u_20260206_nocompile2/epoch_017.pt`
- Sim config:
  - `--frame-split full --eval-days 7 --forecast-horizons 1,4 --moving-average-windows 24,72,168 --min-history-hours 48 --maker-fee 0.0 --edge-mode close --risk-weight 0.5`
- Baseline (no volume cap): intensity=1 offset=0 min_edge=0 → total_return=0.0328, sortino=58.6975 (open_symbol=BTCU)
- Best found (no volume cap): intensity=20 offset=0.00025 min_edge=0.002 → total_return=0.2118, sortino=32.8365 (open_symbol=BTCU; `final_cash=0` implies fully deployed, so treat as optimistic on illiquid pairs)
- With `--max-volume-fraction 0.1` (more realistic fills): intensity=20 offset=0 min_edge=0.002 → total_return=0.1050, sortino=19.5354 (open_symbol=BTCU)

## Chronos2 LoRA (hourly, Alpaca data)
- BTCUSD LoRA: `chronos2_finetuned/BTCUSD_lora_20260203_051412` → Validation MAE% 0.2785, preaug=diff
- ETHUSD LoRA: `chronos2_finetuned/ETHUSD_lora_20260203_051846` → Validation MAE% 0.4450, preaug=diff
- LINKUSD LoRA: `chronos2_finetuned/LINKUSD_lora_20260203_052258` → Validation MAE% 0.4456, preaug=diff
- SOLUSD LoRA: `chronos2_finetuned/SOLUSD_lora_20260203_052715` → Validation MAE% 0.4337, preaug=diff
- UNIUSD LoRA: `chronos2_finetuned/UNIUSD_lora_20260203_091020` → Validation MAE% 0.5416, preaug=detrending
- NFLX LoRA: `chronos2_finetuned/NFLX_lora_20260203_091648` → Validation MAE% 0.1050, preaug=differencing
- NVDA LoRA: `chronos2_finetuned/NVDA_lora_20260203_092111` → Validation MAE% 0.1333, preaug=differencing
- Updated `hyperparams/chronos2/hourly/{BTCUSD,ETHUSD,LINKUSD,SOLUSD,UNIUSD,NFLX,NVDA}.json` to point `model_id` at the LoRA finetuned checkpoints.
- Forecast cache rebuilt for h1 only (BTCUSD/ETHUSD/LINKUSD/SOLUSD/UNIUSD/NFLX/NVDA); h24 recompute deferred (very heavy with long contexts).
- Supervisor `binanceexp1-solusd` updated to cross-pair BTCUSD/ETHUSD/LINKUSD checkpoints (horizon=1, seq=96).

## Chronos2 LoRA (hourly, Binance.US data)
- Data: refreshed `trainingdatahourlybinance/*USDT.csv` to 2026-02-06 09:00 UTC (Binance.US; Global endpoint restricted-region).
- Batch LoRA run `20260206_0818` (15 symbols, `--preaug-eval`; selection metric=val MAE%):
  - Summary: `reports/chronos2_lora_binance/binance_lora_20260206_0818_summary.md`
  - Reports: `hyperparams/chronos2/hourly_lora/*_lora_binance_lora_20260206_0818_*.json`
  - Outputs: `chronos2_finetuned/binance_lora_20260206_0818_<SYMBOL>/finetuned-ckpt`
  - Promoted configs: `hyperparams/chronos2/hourly/{AAVEUSDT,ADAUSDT,APTUSDT,ATOMUSDT,AVAXUSDT,BCHUSDT,BNBUSDT,BTCUSDT,DOTUSDT,ETHUSDT,LINKUSDT,LTCUSDT,POLUSDT,SOLUSDT,UNIUSDT}.json`

| Symbol | Preaug | Val MAE% | Test MAE% | Val ret MAE | Test ret MAE |
|---|---|---:|---:|---:|---:|
| BTCUSDT | differencing | 0.2964 | 0.6417 | 0.0030 | 0.0066 |
| APTUSDT | differencing | 0.3200 | 0.7206 | 0.0032 | 0.0076 |
| BNBUSDT | robust_scaling | 0.3342 | 0.9440 | 0.0034 | 0.0097 |
| LINKUSDT | differencing | 0.4170 | 0.9127 | 0.0042 | 0.0093 |
| AAVEUSDT | differencing | 0.4187 | 1.0117 | 0.0042 | 0.0103 |
| ETHUSDT | differencing | 0.4122 | 0.9218 | 0.0041 | 0.0095 |
| SOLUSDT | differencing | 0.4359 | 0.9292 | 0.0044 | 0.0096 |
| AVAXUSDT | differencing | 0.4549 | 0.8196 | 0.0046 | 0.0083 |
| ADAUSDT | differencing | 0.5210 | 0.9559 | 0.0052 | 0.0097 |
| UNIUSDT | differencing | 0.5560 | 1.0655 | 0.0056 | 0.0108 |
| BCHUSDT | detrending | 0.5805 | 1.1391 | 0.0059 | 0.0115 |
| DOTUSDT | log_returns | 0.6479 | 1.1964 | 0.0065 | 0.0121 |
| POLUSDT | differencing | 0.6772 | 1.1958 | 0.0068 | 0.0122 |
| LTCUSDT | log_returns | 0.7151 | 1.1262 | 0.0072 | 0.0114 |
| ATOMUSDT | differencing | 0.8981 | 1.2665 | 0.0090 | 0.0128 |

- Previous quick runs (superseded by `20260206_0818`):
  - BTCUSDT LoRA: `chronos2_finetuned/BTCUSDT_lora_20260206_074018` → Validation MAE% 0.2960, preaug=differencing
  - SOLUSDT LoRA: `chronos2_finetuned/SOLUSDT_lora_20260206_074244` → Validation MAE% 0.4359, preaug=differencing
  - ETHUSDT LoRA: `chronos2_finetuned/ETHUSDT_lora_20260206_075058` → Validation MAE% 0.4127, preaug=differencing
  - POLUSDT LoRA: `chronos2_finetuned/POLUSDT_lora_20260206_073634` → Validation MAE% 0.6910, preaug=baseline

## Experiments (Chronos2 + Binance neural)

### Policy batch (LoRA 20260206_0818, nano v2 + muon, 90d history)
- Summary: `reports/binance_policy/binance_policy_lora0818_nano_v2_muon_90d_20260206_summary.md`
- Forecast cache root: `binancechronossolexperiment2/forecast_cache_lora_20260206_0818` (h1/h4/h24; per-symbol parquets)

| Symbol | Total Return | Sortino | Last 2d Return | Run |
|---|---:|---:|---:|---|
| ETHUSDT | 0.397039 | 48.423498 | 0.072744 | `ethusdt_lora0818_nano_v2_muon_90d_20260206_0926` |
| SOLUSDT | 0.381142 | 52.623876 | 0.128440 | `solusdt_lora0818_nano_v2_muon_90d_20260206_0930` |
| BNBUSDT | 0.369530 | 34.431249 | 0.028878 | `bnbusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| BTCUSDT | 0.313967 | 47.858053 | 0.084447 | `btcusdt_lora0818_nano_v2_muon_90d_20260206_0918` |
| BCHUSDT | 0.303637 | 24.429731 | 0.067853 | `bchusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| LTCUSDT | 0.287870 | 24.536757 | 0.002231 | `ltcusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| ATOMUSDT | 0.257163 | 13.519264 | 0.041600 | `atomusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| POLUSDT | 0.236184 | 13.309563 | 0.085585 | `polusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| ADAUSDT | 0.214842 | 16.769195 | 0.035352 | `adausdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| LINKUSDT | 0.078683 | 7.235045 | 0.007517 | `linkusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| AVAXUSDT | 0.017097 | 1.715396 | 0.008183 | `avaxusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| DOTUSDT | -0.059666 | -1.980600 | -0.056529 | `dotusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| AAVEUSDT | -0.073214 | -3.007001 | -0.031294 | `aaveusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| UNIUSDT | -0.092815 | -4.354703 | -0.011269 | `uniusdt_lora0818_nano_v2_muon_90d_20260206_0933` |
| APTUSDT | -0.180532 | -5.029216 | -0.097441 | `aptusdt_lora0818_nano_v2_muon_90d_20260206_0933` |

### Baseline (classic, full history)
- Run: `chronos_sol_20260202_041139`
- Metrics (10-day test): total_return=0.3433135070, sortino=136.8685975259, annualized_return=49893.15079393116
- Last 2 days: total_return=0.08457621078, sortino=218.4418841462, annualized_return=2722471.909152695

### Nano + Muon (90d history)
- Run: `chronos_sol_nano_muon_20260202_230000`
- Command:
  `python -m binancechronossolexperiment.run_experiment --symbol SOLUSDT --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --model-id chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt --context-hours 3072 --chronos-batch-size 32 --horizons 1 --sequence-length 72 --val-days 20 --test-days 10 --max-history-days 90 --epochs 6 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --optimizer muon_mix --model-arch nano --num-kv-heads 2 --mlp-ratio 4.0 --logits-softcap 12.0 --muon-lr 0.02 --muon-momentum 0.95 --muon-ns-steps 5 --cache-only --no-compile --run-name chronos_sol_nano_muon_20260202_230000`
- Metrics (10-day test): total_return=0.1933922754, sortino=50.0260710990, annualized_return=651.0995876932
- Last 2 days: total_return=0.045477, sortino=70.023416, annualized_return=3347.674857

### Nano + AdamW (90d history)
- Run: `chronos_sol_nano_adamw_20260202_230500`
- Metrics (10-day test): total_return=0.2216511106, sortino=47.5818631247, annualized_return=1536.6274827896
- Last 2 days: total_return=0.044726, sortino=46.942293, annualized_return=2936.213771

### Nano + Muon (365d history)
- Run: `chronos_sol_nano_muon_20260202_235000`
- Metrics (10-day test): total_return=0.178760, sortino=33.205916, annualized_return=413.887796
- Last 2 days: total_return=0.057223, sortino=166.957110, annualized_return=25727.928457

### Classic + Muon (365d history, retry after pos-encoding fix)
- Run: `chronos_sol_classic_muon_20260202_235500_retry`
- Metrics (10-day test): total_return=0.131389, sortino=38.260542, annualized_return=91.263087
- Last 2 days: total_return=-0.002678, sortino=-2.415733, annualized_return=-0.386949

### Nano + Muon (full history)
- Run: `chronos_sol_nano_muon_full_20260203_001000`
- Metrics (10-day test): total_return=0.163166, sortino=133.889350, annualized_return=253.654864
- Last 2 days: total_return=0.049308, sortino=3301.882512, annualized_return=6527.535363

### Nano + AdamW (full history)
- Run: `chronos_sol_nano_adamw_full_20260203_003000`
- Metrics (10-day test): total_return=0.131405, sortino=106.950196, annualized_return=91.309086
- Last 2 days: total_return=0.039455, sortino=388.584324, annualized_return=1165.838243

### Nano + Muon (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_muon_h1_4_24_20260203_010000`
- Metrics (10-day test): total_return=0.139216, sortino=92.137741, annualized_return=117.786575
- Last 2 days: total_return=0.029039, sortino=1656.626735, annualized_return=184.700605

### Nano + AdamW (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_adamw_h1_4_24_20260203_092000`
- Metrics (10-day test): total_return=0.150219, sortino=139.825174, annualized_return=167.955148
- Last 2 days: total_return=0.056290, sortino=1120.777238, annualized_return=21899.280913

### Nano v2 (nanochat-style upgrades, multi-horizon 1/4/24, full history)
- Run: `chronos_sol_v2_nano_muon_h1_4_24_20260203_110000`
- Settings: muon_mix, residual scalars, value embeddings (every layer), attention_window=64, weight_decay linear->0, amp bfloat16
- Metrics (10-day test): total_return=0.135517, sortino=153.291137, annualized_return=104.436765
- Last 2 days: total_return=0.008194, sortino=376.950354, annualized_return=3.434440

### Nano v2 sweep (4-epoch runs, full history)
- Win=0 alt2: `chronos_sol_v2_win0_alt2_20260203_120000`
  - Metrics (10-day test): total_return=0.168163, sortino=127.621894, annualized_return=296.979450
  - Last 2 days: total_return=0.034318, sortino=4112.864558, annualized_return=471.432240
- Win=128 alt2: `chronos_sol_v2_win128_alt2_20260203_123000`
  - Metrics (10-day test): total_return=0.176363, sortino=131.492666, annualized_return=384.060802
  - Last 2 days: total_return=0.027708, sortino=5468.897139, annualized_return=145.641555
- Win=64 alt1: `chronos_sol_v2_win64_alt1_20260203_130000`
  - Metrics (10-day test): total_return=0.135580, sortino=388.062129, annualized_return=104.651151
  - Last 2 days: total_return=0.026850, sortino=2206.419941, annualized_return=124.898822
- Win=128 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win128_skip01_20260203_133000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=96 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win96_skip01_20260203_150000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=192 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win192_skip01_20260203_153000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=32 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win32_skip01_20260203_160000`
  - Metrics (10-day test): total_return=0.140784, sortino=282.894742, annualized_return=123.930345
  - Last 2 days: total_return=0.029802, sortino=311.508539, annualized_return=211.610349
- Win=48 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win48_skip01_20260203_163000`
  - Metrics (10-day test): total_return=0.152436, sortino=122.565052, annualized_return=180.312070
  - Last 2 days: total_return=0.037831, sortino=14278.210667, annualized_return=876.298951
- Win=128 alt2 + skip_scale_init=0.05: `chronos_sol_v2_win128_skip005_20260203_170000`
  - Metrics (10-day test): total_return=0.148934, sortino=91.914223, annualized_return=161.174633
  - Last 2 days: total_return=0.034244, sortino=12127.487356, annualized_return=465.354448
- Win=128 alt2 + skip_scale_init=0.2: `chronos_sol_v2_win128_skip02_20260203_173000`
  - Metrics (10-day test): total_return=0.153551, sortino=152.815771, annualized_return=186.853345
  - Last 2 days: total_return=0.033610, sortino=2113.671477, annualized_return=415.989073
- Seq=96, Win=64 alt2 + skip_scale_init=0.1: `chronos_sol_v2_seq96_win64_skip01_20260203_180000`
  - Metrics (10-day test): total_return=0.154001, sortino=89.258249, annualized_return=189.558575
  - Last 2 days: total_return=0.060890, sortino=1473.177357, annualized_return=48399.069467

Note: attention_window >= sequence_length behaves like full attention, so Win=96/128/192 runs matched the Win=128 baseline when other settings were identical.

### Nano v2 follow-up sweep (attention window/value embedding/skip-scale, 4-epoch)
- Win=96 + skip_scale_init=0.1: `chronos_sol_v2_win96_skip01_20260203_194200`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=192 + skip_scale_init=0.1: `chronos_sol_v2_win192_skip01_20260203_195500`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=256 + skip_scale_init=0.1: `chronos_sol_v2_win256_skip01_20260203_200500`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=96 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win96_skip01_ve3_20260203_201700`
  - Metrics (10-day test): total_return=0.118162, sortino=170.433597, annualized_return=58.955209
  - Last 2 days: total_return=0.035090, sortino=18497.570350, annualized_return=540.389433
- Win=128 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win128_skip01_ve3_20260203_210500`
  - Metrics (10-day test): total_return=0.118162, sortino=170.433597, annualized_return=58.955209
  - Last 2 days: total_return=0.035090, sortino=18497.570350, annualized_return=540.389433
- Win=128 + skip_scale_init=0.05: `chronos_sol_v2_win128_skip005_20260203_203000`
  - Metrics (10-day test): total_return=0.148934, sortino=91.914223, annualized_return=161.174633
  - Last 2 days: total_return=0.034244, sortino=12127.487356, annualized_return=465.354448
- Win=128 + skip_scale_init=0.2: `chronos_sol_v2_win128_skip02_20260203_204300`
  - Metrics (10-day test): total_return=0.153551, sortino=152.815771, annualized_return=186.853345
  - Last 2 days: total_return=0.033610, sortino=2113.671477, annualized_return=415.989073
- Win=64 + skip_scale_init=0.1: `chronos_sol_v2_win64_skip01_20260203_212000`
  - Metrics (10-day test): total_return=0.102255, sortino=103.215131, annualized_return=34.460601
  - Last 2 days: total_return=0.007902, sortino=535.322223, annualized_return=3.205639

### Nano v2 micro-sweep (attention window/value embedding/Muon LR, 4-epoch)
- Win=24 + skip_scale_init=0.1: `chronos_sol_v2_win24_skip01_20260203_190000`
  - Metrics (10-day test): total_return=0.131183, sortino=49.734845, annualized_return=90.647583
  - Last 2 days: total_return=0.029858, sortino=93.975619, annualized_return=213.718822
- Win=40 + skip_scale_init=0.1: `chronos_sol_v2_win40_skip01_20260204_000000`
  - Metrics (10-day test): total_return=0.156633, sortino=40.493034, annualized_return=206.152934
  - Last 2 days: total_return=0.011449, sortino=8.054279, annualized_return=6.985773
- Win=56 + skip_scale_init=0.1: `chronos_sol_v2_win56_skip01_20260204_003000`
  - Metrics (10-day test): total_return=0.183902, sortino=151.187906, annualized_return=485.653023
  - Last 2 days: total_return=0.053722, sortino=7227.691172, annualized_return=14043.617352
- Win=64 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win64_vale3_skip01_20260204_010000`
  - Metrics (10-day test): total_return=0.106891, sortino=204.396658, annualized_return=40.358946
  - Last 2 days: total_return=0.028697, sortino=7396.888054, annualized_return=173.771543
- Win=64 + skip_scale_init=0.1 + value_embedding_every=4: `chronos_sol_v2_win64_vale4_skip01_20260204_013000`
  - Metrics (10-day test): total_return=0.149532, sortino=116.811572, annualized_return=164.296970
  - Last 2 days: total_return=0.052753, sortino=4335.442502, annualized_return=11872.874383
- Win=128 + skip_scale_init=0.1 + value_embedding_every=1: `chronos_sol_v2_vale1_skip01_20260204_011846`
  - Metrics (10-day test): total_return=0.185369, sortino=160.869717, annualized_return=508.252224
  - Last 2 days: total_return=0.057419, sortino=21575.395855, annualized_return=26614.317587
- Win=128 + skip_scale_init=0.05 + value_embedding_every=1: `chronos_sol_v2_vale1_skip005_20260204_012841`
  - Metrics (10-day test): total_return=0.128222, sortino=88.884578, annualized_return=82.254881
  - Last 2 days: total_return=0.023395, sortino=4226.743686, annualized_return=67.061177
- Win=128 + skip_scale_init=0.15 + value_embedding_every=1: `chronos_sol_v2_vale1_skip015_20260204_013817`
  - Metrics (10-day test): total_return=0.157671, sortino=113.337640, annualized_return=213.079502
  - Last 2 days: total_return=0.031390, sortino=13977.423338, annualized_return=280.653506
- Win=128 + skip_scale_init=0.1 + muon_lr=0.015: `chronos_sol_v2_win128_skip01_muon015_20260204_020000`
  - Metrics (10-day test): total_return=0.144029, sortino=114.980093, annualized_return=137.636905
  - Last 2 days: total_return=0.032975, sortino=1601.106040, annualized_return=371.751577
- Win=128 + skip_scale_init=0.1 + muon_lr=0.03: `chronos_sol_v2_win128_skip01_muon03_20260204_023000`
  - Metrics (10-day test): total_return=0.109021, sortino=65.846097, annualized_return=43.377910
  - Last 2 days: total_return=0.028804, sortino=6587.062715, annualized_return=177.116790
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.93: `chronos_sol_v2_muon093_20260204_014809`
  - Metrics (10-day test): total_return=0.112421, sortino=140.816244, annualized_return=48.646180
  - Last 2 days: total_return=0.032995, sortino=4038.468056, annualized_return=373.074915
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.97: `chronos_sol_v2_muon097_20260204_015606`
  - Metrics (10-day test): total_return=0.193740, sortino=152.846581, annualized_return=658.107035
  - Last 2 days: total_return=0.048688, sortino=5613.438236, annualized_return=5860.035917
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.98: `chronos_sol_v2_muon098_20260204_021639`
  - Metrics (10-day test): total_return=0.172828, sortino=95.283727, annualized_return=343.849245
  - Last 2 days: total_return=0.044075, sortino=14528.643033, annualized_return=2620.416473
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.99: `chronos_sol_v2_muon099_20260204_021641`
  - Metrics (10-day test): total_return=0.129935, sortino=140.445793, annualized_return=87.015755
  - Last 2 days: total_return=0.007348, sortino=1231.628070, annualized_return=2.804469
- Win=128 + skip_scale_init=0.1 + muon_lr=0.025: `chronos_sol_v2_muon025_20260204_021642`
  - Metrics (10-day test): total_return=0.122925, sortino=129.919800, annualized_return=69.063854
  - Last 2 days: total_return=0.027347, sortino=12956.396462, annualized_return=136.527182
- Win=128 + skip_scale_init=0.1 + muon_lr=0.035: `chronos_sol_v2_muon035_20260204_021643`
  - Metrics (10-day test): total_return=0.122698, sortino=58.989053, annualized_return=68.546359
  - Last 2 days: total_return=0.027950, sortino=13908.088305, annualized_return=152.062641
- Win=128 + skip_scale_init=0.1 + context_hours=2048: `chronos_sol_v2_ctx2048_20260204_021644`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=128 + skip_scale_init=0.1 + context_hours=4096: `chronos_sol_v2_ctx4096_20260204_021645`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368

Note: context_hours 2048/4096 matched the best baseline exactly (cache-only run likely reused identical forecast cache).

### Nano v2 sequence length sweep (full attention, 4-epoch)
- Seq=96, Win=96 + skip_scale_init=0.1: `chronos_sol_v2_seq96_win96_skip01_20260204_005446`
  - Metrics (10-day test): total_return=0.147737, sortino=105.345177, annualized_return=155.092648
  - Last 2 days: total_return=0.052962, sortino=235.802923, annualized_return=12311.639256
- Seq=128, Win=128 + skip_scale_init=0.1: `chronos_sol_v2_seq128_win128_skip01_20260204_010516`
  - Metrics (10-day test): total_return=0.150824, sortino=94.192548, annualized_return=171.243700
  - Last 2 days: total_return=0.062229, sortino=202.545430, annualized_return=60922.071574

Best PnL unchanged after micro-sweep + seq-length sweep: total_return=0.224781 (skip_scale_init=0.1, attention_window >= 96, seq_len=72).

### Nano v2 longer test window (20-day test / 20-day val, 4-epoch)
- Baseline (skip_scale_init=0.1): `chronos_sol_v2_test20_skip01_20260204_042613`
  - Metrics (20-day test): total_return=0.283456, sortino=104.303940, annualized_return=94.958200
  - Last 2 days: total_return=0.039501, sortino=10193.434517, annualized_return=1175.415846
- value_embedding_every=1 (skip_scale_init=0.1): `chronos_sol_v2_test20_vale1_skip01_20260204_043452`
  - Metrics (20-day test): total_return=0.272807, sortino=72.602345, annualized_return=81.396613
  - Last 2 days: total_return=0.033666, sortino=10424.777798, annualized_return=420.089043
- muon_momentum=0.97 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon097_20260204_044258`
  - Metrics (20-day test): total_return=0.324214, sortino=92.312041, annualized_return=168.973961
  - Last 2 days: total_return=0.031608, sortino=1188.725587, annualized_return=291.713480
- attention_window=56 (skip_scale_init=0.1): `chronos_sol_v2_test20_win56_skip01_20260204_045236`
  - Metrics (20-day test): total_return=0.282851, sortino=115.827681, annualized_return=94.134199
  - Last 2 days: total_return=0.024458, sortino=11294.884243, annualized_return=81.255827
- muon_momentum=0.98 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon098_20260204_050146`
  - Metrics (20-day test): total_return=0.281030, sortino=262.268200, annualized_return=91.695231
  - Last 2 days: total_return=0.031809, sortino=6890.016604, annualized_return=302.285574
- muon_momentum=0.96 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon096_20260204_110842`
  - Metrics (20-day test): total_return=0.193390, sortino=76.060899, annualized_return=24.363057
  - Last 2 days: total_return=0.032720, sortino=3114.981986, annualized_return=355.327393
- muon_momentum=0.975 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon0975_20260204_111637`
  - Metrics (20-day test): total_return=0.404804, sortino=72.492598, annualized_return=499.736632
  - Last 2 days: total_return=0.034365, sortino=31.250368, annualized_return=475.437671
- value_embedding_every=1, skip_scale_init=0.08: `chronos_sol_v2_test20_vale1_skip008_20260204_112657`
  - Metrics (20-day test): total_return=0.328249, sortino=87.316594, annualized_return=178.698710
  - Last 2 days: total_return=0.058472, sortino=8415.021779, annualized_return=31914.709940
- value_embedding_every=1, skip_scale_init=0.12: `chronos_sol_v2_test20_vale1_skip012_20260204_113544`
  - Metrics (20-day test): total_return=0.269032, sortino=51.479611, annualized_return=77.039409
  - Last 2 days: total_return=0.030477, sortino=9311.769567, annualized_return=238.593876

Best 20-day PnL so far: total_return=0.404804 (muon_momentum=0.975, skip_scale_init=0.1).

### Nano v2 longer test window (30-day test / 20-day val, 4-epoch)
- muon_momentum=0.975 (skip_scale_init=0.1): `chronos_sol_v2_test30_muon0975_20260204_114507`
  - Metrics (30-day test): total_return=0.703753, sortino=106.251893, annualized_return=658.726004
  - Last 2 days: total_return=0.070984, sortino=28667.020760, annualized_return=272505.067898
- baseline (skip_scale_init=0.1): `chronos_sol_v2_test30_skip01_20260204_115545`
  - Metrics (30-day test): total_return=0.534233, sortino=81.648603, annualized_return=183.001900
  - Last 2 days: total_return=0.026818, sortino=4597.584149, annualized_return=124.183249

Best 30-day PnL so far: total_return=0.703753 (muon_momentum=0.975, skip_scale_init=0.1).

### Nano v2 best-config sanity (6-epoch)
- Run: `chronos_sol_v2_win128_skip01_e6_20260203_140000`
- Metrics (10-day test): total_return=0.157931, sortino=118.091167, annualized_return=214.846481
- Last 2 days: total_return=0.062030, sortino=188.092299, annualized_return=58877.772244

## RL Reward Sweep (SOLUSDT hourly)
- Run: `python -m rlsys.run_reward_sweep --csv trainingdatahourlybinance/SOLUSDT.csv`
- Results saved: `rlsys/results/reward_sweep.json`
- Raw reward: episode_reward=-0.026216, episode_sharpe=-2.740867, episode_max_drawdown=-0.199934
- Risk-adjusted reward: episode_reward=-0.061468, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763
- Sharpe-like reward: episode_reward=-0.115716, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763

## Live Binance
- Attempted single-cycle live run failed due to restricted-region Binance API eligibility.
- Supervisor `binanceexp1-solusd` now running with BINANCE_TLD=us; requires Binance.US API keys (current keys rejected with "API-key format invalid").
- See `binanceresults.md` for full details of live attempt and command used.

## Syncs (R2)
- Synced binance checkpoints to R2:
  - `aws s3 sync binanceneural/checkpoints/ s3://models/stock/models/binance/binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"`
- Synced Chronos2 finetunes to R2:
  - `aws s3 sync chronos2_finetuned/ s3://models/stock/models/binance/chronos2_finetuned/ --endpoint-url "$R2_ENDPOINT"`

## Next Experiments
- Longer RL sweeps (e.g., 50k+ timesteps) with a grid over drawdown/volatility penalties.
- Multi-horizon (1/4/24) variants with larger sequence length (96/128) and/or context_hours 2048 vs 3072.
- Run longer backtests (30d+) on the best multi-horizon run and log trade stats + PnL targets.
- Multi-asset selector: tune min_edge/risk_weight, test cooldowns, and compare on matched windows vs SOLUSD h24 baseline.

## Binanceexp1 SOLUSD (marketsimulator, hourly)

Data & validation
- Backfilled hourly gap using Alpaca: 2025-04-16 through 2025-12-21.
- Max gap after backfill: ~3 hours.

Checkpoint
- `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt`

Intensity sweep (price_offset_pct=0, horizon=24, sequence_length=96, min_gap_pct=0.0003)
- 1.0 -> total_return 7.8325, sortino 55.8105
- 1.2 -> total_return 8.3542
- 1.6 -> total_return 9.0235
- 1.8 -> total_return 9.3230
- 2.0 -> total_return 9.5891
- 2.2 -> total_return 9.8570
- 2.5 -> total_return 10.1909
- 3.0 -> total_return 10.6377
- 4.0 -> total_return 11.3114
- 6.0 -> total_return 12.4451
- 8.0 -> total_return 13.5462
- 10.0 -> total_return 14.4308
- 15.0 -> total_return 16.2541, sortino 66.6059
- 20.0 -> total_return 17.7449, sortino 66.1148 (best PnL so far)

Risk mitigation tests
- Probe-mode after loss: reduced total return vs baseline.
- Max-hold forced close: reduced total return vs baseline.

Production target (SOLUSD)
- intensity-scale 20.0, horizon 24, sequence length 96, min-gap-pct 0.0003
- Runner: `scripts/run_binanceexp1_prod_solusd.sh`
- Supervisor: `supervisor/binanceexp1-solusd.conf`
## Binanceexp1 Horizon=1 (Chronos2 LoRA, h1-only cache)

### SOLUSD (full 5 epochs)
- Run: `solusd_h1_ft_20260203` (checkpoint: `binanceneural/checkpoints/solusd_h1_ft_20260203/epoch_005.pt`)
- Baseline eval (intensity=1.0): total_return 8.9329, sortino 47.5089
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 5.4643, sortino 55.0102
  - Best return: intensity 1.40, offset 0.00000 → total_return 10.8068, sortino 48.6080
- Risk mitigations:
  - probe-after-loss (notional 1.0) lowered return (7.5718) and sortino (42.7107)
  - max-hold-hours 24 had no measurable improvement
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 7.1795, sortino 60.5213
  - Best return: intensity 1.80, offset 0.00000 → total_return 14.4635, sortino 53.6626

### BTCUSD (quick 1 epoch / 300 steps)
- Run: `btcusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/btcusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 7.1547, sortino 253.3211
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.1541, sortino 253.5392
  - Best return: intensity 1.40, offset 0.00000 → total_return 8.5504, sortino 234.4700
- Expanded sweep (intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.3499, sortino 230.1740
  - Best return: intensity 1.80, offset 0.00010 → total_return 9.8338, sortino 202.4425
- Aggregate (contexts 64/96/192, trim_ratio 0.2):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.1246, sortino 243.8277
  - Best return: intensity 1.60, offset 0.00020 → total_return 9.1776, sortino 183.7078
- Blend horizons 1/24 (weights 0.7/0.3):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.5609, sortino 231.9530
  - Best return: intensity 1.40, offset 0.00020 → total_return 9.2805, sortino 182.3951

### ETHUSD (quick 1 epoch / 300 steps)
- Run: `ethusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/ethusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 3.6791, sortino 165.9452
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 3.6789, sortino 166.1271
  - Best return: intensity 1.40, offset 0.00000 → total_return 4.5134, sortino 145.5026
- Expanded sweep (intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 4.0286, sortino 165.7779
  - Best return: intensity 1.80, offset 0.00020 → total_return 5.3999, sortino 137.1733
- Aggregate (contexts 64/96/192, trim_ratio 0.2):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 3.7065, sortino 166.4207
  - Best return: intensity 1.60, offset 0.00000 → total_return 4.8140, sortino 143.6667

### LINKUSD (quick 1 epoch / 300 steps)
- Run: `linkusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/linkusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 3.4529, sortino 44.2042
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 2.2955, sortino 46.8543
  - Best return: intensity 1.20, offset 0.00000 → total_return 3.9597, sortino 39.0122
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 2.8358, sortino 38.9894
  - Best return: intensity 1.80, offset 0.00000 → total_return 4.7164, sortino 33.1052

### UNIUSD (quick 1 epoch / 300 steps)
- Run: `uniusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/uniusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 8.0990, sortino 38.5863
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00020 → total_return 5.4407, sortino 45.0316
  - Best return: intensity 1.20, offset 0.00020 → total_return 9.5057, sortino 36.3816
- Backtest (best PnL config): intensity 1.20, offset 0.00020 → total_return 9.5057, sortino 36.3816
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00050 → total_return 6.2135, sortino 42.5577
  - Best return: intensity 1.00, offset 0.00020 → total_return 10.8490, sortino 36.6693

### NFLX (quick 1 epoch / 300 steps)
- Run: `nflx_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/nflx_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 1.0496, sortino 55.6672
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 1.0497, sortino 55.6645
  - Best return: intensity 1.40, offset 0.00000 → total_return 1.1271, sortino 44.4451
- Backtest (best PnL config): intensity 1.40, offset 0.00000 → total_return 1.1271, sortino 44.4451

### NVDA (quick 1 epoch / 300 steps)
- Run: `nvda_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/nvda_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 11.2049, sortino 38.6581
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 6.8115, sortino 41.7632
  - Best return: intensity 1.00, offset 0.00050 → total_return 11.9568, sortino 34.3337
- Backtest (best PnL config): intensity 1.00, offset 0.00050 → total_return 11.9568, sortino 34.3337

## Multi-Asset Best-Trade Selector (BTC/ETH/LINK/UNI/SOL, h1)

Selector inputs
- Symbols: BTCUSD, ETHUSD, LINKUSD, UNIUSD, SOLUSD (h1-only forecast cache).
- Checkpoints: `binanceneural/checkpoints/{btcusd_h1_quick_20260203,ethusd_h1_quick_20260203,linkusd_h1_quick_20260203,uniusd_h1_quick_20260203,solusd_h1_ft_20260203}/epoch_*.pt`.
- Action overrides used (from expanded sweeps):
  - BTCUSD intensity 1.8 offset 0.0001
  - ETHUSD intensity 1.8 offset 0.0002
  - LINKUSD intensity 1.8 offset 0.0000
  - UNIUSD intensity 1.0 offset 0.0002
  - SOLUSD intensity 1.8 offset 0.0000
- CLI: `python -m binanceexp1.run_multiasset_selector ...` (see commands in logs).

Results (no same-bar re-entry by default)
- Baseline (edge_mode=high_low, min_edge=0.0): total_return 5.3570, sortino 29.3199.
- Best conservative (edge_mode=high_low, min_edge=0.002): total_return 13.9718, sortino 24.3030.
- Close-mode (edge_mode=close, min_edge=0.001): total_return 8.2331, sortino 25.0939.
- Aggressive upper bound (allow_reentry_same_bar, min_edge=0.002): total_return 42.1318, sortino 34.9956.

Comparison to best-20 sweep
- Best 20 sweep (SOLUSD horizon=24) total_return 17.7449 (not apples-to-apples; single-symbol, h24).
- Conservative selector is below that baseline; aggressive upper-bound run exceeds it.

## Cross-pair marketsimulator (BTCUSD/ETHUSD/LINKUSD, hourly)

Setup (2026-02-03)
- Checkpoints: `binanceneural/checkpoints/btcusd_h1_quick_20260203/epoch_001.pt`, `binanceneural/checkpoints/ethusd_h1_quick_20260203/epoch_001.pt`, `binanceneural/checkpoints/linkusd_h1_quick_20260203/epoch_001.pt`
- Features: binanceexp1 feature set (forecast_horizons=1, sequence_length=96)
- Validation window (subset): ~2025-05-03 through 2025-10-24

Metrics (per-symbol)
- BTCUSD: total_return 0.1068, sortino 88.6565
- ETHUSD: total_return 0.7707, sortino 26.9238
- LINKUSD: total_return 10.1283, sortino 63.2361

Combined (equal initial cash per symbol)
- total_return 13.1723, sortino 97.3380
