# Kronos vs Toto Benchmark Notes

## Experimental Setup
- Dataset: `trainingdata/BTCUSD.csv` (sorted by `timestamp`).
- Forecast horizon: 1 step (next-day closing price).
- Evaluation range: last 399 observations (299 for sweep, 100 for calibration/test).
- Metrics: Mean Absolute Error (MAE) on absolute prices and derived returns.
- Sampling: All Toto runs use 4,096 Monte‑Carlo samples with `samples_per_batch=512`.

## Kronos Results
Best configuration discovered via `python test_kronos_vs_toto.py` (15 Oct 2025 run):

| Parameter | Value |
|-----------|-------|
| Model ID | `NeoQuasar/Kronos-base` |
| Temperature | **0.152** |
| Top‑p | **0.83** |
| Top‑k | **20** |
| Sample count | **192** |
| Max context | **232** |
| Clip | **1.85** |
| Price MAE | **26.09** |
| Return MAE | **0.00294** |
| Latency (A100) | ~3 ms |

Notes:
- Lower temperatures (<0.2) with tighter top‑p (<0.85) dramatically reduced Kronos over-shooting.
- Truncating context to ≈200 points avoided the drift observed with 512-token contexts.
- Increasing clip beyond 2.0 reintroduced large spikes; 1.8 held the decoder in check.

## Toto Results
The best sweep configuration (15 Oct 2025) relies on a trimmed-mean ensemble:

| Variant | Price MAE | Return MAE | Notes |
|---------|-----------|------------|-------|
| `toto_trimmed10_3072` (`trimmed_mean_0.10`, 3,072 samples) | **162.06** | **0.01825** | `python test_kronos_vs_toto.py` |
| Calibrated Toto (scale 0.972436, bias 693.032) | **422.59** | **0.03675** | Held-out evaluation (`python test_toto_vs_toto_retrain.py`) |

Hyper-parameter training via `python test_hyperparamtraining_kronos_toto.py --symbols BTCUSD` currently selects `toto_trimmed10_3072` (3,072 samples, trimmed mean 10 %) on the BTCUSD validation window and records the choice in the hyperparamstore (see below).

The affine calibration is trained via `python tototraining/train_calibrated_toto.py`. Training MAE improved modestly (659 → 628), validation MAE 572 → 523, but the held-out slice still trails base Toto because of distribution shift near the dataset tail. Calibration artefacts live in `tototraining/artifacts/calibrated_toto.json`.

## Hyper-Parameter Store
- `test_hyperparamtraining_kronos_toto.py` runs the focused search (validation window 20, test window 20) and persists per-symbol winners in `hyperparams/<model>/<symbol>.json` via the new `hyperparamstore` package.
- `hyperparamstore` exposes `save_best_config` / `load_best_config` (see `tests/test_hyperparamstore.py`).
- Current BTCUSD entries:
  - Kronos → `kronos_temp0.145_p0.82_s208_k16_clip1.75_ctx224` (val MAE 359.5, test MAE 268.7).
  - Toto → `toto_trimmed10_3072` (val MAE 332.2, test MAE 259.1).

## Updated Inference Defaults
- `src/models/toto_wrapper.py` keeps the Monte-Carlo defaults (`num_samples=4096`, `samples_per_batch=512`, FP32) but execution-time parameters now come from `hyperparamstore` when available.
- `backtest_test3_inline.py` resolves Toto parameters via `resolve_toto_params(symbol)` and falls back to the environment defaults when no record exists.

## How to Reproduce
1. **Benchmark sweep**
   ```bash
   python test_kronos_vs_toto.py
   ```
   Produces ranked tables for the expanded Kronos/Toto grids (focused around the low-MAE region).

2. **Hyper-parameter selection**
   ```bash
   python test_hyperparamtraining_kronos_toto.py --symbols BTCUSD
   ```
   Stores the per-symbol winners for both models under `hyperparams/`.

3. **Calibrate Toto**
   ```bash
   python tototraining/train_calibrated_toto.py
   ```
   Saves `tototraining/artifacts/calibrated_toto.json` with scale/bias.

4. **Compare Toto vs Calibrated Toto**
   ```bash
   python test_toto_vs_toto_retrain.py
   ```
   Reports MAE/return-MAE for the base model vs the calibrated variant on the latest window.

## Next Steps
- Explore dynamic calibration (e.g., rolling fit) to maintain the ~140 MAE baseline while adapting to drift.
- Consider integrating Kronos’ low-latency settings into production forecasts with automatic fallback to Toto when GPU is saturated.
