Baseline Performance

Purpose
- Establish a reproducible, minimal baseline that verifies training loss decreases and capture key settings to compare future changes against.

Scope
- Model: `hftraining.hf_trainer.TransformerTradingModel`
- Data: synthetic OHLC sequences
- Target: price prediction head (MSE to simple linear target)

Quick Baseline (CI-safe)
- Test: `tests/test_training_baseline.py`
- Settings:
  - `hidden_size=32`, `num_layers=1`, `num_heads=4`
  - `sequence_length=10`, `prediction_horizon=2`, `input_dim=4`
  - Optimizer: `Adam(lr=1e-2)`
  - Steps: 60 on CPU
- Expected: price-prediction loss decreases by >= 50% on synthetic data.

Run Locally
- `pytest -q tests/test_training_baseline.py`

Extended Baseline (manual)
- To sanity-check end-to-end quickly on CPU, you can run a tiny loop similar to the test and log metrics per step. Keep steps ≤ 200 to finish quickly.

Notes
- Keep training/inference feature processing aligned. If enabling `feature_mode="ohlc"` or `use_pct_change=true` in inference, ensure training used the same transforms.
- This baseline is intentionally synthetic to be stable and fast. Real-data baselines (drawdowns, Sharpe, hit rate) should be tracked separately once a dataset is fixed.

