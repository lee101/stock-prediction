# PufferLib Experiment Log — 2025-10-20

## Run Summary
- **Pipeline**: `pufferlibtraining/train_ppo.py` (base & specialist stages skipped).
- **Command**: `python pufferlibtraining/train_ppo.py --skip-base --base-checkpoint pufferlibtraining/models/toto_run/base_models/base_checkpoint_20251017_162009.pth --skip-specialists --trainingdata-dir trainingdata --output-dir pufferlibtraining/models/20251019_puffer_rl400 --tensorboard-dir pufferlibtraining/logs/20251019_puffer_rl400 --rl-epochs 400 --rl-batch-size 256 --rl-learning-rate 7e-4 --rl-optimizer muon_mix --rl-warmup-steps 400 --rl-min-lr 1e-5 --rl-grad-clip 0.25 --transaction-cost-bps 1 --risk-penalty 0.0 --summary-path pufferlibtraining/models/20251019_puffer_rl400/summary.json --device cpu`
- **Base checkpoint**: `pufferlibtraining/models/toto_run/base_models/base_checkpoint_20251017_162009.pth` (scalers reused from the same directory).
- **Pairs trained**: `AAPL/AMZN`, `AMZN/MSFT`.
- **RL metrics** (validation best):
  - AAPL_AMZN — best val profit `+1.17e-3` at epoch 0.
  - AMZN_MSFT — best val profit `-2.32e-3` at epoch 0 (subsequent epochs degrade further).
- **Artifacts**: checkpoints + summary under `pufferlibtraining/models/20251019_puffer_rl400/`; TensorBoard logs at `pufferlibtraining/logs/20251019_puffer_rl400/`.

## Inference PnL (pufferlibinference/run_inference.py)
Simulator inputs used `transaction_cost_bps=1`, `leverage_limit=1.5`, `borrowing_cost=6.75%`, initial value `1.0`, processor scalers from the Toto base run.

| Pair | Final Value | Cumulative Return | Annualised Sharpe | Avg Turnover | Max Drawdown |
| ---- | ----------- | ----------------- | ----------------- | ------------ | ------------- |
| AAPL_AMZN | 1.6737 | +67.37% | 0.90 | 1.76% | -92.95% |
| AMZN_MSFT | 1.0563 | +5.63% | 0.30 | 2.60% | -60.84% |

- Detailed JSON metrics: `pufferlibtraining/models/20251019_puffer_rl400/inference/*_metrics.json`.
- Allocation traces: `.../inference/*_decisions.csv`.

## Findings
- Validation drift: both pairs show best validation profit at the very start of training, then trend negative despite low training loss — suggests overfitting or learning-rate schedule overshooting when Muon mix kicks in.
- Extreme drawdown on AAPL/AMZN despite strong terminal PnL — leverage constraints likely getting hit; turnover remains modest, so review borrowing penalty handling.
- HuggingFace hub import repeatedly warns `HTTPError` missing from `urllib3.exceptions` (new urllib3 drop). Current runs succeed but future hub calls may fail — need pin/patch.
- Packaging fixes: added explicit `tool.setuptools` blocks so `uv pip install -e` now works for `hfshared`, `hftraining`, `traininglib`, `hfinference`, and `pufferlibtraining`.

## Next Steps
1. **Stabilise RL stage**: experiment with lower `--rl-learning-rate` (≤3e-4) and non-mixed Muon/Adam to prevent early metric collapse; keep 400 epochs but enable EMA decay logging to verify plateau.
2. **Risk controls**: reintroduce small `--risk-penalty` (e.g., 0.05) or raise `transaction-cost-bps` during training to curb max drawdown; rerun inference to compare.
3. **Validation sampling**: snapshot checkpoints every 5 epochs and backtest with the inference engine to spot when validation swings negative; consider patience-based early stop.
4. **Hub compatibility**: reconcile HuggingFace/urllib3 versions (pin `urllib3<3` or update hub) so Toto downloads remain reliable.
5. **Additional pairs**: extend run to `NVDA/GOOGL` once RL stability improves; reuse the same base checkpoint and add to the summary.

## Run Summary — LR 3e-4, Risk Penalty 0.05 (2025-10-20 08:21 UTC)
- **Pipeline**: same as above, but with `--rl-learning-rate 3e-4` and `--risk-penalty 0.05`.
- **Command**: `python pufferlibtraining/train_ppo.py --skip-base --base-checkpoint pufferlibtraining/models/toto_run/base_models/base_checkpoint_20251017_162009.pth --skip-specialists --trainingdata-dir trainingdata --output-dir pufferlibtraining/models/20251020_puffer_rl400_lr3e4_risk005 --tensorboard-dir pufferlibtraining/logs/20251020_puffer_rl400_lr3e4_risk005 --rl-epochs 400 --rl-batch-size 256 --rl-learning-rate 3e-4 --rl-optimizer muon_mix --rl-warmup-steps 400 --rl-min-lr 1e-5 --rl-grad-clip 0.25 --transaction-cost-bps 1 --risk-penalty 0.05 --summary-path pufferlibtraining/models/20251020_puffer_rl400_lr3e4_risk005/summary.json --device cpu`
- **Validation best profits**:
  - AAPL_AMZN — `+1.15e-3` (epoch 0).
  - AMZN_MSFT — `+1.21e-3` (epoch 0).
- **Inference PnL** (`transaction_cost_bps=1`, leverage limit `1.5`, borrowing cost `6.75%`):

| Pair | Final Value | Cumulative Return | Annualised Sharpe | Avg Turnover | Max Drawdown |
| ---- | ----------- | ----------------- | ----------------- | ------------ | ------------- |
| AAPL_AMZN | 1.4643 | +46.43% | 1.18 | 2.47% | -52.03% |
| AMZN_MSFT | 0.8057 | -19.43% | -1.61 | 4.82% | -32.35% |

- Artifacts live under `pufferlibtraining/models/20251020_puffer_rl400_lr3e4_risk005/` with inference traces in the `inference/` subdir.
- HuggingFace hub import warning (`HTTPError` missing from `urllib3.exceptions`) persists during training and inference; it is benign for now but needs a compatibility fix before enabling Toto downloads on fresh machines.

### Updated Next Steps
1. **Pair-specific regularisation**: introduce turnover and CVaR penalties that scale per pair (e.g., start AMZN/MSFT at `transaction-cost-bps=5`, `cvar_weight=0.5`, while keeping AAPL/AMZN lighter) and checkpoint every 10 epochs.
2. **Early stopping hooks**: add validation patience logic so we capture checkpoints before profit turns negative; integrate automated inference sweeps into the training loop.
3. **Feature audit**: inspect AMZN/MSFT Toto features and raw returns for regime shifts; consider z-score normalisation or down-weighting extreme bars.
4. **Reporting**: build a small script to aggregate inference metrics across runs into `results/pufferlib_rl.csv` for easier trend analysis.
5. **HF long-run**: launch the refreshed hftraining pipeline (lower LR, longer schedule) and monitor the `stock/hftraining` project for stability.

## Run Summary — LR 3e-4, Risk 0.05, Transaction Cost 5 bps (2025-10-20 09:09 UTC)
- **Command**: `python pufferlibtraining/train_ppo.py ... --rl-learning-rate 3e-4 --rl-optimizer muon_mix --transaction-cost-bps 5 --risk-penalty 0.05`
- **Validation**: AAPL_AMZN best val profit `-4.52e-4` (epoch 0), AMZN_MSFT best val profit `+1.92e-3` (epoch 399).
- **Inference (cost=5 bps)**:

| Pair | Final Value | Cumulative Return | Annualised Sharpe | Avg Turnover | Max Drawdown |
| ---- | ----------- | ----------------- | ----------------- | ------------ | ------------- |
| AAPL_AMZN | 1.5230 | +52.30% | 1.11 | 3.40% | -54.72% |
| AMZN_MSFT | 0.3634 | -63.66% | -0.62 | 4.88% | -93.16% |

- Takeaway: higher transaction cost helped validation late in training for AMZN/MSFT but inference collapsed (portfolio de-levered into catastrophic losses). Turnover still elevated; need stronger regularisation or data rebalancing per pair.

## Run Summary — LR 2e-4, AdamW (2025-10-20 09:26 UTC)
- **Command**: `python pufferlibtraining/train_ppo.py ... --rl-learning-rate 2e-4 --rl-optimizer adamw --risk-penalty 0.05`
- **Validation**: AAPL_AMZN best val profit `-4.55e-4` (epoch 0); AMZN_MSFT best val profit `+1.62e-4` (epoch 0).
- **Inference (cost=1 bps)**:

| Pair | Final Value | Cumulative Return | Annualised Sharpe | Avg Turnover | Max Drawdown |
| ---- | ----------- | ----------------- | ----------------- | ------------ | ------------- |
| AAPL_AMZN | 0.8270 | -17.30% | -1.09 | 3.93% | -43.38% |
| AMZN_MSFT | 1.1111 | +11.11% | 0.58 | 2.96% | -37.69% |

- Takeaway: AdamW stabilised AMZN/MSFT modestly but hurt AAPL/AMZN. Need per-pair hyperparams or curriculum (e.g., freeze weights when validation drops).

## Dependency Fix
- Pinned HTTP stack to `urllib3>=1.26,<2`, added `idna`, `certifi`, and `charset-normalizer` to `pyproject.toml` so HuggingFace Hub loads cleanly (validated via direct `HfApi` import). Future syncs should retain the bundle; rerun `uv pip sync pyproject.toml --extra rl --extra hf --extra forecasting` after edits.
- RL runs now default to the `stock/pufferlib` namespace via `--wandb-project pufferlib --wandb-entity stock`; HF training uses `stock/hftraining`. Override with `--wandb-*` flags or environment variables when branching new experiments.

## PufferLib Daily/Annualised Returns

| run | pair | days | avg_daily_return | annualized_return | cumulative_return |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 20251019_puffer_rl400 | AAPL_AMZN | 317 | 0.003432 | 1.3713 | 0.6737 |
| 20251019_puffer_rl400 | AMZN_MSFT | 317 | 0.000347 | 0.0915 | 0.0563 |
| 20251020_puffer_rl400_lr3e4_risk005 | AAPL_AMZN | 317 | 0.001375 | 0.4137 | 0.4643 |
| 20251020_puffer_rl400_lr3e4_risk005 | AMZN_MSFT | 317 | -0.00066 | -0.1533 | -0.1943 |
| 20251020_puffer_rl400_lr3e4_risk005_tc5 | AAPL_AMZN | 317 | 0.001582 | 0.4895 | 0.5230 |
| 20251020_puffer_rl400_lr3e4_risk005_tc5 | AMZN_MSFT | 317 | -0.001938 | -0.3866 | -0.6366 |
| 20251020_puffer_rl400_lr2e4_adamw | AAPL_AMZN | 317 | -0.000566 | -0.1329 | -0.1730 |
| 20251020_puffer_rl400_lr2e4_adamw | AMZN_MSFT | 317 | 0.000388 | 0.1026 | 0.1111 |

(`aggregate_pufferlib_metrics.csv` contains the raw calculations.)
