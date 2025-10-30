# CX Prompt Training Requests (2025-10-29)

This runbook queues long-horizon training + 1-day validation runs across the main RL stacks. Each block is a ready-to-send `cx prompt` that assumes execution from the repository root (`/home/administrator/code/stock-prediction`) on 2025-10-29 with the existing `.venv*` environments. All installs use `uv pip`, runners avoid `uv run`, and validation windows target the final dataset day (2023-07-14).

## PufferLib v3 Portfolio PPO (Python 3.14)
- Environment: `.venv314`
- Training output: `pufferlibtraining3/runs/20251029_puffer_v3/`
- Validation: 2023-07-14 (single day) via `pufferlibinference.run_inference`

```bash
cx prompt "
You are an automation agent working inside /home/administrator/code/stock-prediction on 2025-10-29.
Goal: train the pufferlibtraining3 PPO policy and validate it on 2023-07-14 market data.
Constraints:
  - Use uv pip (never plain pip) and avoid uv run; activate .venv314 manually.
  - Keep logs and artefacts under pufferlibtraining3/runs/20251029_puffer_v3/.
  - Treat CUDA as optional; fall back to CPU if unavailable.
Steps:
  1. source .venv314/bin/activate
  2. uv pip install -e '.[rl]' -e ./toto
  3. export PYTORCH_ENABLE_MPS_FALLBACK=1; export TOKENIZERS_PARALLELISM=false
  4. mkdir -p pufferlibtraining3/runs/20251029_puffer_v3
  5. python -m pufferlibtraining3.pufferrl \
       --data-root trainingdata \
       --symbol AAPL \
       --mode open_close \
       --total-timesteps 4000000 \
       --num-envs 32 \
       --batch-size 262144 \
       --minibatch-size 65536 \
       --update-epochs 4 \
       --learning-rate 2.5e-4 \
       --seed 1337 \
       --device cuda \
       --log-json pufferlibtraining3/runs/20251029_puffer_v3/summary.json \
       --log-level INFO
  6. jq -r '.model_path' pufferlibtraining3/runs/20251029_puffer_v3/summary.json > /tmp/puffer_ckpt_path
  7. CKPT=$(cat /tmp/puffer_ckpt_path)
     python -m pufferlibinference.run_inference \
       --checkpoint \"$CKPT\" \
       --symbols AAPL \
       --data-dir trainingdata \
       --start-date 2023-07-14 \
       --end-date 2023-07-14 \
       --initial-value 100000 \
       --transaction-cost-bps 10 \
       --output-json pufferlibtraining3/runs/20251029_puffer_v3/validation_2023-07-14.json \
       --decisions-csv pufferlibtraining3/runs/20251029_puffer_v3/validation_2023-07-14_decisions.csv
  8. Summarise key metrics (final portfolio value, turnover, trading_cost, financing_cost, sharpe if available) into pufferlibtraining3/runs/20251029_puffer_v3/README.md.
Return:
  - Path to summary.json, validation JSON, and decisions CSV.
  - Final PnL / turnover / drawdown numbers for 2023-07-14.
"
```

## GymRL PPO Allocator (Python 3.12)
- Environment: `.venv312`
- Training output: `gymrl/artifacts/20251029_cx/`
- Validation: 1 trading day (2023-07-14) via `gymrl.evaluate_policy`

```bash
cx prompt "
You are an automation agent in /home/administrator/code/stock-prediction on 2025-10-29.
Goal: train the gymrl PPO allocator on legacy equity data and evaluate on the last available day (2023-07-14).
Constraints:
  - Activate .venv312; use uv pip for installs.
  - Cache features for reuse and keep artefacts in gymrl/artifacts/20251029_cx/.
Steps:
  1. source .venv312/bin/activate
  2. uv pip install -e '.[rl]' -e ./toto
  3. mkdir -p gymrl/artifacts/20251029_cx
  4. python -m gymrl.train_ppo_allocator \
       --data-dir trainingdata \
       --output-dir gymrl/artifacts/20251029_cx \
       --cache-features-to gymrl/artifacts/20251029_cx/features_latest.npz \
       --num-timesteps 2000000 \
       --train-fraction 0.8 \
       --validation-days 21 \
       --batch-size 512 \
       --n-steps 2048 \
       --ent-coef 0.001 \
       --turnover-penalty 5e-4 \
       --costs-bps 3 \
       --seed 42
  5. python -m gymrl.evaluate_policy \
       --checkpoint gymrl/artifacts/20251029_cx/ppo_allocator_final.zip \
       --features-cache gymrl/artifacts/20251029_cx/features_latest.npz \
       --validation-days 1 \
       --turnover-penalty 5e-4 \
       --weight-cap 0.35 \
       --base-gross-exposure 1.0 \
       --max-gross-leverage 1.5 \
       --intraday-leverage-cap 1.5 \
       --closing-leverage-cap 1.5 \
       --daily-leverage-rate 0.0002 \
       --log-level INFO
     | tee gymrl/artifacts/20251029_cx/validation_2023-07-14.log
  6. Append a short markdown recap (PnL, sharpe, turnover, hit_rate if logged) to gymrl/artifacts/20251029_cx/README.md.
Return:
  - Paths to checkpoint, feature cache, validation log.
  - Core validation metrics for 2023-07-14.
"
```

## Differentiable Market GRPO (Python 3.14)
- Environment: `.venv314`
- Training output: `differentiable_market/runs/20251029_dm/`
- Validation: 1-day rolling window via `differentiable_market.marketsimulator.run`

```bash
cx prompt "
You are an automation agent in /home/administrator/code/stock-prediction on 2025-10-29.
Goal: fit the differentiable-market GRPO policy and backtest the best checkpoint on a 1-day window ending 2023-07-14.
Constraints:
  - Use .venv314 with uv pip.
  - Store artefacts under differentiable_market/runs/20251029_dm/ and evaluation under differentiable_market/evals/20251029_dm/.
Steps:
  1. source .venv314/bin/activate
  2. uv pip install -e . -e ./toto
  3. python -m differentiable_market.train \
       --data-root trainingdata \
       --epochs 1500 \
       --eval-interval 100 \
       --save-dir differentiable_market/runs/20251029_dm \
       --device auto \
       --dtype auto \
       --seed 20251029 \
       --include-cash \
       --max-intraday-leverage 3.0 \
       --max-overnight-leverage 2.0 \
       --risk-aversion 0.05 \
       --drawdown-lambda 0.02 \
       --tensorboard-root tensorboard_logs \
       --tensorboard-subdir 20251029_dm
  4. python -m differentiable_market.marketsimulator.run \
       --checkpoint differentiable_market/runs/20251029_dm/checkpoints/best.pt \
       --data-root trainingdata \
       --window-length 1 \
       --stride 1 \
       --report-dir differentiable_market/evals/20251029_dm \
       --include-cash \
       --risk-aversion 0.05 \
       --drawdown-lambda 0.02
     | tee differentiable_market/evals/20251029_dm/report.log
  5. Record aggregated metrics (cumulative_return, sharpe, turnover, max_drawdown) inside differentiable_market/evals/20251029_dm/README.md.
Return:
  - Paths to best.pt, report.json, windows.json, and report.log.
  - 1-day evaluation metrics covering 2023-07-14.
"
```

## C++ Market Simulator Smoke (LibTorch)
- Build directory: `cppsimulator/build`
- Artefacts: `cppsimulator/runs/20251029_run_sim.txt`

```bash
cx prompt "
You are an automation agent in /home/administrator/code/stock-prediction on 2025-10-29.
Goal: rebuild the C++ market simulator against the current LibTorch from .venv314 and run the synthetic demo to confirm throughput.
Steps:
  1. source .venv314/bin/activate
  2. TORCH_DIR=$(python - <<'PY'
import pathlib, torch
path = pathlib.Path(torch.__file__).resolve().parent / 'share' / 'cmake' / 'Torch'
print(path)
PY
)
  3. cmake -S cppsimulator -B cppsimulator/build -DTorch_DIR=\"$TORCH_DIR\" -DCMAKE_BUILD_TYPE=Release
  4. cmake --build cppsimulator/build -j
  5. ./cppsimulator/build/run_sim | tee cppsimulator/runs/20251029_run_sim.txt
Return:
  - Confirmation that run_sim executed (capture stdout snippet).
  - Location of the build artefacts and timing if reported.
"
```
