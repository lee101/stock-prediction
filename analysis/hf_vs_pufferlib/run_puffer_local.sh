#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source .venv313/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/PufferLib:${PYTHONPATH:-}"

TRAIN_START_DATE="2021-12-03"
TRAIN_END_DATE="2024-12-31"
EVAL_START_DATE="2025-01-02"
EVAL_END_DATE="2025-11-28"
SYMBOLS=(AAPL MSFT NVDA AMZN TSLA)
MODES=(open_close maxdiff)

mkdir -p analysis/hf_vs_pufferlib/evals
mkdir -p analysis/hf_vs_pufferlib/logs

for mode in "${MODES[@]}"; do
  for symbol in "${SYMBOLS[@]}"; do
    run_name="puffer_${mode}_${symbol}_2024train"
    run_dir="analysis/hf_vs_pufferlib/runs/${run_name}"
    mkdir -p "$run_dir"

    train_cmd=(
      python -m pufferlibtraining3.pufferrl
      --symbol "$symbol"
      --data-root trainingdata
      --start-date "$TRAIN_START_DATE"
      --end-date "$TRAIN_END_DATE"
      --mode "$mode"
      --device cpu
      --backend Serial
      --env-backend fast
      --num-envs 16
      --total-timesteps 2000000
      --batch-size 65536
      --minibatch-size 16384
      --update-epochs 3
      --bptt-horizon 128
      --learning-rate 3e-4
      --gamma 0.995
      --gae-lambda 0.95
      --ent-coef 0.005
      --vf-coef 0.8
      --optimizer adam
      --precision float32
      --model-preset small
      --slip-bps 5
      --trading-fee 0.0005
      --intraday-leverage 4
      --overnight-leverage 2
      --annual-leverage-rate 0.065
      --log-json "$run_dir/train_summary.json"
    )
    printf '%q ' "${train_cmd[@]}" > "$run_dir/train_command.txt"
    printf '\n' >> "$run_dir/train_command.txt"
    "${train_cmd[@]}" > "$run_dir/train.log" 2>&1

    eval_json="analysis/hf_vs_pufferlib/evals/${run_name}_eval_2025.json"
    eval_cmd=(
      python compare_hf_pufferlib_marketsim.py evaluate-puffer
      --summary-json "$run_dir/train_summary.json"
      --symbol "$symbol"
      --data-root trainingdata
      --mode "$mode"
      --start-date "$EVAL_START_DATE"
      --end-date "$EVAL_END_DATE"
      --device cpu
      --output-json "$eval_json"
    )
    printf '%q ' "${eval_cmd[@]}" > "$run_dir/eval_command.txt"
    printf '\n' >> "$run_dir/eval_command.txt"
    "${eval_cmd[@]}" > "$run_dir/eval.log" 2>&1
  done
done
