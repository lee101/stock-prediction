#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source .venv313/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/PufferLib:${PYTHONPATH:-}"

TRAIN_DATA_DIR="analysis/hf_vs_pufferlib/data/train_5sym_until_2024-12-31"
TRAIN_END_DATE="2024-12-31"
EVAL_START_DATE="2025-01-02"
EVAL_END_DATE="2025-11-28"
SYMBOLS=(AAPL MSFT NVDA AMZN TSLA)
HF_ACTION_MODES=(alloc_only alloc_signed_by_logits)
EVAL_MODES=(open_close maxdiff)
CONFIGS=(
  "analysis/hf_vs_pufferlib/configs/hf_5sym_nototo_adamw_2024train.json"
  "analysis/hf_vs_pufferlib/configs/hf_5sym_toto_muon_2024train.json"
)

mkdir -p analysis/hf_vs_pufferlib/commands
mkdir -p analysis/hf_vs_pufferlib/evals
mkdir -p analysis/hf_vs_pufferlib/logs

python prepare_hf_puffer_benchmark_data.py \
  --source-root trainingdata \
  --output-dir "$TRAIN_DATA_DIR" \
  --symbols "$(IFS=,; echo "${SYMBOLS[*]}")" \
  --end-date "$TRAIN_END_DATE" \
  > analysis/hf_vs_pufferlib/logs/prepare_train_split.log 2>&1

for config_path in "${CONFIGS[@]}"; do
  run_name="$(python - <<'PY' "$config_path"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    payload = json.load(handle)
print(payload["experiment_name"])
PY
)"
  run_dir="analysis/hf_vs_pufferlib/runs/${run_name}"
  mkdir -p "$run_dir"

  train_cmd=(
    python -m hftraining.run_training
    --config_file "$config_path"
  )
  printf '%q ' "${train_cmd[@]}" > "$run_dir/train_command.txt"
  printf '\n' >> "$run_dir/train_command.txt"
  "${train_cmd[@]}" > "$run_dir/train.log" 2>&1

  checkpoint_path="$run_dir/final_model.pth"
  processor_path="$run_dir/data_processor.pkl"
  config_json="$run_dir/config.json"

  for symbol in "${SYMBOLS[@]}"; do
    for mode in "${EVAL_MODES[@]}"; do
      for action_mode in "${HF_ACTION_MODES[@]}"; do
        eval_json="analysis/hf_vs_pufferlib/evals/${run_name}_${symbol}_${mode}_${action_mode}.json"
        eval_log="analysis/hf_vs_pufferlib/logs/${run_name}_${symbol}_${mode}_${action_mode}.log"
        eval_cmd=(
          python compare_hf_pufferlib_marketsim.py evaluate-hf
          --checkpoint "$checkpoint_path"
          --processor-path "$processor_path"
          --config-json "$config_json"
          --symbol "$symbol"
          --data-root trainingdata
          --mode "$mode"
          --start-date "$EVAL_START_DATE"
          --end-date "$EVAL_END_DATE"
          --hf-action-mode "$action_mode"
          --device cpu
          --trading-fee 0.0005
          --slip-bps 5
          --intraday-leverage 4
          --overnight-leverage 2
          --annual-leverage-rate 0.065
          --output-json "$eval_json"
        )
        printf '%q ' "${eval_cmd[@]}" > "$run_dir/eval_${symbol}_${mode}_${action_mode}_command.txt"
        printf '\n' >> "$run_dir/eval_${symbol}_${mode}_${action_mode}_command.txt"
        "${eval_cmd[@]}" > "$eval_log" 2>&1
      done
    done
  done
done
