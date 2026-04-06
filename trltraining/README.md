# TRL Training

This directory is the repo's GRPO-first TRL layer for trading-plan training.

## Recommended Method

Use `GRPOTrainer` first.

- We already have an online scalar reward from the Binance market simulator.
- We do not yet have strong pairwise preference data for DPO-style training.
- We want direct optimization for return, Sortino, drawdown, and smoothness-aware penalties.

Default recommendation:

- trainer: `GRPOTrainer`
- vLLM: `use_vllm=True`
- vLLM mode: `colocate`
- starter model: `Qwen/Qwen2.5-0.5B-Instruct`

## Install

```bash
source ~/.secretbashrc
source .venv313/bin/activate
uv pip install "trl[vllm]"
```

## First Run

```bash
python -m trltraining.train_grpo \
  --model-preset qwen2_05b_instruct \
  --output-dir trltraining/checkpoints/qwen2_05b_grpo \
  --prompt-variant with_chronos2 \
  --reward-type sortino_drawdown \
  --n-symbols 10 \
  --group-size 8 \
  --use-vllm \
  --vllm-mode colocate
```

## Notes

- This layer reuses `qwen_rl_trading` dataset and simulator reward code instead of forking another reward stack.
- `sortino_drawdown` is the default reward because it penalizes strong returns with unstable equity curves.
- `with_chronos2` is the default prompt variant because the repo already has forecast cache infrastructure.
- For the first serious remote run, keep tensor parallel at `1` until memory pressure is measured on the target GPU.
