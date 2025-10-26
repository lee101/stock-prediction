#!/usr/bin/env bash
set -euo pipefail

# Nanochat-inspired end-to-end speedrun for the stock project.
#  1) bootstrap an isolated environment with uv if available
#  2) run the custom PyTorch loop (training/nano_speedrun.py)
#  3) kick off a lightweight HF training job (hftraining/train_hf.py)
#  4) summarise results in runs/*/report.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "${VENV_DIR}" ]; then
  uv venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

uv pip install --upgrade pip wheel setuptools >/dev/null
uv pip install -r "${ROOT_DIR}/requirements.txt" >/dev/null 2>&1 || true

echo "➤ Running nano speedrun training loop..."
python -m training.nano_speedrun \
  --data-dir "${ROOT_DIR}/trainingdata" \
  --output-dir "${ROOT_DIR}/runs/speedrun" \
  --report "${ROOT_DIR}/runs/speedrun/report.md" \
  --compile \
  --optimizer muon_mix \
  --epochs 3 \
  --device-batch-size 64 \
  --grad-accum 2

echo "➤ Launching HF training with unified optimiser stack..."
python -m hftraining.train_hf > "${ROOT_DIR}/runs/hf_train.log"

echo "➤ Speedrun completed. Reports:"
ls "${ROOT_DIR}"/runs/*/report*.md 2>/dev/null || echo "  (no reports found)"

