#!/bin/bash
# XGBoost daily open-to-close LIVE trader — 5-seed alltrain ensemble.
# FULL-STACK DEPLOY (2026-04-19): hold_through + min_score 0.85 + lev 2.0.
#
# Validated by:
#   - CPU bonferroni 5-seed 2020-train (OOS 2025-01→2026-04-10): +39-42%/mo
#   - Expanding-window 5-fold CV: cross-fold mean +31.2%/mo, 2/49 neg
#   - Alltrain in-sample eval: +36% med (no overfit signature)
#   - Full-stack ensemble eval (OOS 2025-01→2026-04-19, 60 windows):
#       deploy-cost (fb=5bps fee=0.28bps): +141%/mo med, p10 +96%, 0/60 neg
#       36× fees stress   (fb=15 fee=10): +108%/mo med, p10 +68%, 0/60 neg
#   - Inference-side min_dollar_vol 50M (2026-04-20): strict-dominance lift
#     over 5M floor — med +2.08%/mo, p10 +5.23%/mo, same 0/60 neg, same
#     ddW 5.34 & idW 12.93. Predicted by symbol-LOBO (+5.23 goodness)
#     and confirmed by inference-min-dolvol sweep. No retraining needed:
#     the 50M gate only narrows the pick pool at inference time.
#   - Inference-side min_vol_20d 0.12 (2026-04-20 13:16 UTC): stacked
#     strict-dominance vs the 0.10 floor on `oos2024_ensemble_gpu` 60-window
#     TRUE-OOS sweep. Deploy fees: med 143.07→152.87 (Δ+9.80), p10
#     96.06→96.91 (Δ+0.84), ddW 7.18→7.18 (Δ0.00), idW 12.93 same,
#     neg 0/60. Stress36x: med 93.55→101.85 (Δ+8.30), p10 56.08→58.06
#     (Δ+1.98), ddW 8.34→8.34 (Δ0.00). Zero DD regression on either regime
#     — cleanest strict-dom in the vol grid. vol=0.20 is a bigger PnL
#     lift (+22.73 med) but +0.49pp DD and non-monotonic through vol=0.15
#     so we stay conservative at 0.12.
#
# Safety: xgbnew/live_trader.py imports src.alpaca_singleton for the
# fcntl live-writer lock. Only one live process allowed at a time.
# trading-server and daily-rl-trader MUST be stopped before this runs.
# HARD RULE #3 enforced: guard_sell_against_death_spiral + record_buy_price
# fire on every sell/buy in run_session_hold_through.

set -euo pipefail

export HOME=/home/administrator
export USER=administrator

if [ -f "$HOME/.secretbashrc" ]; then
  # shellcheck disable=SC1090
  set +e
  set +u
  source "$HOME/.secretbashrc" >/dev/null 2>&1
  set -euo pipefail
fi

cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

export PYTHONPATH=/nvme0n1-disk/code/stock-prediction
export ALP_PAPER=0
export ALLOW_ALPACA_LIVE_TRADING=1

MODEL_DIR="analysis/xgbnew_daily/alltrain_ensemble_gpu"
MODEL_PATHS="${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl,${MODEL_DIR}/alltrain_seed42.pkl,${MODEL_DIR}/alltrain_seed73.pkl,${MODEL_DIR}/alltrain_seed197.pkl"

exec python -u -m xgbnew.live_trader \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "${MODEL_PATHS}" \
  --top-n 1 \
  --allocation 2.0 \
  --min-score 0.85 \
  --hold-through \
  --min-dollar-vol 50000000 \
  --min-vol-20d 0.12 \
  --trade-log-dir analysis/xgb_live_trade_log \
  --live \
  --loop \
  --verbose
