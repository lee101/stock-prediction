#!/bin/bash
# Watch C s71 and D s29 - both showing 0/20 neg in training val
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
VAL="pufferlib_market/data/stocks17_augmented_val.bin"

echo "[$(date -u +%FT%TZ)] Watching C s71 and D s29..."

eval_ckpt() {
    local ckpt="$1" label="$2"
    local out="${ckpt%.pt}_eval.json"
    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$VAL" \
        --eval-hours 60 --n-windows 50 --fee-rate 0.001 \
        --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
        > "$out" 2>/dev/null
    python3 - "$out" "$label" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med = d.get("median_total_return", 0)*100
p10 = d.get("p10_total_return", 0)*100
neg = d.get("negative_windows", "?")
sort = d.get("median_sortino", 0)
print(f"  {sys.argv[2]}: med={med:.2f}% p10={p10:.2f}% neg={neg}/50 sort={sort:.2f}")
PY
}

# Wait for C s71 to finish (final.pt)
echo "Waiting for C s71 final.pt..."
while [ ! -f "pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s71/final.pt" ]; do
    sleep 120
    best_neg=$(grep "\[val\]" "pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s71/train.log" 2>/dev/null | grep -oP "best_neg=\K\d+" | tail -1)
    step=$(tail -1 "pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s71/train.log" 2>/dev/null | grep -oP '\[\s*\d+/457\]' | head -1)
    echo "  C s71 $step best_neg=$best_neg"
done

echo "[$(date -u +%FT%TZ)] C s71 done! Evaluating all checkpoints..."
for ckpt_name in val_best update_000100 update_000150 update_000200 update_000250 update_000300 update_000350 update_000400 best; do
    ckpt="pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s71/${ckpt_name}.pt"
    [ -f "$ckpt" ] && eval_ckpt "$ckpt" "C s71 $ckpt_name"
done

# Wait for D s29 to finish
echo "Waiting for D s29 final.pt..."
while [ ! -f "pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/final.pt" ]; do
    sleep 120
    best_neg=$(grep "\[val\]" "pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/train.log" 2>/dev/null | grep -oP "best_neg=\K\d+" | tail -1)
    step=$(tail -1 "pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/train.log" 2>/dev/null | grep -oP '\[\s*\d+/457\]' | head -1)
    echo "  D s29 $step best_neg=$best_neg"
done

echo "[$(date -u +%FT%TZ)] D s29 done! Evaluating all checkpoints..."
for ckpt_name in val_best update_000050 update_000100 update_000150 update_000200 update_000250 update_000300 update_000350 update_000400 best; do
    ckpt="pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/${ckpt_name}.pt"
    [ -f "$ckpt" ] && eval_ckpt "$ckpt" "D s29 $ckpt_name"
done

echo "[$(date -u +%FT%TZ)] Done watching C s71 and D s29"
