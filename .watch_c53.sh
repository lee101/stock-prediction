#!/bin/bash
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
DIR="pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s53"
VAL="pufferlib_market/data/stocks17_augmented_val.bin"
echo "[$(date -u +%FT%TZ)] Watching C s53..."
while ! [ -f "$DIR/final.pt" ]; do sleep 30; done
echo "[$(date -u +%FT%TZ)] C s53 done. Evaluating..."
for ckpt_name in val_best update_000100 update_000150 update_000200 update_000250 update_000300 update_000350 update_000400 update_000450 best; do
    ckpt="$DIR/${ckpt_name}.pt"; [ -f "$ckpt" ] || continue
    out="$DIR/${ckpt_name}_eval.json"
    echo -n "  $ckpt_name: "
    python -m pufferlib_market.evaluate_holdout --checkpoint "$ckpt" --data-path "$VAL" --eval-hours 60 --n-windows 50 --fee-rate 0.001 --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop > "$out" 2>/dev/null
    python3 -c "import json; d=json.load(open('$out')); print(f'med={d.get(\"median_total_return\",0)*100:.2f}% p10={d.get(\"p10_total_return\",0)*100:.2f}% neg={d.get(\"negative_windows\",\"?\")} sort={d.get(\"median_sortino\",0):.2f}')"
done
echo "[$(date -u +%FT%TZ)] C s53 eval done."
