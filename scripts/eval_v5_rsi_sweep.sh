#!/bin/bash
# Evaluate all v5_rsi checkpoints against the stocks12_daily_v5_rsi_val data
# Compares with production v4 ensemble performance (p10=66.2% @ fill_bps=5)
set -e
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

VAL_DATA="pufferlib_market/data/stocks12_daily_v5_rsi_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/stocks12_v5_rsi"
OUT_DIR="sweepresults"
mkdir -p "$OUT_DIR"

echo "timestamp,config,med_90d_pct,p10_90d_pct,worst_90d_pct,neg_of_50,med_sortino,checkpoint" > "$OUT_DIR/v5_rsi_leaderboard.csv"

eval_checkpoint() {
    local name=$1
    local ckpt=$2
    local out="$OUT_DIR/v5_rsi_${name}.json"

    if [ ! -f "$ckpt" ]; then
        echo "  $name: no checkpoint, skip"
        return
    fi

    TORCH_COMPILE_DISABLE=1 python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" \
        --data-path "$VAL_DATA" \
        --eval-hours 90 \
        --n-windows 50 \
        --seed 42 \
        --fee-rate 0.001 \
        --fill-buffer-bps 5.0 \
        --deterministic \
        --no-early-stop \
        --out "$out" > /dev/null 2>&1

    if [ ! -f "$out" ]; then
        echo "  $name: eval failed"
        return
    fi

    stats=$(python3 -c "
import json
d=json.load(open('$out'))
s=d['summary']
rets=[w['total_return'] for w in d['windows']]
neg=sum(1 for r in rets if r<0)
worst=min(rets)*100
print(f\"{s['median_total_return']*100:.2f},{s['p10_total_return']*100:.2f},{worst:.2f},{neg},{s['median_sortino']:.2f}\")
" 2>/dev/null)

    if [ -z "$stats" ]; then
        echo "  $name: parse failed"
        return
    fi

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "$ts,$name,$stats,$ckpt" >> "$OUT_DIR/v5_rsi_leaderboard.csv"

    neg=$(echo "$stats" | cut -d, -f4)
    med=$(echo "$stats" | cut -d, -f1)
    p10=$(echo "$stats" | cut -d, -f2)
    sortino=$(echo "$stats" | cut -d, -f5)
    echo "  $name: med=${med}%  p10=${p10}%  neg=${neg}/50  sortino=${sortino}$([ "$neg" = '0' ] && echo ' *** ZERO-NEG ***' || true)"
}

echo "=== Evaluating v5_rsi checkpoints ==="
for dir in "$CKPT_ROOT"/*/; do
    name=$(basename "$dir")
    ckpt="$dir/best.pt"
    eval_checkpoint "$name" "$ckpt"
done

echo ""
echo "=== Leaderboard ==="
sort -t, -k3 -rn "$OUT_DIR/v5_rsi_leaderboard.csv" | head -20
echo ""
echo "Production baseline: 32-model ensemble p10=66.2% @ fill_bps=5 (111-window exhaustive)"
