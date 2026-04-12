#!/bin/bash
# Auto-eval watcher: finds newly completed seeds, evaluates them, highlights promising ones
# Runs continuously. Promising = val_best.pt exists + training val had best_neg <= 4
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

VAL="pufferlib_market/data/stocks17_augmented_val.bin"
LOG="/tmp/auto_eval_watcher.log"
LEADERBOARD="/tmp/promising_seeds.txt"

echo "[$(date -u +%FT%TZ)] Auto-eval watcher started" | tee -a "$LOG"
echo "neg/50|p10%|med%|sortino|checkpoint" > "$LEADERBOARD"

already_evaled() {
    local ckpt_dir="$1"
    [ -f "$ckpt_dir/eval_lag2.json" ] && python3 -c "import json; d=json.load(open('$ckpt_dir/eval_lag2.json')); exit(0 if d.get('negative_windows') is not None else 1)" 2>/dev/null
}

eval_checkpoint() {
    local ckpt="$1"
    local out="${ckpt%.pt}_eval.json"
    [ -f "$out" ] && python3 -c "import json; d=json.load(open('$out')); exit(0 if d.get('negative_windows') is not None else 1)" 2>/dev/null && return 0

    python -m pufferlib_market.evaluate_holdout \
        --checkpoint "$ckpt" --data-path "$VAL" \
        --eval-hours 60 --n-windows 50 --fee-rate 0.001 \
        --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
        > "$out" 2>/dev/null && return 0
    return 1
}

print_result() {
    local out="$1"
    local label="$2"
    python3 - "$out" "$label" << 'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    neg = d.get("negative_windows", "?")
    med = d.get("median_total_return", 0)*100
    p10 = d.get("p10_total_return", 0)*100
    sort = d.get("median_sortino", 0)
    print(f"{sys.argv[2]}: neg={neg}/50 med={med:.2f}% p10={p10:.2f}% sort={sort:.2f}")
    sys.exit(0 if isinstance(neg, int) and neg <= 5 else 1)
except: pass
PY
}

while true; do
    # Scan all stocks17 sweep directories
    for variant_dir in pufferlib_market/checkpoints/stocks17_sweep/C_low_tp pufferlib_market/checkpoints/stocks17_sweep/D_muon; do
        variant=$(basename "$variant_dir" | sed 's/_.*//; s/C/C/; s/D/D/')

        for seed_dir in "$variant_dir"/s*/; do
            [ -d "$seed_dir" ] || continue
            seed=$(basename "$seed_dir")

            # Skip if training not complete
            grep -q "Training complete" "$seed_dir/train.log" 2>/dev/null || continue

            # Skip if already has good eval
            already_evaled "$seed_dir" && continue

            # Get best_neg from training val
            best_neg=$(grep "\[val\]" "$seed_dir/train.log" 2>/dev/null | grep -oP "best_neg=\K\d+" | tail -1)

            echo "[$(date -u +%FT%TZ)] Evaluating $seed_dir (best_neg=$best_neg in training)" | tee -a "$LOG"

            ckpt="$seed_dir/val_best.pt"
            [ -f "$ckpt" ] || ckpt="$seed_dir/best.pt"
            [ -f "$ckpt" ] || continue

            if eval_checkpoint "$ckpt"; then
                result=$(print_result "${ckpt%.pt}_eval.json" "$seed_dir")
                echo "  $result" | tee -a "$LOG"
                # Check if promising (neg <= 5)
                neg=$(python3 -c "import json; d=json.load(open('${ckpt%.pt}_eval.json')); print(d.get('negative_windows', 99))" 2>/dev/null)
                if [ -n "$neg" ] && [ "$neg" -le 5 ] 2>/dev/null; then
                    echo "  *** PROMISING: $seed_dir neg=$neg ***" | tee -a "$LOG"
                    # For D seeds, also eval periodic checkpoints
                    if echo "$seed_dir" | grep -q "D_muon"; then
                        for pckpt in "$seed_dir"/update_000*.pt; do
                            [ -f "$pckpt" ] || continue
                            pname=$(basename "$pckpt" .pt)
                            echo "  Evaluating periodic $pname..." | tee -a "$LOG"
                            eval_checkpoint "$pckpt"
                            print_result "${pckpt%.pt}_eval.json" "  $seed_dir/$pname" | tee -a "$LOG"
                        done
                    fi
                fi
                # Copy to seed's eval_lag2.json if not set
                [ -f "$seed_dir/eval_lag2.json" ] || cp "${ckpt%.pt}_eval.json" "$seed_dir/eval_lag2.json" 2>/dev/null
            fi
        done
    done

    sleep 300  # check every 5 minutes
done
