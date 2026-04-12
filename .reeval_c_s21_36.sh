#!/bin/bash
export TMPDIR=/nvme0n1-disk/code/stock-prediction/.tmp_train
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate
VAL="pufferlib_market/data/stocks17_augmented_val.bin"
CKPT_ROOT="pufferlib_market/checkpoints/stocks17_sweep/C_low_tp"
LOG="/nvme0n1-disk/code/stock-prediction/.reeval_c_results.txt"
echo "=== C s21-36 Re-eval $(date -u +%FT%TZ) ===" > "$LOG"
for seed in 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36; do
    f="$CKPT_ROOT/s${seed}/eval_lag2.json"
    has_neg=$(python3 -c "import json; d=json.load(open('$f')); print('yes' if 'negative_windows' in d else 'no')" 2>/dev/null || echo "no")
    ckpt="$CKPT_ROOT/s${seed}/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$CKPT_ROOT/s${seed}/best.pt"
    if [ "$has_neg" = "no" ] && [ -f "$ckpt" ]; then
        echo -n "  s$seed: " | tee -a "$LOG"
        result=$(python -m pufferlib_market.evaluate_holdout --checkpoint "$ckpt" --data-path "$VAL" --eval-hours 60 --n-windows 50 --fee-rate 0.001 --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop 2>/dev/null)
        echo "$result" > "$f"
        echo "$result" | python3 -c "
import json,sys
d=json.load(sys.stdin)
med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100
neg=d.get('negative_windows','?'); sort=d.get('median_sortino',0)
flag='' 
if neg=='?': flag=' [NO NEG]'
elif neg==0: flag=' ***'
elif neg<=3: flag=' **'
elif neg<=10: flag=' *'
print(f'med={med:.1f}% p10={p10:.1f}% neg={neg}/50 sort={sort:.1f}{flag}')
" | tee -a "$LOG"
    else
        python3 -c "
import json; d=json.load(open('$f'))
med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100
neg=d.get('negative_windows','?'); sort=d.get('median_sortino',0)
flag='' 
if neg==0: flag=' ***'
elif neg<=3: flag=' **'
elif neg<=10: flag=' *'
print(f'  s$seed (ok): med={med:.1f}% p10={p10:.1f}% neg={neg}/50 sort={sort:.1f}{flag}')
" | tee -a "$LOG"
    fi
done
echo "Done $(date -u +%FT%TZ)" >> "$LOG"
