#!/bin/bash
# Quick check of all wide73 seed results so far.
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

check_evals() {
    local root="$1"
    local variant="$2"
    local found=0
    for s in $(seq 1 20); do
        f="$root/s${s}/eval_lag2.json"
        [ -f "$f" ] || continue
        found=1
        python3 -c "
import json, sys
d = json.load(open('$f'))
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
worst = (d.get('worst_window') or {}).get('total_return',0)*100
neg = d.get('negative_windows',0)
sort = d.get('median_sortino',0)
flag = ' *** ALL-POS' if neg==0 and p10>0 else (' *** 0-NEG' if neg==0 else '')
print(f's$s: med={med:.2f}% p10={p10:.2f}% worst={worst:.2f}% neg={neg}/50 sort={sort:.2f}{flag}')
" 2>/dev/null
    done
    [ $found -eq 0 ] && echo "  (no eval results yet)"
}

echo "=== $(date -u +%FT%TZ) ==="
echo ""
echo "=== Wide73 C_low_tp (tp=0.02 adamw) ==="
check_evals "pufferlib_market/checkpoints/wide73_sweep/C_low_tp" C
echo ""
echo "=== Wide73 F_psn_lotp (tp=0.02 adamw + per-sym-norm) ==="
check_evals "pufferlib_market/checkpoints/wide73_sweep/F_psn_lotp" F
echo ""
echo "=== Wide73 D_muon (tp=0.05 muon) ==="
check_evals "pufferlib_market/checkpoints/wide73_sweep/D_muon" D
echo ""
echo "=== Currently training ==="
ps aux | grep "pufferlib_market.train" | grep -v grep | awk '{
    for(i=1;i<=NF;i++) {
        if($i ~ /--seed/) seed=$(i+1)
        if($i ~ /--checkpoint-dir/) ckpt=$(i+1)
    }
    gsub(/.*wide73_sweep\//, "wide73/", ckpt)
    gsub(/.*stocks17_sweep\//, "s17/", ckpt)
    print "  " ckpt " seed=" seed
}' | grep -v "^\s*$"
