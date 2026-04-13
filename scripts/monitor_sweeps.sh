#!/bin/bash
# Auto-monitor and eval all sweep seeds that have finished training.
# Tests anchors (neg<25) in the current 8-model production ensemble.
# Usage: nohup bash scripts/monitor_sweeps.sh > /tmp/monitor_sweeps.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
LOG="pufferlib_market/checkpoints/monitor_sweeps.log"
ANCHOR_THRESH=25   # neg/100 threshold for "anchor" evaluation

# Current 8-model production ensemble (updated 2026-04-13: D_s5->I_s3 swap)
PROD_PRIMARY="pufferlib_market/prod_ensemble_screened32/C_s7.pt"
PROD_EXTRAS=(
  "pufferlib_market/prod_ensemble_screened32/D_s16.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s42.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s3.pt"
  "pufferlib_market/prod_ensemble_screened32/I_s3.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s2.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s14.pt"
  "pufferlib_market/prod_ensemble_screened32/D_s28.pt"
)

eval_one() {
  local ckpt="$1"
  local out="$2"
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$ckpt" \
    --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    > "$out" 2>/dev/null
}

test_in_ensemble() {
  local new_ckpt="$1"
  local label="$2"
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint "$PROD_PRIMARY" \
    --extra-checkpoints "${PROD_EXTRAS[@]}" "$new_ckpt" \
    --data-path "$FULL_VAL" \
    --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
    --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
    2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
med = d.get('median_total_return',0)*100
p10 = d.get('p10_total_return',0)*100
neg = d.get('negative_windows',0)
s = d.get('median_sortino',0)
print(f'  9-model+$label: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')
" 2>/dev/null
}

echo "" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === Monitor sweep run ===" | tee -a "$LOG"

# v1 sweeps
for variant in C D E F G H I; do
  sweep_dir="pufferlib_market/checkpoints/screened32_sweep/${variant}"
  lb="$sweep_dir/leaderboard_fulloos.csv"
  [ -d "$sweep_dir" ] || continue
  [ -f "$lb" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$lb"

  for seed_dir in "$sweep_dir"/s*/; do
    seed=$(basename "$seed_dir" | sed 's/s//')
    [ -f "$seed_dir/SKIPPED_EARLY_TERM" ] && continue
    [ -f "$seed_dir/final.pt" ] || continue
    out="$seed_dir/eval_full.json"
    [ -f "$out" ] && [ -s "$out" ] && continue  # already done

    ckpt="$seed_dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$seed_dir/best.pt"
    [ -f "$ckpt" ] || { echo "  ${variant}/s${seed}: no checkpoint, skip" | tee -a "$LOG"; continue; }

    echo "[$(date -u +%FT%TZ)] Evaluating ${variant}/s${seed}..." | tee -a "$LOG"
    eval_one "$ckpt" "$out"
    if [ -s "$out" ]; then
      result=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); w=d.get('worst_window',{}).get('total_return',0)*100; print(f'${variant} s${seed}: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')" 2>/dev/null)
      echo "  $result" | tee -a "$LOG"

      # Update leaderboard
      ts=$(date -u +%FT%TZ)
      stats=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); w=d.get('worst_window',{}).get('total_return',0)*100; print(f'{med:.2f},{p10:.2f},{w:.2f},{neg},{s:.2f}')" 2>/dev/null)
      [ -n "$stats" ] && echo "$ts,$variant,$seed,$stats,$ckpt" >> "$lb"

      # If promising anchor, test in ensemble
      neg=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('negative_windows',0))" 2>/dev/null || echo 99)
      if [ "$neg" -lt "$ANCHOR_THRESH" ]; then
        echo "  [ANCHOR CANDIDATE] ${variant}/s${seed}: neg=${neg} < ${ANCHOR_THRESH}" | tee -a "$LOG"
        test_in_ensemble "$ckpt" "${variant}_s${seed}" | tee -a "$LOG"
      fi
    else
      echo "  ${variant}/s${seed}: eval failed" | tee -a "$LOG"
    fi
  done
done

# v2 sweeps
for variant in C D; do
  sweep_dir="pufferlib_market/checkpoints/screened32_v2_sweep/${variant}"
  lb="$sweep_dir/leaderboard_fulloos.csv"
  [ -d "$sweep_dir" ] || continue
  [ -f "$lb" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$lb"

  for seed_dir in "$sweep_dir"/s*/; do
    seed=$(basename "$seed_dir" | sed 's/s//')
    [ -f "$seed_dir/SKIPPED_EARLY_TERM" ] && continue
    [ -f "$seed_dir/final.pt" ] || continue
    out="$seed_dir/eval_full.json"
    [ -f "$out" ] && [ -s "$out" ] && continue

    ckpt="$seed_dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$seed_dir/best.pt"
    [ -f "$ckpt" ] || continue

    echo "[$(date -u +%FT%TZ)] Evaluating v2_${variant}/s${seed}..." | tee -a "$LOG"
    eval_one "$ckpt" "$out"
    if [ -s "$out" ]; then
      result=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); print(f'v2_${variant} s${seed}: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')" 2>/dev/null)
      echo "  $result" | tee -a "$LOG"
      ts=$(date -u +%FT%TZ)
      stats=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); w=d.get('worst_window',{}).get('total_return',0)*100; print(f'{med:.2f},{p10:.2f},{w:.2f},{neg},{s:.2f}')" 2>/dev/null)
      [ -n "$stats" ] && echo "$ts,v2_$variant,$seed,$stats,$ckpt" >> "$lb"
      neg=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('negative_windows',0))" 2>/dev/null || echo 99)
      if [ "$neg" -lt "$ANCHOR_THRESH" ]; then
        echo "  [ANCHOR CANDIDATE] v2_${variant}/s${seed}: neg=${neg}" | tee -a "$LOG"
        test_in_ensemble "$ckpt" "v2_${variant}_s${seed}" | tee -a "$LOG"
      fi
    fi
  done
done

# gru sweeps
for variant in C D; do
  sweep_dir="pufferlib_market/checkpoints/screened32_gru_sweep/${variant}"
  lb="$sweep_dir/leaderboard_fulloos.csv"
  [ -d "$sweep_dir" ] || continue
  [ -f "$lb" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$lb"

  for seed_dir in "$sweep_dir"/s*/; do
    seed=$(basename "$seed_dir" | sed 's/s//')
    [ -f "$seed_dir/SKIPPED_EARLY_TERM" ] && continue
    [ -f "$seed_dir/final.pt" ] || continue
    out="$seed_dir/eval_full.json"
    [ -f "$out" ] && [ -s "$out" ] && continue

    ckpt="$seed_dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$seed_dir/best.pt"
    [ -f "$ckpt" ] || continue

    echo "[$(date -u +%FT%TZ)] Evaluating gru_${variant}/s${seed}..." | tee -a "$LOG"
    eval_one "$ckpt" "$out"
    if [ -s "$out" ]; then
      result=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); print(f'gru_${variant} s${seed}: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')" 2>/dev/null)
      echo "  $result" | tee -a "$LOG"
      ts=$(date -u +%FT%TZ)
      stats=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); w=d.get('worst_window',{}).get('total_return',0)*100; print(f'{med:.2f},{p10:.2f},{w:.2f},{neg},{s:.2f}')" 2>/dev/null)
      [ -n "$stats" ] && echo "$ts,gru_$variant,$seed,$stats,$ckpt" >> "$lb"
      neg=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('negative_windows',0))" 2>/dev/null || echo 99)
      if [ "$neg" -lt "$ANCHOR_THRESH" ]; then
        echo "  [ANCHOR CANDIDATE] gru_${variant}/s${seed}: neg=${neg}" | tee -a "$LOG"
        test_in_ensemble "$ckpt" "gru_${variant}_s${seed}" | tee -a "$LOG"
      fi
    fi
  done
done

# transformer sweeps
for variant in C D; do
  sweep_dir="pufferlib_market/checkpoints/screened32_transformer_sweep/${variant}"
  lb="$sweep_dir/leaderboard_fulloos.csv"
  [ -d "$sweep_dir" ] || continue
  [ -f "$lb" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$lb"

  for seed_dir in "$sweep_dir"/s*/; do
    seed=$(basename "$seed_dir" | sed 's/s//')
    [ -f "$seed_dir/SKIPPED_EARLY_TERM" ] && continue
    [ -f "$seed_dir/final.pt" ] || continue
    out="$seed_dir/eval_full.json"
    [ -f "$out" ] && [ -s "$out" ] && continue

    ckpt="$seed_dir/val_best.pt"
    [ -f "$ckpt" ] || ckpt="$seed_dir/best.pt"
    [ -f "$ckpt" ] || continue

    echo "[$(date -u +%FT%TZ)] Evaluating transformer_${variant}/s${seed}..." | tee -a "$LOG"
    eval_one "$ckpt" "$out"
    if [ -s "$out" ]; then
      result=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); print(f'transformer_${variant} s${seed}: med={med:.2f}% p10={p10:.2f}% neg={neg}/100 sort={s:.2f}')" 2>/dev/null)
      echo "  $result" | tee -a "$LOG"
      ts=$(date -u +%FT%TZ)
      stats=$(python3 -c "import json; d=json.load(open('$out')); med=d.get('median_total_return',0)*100; p10=d.get('p10_total_return',0)*100; neg=d.get('negative_windows',0); s=d.get('median_sortino',0); w=d.get('worst_window',{}).get('total_return',0)*100; print(f'{med:.2f},{p10:.2f},{w:.2f},{neg},{s:.2f}')" 2>/dev/null)
      [ -n "$stats" ] && echo "$ts,transformer_$variant,$seed,$stats,$ckpt" >> "$lb"
      neg=$(python3 -c "import json; d=json.load(open('$out')); print(d.get('negative_windows',0))" 2>/dev/null || echo 99)
      if [ "$neg" -lt "$ANCHOR_THRESH" ]; then
        echo "  [ANCHOR CANDIDATE] transformer_${variant}/s${seed}: neg=${neg}" | tee -a "$LOG"
        test_in_ensemble "$ckpt" "transformer_${variant}_s${seed}" | tee -a "$LOG"
      fi
    fi
  done
done

echo "[$(date -u +%FT%TZ)] === Monitor run complete ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Print current leaderboard (anchors only)
echo "=== CURRENT ANCHORS (neg < $ANCHOR_THRESH) ===" | tee -a "$LOG"
for variant in C D E F G H I; do
  lb="pufferlib_market/checkpoints/screened32_sweep/${variant}/leaderboard_fulloos.csv"
  [ -f "$lb" ] && awk -F, -v thresh="$ANCHOR_THRESH" 'NR>1 && $7+0 < thresh {print}' "$lb" | sort -t, -k4 -rn | while read line; do echo "  $line" | tee -a "$LOG"; done
done
for variant in C D; do
  lb="pufferlib_market/checkpoints/screened32_v2_sweep/${variant}/leaderboard_fulloos.csv"
  [ -f "$lb" ] && awk -F, -v thresh="$ANCHOR_THRESH" 'NR>1 && $7+0 < thresh {print}' "$lb" | sort -t, -k4 -rn | while read line; do echo "  v2: $line" | tee -a "$LOG"; done
done
for variant in C D; do
  lb="pufferlib_market/checkpoints/screened32_transformer_sweep/${variant}/leaderboard_fulloos.csv"
  [ -f "$lb" ] && awk -F, -v thresh="$ANCHOR_THRESH" 'NR>1 && $7+0 < thresh {print}' "$lb" | sort -t, -k4 -rn | while read line; do echo "  transformer: $line" | tee -a "$LOG"; done
done
for variant in C D; do
  lb="pufferlib_market/checkpoints/screened32_gru_sweep/${variant}/leaderboard_fulloos.csv"
  [ -f "$lb" ] && awk -F, -v thresh="$ANCHOR_THRESH" 'NR>1 && $7+0 < thresh {print}' "$lb" | sort -t, -k4 -rn | while read line; do echo "  gru: $line" | tee -a "$LOG"; done
done
