#!/bin/bash
# Seed sweep for screened32 augmented training.
# 32 curated stocks × 35x augmentation (5 shifts × 7 vol scales)
# 20,377 train timesteps vs 2,911 for stocks17.
#
# Variants:
#   C: tp=0.02 adamw (most stable, stocks17 champion)
#   D: tp=0.05 muon  (best median in stocks17 D_muon)
#
# Usage:
#   nohup bash scripts/sweep_screened32.sh C > /tmp/s32_sweep_C.log 2>&1 &
#   nohup bash scripts/sweep_screened32.sh D > /tmp/s32_sweep_D.log 2>&1 &

set -u
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

# earlyoom kills training processes under memory pressure — stop it
echo "ilu" | sudo -S systemctl stop earlyoom 2>/dev/null && echo "[sweep] earlyoom stopped" || echo "[sweep] earlyoom already stopped"

export TMPDIR="$(pwd)/.tmp_train"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$(pwd)/.tmp_train/triton_cache"
export TORCH_COMPILE_DEBUG=0
mkdir -p "$TRITON_CACHE_DIR"

VARIANT="${1:-C}"
TRAIN="pufferlib_market/data/screened32_augmented_train.bin"
VAL="pufferlib_market/data/screened32_augmented_val.bin"
FULL_VAL="pufferlib_market/data/screened32_single_offset_val_full.bin"
ALLOW_SHORTS=0  # default: shorts disabled (--disable-shorts passed)

case "$VARIANT" in
  C) TP=0.02; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=() ;;
  D) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;
  E) TP=0.02; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;  # Muon at C's trade-penalty
  F) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=() ;;                  # AdamW at D's trade-penalty
  G) TP=0.07; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;  # Muon higher tp (more selective)
  H) TP=0.10; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;  # Muon very high tp (fewest trades)
  I) TP=0.03; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;  # Muon mid tp (between C and D)
  J) TP=0.04; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon) ;;  # Muon between I(0.03) and D(0.05)
  K) TP=0.02; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--weight-decay 0.005) ;;  # AdamW+wd: C with regularization
  L) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--weight-decay 0.005) ;;  # AdamW+wd at D's trade-penalty
  P) TP=0.05; STEPS=15000000; MAX_STEPS=50; EXTRA_FLAGS=(--optimizer muon --val-eval-windows 100) ;;  # D-equiv short 50-day episodes (matches 50-day eval window)
  R) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon); VAL="pufferlib_market/data/screened32_recent_val.bin" ;;  # D-equiv but recent val (Dec-Apr 2026)
  S) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --downside-penalty 0.05) ;;  # D + downside penalty (Sortino focus)
  T) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --smooth-downside-penalty 0.05) ;;  # D + smooth downside penalty
  U) TP=0.03; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --downside-penalty 0.05) ;;  # I + downside penalty
  V) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --arch transformer --hidden-size 512) ;;  # D-equiv but transformer symbol-attention arch
  W) TP=0.02; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--arch transformer --hidden-size 512) ;;                   # C-equiv but transformer arch
  X) TP=0.02; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --downside-penalty 0.05) ;;  # E + downside penalty (low tp + muon + dp)
  Y) TP=0.03; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --smooth-downside-penalty 0.05) ;;  # U + smooth dp variant
  Z) TP=0.04; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --downside-penalty 0.05) ;;  # J + downside penalty
  N) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --val-eval-windows 100) ;;  # D + 100 val windows (harder to overfit val)
  M) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --val-eval-windows 100); VAL="pufferlib_market/data/screened32_recent_val.bin" ;;  # N + recent val (Dec-Apr 2026)
  A) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon); ALLOW_SHORTS=1 ;;  # D-equiv WITH short selling (diversity via bear-market shorting)
  B) TP=0.03; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon); ALLOW_SHORTS=1 ;;  # I-equiv WITH short selling
  Q) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --hidden-size 2048) ;;  # D-equiv but wider MLP (h=2048 vs h=1024)
  AA) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --lr-schedule cosine --anneal-ent --anneal-clip) ;;  # D + cosine LR + ent/clip anneal (E1 training-knob bundle)
  AB) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --group-relative-mix 0.3) ;;  # D + group-relative advantage (mix=0.3) per E1
  AC) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon --lr-schedule cosine --anneal-ent --anneal-clip --group-relative-mix 0.3) ;;  # D + full E1 stack
  AD) TP=0.05; STEPS=15000000; MAX_STEPS=252; EXTRA_FLAGS=(--optimizer muon); TRAIN="pufferlib_market/data/screened32_aprcrash_augmented_train.bin"; VAL="pufferlib_market/data/screened32_aprcrash_augmented_val.bin" ;;  # D baseline on aprcrash data (train through 2026-02-28 incl Mar-Apr crash context)
  *) echo "Unknown variant $VARIANT (use A B C D E F G H I J K L M N P Q R S T U V W X Y Z AA AB AC AD)"; exit 1 ;;
esac

SEEDS=${SEEDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}

CKPT_ROOT="pufferlib_market/checkpoints/screened32_sweep/${VARIANT}"
LOG="$CKPT_ROOT/leaderboard_fulloos.csv"
mkdir -p "$CKPT_ROOT"
[ -f "$LOG" ] || echo "timestamp,variant,seed,med_pct,p10_pct,worst_pct,neg_count,med_sortino,checkpoint" > "$LOG"

train_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"
  mkdir -p "$dir"
  local attempt=1
  local max_attempts=20
  while [ $attempt -le $max_attempts ]; do
    [ -f "$dir/final.pt" ] && { echo "  s${seed}: final.pt exists, training complete"; return 0; }
    echo "[$(date -u +%FT%TZ)] [${VARIANT}] seed ${seed} training (attempt ${attempt})..."
    local shorts_flag=("--disable-shorts")
    [ "${ALLOW_SHORTS:-0}" -eq 1 ] && shorts_flag=()
    python -u -m pufferlib_market.train \
        --data-path "$TRAIN" --val-data-path "$VAL" \
        --total-timesteps "$STEPS" \
        --max-steps "${MAX_STEPS:-252}" \
        --trade-penalty "$TP" \
        --hidden-size 1024 \
        --anneal-lr \
        "${shorts_flag[@]}" \
        --val-eval-windows 50 \
        --early-stop-val-neg-threshold "${EARLY_STOP_NEG:-25}" \
        --early-stop-val-neg-patience "${EARLY_STOP_PATIENCE:-2}" \
        "${EXTRA_FLAGS[@]}" \
        --num-envs 128 \
        --seed "$seed" \
        --checkpoint-dir "$dir" >> "$dir/train.log" 2>&1 &
    local train_pid=$!
    # Early termination: check first val (update 50)
    local first_val_checked=0
    while kill -0 $train_pid 2>/dev/null; do
      sleep 45
      if [ "$first_val_checked" -eq 0 ]; then
        local first_val
        first_val=$(grep -m1 "\[val\]" "$dir/train.log" 2>/dev/null)
        if [ -n "$first_val" ]; then
          first_val_checked=1
          local neg n_val
          neg=$(echo "$first_val" | grep -oP 'neg=\K[0-9]+(?=/)')
          n_val=$(echo "$first_val" | grep -oP 'neg=[0-9]+/\K[0-9]+')
          echo "  s${seed}: first val neg=${neg}/${n_val}"
          # Skip if >80% windows are negative (definitely bad). Scale threshold with n_val.
          local thresh=$(( (${n_val:-50} * 80) / 100 ))
          if [ -n "$neg" ] && [ "$neg" -gt "$thresh" ]; then
            echo "  s${seed}: early termination (neg=${neg}/${n_val} > ${thresh})"
            kill "$train_pid" 2>/dev/null
            wait "$train_pid" 2>/dev/null
            touch "$dir/SKIPPED_EARLY_TERM"
            return 1
          fi
        fi
      fi
    done
    wait "$train_pid"
    local exit_code=$?
    [ -f "$dir/final.pt" ] && { echo "[$(date -u +%FT%TZ)] [${VARIANT}] seed ${seed} done"; return 0; }
    [ -f "$dir/SKIPPED_EARLY_TERM" ] && { echo "  s${seed}: skipped by early termination"; return 1; }
    echo "[$(date -u +%FT%TZ)] [${VARIANT}] seed ${seed} exit_code=${exit_code}, retrying (attempt $((attempt+1)))..."
    attempt=$((attempt + 1))
  done
  echo "[$(date -u +%FT%TZ)] [${VARIANT}] seed ${seed} FAILED after ${max_attempts} attempts"
  return 1
}

eval_one() {
  local seed=$1
  local dir="$CKPT_ROOT/s${seed}"

  # Collect candidates: val_best.pt + late-training checkpoints
  local candidates=()
  for f in "$dir/val_best.pt" "$dir/best_neg.pt" "$dir/best.pt"; do
    [ -f "$f" ] && candidates+=("$f")
  done
  # Also add the highest-numbered update checkpoint (late model)
  local late_ckpt
  late_ckpt=$(ls "$dir"/update_*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$late_ckpt" ] && candidates+=("$late_ckpt")

  [ ${#candidates[@]} -eq 0 ] && { echo "  s${seed}: no checkpoint"; return; }

  local best_ckpt="" best_neg=999 best_med=-99 best_stats=""
  for ckpt in "${candidates[@]}"; do
    local tmp_out="${ckpt%.pt}_oos_eval.json"
    if [ ! -f "$tmp_out" ]; then
      python -m pufferlib_market.evaluate_holdout \
          --checkpoint "$ckpt" --data-path "$FULL_VAL" \
          --eval-hours 50 --n-windows 100 --fee-rate 0.001 \
          --fill-buffer-bps 5.0 --decision-lag 2 --deterministic --no-early-stop \
          > "$tmp_out" 2>/dev/null || continue
    fi
    local stats neg med
    stats=$(python3 - "$tmp_out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
med   = d.get('median_total_return', 0)*100
p10   = d.get('p10_total_return', 0)*100
worst = d.get('worst_window',{}).get('total_return', 0)*100
neg   = d.get('negative_windows', 0)
sort  = d.get('median_sortino', 0)
print(f"{med:.2f},{p10:.2f},{worst:.2f},{neg},{sort:.2f}")
PY
    )
    [ -z "$stats" ] && continue
    neg=$(echo "$stats" | cut -d, -f4)
    med=$(echo "$stats" | cut -d, -f1 | awk '{printf "%d", $1*100}')
    echo "    s${seed} $(basename "$ckpt"): neg=$neg med=${stats%%,*}%"
    if [ "$neg" -lt "$best_neg" ] || { [ "$neg" -eq "$best_neg" ] && [ "$med" -gt "$best_med" ]; }; then
      best_neg="$neg"; best_med="$med"; best_ckpt="$ckpt"; best_stats="$stats"
    fi
  done

  [ -z "$best_ckpt" ] && { echo "  s${seed}: all evals failed"; return; }
  cp "$best_ckpt" "$dir/eval_best.pt" 2>/dev/null
  # Write canonical eval_full.json for skip-check
  local src_json="${best_ckpt%.pt}_oos_eval.json"
  [ -f "$src_json" ] && cp "$src_json" "$dir/eval_full.json"

  ts=$(date -u +%FT%TZ)
  echo "$ts,$VARIANT,$seed,$best_stats,$best_ckpt" >> "$LOG"
  echo "  s${seed}: BEST neg=$best_neg  $best_stats  ($(basename "$best_ckpt"))"
}

nts=$(python3 -c "import struct; h=open('$TRAIN','rb').read(24); v=struct.unpack_from('<IIII',h,4); print(f'{v[1]} syms, {v[2]} ts, {v[3]} feats')" 2>/dev/null || echo "?")
echo "=== screened32 variant $VARIANT : tp=$TP steps=$STEPS ==="
echo "    train=$TRAIN ($nts)"
echo

for s in $SEEDS; do
  [ -f "$CKPT_ROOT/s${s}/eval_full.json" ] && [ -s "$CKPT_ROOT/s${s}/eval_full.json" ] && { echo "[$(date -u +%FT%TZ)] s${s}: already evaled on full OOS, skip"; continue; }
  [ -f "$CKPT_ROOT/s${s}/SKIPPED_EARLY_TERM" ] && { echo "  s${s}: skipped by early termination, skip"; continue; }
  if [ ! -f "$CKPT_ROOT/s${s}/final.pt" ]; then
    train_one "$s" || continue
  else
    echo "  s${s}: final.pt exists, skipping training"
  fi
  eval_one "$s"
done

echo
echo "=== [${VARIANT}] LEADERBOARD (full OOS, 100 windows) ==="
sort -t, -k4 -rn "$LOG" | head -10
echo
echo "=== Positive-p10 (p10>0, neg<10) ==="
awk -F, 'NR>1 && $5+0>0 && $7+0<10 {print}' "$LOG" | sort -t, -k4 -rn
