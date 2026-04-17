#!/bin/bash
# Auto-test new screened32 sweep candidates as 14th member of the prod
# 13-model v5 ensemble. Watches all variant directories under
# pufferlib_market/checkpoints/screened32_sweep/ and triggers a deploy-gate
# (fb=5, lev=1.0,1.5, lag=2, 263 windows) test for any new seed whose
# standalone OOS eval has neg <= 17 (baseline + 6).
#
# Designed to run in a loop or as a cron. Each candidate is tested at most
# once (state tracked via a marker file). Logs to docs/auto_test_log.md.
#
# Usage:
#   bash scripts/auto_test_screened32_candidates.sh                # one pass
#   while true; do bash $0; sleep 600; done                        # continuous
#
# Tighter neg threshold can be set via NEG_BAR env var (default 17).

set -uo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

SWEEP_ROOT="pufferlib_market/checkpoints/screened32_sweep"
LOG_DIR="docs/auto_test_screened32"
LOG_MD="$LOG_DIR/auto_test_log.md"
mkdir -p "$LOG_DIR"

NEG_BAR="${NEG_BAR:-17}"  # baseline 11 + 6 cushion
BASELINE_MED_MO="0.0752"  # 13-model v6 deploy gate (D_s3→AD_s4 swap, 2026-04-17)

[ -f "$LOG_MD" ] || cat > "$LOG_MD" <<EOF
# Auto-test log: screened32 candidates as 14th ensemble member

Watches \`$SWEEP_ROOT\` for newly-evaluated seeds and tests any with
standalone neg <= \$NEG_BAR (default $NEG_BAR) at the deploy-gate cell
(fb=5, lev=1.0+1.5, lag=2, 263 windows). Threshold motivation:
AD_s9 with neg=11 standalone failed the 14m gate, so candidates with
neg > 17 standalone are essentially guaranteed to fail too.

| timestamp | candidate | standalone neg | 14m med (1x) | 14m neg | 14m sortino | 14m verdict |
|---|---|---:|---:|---:|---:|---|
EOF

# Read current 13-model defaults from src/daily_stock_defaults.py.
mapfile -t BASE_CKPTS < <(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS
print(DEFAULT_CHECKPOINT)
for c in DEFAULT_EXTRA_CHECKPOINTS:
    print(c)
PY
)

if [ ${#BASE_CKPTS[@]} -ne 13 ]; then
  echo "auto_test: expected 13 base ckpts, got ${#BASE_CKPTS[@]}" >&2
  exit 2
fi

candidates_tested=0
candidates_skipped=0
candidates_promising=0

for variant_dir in "$SWEEP_ROOT"/*/; do
  variant=$(basename "$variant_dir")
  # Skip non-variant dirs
  [[ "$variant" == "relu_sq_D" || "$variant" == "resmlp_D" ]] && continue
  [ -d "$variant_dir" ] || continue

  for seed_dir in "$variant_dir"s*/; do
    [ -d "$seed_dir" ] || continue
    seed=$(basename "$seed_dir" | sed 's/^s//')

    eval_json="$seed_dir/eval_full.json"
    [ -s "$eval_json" ] || continue  # not yet evaluated by sweep

    # Marker for already-tested
    marker="$seed_dir/.auto_test_14m_done"
    [ -f "$marker" ] && { candidates_skipped=$((candidates_skipped + 1)); continue; }

    # Read standalone neg + checkpoint path from eval_full.json
    info=$(python3 - "$eval_json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
neg = int(d.get('negative_windows', 999))
ckpt = d.get('checkpoint', '')
med = float(d.get('median_total_return', 0.0)) * 100
print(f"{neg}|{ckpt}|{med:.2f}")
PY
)
    [ -z "$info" ] && continue
    standalone_neg=$(echo "$info" | cut -d'|' -f1)
    candidate_ckpt=$(echo "$info" | cut -d'|' -f2)
    standalone_med=$(echo "$info" | cut -d'|' -f3)

    [ -z "$candidate_ckpt" ] && continue
    [ -f "$candidate_ckpt" ] || continue

    # Skip if candidate is already a prod ensemble member (deduped by basename)
    candidate_base=$(basename "$candidate_ckpt" .pt)
    is_prod=0
    for prod_ckpt in "${BASE_CKPTS[@]}"; do
      prod_base=$(basename "$prod_ckpt" .pt)
      # Match by prod ensemble naming convention: {VARIANT}_s{seed} matches sweep dir variant/s{seed}
      if [ "$prod_base" = "${variant}_s${seed}" ]; then
        is_prod=1
        break
      fi
    done
    if [ "$is_prod" -eq 1 ]; then
      ts=$(date -u +%FT%TZ)
      echo "$ts $variant/s$seed already in prod ensemble — skip"
      echo "| $ts | $variant/s$seed | $standalone_neg | — | — | — | skip (in prod ensemble) |" >> "$LOG_MD"
      touch "$marker"
      candidates_skipped=$((candidates_skipped + 1))
      continue
    fi

    # Filter: standalone neg too high → skip with marker
    if [ "$standalone_neg" -gt "$NEG_BAR" ]; then
      ts=$(date -u +%FT%TZ)
      echo "$ts $variant/s$seed standalone neg=$standalone_neg > $NEG_BAR — skip (no 14m test)"
      echo "| $ts | $variant/s$seed | $standalone_neg | — | — | — | skip (over neg bar) |" >> "$LOG_MD"
      touch "$marker"
      candidates_skipped=$((candidates_skipped + 1))
      continue
    fi

    # Run 14m gate at fb=5, lev=1.0,1.5
    out_dir="docs/realism_gate_${variant}_s${seed}_14m"
    if [ ! -f "$out_dir/screened32_single_offset_val_full_realism_gate.json" ]; then
      mkdir -p "$out_dir"
      echo "[$(date -u +%FT%TZ)] AUTO-TEST $variant/s$seed (standalone neg=$standalone_neg, med=$standalone_med%)..."
      python scripts/screened32_realism_gate.py \
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \
        --window-days 50 \
        --fill-buffer-bps-grid 5 \
        --max-leverage-grid 1.0,1.5 \
        --decision-lag 2 \
        --out-dir "$out_dir" \
        --checkpoints "${BASE_CKPTS[@]}" "$candidate_ckpt" \
        > "$out_dir/run.log" 2>&1
      candidates_tested=$((candidates_tested + 1))
    fi

    # Extract verdict from JSON
    verdict_line=$(python3 - "$out_dir/screened32_single_offset_val_full_realism_gate.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cells = d.get('cells', [])
# Find fb=5 lev=1
for c in cells:
    if abs(c.get('fill_buffer_bps', 0) - 5) < 0.1 and abs(c.get('max_leverage', 0) - 1.0) < 0.1:
        med_mo = float(c.get('median_monthly_return', 0))
        n_neg  = int(c.get('n_neg', 0))
        sort   = float(c.get('median_sortino', 0))
        baseline = 0.0752  # v6 deploy gate
        d_med = med_mo - baseline
        verdict = "PROMISING" if (d_med > 0.001 and n_neg <= 11) else "reject"
        print(f"{med_mo:.4f}|{n_neg}|{sort:.2f}|{verdict}")
        break
PY
)
    [ -z "$verdict_line" ] && continue

    med_14m=$(echo "$verdict_line" | cut -d'|' -f1)
    neg_14m=$(echo "$verdict_line" | cut -d'|' -f2)
    sort_14m=$(echo "$verdict_line" | cut -d'|' -f3)
    verdict=$(echo "$verdict_line" | cut -d'|' -f4)

    ts=$(date -u +%FT%TZ)
    echo "$ts $variant/s$seed → 14m med=$med_14m neg=$neg_14m sortino=$sort_14m → $verdict"
    echo "| $ts | $variant/s$seed | $standalone_neg | ${med_14m} | $neg_14m | $sort_14m | $verdict |" >> "$LOG_MD"
    [ "$verdict" = "PROMISING" ] && candidates_promising=$((candidates_promising + 1))
    touch "$marker"
  done
done

echo "[$(date -u +%FT%TZ)] auto_test pass: tested=$candidates_tested skipped=$candidates_skipped promising=$candidates_promising"
