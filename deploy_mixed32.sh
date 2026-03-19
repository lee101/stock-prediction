#!/bin/bash
# deploy_mixed32.sh — Safe deployment rollout for mixed32 daily RL checkpoint.
#
# Verifies the checkpoint exists, runs market-sim eval on CPU to confirm
# Sortino, backs up the current orchestrator config, runs a dry-run cycle,
# and prints deployment/rollback instructions.  Does NOT auto-restart the
# live trading bot — that is left to the operator.
#
# Usage:
#   bash deploy_mixed32.sh
#
# Requirements: .venv313 with all pufferlib_market + unified_orchestrator deps.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

CHECKPOINT="pufferlib_market/checkpoints/autoresearch_mixed32_daily/wd_05/best.pt"
ORCH_CONFIG="unified_orchestrator/orchestrator.py"
BACKUP_DIR="backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="${BACKUP_DIR}/orchestrator_${TIMESTAMP}.py"

# Colours for terminal output (no-op if not a tty)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' NC=''
fi

ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# -----------------------------------------------------------------------
# 0. Activate virtual environment
# -----------------------------------------------------------------------
echo "=== Step 0: Activate .venv313 ==="
if [ -f ".venv313/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv313/bin/activate
    ok "Activated .venv313 (python=$(python --version 2>&1))"
else
    fail ".venv313 not found. Please create it first."
fi

# -----------------------------------------------------------------------
# 1. Verify checkpoint exists
# -----------------------------------------------------------------------
echo ""
echo "=== Step 1: Verify checkpoint ==="
if [ -f "$CHECKPOINT" ]; then
    CKPT_SIZE=$(stat --printf='%s' "$CHECKPOINT" 2>/dev/null || stat -f '%z' "$CHECKPOINT" 2>/dev/null || echo "?")
    CKPT_MTIME=$(stat --printf='%y' "$CHECKPOINT" 2>/dev/null || stat -f '%Sm' "$CHECKPOINT" 2>/dev/null || echo "?")
    ok "Checkpoint found: $CHECKPOINT"
    echo "    Size:     $CKPT_SIZE bytes"
    echo "    Modified: $CKPT_MTIME"
else
    fail "Checkpoint not found: $CHECKPOINT"
fi

# -----------------------------------------------------------------------
# 2. Run fast_marketsim_eval on CPU to confirm Sortino
# -----------------------------------------------------------------------
echo ""
echo "=== Step 2: Market-sim evaluation (CPU) ==="
EVAL_OUTPUT="/tmp/deploy_mixed32_eval_${TIMESTAMP}.csv"
echo "Running fast_marketsim_eval.py for mixed32 checkpoints..."
echo "(this may take a few minutes)"

python -m pufferlib_market.fast_marketsim_eval \
    --root "$REPO" \
    --checkpoint-dirs "pufferlib_market/checkpoints/autoresearch_mixed32_daily" \
    --output "$EVAL_OUTPUT" \
    --periods "60,90,120,180" \
    --sequential \
    --no-compile \
    --sort-period 120 \
    --cache-path "/tmp/deploy_mixed32_eval_cache_${TIMESTAMP}.json"

WD05_SORTINO="N/A"
if [ -f "$EVAL_OUTPUT" ]; then
    ok "Eval results written to $EVAL_OUTPUT"
    echo ""
    echo "--- wd_05 results ---"
    # Show rows matching wd_05
    head -1 "$EVAL_OUTPUT"
    grep "wd_05" "$EVAL_OUTPUT" || warn "No wd_05 rows found in eval output"
    echo ""
    # Extract 120d Sortino for wd_05
    WD05_SORTINO=$(python -c "
import csv, sys
with open('$EVAL_OUTPUT') as f:
    for row in csv.DictReader(f):
        if 'wd_05' in row.get('checkpoint','') and row.get('period','') == '120':
            print(row['sortino'])
            sys.exit(0)
print('N/A')
" 2>/dev/null || echo "N/A")
    echo "  wd_05 120d Sortino: $WD05_SORTINO"
    if [ "$WD05_SORTINO" = "N/A" ]; then
        warn "Could not extract 120d Sortino for wd_05. Review $EVAL_OUTPUT manually."
    fi
else
    warn "Eval output not created. Check for errors above."
fi

# -----------------------------------------------------------------------
# 3. Backup current orchestrator config
# -----------------------------------------------------------------------
echo ""
echo "=== Step 3: Backup orchestrator config ==="
mkdir -p "$BACKUP_DIR"
if [ -f "$ORCH_CONFIG" ]; then
    cp "$ORCH_CONFIG" "$BACKUP_PATH"
    ok "Backed up $ORCH_CONFIG -> $BACKUP_PATH"
else
    warn "Orchestrator config not found at $ORCH_CONFIG (may be normal in worktree)"
fi

# -----------------------------------------------------------------------
# 4. Dry-run: single cycle of unified orchestrator
# -----------------------------------------------------------------------
echo ""
echo "=== Step 4: Dry-run orchestrator (60s timeout) ==="
echo "Running: timeout 60 python -m unified_orchestrator.orchestrator --dry-run --once"
set +e
timeout 60 python -m unified_orchestrator.orchestrator --dry-run --once 2>&1 | tail -80
DRY_RUN_EXIT=$?
set -e

if [ $DRY_RUN_EXIT -eq 0 ]; then
    ok "Dry-run completed successfully"
elif [ $DRY_RUN_EXIT -eq 124 ]; then
    warn "Dry-run timed out after 60s (expected if waiting for market data or LLM)"
else
    warn "Dry-run exited with code $DRY_RUN_EXIT (review output above)"
fi

# -----------------------------------------------------------------------
# 5. Deployment summary
# -----------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  DEPLOYMENT SUMMARY"
echo "=================================================================="
echo ""
echo "  Checkpoint:  $CHECKPOINT"
echo "  Eval output: $EVAL_OUTPUT"
echo "  Config backup: $BACKUP_PATH"
echo "  120d Sortino (wd_05): $WD05_SORTINO"
echo ""
echo "  To deploy this checkpoint to production:"
echo ""
echo "    1. Edit unified_orchestrator/orchestrator.py"
echo "       Add to CRYPTO_CHECKPOINT_CANDIDATES (at the top):"
echo ""
echo '       REPO / "pufferlib_market/checkpoints/autoresearch_mixed32_daily/wd_05/best.pt",'
echo ""
echo "    2. If replacing the current primary checkpoint, move the old"
echo "       entry down or comment it out."
echo ""
echo "    3. Kill the running orchestrator:"
echo "         pgrep -f 'unified_orchestrator.orchestrator' | xargs kill"
echo ""
echo "    4. Restart with:"
echo "         bash unified_orchestrator/deploy_and_monitor.sh --live"
echo ""
echo "    5. Monitor for 1h:"
echo "         tail -f logs/unified_orchestrator.log"
echo ""

# -----------------------------------------------------------------------
# 6. Rollback instructions
# -----------------------------------------------------------------------
echo "=================================================================="
echo "  ROLLBACK INSTRUCTIONS"
echo "=================================================================="
echo ""
echo "  If the new checkpoint underperforms or causes issues:"
echo ""
echo "    1. Restore the orchestrator config backup:"
echo "         cp $BACKUP_PATH $ORCH_CONFIG"
echo ""
echo "    2. Kill the running orchestrator:"
echo "         pgrep -f 'unified_orchestrator.orchestrator' | xargs kill"
echo ""
echo "    3. Restart with the old config:"
echo "         bash unified_orchestrator/deploy_and_monitor.sh --live"
echo ""
echo "    4. Verify the old checkpoint is loaded in the logs:"
echo "         grep 'RL bridge loaded' logs/unified_orchestrator.log | tail -3"
echo ""
echo "=================================================================="
echo "  Script complete. No live changes were made."
echo "=================================================================="
