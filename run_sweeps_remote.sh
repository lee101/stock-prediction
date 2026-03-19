#!/bin/bash
# Orchestrate sweeps on the 5090 machine (93.127.141.100)
# Usage: bash scripts/run_sweeps_remote.sh [sweep_type]
# sweep_type: all, seed, rw, lr, epochs, seq, arch, fine, lora

REMOTE="administrator@93.127.141.100"
REMOTE_PASS="zf@nVLk01@"
REMOTE_DIR="/nvme0n1-disk/code/stock-prediction"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH="sshpass -p '$REMOTE_PASS' ssh -o StrictHostKeyChecking=no $REMOTE"
SCP="sshpass -p '$REMOTE_PASS' scp -o StrictHostKeyChecking=no"

SWEEP_TYPE="${1:-all}"

echo "=== SUI Sweep Orchestrator ==="
echo "Sweep type: $SWEEP_TYPE"
echo "Remote: $REMOTE:$REMOTE_DIR"

# Step 1: Sync code to remote
echo ""
echo "=== Step 1: Syncing code ==="
sshpass -p "$REMOTE_PASS" rsync -avz --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='tensorboard_logs' --exclude='wandb' \
    --exclude='checkpoints' --exclude='chronos2_finetuned' \
    --include='binancechronossolexperiment/***' \
    --include='binanceneural/***' \
    --include='differentiable_loss_utils.py' \
    --include='scripts/***' \
    --include='src/***' \
    --include='preaug/***' \
    --include='traininglib/***' \
    --include='nanochat/***' \
    --include='requirements.txt' \
    -e "ssh -o StrictHostKeyChecking=no" \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

# Step 2: Sync data (forecast caches, training data)
echo ""
echo "=== Step 2: Syncing data ==="
sshpass -p "$REMOTE_PASS" rsync -avz \
    --include='trainingdatahourlybinance/***' \
    --include='binancechronossolexperiment/forecast_cache_sui_10bp/***' \
    -e "ssh -o StrictHostKeyChecking=no" \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

# Step 3: Run sweep
echo ""
echo "=== Step 3: Running sweep ($SWEEP_TYPE) ==="
if [ "$SWEEP_TYPE" = "lora" ]; then
    sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" \
        "cd $REMOTE_DIR && python3 binancechronossolexperiment/run_lora_aug_sweep.py --symbol SUIUSDT --lora-only" &
    REMOTE_PID=$!
    echo "Remote PID: $REMOTE_PID (LoRA sweep)"
else
    sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" \
        "cd $REMOTE_DIR && python3 binancechronossolexperiment/run_comprehensive_sweep.py --sweep-type $SWEEP_TYPE --symbol SUIUSDT" &
    REMOTE_PID=$!
    echo "Remote PID: $REMOTE_PID (policy sweep: $SWEEP_TYPE)"
fi

echo ""
echo "Sweep running in background (PID $REMOTE_PID)"
echo "To check progress: sshpass -p '$REMOTE_PASS' ssh $REMOTE 'tail -20 $REMOTE_DIR/sweep_*.json 2>/dev/null || echo running...'"
echo "To fetch results:  sshpass -p '$REMOTE_PASS' scp $REMOTE:$REMOTE_DIR/binancechronossolexperiment/sweep_*.json binancechronossolexperiment/"

wait $REMOTE_PID
echo "Sweep complete!"

# Step 4: Fetch results
echo ""
echo "=== Step 4: Fetching results ==="
sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no \
    "$REMOTE:$REMOTE_DIR/binancechronossolexperiment/sweep_*.json" \
    "$LOCAL_DIR/binancechronossolexperiment/" 2>/dev/null

sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no \
    "$REMOTE:$REMOTE_DIR/binancechronossolexperiment/lora_sweep_*.json" \
    "$LOCAL_DIR/binancechronossolexperiment/" 2>/dev/null

echo "Done! Results in binancechronossolexperiment/"
