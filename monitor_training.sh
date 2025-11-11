#!/bin/bash
# Monitor training process and GPU usage

echo "================================================================================"
echo "Training Process Monitor"
echo "================================================================================"
echo ""

# Find the training process
PID=$(pgrep -f "train_crypto_direct.py" | head -1)

if [ -z "$PID" ]; then
    echo "✅ No training process found - GPU is free!"
    echo ""
    echo "Ready to run MAE tests:"
    echo "  ./run_mae_tests_when_ready.sh"
    exit 0
fi

echo "Training Process Details:"
echo "-------------------------"
ps -p $PID -o pid,etime,pmem,rss,cmd

echo ""
echo "GPU Status:"
echo "-------------------------"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv 2>/dev/null || echo "GPU not accessible via nvidia-smi"

echo ""
echo "GPU Process List:"
echo "-------------------------"
fuser -v /dev/nvidia* 2>&1 | head -10

echo ""
echo "================================================================================"
echo "Options:"
echo "================================================================================"
echo "1. Wait for training to finish: ./run_mae_tests_when_ready.sh"
echo "2. Kill training (if needed):   kill -9 $PID"
echo "3. Check progress:              tail -f <log_file>"
echo "4. Run this again:              ./monitor_training.sh"
echo ""
