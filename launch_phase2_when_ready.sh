#!/bin/bash
# Wait for stocks10 sweep to finish, then launch phase 2
STOCKS10_PID=4014799
echo "Waiting for stocks10 sweep (PID $STOCKS10_PID) to finish..."
while kill -0 $STOCKS10_PID 2>/dev/null; do
    sleep 30
    echo -n "."
done
echo ""
echo "[$(date)] stocks10 sweep done. Launching Phase 2..."
nohup bash /nvme0n1-disk/code/stock-prediction/run_phase2_sweeps.sh \
  > /nvme0n1-disk/code/stock-prediction/phase2_output.log 2>&1 &
echo "Phase 2 launched as PID $!"
