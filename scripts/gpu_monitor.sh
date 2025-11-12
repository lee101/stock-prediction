#!/bin/bash
# GPU Health Monitor - Detects when GPU fails and logs diagnostics

LOG_FILE="$HOME/gpu_monitor.log"
CHECK_INTERVAL=60  # seconds

echo "[$(date)] GPU Monitor started" | tee -a "$LOG_FILE"

while true; do
    if ! nvidia-smi &>/dev/null; then
        echo "" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"
        echo "[$(date)] GPU FAILURE DETECTED" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        # Log system state
        echo "lspci GPU check:" | tee -a "$LOG_FILE"
        lspci | grep -i nvidia | tee -a "$LOG_FILE"

        echo "nvidia modules:" | tee -a "$LOG_FILE"
        lsmod | grep nvidia | tee -a "$LOG_FILE"

        echo "Recent NVRM errors:" | tee -a "$LOG_FILE"
        sudo dmesg -T | grep -i 'nvrm\|xid\|msi-x' | tail -20 | tee -a "$LOG_FILE"

        echo "" | tee -a "$LOG_FILE"
        echo "ACTION REQUIRED: GPU has fallen off the bus. Reboot required." | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        # Optionally send alert (uncomment and configure)
        # echo "GPU DOWN on $(hostname) - Reboot required" | mail -s "GPU Alert" admin@example.com

        # Stop monitoring after first failure
        exit 1
    fi

    sleep "$CHECK_INTERVAL"
done
