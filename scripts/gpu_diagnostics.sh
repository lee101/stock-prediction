#!/bin/bash
# GPU Diagnostics - Run this when GPU fails to capture state

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="gpu_failure_${TIMESTAMP}.log"

echo "Capturing GPU failure diagnostics to $REPORT_FILE"

{
    echo "=========================================="
    echo "GPU FAILURE DIAGNOSTICS"
    echo "Timestamp: $(date)"
    echo "Hostname: $(hostname)"
    echo "=========================================="
    echo ""

    echo "=== NVIDIA SMI ==="
    nvidia-smi 2>&1 || echo "nvidia-smi failed"
    echo ""

    echo "=== LSPCI GPU ==="
    lspci | grep -i nvidia
    echo ""

    echo "=== NVIDIA MODULES ==="
    lsmod | grep nvidia
    echo ""

    echo "=== /dev/nvidia* ==="
    ls -l /dev/nvidia* 2>&1
    echo ""

    echo "=== DRIVER VERSION ==="
    cat /proc/driver/nvidia/version 2>&1 || echo "No driver loaded"
    echo ""

    echo "=== KERNEL VERSION ==="
    uname -r
    echo ""

    echo "=== DMESG NVIDIA ERRORS (last 50) ==="
    sudo dmesg -T | egrep -i 'nvrm|nvidia|xid|pcie|aer' | tail -50
    echo ""

    echo "=== JOURNALCTL NVIDIA ERRORS (last 50) ==="
    sudo journalctl -k -b | egrep -i 'nvrm|nvidia|xid|pcie|aer' | tail -50
    echo ""

    echo "=== PCIe LINK STATUS (if GPU visible) ==="
    sudo lspci -vvv -s 82:00.0 2>&1 | grep -E "LnkCap|LnkSta|LnkCtl|MSI-X|IntCtl" || echo "GPU not found in lspci"
    echo ""

    echo "=========================================="
    echo "DIAGNOSTICS COMPLETE"
    echo "Next step: Reboot the system to recover GPU"
    echo "=========================================="

} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"
cat "$REPORT_FILE"
