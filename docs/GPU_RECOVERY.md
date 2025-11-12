# GPU Recovery & Stability Guide

## Problem Summary

**Symptom:** `nvidia-smi` shows "No devices were found" after heavy training workloads.

**Root Cause:** `Failed to enable MSI-X` - The GPU's PCIe Message Signaled Interrupts fail under load, causing the driver to lose communication with the GPU. This is a **hardware/PCIe stability issue**, not a driver bug.

**GPU:** NVIDIA Device 10de:2b85 (RTX 50-series/Blackwell)
**Driver:** 575.51.02 (Open Kernel Module)
**Kernel:** 6.8.0-87-generic

---

## Diagnostic Log Findings

```
[Tue Nov 11 23:25:43 2025] NVRM: GPU 0000:82:00.0: Failed to enable MSI-X.
[Tue Nov 11 23:25:43 2025] NVRM: osInitNvMapping: *** Cannot attach gpu
[Tue Nov 11 23:25:43 2025] NVRM: RmInitAdapter failed! (0x22:0x56:742)
```

Repeated failures indicate the GPU "fell off the bus" under training load.

---

## Immediate Recovery (REQUIRES REBOOT)

Once the GPU is in this state, **only a full system reboot** will recover it. Hot PCIe reset doesn't work because the MSI-X initialization is broken at the hardware level.

```bash
sudo reboot
```

---

## Prevention Measures (ALREADY APPLIED)

### 1. ✅ NVIDIA Persistence Mode

**Installed:** `nvidia-compute-utils-575` (provides `nvidia-persistenced`)

After each reboot, enable persistence mode:

```bash
sudo nvidia-smi -pm 1
```

This prevents the driver from unloading between jobs and reduces the chance of MSI-X failures.

**Auto-enable on boot:** Add to `/etc/rc.local` or create a systemd service (see below).

---

## Robustness Scripts

### Auto-enable Persistence Mode on Boot

Create `/etc/systemd/system/nvidia-persistence-mode.service`:

```ini
[Unit]
Description=Enable NVIDIA Persistence Mode
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nvidia-persistence-mode.service
```

---

### GPU Health Monitor

Create `scripts/gpu_monitor.sh`:

```bash
#!/bin/bash
# Monitor GPU health and alert if it disappears

LOG_FILE="$HOME/gpu_monitor.log"

while true; do
    if ! nvidia-smi &>/dev/null; then
        echo "[$(date)] GPU FAILURE DETECTED - nvidia-smi failed" | tee -a "$LOG_FILE"
        # Send alert (email, Slack, etc.)
        echo "GPU DOWN - Manual reboot required" | mail -s "GPU Alert" admin@example.com
        # Log dmesg errors
        sudo dmesg -T | grep -i 'nvrm\|xid\|msi-x' | tail -50 >> "$LOG_FILE"
    fi
    sleep 60  # Check every minute
done
```

Run in background:
```bash
nohup bash scripts/gpu_monitor.sh &
```

---

### Training Stability Tweaks

Add to your training scripts or shell environment:

```bash
# Reduce aggressive CUDA memory bursts
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Monitor GPU during training
watch -n 5 nvidia-smi  # In separate terminal
```

---

## BIOS/Hardware Fixes (Provider Side)

Contact your server provider and request:

1. **Disable PCIe ASPM** (Active State Power Management) for slot 82:00.0
2. **Enable Above 4G Decoding** if not already enabled
3. **Check Resizable BAR** settings
4. **Verify PSU stability** - MSI-X failures can be caused by power spikes
5. **Check PCIe slot/riser quality** - ensure clean electrical connection
6. **Verify cooling** - thermal throttling can trigger bus errors

Note from logs:
```
acpi PNP0A08:00: _OSC: platform does not support [AER LTR DPC]
```
This system lacks Advanced Error Reporting (AER), which makes debugging harder.

---

## What Doesn't Work

❌ **Hot PCIe Reset:** `echo 1 > /sys/bus/pci/devices/0000:82:00.0/remove && echo 1 > /sys/bus/pci/rescan`
   - GPU disappears completely after removal

❌ **Module Reload:** `rmmod nvidia && modprobe nvidia`
   - MSI-X is still broken after reload

❌ **nvidia-persistenced restart**
   - Can't start when GPU is already failed

---

## Diagnostic Commands

When the GPU fails again, run these BEFORE rebooting:

```bash
# Capture error logs
sudo dmesg -T | egrep -i 'nvrm|nvidia|xid|pcie|aer' > gpu_failure_$(date +%Y%m%d_%H%M%S).log

# Check PCIe link status
sudo lspci -vvv -s 82:00.0 | grep -E "LnkCap|LnkSta|MSI-X"

# Check module state
lsmod | grep nvidia
ls -l /dev/nvidia*

# Capture full system state
nvidia-smi -q > nvidia_state.log 2>&1 || echo "nvidia-smi failed"
```

---

## Long-term Solution

If this issue persists frequently:

1. **Contact provider** about PCIe stability and power quality
2. **Consider driver downgrade** to a more stable version (e.g., 535 LTS)
3. **Cap GPU power** to reduce electrical stress:
   ```bash
   sudo nvidia-smi -pl 300  # Cap at 300W (adjust for your card)
   ```
4. **Upgrade to a different server** with better PCIe/power infrastructure

---

## References

- NVIDIA Error Code 0x22:0x56:742 = RM_ERR_INSUFFICIENT_RESOURCES / osInitNvMapping failure
- MSI-X failure typically indicates: PCIe signal integrity, BIOS configuration, or hardware fault
- PCIe ACS workaround in logs suggests Intel PCH quirks
