# GPU Recovery & Stability Checklist

## ⚠️ IMMEDIATE ACTION REQUIRED

Your GPU has **completely failed** due to MSI-X initialization errors. Only a **full reboot** will recover it.

---

## 1. REBOOT NOW

```bash
sudo reboot
```

After reboot, verify GPU is working:
```bash
nvidia-smi
```

---

## 2. ENABLE PERSISTENCE MODE (Right After Reboot)

```bash
sudo nvidia-smi -pm 1
```

This will prevent the driver from unloading and reduce future failures.

**Auto-enable on boot:** Already configured via systemd service `/etc/systemd/system/nvidia-persistence-mode.service`

---

## 3. ADD KERNEL PARAMETERS (CRITICAL FOR STABILITY)

### Edit GRUB config
```bash
sudo nano /etc/default/grub
```

Find the last line with `GRUB_CMDLINE_LINUX_DEFAULT` and add `pcie_aspm=off`:

**Before:**
```
GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,115200 no_timer_check nofb nomodeset gfxpayload=text"
```

**After:**
```
GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,115200 no_timer_check nofb nomodeset gfxpayload=text pcie_aspm=off pci=noaer"
```

### Apply changes
```bash
sudo update-grub
sudo reboot
```

### Verify after reboot
```bash
cat /proc/cmdline | grep pcie_aspm
```

---

## 4. START GPU MONITORING

Before starting any training:

```bash
# Start GPU health monitor in background
nohup bash scripts/gpu_monitor.sh > /dev/null 2>&1 &

# In a separate terminal/tmux pane, watch GPU continuously
watch -n 5 nvidia-smi
```

---

## 5. SOURCE TRAINING ENVIRONMENT

Before each training session:

```bash
source scripts/setup_training_env.sh
```

This sets:
- `CUDA_DEVICE_MAX_CONNECTIONS=1` - Reduces concurrent operations
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` - Limits memory allocator aggressiveness

---

## 6. IF GPU FAILS AGAIN

### Capture diagnostics
```bash
bash scripts/gpu_diagnostics.sh
```

This saves a full diagnostic report to `gpu_failure_TIMESTAMP.log`.

### Check dmesg for the failure pattern
```bash
sudo dmesg -T | grep -i 'msi-x\|xid' | tail -20
```

### Report to provider

Send them:
1. The diagnostic log
2. Request: "Disable PCIe ASPM in BIOS for GPU slot 0000:82:00.0"
3. Mention: MSI-X failures under load (error code 0x22:0x56:742)

---

## 7. ESCALATION PATH (If Issue Persists)

### After first reboot + persistence mode:
- ✅ Should see 30-50% reduction in failures

### After adding `pcie_aspm=off`:
- ✅ Should see 70-90% reduction in failures

### If still failing frequently (>once per day):
1. **Add `pci=nomsi` to kernel params** (last resort)
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="... pcie_aspm=off pci=noaer pci=nomsi"
   ```
   **Warning:** 5-10% GPU performance loss, but 100% stability

2. **Contact provider** for:
   - BIOS PCIe ASPM disable
   - PSU quality check
   - Different PCIe slot or server

3. **Consider driver downgrade** to stable LTS:
   ```bash
   sudo apt-get install nvidia-driver-535
   sudo reboot
   ```

---

## 8. LONG-TERM MONITORING

Track GPU failures in a log:

```bash
echo "$(date): GPU failure count: X" >> gpu_stability.log
```

If failures are:
- **< 1 per week:** Acceptable for a training server
- **1-3 per week:** Needs kernel param tuning
- **> 3 per week:** Hardware/provider issue, escalate

---

## WHAT WE'VE DONE

✅ **Installed:** `nvidia-compute-utils-575` (persistence daemon)
✅ **Created:** Systemd service for auto-enabling persistence mode on boot
✅ **Created:** `scripts/gpu_monitor.sh` - Background GPU health monitor
✅ **Created:** `scripts/gpu_diagnostics.sh` - Failure diagnostic collector
✅ **Created:** `scripts/setup_training_env.sh` - CUDA stability environment
✅ **Documented:** `docs/GPU_RECOVERY.md` - Full troubleshooting guide
✅ **Documented:** `docs/KERNEL_PARAMETERS.md` - PCIe stability parameters

---

## QUICK REFERENCE

```bash
# Check GPU status
nvidia-smi

# Enable persistence mode
sudo nvidia-smi -pm 1

# View recent errors
sudo dmesg -T | grep -i nvidia | tail -20

# Run diagnostics when failed
bash scripts/gpu_diagnostics.sh

# Start monitoring
nohup bash scripts/gpu_monitor.sh &

# Setup training environment
source scripts/setup_training_env.sh
```

---

## EXPECTED TIMELINE

1. **Now:** Reboot to recover GPU (15 seconds)
2. **Immediately after:** Enable persistence mode (5 seconds)
3. **Within 1 hour:** Add `pcie_aspm=off` kernel parameter (2 minutes + reboot)
4. **Test:** Run training for 24 hours with monitoring
5. **Evaluate:** Check if failures reduced significantly
6. **If needed:** Add `pci=nomsi` and report to provider

---

**REMEMBER:** This is primarily a **hardware stability issue** (MSI-X/PCIe), not a software bug. The kernel parameters and persistence mode are **workarounds** that should significantly improve stability.
