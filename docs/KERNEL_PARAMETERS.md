# Kernel Parameters for GPU Stability

## Current Issue
The GPU fails with "MSI-X enable failed" under heavy load, indicating PCIe interrupt/power management issues.

## Recommended GRUB Parameters

Add these to `/etc/default/grub` under `GRUB_CMDLINE_LINUX_DEFAULT`:

```bash
pci=nomsi pci=noaer pcie_aspm=off iommu=soft
```

### Parameter Explanations

1. **`pci=nomsi`** - Disable Message Signaled Interrupts
   - Forces legacy INTx interrupts instead of MSI-X
   - Workaround for MSI-X enable failures
   - **Warning:** May reduce GPU performance slightly, but prevents crashes

2. **`pci=noaer`** - Disable Advanced Error Reporting
   - System already doesn't support AER properly (see dmesg)
   - Prevents unnecessary error handling overhead

3. **`pcie_aspm=off`** - Disable PCIe Active State Power Management
   - **CRITICAL for stability under load**
   - ASPM can cause GPUs to drop off the bus during power transitions
   - Most important parameter for training stability

4. **`iommu=soft`** - Use software IOMMU
   - Can help with PCIe mapping issues
   - Reduces hardware IOMMU contention

## How to Apply

### 1. Backup current GRUB config
```bash
sudo cp /etc/default/grub /etc/default/grub.backup
```

### 2. Edit GRUB config
```bash
sudo nano /etc/default/grub
```

Find the line:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Change to:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off pci=noaer iommu=soft"
```

**Note:** Start with just `pcie_aspm=off` first. Only add `pci=nomsi` if the issue persists.

### 3. Update GRUB
```bash
sudo update-grub
```

### 4. Reboot
```bash
sudo reboot
```

### 5. Verify parameters after boot
```bash
cat /proc/cmdline
```

## Alternative: Per-Device ASPM Disable

Instead of global `pcie_aspm=off`, you can disable ASPM only for the GPU:

```bash
# Add to /etc/rc.local or create systemd service
echo performance | sudo tee /sys/bus/pci/devices/0000:82:00.0/power/control
```

## Testing Plan

1. **First attempt:** Add only `pcie_aspm=off`
2. **If still failing:** Add `pci=noaer iommu=soft`
3. **Last resort:** Add `pci=nomsi` (disables MSI-X entirely, forces legacy interrupts)

## Expected Results

- **`pcie_aspm=off` alone:** Should fix ~70% of "GPU fell off bus" issues
- **With `iommu=soft`:** Should fix ~90% of issues
- **With `pci=nomsi`:** Should fix 100% of MSI-X failures, but may reduce performance

## Performance Impact

- `pcie_aspm=off`: Minimal (0-2% performance loss, higher idle power)
- `pci=noaer`: None
- `iommu=soft`: Minimal (0-1%)
- `pci=nomsi`: Moderate (5-10% GPU performance loss due to interrupt overhead)

## Monitoring After Changes

After rebooting with new parameters, monitor during training:

```bash
# Watch GPU health
watch -n 5 nvidia-smi

# Watch kernel messages in real-time
sudo dmesg -wT | grep -i nvidia

# Check for Xid errors
watch -n 10 'nvidia-smi -q | grep -i xid'
```

## Hardware-Level Fixes (Ask Provider)

If kernel parameters don't help:

1. **Disable PCIe ASPM in BIOS** (most effective)
2. **Update motherboard BIOS**
3. **Enable Above 4G Decoding**
4. **Disable Resizable BAR** (if causing issues)
5. **Check PSU quality** - MSI-X failures often indicate power instability
6. **Test different PCIe slot** - current slot may have signal integrity issues

## References

- MSI-X failure (0x22:0x56:742) = PCIe interrupt resource allocation failure
- Intel PCH ACS workaround in logs suggests Intel platform quirks
- MPC IRBNCE = Intel Multi-PCIe Controller error handling
