"""PyTorch fine-tune step baseline (same data, schedule, clipping, AdamW as mojo/src/train.mojo)."""
import sys, time, math, numpy as np, torch
sys.path.insert(0, "mojo")
import train_ref as tr
from timestral import TimestralForecaster
mode, B, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lr_peak = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-5
eval_every = int(sys.argv[5]) if len(sys.argv) > 5 else 0
CTX = 512
d = "mojo/data/train"
Zt = torch.tensor(np.fromfile(f"{d}/train_Z_{CTX}.f32", dtype=np.float32).reshape(-1, CTX), device="cuda")
Yt = torch.tensor(np.fromfile(f"{d}/train_Y_{CTX}.f32", dtype=np.float32).reshape(-1, 32), device="cuda")
Vt = torch.tensor(np.fromfile(f"{d}/train_V_{CTX}.f32", dtype=np.float32).reshape(-1, 32), device="cuda")
Zv = torch.tensor(np.fromfile(f"{d}/val_Z_{CTX}.f32", dtype=np.float32).reshape(-1, CTX), device="cuda")
Yv = torch.tensor(np.fromfile(f"{d}/val_Y_{CTX}.f32", dtype=np.float32).reshape(-1, 32), device="cuda")
Vv = torch.tensor(np.fromfile(f"{d}/val_V_{CTX}.f32", dtype=np.float32).reshape(-1, 32), device="cuda")
idx = torch.tensor(np.fromfile(f"{d}/train_idx.i32", dtype=np.int32).astype(np.int64), device="cuda")
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
P = tr.build_params(fc.model)
for v in P.values(): v.requires_grad_(True)
del fc
cdt = torch.bfloat16
lossf = lambda P_, z, y, v: tr.forward_loss(P_, z, y, v, cdt, fast=True)
if mode == "compile":
    lossf = torch.compile(lossf, mode="max-autotune-no-cudagraphs")
decay = [p for p in P.values() if p.dim() == 2]; nodecay = [p for p in P.values() if p.dim() == 1]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}], lr=lr_peak, betas=(0.9, 0.95), eps=1e-8, fused=True)
params = list(P.values())
def evaluate():
    tot = 0.0; wsum = 0.0; mae = 0.0; mae0 = 0.0
    with torch.no_grad():
        for i in range(0, (len(Zv) // B) * B, B):
            loss, q = tr.forward_loss(P, Zv[i:i+B], Yv[i:i+B], Vv[i:i+B], cdt, fast=True)
            w = torch.full((32,), 0.5 / 31, device="cuda"); w[0] = 1.0
            ws = (w[None] * Vv[i:i+B]).sum().item()
            tot += loss.item() * ws; wsum += ws
            m0 = q[:, 0, 10]; mae += (Yv[i:i+B, 0] - m0).abs().sum().item(); mae0 += Yv[i:i+B, 0].abs().sum().item()
    return tot / wsum, mae / mae0
times = []
for step in range(1, steps + 1):
    lr = lr_peak * step / 100 if step < 100 else lr_peak * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(steps - 100, 1)))
    for g in opt.param_groups: g["lr"] = lr
    ids = idx[(step - 1) * B:(step - 1) * B + B] if (step * B) <= len(idx) else idx[torch.arange((step - 1) * B, step * B, device="cuda") % len(idx)]
    torch.cuda.synchronize(); t0 = time.time()
    loss, _ = lossf(P, Zt[ids], Yt[ids], Vt[ids])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    torch.cuda.synchronize(); dt = time.time() - t0
    if step > 3: times.append(dt)
    if eval_every and (step % eval_every == 0 or step == 1):
        vl, vm = evaluate()
        print(f"step {step} lr {lr:.3g} train_loss {loss.item():.5f} gnorm {gn.item():.4f} | val loss {vl:.5f} mae_vs_zero {vm:.5f} | ms/step {np.mean(times)*1e3 if times else 0:.1f}", flush=True)
print(f"RESULT torch-{mode} B={B}: min {np.min(times)*1e3:.1f} ms/step = {np.min(times)/B*1e6:.0f} us/sample; median {np.median(times)*1e3:.1f} ms/step = {np.median(times)/B*1e6:.0f} us/sample", flush=True)
