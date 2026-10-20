"""Export fp32 master params, a test batch + fp32 oracle gradients/loss, and real train/val windows for the Mojo trainer."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
import train_ref as tr
from timestral import TimestralForecaster

out = "mojo/data/train"; os.makedirs(out, exist_ok=True)
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
P = tr.build_params(fc.model)
for k, v in P.items():
    v.detach().cpu().numpy().astype(np.float32).tofile(f"{out}/{k}.f32")
json.dump({k: list(v.shape) for k, v in P.items()}, open(f"{out}/manifest.json", "w"))
closes = load_closes(); train, test = split_symbols(closes)
# oracle batch
rng = np.random.default_rng(0)
Z, Y, V = tr.make_batch({n: closes[n] for n in train}, rng, 64, CTX)
for name, t in (("tb_Z", Z), ("tb_Y", Y), ("tb_V", V)): t.cpu().numpy().astype(np.float32).tofile(f"{out}/{name}_{CTX}.f32")
for k in P: P[k].requires_grad_(True)
loss, q = tr.forward_loss(P, Z, Y, V, torch.float32); loss.backward()
np.array([loss.item()], np.float32).tofile(f"{out}/ref_loss_{CTX}.f32")
q.detach().cpu().numpy().astype(np.float32).tofile(f"{out}/ref_q_{CTX}.f32")
os.makedirs(f"{out}/grads_{CTX}", exist_ok=True)
for k, v in P.items(): v.grad.cpu().numpy().astype(np.float32).tofile(f"{out}/grads_{CTX}/{k}.f32")
print("oracle loss", loss.item())
# real training windows (train symbols) and validation windows (held-out symbols, tuning period)
def windows(names, lo, hi, n, seed):
    r = np.random.default_rng(seed); pool = []
    for nm in names:
        c = closes[nm]
        for t in range(max(CTX + 2, len(c) - hi), len(c) - lo - 1):
            pool.append((nm, t))
    sel = r.permutation(len(pool))[:n]
    Zs = np.zeros((len(sel), CTX), np.float32); Ys = np.zeros((len(sel), 32), np.float32); Vs = np.zeros((len(sel), 32), np.float32)
    for i, j in enumerate(sel):
        nm, t = pool[j]; c = closes[nm]
        Zs[i] = np.diff(np.log(c[t - CTX - 1:t]))
        fut = np.diff(np.log(c[t - 1:min(t + 32, len(c))])); Ys[i, :len(fut)] = fut; Vs[i, :len(fut)] = 1
    return Zs, Ys, Vs
# train windows: targets end before the validation period (positions from the end > 2*NV+32)
for name, (names, lo, hi, n) in {"train": (train, 232, 10**9, 32768), "val": (test, 100 + 32, 200, 4096)}.items():
    Zs, Ys, Vs = windows(names, lo, hi, n, 1)
    Zs.tofile(f"{out}/{name}_Z_{CTX}.f32"); Ys.tofile(f"{out}/{name}_Y_{CTX}.f32"); Vs.tofile(f"{out}/{name}_V_{CTX}.f32")
    print(name, Zs.shape, "valid frac", Vs.mean())
