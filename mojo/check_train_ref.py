import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
import train_ref as tr
from timestral import TimestralForecaster
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
P = tr.build_params(fc.model)
closes = load_closes(); train, _ = split_symbols(closes)
rng = np.random.default_rng(0)
Z, Y, V = tr.make_batch({n: closes[n] for n in train}, rng, 64, 512)
for k in P: P[k].requires_grad_(True)
loss, q = tr.forward_loss(P, Z, Y, V, torch.float32)
loss.backward()
print("fp32 loss", loss.item(), "q shape", tuple(q.shape))
g32 = {k: v.grad.clone() for k, v in P.items()}
for v in P.values(): v.grad = None
loss16, _ = tr.forward_loss(P, Z, Y, V, torch.bfloat16); loss16.backward()
print("bf16 loss", loss16.item())
worst = []
for k, g in g32.items():
    gb = P[k].grad
    rel = ((gb - g).norm() / (g.norm() + 1e-12)).item()
    worst.append((rel, k, g.norm().item()))
worst.sort(reverse=True)
print("bf16-vs-fp32 grad rel err (worst 8):", [(round(r, 4), k) for r, k, _ in worst[:8]])
print("median rel err", np.median([w[0] for w in worst]))
