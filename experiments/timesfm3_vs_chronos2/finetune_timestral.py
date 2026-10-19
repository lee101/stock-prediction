import sys, time, json, argparse, math, copy
import numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common2 import *
from timestral import TimestralForecaster

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="209M")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--aux", type=float, default=0.5)
ap.add_argument("--freeze", type=int, default=0, help="freeze first N transformer blocks")
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--tag", default="ft")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

closes = load_closes(); train, test = split_symbols(closes)
H = 32
cand = []
for n in train:
    c = closes[n]; r = np.abs(c[1:] / c[:-1] - 1)
    cr = np.concatenate([[0], np.cumsum(r > MAXJUMP)])
    for t in range(130, len(c) - 2 * NV):
        lo = max(0, t - CMAX)
        if cr[min(t + H, len(c) - 1)] - cr[lo] == 0:
            cand.append((n, t))
print("train windows", len(cand), "symbols", len(train), flush=True)
sets = {"val": window_index(closes, test, NV, 2 * NV), "test": window_index(closes, test, 0, NV)}
ev = {s: build(closes, idx, a.ctx) for s, idx in sets.items()}

fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{a.size}", device="cuda", per_shot_horizon_patches=1)
model = fc.model
levels = torch.tensor(fc.quantile_levels, device="cuda")
for p in model.parameters():
    p.requires_grad_(True)
if a.freeze:
    for blk in list(model.transformer.layers)[:a.freeze]:
        for p in blk.parameters():
            p.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
print("trainable", sum(p.numel() for p in params) / 1e6, "M", flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
warm = 100
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))


def batch():
    ids = rng.integers(0, len(cand), a.bs)
    X = np.zeros((a.bs, a.ctx), np.float32); Y = np.full((a.bs, H), np.nan, np.float32); L = np.zeros(a.bs, np.float32)
    for i, j in enumerate(ids):
        n, t = cand[j]; c = closes[n]
        seg = c[max(0, t - a.ctx):t]
        X[i, :] = np.nan; X[i, a.ctx - len(seg):] = seg
        fut = c[t:min(t + H, len(c) - 2 * NV + 0)]
        L[i] = c[t - 1]; Y[i, :len(fut)] = fut / c[t - 1] - 1
    return X, Y, L


def evaluate(split, sa):
    X, act, _ = ev[split]; out = []
    for i in range(0, len(X), 256):
        f = fc.forecast(X[i:i + 256], prediction_length=1, symmetric_averaging=sa, clip_nonnegative=True)
        q = f.quantiles[:, 0, 0, :].float().cpu().numpy()
        out.append(q / X[i:i + 256, -1:] - 1)
    q = np.concatenate(out)
    return metrics(act, q[:, q.shape[1] // 2]), q


best, best_state, hist = 1e9, None, []
t0 = time.time()
for step in range(a.steps + 1):
    if step % a.eval_every == 0:
        model.eval()
        m, _ = evaluate("val", False)
        hist.append(dict(step=step, **m)); print(f"step {step} val mae_vs_zero {m['mae_vs_zero']:.5f} ic {m['ic']:.4f} dir {m['dir_acc']:.4f} t={time.time()-t0:.0f}s", flush=True)
        if m["mae_vs_zero"] < best:
            best = m["mae_vs_zero"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_step = step
    if step == a.steps:
        break
    model.train()
    X, Y, L = batch()
    ctx = torch.tensor(X, device="cuda"); obs = torch.isfinite(ctx); ctx = torch.where(obs, ctx, torch.zeros_like(ctx))
    ctx_f = torch.tensor(L, device="cuda")
    gid = torch.arange(a.bs, device="cuda")
    # ffill leading NaN-free: contexts are full length, so mask is all True except left pad
    q = model.predict_block(ctx, obs, gid, 1)               # (B, 32, Q) raw price space
    qr = q / ctx_f[:, None, None] - 1
    y = torch.tensor(Y, device="cuda"); ok = torch.isfinite(y)
    e = torch.where(ok, y, torch.zeros_like(y))[:, :, None] - qr
    pin = torch.maximum(levels * e, (levels - 1) * e).mean(-1) / 0.02   # (B, 32)
    w = torch.full((H,), a.aux / (H - 1), device="cuda"); w[0] = 1.0
    loss = (pin * ok * w).sum() / (ok * w).sum()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)

model.load_state_dict(best_state); model.eval()
res = dict(args=vars(a), best_step=best_step, hist=hist)
for sa in (False, True):
    for s in ("val", "test"):
        m, q = evaluate(s, sa); res[f"{s}_sa{int(sa)}"] = m; print(s, sa, {k: round(v, 4) for k, v in m.items()}, flush=True)
        np.save(f"experiments/timesfm3_vs_chronos2/ftq_{a.tag}_{s}_sa{int(sa)}.npy", q)
json.dump(res, open(f"experiments/timesfm3_vs_chronos2/ft_{a.tag}.json", "w"))
torch.save(best_state, f"experiments/timesfm3_vs_chronos2/ft_{a.tag}.pt")
