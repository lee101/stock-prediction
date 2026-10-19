import sys, time
import numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common import load_closes, split_symbols
from timestral import TimestralForecaster

closes = load_closes(); _, test = split_symbols(closes)
CTX = 512
X = []
for n in test:
    c = closes[n]
    for t in range(len(c) - 100, len(c)):
        X.append(c[t - CTX:t])
X = np.array(X, dtype=np.float32)[:2048]


def bench(m, B, sa=True, autocast=None, reps=1):
    out = []
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        out = []
        for i in range(0, len(X), B):
            with torch.autocast("cuda", dtype=autocast, enabled=autocast is not None):
                f = m.forecast(X[i:i + B], prediction_length=1, symmetric_averaging=sa)
            out.append(f.median[:, 0, 0].float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(out), (time.time() - t0) / reps


base = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda")
ref, t = bench(base, 128); print("baseline ph=32 B=128", round(t, 2), flush=True)
fast = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
for B in (128, 256, 512):
    o, t = bench(fast, B); print(f"ph=1 B={B}", round(t, 2), "maxrel", float(np.abs(o / ref - 1).max()), flush=True)
for dt in (torch.bfloat16, torch.float16):
    o, t = bench(fast, 256, autocast=dt); print(f"ph=1 {dt} B=256", round(t, 2), "maxrel", float(np.abs(o / ref - 1).max()), "mean|d|/ret", float(np.abs(o - ref).mean() / np.abs(ref / X[:, -1] - 1 + 1e-9).mean() if False else np.abs(o - ref).mean()), flush=True)
o, t = bench(fast, 256, sa=False); print("ph=1 no-symavg B=256", round(t, 2), flush=True)
