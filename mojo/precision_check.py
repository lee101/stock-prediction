import sys, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common2 import *
from timestral import TimestralForecaster
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:2048]
X, act, _ = build(closes, idx, 512); X = X.astype(np.float32)
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
def run(autocast=None, rep="price"):
    Z = X if rep == "price" else np.diff(np.log(X), axis=1)
    out = []
    for i in range(0, len(Z), 256):
        with torch.autocast("cuda", dtype=autocast, enabled=autocast is not None):
            q = fc.forecast(Z[i:i+256], prediction_length=1, symmetric_averaging=False).quantiles[:, 0, 0].float().cpu().numpy()
        out.append(q)
    q = np.concatenate(out)
    r = q / X[:, -1:] - 1 if rep == "price" else np.exp(q) - 1
    return r
for rep in ("price", "logret"):
    ref = run(None, rep); med = ref[:, 10]
    print(rep, "fp32 median ret std", med.std(), "mae_vs_zero", np.abs(act - med).mean() / np.abs(act).mean())
    for dt in (torch.float16, torch.bfloat16):
        r = run(dt, rep)
        d = r - ref
        print(rep, dt, "ret err rms", np.sqrt((d**2).mean()), "median err rms", np.sqrt((d[:,10]**2).mean()), "corr w/ fp32 median", np.corrcoef(r[:,10], med)[0,1], "mae_vs_zero", np.abs(act - r[:,10]).mean() / np.abs(act).mean())
