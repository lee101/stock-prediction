import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common2 import *
from timestral import TimestralForecaster
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:2048]
X, act, _ = build(closes, idx, 512); X = X.astype(np.float32)
ref_in = np.log(X)
def run(fc, B=256, reps=3, **kw):
    for _ in range(1):
        for i in range(0, len(X), B): fc.forecast(ref_in[i:i+B], prediction_length=1, symmetric_averaging=False)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        out = []
        for i in range(0, len(X), B):
            out.append(fc.forecast(ref_in[i:i+B], prediction_length=1, symmetric_averaging=False).quantiles[:, 0, 0].float().cpu().numpy())
    torch.cuda.synchronize(); return np.concatenate(out), (time.time() - t0) / reps
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
ref, t = run(fc); print("fp32 ieee", round(t, 3), "s / 2048 windows =", round(t / 2048 * 1e6), "us/window", flush=True)
torch.set_float32_matmul_precision("high")
o, t = run(fc); print("tf32", round(t, 3), "maxabs q diff", float(np.abs(o - ref).max()), "mean", float(np.abs(o - ref).mean()), "scale", float(np.abs(ref).mean()), flush=True)
torch.set_float32_matmul_precision("highest")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
    fc.forecast(ref_in[:256], prediction_length=1, symmetric_averaging=False); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60))
