import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
from timestral import TimestralForecaster
import torch_fast as tf
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:2048]
X, act, _ = build(closes, idx, 512)
Z = np.diff(np.log(X), axis=1)[:, -512:] if False else np.log(X)[:, 1:]
Z = np.concatenate([Z, Z[:, -1:]], 1)[:, :512].astype(np.float32) if False else np.diff(np.log(np.concatenate([X[:, :1], X], 1)), axis=1).astype(np.float32)
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
Zt = torch.tensor(Z, device="cuda")
def official(Zt, sa):
    out = []
    for i in range(0, len(Zt), 256):
        out.append(fc.forecast(Zt[i:i+256], prediction_length=1, symmetric_averaging=sa, clip_nonnegative=False).quantiles[:, 0, 0])
    return torch.cat(out)
def timeit(f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return r, (time.time() - t) / n
for sa in (False, True):
    ref, t_ref = timeit(lambda: official(Zt, sa))
    print(f"official sa={sa}: {t_ref*1000:.0f} ms / {len(Zt)} windows = {t_ref/len(Zt)*1e6:.0f} us/window", flush=True)
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        W = tf.prep_weights(fc.model, dt)
        def run():
            with torch.no_grad():
                return torch.cat([tf.forecast(W, Zt[i:i+512], sa) for i in range(0, len(Zt), 512)])
        out, t = timeit(run)
        d = (out - ref).abs()
        print(f"  mine {name}: {t*1000:.0f} ms ({t/len(Zt)*1e6:.0f} us/win, {t_ref/t:.2f}x) maxabs {d.max().item():.3e} med-err rms {((out[:,10]-ref[:,10])**2).mean().sqrt().item():.3e} med std {ref[:,10].std().item():.3e}", flush=True)
