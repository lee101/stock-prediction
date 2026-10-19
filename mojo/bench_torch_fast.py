import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
from timestral import TimestralForecaster
import torch_fast as tf
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:2048]
X, act, _ = build(closes, idx, 512)
Z = np.diff(np.log(np.concatenate([X[:, :1], X], 1)), axis=1).astype(np.float32)
Zt = torch.tensor(Z, device="cuda")
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
def timeit(f, n=5):
    f(); f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return r, (time.time() - t) / n
def official(sa):
    return torch.cat([fc.forecast(Zt[i:i+256], prediction_length=1, symmetric_averaging=sa, clip_nonnegative=False).quantiles[:, 0, 0] for i in range(0, len(Zt), 256)])
for sa in (False, True):
    ref, t0 = timeit(lambda: official(sa))
    print(f"official sa={sa}: {t0/len(Zt)*1e6:.0f} us/win", flush=True)
    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        W = tf.prep_weights(fc.model, dt)
        for B in (256, 512, 1024):
            def run():
                with torch.no_grad():
                    return torch.cat([tf.forecast(W, Zt[i:i+B], sa) for i in range(0, len(Zt), B)])
            out, t = timeit(run)
            print(f"  eager {name} B={B}: {t/len(Zt)*1e6:.0f} us/win  ({t0/t:.2f}x)  med err {((out[:,10]-ref[:,10])**2).mean().sqrt().item():.2e}", flush=True)
    W = tf.prep_weights(fc.model, torch.float16)
    cf = torch.compile(lambda z: tf.forecast(W, z, sa), mode="max-autotune-no-cudagraphs")
    B = 512
    def runc():
        with torch.no_grad():
            return torch.cat([cf(Zt[i:i+B]) for i in range(0, len(Zt), B)])
    try:
        out, t = timeit(runc, 3)
        print(f"  compiled fp16 B={B}: {t/len(Zt)*1e6:.0f} us/win ({t0/t:.2f}x) med err {((out[:,10]-ref[:,10])**2).mean().sqrt().item():.2e}", flush=True)
    except Exception as e:
        print("compile failed", repr(e)[:200])
