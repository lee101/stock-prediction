import sys, time
import numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common import *

closes = load_closes()
train, test = split_symbols(closes)
ctxs, act, sym = windows(closes, test)
print(f"symbols train={len(train)} test={len(test)} windows={len(act)}", flush=True)
last = np.array([c[-1] for c in ctxs])
res = {"actual": act}

from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
t0 = time.time(); out = []
B = 64
for i in range(0, len(ctxs), B):
    q, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32)[None] for c in ctxs[i:i+B]], prediction_length=1, quantile_levels=[0.5])
    out += [float(x[0, 0, 0]) for x in q]
res["chronos2"] = np.array(out) / last - 1
print("chronos2 time", time.time() - t0, flush=True)
del pipe; torch.cuda.empty_cache()

from timesfm3 import TimesFM3Forecaster
m = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda", per_core_batch_size=64)
t0 = time.time()
outs = list(m.predict_batch(contexts=[c.astype(np.float32) for c in ctxs], horizon=1))
res["timesfm3"] = np.array([float(o.forecast[0]) for o in outs]) / last - 1
print("timesfm3 time", time.time() - t0, flush=True)

np.savez("experiments/timesfm3_vs_chronos2/zeroshot_preds.npz", sym=sym, **res)
for k, v in res.items():
    if k != "actual":
        print(k, {a: round(b, 5) for a, b in metrics(act, v).items()})
print("zero", {a: round(b, 5) for a, b in metrics(act, np.zeros_like(act)).items()})
