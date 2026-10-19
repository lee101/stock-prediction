import sys, time, json, itertools, pickle
import numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common2 import *

closes = load_closes(); _, test = split_symbols(closes)
CTXS = [128, 256, 384, 512, 768, 1024]
REPS = ["price", "logprice", "logret"]
sets = {"val": window_index(closes, test, NV, 2 * NV), "test": window_index(closes, test, 0, NV)}
data = {(s, c): build(closes, idx, c) for s, idx in sets.items() for c in CTXS}
print({s: len(i) for s, i in sets.items()}, flush=True)
B = 256
store = {}   # (model, rep, ctx, sa, split) -> (N,Q) return quantiles ; Q=21 for timestral, 1 otherwise


def log(model, rep, ctx, sa, split, q, act, dt):
    store[(model, rep, ctx, sa, split)] = q
    mid = q[:, q.shape[1] // 2]
    m = metrics(act, mid); print(json.dumps(dict(model=model, rep=rep, ctx=ctx, sa=sa, split=split, sec=round(dt, 1), **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})), flush=True)


def runall(model, name, fn, sas=(None,)):
    for sa, rep, ctx, split in itertools.product(sas, REPS, CTXS, sets):
        X, act, _ = data[(split, ctx)]
        Z = to_rep(X, rep); t0 = time.time(); out = []
        for i in range(0, len(Z), B):
            out.append(fn(Z[i:i + B], sa, rep))
        log(name, rep, ctx, sa, split, from_rep(np.concatenate(out), X, rep), act, time.time() - t0)
    pickle.dump(store, open("experiments/timesfm3_vs_chronos2/sweep_zs.pkl", "wb"))


from timestral import TimestralForecaster
m = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
def ts(Z, sa, rep):
    f = m.forecast(Z, prediction_length=1, symmetric_averaging=sa, clip_nonnegative=(rep == "price"))
    return f.quantiles[:, 0, 0, :].float().cpu().numpy()
runall(m, "timestral-209M", ts, sas=(True, False))
del m; torch.cuda.empty_cache()

from timesfm3 import TimesFM3Forecaster
tm = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda", per_core_batch_size=64)
def tf(Z, sa, rep):
    o = list(tm.predict_batch(contexts=[z[~np.isnan(z)].astype(np.float32) for z in Z], horizon=1))
    return np.array([[float(x.forecast[0])] for x in o])
runall(tm, "timesfm3", tf)
del tm; torch.cuda.empty_cache()

from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
def ch(Z, sa, rep):
    q, _ = pipe.predict_quantiles([torch.tensor(z[~np.isnan(z)], dtype=torch.float32)[None] for z in Z], prediction_length=1, quantile_levels=[0.5])
    return np.array([[float(x[0, 0, 0])] for x in q])
runall(pipe, "chronos2", ch)
for split in sets:
    act = data[(split, CTXS[0])][1]
    store[("zero", "-", 0, None, split)] = np.zeros((len(act), 1))
pickle.dump({"store": store, "act": {s: data[(s, CTXS[0])][1] for s in sets}, "sym": {s: data[(s, CTXS[0])][2] for s in sets}}, open("experiments/timesfm3_vs_chronos2/sweep_zs.pkl", "wb"))
print("DONE")
