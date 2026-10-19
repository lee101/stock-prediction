import sys, time, json, itertools
import numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common import load_closes, split_symbols, metrics

NTEST, B, MAXJUMP = 100, 128, 0.4
closes = load_closes()
_, test = split_symbols(closes)
CTXS = [64, 128, 256, 512, 1024]
CMAX = max(CTXS)


def build(ctx):
    X, act, last, sym = [], [], [], []
    for n in test:
        c = closes[n]
        r = np.abs(c[1:] / c[:-1] - 1)
        for t in range(len(c) - NTEST, len(c)):
            if r[max(0, t - CMAX):t].max() > MAXJUMP:
                continue
            w = np.full(ctx, np.nan); seg = c[max(0, t - ctx):t]; w[ctx - len(seg):] = seg
            X.append(w); act.append(c[t] / c[t - 1] - 1); last.append(c[t - 1]); sym.append(n)
    return np.array(X), np.array(act), np.array(last), np.array(sym)


def to_rep(X, rep):
    if rep == "price":
        return X
    lp = np.log(X)
    return lp if rep == "logprice" else np.concatenate([np.full((len(X), 1), np.nan), np.diff(lp, axis=1)], 1)[:, 1:] if False else np.diff(lp, axis=1)


def from_rep(f, X, rep):
    last = X[:, -1]
    if rep == "price": return f / last - 1
    if rep == "logprice": return np.exp(f - np.log(last)) - 1
    return np.exp(f) - 1


def run(predict, rep, ctx, X):
    Z = to_rep(X, rep)
    out = []
    for i in range(0, len(Z), B):
        out.append(predict(Z[i:i + B]))
    return from_rep(np.concatenate(out), X, rep)


rows = []
def record(model, rep, ctx, sa, act, pred, dt):
    m = metrics(act, pred); m.update(model=model, rep=rep, ctx=ctx, sym_avg=sa, sec=round(dt, 1))
    rows.append(m); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
    json.dump(rows, open("experiments/timesfm3_vs_chronos2/grid_results.json", "w"))


data = {c: build(c) for c in CTXS}
act = data[CTXS[0]][1]
print("windows", len(act), flush=True)
record("zero", "-", 0, "-", act, np.zeros_like(act), 0)

from chronos import Chronos2Pipeline
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
def chr_pred(Z):
    q, _ = pipe.predict_quantiles([torch.tensor(np.nan_to_num(z[~np.isnan(z)]) if False else z[~np.isnan(z)], dtype=torch.float32)[None] for z in Z], prediction_length=1, quantile_levels=[0.5])
    return np.array([float(x[0, 0, 0]) for x in q])
for rep, ctx in itertools.product(["price", "logprice", "logret"], CTXS):
    X = data[ctx][0]; t0 = time.time()
    record("chronos2", rep, ctx, "-", act, run(chr_pred, rep, ctx, X), time.time() - t0)
del pipe; torch.cuda.empty_cache()

from timestral import TimestralForecaster
for size in ["5M", "28M", "123M", "209M"]:
    m = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device="cuda")
    for sa, rep, ctx in itertools.product([True, False], ["price", "logprice", "logret"], CTXS):
        X = data[ctx][0]; t0 = time.time()
        def pr(Z):
            f = m.forecast(Z, prediction_length=1, symmetric_averaging=sa, clip_nonnegative=(rep == "price"))
            return f.median[:, 0, 0].float().cpu().numpy()
        record(f"timestral-{size}", rep, ctx, sa, act, run(pr, rep, ctx, X), time.time() - t0)
    del m; torch.cuda.empty_cache()
