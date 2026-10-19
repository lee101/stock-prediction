import glob
import numpy as np
import pandas as pd

CTX = 512
NTEST = 100


def load_closes(min_rows=1000):
    out = {}
    for f in sorted(glob.glob("trainingdata/*.csv")):
        try:
            d = pd.read_csv(f, usecols=["timestamp", "close"])
        except Exception:
            continue
        if len(d) < min_rows:
            continue
        d = d.dropna().sort_values("timestamp")
        c = d.close.to_numpy(np.float64)
        if (c <= 0).any():
            continue
        out[f.split("/")[-1][:-4]] = c
    return out


def split_symbols(closes, stride=5):
    names = sorted(closes)
    test = names[::stride]
    train = [n for n in names if n not in set(test)]
    return train, test


def windows(closes, names, ntest=NTEST, ctx=CTX):
    ctxs, act, sym = [], [], []
    for n in names:
        c = closes[n]
        for t in range(len(c) - ntest, len(c)):
            ctxs.append(c[max(0, t - ctx):t])
            act.append(c[t] / c[t - 1] - 1)
            sym.append(n)
    return ctxs, np.array(act), np.array(sym)


def metrics(act, pred):
    ok = np.isfinite(pred)
    a, p = act[ok], pred[ok]
    s = np.sign(p) * a
    return dict(
        n=int(ok.sum()),
        mae=float(np.abs(a - p).mean()),
        mae_vs_zero=float(np.abs(a - p).mean() / np.abs(a).mean()),
        dir_acc=float((np.sign(a) == np.sign(p)).mean()),
        ic=float(np.corrcoef(a, p)[0, 1]) if p.std() > 0 else float("nan"),
        ls_mean_bps=float(s.mean() * 1e4),
        ls_sharpe_ann=float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else float("nan"),
    )
