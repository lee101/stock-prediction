import glob
import numpy as np
import pandas as pd

NV = 100
MAXJUMP = 0.4
CMAX = 1024


def load_closes(min_rows=1000, min_end="2025-11-25"):
    out = {}
    for f in sorted(glob.glob("trainingdata/*.csv")):
        try:
            d = pd.read_csv(f, usecols=["timestamp", "close"])
        except Exception:
            continue
        if len(d) < min_rows:
            continue
        d = d.dropna().sort_values("timestamp")
        if d.timestamp.iloc[-1][:10] < min_end:
            continue
        c = d.close.to_numpy(np.float64)
        if (c <= 0).any():
            continue
        out[f.split("/")[-1][:-4]] = c
    return out


def split_symbols(closes, stride=5):
    names = sorted(closes)
    test = names[::stride]
    return [n for n in names if n not in set(test)], test


def window_index(closes, names, lo, hi):
    """(sym, t) targets with lo < len-t <= hi; excludes windows with a jump>MAXJUMP in the lookback."""
    idx = []
    for n in names:
        c = closes[n]
        r = np.abs(c[1:] / c[:-1] - 1)
        for t in range(len(c) - hi, len(c) - lo):
            if t < 130:
                continue
            if r[max(0, t - CMAX):t].max() > MAXJUMP:
                continue
            idx.append((n, t))
    return idx


def build(closes, idx, ctx):
    X = np.full((len(idx), ctx), np.nan)
    act = np.zeros(len(idx)); sym = []
    for i, (n, t) in enumerate(idx):
        c = closes[n]; seg = c[max(0, t - ctx):t]
        X[i, ctx - len(seg):] = seg
        act[i] = c[t] / c[t - 1] - 1; sym.append(n)
    return X, act, np.array(sym)


def to_rep(X, rep):
    if rep == "price":
        return X
    lp = np.log(X)
    return lp if rep == "logprice" else np.diff(lp, axis=1)


def from_rep(f, X, rep):
    last = X[:, -1]
    if rep == "price":
        return f / last[:, None] - 1 if f.ndim == 2 else f / last - 1
    if rep == "logprice":
        return np.exp(f - np.log(last)[:, None]) - 1 if f.ndim == 2 else np.exp(f - np.log(last)) - 1
    return np.exp(f) - 1


def metrics(act, pred):
    ok = np.isfinite(pred); a, p = act[ok], pred[ok]
    s = np.sign(p) * a
    return dict(
        n=int(ok.sum()), mae=float(np.abs(a - p).mean()),
        mae_vs_zero=float(np.abs(a - p).mean() / np.abs(a).mean()),
        dir_acc=float((np.sign(a) == np.sign(p)).mean()),
        ic=float(np.corrcoef(a, p)[0, 1]) if p.std() > 0 else float("nan"),
        ls_sharpe_ann=float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else float("nan"),
    )
