"""Emulate W8A8 (per-out-channel weight scale, per-token dynamic activation scale) in torch_fast and measure output error."""
import sys, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
from timestral import TimestralForecaster
import torch_fast as tf
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:1024]
X, act, _ = build(closes, idx, 512)
Z = torch.tensor(np.diff(np.log(np.concatenate([X[:, :1], X], 1)), axis=1).astype(np.float32), device="cuda")
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
W = tf.prep_weights(fc.model, torch.float32)
with torch.no_grad():
    ref = tf.forecast(W, Z[:512], False)
orig_lin = tf.lin
def make_lin(a_bits, w_bits, skip=()):
    qa, qw = 2 ** (a_bits - 1) - 1, 2 ** (w_bits - 1) - 1
    cache = {}
    def lin(x, w):
        if w.shape[0] in skip: return orig_lin(x, w)
        key = id(w)
        if key not in cache:
            s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / qw
            cache[key] = ((w / s).round().clamp(-qw, qw), s)
        wq, ws = cache[key]
        sx = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qa
        xq = (x / sx).round().clamp(-qa, qa)
        return (xq @ wq.t()) * sx * ws.t()
    return lin
for name, (ab, wb) in {"w8a8": (8, 8), "w8a16(fake)": (16, 8), "w4a8": (8, 4), "w8a8 skip-small": (8, 8)}.items():
    tf.lin = make_lin(ab, wb)
    with torch.no_grad():
        out = tf.forecast(W, Z[:512], False)
    d = out[:, 10] - ref[:, 10]
    print(name, "median err rms", d.pow(2).mean().sqrt().item(), "signal std", ref[:, 10].std().item(), "corr", np.corrcoef(out[:, 10].cpu(), ref[:, 10].cpu())[0, 1])
tf.lin = orig_lin
