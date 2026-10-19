"""Dump Timestral-209M weights (prepped for the univariate one-step forward) and a test batch to mojo/data/."""
import sys, json, os, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
import torch_fast as tf
from timestral import TimestralForecaster

out = "mojo/data"; os.makedirs(out, exist_ok=True)
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu", per_shot_horizon_patches=1)
W = tf.prep_weights(fc.model, torch.float32, device="cpu")
man = {}
def put(name, t, dtypes=("f32",)):
    t = t.contiguous()
    for d in dtypes:
        if d == "f32": a = t.float().numpy()
        elif d == "f16": a = t.half().numpy()
        elif d == "bf16": a = t.bfloat16().view(torch.int16).numpy()
        a.tofile(f"{out}/{name}.{d}");
    man[name] = list(t.shape)
for k in ("fq", "e1b", "e2b", "erb", "on", "cos", "sin", "qw"): put(k, W[k])
def pad(w, n=32): return torch.cat([w, torch.zeros(n - w.shape[0], w.shape[1])], 0)
for k in ("e1", "e2", "er", "h1"): put(k, W[k], ("f16", "bf16"))
put("h2", pad(W["h2"]), ("f16", "bf16")); put("hr", pad(W["hr"]), ("f16", "bf16"))
for i, ly in enumerate(W["layers"]):
    put(f"l{i}_n0", ly["n0"]); put(f"l{i}_nm", ly["nm"])
    # fused [q;k;v;gate] projection for the time attention, rows (c,h,d)
    put(f"l{i}_qkvg", torch.cat([ly["qkv0"], ly["g0"]], 0), ("f16", "bf16"))
    put(f"l{i}_o0", ly["o0"], ("f16", "bf16"))
    put(f"l{i}_w12", ly["w12"], ("f16", "bf16")); put(f"l{i}_w3", ly["w3"], ("f16", "bf16"))
    if ly["variate"]:
        put(f"l{i}_n1", ly["n1"])
        put(f"l{i}_vg", torch.cat([ly["v1"], ly["g1"]], 0), ("f16", "bf16"))
        put(f"l{i}_o1", ly["o1"], ("f16", "bf16"))
json.dump(dict(shapes=man, variate=[l["variate"] for l in W["layers"]]), open(f"{out}/manifest.json", "w"))
print("tensors", len(man))

# test batch + fp32 oracle outputs
sys.path.insert(0, "experiments/timesfm3_vs_chronos2")
from common2 import *
closes = load_closes(); _, test = split_symbols(closes)
idx = window_index(closes, test, 0, 100)[:1024]
X, act, _ = build(closes, idx, 512)
Z = np.diff(np.log(np.concatenate([X[:, :1], X], 1)), axis=1).astype(np.float32)
Z.tofile(f"{out}/test_in.f32")
Wc = tf.prep_weights(fc.model, torch.float32, device="cuda"); Zt = torch.tensor(Z, device="cuda")
with torch.no_grad():
    for sa in (0, 1):
        q = torch.cat([tf.forecast(Wc, Zt[i:i+256], bool(sa)) for i in range(0, len(Zt), 256)]).cpu().numpy()
        q.astype(np.float32).tofile(f"{out}/test_ref_sa{sa}.f32")
print("test batch", Z.shape)
