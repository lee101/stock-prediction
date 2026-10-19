"""End-to-end: official Timestral-209M vs the Mojo path on real held-out windows (logret input), same metrics."""
import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
from common2 import *
from timestral import TimestralForecaster
import mojo_timestral
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 768
closes = load_closes(); _, test = split_symbols(closes)
rows = []
for n in test:
    c = closes[n]
    for t in range(len(c) - 100, len(c)):
        if t >= ctx + 1 and np.abs(c[1:t] / c[:t - 1] - 1)[max(0, t - 1024):].max() <= MAXJUMP:
            rows.append((n, t))
Z = np.stack([np.diff(np.log(closes[n][t - ctx - 1:t])) for n, t in rows]).astype(np.float32)
act = np.array([closes[n][t] / closes[n][t - 1] - 1 for n, t in rows])
print("windows", len(rows), "ctx", ctx, flush=True)
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
for sa in (False, True):
    t0 = time.time()
    ref = torch.cat([fc.forecast(torch.tensor(Z[i:i + 128], device="cuda"), prediction_length=1, symmetric_averaging=sa, clip_nonnegative=False).quantiles[:, 0, 0] for i in range(0, len(Z), 128)]).cpu().numpy()
    t_off = time.time() - t0
    t0 = time.time(); mj = mojo_timestral.forecast(Z, sa); t_mj = time.time() - t0
    for name, q in (("official", ref), ("mojo", mj)):
        r = np.exp(q[:, 10]) - 1
        m = metrics(act, r)
        print(f"sa={sa} {name:9s} mae_vs_zero {m['mae_vs_zero']:.5f} dir {m['dir_acc']:.4f} ic {m['ic']:.4f}  wall {t_off if name=='official' else t_mj:.2f}s", flush=True)
    d = np.abs(np.exp(mj[:, 10]) - np.exp(ref[:, 10]))
    print(f"   median-return diff: max {d.max():.2e} mean {d.mean():.2e}; corr {np.corrcoef(mj[:,10], ref[:,10])[0,1]:.6f}; sign agree {(np.sign(mj[:,10])==np.sign(ref[:,10])).mean():.4f}")
