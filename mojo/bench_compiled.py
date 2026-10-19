import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/timesfm3_vs_chronos2"); sys.path.insert(0, "mojo")
import torch_fast as tf
from timestral import TimestralForecaster
Z = torch.tensor(np.fromfile("mojo/data/test_in.f32", dtype=np.float32).reshape(1024, 512), device="cuda")
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
for dt in (torch.float16, torch.bfloat16):
    W = tf.prep_weights(fc.model, dt)
    for B in (256, 512, 1024):
        cf = torch.compile(lambda z: tf.forecast(W, z, False), mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            cf(Z[:B]); cf(Z[:B]); torch.cuda.synchronize(); t = time.time()
            for _ in range(10): cf(Z[:B])
            torch.cuda.synchronize()
        print(f"RESULT compiled {dt} B={B}: {(time.time()-t)/10/B*1e6:.1f} us/win", flush=True)
