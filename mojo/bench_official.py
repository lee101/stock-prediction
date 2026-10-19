import sys, time, numpy as np, torch
from timestral import TimestralForecaster
fc = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cuda", per_shot_horizon_patches=1)
for ctx in (256, 512, 768, 1024):
    Z = torch.tensor(np.fromfile(f"mojo/data/test_in_{ctx}.f32", dtype=np.float32).reshape(1024, ctx), device="cuda")
    for sa in (False, True):
        def run():
            for i in range(0, 1024, 256): fc.forecast(Z[i:i+256], prediction_length=1, symmetric_averaging=sa, clip_nonnegative=False)
        run(); torch.cuda.synchronize(); t = time.time()
        for _ in range(3): run()
        torch.cuda.synchronize()
        print(f"RESULT official ctx={ctx} sa={int(sa)}: {(time.time()-t)/3/1024*1e6:.0f} us/win", flush=True)
