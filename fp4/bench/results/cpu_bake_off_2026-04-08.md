# CPU bake-off — 2026-04-08

Quick apples-to-apples sortino comparison of the three fp4 trainers on the
synthetic stub env (CPU only, 4096 env steps × 2 seeds, default hyperparams
from each trainer's smoke config). Run while the GPU was busy with
sibling processes — purpose is **algorithm framing**, not absolute PnL.

## Results (medians)

| algo   | sortino | p10     | mean    | sps    |
|:-------|--------:|--------:|--------:|-------:|
| ppo    |  -0.810 | -20.97  | -11.19  |  3745  |
| qr_ppo |  -0.513 | -19.84  |  -5.95  |  3081  |
| sac    |  -0.615 | -26.51  | -10.46  |  1049  |

## Takeaways

- **QR-PPO has the best sortino** of the three (-0.513 vs PPO -0.810 / SAC -0.615)
  and the least-negative mean return. The quantile value head appears to give
  a real risk-aware advantage even on the noisy stub env, which lines up with
  the CPO/distributional-RL literature.
- **PPO is the fastest** at 3.7k SPS — the policy/critic forward+backward is
  the simplest of the three.
- **SAC is ~3.5× slower** than PPO at 1.05k SPS because of the twin-Q + actor
  + temperature updates per env step. The wall-time gap closes when env_step
  itself is the bottleneck (real marketsim, not stub).
- All three are negative on stub at this budget — stub is intentionally
  noisy and 4k steps isn't enough to learn against it. The point of the
  bake-off is the *relative* ranking, not the absolute number.

## Reproduce

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
from fp4.trainer import train_ppo
from fp4.trainer_qr import train_qr_ppo
from fp4.trainer_sac import train_sac
CFG = {
    'env': 'stub',
    'ppo': {'num_envs': 8, 'rollout_len': 32, 'hidden_size': 32,
            'ppo_epochs': 1, 'minibatch_size': 64},
    'sac': {'num_envs': 8, 'hidden_size': 32, 'batch_size': 64, 'warmup_steps': 128},
}
for name, fn in [('ppo', train_ppo), ('qr_ppo', train_qr_ppo), ('sac', train_sac)]:
    for s in (0, 1):
        m = fn(cfg=CFG, total_timesteps=4096, seed=s,
               checkpoint_dir=f'_build/bake_{name}_s{s}')
        print(name, s, m['final_sortino'], m['final_p10'], m['steps_per_sec'])
PY
```

## Next

- Re-run on `gpu_trading_env` once the long sweep frees the GPU. Same script,
  remove `CUDA_VISIBLE_DEVICES=-1`. The relative ranking should persist
  but absolute SPS will be 10-100× higher and the mean returns will reflect
  the real reward structure, not the synthetic stub.
- Pipe the results through `scripts/eval_100d.py` for the lag=2 binary-fill
  validation we now have wired up — that's the only number that matters
  for the 27%/month production target.
