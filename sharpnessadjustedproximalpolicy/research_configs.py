"""Research-based experiment configs for SAPP round 5.

New techniques:
1. Spectral regularization: penalize spectral norm of weight matrices
2. Multi-period loss: train on multiple window lengths simultaneously
3. Gradient diversity: decorrelate gradients across minibatches
4. Return-conditioned augmentation: shift returns target during training
5. Aggressive regularization combos that worked in per-symbol sweep
"""

# -- Round 5a: Spectral regularization --
# Encourages smoother decision boundaries, reduces sensitivity to input noise.
# Implemented as a spectral_reg_weight penalty term on the largest singular value
# of each weight matrix.
SPECTRAL_CONFIGS = [
    {
        "name": "spectral_1e3",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "spectral_reg_weight": 1e-3,
    },
    {
        "name": "spectral_1e2",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "spectral_reg_weight": 1e-2,
    },
    {
        "name": "spectral_1e2_baseline",
        "sam_mode": "none", "weight_decay": 0.1,
        "spectral_reg_weight": 1e-2,
    },
]

# -- Round 5b: Multi-period loss --
# Instead of training on a single seq_len window, compute loss on overlapping
# sub-windows (short-term 24h + medium 72h + full seq_len). Forces the model
# to be profitable at multiple horizons, not just the training window.
MULTIPERIOD_CONFIGS = [
    {
        "name": "multiperiod_24_72",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "multi_period_windows": [24, 72],
        "multi_period_weight": 0.3,  # weight for sub-window losses
    },
    {
        "name": "multiperiod_12_36_72",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "multi_period_windows": [12, 36, 72],
        "multi_period_weight": 0.2,
    },
    {
        "name": "multiperiod_24_72_baseline",
        "sam_mode": "none", "weight_decay": 0.1,
        "multi_period_windows": [24, 72],
        "multi_period_weight": 0.3,
    },
]

# -- Round 5c: Aggressive regularization combos --
# Combine techniques that individually showed promise.
COMBO_CONFIGS = [
    # SAM + SWA + high wd (kitchen sink that might actually work)
    {
        "name": "sam_swa_wd015",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.15,
        "use_swa": True, "swa_start_frac": 0.4,
    },
    # SAM + spectral + multiperiod (full stack)
    {
        "name": "fullstack_spec_multi",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "spectral_reg_weight": 1e-3,
        "multi_period_windows": [24, 72],
        "multi_period_weight": 0.2,
    },
    # Barshift + SAM + SWA (data aug + flat minima)
    {
        "name": "barshift5_sam_swa",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "bar_shift_range": 5, "use_swa": True,
    },
    # Higher lr + SAM + cosine (explore more, flatten more)
    {
        "name": "lr5e4_sam_cosine_wd01",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "learning_rate": 5e-4, "lr_schedule": "cosine",
    },
    # Longer training + lower lr + SAM (more steps, gentler)
    {
        "name": "long40ep_lr1e4_sam",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "learning_rate": 1e-4, "epochs": 40,
    },
]

# -- Round 5d: Per-symbol champion tuning --
# Take the known winning configs and push further with focused tweaks.
CHAMPION_TUNE_CONFIGS = [
    # periodic_wd01 was the universal winner -- try variants
    {
        "name": "periodic_wd01_rho01",
        "sam_mode": "periodic", "rho": 0.1, "weight_decay": 0.1,
        "probe_every": 10,
    },
    {
        "name": "periodic_wd01_rho02",
        "sam_mode": "periodic", "rho": 0.2, "weight_decay": 0.1,
        "probe_every": 10,
    },
    {
        "name": "periodic_wd01_probe5",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "probe_every": 5,  # more frequent probing
    },
    {
        "name": "periodic_wd01_target05",
        "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1,
        "target_sharpness": 0.5,  # tighter tolerance
    },
    # Full SAM variants (2x compute but strongest signal)
    {
        "name": "fullsam_wd01",
        "sam_mode": "full_sam", "rho": 0.05, "weight_decay": 0.1,
    },
    {
        "name": "fullsam_wd01_adaptive",
        "sam_mode": "full_sam", "rho": 0.05, "weight_decay": 0.1,
        "adaptive_sam": True,
    },
]

# -- Round 5e: WARP (Weight Averaged Rewarded Policies) --
# Not a training config -- this is a post-hoc technique.
# Average state dicts from top-K checkpoints per symbol.
# Implemented in portfolio_eval.py --warp flag.

ALL_R5_CONFIGS = SPECTRAL_CONFIGS + MULTIPERIOD_CONFIGS + COMBO_CONFIGS + CHAMPION_TUNE_CONFIGS
