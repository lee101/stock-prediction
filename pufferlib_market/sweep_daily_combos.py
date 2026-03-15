"""
Targeted daily RL sweep: combinations of best-performing hyperparameters.

Based on autoresearch results:
- trade_pen_05 (#1): +20.0% OOS, 100% profitable
- cosine_lr (#2): +5.18% OOS, 98% profitable
- fee_2x (#3): +3.31% OOS, 83% profitable
- ent_001 (#4): +1.17% OOS, 58% profitable

This script tests combinations of these winning ingredients.
"""

DAILY_EXPERIMENTS = [
    # ===== Trade penalty sweep =====
    {"description": "tp_03", "trade_penalty": 0.03},
    {"description": "tp_04", "trade_penalty": 0.04},
    {"description": "tp_06", "trade_penalty": 0.06},
    {"description": "tp_08", "trade_penalty": 0.08},
    {"description": "tp_10", "trade_penalty": 0.10},
    {"description": "tp_15", "trade_penalty": 0.15},
    {"description": "tp_20", "trade_penalty": 0.20},

    # ===== trade_pen + cosine_lr combos =====
    {"description": "tp05_cosine", "trade_penalty": 0.05, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "tp04_cosine", "trade_penalty": 0.04, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "tp06_cosine", "trade_penalty": 0.06, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "tp08_cosine", "trade_penalty": 0.08, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},
    {"description": "tp10_cosine", "trade_penalty": 0.10, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05},

    # ===== trade_pen + fee_2x combos =====
    {"description": "tp05_fee2x", "trade_penalty": 0.05, "fee_rate": 0.002},
    {"description": "tp08_fee2x", "trade_penalty": 0.08, "fee_rate": 0.002},

    # ===== trade_pen + low entropy combos =====
    {"description": "tp05_ent001", "trade_penalty": 0.05, "ent_coef": 0.01},
    {"description": "tp05_ent003", "trade_penalty": 0.05, "ent_coef": 0.03},

    # ===== trade_pen + cosine + fee_2x (triple combo) =====
    {"description": "tp05_cos_fee2x", "trade_penalty": 0.05, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "fee_rate": 0.002},
    {"description": "tp08_cos_fee2x", "trade_penalty": 0.08, "lr_schedule": "cosine",
     "lr_warmup_frac": 0.02, "lr_min_ratio": 0.05, "fee_rate": 0.002},

    # ===== trade_pen + weight decay =====
    {"description": "tp05_wd01", "trade_penalty": 0.05, "weight_decay": 0.01},
    {"description": "tp05_wd005", "trade_penalty": 0.05, "weight_decay": 0.005},

    # ===== trade_pen + longer episode =====
    {"description": "tp05_ep120", "trade_penalty": 0.05, "max_steps": 120},
    {"description": "tp05_ep60", "trade_penalty": 0.05, "max_steps": 60},

    # ===== Smaller models with trade_pen =====
    {"description": "tp05_h512", "trade_penalty": 0.05, "hidden_size": 512},
    {"description": "tp05_h256", "trade_penalty": 0.05, "hidden_size": 256},

    # ===== Seeds for reproducibility =====
    {"description": "tp05_s123", "trade_penalty": 0.05, "seed": 123},
    {"description": "tp05_s7", "trade_penalty": 0.05, "seed": 7},
    {"description": "tp05_s2024", "trade_penalty": 0.05, "seed": 2024},

    # ===== trade_pen + smoothness =====
    {"description": "tp05_smooth", "trade_penalty": 0.05, "smoothness_penalty": 0.01},

    # ===== Higher gamma with trade_pen =====
    {"description": "tp05_g995", "trade_penalty": 0.05, "gamma": 0.995},

    # ===== Longer training time (10 min instead of 5) =====
    # These need --time-budget override
]

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    # Ensure repo root is on path
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import pufferlib_market.autoresearch_rl as ar
    ar.EXPERIMENTS = DAILY_EXPERIMENTS

    # Run with daily settings
    sys.argv = [
        sys.argv[0],
        "--train-data", "pufferlib_market/data/crypto5_daily_train.bin",
        "--val-data", "pufferlib_market/data/crypto5_daily_val.bin",
        "--time-budget", "300",
        "--max-trials", "35",
        "--leaderboard", "pufferlib_market/autoresearch_daily_combos.csv",
        "--checkpoint-root", "pufferlib_market/checkpoints/autoresearch_daily_combos",
        "--periods-per-year", "365.0",
        "--max-steps-override", "90",
    ]
    ar.main()
