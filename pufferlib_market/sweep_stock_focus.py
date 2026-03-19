"""
Targeted stock RL sweep aligned with the stricter 30-day market validator.

This now focuses on conservative fine-tuning around the stock checkpoints
that have held up best under realistic replay. The first broad resumed sweep
showed that high-entropy updates from the current stock leader quickly destroy
the low-turnover behavior that keeps 30-day replay close to flat.
"""

from __future__ import annotations

STOCK_BASELINES = {
    "featlag": {
        "resume_from": "pufferlib_market/checkpoints/stocks13_featlag1_fee5bps_longonly_run4/best.pt",
        "train_data": "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915_featlag1.bin",
    },
    "issuedat_featlag": {
        "resume_from": "pufferlib_market/checkpoints/stocks13_issuedat_featlag1_fee5bps_longonly_run5/best.pt",
        "train_data": "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915_issuedat_featlag1.bin",
    },
    "recent_generic": {
        "resume_from": "pufferlib_market/checkpoints/stocks13_recent_longonly_run2/best.pt",
        "train_data": "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915.bin",
    },
}


def stock_experiment(
    description: str,
    *,
    baseline: str,
    **overrides: object,
) -> dict[str, object]:
    source = STOCK_BASELINES[baseline]
    config: dict[str, object] = {
        "description": description,
        "arch": "resmlp",
        "hidden_size": 256,
        "disable_shorts": True,
        "lr": 1e-4,
        "ent_coef": 0.001,
        "resume_from": source["resume_from"],
        "train_data": source["train_data"],
    }
    config.update(overrides)
    return config


STOCK_EXPERIMENTS = [
    stock_experiment(
        "featlag_ft_lr1e4_ent001_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr1e4_tradepen_0005_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        trade_penalty=0.0005,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr1e4_tradepen_0010_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        trade_penalty=0.001,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr1e4_wd001_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        weight_decay=0.001,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr1e4_gamma995_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        gamma=0.995,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr1e4_cosine_t240",
        baseline="featlag",
        lr=1e-4,
        ent_coef=0.001,
        time_budget=240,
        lr_schedule="cosine",
        lr_warmup_frac=0.02,
        lr_min_ratio=0.2,
    ),
    stock_experiment(
        "featlag_ft_lr5e5_ent001_t240",
        baseline="featlag",
        lr=5e-5,
        ent_coef=0.001,
        time_budget=240,
    ),
    stock_experiment(
        "featlag_ft_lr2e5_ent0005_t120",
        baseline="featlag",
        lr=2e-5,
        ent_coef=0.0005,
        time_budget=120,
    ),
    stock_experiment(
        "featlag_ft_lr1e5_ent0005_t120",
        baseline="featlag",
        lr=1e-5,
        ent_coef=0.0005,
        time_budget=120,
    ),
    stock_experiment(
        "issuedat_ft_lr2e5_ent0005_t120",
        baseline="issuedat_featlag",
        lr=2e-5,
        ent_coef=0.0005,
        time_budget=120,
    ),
    stock_experiment(
        "recent_generic_ft_lr2e5_ent0005_t120",
        baseline="recent_generic",
        lr=2e-5,
        ent_coef=0.0005,
        time_budget=120,
    ),
]


def build_default_argv(script_name: str) -> list[str]:
    return [
        script_name,
        "--train-data", "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_start20250915_featlag1.bin",
        "--val-data", "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_20260214_cov019.bin",
        "--holdout-data", "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_20260214_cov019.bin",
        "--time-budget", "300",
        "--max-trials", str(len(STOCK_EXPERIMENTS)),
        "--leaderboard", "pufferlib_market/autoresearch_stocks13_focus.csv",
        "--checkpoint-root", "pufferlib_market/checkpoints/autoresearch_stocks13_focus",
        "--holdout-n-windows", "8",
        "--holdout-eval-steps", "720",
        "--holdout-end-within-steps", "1440",
        "--holdout-fee-rate", "0.0005",
        "--holdout-max-leverage", "2.0",
        "--rank-metric", "auto",
        "--market-validation-asset-class", "stock",
        "--market-validation-days", "30",
        "--market-validation-symbols", "NVDA,PLTR,META,MSFT,NET",
        "--fee-rate-override", "0.0005",
        "--max-leverage-override", "2.0",
        "--disable-shorts-override",
    ]


if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import pufferlib_market.autoresearch_rl as ar

    ar.EXPERIMENTS = STOCK_EXPERIMENTS
    sys.argv = build_default_argv(sys.argv[0]) + sys.argv[1:]
    ar.main()
