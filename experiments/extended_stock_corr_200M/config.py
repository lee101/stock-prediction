"""Extended stock training config: 30+ stocks, shorts, 200M steps, correlation-aware."""

# Full stock universe with direction constraints
LONG_ONLY_STOCKS = [
    "AAPL", "AMD", "AMZN", "GOOG", "GOOGL", "META", "MSFT", "NET", "NVDA", "TSLA",
    "NFLX", "V", "JPM", "WMT",
]

SHORT_ONLY_STOCKS = [
    "ANGI", "BKNG", "EBAY", "EXPE", "KIND", "MTCH", "NWSA", "NYT", "TRIP", "Z",
    "YELP",
]

CRYPTO_SYMBOLS = [
    "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "UNIUSD",
]

ALL_SYMBOLS = LONG_ONLY_STOCKS + SHORT_ONLY_STOCKS + CRYPTO_SYMBOLS

# Training config
TRAINING_CONFIG = {
    "total_timesteps": 200_000_000,
    "num_envs": 64,
    "rollout_len": 512,
    "max_steps": 720,  # 30 days

    # Architecture (best from 100M_robust)
    "hidden_size": 512,
    "num_blocks": 3,
    "architecture": "resmlp",

    # Reward shaping (best from 100M_robust)
    "downside_penalty": 1.5,
    "smoothness_penalty": 0.5,
    "drawdown_penalty": 0.05,
    "trade_penalty": 0.001,
    "cash_penalty": 0.005,

    # Robustness
    "obs_noise_std": 0.01,

    # PPO params
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "ppo_epochs": 4,
    "minibatch_size": 4096,

    # Train/val split
    "train_split": 0.8,

    # Shorting enabled for stocks
    "allow_short": True,
}
