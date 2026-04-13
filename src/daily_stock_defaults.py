DEFAULT_SYMBOLS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "PLTR",
    "JPM",
    "V",
    "AMZN",
    "AMD",
    "NFLX",
    "COIN",
    "CRWD",
    "UBER",
)

# 2026-04-13: Switched from 32-model stocks12 legacy ensemble → 2-model stocks17 RSI ensemble.
# Old 32-model had feature mismatch (pre-RSI features) and showed ~0.04%/month in 120d replay.
# New 2-model: C_s31 (AdamW) + D_s29 u200 (Muon) — cross-variant diversity.
# Eval: lag=2, binary fills, fee=10bps, 60d×50 windows: med=18.84%, p10=6.25%, 0/50 neg, Sortino=48.81.
# Slippage stress PASSED at 0/5/10/20bps (all 0/50 neg).
# Feature schema: rsi_v5 (RSI(14) replaces duplicate trend_20d). 17 symbols, obs_dim=294.
DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/stocks17_sweep/C_low_tp/s31/val_best.pt"

# Old 32-model ensemble (stocks12, legacy features) — kept for reference, not used:
# Members: tp10+s15+s36+gamma_995+muon_wd_005+h1024_a40+s1731+gamma995_s2006+s1401+s1726+s1523+s2617+s2033+s2495+s1835+s2827+s2722+s3668+s3411+s4011+s4777+s4080+s4533+s4813+s5045+s5337+s5199+s5019+s6808+s3456+s7159+s6758
DEFAULT_EXTRA_CHECKPOINTS = (
    # D_s29 champion_u200: med=15.63%, p10=7.88%, 0/50 neg, Sortino=23.74 (Muon variant)
    "pufferlib_market/checkpoints/stocks17_sweep/D_muon/s29/champion_u200.pt",
)

DEFAULT_DATA_DIR = "trainingdata"
# 2-model stocks17 ensemble (17 symbols, 25 actions with disable_shorts).
# Uniform = 1/25 = 0.04. Observed confidence ~0.08-0.12 for typical signals.
# Gate set just above uniform; rejects only near-random signals.
DEFAULT_MIN_OPEN_CONFIDENCE = 0.05
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = 0.0
