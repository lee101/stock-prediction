DEFAULT_SYMBOLS = (
    # Screened32: 32 curated stocks with strong learnability (trend correlation, Sharpe)
    # Healthcare / Pharma (defensive, tariff-resilient)
    "LLY", "BSX", "ABBV", "VRTX", "SYK", "WELL",
    # Finance (cyclical but trendy)
    "JPM", "GS", "V", "MA", "AXP", "MS",
    # Technology (core winners)
    "AAPL", "MSFT", "NVDA", "KLAC", "CRWD", "META",
    # Consumer / Retail (steady compounders)
    "COST", "AZO", "TJX",
    # Industrial / Defense (steady bull)
    "CAT", "PH", "RTX",
    # Travel / Hotels (recovery plays)
    "BKNG", "MAR", "HLT",
    # Broad market + tech
    "PLTR", "SPY", "QQQ", "AMZN", "GOOG",
)

# 2026-04-13: Upgraded to 7-model screened32 ensemble (+D_s14).
# 7-model OOS (Jun 2025-Apr 2026, all 263 windows, lag=2, fill_bps=5, fee=10bps):
#   med=12.62%, p10=-1.30%, neg=31/263, sortino=20.98
# vs 6-model baseline: med=12.27%, p10=-1.24%, neg=33/263, sortino=22.93
# 100-window sample: 7-model med=13.08%, p10=0.72%, neg=8/100, sort=19.88
#                    6-model med=13.73%, p10=0.80%, neg=9/100, sort=27.39
# D_s14 adds marginal neg improvement (33→31 on full OOS), mixed on sortino.
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s13, D_s3, D_s5, D_s2, D_s14 (Muon, tp=0.05)
# All models: disable_shorts=True, 65 actions (masked shorts), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s13.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s3.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s5.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s2.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s14.pt",  # +D_s14: neg→4/263, p10→+3.38%
)

DEFAULT_DATA_DIR = "trainingdata"
# 5-model screened32 ensemble (32 symbols, 33 actions with disable_shorts).
# Uniform = 1/33 ≈ 0.030. Gate at 0.05 = ~1.67x uniform; rejects near-random signals.
DEFAULT_MIN_OPEN_CONFIDENCE = 0.05
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = 0.0
