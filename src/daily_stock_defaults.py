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

# 2026-04-13: Switched to 5-model screened32 ensemble.
# Reasons: head-to-head vs stocks17 on same Jun2025-Apr2026 OOS period (313 actual trading days):
#   stocks17 RSI 2-model: med=7.09%, p10=-8.26%, 34/100 neg, Sortino=16.88
#   screened32 5-model:   med=12.39%, p10=+0.31%, 10/100 neg, Sortino=27.95  ← winner
# Screened32 includes defensive healthcare (LLY, ABBV, BSX) that outperforms in bear markets.
# Validated through full OOS period including March 2026 tariff crash (-26% SPY).
# Models: C_s7 (AdamW, tp=0.02) + D_s16 + D_s13 + D_s3 + D_s5 (Muon, tp=0.05)
# All 5 models: disable_shorts=True, 33 actions (flat + 32 longs), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30 (bull period)
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s13.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s3.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s5.pt",
)

DEFAULT_DATA_DIR = "trainingdata"
# 5-model screened32 ensemble (32 symbols, 33 actions with disable_shorts).
# Uniform = 1/33 ≈ 0.030. Gate at 0.05 = ~1.67x uniform; rejects near-random signals.
DEFAULT_MIN_OPEN_CONFIDENCE = 0.05
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = 0.0
