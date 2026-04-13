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

# 2026-04-13: Upgraded to 8-model screened32 ensemble (D_s13→D_s42 swap).
# D_s42 individual OOS: neg=24/100, med=10.28% (2nd best D after D_s28=16).
# 8-model with D_s42 replacing D_s13 (OOS 100 windows):
#   med=15.81%, p10=+5.14%, neg=7/100, sortino=27.65
# 8-model exhaustive 263-window: med=15.28%, p10=+2.72%, neg=15/263, sortino=26.52
# vs previous 8-model (D_s13): med=14.23%, p10=+1.58%, neg=22/263, sortino=22.75
# Key improvements: neg 22→15/263, p10 +1.14%, med +1.05%, sort +3.77.
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s42, D_s3, D_s5, D_s2, D_s14, D_s28 (Muon, tp=0.05)
# All models: disable_shorts=True, 65 actions (masked shorts), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s42.pt",  # D_s13→D_s42: neg 22→15/263, p10→+2.72%
    "pufferlib_market/prod_ensemble_screened32/D_s3.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s5.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s2.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s14.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s28.pt",
)

DEFAULT_DATA_DIR = "trainingdata"
# 5-model screened32 ensemble (32 symbols, 33 actions with disable_shorts).
# Uniform = 1/33 ≈ 0.030. Gate at 0.05 = ~1.67x uniform; rejects near-random signals.
DEFAULT_MIN_OPEN_CONFIDENCE = 0.05
# Value estimate gate: RL critic V(s) is miscalibrated OOS — systematically negative
# even when ensemble OOS returns are strongly positive (15.81% med). Disable gate by
# setting to -1.0 (blocks only catastrophic predictions < -100%). Confidence gate is
# the real quality filter.
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = -1.0
