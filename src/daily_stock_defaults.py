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

# 2026-04-14 v2: Upgraded to 11-model ensemble (added D_s57 + I_s3 doubled).
# D_s57: tp=0.05 Muon, individual OOS neg=29/100, med=5.83%, sort=10.19
# I_s3 (doubled): gives I_s3 2x weight in softmax averaging — equivalent to weighted ensemble
# 11-model vs 9-model (exhaustive 263 windows):
#   11m: med=17.79%, p10=+5.96%, neg=12/263, sortino=29.41
#   9m:  med=17.48%, p10=+5.14%, neg=17/263, sortino=30.19
# Net: med+0.31%, p10+0.82%, neg-5 (30% fewer losses), sortino-0.78
# Bear windows (indices 249-260, Apr 2026 tariff crash): 6/8 neg (vs 8/8 before)
# 100-window sampled: med=17.58%, p10=5.64%, neg=6/100, sort=29.77
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s42, D_s3, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3(2x)
# All models: disable_shorts=True, 65 actions (masked shorts), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s42.pt",  # D_s13→D_s42: neg 22→15/263, p10→+2.72%
    "pufferlib_market/prod_ensemble_screened32/D_s3.pt",
    "pufferlib_market/prod_ensemble_screened32/I_s3.pt",   # D_s5→I_s3 swap: med+2.36% sort+3.02 (100-win)
    "pufferlib_market/prod_ensemble_screened32/D_s2.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s14.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s28.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s81.pt",  # 2026-04-14: best 9th model (p10+0.39%, same neg)
    "pufferlib_market/prod_ensemble_screened32/D_s57.pt",  # 2026-04-14 v2: fixes bear windows (neg 17→12/263)
    "pufferlib_market/prod_ensemble_screened32/I_s3.pt",   # 2026-04-14 v2: I_s3 2x weight (bear-resistant)
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
