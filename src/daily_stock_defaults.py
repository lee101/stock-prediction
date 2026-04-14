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

# 2026-04-14 v4: Upgraded to 13-model ensemble (added D_s57 + I_s3 doubled + D_s64 + D_s27).
# D_s57: tp=0.05 Muon, individual OOS neg=29/100, med=5.83%, sort=10.19 — bear-resistant
# I_s3 (doubled): gives I_s3 2x weight in softmax averaging — equivalent to weighted ensemble
# D_s64: tp=0.05 Muon, individual OOS neg=33/100, med=3.03%, sort=6.56 — diverse, high sort in ensemble
# D_s27: tp=0.05 Muon, individual OOS neg=38/100, med=1.55%, sort=6.56 — further diversity
# 13-model vs 9-model (exhaustive 263 windows):
#   13m: med=19.02%, p10=+8.11%, neg=10/263, sortino=33.31
#   12m: med=18.87%, p10=+7.88%, neg=10/263, sortino=32.35
#   9m:  med=17.48%, p10=+5.14%, neg=17/263, sortino=30.19
# Net vs 9m: med+1.54%, p10+2.97%, neg-7 (41% fewer losses), sortino+3.12
# Bear windows (indices 249-260, Apr 2026 tariff crash): 6/8 neg (vs 8/8 for 9m)
# 100-window sampled: med=19.08%, p10=5.34%, neg=6/100, sort=34.30
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s42, D_s3, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3(2x), D_s64, D_s27
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
    "pufferlib_market/prod_ensemble_screened32/D_s64.pt",  # 2026-04-14 v3: med+1.39% p10+2.74% sort+2.94
    "pufferlib_market/prod_ensemble_screened32/D_s27.pt",  # 2026-04-14 v4: p10+0.23% sort+0.96 (263w)
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
