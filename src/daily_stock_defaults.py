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

# 2026-04-17 v6: Swapped D_s3 → AD_s4 (13-model). LOO showed D_s3 was a free-drop
# (Δmed +0.14%, Δneg 0, Δsortino +0.17, Δmax_dd −0.68%). AD_s4 (sweep AD/s4, Muon
# trained on aprcrash data) had best individual standalone (med 8.11%, neg 12) of
# the A* batch and was the only candidate with positive Δmed as 14th member.
# 13-model v6 263w deploy gate (fb=5, lev=1): med +7.52%, p10 +2.72%, neg 11/263,
# sortino 6.55, max_dd 5.71%. vs v5 baseline: +0.63% med, +0.38% p10, +0.45 sort,
# −0.57% max_dd, neg unchanged. 1.5× cell preserved (med +10.33%, sortino 6.18).
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s81, D_s57, I_s3(2x), D_s64, I_s32
# All models: disable_shorts=True, 65 actions (masked shorts), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30
# AD_s4 trained on aprcrash augmented data (through 2026-02-28) — gives ensemble
# fresh exposure to Mar-Apr 2026 tariff crash dynamics without contaminating OOS val.
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s42.pt",  # D_s13→D_s42: neg 22→15/263, p10→+2.72%
    "pufferlib_market/prod_ensemble_screened32/AD_s4.pt",  # 2026-04-17 v6: swap D_s3→AD_s4, +0.63% med, +0.45 sort, neg same
    "pufferlib_market/prod_ensemble_screened32/I_s3.pt",   # D_s5→I_s3 swap: med+2.36% sort+3.02 (100-win)
    "pufferlib_market/prod_ensemble_screened32/D_s2.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s14.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s28.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s81.pt",  # 2026-04-14: best 9th model (p10+0.39%, same neg)
    "pufferlib_market/prod_ensemble_screened32/D_s57.pt",  # 2026-04-14 v2: fixes bear windows (neg 17→12/263)
    "pufferlib_market/prod_ensemble_screened32/I_s3.pt",   # 2026-04-14 v2: I_s3 2x weight (bear-resistant)
    "pufferlib_market/prod_ensemble_screened32/D_s64.pt",  # 2026-04-14 v3: med+1.39% p10+2.74% sort+2.94
    "pufferlib_market/prod_ensemble_screened32/I_s32.pt",  # 2026-04-14 v5: swap D_s27→I_s32, neg 10→8/263
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
