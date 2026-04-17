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

# 2026-04-17 v7: Dropped D_s81 (12-model). LOO v6 showed D_s81 was net-negative
# (Δmed −0.05%, Δp10 +0.46%, Δneg −1, Δsortino +0.19) — strict p10/neg/sortino win
# with trivial med cost. Deploy gate confirmed: 1× med +7.47%, p10 +3.18%,
# neg 10/263, sortino 6.74, max_dd 5.71%. 1.5×: med +10.31%, p10 +4.09%, neg 13.
# Prod runs at 1× leverage, so v7 strictly dominates v6 in the deployed cell.
# v6→v7 rationale: fewer members = sharper softmax-avg posterior, helping break
# flat-ties in tariff-crash OOS regime (3 consecutive flat days Apr 14-16 motivated
# LOO re-scan). Reliability > marginal mean: v7 has fewer negative windows at 1×.
# 2026-04-17 v6 (prior): Swapped D_s3 → AD_s4. LOO v5 showed D_s3 was free-drop.
# AD_s4 trained on aprcrash augmented data (through 2026-02-28) — gives ensemble
# fresh exposure to Mar-Apr 2026 tariff crash dynamics without contaminating OOS val.
# Models: C_s7 (AdamW, tp=0.02) + D_s16, D_s42, AD_s4, I_s3, D_s2, D_s14, D_s28, D_s57, I_s3(2x), D_s64, I_s32
# All models: disable_shorts=True, 65 actions (masked shorts), features_per_sym=16
# Trained on data through 2025-05-31, val 2025-06-01 to 2025-11-30
DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble_screened32/C_s7.pt"

DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble_screened32/D_s16.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s42.pt",  # D_s13→D_s42: neg 22→15/263, p10→+2.72%
    "pufferlib_market/prod_ensemble_screened32/AD_s4.pt",  # 2026-04-17 v6: swap D_s3→AD_s4, +0.63% med, +0.45 sort, neg same
    "pufferlib_market/prod_ensemble_screened32/I_s3.pt",   # D_s5→I_s3 swap: med+2.36% sort+3.02 (100-win)
    "pufferlib_market/prod_ensemble_screened32/D_s2.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s14.pt",
    "pufferlib_market/prod_ensemble_screened32/D_s28.pt",
    # 2026-04-17 v7: D_s81 removed — LOO net-negative (Δp10 +0.46%, Δneg −1, Δsortino +0.19)
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
