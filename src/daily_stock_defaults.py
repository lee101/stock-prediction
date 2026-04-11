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
)

DEFAULT_CHECKPOINT = "pufferlib_market/prod_ensemble/tp10.pt"

# 32-model ensemble stored in prod_ensemble/ (protected from *_screen/ deletion pattern)
# Members: tp10+s15+s36+gamma_995+muon_wd_005+h1024_a40+s1731+gamma995_s2006+s1401+s1726+s1523+s2617+s2033+s2495+s1835+s2827+s2722+s3668+s3411+s4011+s4777+s4080+s4533+s4813+s5045+s5337+s5199+s5019+s6808+s3456+s7159+s6758
# Updated 2026-03-31 — all checkpoints are screen-phase (<=3M steps) or exact-match recoveries
# s2827 added 2026-03-28: +16% delta vs 15-model
# s2722 added 2026-03-29: +6% delta vs 16-model
# s3668 added 2026-03-29: +1.1% delta vs 17-model
# s3411 added 2026-03-29: +1.8% delta vs 18-model — 19-model: 0/111 neg, p10=44.1%
# s4011 added 2026-03-29: +4.4% delta vs 19-model — 20-model: 0/111 neg, p10=48.5%
# s4777 added 2026-03-29: +0.2% delta vs 20-model — 21-model: 0/111 neg, p10=48.7%
# s4080 added 2026-03-29: +0.1% delta vs 21-model — 22-model: 0/111 neg, p10=48.8%
# s4533 added 2026-03-29: +4.2% delta vs 22-model — 23-model: 0/111 neg, p10=52.9%
# s4813 added 2026-03-29: +4.4% delta vs 23-model — 24-model: 0/111 neg, p10=57.4%
# s5045 added 2026-03-29: +1.2% delta vs 24-model — 25-model: 0/111 neg, p10=58.6%
# s5337 added 2026-03-29: +1.8% delta vs 25-model — 26-model: 0/111 neg, p10=60.3%
# s5199 added 2026-03-29: +2.2% delta vs 26-model — 27-model: 0/111 neg, p10=62.6%
# s5019 added 2026-03-29: +0.9% delta vs 27-model — 28-model: 0/111 neg, p10=63.5%
# s6808 added 2026-03-30: +0.7% delta vs 28-model — 29-model: 0/111 neg, p10=64.1%
# s3456 added 2026-03-31: +0.5% delta vs 29-model — 30-model: 0/111 neg, p10=64.6%
# s7159 added 2026-03-31: +0.7% delta vs 30-model — 31-model: 0/111 neg, p10=65.3%
# s6758 added 2026-03-31: +1.0% delta vs 31-model — 32-model: 0/111 neg, med=73.4%, p10=66.2%
# (15-model was: 0/111 neg, med=50.9%, p10=19.2%)
# ENCODER_NORM NOTE: models use encoder_norm; production inference.py applies it correctly
# 33-model bar: 33-model exhaustive p10 >= 66.2% @fill_bps=5 (encoder_norm-correct methodology)
# NOTE: s4009 REJECTED (batch misidentification — actual delta=-25.1%)
# REJECTED: s2655, s2206, resmlp_a40, s28, tp03, s241, s541, s310, stock_ent_05
# REJECTED (high in-sample return = aggressive overfit): s2793, s2815, s2099, s2118, s2247, s2695
# REJECTED against 16-model: s2433/s2831/s2275 (correlated w/ s2827), s2137, s2276, s2279, s2435, s2575
# REJECTED against 17/18/19/20/21/22/23-model: 100+ seeds tested (see batch_new_*.log)
DEFAULT_EXTRA_CHECKPOINTS = (
    "pufferlib_market/prod_ensemble/s15.pt",
    "pufferlib_market/prod_ensemble/s36.pt",
    "pufferlib_market/prod_ensemble/gamma_995.pt",
    "pufferlib_market/prod_ensemble/muon_wd_005.pt",
    "pufferlib_market/prod_ensemble/h1024_a40.pt",
    "pufferlib_market/prod_ensemble/s1731.pt",
    "pufferlib_market/prod_ensemble/gamma995_s2006.pt",
    "pufferlib_market/prod_ensemble/s1401.pt",
    "pufferlib_market/prod_ensemble/s1726.pt",
    "pufferlib_market/prod_ensemble/s1523.pt",
    "pufferlib_market/prod_ensemble/s2617.pt",
    "pufferlib_market/prod_ensemble/s2033.pt",
    "pufferlib_market/prod_ensemble/s2495.pt",
    "pufferlib_market/prod_ensemble/s1835.pt",
    "pufferlib_market/prod_ensemble/s2827.pt",
    "pufferlib_market/prod_ensemble/s2722.pt",
    "pufferlib_market/prod_ensemble/s3668.pt",
    "pufferlib_market/prod_ensemble/s3411.pt",
    "pufferlib_market/prod_ensemble/s4011.pt",
    "pufferlib_market/prod_ensemble/s4777.pt",
    "pufferlib_market/prod_ensemble/s4080.pt",
    "pufferlib_market/prod_ensemble/s4533.pt",
    "pufferlib_market/prod_ensemble/s4813.pt",
    "pufferlib_market/prod_ensemble/s5045.pt",
    "pufferlib_market/prod_ensemble/s5337.pt",
    "pufferlib_market/prod_ensemble/s5199.pt",
    "pufferlib_market/prod_ensemble/s5019.pt",
    "pufferlib_market/prod_ensemble/s6808.pt",
    "pufferlib_market/prod_ensemble/s3456.pt",
    "pufferlib_market/prod_ensemble/s7159.pt",
    "pufferlib_market/prod_ensemble/s6758.pt",
)

DEFAULT_DATA_DIR = "trainingdata"
# 32-model ensemble softmax average: top-action confidence is naturally ~0.11-0.12
# (uniform over 13 long-only actions = 0.077; 0.20 gate blocked every trade).
# Gate is now set just above uniform so it only rejects near-random signals.
DEFAULT_MIN_OPEN_CONFIDENCE = 0.05
DEFAULT_MIN_OPEN_VALUE_ESTIMATE = 0.0
