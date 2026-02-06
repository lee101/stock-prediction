# Binance Progress Log

Updated: 2026-02-06

## Notes / Fixes
- Refreshed `trainingdatahourly/` via Alpaca (up to 2026-02-06) and rebuilt `trainingdatahourlybinance/` (Binance symbol aliases as symlinks to Alpaca hourly CSVs, while preserving any real Binance-downloaded files already present).
- Fixed Chronos multi-step horizon alignment in `binanceneural/forecasts.py` (previously multi-horizon caches effectively used the 1-step prediction for every horizon). Any cached forecasts for horizons >1 generated before 2026-02-06 should be considered invalid and rebuilt before comparing multi-horizon PnL/MAE runs.
- Added stable-quote symbol utilities + tests, plus a builder script for `trainingdatahourlybinance/`.
  - New: `src/binance_symbol_utils.py`, `tests/test_binance_symbol_utils.py`, `scripts/build_trainingdatahourlybinance.py`.
- Fixed `refresh_daily_inputs.py` / `update_key_forecasts.py` to run forecast refresh under the active venv interpreter (was calling system `python`, breaking `chronos` imports) and corrected log formatting.
- Added `binancecrosslearning/` pipeline (multi-symbol Chronos2 fine-tune + global policy + selector) with Binance defaults.
- Extended crypto symbol detection + fee heuristics for stable-quote pairs (USDT/FDUSD/USDC/etc); added tests.
- Added `state_dict`-aware max_len inference when loading classic models so positional encoding buffers match checkpoint shapes.
  - Updated: `binanceneural/model.py`, `binancechronossolexperiment/inference.py`, `binanceneural/run_simulation.py`, `binanceneural/trade_binance_hourly.py`, `binanceneural/sweep.py`.
- Added max_len inference for binanceexp1 checkpoint reloads (prevents positional encoding size mismatch).
  - Updated: `binanceexp1/run_experiment.py`, `binanceexp1/sweep.py`.
- Added `value_embedding`-aware max_len inference for nano policy checkpoints (avoids size mismatch on reload).
- New PnL/RL utility library (`src/tradinglib/`) with tests (metrics, rewards, benchmarks, target gating).
- Added input-dim alignment for checkpoint embeddings when feature sets drift (pads/trims `embed.weight`).
  - Updated: `binanceneural/model.py`, `binanceneural/sweep.py`, `binanceneural/run_simulation.py`,
    `binanceexp1/sweep.py`, `binanceexp1/run_experiment.py`, `binanceexp1/run_multiasset_selector.py`.
- Added multi-asset best-trade selector simulator + CLI + tests.
  - New: `binanceneural/marketsimulator/selector.py`, `binanceexp1/run_multiasset_selector.py`,
    `tests/test_binance_multiasset_selector.py`.

## Chronos2 LoRA (hourly, Alpaca data)
- BTCUSD LoRA: `chronos2_finetuned/BTCUSD_lora_20260203_051412` → Validation MAE% 0.2785, preaug=diff
- ETHUSD LoRA: `chronos2_finetuned/ETHUSD_lora_20260203_051846` → Validation MAE% 0.4450, preaug=diff
- LINKUSD LoRA: `chronos2_finetuned/LINKUSD_lora_20260203_052258` → Validation MAE% 0.4456, preaug=diff
- SOLUSD LoRA: `chronos2_finetuned/SOLUSD_lora_20260203_052715` → Validation MAE% 0.4337, preaug=diff
- UNIUSD LoRA: `chronos2_finetuned/UNIUSD_lora_20260203_091020` → Validation MAE% 0.5416, preaug=detrending
- NFLX LoRA: `chronos2_finetuned/NFLX_lora_20260203_091648` → Validation MAE% 0.1050, preaug=differencing
- NVDA LoRA: `chronos2_finetuned/NVDA_lora_20260203_092111` → Validation MAE% 0.1333, preaug=differencing
- Updated `hyperparams/chronos2/hourly/{BTCUSD,ETHUSD,LINKUSD,SOLUSD,UNIUSD,NFLX,NVDA}.json` to point `model_id` at the LoRA finetuned checkpoints.
- Forecast cache rebuilt for h1 only (BTCUSD/ETHUSD/LINKUSD/SOLUSD/UNIUSD/NFLX/NVDA); h24 recompute deferred (very heavy with long contexts).
- Supervisor `binanceexp1-solusd` updated to cross-pair BTCUSD/ETHUSD/LINKUSD checkpoints (horizon=1, seq=96).

## Experiments (Chronos2 + Binance neural)

### Baseline (classic, full history)
- Run: `chronos_sol_20260202_041139`
- Metrics (10-day test): total_return=0.3433135070, sortino=136.8685975259, annualized_return=49893.15079393116
- Last 2 days: total_return=0.08457621078, sortino=218.4418841462, annualized_return=2722471.909152695

### Nano + Muon (90d history)
- Run: `chronos_sol_nano_muon_20260202_230000`
- Command:
  `python -m binancechronossolexperiment.run_experiment --symbol SOLUSDT --data-root trainingdatahourlybinance --forecast-cache-root binancechronossolexperiment/forecast_cache --model-id chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt --context-hours 3072 --chronos-batch-size 32 --horizons 1 --sequence-length 72 --val-days 20 --test-days 10 --max-history-days 90 --epochs 6 --batch-size 128 --learning-rate 3e-4 --weight-decay 1e-4 --optimizer muon_mix --model-arch nano --num-kv-heads 2 --mlp-ratio 4.0 --logits-softcap 12.0 --muon-lr 0.02 --muon-momentum 0.95 --muon-ns-steps 5 --cache-only --no-compile --run-name chronos_sol_nano_muon_20260202_230000`
- Metrics (10-day test): total_return=0.1933922754, sortino=50.0260710990, annualized_return=651.0995876932
- Last 2 days: total_return=0.045477, sortino=70.023416, annualized_return=3347.674857

### Nano + AdamW (90d history)
- Run: `chronos_sol_nano_adamw_20260202_230500`
- Metrics (10-day test): total_return=0.2216511106, sortino=47.5818631247, annualized_return=1536.6274827896
- Last 2 days: total_return=0.044726, sortino=46.942293, annualized_return=2936.213771

### Nano + Muon (365d history)
- Run: `chronos_sol_nano_muon_20260202_235000`
- Metrics (10-day test): total_return=0.178760, sortino=33.205916, annualized_return=413.887796
- Last 2 days: total_return=0.057223, sortino=166.957110, annualized_return=25727.928457

### Classic + Muon (365d history, retry after pos-encoding fix)
- Run: `chronos_sol_classic_muon_20260202_235500_retry`
- Metrics (10-day test): total_return=0.131389, sortino=38.260542, annualized_return=91.263087
- Last 2 days: total_return=-0.002678, sortino=-2.415733, annualized_return=-0.386949

### Nano + Muon (full history)
- Run: `chronos_sol_nano_muon_full_20260203_001000`
- Metrics (10-day test): total_return=0.163166, sortino=133.889350, annualized_return=253.654864
- Last 2 days: total_return=0.049308, sortino=3301.882512, annualized_return=6527.535363

### Nano + AdamW (full history)
- Run: `chronos_sol_nano_adamw_full_20260203_003000`
- Metrics (10-day test): total_return=0.131405, sortino=106.950196, annualized_return=91.309086
- Last 2 days: total_return=0.039455, sortino=388.584324, annualized_return=1165.838243

### Nano + Muon (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_muon_h1_4_24_20260203_010000`
- Metrics (10-day test): total_return=0.139216, sortino=92.137741, annualized_return=117.786575
- Last 2 days: total_return=0.029039, sortino=1656.626735, annualized_return=184.700605

### Nano + AdamW (multi-horizon 1/4/24, full history)
- Run: `chronos_sol_nano_adamw_h1_4_24_20260203_092000`
- Metrics (10-day test): total_return=0.150219, sortino=139.825174, annualized_return=167.955148
- Last 2 days: total_return=0.056290, sortino=1120.777238, annualized_return=21899.280913

### Nano v2 (nanochat-style upgrades, multi-horizon 1/4/24, full history)
- Run: `chronos_sol_v2_nano_muon_h1_4_24_20260203_110000`
- Settings: muon_mix, residual scalars, value embeddings (every layer), attention_window=64, weight_decay linear->0, amp bfloat16
- Metrics (10-day test): total_return=0.135517, sortino=153.291137, annualized_return=104.436765
- Last 2 days: total_return=0.008194, sortino=376.950354, annualized_return=3.434440

### Nano v2 sweep (4-epoch runs, full history)
- Win=0 alt2: `chronos_sol_v2_win0_alt2_20260203_120000`
  - Metrics (10-day test): total_return=0.168163, sortino=127.621894, annualized_return=296.979450
  - Last 2 days: total_return=0.034318, sortino=4112.864558, annualized_return=471.432240
- Win=128 alt2: `chronos_sol_v2_win128_alt2_20260203_123000`
  - Metrics (10-day test): total_return=0.176363, sortino=131.492666, annualized_return=384.060802
  - Last 2 days: total_return=0.027708, sortino=5468.897139, annualized_return=145.641555
- Win=64 alt1: `chronos_sol_v2_win64_alt1_20260203_130000`
  - Metrics (10-day test): total_return=0.135580, sortino=388.062129, annualized_return=104.651151
  - Last 2 days: total_return=0.026850, sortino=2206.419941, annualized_return=124.898822
- Win=128 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win128_skip01_20260203_133000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=96 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win96_skip01_20260203_150000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=192 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win192_skip01_20260203_153000`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=32 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win32_skip01_20260203_160000`
  - Metrics (10-day test): total_return=0.140784, sortino=282.894742, annualized_return=123.930345
  - Last 2 days: total_return=0.029802, sortino=311.508539, annualized_return=211.610349
- Win=48 alt2 + skip_scale_init=0.1: `chronos_sol_v2_win48_skip01_20260203_163000`
  - Metrics (10-day test): total_return=0.152436, sortino=122.565052, annualized_return=180.312070
  - Last 2 days: total_return=0.037831, sortino=14278.210667, annualized_return=876.298951
- Win=128 alt2 + skip_scale_init=0.05: `chronos_sol_v2_win128_skip005_20260203_170000`
  - Metrics (10-day test): total_return=0.148934, sortino=91.914223, annualized_return=161.174633
  - Last 2 days: total_return=0.034244, sortino=12127.487356, annualized_return=465.354448
- Win=128 alt2 + skip_scale_init=0.2: `chronos_sol_v2_win128_skip02_20260203_173000`
  - Metrics (10-day test): total_return=0.153551, sortino=152.815771, annualized_return=186.853345
  - Last 2 days: total_return=0.033610, sortino=2113.671477, annualized_return=415.989073
- Seq=96, Win=64 alt2 + skip_scale_init=0.1: `chronos_sol_v2_seq96_win64_skip01_20260203_180000`
  - Metrics (10-day test): total_return=0.154001, sortino=89.258249, annualized_return=189.558575
  - Last 2 days: total_return=0.060890, sortino=1473.177357, annualized_return=48399.069467

Note: attention_window >= sequence_length behaves like full attention, so Win=96/128/192 runs matched the Win=128 baseline when other settings were identical.

### Nano v2 follow-up sweep (attention window/value embedding/skip-scale, 4-epoch)
- Win=96 + skip_scale_init=0.1: `chronos_sol_v2_win96_skip01_20260203_194200`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=192 + skip_scale_init=0.1: `chronos_sol_v2_win192_skip01_20260203_195500`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=256 + skip_scale_init=0.1: `chronos_sol_v2_win256_skip01_20260203_200500`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=96 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win96_skip01_ve3_20260203_201700`
  - Metrics (10-day test): total_return=0.118162, sortino=170.433597, annualized_return=58.955209
  - Last 2 days: total_return=0.035090, sortino=18497.570350, annualized_return=540.389433
- Win=128 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win128_skip01_ve3_20260203_210500`
  - Metrics (10-day test): total_return=0.118162, sortino=170.433597, annualized_return=58.955209
  - Last 2 days: total_return=0.035090, sortino=18497.570350, annualized_return=540.389433
- Win=128 + skip_scale_init=0.05: `chronos_sol_v2_win128_skip005_20260203_203000`
  - Metrics (10-day test): total_return=0.148934, sortino=91.914223, annualized_return=161.174633
  - Last 2 days: total_return=0.034244, sortino=12127.487356, annualized_return=465.354448
- Win=128 + skip_scale_init=0.2: `chronos_sol_v2_win128_skip02_20260203_204300`
  - Metrics (10-day test): total_return=0.153551, sortino=152.815771, annualized_return=186.853345
  - Last 2 days: total_return=0.033610, sortino=2113.671477, annualized_return=415.989073
- Win=64 + skip_scale_init=0.1: `chronos_sol_v2_win64_skip01_20260203_212000`
  - Metrics (10-day test): total_return=0.102255, sortino=103.215131, annualized_return=34.460601
  - Last 2 days: total_return=0.007902, sortino=535.322223, annualized_return=3.205639

### Nano v2 micro-sweep (attention window/value embedding/Muon LR, 4-epoch)
- Win=24 + skip_scale_init=0.1: `chronos_sol_v2_win24_skip01_20260203_190000`
  - Metrics (10-day test): total_return=0.131183, sortino=49.734845, annualized_return=90.647583
  - Last 2 days: total_return=0.029858, sortino=93.975619, annualized_return=213.718822
- Win=40 + skip_scale_init=0.1: `chronos_sol_v2_win40_skip01_20260204_000000`
  - Metrics (10-day test): total_return=0.156633, sortino=40.493034, annualized_return=206.152934
  - Last 2 days: total_return=0.011449, sortino=8.054279, annualized_return=6.985773
- Win=56 + skip_scale_init=0.1: `chronos_sol_v2_win56_skip01_20260204_003000`
  - Metrics (10-day test): total_return=0.183902, sortino=151.187906, annualized_return=485.653023
  - Last 2 days: total_return=0.053722, sortino=7227.691172, annualized_return=14043.617352
- Win=64 + skip_scale_init=0.1 + value_embedding_every=3: `chronos_sol_v2_win64_vale3_skip01_20260204_010000`
  - Metrics (10-day test): total_return=0.106891, sortino=204.396658, annualized_return=40.358946
  - Last 2 days: total_return=0.028697, sortino=7396.888054, annualized_return=173.771543
- Win=64 + skip_scale_init=0.1 + value_embedding_every=4: `chronos_sol_v2_win64_vale4_skip01_20260204_013000`
  - Metrics (10-day test): total_return=0.149532, sortino=116.811572, annualized_return=164.296970
  - Last 2 days: total_return=0.052753, sortino=4335.442502, annualized_return=11872.874383
- Win=128 + skip_scale_init=0.1 + value_embedding_every=1: `chronos_sol_v2_vale1_skip01_20260204_011846`
  - Metrics (10-day test): total_return=0.185369, sortino=160.869717, annualized_return=508.252224
  - Last 2 days: total_return=0.057419, sortino=21575.395855, annualized_return=26614.317587
- Win=128 + skip_scale_init=0.05 + value_embedding_every=1: `chronos_sol_v2_vale1_skip005_20260204_012841`
  - Metrics (10-day test): total_return=0.128222, sortino=88.884578, annualized_return=82.254881
  - Last 2 days: total_return=0.023395, sortino=4226.743686, annualized_return=67.061177
- Win=128 + skip_scale_init=0.15 + value_embedding_every=1: `chronos_sol_v2_vale1_skip015_20260204_013817`
  - Metrics (10-day test): total_return=0.157671, sortino=113.337640, annualized_return=213.079502
  - Last 2 days: total_return=0.031390, sortino=13977.423338, annualized_return=280.653506
- Win=128 + skip_scale_init=0.1 + muon_lr=0.015: `chronos_sol_v2_win128_skip01_muon015_20260204_020000`
  - Metrics (10-day test): total_return=0.144029, sortino=114.980093, annualized_return=137.636905
  - Last 2 days: total_return=0.032975, sortino=1601.106040, annualized_return=371.751577
- Win=128 + skip_scale_init=0.1 + muon_lr=0.03: `chronos_sol_v2_win128_skip01_muon03_20260204_023000`
  - Metrics (10-day test): total_return=0.109021, sortino=65.846097, annualized_return=43.377910
  - Last 2 days: total_return=0.028804, sortino=6587.062715, annualized_return=177.116790
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.93: `chronos_sol_v2_muon093_20260204_014809`
  - Metrics (10-day test): total_return=0.112421, sortino=140.816244, annualized_return=48.646180
  - Last 2 days: total_return=0.032995, sortino=4038.468056, annualized_return=373.074915
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.97: `chronos_sol_v2_muon097_20260204_015606`
  - Metrics (10-day test): total_return=0.193740, sortino=152.846581, annualized_return=658.107035
  - Last 2 days: total_return=0.048688, sortino=5613.438236, annualized_return=5860.035917
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.98: `chronos_sol_v2_muon098_20260204_021639`
  - Metrics (10-day test): total_return=0.172828, sortino=95.283727, annualized_return=343.849245
  - Last 2 days: total_return=0.044075, sortino=14528.643033, annualized_return=2620.416473
- Win=128 + skip_scale_init=0.1 + muon_momentum=0.99: `chronos_sol_v2_muon099_20260204_021641`
  - Metrics (10-day test): total_return=0.129935, sortino=140.445793, annualized_return=87.015755
  - Last 2 days: total_return=0.007348, sortino=1231.628070, annualized_return=2.804469
- Win=128 + skip_scale_init=0.1 + muon_lr=0.025: `chronos_sol_v2_muon025_20260204_021642`
  - Metrics (10-day test): total_return=0.122925, sortino=129.919800, annualized_return=69.063854
  - Last 2 days: total_return=0.027347, sortino=12956.396462, annualized_return=136.527182
- Win=128 + skip_scale_init=0.1 + muon_lr=0.035: `chronos_sol_v2_muon035_20260204_021643`
  - Metrics (10-day test): total_return=0.122698, sortino=58.989053, annualized_return=68.546359
  - Last 2 days: total_return=0.027950, sortino=13908.088305, annualized_return=152.062641
- Win=128 + skip_scale_init=0.1 + context_hours=2048: `chronos_sol_v2_ctx2048_20260204_021644`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368
- Win=128 + skip_scale_init=0.1 + context_hours=4096: `chronos_sol_v2_ctx4096_20260204_021645`
  - Metrics (10-day test): total_return=0.224781, sortino=110.011380, annualized_return=1687.823288
  - Last 2 days: total_return=0.055692, sortino=32793.242972, annualized_return=19748.420368

Note: context_hours 2048/4096 matched the best baseline exactly (cache-only run likely reused identical forecast cache).

### Nano v2 sequence length sweep (full attention, 4-epoch)
- Seq=96, Win=96 + skip_scale_init=0.1: `chronos_sol_v2_seq96_win96_skip01_20260204_005446`
  - Metrics (10-day test): total_return=0.147737, sortino=105.345177, annualized_return=155.092648
  - Last 2 days: total_return=0.052962, sortino=235.802923, annualized_return=12311.639256
- Seq=128, Win=128 + skip_scale_init=0.1: `chronos_sol_v2_seq128_win128_skip01_20260204_010516`
  - Metrics (10-day test): total_return=0.150824, sortino=94.192548, annualized_return=171.243700
  - Last 2 days: total_return=0.062229, sortino=202.545430, annualized_return=60922.071574

Best PnL unchanged after micro-sweep + seq-length sweep: total_return=0.224781 (skip_scale_init=0.1, attention_window >= 96, seq_len=72).

### Nano v2 longer test window (20-day test / 20-day val, 4-epoch)
- Baseline (skip_scale_init=0.1): `chronos_sol_v2_test20_skip01_20260204_042613`
  - Metrics (20-day test): total_return=0.283456, sortino=104.303940, annualized_return=94.958200
  - Last 2 days: total_return=0.039501, sortino=10193.434517, annualized_return=1175.415846
- value_embedding_every=1 (skip_scale_init=0.1): `chronos_sol_v2_test20_vale1_skip01_20260204_043452`
  - Metrics (20-day test): total_return=0.272807, sortino=72.602345, annualized_return=81.396613
  - Last 2 days: total_return=0.033666, sortino=10424.777798, annualized_return=420.089043
- muon_momentum=0.97 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon097_20260204_044258`
  - Metrics (20-day test): total_return=0.324214, sortino=92.312041, annualized_return=168.973961
  - Last 2 days: total_return=0.031608, sortino=1188.725587, annualized_return=291.713480
- attention_window=56 (skip_scale_init=0.1): `chronos_sol_v2_test20_win56_skip01_20260204_045236`
  - Metrics (20-day test): total_return=0.282851, sortino=115.827681, annualized_return=94.134199
  - Last 2 days: total_return=0.024458, sortino=11294.884243, annualized_return=81.255827
- muon_momentum=0.98 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon098_20260204_050146`
  - Metrics (20-day test): total_return=0.281030, sortino=262.268200, annualized_return=91.695231
  - Last 2 days: total_return=0.031809, sortino=6890.016604, annualized_return=302.285574
- muon_momentum=0.96 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon096_20260204_110842`
  - Metrics (20-day test): total_return=0.193390, sortino=76.060899, annualized_return=24.363057
  - Last 2 days: total_return=0.032720, sortino=3114.981986, annualized_return=355.327393
- muon_momentum=0.975 (skip_scale_init=0.1): `chronos_sol_v2_test20_muon0975_20260204_111637`
  - Metrics (20-day test): total_return=0.404804, sortino=72.492598, annualized_return=499.736632
  - Last 2 days: total_return=0.034365, sortino=31.250368, annualized_return=475.437671
- value_embedding_every=1, skip_scale_init=0.08: `chronos_sol_v2_test20_vale1_skip008_20260204_112657`
  - Metrics (20-day test): total_return=0.328249, sortino=87.316594, annualized_return=178.698710
  - Last 2 days: total_return=0.058472, sortino=8415.021779, annualized_return=31914.709940
- value_embedding_every=1, skip_scale_init=0.12: `chronos_sol_v2_test20_vale1_skip012_20260204_113544`
  - Metrics (20-day test): total_return=0.269032, sortino=51.479611, annualized_return=77.039409
  - Last 2 days: total_return=0.030477, sortino=9311.769567, annualized_return=238.593876

Best 20-day PnL so far: total_return=0.404804 (muon_momentum=0.975, skip_scale_init=0.1).

### Nano v2 longer test window (30-day test / 20-day val, 4-epoch)
- muon_momentum=0.975 (skip_scale_init=0.1): `chronos_sol_v2_test30_muon0975_20260204_114507`
  - Metrics (30-day test): total_return=0.703753, sortino=106.251893, annualized_return=658.726004
  - Last 2 days: total_return=0.070984, sortino=28667.020760, annualized_return=272505.067898
- baseline (skip_scale_init=0.1): `chronos_sol_v2_test30_skip01_20260204_115545`
  - Metrics (30-day test): total_return=0.534233, sortino=81.648603, annualized_return=183.001900
  - Last 2 days: total_return=0.026818, sortino=4597.584149, annualized_return=124.183249

Best 30-day PnL so far: total_return=0.703753 (muon_momentum=0.975, skip_scale_init=0.1).

### Nano v2 best-config sanity (6-epoch)
- Run: `chronos_sol_v2_win128_skip01_e6_20260203_140000`
- Metrics (10-day test): total_return=0.157931, sortino=118.091167, annualized_return=214.846481
- Last 2 days: total_return=0.062030, sortino=188.092299, annualized_return=58877.772244

## RL Reward Sweep (SOLUSDT hourly)
- Run: `python -m rlsys.run_reward_sweep --csv trainingdatahourlybinance/SOLUSDT.csv`
- Results saved: `rlsys/results/reward_sweep.json`
- Raw reward: episode_reward=-0.026216, episode_sharpe=-2.740867, episode_max_drawdown=-0.199934
- Risk-adjusted reward: episode_reward=-0.061468, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763
- Sharpe-like reward: episode_reward=-0.115716, episode_sharpe=-2.304354, episode_max_drawdown=-0.222763

## Live Binance
- Attempted single-cycle live run failed due to restricted-region Binance API eligibility.
- Supervisor `binanceexp1-solusd` now running with BINANCE_TLD=us; requires Binance.US API keys (current keys rejected with "API-key format invalid").
- See `binanceresults.md` for full details of live attempt and command used.

## Syncs (R2)
- Synced binance checkpoints to R2:
  - `aws s3 sync binanceneural/checkpoints/ s3://models/stock/models/binance/binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"`
- Synced Chronos2 finetunes to R2:
  - `aws s3 sync chronos2_finetuned/ s3://models/stock/models/binance/chronos2_finetuned/ --endpoint-url "$R2_ENDPOINT"`

## Next Experiments
- Longer RL sweeps (e.g., 50k+ timesteps) with a grid over drawdown/volatility penalties.
- Multi-horizon (1/4/24) variants with larger sequence length (96/128) and/or context_hours 2048 vs 3072.
- Run longer backtests (30d+) on the best multi-horizon run and log trade stats + PnL targets.
- Multi-asset selector: tune min_edge/risk_weight, test cooldowns, and compare on matched windows vs SOLUSD h24 baseline.

## Binanceexp1 SOLUSD (marketsimulator, hourly)

Data & validation
- Backfilled hourly gap using Alpaca: 2025-04-16 through 2025-12-21.
- Max gap after backfill: ~3 hours.

Checkpoint
- `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt`

Intensity sweep (price_offset_pct=0, horizon=24, sequence_length=96, min_gap_pct=0.0003)
- 1.0 -> total_return 7.8325, sortino 55.8105
- 1.2 -> total_return 8.3542
- 1.6 -> total_return 9.0235
- 1.8 -> total_return 9.3230
- 2.0 -> total_return 9.5891
- 2.2 -> total_return 9.8570
- 2.5 -> total_return 10.1909
- 3.0 -> total_return 10.6377
- 4.0 -> total_return 11.3114
- 6.0 -> total_return 12.4451
- 8.0 -> total_return 13.5462
- 10.0 -> total_return 14.4308
- 15.0 -> total_return 16.2541, sortino 66.6059
- 20.0 -> total_return 17.7449, sortino 66.1148 (best PnL so far)

Risk mitigation tests
- Probe-mode after loss: reduced total return vs baseline.
- Max-hold forced close: reduced total return vs baseline.

Production target (SOLUSD)
- intensity-scale 20.0, horizon 24, sequence length 96, min-gap-pct 0.0003
- Runner: `scripts/run_binanceexp1_prod_solusd.sh`
- Supervisor: `supervisor/binanceexp1-solusd.conf`
## Binanceexp1 Horizon=1 (Chronos2 LoRA, h1-only cache)

### SOLUSD (full 5 epochs)
- Run: `solusd_h1_ft_20260203` (checkpoint: `binanceneural/checkpoints/solusd_h1_ft_20260203/epoch_005.pt`)
- Baseline eval (intensity=1.0): total_return 8.9329, sortino 47.5089
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 5.4643, sortino 55.0102
  - Best return: intensity 1.40, offset 0.00000 → total_return 10.8068, sortino 48.6080
- Risk mitigations:
  - probe-after-loss (notional 1.0) lowered return (7.5718) and sortino (42.7107)
  - max-hold-hours 24 had no measurable improvement
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 7.1795, sortino 60.5213
  - Best return: intensity 1.80, offset 0.00000 → total_return 14.4635, sortino 53.6626

### BTCUSD (quick 1 epoch / 300 steps)
- Run: `btcusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/btcusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 7.1547, sortino 253.3211
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.1541, sortino 253.5392
  - Best return: intensity 1.40, offset 0.00000 → total_return 8.5504, sortino 234.4700
- Expanded sweep (intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.3499, sortino 230.1740
  - Best return: intensity 1.80, offset 0.00010 → total_return 9.8338, sortino 202.4425
- Aggregate (contexts 64/96/192, trim_ratio 0.2):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.1246, sortino 243.8277
  - Best return: intensity 1.60, offset 0.00020 → total_return 9.1776, sortino 183.7078
- Blend horizons 1/24 (weights 0.7/0.3):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 7.5609, sortino 231.9530
  - Best return: intensity 1.40, offset 0.00020 → total_return 9.2805, sortino 182.3951

### ETHUSD (quick 1 epoch / 300 steps)
- Run: `ethusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/ethusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 3.6791, sortino 165.9452
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 3.6789, sortino 166.1271
  - Best return: intensity 1.40, offset 0.00000 → total_return 4.5134, sortino 145.5026
- Expanded sweep (intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 4.0286, sortino 165.7779
  - Best return: intensity 1.80, offset 0.00020 → total_return 5.3999, sortino 137.1733
- Aggregate (contexts 64/96/192, trim_ratio 0.2):
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 3.7065, sortino 166.4207
  - Best return: intensity 1.60, offset 0.00000 → total_return 4.8140, sortino 143.6667

### LINKUSD (quick 1 epoch / 300 steps)
- Run: `linkusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/linkusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 3.4529, sortino 44.2042
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 2.2955, sortino 46.8543
  - Best return: intensity 1.20, offset 0.00000 → total_return 3.9597, sortino 39.0122
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 2.8358, sortino 38.9894
  - Best return: intensity 1.80, offset 0.00000 → total_return 4.7164, sortino 33.1052

### UNIUSD (quick 1 epoch / 300 steps)
- Run: `uniusd_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/uniusd_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 8.0990, sortino 38.5863
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00020 → total_return 5.4407, sortino 45.0316
  - Best return: intensity 1.20, offset 0.00020 → total_return 9.5057, sortino 36.3816
- Backtest (best PnL config): intensity 1.20, offset 0.00020 → total_return 9.5057, sortino 36.3816
- Expanded sweep (h1-only cache; intensity 0.6-1.8, offsets 0-0.0005):
  - Best sortino: intensity 0.80, offset 0.00050 → total_return 6.2135, sortino 42.5577
  - Best return: intensity 1.00, offset 0.00020 → total_return 10.8490, sortino 36.6693

### NFLX (quick 1 epoch / 300 steps)
- Run: `nflx_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/nflx_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 1.0496, sortino 55.6672
- Sweep highlights:
  - Best sortino: intensity 1.00, offset 0.00000 → total_return 1.0497, sortino 55.6645
  - Best return: intensity 1.40, offset 0.00000 → total_return 1.1271, sortino 44.4451
- Backtest (best PnL config): intensity 1.40, offset 0.00000 → total_return 1.1271, sortino 44.4451

### NVDA (quick 1 epoch / 300 steps)
- Run: `nvda_h1_quick_20260203` (checkpoint: `binanceneural/checkpoints/nvda_h1_quick_20260203/epoch_001.pt`)
- Baseline eval (intensity=1.0): total_return 11.2049, sortino 38.6581
- Sweep highlights:
  - Best sortino: intensity 0.80, offset 0.00000 → total_return 6.8115, sortino 41.7632
  - Best return: intensity 1.00, offset 0.00050 → total_return 11.9568, sortino 34.3337
- Backtest (best PnL config): intensity 1.00, offset 0.00050 → total_return 11.9568, sortino 34.3337

## Multi-Asset Best-Trade Selector (BTC/ETH/LINK/UNI/SOL, h1)

Selector inputs
- Symbols: BTCUSD, ETHUSD, LINKUSD, UNIUSD, SOLUSD (h1-only forecast cache).
- Checkpoints: `binanceneural/checkpoints/{btcusd_h1_quick_20260203,ethusd_h1_quick_20260203,linkusd_h1_quick_20260203,uniusd_h1_quick_20260203,solusd_h1_ft_20260203}/epoch_*.pt`.
- Action overrides used (from expanded sweeps):
  - BTCUSD intensity 1.8 offset 0.0001
  - ETHUSD intensity 1.8 offset 0.0002
  - LINKUSD intensity 1.8 offset 0.0000
  - UNIUSD intensity 1.0 offset 0.0002
  - SOLUSD intensity 1.8 offset 0.0000
- CLI: `python -m binanceexp1.run_multiasset_selector ...` (see commands in logs).

Results (no same-bar re-entry by default)
- Baseline (edge_mode=high_low, min_edge=0.0): total_return 5.3570, sortino 29.3199.
- Best conservative (edge_mode=high_low, min_edge=0.002): total_return 13.9718, sortino 24.3030.
- Close-mode (edge_mode=close, min_edge=0.001): total_return 8.2331, sortino 25.0939.
- Aggressive upper bound (allow_reentry_same_bar, min_edge=0.002): total_return 42.1318, sortino 34.9956.

Comparison to best-20 sweep
- Best 20 sweep (SOLUSD horizon=24) total_return 17.7449 (not apples-to-apples; single-symbol, h24).
- Conservative selector is below that baseline; aggressive upper-bound run exceeds it.

## Cross-pair marketsimulator (BTCUSD/ETHUSD/LINKUSD, hourly)

Setup (2026-02-03)
- Checkpoints: `binanceneural/checkpoints/btcusd_h1_quick_20260203/epoch_001.pt`, `binanceneural/checkpoints/ethusd_h1_quick_20260203/epoch_001.pt`, `binanceneural/checkpoints/linkusd_h1_quick_20260203/epoch_001.pt`
- Features: binanceexp1 feature set (forecast_horizons=1, sequence_length=96)
- Validation window (subset): ~2025-05-03 through 2025-10-24

Metrics (per-symbol)
- BTCUSD: total_return 0.1068, sortino 88.6565
- ETHUSD: total_return 0.7707, sortino 26.9238
- LINKUSD: total_return 10.1283, sortino 63.2361

Combined (equal initial cash per symbol)
- total_return 13.1723, sortino 97.3380
