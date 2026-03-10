# FastForecaster Experiments (2026-03-04)

## Key findings

- The largest MAE issue came from an invalid filename in `trainingdatahourly/stocks`:
  `AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,JPM,V,WMT.csv`.
- FastForecaster now skips invalid symbol filenames via a strict symbol regex.
- On cleaned symbols, `return_loss_weight=0.2`, `direction_loss_weight=0.0`, EMA eval, and QK norm gave the best test MAE/RMSE in these runs.
- A focused regularization sweep (`sweep_20260304_095504`) improved best validation MAE from `2.62699` to `2.62374`.
- Multi-seed reruns of that config found `2.62070` at seed `1701`.
- Additional local refinement around horizon weighting found a new low:
  `2.61709` at `horizon_weight_power=0.30`, `weight_decay=0.01`, `seed=1701`
  (`-0.138%` vs `2.62070`, `-0.401%` vs `2.62765`).
- Extending the same cleaned 11-symbol setup to 5 epochs improved validation MAE further to `2.61487`.
- Increasing data budget (`max_symbols=24`, `max_train_windows_per_symbol=7000`) reached `2.19255` on a larger 22-symbol split.
  This is a different evaluation distribution and should be compared separately from the 11-symbol setup.
- A follow-up 4-seed scan of the 5-epoch 11-symbol setup confirmed `seed=1701` remains best:
  `1701: 2.61487`, `2026: 2.61590`, `1337: 2.62252`, `3333: 2.62776`.
- Frontier7 same-split refinement found a lower validation MAE at `2.60819` with
  `lr=2.5e-4`, `dropout=0.05`, `weight_decay=0.01`, `horizon_weight_power=0.28`, `return_loss_weight=0.2`, `direction_loss_weight=0.0`, EMA, and QK-norm.
- A seed scan of that Frontier7 best config still showed high seed sensitivity:
  `1701: 2.60819`, `2026: 2.62785`, `1337: 2.62805`, `3333: 2.62805`.
- Frontier8 and Frontier9 local retunes (longer epochs and nearby LR/WD/dropout/horizon values) did not beat the Frontier7 low.
- Frontier10 C++ objective ablation (`use_cpp_kernels` on/off) produced identical results on the tested config.
- Frontier11 return-loss scan recovered a stronger setting at `return_loss_weight=0.2` with `2.62013`.
- Frontier12 horizon scan improved current-path best validation MAE to `2.61726` at `horizon_weight_power=0.30`.
- A Frontier12 seed scan again showed strong seed sensitivity:
  `1701: 2.61726`, `2026: 2.62755`, `1337: 2.62805`, `3333: 2.62805`.
- Frontier13 local refinement around the `hwp=0.30` winner (LR/WD/dropout/hwp micro-variants) did not beat Frontier12;
  best was `2.62213`.
- Frontier14 architecture scan (depth/width/FF variants) did not beat Frontier12;
  best was `2.62035` (`hidden_dim=384`, `num_layers=6`, `ff_multiplier=4`, `num_heads=8`).
- Frontier15 targeted regularization/horizon scan on the `hidden_dim=384`, `num_layers=6` architecture improved
  current-path best validation MAE to `2.61605` at `horizon_weight_power=0.26`.
- Frontier15 seed sweep on that winner remained highly seed-sensitive:
  `1701: 2.61605`, `3333: 2.62220`, `1337: 2.62762`, `2026: 2.62767`.
- Frontier16 nearby horizon/dropout follow-up (`hwp=0.24/0.25/0.27` and `dropout=0.07` at `hwp=0.26`)
  did not improve on Frontier15; best was `2.62146`.
- Frontier17 weight-decay and learning-rate scan around the Frontier15 winner also did not improve;
  best was `2.61835` (`wd=0.011`, `lr=2.5e-4`).
- Frontier18 objective-weight scan (`return_loss_weight` and `direction_loss_weight`) did not improve either;
  best was `2.61612` at `return_loss_weight=0.2`, `direction_loss_weight=0.01`, still above Frontier15 by `+0.00007`.
- Frontier19 micro-scan around the Frontier18 near-tie setting found a new current-path low at `2.61576`
  with `lr=2.55e-4`, `return_loss_weight=0.2`, `direction_loss_weight=0.01`, and `horizon_weight_power=0.26`.

## Run summary

| Run | Config highlights | best_val_mae | test_mae | test_rmse | test_mape | notes |
|---|---|---:|---:|---:|---:|---|
| baseline_hourly_v1 | no aux losses, no EMA, no QK | 1.9301 | 2.1444 | 4.1643 | 2.0837 | 5-symbol control |
| enhanced_hourly_v2 | return=0.2, dir=0.02, EMA, QK | 1.8042 | 2.1077 | 4.0422 | 2.0175 | 5-symbol control |
| enhanced_hourly_v5_retonly_noqk | return=0.2, dir=0.0, EMA, no QK | 1.8061 | 2.1064 | 4.0301 | 2.0197 | best 5-symbol test MAE/RMSE at time |
| enhanced_hourly_12sym_e6_retonly_noqk | return=0.2, dir=0.0, EMA, no QK | 21.0411 | 24.6126 | 76.8854 | 9.4178 | contaminated by invalid symbol file |
| enhanced_hourly_12sym_e6_retonly_noqk_cleansym | return=0.2, dir=0.0, EMA, no QK | 2.6265 | 2.5329 | 3.9947 | 1.7348 | cleaned symbol universe |
| enhanced_hourly_12sym_e6_best_qk_retonly | return=0.2, dir=0.0, EMA, QK | 2.6276 | 2.5100 | 3.9621 | 1.7235 | current best 11-symbol test metrics |
| sweep_20260304_095504 trial012 | lr=2.5e-4, wd=0.01, dropout=0.05, hwp=0.25, EMA, QK | 2.6237 | 2.5194 | 3.9752 | 1.7273 | best validation trial from focused sweep |
| hourly_bestcfg_seed1701 | same as trial012 + seed=1701 | 2.6207 | 2.5183 | 3.9752 | 1.7269 | interim validation-MAE best |
| hourly_refine_seed1701_hwp0p30_wd0p010 | lr=2.5e-4, wd=0.01, hwp=0.30, seed=1701 | 2.6171 | 2.5133 | 3.9670 | 1.7243 | new best validation MAE |
| hourly_frontier6_cppcheck_seed1701_hwp0p30_wd0p010 | refine-best config + C++ weighted-MAE path | 2.6174 | 2.5132 | 3.9689 | 1.7244 | objective parity check with compiled kernels |
| hourly_frontier6_epoch5_seed1701_hwp0p30_wd0p010 | refine-best config, 5 epochs | 2.6149 | 2.5275 | 4.0017 | 1.7299 | best val MAE on same 11-symbol split |
| seed_sweep_20260304_113613 | same as frontier6_epoch5 over seeds 1337/1701/2026/3333 | 2.6149 | 2.5275 | 4.0017 | 1.7299 | best seed remained 1701 |
| hourly_frontier7_seed1701_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p20 | frontier7 same-split best (hwp=0.28) | 2.6082 | 2.5308 | 4.0135 | 1.7319 | best validation MAE on cleaned 11-symbol split |
| seed_sweep_20260304_120817 | frontier7 hwp=0.28 over seeds 1337/1701/2026/3333 | 2.6082 | 2.5308 | 4.0135 | 1.7319 | only seed 1701 reproduced the gain |
| hourly_frontier10_seed1701_repro_hwp0p28_nocpp | hwp=0.28 repro, pure PyTorch objective path | 2.6228 | 2.5188 | 3.9651 | 1.7259 | baseline for C++ objective ablation |
| hourly_frontier10_seed1701_repro_hwp0p28_cpp | same as above with C++ weighted-MAE path | 2.6228 | 2.5188 | 3.9651 | 1.7259 | matched no-C++ metrics exactly |
| hourly_frontier11_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p20 | return-loss scan winner on current path | 2.6201 | 2.5166 | 3.9632 | 1.7245 | best `return_loss_weight` in scan |
| hourly_frontier12_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p30_rlw0p20 | horizon-power retune winner (`hwp=0.30`) | 2.6173 | 2.5212 | 3.9799 | 1.7262 | best current-path validation MAE |
| hourly_frontier13_seed1701_ep5_lr0p000255_do0p05_wd0p010_hwp0p30_rlw0p20 | local refine winner around Frontier12 (`lr=2.55e-4`) | 2.6221 | 2.5123 | 3.9643 | 1.7228 | no improvement vs Frontier12 val MAE |
| hourly_frontier14_seed1701_hd384_l6_ff4_h8_hwp0p30 | architecture scan winner (`hidden=384`, `layers=6`, `ff=4`) | 2.6204 | 2.5313 | 4.0017 | 1.7343 | no improvement vs Frontier12 val MAE |
| hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p26 | Frontier15 winner (`hidden=384`, `layers=6`, `hwp=0.26`) | 2.6160 | 2.5177 | 3.9821 | 1.7257 | previous current-path best (later beaten in frontier19) |
| seed_sweep_20260304_142244 | frontier15 hwp=0.26 over seeds 1337/1701/2026/3333 | 2.6160 | 2.5177 | 3.9821 | 1.7257 | only seed 1701 reproduced the low |
| hourly_frontier16_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p25 | frontier16 nearby-hwp scan best (`hwp=0.25`) | 2.6215 | 2.5179 | 3.9816 | 1.7253 | no improvement vs frontier15 |
| hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p011_do0p05_hwp0p26 | frontier17 wd/lr scan best (`wd=0.011`, `lr=2.5e-4`) | 2.6184 | 2.5149 | 3.9784 | 1.7242 | no improvement vs frontier15 |
| hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p20_dlw0p01_hwp0p26 | frontier18 objective-weight scan best (`rlw=0.20`, `dlw=0.01`) | 2.6161 | 2.5152 | 3.9744 | 1.7243 | near tie, but still above frontier15 |
| hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p000255_wd0p010_do0p05_rlw0p20_dlw0p01_hwp0p26 | frontier19 micro-scan winner (`lr=2.55e-4`, `rlw=0.20`, `dlw=0.01`) | 2.6158 | 2.5138 | 3.9653 | 1.7213 | new current-path best validation MAE |
| seed_sweep_20260304_132313 | frontier12 hwp=0.30 over seeds 1337/1701/2026/3333 | 2.6173 | 2.5212 | 3.9799 | 1.7262 | only seed 1701 achieved the low |
| hourly_frontier6_moredata_ms24_seed1701_hwp0p30_wd0p010 | max_symbols=24, train_windows/sym=7000 | 2.1926 | 2.1064 | 3.4963 | 1.8079 | larger 22-symbol split (not directly comparable) |
| hourly_12sym_long_e10_newbestcfg | trial012 config, 10 epochs | 2.6247 | 2.5419 | 4.0064 | 1.7401 | longer run overfit after epoch 4 |
| daily_12sym_e4_auto_minrows | return=0.2, dir=0.0, EMA, QK | 7.7279 | 6.0505 | 10.5435 | 3.9981 | daily mode with auto min-rows default |

## W&B runs

- `fastforecaster_hourly_12sym_e6_20260304`:
  https://wandb.ai/lee101p/stock/runs/y3tdkyog
- `fastforecaster_hourly_12sym_e6_retonly_noqk_20260304`:
  https://wandb.ai/lee101p/stock/runs/ipdzwvhi
- `fastforecaster_hourly_12sym_e6_retonly_noqk_cleansym_20260304`:
  https://wandb.ai/lee101p/stock/runs/ymztpx6n
- `fastforecaster_hourly_12sym_e6_best_qk_retonly_20260304`:
  https://wandb.ai/lee101p/stock/runs/92f31eeu
- `fastforecaster_daily_12sym_e4_auto_minrows_20260304`:
  https://wandb.ai/lee101p/stock/runs/8659jbne
- `fastforecaster_hourly_12sym_long_e12_besttrial_nocompile`:
  https://wandb.ai/lee101p/stock/runs/zxs0b9ah
- `fastforecaster_hourly_12sym_long_e10_newbestcfg`:
  https://wandb.ai/lee101p/stock/runs/yfcv4zun
- `trial012_rlw0p200_dlw0p000_dms16p0_lr0p00025_wd0p0100_do0p050_hwp0p25_qk1_ema1`:
  https://wandb.ai/lee101p/stock/runs/m2zw1082
- `fastforecaster_hourly_bestcfg_seed1701`:
  https://wandb.ai/lee101p/stock/runs/16tzcts6
- `fastforecaster_hourly_refine_seed1701_hwp0p30_wd0p010`:
  https://wandb.ai/lee101p/stock/runs/bhzcc76f
- `fastforecaster_hourly_frontier6_cppcheck_seed1701_hwp0p30_wd0p010`:
  https://wandb.ai/lee101p/stock/runs/tciro273
- `fastforecaster_hourly_frontier6_epoch5_seed1701_hwp0p30_wd0p010`:
  https://wandb.ai/lee101p/stock/runs/5p02rjhl
- `fastforecaster_hourly_frontier6_moredata_ms24_seed1701_hwp0p30_wd0p010`:
  https://wandb.ai/lee101p/stock/runs/pafh5q6v
- `fastforecaster_hourly_frontier6_epoch5_seedscan_seed1337`:
  https://wandb.ai/lee101p/stock/runs/crxovsbh
- `fastforecaster_hourly_frontier6_epoch5_seedscan_seed1701`:
  https://wandb.ai/lee101p/stock/runs/hinktiju
- `fastforecaster_hourly_frontier6_epoch5_seedscan_seed2026`:
  https://wandb.ai/lee101p/stock/runs/jnfkfubh
- `fastforecaster_hourly_frontier6_epoch5_seedscan_seed3333`:
  https://wandb.ai/lee101p/stock/runs/max66f8k
- `fastforecaster_hourly_frontier7_seed1701_lr0p00024_do0p05_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/wauzepos
- `fastforecaster_hourly_frontier7_seed1701_lr0p00026_do0p05_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/5swtnkrk
- `fastforecaster_hourly_frontier7_seed1701_lr0p00025_do0p04_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/l53xcazp
- `fastforecaster_hourly_frontier7_seed1701_lr0p00025_do0p05_wd0p009_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/cxwohr8c
- `fastforecaster_hourly_frontier7_seed1701_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/b5p6iwyz
- `fastforecaster_hourly_frontier7_seed1701_lr0p00025_do0p05_wd0p010_hwp0p30_rlw0p15`:
  https://wandb.ai/lee101p/stock/runs/6jlt5lfa
- `fastforecaster_hourly_frontier7_hwp028_seedscan_seed1337`:
  https://wandb.ai/lee101p/stock/runs/wuhnksof
- `fastforecaster_hourly_frontier7_hwp028_seedscan_seed1701`:
  https://wandb.ai/lee101p/stock/runs/1o2z4b6o
- `fastforecaster_hourly_frontier7_hwp028_seedscan_seed2026`:
  https://wandb.ai/lee101p/stock/runs/8f0w105g
- `fastforecaster_hourly_frontier7_hwp028_seedscan_seed3333`:
  https://wandb.ai/lee101p/stock/runs/9u51otem
- `fastforecaster_hourly_frontier10_seed1701_repro_hwp0p28_nocpp`:
  https://wandb.ai/lee101p/stock/runs/ksnp4tzd
- `fastforecaster_hourly_frontier10_seed1701_repro_hwp0p28_cpp`:
  https://wandb.ai/lee101p/stock/runs/2wl1jjnm
- `fastforecaster_hourly_frontier11_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p00`:
  https://wandb.ai/lee101p/stock/runs/ibobigr4
- `fastforecaster_hourly_frontier11_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p10`:
  https://wandb.ai/lee101p/stock/runs/76tyfl6f
- `fastforecaster_hourly_frontier11_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/x6k0200p
- `fastforecaster_hourly_frontier11_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p28_rlw0p30`:
  https://wandb.ai/lee101p/stock/runs/y3hwthjh
- `fastforecaster_hourly_frontier12_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/ya3lc93e
- `fastforecaster_hourly_frontier12_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p35_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/anq4s24e
- `fastforecaster_hourly_frontier12_hwp030_seedscan_seed1337`:
  https://wandb.ai/lee101p/stock/runs/njgmpqjx
- `fastforecaster_hourly_frontier12_hwp030_seedscan_seed1701`:
  https://wandb.ai/lee101p/stock/runs/ma4zu0yn
- `fastforecaster_hourly_frontier12_hwp030_seedscan_seed2026`:
  https://wandb.ai/lee101p/stock/runs/7bky76kv
- `fastforecaster_hourly_frontier12_hwp030_seedscan_seed3333`:
  https://wandb.ai/lee101p/stock/runs/dppfnvx9
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p29_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/offz8dwv
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p00025_do0p05_wd0p010_hwp0p31_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/0z391616
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p000245_do0p05_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/tl2rwths
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p000255_do0p05_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/q5n382gy
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p00025_do0p05_wd0p009_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/kmw33wwv
- `fastforecaster_hourly_frontier13_seed1701_ep5_lr0p00025_do0p04_wd0p010_hwp0p30_rlw0p20`:
  https://wandb.ai/lee101p/stock/runs/cnpaz9di
- `fastforecaster_hourly_frontier14_seed1701_hd320_l8_ff4_h8_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/g9fv121p
- `fastforecaster_hourly_frontier14_seed1701_hd448_l8_ff4_h8_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/e5mz4zsn
- `fastforecaster_hourly_frontier14_seed1701_hd384_l6_ff4_h8_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/vyvbb1j5
- `fastforecaster_hourly_frontier14_seed1701_hd384_l8_ff3_h8_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/d058ztsk
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p06_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/obj10k09
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p07_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/f3kfvfn9
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p012_do0p05_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/g0ky8dx7
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p28`:
  https://wandb.ai/lee101p/stock/runs/9nszh08h
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/11fch0fq
- `fastforecaster_hourly_frontier15_seed1701_hd384_l6_ff4_h8_lr0p00023_wd0p010_do0p05_hwp0p30`:
  https://wandb.ai/lee101p/stock/runs/820b2kr6
- `fastforecaster_hourly_frontier15_hwp026_seedscan_seed1337`:
  https://wandb.ai/lee101p/stock/runs/w5novn9h
- `fastforecaster_hourly_frontier15_hwp026_seedscan_seed1701`:
  https://wandb.ai/lee101p/stock/runs/4jp0xlo1
- `fastforecaster_hourly_frontier15_hwp026_seedscan_seed2026`:
  https://wandb.ai/lee101p/stock/runs/pj2dr3ki
- `fastforecaster_hourly_frontier15_hwp026_seedscan_seed3333`:
  https://wandb.ai/lee101p/stock/runs/bzrfqr0v
- `fastforecaster_hourly_frontier16_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p24`:
  https://wandb.ai/lee101p/stock/runs/lmzsjtdl
- `fastforecaster_hourly_frontier16_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p25`:
  https://wandb.ai/lee101p/stock/runs/hf85dt1h
- `fastforecaster_hourly_frontier16_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_hwp0p27`:
  https://wandb.ai/lee101p/stock/runs/slz8r64v
- `fastforecaster_hourly_frontier16_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p07_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/nr9aofy4
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p008_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/gge9vj66
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p009_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/rt5e1ux0
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p011_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/5gih4pa7
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p013_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/81hd4y49
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00024_wd0p010_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/xyx3nop4
- `fastforecaster_hourly_frontier17_seed1701_hd384_l6_ff4_h8_lr0p00026_wd0p010_do0p05_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/7ht05w53
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p15_dlw0p00_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/2ymzahlg
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p18_dlw0p00_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/5xsivyir
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p22_dlw0p00_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/1bm5y842
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p25_dlw0p00_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/msb1hn41
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p20_dlw0p01_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/eajj6bbr
- `fastforecaster_hourly_frontier18_seed1701_hd384_l6_ff4_h8_rlw0p20_dlw0p02_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/wwjlsn0j
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_rlw0p20_dlw0p01_hwp0p255`:
  https://wandb.ai/lee101p/stock/runs/c0bgv9a7
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p05_rlw0p20_dlw0p01_hwp0p265`:
  https://wandb.ai/lee101p/stock/runs/o0z61q5o
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p000245_wd0p010_do0p05_rlw0p20_dlw0p01_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/yxvabw6w
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p000255_wd0p010_do0p05_rlw0p20_dlw0p01_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/0inw0qjw
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p0095_do0p05_rlw0p20_dlw0p01_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/0d73mqsw
- `fastforecaster_hourly_frontier19_seed1701_hd384_l6_ff4_h8_lr0p00025_wd0p010_do0p045_rlw0p20_dlw0p01_hwp0p26`:
  https://wandb.ai/lee101p/stock/runs/je1qxm9v

## Comparison files

- `FastForecaster/experiments/comparison_baseline_v1_vs_enhanced_v2.json`
- `FastForecaster/experiments/comparison_contaminated_vs_cleansym.json`
- `FastForecaster/experiments/comparison_clean_noqk_vs_qk.json`
- `FastForecaster/experiments/comparison_contaminated_vs_clean_qk.json`
- `FastForecaster/experiments/comparison_hourly_bestcfg_seed_scan_20260304.json`
- `FastForecaster/experiments/comparison_prev_best_vs_seed1701_20260304.json`
- `FastForecaster/experiments/comparison_hourly_seed1701_refinement_20260304.json`
- `FastForecaster/experiments/comparison_prev_best_vs_refine_best_20260304.json`
- `FastForecaster/experiments/comparison_frontier6_20260304.json`
- `FastForecaster/experiments/comparison_frontier6_epoch5_seedscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier7_20260304.json`
- `FastForecaster/experiments/comparison_frontier7_hwp028_seedscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier10_cpp_ablation_20260304.json`
- `FastForecaster/experiments/comparison_frontier11_rlw_scan_20260304.json`
- `FastForecaster/experiments/comparison_frontier12_hwp_scan_20260304.json`
- `FastForecaster/experiments/comparison_frontier12_hwp030_seedscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier13_hwp030_localrefine_20260304.json`
- `FastForecaster/experiments/comparison_frontier14_architecture_scan_20260304.json`
- `FastForecaster/experiments/comparison_frontier15_arch6_regscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier15_hwp026_seedscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier16_hwp_localscan_20260304.json`
- `FastForecaster/experiments/comparison_frontier17_wdlr_scan_20260304.json`
- `FastForecaster/experiments/comparison_frontier18_objective_scan_20260304.json`
- `FastForecaster/experiments/comparison_frontier19_micro_scan_20260304.json`
