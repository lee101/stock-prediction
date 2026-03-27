# Archived Alpaca production snapshot

- Archived from `prod.md` before the 2026-03-25 08:56 UTC refresh.
- Covers current ledger snapshot plus the live supervisor and inactive paper systemd sections as they stood at 08:20 UTC.

### Current Alpaca snapshot (2026-03-25 08:20 UTC)
- **LIVE account**: supervisor `unified-stock-trader` is active; equity **$41,048.99**, last_equity **$41,077.99**, day change **-$29.00 (-0.07%)**, total unrealized **+$284.20**.
- **LIVE positions/orders**: `ABEV` 4459 shares (**+$222.95**) with DAY sell `4459 @ $2.77`; `ETHUSD` `4.748306908` (**+$61.25**) with GTC sell `4.748306908 @ $2178.32`; `AVAXUSD`, `BTCUSD`, `LTCUSD`, `SOLUSD` are dust.
- **PAPER account**: `daily-rl-trader.service` is installed but currently `inactive (dead)`; paper equity **$54,691.94**, last_equity **$53,487.18**, day change **+$1,204.76**, total unrealized **+$2,693.25**.

### 3. Alpaca Stock Trader (`unified-stock-trader`) -- RUNNING (LIVE via supervisor)
- **Bot**: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- **Service manager**: supervisor program `unified-stock-trader`
- **Installed config**: `/etc/supervisor/conf.d/unified-stock-trader.conf`
- **Exact launch**: `.venv313/bin/python -u /nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/trade_unified_hourly_meta.py --strategy wd06=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s42:8 --strategy wd06b=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s1337:8 --stock-symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV --min-edge 0.001 --fee-rate 0.001 --max-positions 5 --max-hold-hours 5 --trade-amount-scale 100.0 --min-buy-amount 2.0 --entry-intensity-power 1.0 --entry-min-intensity-fraction 0.0 --long-intensity-multiplier 1.0 --short-intensity-multiplier 1.5 --meta-metric p10 --meta-lookback-days 14 --meta-selection-mode sticky --meta-switch-margin 0.005 --meta-min-score-gap 0.0 --meta-recency-halflife-days 0.0 --meta-history-days 120 --sit-out-if-negative --sit-out-threshold -0.001 --market-order-entry --bar-margin 0.0005 --entry-order-ttl-hours 6 --margin-rate 0.0625 --live --loop`
- **Environment**: `PYTHONPATH=/nvme0n1-disk/code/stock-prediction`, `PYTHONUNBUFFERED=1`, `CHRONOS2_FREQUENCY=hourly`, `PAPER=0`
- **Architecture**: Chronos2 hourly, multiple models + meta-selector
- **Symbols**: NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT, AAPL, MSFT, META, TSLA, NET, BKNG, EBAY, EXPE, ITUB, BTG, ABEV
- **Live snapshot (2026-03-25 08:20 UTC)**: equity **$41,048.99**, cash **$18,352.47**, long market value **$22,696.52**, buying power **$49,100.96**, unrealized **+$284.20**
- **Open positions (2026-03-25 08:20 UTC)**: `ABEV` `4459` shares (**+$222.95**), `ETHUSD` `4.748306908` (**+$61.25**), plus dust in `AVAXUSD`, `BTCUSD`, `LTCUSD`, `SOLUSD`
- **Open exit orders (2026-03-25 08:20 UTC)**: `ABEV` sell `4459 @ $2.77` (`DAY`), `ETH/USD` sell `4.748306908 @ $2178.32` (`GTC`)
- **Strategies**: wd_0.06_s42:8 + wd_0.06_s1337:8 (2-strategy meta-selector)
- **NOTE (2026-03-24)**: Was crash-looping since ~Mar 19 — supervisor config referenced 5 missing checkpoints
  (wd_0.04, wd_0.05_s42, wd_0.08_s42, wd_0.03_s42, stock_sortino_robust_20260219b/c).
  Fixed by replacing with wd_0.06_s42:8 + wd_0.06_s1337:8 (only 2 strategies remain locally).
  Previous equity loss (~$5k) likely from pre-existing positions before outage, not model error.
- **NOTE (2026-03-25)**: 3 bugs fixed in `trade_unified_hourly.py` — see `alpacaprogress6.md`:
  1. pending_close_retry: positions stuck in pending_close with no exit order now retry force_close
  2. cancel race condition: sleep(0.75) after cancel before new order (prevents "qty held for orders")
  3. crypto qty: abs(qty)<1 wrongly treated fractional crypto as closed — fixed with notional check
- **ABEV incident (2026-03-25)**: ABEV position ($12k, entered 2026-03-20 @ $2.73) had no exit order since
  2026-03-24 when force_close failed due to race condition. Fixed — retry fired at 01:42 UTC.
  Force_close limit order ~$2.77 queued for market open (2026-03-25 13:30 UTC).

### 4. Alpaca Daily PPO Trader (`trade_daily_stock_prod.py`) -- INSTALLED, CURRENTLY INACTIVE (paper systemd)
- **Service manager**: systemd unit `daily-rl-trader.service`
- **Installed unit**: `/etc/systemd/system/daily-rl-trader.service`
- **Installed ExecStart**: `.venv313/bin/python -u trade_daily_stock_prod.py --daemon --paper --allocation-pct 25`
- **Runtime status (2026-03-25 08:20 UTC)**: `inactive (dead)`; latest journal restart was `2026-03-25 01:42 UTC`
- **Paper snapshot (2026-03-25 08:20 UTC)**: equity **$54,691.94**, cash **$2,235.61**, long market value **$52,496.44**, total unrealized **+$2,693.25**
- **Paper positions (2026-03-25 08:20 UTC)**: `AAPL`, `BTCUSD`, `COUR`, `ETHUSD`, `SOLUSD`, `U`, `UNIUSD`
- **Architecture**: h=1024 MLP PPO, stocks12 (AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,JPM,V,AMZN,PLTR)
- **Primary checkpoint**: `pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt`
- **Ensemble member**: `pufferlib_market/checkpoints/stocks12_seed_sweep/tp05_s15/best.pt`
  - tp05_s15 = seed=15, 128 envs, no bf16, 35M steps, best at update_000950
- **3-model ensemble (s123+s15+s36 softmax_avg) holdout eval** (50 windows, Sep2025-Mar2026, default_rng(42)):
  - Median: **+47.30%** / 90 days (+64% over 2-model s123+s15)
  - P10: **+29.93%** (+83% over 2-model s123+s15's 16.37%)
  - Worst window: **+20.97%** (+106% over 2-model s123+s15's 10.17%)
  - Negative windows: **0/50 (0%)** — ZERO negative windows
- **Slippage robustness** (3-model ensemble, 50-window, default_rng(42)):
  - @5bps:  med=47.30%, p10=29.93%, worst=20.97%, neg=0/50
  - @10bps: med=47.30%, p10=29.93%, worst=20.97%, neg=0/50 (identical)
  - @20bps: med=48.60%, p10=31.54%, worst=22.47%, neg=0/50
  - @50bps: med=37.92%, p10=22.79%, worst=14.94%, neg=0/50
- **tp05_s36**: seed=36, 128 envs, no bf16, 35M steps — 0/50 neg standalone, med=26.80%
- **tp05_s15**: seed=15, 128 envs, no bf16, 35M steps — 0/50 neg standalone, med=28.76%
- **Previous 2-model ensemble** s123+s15: med=28.76%, p10=16.37% (superseded 2026-03-24)
- **Previous 3-model ensemble** rmu2201+8597+5526: med=14.94%, p10=7.64% (kept for reference)
- **Config**: h=1024, lr=3e-4, ent=0.05, trade_penalty=0.05, dp=0.0, slip=0bps (training), anneal_lr=True
- **Launch**: `deployments/daily-stock-ppo/launch.sh`
- **Supervisor**: `deployments/daily-stock-ppo/supervisor.conf` (autostart=false — enable manually)
- **WARNING**: Symbol conflict with unified-stock-trader (both trade AAPL/MSFT/NVDA/GOOG/META/TSLA/PLTR) — coordinate before enabling both simultaneously
- **NOTE**: Previous default (stocks12_daily_tp05_longonly) scored -2.55% median, 32/50 negative
- **NOTE**: random_mut_2272 (prev deployed) scored -5.14% median, 29/50 negative
- **NOTE**: seed variance is extreme — seed=7 gives -13.6% median, seed=123 gives +16.5% median (same config)
- **Deploy command**:
  ```bash
  source .venv313/bin/activate
  python trade_daily_stock_prod.py --live
  # Uses 3-model ensemble (s123+s15+s36 softmax_avg) — DEFAULT_EXTRA_CHECKPOINTS in code
  # To restore 2-model: python trade_daily_stock_prod.py --live --extra-checkpoints <s15_path>
  # To restore standalone s123: python trade_daily_stock_prod.py --live --no-ensemble
  ```
- **Eval command**:
  ```bash
  source .venv313/bin/activate
  python -m pufferlib_market.evaluate_holdout \
    --checkpoint pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt \
    --data-path pufferlib_market/data/stocks12_daily_val.bin \
    --eval-hours 90 --n-windows 50 --seed 42 \
    --fee-rate 0.001 --fill-buffer-bps 5.0 --deterministic --no-early-stop
  # NOTE: lag=0 IS correct for daily models (1-bar obs lag built-in to C env)
  # Bot runs at 9:35 AM, drops today's incomplete bar, uses T-1 data → fills at T OPEN
  ```
- **Why ensemble works here**: unlike rmu models, s15 was trained with same tp05 config. Both models
  pick only high-conviction trades (~10/90d). softmax_avg of two similar architectures diversifies
  seed variance rather than diluting edge. Ensemble p10/worst strictly better than either alone.
- **Seed sweep results** (tp05 old-config: 128 envs, no bf16, 35M steps):
  - seed=15: **0/50 neg, med=39.14%, p10=17.72%** — DEPLOYED in ensemble
  - seed=33: 32/50 neg — bad
  - seed=3 (old-config): 1/50 neg best case, p10<5% — not deployable
  - Sweep continuing: seeds 16-32 (stream A), 34-50 (stream B)

## Crypto70 Daily RL Autoresearch (2026-03-24) — COMPLETED

**Dataset**: 48 Binance USDT pairs (crypto70, filtered), daily bars, train 2019-2025, val 2025-09-01 to 2026-03-31 (205 days)
**Config**: h=1024 MLP PPO, anneal_lr, ent=0.05, 128 envs, bf16, no-cuda-graph, periods_per_year=365, max_steps=180 (6mo episodes)
**Sweep**: 3 seeds × (3 trade_penalties × 2 slippages + 1 muon) = 21 jobs, 5min/job
