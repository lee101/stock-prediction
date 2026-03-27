# Archived Alpaca Production Snapshot

Archived from `prod.md` on 2026-03-26 04:22 UTC before refreshing the live Alpaca ledger.

## Previous current Alpaca snapshot (2026-03-25 08:56 UTC)

- LIVE account: supervisor `unified-stock-trader` active; equity `$41,081.47`, last_equity `$41,077.99`, day change `+$3.48 (+0.01%)`, total unrealized `+$222.95`.
- LIVE positions/orders: `ABEV` `4459` shares (`+$222.95`) with DAY sell `4459 @ $2.77`; `AVAXUSD`, `BTCUSD`, `ETHUSD`, `LTCUSD`, `SOLUSD` remained only as dust.
- PAPER account: `daily-rl-trader.service` installed but `inactive (dead)`; paper equity `$55,268.15`, last_equity `$54,078.69`, day change `+$1,189.46 (+2.20%)`, total unrealized `+$3,269.46`.

## Previous Alpaca trader section

- Bot: `unified_hourly_experiment/trade_unified_hourly_meta.py`
- Service manager: supervisor program `unified-stock-trader`
- Installed config: `/etc/supervisor/conf.d/unified-stock-trader.conf`
- Exact launch: `.venv313/bin/python -u /nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/trade_unified_hourly_meta.py --strategy wd06=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s42:8 --strategy wd06b=/nvme0n1-disk/code/stock-prediction/unified_hourly_experiment/checkpoints/wd_0.06_s1337:8 --stock-symbols NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV --min-edge 0.001 --fee-rate 0.001 --max-positions 5 --max-hold-hours 5 --trade-amount-scale 100.0 --min-buy-amount 2.0 --entry-intensity-power 1.0 --entry-min-intensity-fraction 0.0 --long-intensity-multiplier 1.0 --short-intensity-multiplier 1.5 --meta-metric p10 --meta-lookback-days 14 --meta-selection-mode sticky --meta-switch-margin 0.005 --meta-min-score-gap 0.0 --meta-recency-halflife-days 0.0 --meta-history-days 120 --sit-out-if-negative --sit-out-threshold -0.001 --market-order-entry --bar-margin 0.0005 --entry-order-ttl-hours 6 --margin-rate 0.0625 --live --loop`
- Environment: `PYTHONPATH=/nvme0n1-disk/code/stock-prediction`, `PYTHONUNBUFFERED=1`, `CHRONOS2_FREQUENCY=hourly`, `PAPER=0`
- Architecture: Chronos2 hourly, multiple models + meta-selector
- Symbols: `NVDA, PLTR, GOOG, DBX, TRIP, MTCH, NYT, AAPL, MSFT, META, TSLA, NET, BKNG, EBAY, EXPE, ITUB, BTG, ABEV`
- Live snapshot: equity `$41,081.47`, cash `$28,685.45`, long market value `$12,396.02`, buying power `$69,766.92`, unrealized `+$222.95`
- Open positions: `ABEV` `4459` shares, plus dust in `AVAXUSD`, `BTCUSD`, `ETHUSD`, `LTCUSD`, `SOLUSD`
- Open exit orders: `ABEV` sell `4459 @ $2.77` (`DAY`)
- Strategies: `wd_0.06_s42:8` + `wd_0.06_s1337:8`
