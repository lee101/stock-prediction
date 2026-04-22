#!/bin/bash
# Crypto weekend-only LIVE trader — BTC+ETH+SOL trend-follow, Sat entry / Mon exit.
#
# Strategy (crypto_weekend/backtest.py): fri_close > SMA20 × 1.05 AND vol_20d ≤ 3%.
# OOS 2022-07 → 2026-04 (198 weekends, daily Binance bars, 10bps fee):
#   mean +0.21%/week (~+0.9%/mo), median 0, worst week −2.91%, max DD 7.44%,
#   neg-weekend rate 8.6%, trades only 20% of weekends (cash-heavy).
# Sample is thin (39 OOS trades). Live monitor essential — see kill-switch below.
#
# Windows (UTC):
#   BUY  : Saturday 00:00-23:00 (Friday daily bar fully closed)
#   SELL : Monday   00:00-12:30 (well before US stock open 13:30)
#   FLAT : Tue-Fri (no crypto positions held overnight midweek)
#
# Safety / isolation:
#   - ALPACA_ACCOUNT_NAME=alpaca_crypto_writer → separate fcntl lock
#     from the xgb-daily-trader-live stock bot (alpaca_live_writer).
#     The two daemons coexist safely on the same Alpaca account.
#   - Sizing caps at min(cash, equity × max_gross) so a concurrently-levered
#     stock bot does not cause us to over-lever crypto.
#   - close_position_violently() is used for sells; it bypasses the stock
#     bot's death-spiral guard, which is stock-specific (crypto can legitimately
#     move >5% over a weekend).
#
# Kill-switch criteria (flip autostart=false + stop the unit):
#   - Live median weekly PnL < 0 after first 10 trades
#   - Realized max DD > 10% at any point (OOS was 7.4%)
#   - Any buy > $100 over the computed reference price (Alpaca slippage)

set -euo pipefail

export HOME=/home/administrator
export USER=administrator

if [ -f "$HOME/.secretbashrc" ]; then
  # shellcheck disable=SC1090
  set +e
  set +u
  source "$HOME/.secretbashrc" >/dev/null 2>&1
  set -euo pipefail
fi

cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

export PYTHONPATH=/nvme0n1-disk/code/stock-prediction
export ALP_PAPER=0
export ALLOW_ALPACA_LIVE_TRADING=1
export ALPACA_ACCOUNT_NAME=alpaca_crypto_writer
export ALPACA_SERVICE_NAME=crypto-weekend-live

exec python -u -m crypto_weekend.live_trader \
  --poll-seconds 300 \
  --max-gross 0.5
