# Binance Trading Runbook

Updated: 2026-02-02

## Overview

This runbook covers setting up and running live trading on Binance.com or Binance.US using the `binanceexp1` trading system.

**⚠️ Important Notes:**
- Binance.com is geo-blocked from US IPs. Use Binance.US keys if trading from the US, or run on a non-US host.
- Always test with small amounts first (use `--skip-order` flag for dry runs).
- Monitor logs closely during the first 24 hours of live trading.

## Best SOLUSD model (marketsimulator)

- Checkpoint: `binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt`
- Best PnL config: `intensity-scale=20.0`, `horizon=24`, `sequence-length=96`, `min-gap-pct=0.0003`
- Performance: total_return 17.7449, sortino 66.1148

## Prerequisites

1. **Binance Account & API Keys**
   - Create account on [Binance.com](https://binance.com) or [Binance.US](https://binance.us)
   - Generate API keys with spot trading permissions
   - Whitelist your server IP in Binance security settings (recommended)
   - Enable "Enable Spot & Margin Trading" on the API key

2. **Server Requirements**
   - Python 3.13 (or 3.12)
   - 4GB+ RAM
   - Stable internet connection
   - Non-US IP for Binance.com (or use Binance.US)

3. **Environment Variables**
   ```bash
   export BINANCE_API_KEY="your_api_key_here"
   export BINANCE_SECRET="your_secret_key_here"
   
   # For Binance.US users:
   export BINANCE_TLD="us"
   
   # Optional for testnet:
   export BINANCE_TESTNET=1
   ```

## Clone + Setup (inference host)

```bash
# Clone repository
git clone <REPO_URL> stock-prediction
cd stock-prediction

# Create Python 3.13 virtual environment
python3.13 -m venv .venv313
source .venv313/bin/activate

# Install dependencies
uv pip install -e .
uv pip install -e toto/
```

## Sync Assets from R2

Download required training data and model checkpoints:

```bash
# Set R2 endpoint (should already be configured)
export R2_ENDPOINT="<your_r2_endpoint>"

# Sync training data (hourly Binance data)
aws s3 sync s3://models/stock/trainingdata/ trainingdata/ --endpoint-url "$R2_ENDPOINT"

# Sync model checkpoints
aws s3 sync s3://models/stock/models/binance/binanceneural/checkpoints/ binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"

# Optional: sync forecast cache to avoid rebuilding Chronos forecasts
aws s3 sync s3://models/stock/models/binance/binanceneural/forecast_cache/ binanceneural/forecast_cache/ --endpoint-url "$R2_ENDPOINT"
```

Verify checkpoint exists:
```bash
ls -lh binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt
```

Optional (Chronos2 + hyperparams):
```bash
aws s3 sync s3://models/stock/models/binance/chronos2_finetuned/ chronos2_finetuned/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/hyperparams/chronos2/ hyperparams/chronos2/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/preaugstrategies/chronos2/hourly/ preaugstrategies/chronos2/hourly/ --endpoint-url "$R2_ENDPOINT"
```

Note: if you want to avoid rebuilding Chronos forecasts on the inference host, sync `binanceneural/forecast_cache/` as well (if present).

Minimal SOLUSD-only sync (optional):

```bash
aws s3 sync s3://models/stock/models/binance/binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/ binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/ --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/trainingdatahourly/crypto/ trainingdatahourly/crypto/ --exclude "*" --include "SOLUSD.csv" --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/hyperparams/chronos2/hourly/ hyperparams/chronos2/hourly/ --exclude "*" --include "SOLUSD.json" --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/preaugstrategies/chronos2/hourly/ preaugstrategies/chronos2/hourly/ --exclude "*" --include "SOLUSDT.json" --endpoint-url "$R2_ENDPOINT"
aws s3 sync s3://models/stock/models/binance/chronos2_finetuned/ chronos2_finetuned/ --exclude "*" --include "SOLUSDT_lora_*/*" --endpoint-url "$R2_ENDPOINT"
```
## Inference-only (no live orders)

For testing predictions without placing real orders:

```bash
source .venv313/bin/activate
PYTHONPATH=/path/to/stock-prediction \
python -m binanceexp1.trade_binance_hourly \
  --symbols SOLUSD \
  --checkpoint binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt \
  --horizon 24 \
  --sequence-length 96 \
  --intensity-scale 20.0 \
  --min-gap-pct 0.0003 \
  --once \
  --dry-run
```

Optional: add `--cache-only` if you have a populated `binanceneural/forecast_cache/`.

## Testing API Connection

Before running live trading, verify your Binance API credentials work:

```bash
source .venv313/bin/activate

# Test 1: Fetch account balance
python -c "from src.binan import binance_wrapper; print(binance_wrapper.get_account_balances())"

# Test 2: Get SOLUSD price
python -c "from src.binan import binance_wrapper; print(binance_wrapper.get_symbol_price('SOLUSDT'))"

# Test 3: Check account value in USDT
python -c "from src.binan import binance_wrapper; import json; print(json.dumps(binance_wrapper.get_account_value_usdt(), indent=2))"
```

### Smoke Test (place + cancel tiny order)

This will place a tiny limit order far from market price and immediately cancel it:

```bash
source .venv313/bin/activate

# For selling (requires SOL balance):
python scripts/binance_api_smoketest.py \
  --symbol SOLUSDT \
  --side sell \
  --notional 1 \
  --price-multiplier 1.15

# For buying (requires USDT balance):
python scripts/binance_api_smoketest.py \
  --symbol SOLUSDT \
  --side buy \
  --notional 1 \
  --price-multiplier 0.85

# Skip order placement (data fetch only):
python scripts/binance_api_smoketest.py \
  --symbol SOLUSDT \
  --skip-order
```

If any test fails:
- Check API key permissions (need "Enable Spot & Margin Trading")
- Verify IP whitelist settings on Binance
- For Binance.US users: ensure `BINANCE_TLD=us` is set
- Check error messages for "API-key format invalid" (wrong region) or "Signature invalid" (wrong secret)

## Running Live Trading

### Option 1: Manual Run (foreground, with logging)

**Step 1:** Set environment variables
```bash
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_SECRET="your_secret_key_here"

# For Binance.US:
export BINANCE_TLD="us"
```

**Step 2:** Activate environment and run
```bash
source .venv313/bin/activate

python -m binanceexp1.trade_binance_hourly \
  --symbols SOLUSD \
  --checkpoint binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt \
  --horizon 24 \
  --sequence-length 96 \
  --intensity-scale 20.0 \
  --min-gap-pct 0.0003 \
  --poll-seconds 30 \
  --expiry-minutes 90 \
  --log-metrics \
  --metrics-log-path strategy_state/binanceexp1-solusd-metrics.csv
```

**Key Parameters:**
- `--symbols SOLUSD`: Trading pair (internally maps to SOLUSDT)
- `--checkpoint`: Path to trained model
- `--intensity-scale 20.0`: Position sizing multiplier (higher = more aggressive)
- `--min-gap-pct 0.0003`: Minimum price movement (0.03%) to trigger trades
- `--poll-seconds 30`: How often to check for trading opportunities
- `--expiry-minutes 90`: Cancel unfilled orders after this many minutes
- `--log-metrics`: Enable metrics logging
- `--metrics-log-path`: CSV file for trade metrics

### Option 2: Quick Start Script

Use the provided convenience script:

```bash
source .venv313/bin/activate
bash scripts/run_binanceexp1_prod_solusd.sh
```

This runs the same command as manual mode but is easier to restart.

### Option 3: Background Run (tmux/screen)

For persistent sessions that survive disconnections:

```bash
# Using tmux:
tmux new -s binance-trader
source .venv313/bin/activate
bash scripts/run_binanceexp1_prod_solusd.sh
# Press Ctrl+B, then D to detach

# Reattach later:
tmux attach -t binance-trader

# Using screen:
screen -S binance-trader
source .venv313/bin/activate
bash scripts/run_binanceexp1_prod_solusd.sh
# Press Ctrl+A, then D to detach

# Reattach later:
screen -r binance-trader
```

## Option 4: Production Run with Supervisor

For production deployments with automatic restarts and log rotation:

**Step 1:** Copy and edit supervisor config
```bash
# Copy template
cp supervisor/binanceexp1-solusd.conf /etc/supervisor/conf.d/

# Edit paths to match your setup:
sudo nano /etc/supervisor/conf.d/binanceexp1-solusd.conf
```

**Step 2:** Update the config file with your paths:
```ini
[program:binanceexp1-solusd]
command=/path/to/stock-prediction/.venv313/bin/python -m binanceexp1.trade_binance_hourly --symbols SOLUSD --checkpoint /path/to/stock-prediction/binanceneural/checkpoints/binanceexp1_regime_solusd_backfill/epoch_005.pt --horizon 24 --sequence-length 96 --intensity-scale 20.0 --min-gap-pct 0.0003 --poll-seconds 30 --expiry-minutes 90 --log-metrics --metrics-log-path /path/to/stock-prediction/strategy_state/binanceexp1-solusd-metrics.csv
directory=/path/to/stock-prediction
user=yourusername
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/binanceexp1-solusd-error.log
stdout_logfile=/var/log/supervisor/binanceexp1-solusd.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
environment=HOME="/home/yourusername",PYTHONPATH="/path/to/stock-prediction",BINANCE_API_KEY="your_key",BINANCE_SECRET="your_secret",BINANCE_TLD="us"
```

**Important:** Add your actual API credentials to the `environment` line, or use a `.env` file approach.

**Step 3:** Start with supervisor
```bash
# Reload supervisor config
sudo supervisorctl reread
sudo supervisorctl update

# Start the bot
sudo supervisorctl start binanceexp1-solusd

# Check status
sudo supervisorctl status binanceexp1-solusd

# View logs
sudo supervisorctl tail -f binanceexp1-solusd stdout
sudo supervisorctl tail -f binanceexp1-solusd stderr
```

**Supervisor Commands:**
```bash
# Stop trading
sudo supervisorctl stop binanceexp1-solusd

# Restart trading
sudo supervisorctl restart binanceexp1-solusd

# View real-time logs
sudo tail -f /var/log/supervisor/binanceexp1-solusd.log
sudo tail -f /var/log/supervisor/binanceexp1-solusd-error.log
```

## Monitoring Your Trading Bot

### Watch Live Output

If running in foreground or tmux/screen:
```bash
# Output shows:
# - Current price predictions
# - Trading signals (BUY/SELL)
# - Order placements and fills
# - Position updates
# - PnL tracking
```

### Check Metrics Log

The bot writes trade metrics to CSV:
```bash
# View latest entries
tail -20 strategy_state/binanceexp1-solusd-metrics.csv

# Monitor in real-time
tail -f strategy_state/binanceexp1-solusd-metrics.csv

# Count trades today
grep "$(date +%Y-%m-%d)" strategy_state/binanceexp1-solusd-metrics.csv | wc -l
```

### Check Open Orders

```bash
python -c "from src.binan import binance_wrapper; import json; orders = binance_wrapper.get_all_orders('SOLUSDT'); print(json.dumps([o for o in orders if o['status'] not in ['CANCELED', 'FILLED']], indent=2))"
```

### Check Current Positions & Balance

```bash
# Total account value
python -c "from src.binan import binance_wrapper; import json; print(json.dumps(binance_wrapper.get_account_value_usdt(), indent=2))"

# Specific asset balances
python -c "from src.binan import binance_wrapper; print(f\"SOL: {binance_wrapper.get_asset_free_balance('SOL'):.6f}\"); print(f\"USDT: {binance_wrapper.get_asset_free_balance('USDT'):.2f}\")"
```

### Alert Setup (Optional)

Monitor for crashes and restart:
```bash
# Simple cron job to check if process is running (add to crontab)
*/5 * * * * pgrep -f "trade_binance_hourly" || /path/to/restart_script.sh
```

For supervisor, this is automatic via `autorestart=true`.

## Troubleshooting

### Common Issues

**1. "API-key format invalid"**
- Solution: Using wrong region. Set `BINANCE_TLD=us` for Binance.US or remove it for Binance.com
- Check: Your API keys match the platform (US vs international)

**2. "Signature for this request is not valid"**
- Solution: Check `BINANCE_SECRET` is correct
- Check: Server time is synchronized (`timedatectl status`)
- Fix: `sudo ntpdate -s time.nist.gov` or `sudo timedatectl set-ntp true`

**3. "Timestamp for this request is outside of the recvWindow"**
- Solution: Server clock is out of sync
- Fix: Sync time with NTP (see above)

**4. "Insufficient balance"**
- Check current balance: Run balance check command above
- Ensure you have both SOL and USDT for trading
- Min notional requirement: ~$5-10 USD per trade

**5. "IP address not whitelisted"**
- Solution: Add your server IP to Binance API whitelist
- Or: Remove IP restriction (less secure)

**6. Orders not filling**
- Check `--intensity-scale` may be too conservative
- Check `--min-gap-pct` threshold may be too high
- Review market conditions (low volume periods)
- Monitor with: `binance_wrapper.get_all_orders('SOLUSDT')`

**7. High frequency of orders**
- Reduce `--intensity-scale` (lower = less aggressive)
- Increase `--min-gap-pct` (higher = fewer trades)
- Increase `--poll-seconds` (less frequent checks)

### Emergency Stop

```bash
# If using supervisor:
sudo supervisorctl stop binanceexp1-solusd

# If using tmux/screen/manual:
# Find process ID
ps aux | grep trade_binance_hourly
# Kill it
kill <PID>

# Cancel all open orders immediately:
python -c "from src.binan import binance_wrapper; binance_wrapper.cancel_all_orders()"

# Close position at market (if needed):
# Sell all SOL:
# python -c "from src.binan import binance_wrapper; binance_wrapper.create_all_in_order('SOLUSDT', 'SELL', <current_price>)"
```

### Log Analysis

```bash
# Find errors in logs
grep -i "error\|exception\|failed" /var/log/supervisor/binanceexp1-solusd-error.log | tail -20

# Check trade execution
grep -i "order\|trade\|fill" /var/log/supervisor/binanceexp1-solusd.log | tail -20

# Monitor PnL updates
grep -i "pnl\|profit\|return" /var/log/supervisor/binanceexp1-solusd.log | tail -20
```

## Backup Strategy State

Before major changes, backup your metrics:
```bash
# Backup metrics log
cp strategy_state/binanceexp1-solusd-metrics.csv \
   strategy_state/binanceexp1-solusd-metrics-backup-$(date +%Y%m%d).csv

# Upload to R2
aws s3 cp strategy_state/binanceexp1-solusd-metrics.csv \
   s3://models/stock/strategy_state/ --endpoint-url "$R2_ENDPOINT"
```

## S3/R2 Sync Commands

Note: `R2_ENDPOINT` must be set in the environment.

- Training data:
```bash
aws s3 sync trainingdata/ s3://models/stock/trainingdata/ --endpoint-url "$R2_ENDPOINT"
```

- Market simulator logs:
```bash
aws s3 sync marketsimulator/run_logs/ s3://models/stock/marketsimulator/logs/manual-20251025/ --endpoint-url "$R2_ENDPOINT"
```

- Models (mirror local path under `/models/stock/models/binance/`):
```bash
aws s3 sync binanceneural/checkpoints/ s3://models/stock/models/binance/binanceneural/checkpoints/ --endpoint-url "$R2_ENDPOINT"
```

## Best Practices

### Risk Management

1. **Start Small**
   - Begin with minimum position sizes
   - Test for 24-48 hours before increasing capital
   - Monitor every trade during the first day

2. **Position Sizing**
   - `--intensity-scale 20.0` is aggressive (tested in simulation)
   - Consider starting at 5.0-10.0 for live trading
   - Scale up gradually as confidence builds

3. **Monitoring Schedule**
   - First 24 hours: Check every 2-3 hours
   - First week: Check 2-3 times daily
   - After stable: Daily review of metrics

4. **Capital Allocation**
   - Keep majority of funds in USDT
   - Maintain buffer for margin calls
   - Don't use more than you can afford to lose

### Security

1. **API Key Best Practices**
   - Use separate API keys for trading vs monitoring
   - Enable IP whitelisting
   - Disable withdrawal permissions (trade-only)
   - Rotate keys periodically (monthly)

2. **Server Security**
   - Use firewall (allow only necessary ports)
   - Keep system updated (`apt update && apt upgrade`)
   - Use SSH keys instead of passwords
   - Monitor for unauthorized access

3. **Secret Management**
   - Never commit API keys to git
   - Use environment variables or secret management
   - Restrict file permissions: `chmod 600 .env`

### Performance Tuning

**Conservative Settings** (lower risk, fewer trades):
```bash
--intensity-scale 5.0 \
--min-gap-pct 0.001 \
--poll-seconds 60 \
--expiry-minutes 120
```

**Aggressive Settings** (higher risk, more trades):
```bash
--intensity-scale 20.0 \
--min-gap-pct 0.0001 \
--poll-seconds 30 \
--expiry-minutes 60
```

**Recommended Starting Point**:
```bash
--intensity-scale 10.0 \
--min-gap-pct 0.0005 \
--poll-seconds 30 \
--expiry-minutes 90
```

### Maintenance Schedule

**Daily:**
- Check bot is running (`supervisorctl status` or `tmux ls`)
- Review trades in metrics log
- Verify no errors in logs

**Weekly:**
- Analyze PnL and Sharpe ratio
- Check for any anomalies or unusual patterns
- Review open orders and cancel stale ones
- Backup metrics to R2

**Monthly:**
- Evaluate model performance vs benchmark
- Consider retraining with new data
- Rotate API keys
- Review and optimize parameters

## Notes

- **Region Restrictions**: Binance.com is geo-blocked from US IPs. Use a non-US host or Binance.US keys if running from the US.
- **Model Updates**: When the best model/config changes, update this file and the production command.
- **Market Conditions**: The bot was optimized on historical data. Performance may vary in different market regimes.
- **Trading Hours**: Crypto trades 24/7, but expect lower volume during US night hours.
- **Funding**: Binance has no funding rates for spot trading, but watch for sudden price moves during high volatility.

## Support & Resources

- **Binance API Docs**: https://binance-docs.github.io/apidocs/spot/en/
- **Binance.US API Docs**: https://docs.binance.us/
- **Model Training**: See `binanceprogress.md` for training details
- **Issue Tracking**: Check git history and open issues for known problems

## Quick Reference Card

```bash
# Start bot (manual)
source .venv313/bin/activate && bash scripts/run_binanceexp1_prod_solusd.sh

# Start bot (supervisor)
sudo supervisorctl start binanceexp1-solusd

# Check status
sudo supervisorctl status binanceexp1-solusd

# View logs
sudo supervisorctl tail -f binanceexp1-solusd

# Emergency stop
sudo supervisorctl stop binanceexp1-solusd

# Cancel all orders
python -c "from src.binan import binance_wrapper; binance_wrapper.cancel_all_orders()"

# Check balance
python -c "from src.binan import binance_wrapper; import json; print(json.dumps(binance_wrapper.get_account_value_usdt(), indent=2))"

# View recent metrics
tail -20 strategy_state/binanceexp1-solusd-metrics.csv
```
