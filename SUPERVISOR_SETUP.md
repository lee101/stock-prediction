# Supervisor Setup for Binance Trading Bot

This guide helps you set up the Binance trading bot with supervisor for production deployment.

## Why Supervisor?

✅ **Auto-restart** if bot crashes  
✅ **Auto-start** on system reboot  
✅ **Centralized logging**  
✅ **Easy process management**  
✅ **Better than tmux** for production  

## Quick Setup

```bash
# Run the setup script (will prompt for sudo)
./scripts/setup_supervisor.sh
```

That's it! The bot will now run under supervisor.

## Manual Setup

If you prefer to set it up manually:

```bash
# 1. Copy config to supervisor directory
sudo cp supervisor/binanceexp1-solusd.conf /etc/supervisor/conf.d/

# 2. Reload supervisor
sudo supervisorctl reread

# 3. Add the program
sudo supervisorctl update

# 4. Check status
sudo supervisorctl status binanceexp1-solusd
```

## Management Commands

### Check Status
```bash
sudo supervisorctl status binanceexp1-solusd
```

### View Live Logs
```bash
sudo supervisorctl tail -f binanceexp1-solusd stdout
```

### Restart Bot
```bash
sudo supervisorctl restart binanceexp1-solusd
```

### Stop Bot
```bash
sudo supervisorctl stop binanceexp1-solusd
```

### Start Bot
```bash
sudo supervisorctl start binanceexp1-solusd
```

### View Error Logs
```bash
sudo tail -f /var/log/supervisor/binanceexp1-solusd-error.log
```

### View Output Logs
```bash
sudo tail -f /var/log/supervisor/binanceexp1-solusd.log
```

## Configuration File

The supervisor config is located at:
- **Repository:** `supervisor/binanceexp1-solusd.conf`
- **Installed:** `/etc/supervisor/conf.d/binanceexp1-solusd.conf`

### Key Settings

```ini
[program:binanceexp1-solusd]
command=/home/lee/code/stock/.venv313/bin/python -m binanceexp1.trade_binance_hourly ...
directory=/home/lee/code/stock
user=lee
autostart=true       # Start on boot
autorestart=true     # Restart if crashes
```

### Updating Configuration

After editing `supervisor/binanceexp1-solusd.conf`:

```bash
# Copy updated config
sudo cp supervisor/binanceexp1-solusd.conf /etc/supervisor/conf.d/

# Reload
sudo supervisorctl reread
sudo supervisorctl update

# Restart to apply changes
sudo supervisorctl restart binanceexp1-solusd
```

## Logs

All logs are stored in `/var/log/supervisor/`:

| File | Description |
|------|-------------|
| `binanceexp1-solusd.log` | Standard output (predictions, trades, status) |
| `binanceexp1-solusd-error.log` | Errors and exceptions |

Logs rotate automatically:
- Max size: 10MB per file
- Backups: 3 files kept

## Troubleshooting

### Bot not starting?

Check logs for errors:
```bash
sudo tail -50 /var/log/supervisor/binanceexp1-solusd-error.log
```

### Check if supervisor is running:
```bash
sudo systemctl status supervisor
```

### Restart supervisor:
```bash
sudo systemctl restart supervisor
```

### View all supervisor programs:
```bash
sudo supervisorctl status
```

## Monitoring

Monitor the bot's performance:

```bash
# Watch metrics in real-time
tail -f strategy_state/binanceexp1-solusd-metrics.csv

# Check account balance
source .venv313/bin/activate
python -c "from src.binan import binance_wrapper; import json; \
print(json.dumps(binance_wrapper.get_account_value_usdt(), indent=2))"

# View recent log output
sudo supervisorctl tail binanceexp1-solusd
```

## Switching from tmux

If you're currently running the bot in tmux:

```bash
# Stop tmux session
./scripts/manage_binance_trader.sh stop

# Setup supervisor
./scripts/setup_supervisor.sh

# Done! Bot now runs under supervisor
```

## Emergency Stop

```bash
# Stop the bot
sudo supervisorctl stop binanceexp1-solusd

# Cancel all open orders
source .venv313/bin/activate
python -c "from src.binan import binance_wrapper; binance_wrapper.cancel_all_orders()"
```

## Benefits Summary

| Feature | tmux | Supervisor |
|---------|------|------------|
| Auto-restart on crash | ❌ | ✅ |
| Auto-start on boot | ❌ | ✅ |
| Centralized logging | ❌ | ✅ |
| Log rotation | ❌ | ✅ |
| Easy management | ⚠️ | ✅ |
| Production ready | ⚠️ | ✅ |

## Related Documentation

- **Production Status:** `BINANCE_PRODUCTION_STATUS.md`
- **Trading Runbook:** `binance_latest.md`
- **Model Details:** `binanceprogress.md`
