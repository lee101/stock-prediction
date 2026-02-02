#!/bin/bash
# Setup Binance trading bot with supervisor

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Setting up Binance Trading Bot with Supervisor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run this script as root/sudo"
    echo "   It will prompt for sudo when needed"
    exit 1
fi

# Check if supervisor is installed
if ! command -v supervisorctl &> /dev/null; then
    echo "❌ supervisorctl not found. Install supervisor first:"
    echo "   sudo apt install supervisor"
    exit 1
fi

# Stop tmux session if running
if tmux has-session -t binance-trader 2>/dev/null; then
    echo "🛑 Stopping tmux session 'binance-trader'..."
    tmux kill-session -t binance-trader
fi

# Copy supervisor config
echo "📋 Installing supervisor config..."
sudo cp supervisor/binanceexp1-solusd.conf /etc/supervisor/conf.d/

# Reload supervisor
echo "🔄 Reloading supervisor..."
sudo supervisorctl reread

# Update supervisor (add new programs)
echo "✅ Updating supervisor..."
sudo supervisorctl update

# Check status
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sudo supervisorctl status binanceexp1-solusd

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   ✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Useful commands:"
echo "  sudo supervisorctl status binanceexp1-solusd      # Check status"
echo "  sudo supervisorctl tail -f binanceexp1-solusd    # View live logs"
echo "  sudo supervisorctl restart binanceexp1-solusd    # Restart bot"
echo "  sudo supervisorctl stop binanceexp1-solusd       # Stop bot"
echo "  sudo supervisorctl start binanceexp1-solusd      # Start bot"
echo
echo "Log files:"
echo "  /var/log/supervisor/binanceexp1-solusd.log       # Output"
echo "  /var/log/supervisor/binanceexp1-solusd-error.log # Errors"
echo
