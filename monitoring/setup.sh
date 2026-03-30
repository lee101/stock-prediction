#!/bin/bash
# Install/manage Alpaca trading monitor systemd timer.
# Usage:
#   ./monitoring/setup.sh              # install and start timer
#   ./monitoring/setup.sh --status     # show timer status
#   ./monitoring/setup.sh --remove     # stop and remove timer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="$SCRIPT_DIR/systemd"
SERVICE_NAME="alpaca-monitor"

case "${1:-install}" in
    --status)
        echo "Timer status:"
        systemctl list-timers --all | grep "$SERVICE_NAME" || echo "  (not installed)"
        echo ""
        echo "Last run:"
        journalctl -u "$SERVICE_NAME.service" --no-pager -n 5 2>/dev/null || echo "  (no logs)"
        ;;
    --remove)
        echo "Stopping and removing $SERVICE_NAME timer..."
        sudo systemctl stop "$SERVICE_NAME.timer" 2>/dev/null || true
        sudo systemctl disable "$SERVICE_NAME.timer" 2>/dev/null || true
        sudo rm -f "/etc/systemd/system/$SERVICE_NAME.service" "/etc/systemd/system/$SERVICE_NAME.timer"
        sudo systemctl daemon-reload
        echo "Removed."
        ;;
    install|"")
        echo "Installing $SERVICE_NAME timer..."
        sudo cp "$SYSTEMD_DIR/$SERVICE_NAME.service" /etc/systemd/system/
        sudo cp "$SYSTEMD_DIR/$SERVICE_NAME.timer" /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable "$SERVICE_NAME.timer"
        sudo systemctl start "$SERVICE_NAME.timer"
        echo "Installed. Next run times:"
        systemctl list-timers --all | grep "$SERVICE_NAME"
        ;;
    *)
        echo "Usage: $0 [--status|--remove]"
        exit 1
        ;;
esac
