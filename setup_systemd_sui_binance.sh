#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="sui-binance-trader"
SERVICE_SRC="$(pwd)/systemd/${SERVICE_NAME}.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found. Run manually:" >&2
  echo "  .venv313/bin/python -m bitbankstylelongsuitrain.trade_binance_sui --checkpoint binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt" >&2
  exit 1
fi

if [ ! -f "$SERVICE_SRC" ]; then
  echo "Missing service file: $SERVICE_SRC" >&2
  exit 1
fi

echo "Installing ${SERVICE_NAME}..."
sudo cp "$SERVICE_SRC" "$SERVICE_DST"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo "Status:"
sudo systemctl --no-pager status "$SERVICE_NAME"

echo ""
echo "Useful commands:"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
