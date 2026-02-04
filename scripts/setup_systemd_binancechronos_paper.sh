#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="binancechronos-solusdt-paper"
SERVICE_SRC="$(pwd)/systemd/${SERVICE_NAME}.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found. Install systemd or use the manual script:" >&2
  echo "  scripts/run_binancechronos_best_paper_solusdt.sh" >&2
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

echo "Useful commands:"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
