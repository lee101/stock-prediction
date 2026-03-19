#!/usr/bin/env bash
set -euo pipefail

# Disables Bags.fm live trading services managed by supervisor/systemd.
# This is a production safety tool; it is intentionally conservative and only
# targets known Bags.fm trading entrypoints.

if [ "$(id -u)" -ne 0 ]; then
  echo "This script must run as root." >&2
  echo "Run: sudo bash scripts/disable_bagsfm_trading.sh" >&2
  exit 1
fi

PATTERNS=(
  "run_bags_trader.py"
  "run_neural_trader.py"
  "run_neural_trader_v3.py"
  "bagsfm.trader"
  "SOLANA_PRIVATE_KEY"
)

_grep_args=()
for p in "${PATTERNS[@]}"; do
  _grep_args+=(-e "$p")
done

echo "Disabling Bags.fm trading (supervisor + systemd)..."

echo ""
echo "[1/2] Supervisor"
if command -v supervisorctl >/dev/null 2>&1; then
  SUP_CONF_DIR="/etc/supervisor/conf.d"
  if [ -d "$SUP_CONF_DIR" ]; then
    mapfile -t sup_matches < <(grep -RIl "${_grep_args[@]}" "$SUP_CONF_DIR" 2>/dev/null || true)

    if [ "${#sup_matches[@]}" -eq 0 ]; then
      echo "No matching supervisor configs found in $SUP_CONF_DIR"
    else
      for conf_file in "${sup_matches[@]}"; do
        if [[ "$conf_file" == *.disabled ]]; then
          echo "Already disabled: $conf_file"
          continue
        fi

        prog_name="$(
          sed -n 's/^[[:space:]]*\\[program:\\([^]]\\+\\)\\][[:space:]]*$/\\1/p' "$conf_file" \
            | head -n 1 || true
        )"

        if [ -n "${prog_name:-}" ]; then
          echo "Stopping supervisor program: $prog_name"
          supervisorctl stop "$prog_name" >/dev/null 2>&1 || true
        else
          echo "Could not parse [program:...] name from $conf_file (will still disable config)"
        fi

        echo "Disabling supervisor config: $conf_file"
        mv -f "$conf_file" "${conf_file}.disabled"
      done

      echo "Reloading supervisor config (reread/update)..."
      supervisorctl reread >/dev/null 2>&1 || true
      supervisorctl update >/dev/null 2>&1 || true
    fi
  else
    echo "Supervisor config dir not found: $SUP_CONF_DIR (skipping)"
  fi
else
  echo "supervisorctl not found (skipping)"
fi

echo ""
echo "[2/2] systemd"
if command -v systemctl >/dev/null 2>&1; then
  UNIT_DIRS=(
    "/etc/systemd/system"
    "/lib/systemd/system"
    "/usr/lib/systemd/system"
  )

  unit_files=()
  for d in "${UNIT_DIRS[@]}"; do
    if [ -d "$d" ]; then
      while IFS= read -r f; do
        unit_files+=("$f")
      done < <(grep -RIl "${_grep_args[@]}" "$d" 2>/dev/null || true)
    fi
  done

  if [ "${#unit_files[@]}" -eq 0 ]; then
    echo "No matching systemd unit files found."
  else
    # Deduplicate file list.
    mapfile -t unit_files < <(printf "%s\n" "${unit_files[@]}" | awk '!seen[$0]++')

    for f in "${unit_files[@]}"; do
      unit="$(basename "$f")"
      echo "Stopping/disabling/masking: $unit ($f)"
      systemctl stop "$unit" >/dev/null 2>&1 || true
      systemctl disable "$unit" >/dev/null 2>&1 || true
      systemctl mask "$unit" >/dev/null 2>&1 || true
    done

    echo "Reloading systemd daemon..."
    systemctl daemon-reload >/dev/null 2>&1 || true
  fi
else
  echo "systemctl not found (skipping)"
fi

echo ""
echo "Done. Note: the codebase also enforces a hard kill switch via BAGSFM_TRADING_DISABLED=1."

