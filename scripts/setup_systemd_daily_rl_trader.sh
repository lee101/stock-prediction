#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="daily-rl-trader"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_SRC="${REPO_ROOT}/systemd/${SERVICE_NAME}.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"
RESTART_AFTER_INSTALL=0
SKIP_PREFLIGHT=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart)
      RESTART_AFTER_INSTALL=1
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      ;;
    --check-only)
      CHECK_ONLY=1
      ;;
    *)
      echo "Usage: $0 [--restart] [--skip-preflight] [--check-only]" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "${SERVICE_SRC}" ]]; then
  echo "Missing service file: ${SERVICE_SRC}" >&2
  exit 1
fi

build_preflight_fragment() {
  python - "${SERVICE_SRC}" <<'PY'
from pathlib import Path
import shlex
import sys

service_path = Path(sys.argv[1])
text = service_path.read_text(encoding="utf-8")
execstart = None
for line in text.splitlines():
    if line.startswith("ExecStart="):
        execstart = line.split("=", 1)[1].strip()
        break
if not execstart:
    raise SystemExit(f"ExecStart missing from {service_path}")

tokens = shlex.split(execstart)
if len(tokens) < 3 or tokens[0] != "/bin/bash" or tokens[1] != "-lc":
    raise SystemExit(f"Unsupported ExecStart format in {service_path}: {execstart}")

fragment = tokens[2]
marker = "exec python "
marker_idx = fragment.find(marker)
if marker_idx == -1:
    raise SystemExit(f"ExecStart does not contain 'exec python': {execstart}")

python_cmd = shlex.split(fragment[marker_idx + len("exec "):])
if "--daemon" in python_cmd:
    python_cmd.remove("--daemon")
python_cmd.extend(["--check-config", "--check-config-text"])
fragment = fragment[:marker_idx] + "exec " + shlex.join(python_cmd)
print(fragment)
PY
}

run_preflight() {
  local preflight_fragment
  preflight_fragment="$(build_preflight_fragment)"
  echo "Running ${SERVICE_NAME} preflight from checked-in unit..."
  /bin/bash -lc "${preflight_fragment}"
}

if command -v systemd-analyze >/dev/null 2>&1; then
  echo "Verifying unit syntax with systemd-analyze..."
  systemd-analyze verify "${SERVICE_SRC}"
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  run_preflight
else
  echo "Skipping runtime preflight (--skip-preflight)."
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Preflight completed; not installing ${SERVICE_NAME} (--check-only)."
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not found. Preflight passed, but install must be done manually." >&2
  echo "Manual command:" >&2
  echo "  sudo cp \"${SERVICE_SRC}\" \"${SERVICE_DST}\"" >&2
  exit 1
fi

echo "Installing ${SERVICE_NAME} from ${SERVICE_SRC}..."
if [[ -f "${SERVICE_DST}" ]] && cmp -s "${SERVICE_SRC}" "${SERVICE_DST}"; then
  echo "${SERVICE_DST} is already up to date."
else
  sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
fi
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"

if [[ "${RESTART_AFTER_INSTALL}" == "1" ]]; then
  echo "Restarting ${SERVICE_NAME}..."
  sudo systemctl restart "${SERVICE_NAME}"
else
  echo "Installed and enabled ${SERVICE_NAME}, but did not start it."
  echo "Pass --restart to restart immediately after install."
fi

echo "Status:"
sudo systemctl --no-pager status "${SERVICE_NAME}" || true

echo "Useful commands:"
echo "  ${0} --check-only"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
