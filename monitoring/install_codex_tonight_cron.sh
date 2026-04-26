#!/usr/bin/env bash
# Install one-shot Codex production checks for the evening of 2026-04-25 UTC.
set -euo pipefail

REPO=/nvme0n1-disk/code/stock-prediction
CMD="$REPO/monitoring/codex_prod_check.sh >> $REPO/monitoring/logs/codex_cron.log 2>&1"
MARK_BEGIN="# BEGIN one-shot codex prod checks 2026-04-25"
MARK_END="# END one-shot codex prod checks 2026-04-25"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

crontab -l 2>/dev/null | sed "/^$MARK_BEGIN\$/,/^$MARK_END\$/d" > "$tmp"
cat >> "$tmp" <<EOF
$MARK_BEGIN
30 16 25 4 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202604251630" && $CMD
30 18 25 4 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202604251830" && $CMD
30 21 25 4 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202604252130" && $CMD
30 0 26 4 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202604260030" && $CMD
$MARK_END
EOF

crontab "$tmp"
echo "Installed one-shot Codex prod checks:"
crontab -l | sed -n "/^$MARK_BEGIN\$/,/^$MARK_END\$/p"
