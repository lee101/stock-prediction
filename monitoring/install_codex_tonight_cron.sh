#!/usr/bin/env bash
# Install one-shot Codex production checks for tonight's follow-ups.
set -euo pipefail

REPO=/nvme0n1-disk/code/stock-prediction
CMD="$REPO/monitoring/codex_prod_check.sh >> $REPO/monitoring/logs/codex_cron.log 2>&1"
MARK_BEGIN="# BEGIN one-shot codex prod checks 2026-05-04"
MARK_END="# END one-shot codex prod checks 2026-05-04"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

crontab -l 2>/dev/null | sed "/^$MARK_BEGIN\$/,/^$MARK_END\$/d" > "$tmp"
cat >> "$tmp" <<EOF
$MARK_BEGIN
30 22 4 5 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202605042230" && $CMD
30 1 5 5 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202605050130" && $CMD
30 4 5 5 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202605050430" && $CMD
30 7 5 5 * test "\$(date -u +\\%Y\\%m\\%d\\%H\\%M)" = "202605050730" && $CMD
$MARK_END
EOF

crontab "$tmp"
echo "Installed one-shot Codex prod checks:"
crontab -l | sed -n "/^$MARK_BEGIN\$/,/^$MARK_END\$/p"
