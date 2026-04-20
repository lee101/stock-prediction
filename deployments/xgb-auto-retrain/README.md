# xgb-auto-retrain

Weekly retrain of the live XGB 5-seed ensemble. Schedule: Sunday 23:00 UTC
(≈10 hours before Monday pre-market).

## Why

`project_xgb_recency_monotonic.md` — a 1-year-stale ensemble loses
~+86 goodness/year during regime shifts (tariff-crash-type events), and
near-zero in calm weeks. Weekly retrain is cheap (~2 min) asymmetric-
upside insurance.

## Pieces

- `scripts/xgb_weekly_retrain.sh` — the worker: trains a new
  5-seed ensemble to a staging dir, verifies 5×pkl + manifest, atomically
  rotates the live dir (snapshot → `_prev_<ts>/`), prunes backups older
  than 28 days, and restarts `xgb-daily-trader-live` via supervisorctl.
- `xgb-auto-retrain.service` — oneshot systemd unit that runs the script.
- `xgb-auto-retrain.timer` — weekly timer (Sun 23:00 UTC).

## Install

```bash
sudo cp deployments/xgb-auto-retrain/xgb-auto-retrain.service \
        deployments/xgb-auto-retrain/xgb-auto-retrain.timer \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now xgb-auto-retrain.timer
sudo systemctl list-timers xgb-auto-retrain.timer
```

## Manual ops

```bash
# Dry-run (train to staging + verify, no rotate, no supervisor restart):
DRY_RUN=1 ./scripts/xgb_weekly_retrain.sh

# Force a real retrain now:
sudo systemctl start xgb-auto-retrain.service

# Check when it'll next fire / recently fired:
sudo systemctl list-timers xgb-auto-retrain.timer
sudo journalctl -u xgb-auto-retrain -n 100

# Rollback to the prior ensemble:
LIVE=analysis/xgbnew_daily/alltrain_ensemble_gpu
ls -dt "${LIVE}"_prev_* | head -n1          # find most recent backup
mv "${LIVE}" "${LIVE}_bad_$(date -u +%s)"   # park the bad one
mv <the backup>  "${LIVE}"                  # restore
sudo supervisorctl restart xgb-daily-trader-live
```

## Prerequisites

- `xgb-daily-trader-live` must already be registered in supervisor
  (`deployments/xgb-daily-trader-live/supervisor.conf`).
- The user `administrator` needs `sudo -n supervisorctl` rights
  (already configured; dry-run doesn't require sudo).
- GPU visible to the .venv — `python -c "import torch; torch.cuda.is_available()"`.
