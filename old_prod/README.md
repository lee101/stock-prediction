# old_prod archive

This directory stores dated production snapshots that were previously the current state in `prod.md`.

Use filenames like:
- `YYYY-MM-DD-<slug>.md`
- `YYYY-MM-DD-HHMM-<slug>.md`

Each archived snapshot should include:
- the exact service/supervisor/unit name
- the exact launch command and environment notes
- the active checkpoint(s) or model identifiers
- whether the deployment was live or paper
- a timestamped PnL / equity / open-position snapshot
- the reason the snapshot was superseded

`prod.md` stays as the current production ledger; `old_prod/` preserves the historical trail.
