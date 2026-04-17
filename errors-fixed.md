# Errors Fixed

2026-04-15 stock Alpaca production audit on `leaf-gpu` found and fixed the following deployment drift:

- `scripts/alpaca_deploy_preflight.py` still treated `daily-rl-trader` as a systemd unit. The real live manager is supervisor, so preflight was falsely reporting the daily bot as down even while the supervisor process was resident.
- `unified_orchestrator/service_config.json` still advertised the old 12-symbol stock universe for `daily-rl-trader`. That drift no longer matched the live screened32 universe and produced misleading ownership context.
- The checked-in `systemd/daily-rl-trader.service` still pointed at the old direct-Alpaca `ExecStart` path. It now delegates to the same checked-in launcher used by supervisor so the fallback unit matches the real bot.
- The deployment launchers were brittle around environment bootstrap. Historical supervisor logs show failures from `/root/.secretbashrc` lookups and `HOME: unbound variable`. The checked-in launchers now set `HOME`/`USER` defensively before sourcing `~/.secretbashrc`.

What was diagnosed but not a bug:

- The live stock RL process is currently not trading because the model output is `flat`, not because the daemon is dead. A direct dry run on 2026-04-15 06:59 UTC against fresh Alpaca data loaded the live 13-model ensemble cleanly and returned `flat` with `confidence=5.47%`.
- The trading server is the single live Alpaca writer and is healthy. It is refreshing live quotes continuously; the daily bot talks to it over loopback with `account=live_prod` and `bot_id=daily_stock_sortino_v1`.

Open production caveats:

- `daily-rl-trader.service` on the host is still an inactive legacy unit unless it is explicitly reinstalled from the updated checked-in file.
- The current stock champion is the best stock candidate we have, but it still does not satisfy the repo-wide `27%/month` unseen-data gate. It remains the best validated stock deployment candidate, not a universal target-clearing model.
