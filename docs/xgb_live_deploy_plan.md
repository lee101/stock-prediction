# XGB Live-Deploy Plan (draft 2026-04-18)

User directive: `PAPER=0` real trading for XGB. Paper creds (`PKEAJPSXTOHMZ2V4DDTJ25NECZ`) verified working — paper equity $52,417. Live creds verified — live equity $28,679. The block is not credentials; it's the **writer singleton**.

## Architecture reality check

The singleton is `<state>/account_locks/alpaca_live_writer.lock` (fcntl), currently held by `trading_server` (supervisor pid 2325192, uptime 3+ days). Per HARD RULE #2 there can be exactly one live writer at a time. `trading_server` IS that writer for everyone; `daily-rl-trader` is a thin HTTP client of it.

`xgbnew/live_trader.py:473–478` calls `enforce_live_singleton` directly when `--live`. That path fights `trading_server` for the fcntl lock → exit 42.

So "XGB live + RL live both trading" is not a config flip — it needs xgb to become a trading_server client.

## Accounts registry

`config/trading_server/accounts.json`:
- `live_prod` → mode=live, `allowed_bot_id: daily_stock_sortino_v1` (RL owns this slot)
- `paper_main`, `paper_shadow`, `paper_sortino_daily` (no live XGB slot)

There is no `live_xgb` account entry. Adding one points at the same Alpaca live keys (only one live account exists) and still serializes through the single `trading_server` process.

## Three options

### Option A — Swap RL out, ship XGB as sole live writer
**What**: stop `daily-rl-trader` (supervisor), flip `xgb-daily-trader.service` to `ALP_PAPER=0`, apply the queued config (top_n=1, `live_model_v2_n400_d5_lr003_top1.pkl`, allocation=1.0), start xgb. xgb will still fight the `trading_server` singleton unless we also route it through trading_server.

- Cleanest way: add a new `live_prod` `allowed_bot_id: xgb_daily_top1` OR add a dedicated `live_xgb` account and make xgb a trading_server client (Option B for xgb alone, minus the daemon-daemon coexistence constraint).
- Alternative: bypass `trading_server` and have xgb use `alpaca_wrapper` direct — the trading_server must be STOPPED so the fcntl lock is free. That orphans the RL daemon (it can't delegate writes).

**PnL math**: XGB realized at allocation=1.0 lev=1 on $28,679 → median $9,226/mo (32.2%). RL at 12.5% alloc lev=1 → median $267/mo (0.93% of account, from 7.47% × 12.5%). **34× expected lift**, but single-stock concentration.

**Risk**: concentrated 1-of-846-stock per session. DD sweep queued in `analysis/xgbnew_dd_sweep/` — round 2 (MA50 gate + vol-target) finishes ~12:15 UTC today. Deploying bare lev=1 now means accepting the 31.9% worst-window DD without the regime filter.

**Decision gate**: ship only if (a) user approves the RL-off swap, (b) DD round-2 fails to produce a strict winner — otherwise wait ~2h and deploy the DD-reduced variant.

### Option B — Cohabit: xgb as trading_server client
**What**: patch `xgbnew/live_trader.py` to use `src/trading_server/client.TradingServerClient` instead of `alpaca_wrapper`. Register a new account entry in `accounts.json` (either a second `allowed_bot_id` on `live_prod` if trading_server supports that, or a dedicated `live_prod_xgb` entry). Size-split equity between RL and XGB.

**Code work** (estimated ~150 lines):
- Replace `_build_trading_client`, `_get_account`, `_submit_market_order`, `_get_positions` with `TradingServerClient` equivalents (`get_account`, `submit_limit_order` — note: no `submit_market_order` exists on the client; need to either (i) emulate with marketable limits, or (ii) extend server).
- Acquire a writer lease at pre-open, release post-close (daily cycle, doesn't need to hold across market hours).
- Account registry surgery — verify trading_server supports dual bot_ids on `live_prod`, or create a second account entry that aliases the same Alpaca key.

**Tests**: add `tests/test_xgb_live_trader_trading_server.py` covering claim/heartbeat/submit/release cycle against `InMemoryTradingServerClient` fixture.

**PnL**: if XGB gets 50% equity → $4,618/mo expected (16% of account) + RL's $134/mo → $4,752/mo total. If XGB gets 75% → $6,926 + $67 → $6,993. The split degrades XGB's concentration benefit proportionally; still 4–7× current pace.

**Risk**: week of coding + weekend deploy window. Higher total complexity for a modest PnL gain over Option A, BUT preserves the RL bet (regime diversification) and keeps the currently-validated v7 ensemble live.

### Option C — Paper-validate XGB first, then decide
**What**: flip xgb unit to `ALP_PAPER=1`, verify `PKEAJPSXTOHMZ2V4DDTJ25NECZ` is the effective key (no shell env shadow in systemd — confirmed clean in unit), ship with top_n=1 queued config. Paper runs alongside live RL (no singleton conflict; paper bypasses singleton per `alpaca_singleton.py:176`). After one paper week (~5 sessions), compare live fills to backtest, then execute Option A or B.

**PnL**: 0 for the paper week. But: direct measurement of slippage vs backtest on live market microstructure, which is the one realism check we can't do from the sim.

User said "we are talking about prod here anyway" — this option was explicitly rejected. Leaving it listed for completeness.

## Recommendation

The call is **Option A** — XGB's OOS edge is 3.6× RL at fee parity (see `project_xgb_vs_rl_fee_aligned.md`) and seed-robust per 16-seed Bonferroni. The RL v7 ensemble took months of sweeps to get to +7.47%/mo; XGB matched 4.3× that in a week of hyperparam work. Waiting to cohabit (Option B) is ~1 week of coding that gains ~$1k/mo over shipping XGB solo.

**But I will NOT flip the live writer autonomously.** This is a change to what algorithm trades real capital tomorrow morning. User-approved deploy is required.

### If user approves Option A (checklist)

1. `sudo supervisorctl stop daily-rl-trader` (releases its execution-client connection; trading_server still runs, will release writer when daemon disconnects).
2. Verify `trading_server` still holds singleton; that's fine — xgb will replace the bot that talks to it, not the server itself.
3. Edit `config/trading_server/accounts.json`: change `live_prod.allowed_bot_id` from `daily_stock_sortino_v1` to `xgb_daily_top1` (or add xgb as alternate — verify server code supports list). If the server only accepts a single bot_id string, change it fully.
4. Edit `xgbnew/live_trader.py` — swap `_build_trading_client` → `TradingServerClient`. Add command-line flags mirroring `--server-account live_prod --server-bot-id xgb_daily_top1`.
5. Edit `/etc/systemd/system/xgb-daily-trader.service`: remove `Environment=ALP_PAPER=1`; apply queued model-path/top-n/allocation diff from alpacaprod.md block 2.
6. `sudo systemctl daemon-reload && sudo systemctl start xgb-daily-trader`.
7. At next 13:30 UTC (market open Monday), watch `journalctl -u xgb-daily-trader -f` for the first BUY.
8. Post-close (~20:00 UTC), check Alpaca positions = 0, fills = 1 buy + 1 sell of the same symbol, realized P&L recorded.
9. Update `alpacaprod.md` with a new dated block: "v8 — XGB top_n=1 live. RL v7 stood down." Move old RL block to `old_prod/`.
10. `monitoring/current_algorithms.md` — swap primary algorithm entry.
11. Commit + push everything.

### Rollback
`supervisorctl start daily-rl-trader && sudo systemctl stop xgb-daily-trader && ` revert `accounts.json` + `xgb-daily-trader.service`. RL resumes on next open.

## Outstanding pre-deploy questions (user's to answer)

1. Approve Option A, or prefer B's cohabit?
2. Ship at bare lev=1 now, OR wait ~2h for `analysis/xgbnew_dd_sweep/` round 2 result? (Round 2 tests MA50 regime gate + vol-target sizing — if any cell beats baseline on Δsortino ≥ 0 AND Δneg ≤ 0 at equal median, deploy that variant.)
3. Accept the 31.9% worst-window DD, or require lev<1 to cap it further?

---

## Update 2026-04-18 ~10:55 UTC — DD campaign closed, seed=2 leads

The DD-reduction campaign (commit `0d9026bd`) finished. Summary:

- **All 7 DD-reduction knobs FAILED the ship gate.** `ma50`, `ma20`, `voltarget{010,015,020}`, `ma50+voltarget015`, `ma50+lev1.25` — every one posts Δsortino strictly < 0. Regime gates and vol-target sizers sat out profitable days the `top_n=1` argmax had correctly identified, so the tail cost exceeded the DD benefit. Question 2 above is answered: **do NOT ship a DD-reduction variant.**
- **Seed=2 baseline passes the strict ship rule.** Same cell, seed=2 (file: `analysis/xgbnew_dd_sweep/baseline_s2/multiwindow_20260418_104652.json`):
  - Δsortino **+0.39** (8.67 vs 8.28)
  - Δworst_dd **−4.48pt** (27.39% vs 31.87% — **14% relative DD reduction**)
  - Δneg 0, Δp10 +0.09, Δmed −0.65 (still +32.15%/mo)
- **Three-seed trend on worst_dd**: s0=31.87 → s1=30.19 → s2=27.39. Axis looks real, not a lucky spike; but N=3 is small.
- **Seed-robustness sweep in flight.** `scripts/xgb_baseline_seeds_ext.sh` (driver PID 488196) is running seeds 3, 4, 5, 6 at the same cell. Seed=3 is mid-run (PID 488210); full harvest in ~45 min. Ship rule for seed=2 promotion: must be in top quartile by (sortino, −worst_dd) across the 8-seed pool AND the median sortino across 8 must be ≥ baseline.

### Updated Option A deploy recipe

If user approves Option A and the 8-seed sweep confirms seed=2:

- Swap the queued `--model-path` from `live_model_v2_n400_d5_lr003_top1.pkl` (seed=0) to the seed=2 artifact. The DD reduction is free PnL smoothness; the 0.65%/mo median hit is noise-level (≪ seed σ).
- Keep everything else in the checklist identical (bot_id, allocation=1.0, lev=1).

If user wants to ship today regardless of the seed-sweep, stay on seed=0 — it's the seed that every prior Bonferroni/leverage/DSR check used, so it's the best-validated artifact even though seed=2 has tighter DD on the latest 34-window OOS.

### Recommendation refinement

Still **Option A**. Question 2 now has a concrete answer (no DD-knob ships, hold at lev=1). Question 3's answer: the 27.39% seed=2 DD is a 14% relative improvement for free, and no lower-lev variant was tested this round — the existing `project_xgb_leverage_scaling.md` memo already documented lev<1 as a proportional median-cut with no DD/sortino benefit, so lev=1 + seed=2 is the cleanest deploy.
