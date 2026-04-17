# Stock Alpaca Deployment

Current stock production on `leaf-gpu` is the screened32 daily RL ensemble in `pufferlib_market/prod_ensemble_screened32/`, with the 2026-04-14 v5 swap `D_s27 -> I_s32`. The live command path is `deployments/daily-rl-trader/launch.sh`, which runs `trade_daily_stock_prod.py --daemon --live --execution-backend trading_server --server-account live_prod --server-bot-id daily_stock_sortino_v1`. The broker boundary is `deployments/trading-server/launch.sh`; that process is the only live Alpaca writer and owns `alpaca_live_writer`.

The best validated stock model family in the repo is still RL, not Chronos2 and not XGBoost. Chronos2 directional calibration is promising but not production-qualified, and the XGBoost daily trader is explicitly marked as needing production-faithful validation before promotion. The current champion recorded in [alpacaprod.md](/nvme0n1-disk/code/stock-prediction/alpacaprod.md) is the 13-model screened32 v5 ensemble with exhaustive 263-window stats at `lag=2`, binary fills, `fee=10bps`, `slip=5bps`: median `+19.57%`, p10 `+7.68%`, `8/263` negative windows, Sortino `34.07`.

Deployment steps on `leaf-gpu`:

```bash
cd /nvme0n1-disk/code/stock-prediction
source ~/.secretbashrc
source .venv313/bin/activate
pytest -q tests/test_alpaca_deploy_preflight.py tests/test_trade_daily_stock_prod.py tests/test_alpaca_singleton.py
sudo cp deployments/trading-server/supervisor.conf /etc/supervisor/conf.d/trading-server.conf
sudo cp deployments/daily-rl-trader/supervisor.conf /etc/supervisor/conf.d/daily-rl-trader.conf
sudo cp systemd/daily-rl-trader.service /etc/systemd/system/daily-rl-trader.service
sudo systemctl daemon-reload
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart trading-server daily-rl-trader
sudo supervisorctl status trading-server daily-rl-trader
```

Smoke-check the exact live runtime:

```bash
source ~/.secretbashrc
source .venv313/bin/activate
ALLOW_ALPACA_LIVE_TRADING=1 python trade_daily_stock_prod.py \
  --once --dry-run --live \
  --execution-backend trading_server \
  --server-account live_prod \
  --server-bot-id daily_stock_sortino_v1
tail -n 80 /var/log/supervisor/trading-server.log
tail -n 120 /var/log/supervisor/daily-rl-trader.log
tail -n 80 /var/log/supervisor/daily-rl-trader-error.log
```

Production-faithful simulation commands:

```bash
source ~/.secretbashrc
source .venv313/bin/activate
python scripts/eval_100d.py \
  --checkpoint pufferlib_market/prod_ensemble_screened32/C_s7.pt \
  --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \
  --n-windows 30 \
  --window-days 100 \
  --monthly-target 0.27 \
  --fail-fast-max-dd 0.20
```

Use `trade_daily_stock_prod.py --backtest` and `tests/test_trade_daily_stock_prod.py` for execution-path parity checks, but treat `scripts/eval_100d.py` and the binary-fill marketsim as the promotion gate.
