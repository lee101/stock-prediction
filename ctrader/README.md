# ctrader

Zero-Python production trading bot in C. Uses libcurl for Binance REST API, libtorch for policy inference, and a ported C market simulator.

## Build

```bash
make              # build ctrader binary (stubs for curl/torch)
make test         # compile and run tests
make test_valgrind # run tests under valgrind
```

With libcurl + libtorch:
```bash
make USE_CURL=1 TORCH_DIR=/usr/local/lib/libtorch
```

## Components

- `market_sim.c` -- single-symbol and multi-symbol portfolio simulator (ported from csim/market_sim.c)
- `binance_rest.c` -- Binance REST client (stubs, needs libcurl + openssl)
- `policy_infer.c` -- libtorch policy inference wrapper (stubs, needs libtorch)
- `trade_loop.c` -- main hourly trading cycle
- `main.c` -- CLI entry point

## Usage

```bash
./ctrader --symbols BTCUSDT,ETHUSDT,SOLUSDT --model policy.pt --dry-run
./ctrader --symbols BTCUSDT --api-key KEY --secret-key SECRET --live
```
