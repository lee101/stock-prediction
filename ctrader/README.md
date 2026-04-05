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
- `market_sim.c` also includes a continuous target-weight portfolio simulator for PPO/SAC-style research
- `binance_rest.c` -- Binance REST client (stubs, needs libcurl + openssl)
- `policy_infer.c` -- libtorch policy inference wrapper (stubs, needs libtorch)
- `trade_loop.c` -- main hourly trading cycle
- `main.c` -- CLI entry point

## Research Direction

The low-level simulator now supports two distinct paths:

- discrete buy/sell amount simulation for parity with the existing `ctrader` execution loop
- continuous target-weight rebalancing for multi-asset RL research

The target-weight simulator is intended as the baseline environment for GPU-first PPO/SAC experiments:

- action: target weight per symbol at bar `t`
- execution model: rebalance at bar `t`, then mark to market over `t -> t+1`
- constraints: short clamp, gross leverage clamp
- costs: turnover fees plus optional borrow cost for leverage/short exposure
- outputs: total return, annualized return, sortino, drawdown, turnover, fees

There is now a matching research trainer in `train_weight_ppo.py`:

- data source: aligned Binance CSV closes from `trainingdatahourlybinance`
- policy: continuous Gaussian PPO over raw action scores
- portfolio mapping: long-only softmax weights by default, optional short path later
- checkpoint ranking: validation annualized return from the C simulator, not only in-loop reward
- purpose: establish a stable continuous-weight baseline before moving the same interface into C++/LibTorch

The native C port now also exposes a stateful target-weight environment in `market_sim.c`:

- `weight_env_init` / `weight_env_free` for lifecycle
- `weight_env_reset` for deterministic episode starts
- `weight_env_get_obs` for encoded lookback observations
- `weight_env_step` for native reward/equity transitions with terminal summary metrics

That gives the eventual C++ trainer a direct reset/obs/step substrate without reimplementing portfolio semantics again.

## Usage

```bash
./ctrader --symbols BTCUSDT,ETHUSDT,SOLUSDT --model policy.pt --dry-run
./ctrader --symbols BTCUSDT --api-key KEY --secret-key SECRET --live
```
