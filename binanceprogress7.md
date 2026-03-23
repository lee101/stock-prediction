# Binance Model Refresh Pass (2026-03-23)

## Runtime Findings

- Current hybrid live allocator is effectively **not trading** on `2026-03-23`.
  - Evidence: `strategy_state/hybrid_trade_cycles/hybrid-cycle_20260323.jsonl`
  - Repeated status: `invalid_allocation_plan_rl_flat`
  - Root cause in log payload:
    - Gemini API error `403 PERMISSION_DENIED`
    - Message: API key was reported as leaked
  - Result: no orders placed, allocation falls back to cash.

- Legacy `binanceneural` live state did lose money before that handoff.
  - Evidence: `strategy_state/binanceneural_pnl_state_live.json`
  - Realized PnL total: `-1232.47`
  - Biggest losers:
    - `DOGEUSDT`: `-670.91`
    - `SOLUSD`: `-391.06`
    - `AAVEUSDT`: `-267.57`
    - `BTCUSD`: `-260.92`

## Validation Harness Work

### Code changes

- `binanceneural.marketsimulator`
  - Added seeded-start support via:
    - `initial_inventory_by_symbol`
    - `initial_cost_basis_by_symbol`
  - Works in both single-symbol and shared-cash simulation.

- New script:
  - `scripts/run_binanceneural_robustness_sweep.py`
  - Evaluates checkpoint candidates across:
    - multiple windows
    - seeded start states
    - decision lags
    - fill-buffer / slippage assumptions
    - action intensity / offset overrides

- `scripts/run_deep_binanceneural_sweep.py`
  - Added modern architecture flags:
    - `--model-arch`
    - `--num-memory-tokens`
    - `--dilated-strides`
    - `--attention-window`
    - `--use-compile`
    - `--use-vectorized-sim`
    - `--accumulation-steps`
    - `--decision-lag-range`
    - `--forecast-horizons`

- `binanceneural/benchmark_training.py`
  - Added matching architecture flags for throughput probes.
  - Enabled TF32 matmul precision so benchmark matches real trainer settings more closely.

### Tests

- `pytest tests/test_marketsimulator.py tests/test_run_binanceneural_robustness_sweep.py -q`
- Result: `56 passed`

## Robustness Baseline

Command run:

```bash
source .venv313/bin/activate
python scripts/run_binanceneural_robustness_sweep.py \
  --checkpoints binanceneural/checkpoints/crypto_portfolio_6sym \
  --sample-epochs 12 \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --sequence-length 48 \
  --window-hours 168,336,720 \
  --decision-lag-list 1,2 \
  --fill-buffer-bps-list 0,5,10 \
  --output-dir analysis/binanceneural_robustness_crypto_portfolio_6sym
```

Artifacts:

- Summary: `analysis/binanceneural_robustness_crypto_portfolio_6sym/summary.csv`
- Scenarios: `analysis/binanceneural_robustness_crypto_portfolio_6sym/scenarios.csv`

Result for `crypto_portfolio_6sym/epoch_012.pt`:

- `selection_score`: `-133.56`
- `return_mean_pct`: `-8.64%`
- `return_worst_pct`: `-17.60%`
- `max_drawdown_worst_pct`: `18.63%`
- `negative_return_rate`: `1.0`

Key scenario read:

- Worst cluster is the `168h` window with `lag=2` and `fill_buffer_bps=10`.
- Even the best cluster stayed negative:
  - around `-0.36%` return with `lag=1`, `fill_buffer_bps=0`
- Conclusion:
  - this checkpoint is not a promotion candidate
  - do not use it as the “best algorithm” baseline

## Training Efficiency Read

Local GPU: `RTX 3090 Ti 24GB`

Benchmark commands:

```bash
source .venv313/bin/activate
python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 16 --seq-len 72 \
  --hidden-dim 256 --num-layers 4 --num-heads 8 \
  --use-fast-sim --use-compile

python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 16 --seq-len 72 \
  --hidden-dim 256 --num-layers 4 --num-heads 8 \
  --model-arch nano --num-memory-tokens 4 \
  --dilated-strides 1,4,24 \
  --use-fast-sim --use-compile

python binanceneural/benchmark_training.py \
  --steps 20 --warmup 5 \
  --batch-size 8 --seq-len 96 \
  --hidden-dim 384 --num-layers 6 --num-heads 8 \
  --model-arch nano --num-memory-tokens 8 \
  --dilated-strides 1,4,24,72 --attention-window 96 \
  --use-fast-sim --use-compile
```

Observed throughput:

| Config | Steps/s | Samples/s | Peak VRAM |
|--------|--------:|----------:|----------:|
| classic `256x4` | `8.13` | `130.14` | `139.1 MB` |
| nano `256x4` + mem4 + dilated | `7.89` | `126.18` | `34.9 MB` |
| nano `384x6` + mem8 + dilated | `5.84` | `46.73` | `71.2 MB` |

Read:

- `256` width is the current sweet spot.
- `384x6` is too expensive for the first search pass.
- Nano is slightly slower than classic at this size on the 3090 Ti, but the VRAM footprint is much lower.
- That means the next sweep should focus on:
  - `192`
  - `256`
  - `320`
  - not `384x6` yet

## Local Proof Sweep

Command running:

```bash
source .venv313/bin/activate
python scripts/run_deep_binanceneural_sweep.py \
  --symbols SOLUSD \
  --dims 192 256 384 \
  --wds 0.03 \
  --lrs 1e-4 \
  --forecast-horizons 1,24 \
  --sequence-length 96 \
  --batch-size 16 \
  --transformer-layers 4 \
  --transformer-heads 8 \
  --validation-days 90 \
  --model-arch nano \
  --num-memory-tokens 4 \
  --dilated-strides 1,4,24 \
  --attention-window 96 \
  --return-weight 0.08 \
  --fill-buffer-pct 0.0005 \
  --decision-lag-bars 1 \
  --decision-lag-range 0,1,2 \
  --loss-type sortino \
  --use-compile \
  --use-vectorized-sim \
  --epochs 4 \
  --patience 2 \
  --results-dir analysis/solusd_modern_size_sweep_local \
  --no-wandb
```

Current status at last check:

- process still active locally
- no checkpoint files written yet
- first config had reached:
  - `Training: 2043 train batches, 136 val batches`

## Remote Status

Attempted remote connectivity checks from this shell did not return usable output:

```bash
ssh -o StrictHostKeyChecking=no administrator@93.127.141.100 \
  'cd /nvme0n1-disk/code/stock-prediction && source .venv313/bin/activate && python -V'
```

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no administrator@93.127.141.100 'echo ok'
```

Result:

- remote training was **not launched** from this shell
- no reproducible remote run id / log path exists yet
- next step is to resolve the SSH/auth/connectivity issue, then launch the `192/256/320` trio sweep remotely
