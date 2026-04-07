# gpu_trading_env

SoA fused `env_step` CUDA kernel for GPU-resident RL trading (Blackwell SM120,
embedded PTX for forward compat).

One thread per episode. State (SoA):
`pos_qty, pos_entry_px, cash, equity, dd_peak, drawdown, t_idx, done`.

Action: `[B, 4] = (p_bid_frac, p_ask_frac, q_bid_frac, q_ask_frac)` where the
price fracs are in `[-1, 1]` and scale a `max_quote_offset_bps` around the bar
mid, and the size fracs are in `[0, 1]` scaled by the per-side leverage budget.

Buffered limit-fill (5 bps default): buy iff `p_bid >= L*(1+5bps)`,
sell iff `p_ask <= H*(1-5bps)`. Conservative fill price.

Fees on filled notional, leverage cap at `max_leverage=5x` (force-reduce at
close if exceeded), maintenance margin -> liquidation at punitive bar extreme
plus a `liq_penalty` fraction of prior equity, drawdown tracking via `dd_peak`,
reward is `log(equity_t / equity_{t-1})`.

Optional `cost[B, 4]` output: `(drawdown_increment, liquidation_indicator,
leverage_violation, turnover = filled_notional / equity)`. These feed the
constrained-MDP losses in Unit P4-3.

## Build

```bash
source .venv/bin/activate
cd gpu_trading_env
TMPDIR=$PWD/tmp uv pip install -e . --no-build-isolation
```

## Usage

```python
import gpu_trading_env, torch
env = gpu_trading_env.make(B=1024, ohlc_path_or_tensor=None)
a = torch.zeros(1024, 4, device="cuda")
obs, reward, done, cost = env.step(a)
print(env.state["equity"].mean().item())
```
