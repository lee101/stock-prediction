#!/usr/bin/env python3
"""Fine-grain trade penalty sweep around the tp=0.15 optimum.

The mass_daily grid tested tp=0.05, 0.10, 0.15, 0.20 and tp=0.15 won.
This script sweeps the neighborhood [0.11..0.19] (excluding 0.15 which
is already trained) with 3 seeds each, to find the exact optimum.

Each config trains for 300s then is evaluated on crypto5_daily_val.bin
via both the C env evaluator and the pure-python market simulator.

Usage:
    source .venv313/bin/activate
    python sweep_tp_fine_grain.py
"""

import csv
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ── Config ──────────────────────────────────────────────────────────────

TRADE_PENALTIES = [0.11, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.19]
SEEDS = [42, 314, 123]
TIME_BUDGET_S = 300

REPO = Path(__file__).resolve().parent
TRAIN_DATA = "pufferlib_market/data/crypto5_daily_train.bin"
VAL_DATA = "pufferlib_market/data/crypto5_daily_val.bin"
CKPT_ROOT = Path("pufferlib_market/checkpoints/tp_fine")
LEADERBOARD = Path("pufferlib_market/tp_fine_leaderboard.csv")

# Market sim settings
FEE_RATE = 0.001
FILL_BUFFER_BPS = 8.0
PERIODS_PER_YEAR = 365.0
MAX_LEVERAGE = 1.0

LEADERBOARD_FIELDS = [
    "name", "trade_penalty", "seed",
    "train_return", "train_sortino", "train_wr", "train_steps", "elapsed_s",
    "val_return", "val_sortino", "val_wr", "val_profitable_pct",
    "sim_60d_ret_pct", "sim_60d_sortino",
    "sim_90d_ret_pct", "sim_90d_sortino",
    "sim_120d_ret_pct", "sim_120d_sortino", "sim_120d_max_dd_pct",
    "sim_120d_trades", "sim_120d_wr",
    "sim_180d_ret_pct", "sim_180d_sortino",
    "error",
]

SIM_PERIODS = {"60d": 60, "90d": 90, "120d": 120, "180d": 180}


# ── Helpers ─────────────────────────────────────────────────────────────

def ckpt_dir_name(tp: float, seed: int) -> str:
    return f"tp{tp:.2f}_s{seed}"


def resolve_checkpoint(tp: float, seed: int) -> Path | None:
    """Find the best available checkpoint for a (tp, seed) combo."""
    base = CKPT_ROOT / ckpt_dir_name(tp, seed)
    for name in ("best.pt", "final.pt"):
        p = base / name
        if p.exists():
            return p
    pts = list(base.glob("*.pt"))
    if pts:
        return max(pts, key=lambda p: p.stat().st_mtime)
    return None


def fmt_val(v, width=8):
    """Format a numeric value for table display."""
    try:
        return f"{float(v):>+{width}.1f}"
    except (ValueError, TypeError):
        return f"{'N/A':>{width}}"


def run_training(tp: float, seed: int) -> dict:
    """Run a single training job with SIGTERM after TIME_BUDGET_S."""
    name = ckpt_dir_name(tp, seed)
    checkpoint_dir = str(CKPT_ROOT / name)

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", TRAIN_DATA,
        "--total-timesteps", "999999999",
        "--max-steps", "720",
        "--hidden-size", "1024",
        "--lr", "3e-4",
        "--ent-coef", "0.05",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--num-envs", "128",
        "--rollout-len", "256",
        "--ppo-epochs", "4",
        "--seed", str(seed),
        "--reward-scale", "10.0",
        "--reward-clip", "5.0",
        "--cash-penalty", "0.01",
        "--fee-rate", str(FEE_RATE),
        "--trade-penalty", str(tp),
        "--anneal-lr",
        "--checkpoint-dir", checkpoint_dir,
        "--periods-per-year", str(PERIODS_PER_YEAR),
    ]

    print(f"\n{'='*60}")
    print(f"  TRAIN  tp={tp:.2f}  seed={seed}  ({TIME_BUDGET_S}s budget)")
    print(f"{'='*60}")

    t0 = time.time()
    train_return = None
    train_sortino = None
    train_wr = None
    total_steps = 0

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO), preexec_fn=os.setsid,
        )
        stdout_lines = []
        while time.time() - t0 < TIME_BUDGET_S:
            if proc.poll() is not None:
                break
            try:
                line = proc.stdout.readline()
                if line:
                    decoded = line.decode("utf-8", errors="replace").strip()
                    stdout_lines.append(decoded)
            except Exception:
                pass

        # Kill if still running
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

        elapsed = time.time() - t0

        # Parse training stats from last logged line with "ret="
        for line in reversed(stdout_lines):
            if "ret=" in line and train_return is None:
                try:
                    for part in line.split():
                        if part.startswith("ret="):
                            train_return = float(part.split("=")[1])
                        elif part.startswith("sortino="):
                            train_sortino = float(part.split("=")[1])
                        elif part.startswith("wr="):
                            train_wr = float(part.split("=")[1])
                        elif part.startswith("step="):
                            total_steps = int(part.split("=")[1].replace(",", ""))
                except Exception:
                    pass
                if train_return is not None:
                    break

        print(f"  Training done: {elapsed:.0f}s, {total_steps:,} steps, "
              f"ret={train_return}, sortino={train_sortino}, wr={train_wr}")

    except Exception as e:
        print(f"  Training ERROR: {e}")
        return {"error": str(e)}

    return {
        "train_return": train_return,
        "train_sortino": train_sortino,
        "train_wr": train_wr,
        "train_steps": total_steps,
        "elapsed_s": elapsed,
    }


def run_c_env_eval(tp: float, seed: int) -> dict:
    """Evaluate checkpoint using the C env evaluator (random episodes)."""
    cp = resolve_checkpoint(tp, seed)
    if cp is None:
        return {"error": "no checkpoint"}

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.evaluate",
        "--checkpoint", str(cp),
        "--data-path", VAL_DATA,
        "--deterministic",
        "--hidden-size", "1024",
        "--max-steps", "720",
        "--num-episodes", "100",
        "--seed", "42",
        "--fill-slippage-bps", "8",
        "--periods-per-year", str(PERIODS_PER_YEAR),
    ]

    val_return = None
    val_sortino = None
    val_wr = None
    val_profitable_pct = None

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(REPO), timeout=300,
        )
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "Return:" in line and "mean=" in line:
                try:
                    val_return = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Win rate:" in line and "mean=" in line:
                try:
                    val_wr = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if "Sortino:" in line and "mean=" in line:
                try:
                    val_sortino = float(line.split("mean=")[1].split()[0])
                except Exception:
                    pass
            if ">0:" in line:
                try:
                    pct_str = line.split("(")[1].split("%")[0]
                    val_profitable_pct = float(pct_str)
                except Exception:
                    pass
        if result.returncode != 0 and val_return is None:
            return {"error": f"eval exit {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"error": "eval timeout"}
    except Exception as e:
        return {"error": str(e)}

    print(f"  C-env val: ret={val_return}, sortino={val_sortino}, "
          f"wr={val_wr}, profitable={val_profitable_pct}%")

    return {
        "val_return": val_return,
        "val_sortino": val_sortino,
        "val_wr": val_wr,
        "val_profitable_pct": val_profitable_pct,
    }


def run_market_sim_eval(tp: float, seed: int, data, sim_deps) -> dict:
    """Evaluate checkpoint on the market simulator (deterministic, full val).

    Args:
        data: Pre-loaded MktdData from read_mktd.
        sim_deps: Tuple of (simulate_daily_policy, annualize_total_return,
                  TradingPolicy, _infer_num_actions, _infer_arch,
                  _infer_hidden_size, _slice_tail).
    """
    cp = resolve_checkpoint(tp, seed)
    if cp is None:
        return {"error": "no checkpoint"}

    (simulate_daily_policy, annualize_total_return,
     TradingPolicy, _infer_num_actions, _infer_arch,
     _infer_hidden_size, _slice_tail) = sim_deps

    try:
        device = torch.device("cpu")
        nsym = data.num_symbols

        payload = torch.load(str(cp), map_location=device, weights_only=False)
        sd = payload.get("model", payload)
        obs_size = nsym * 16 + 5 + nsym
        na = _infer_num_actions(sd, fallback=1 + 2 * nsym)
        arch = _infer_arch(sd)
        h = _infer_hidden_size(sd, arch=arch)
        policy = TradingPolicy(obs_size, na, hidden=h).to(device)
        policy.load_state_dict(sd)
        policy.eval()

        def policy_fn(obs):
            t = torch.from_numpy(obs.astype(np.float32)).to(device).view(1, -1)
            with torch.no_grad():
                logits, _ = policy(t)
            return int(torch.argmax(logits, dim=-1).item())

        results = {}
        for pname, steps in SIM_PERIODS.items():
            if data.num_timesteps < steps + 1:
                results[pname] = {"error": "data too short"}
                continue
            tail = _slice_tail(data, steps=steps)
            sim = simulate_daily_policy(
                tail, policy_fn, max_steps=steps,
                fee_rate=FEE_RATE, fill_buffer_bps=FILL_BUFFER_BPS,
                max_leverage=MAX_LEVERAGE, periods_per_year=PERIODS_PER_YEAR,
            )
            ann = annualize_total_return(
                float(sim.total_return), periods=float(steps),
                periods_per_year=PERIODS_PER_YEAR,
            )
            results[pname] = {
                "return_pct": sim.total_return * 100,
                "annualized_pct": ann * 100,
                "sortino": sim.sortino,
                "max_dd_pct": sim.max_drawdown * 100,
                "trades": sim.num_trades,
                "wr": sim.win_rate,
            }

        r60 = results.get("60d", {})
        r90 = results.get("90d", {})
        r120 = results.get("120d", {})
        r180 = results.get("180d", {})

        def safe_fmt(d, key):
            v = d.get(key)
            return f"{v:.1f}" if v is not None else "N/A"

        print(f"  MarketSim: 60d={safe_fmt(r60, 'return_pct')}%, "
              f"90d={safe_fmt(r90, 'return_pct')}%, "
              f"120d={safe_fmt(r120, 'return_pct')}%, "
              f"180d={safe_fmt(r180, 'return_pct')}%")

        return {
            "sim_60d_ret_pct": r60.get("return_pct"),
            "sim_60d_sortino": r60.get("sortino"),
            "sim_90d_ret_pct": r90.get("return_pct"),
            "sim_90d_sortino": r90.get("sortino"),
            "sim_120d_ret_pct": r120.get("return_pct"),
            "sim_120d_sortino": r120.get("sortino"),
            "sim_120d_max_dd_pct": r120.get("max_dd_pct"),
            "sim_120d_trades": r120.get("trades"),
            "sim_120d_wr": r120.get("wr"),
            "sim_180d_ret_pct": r180.get("return_pct"),
            "sim_180d_sortino": r180.get("sortino"),
        }
    except Exception as e:
        print(f"  MarketSim ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def append_leaderboard(row: dict) -> None:
    """Append a row to the leaderboard CSV, creating headers if needed."""
    file_exists = LEADERBOARD.exists() and LEADERBOARD.stat().st_size > 0
    with open(LEADERBOARD, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_leaderboard() -> None:
    """Print final leaderboard sorted by 120d Sortino."""
    if not LEADERBOARD.exists():
        print("No leaderboard file found.")
        return

    rows = []
    with open(LEADERBOARD, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def sort_key(r):
        try:
            return float(r.get("sim_120d_sortino") or "-inf")
        except (ValueError, TypeError):
            return float("-inf")

    rows.sort(key=sort_key, reverse=True)

    print(f"\n{'='*100}")
    print(f"  LEADERBOARD -- sorted by 120d Sortino ({len(rows)} configs)")
    print(f"{'='*100}")
    print(f"{'Name':<18} {'TP':>5} {'Seed':>5} "
          f"{'ValRet':>8} {'60dRet%':>8} {'90dRet%':>8} "
          f"{'120dRet%':>9} {'120dSort':>9} {'120dDD%':>8} "
          f"{'180dRet%':>9}")
    print("-" * 100)

    for r in rows:
        print(f"{r.get('name', '?'):<18} "
              f"{r.get('trade_penalty', '?'):>5} "
              f"{r.get('seed', '?'):>5} "
              f"{fmt_val(r.get('val_return', ''))}  "
              f"{fmt_val(r.get('sim_60d_ret_pct', ''))}  "
              f"{fmt_val(r.get('sim_90d_ret_pct', ''))}  "
              f"{fmt_val(r.get('sim_120d_ret_pct', ''), 9)}  "
              f"{fmt_val(r.get('sim_120d_sortino', ''), 9)}  "
              f"{fmt_val(r.get('sim_120d_max_dd_pct', ''))}  "
              f"{fmt_val(r.get('sim_180d_ret_pct', ''), 9)}")


def print_tp_summary():
    """Print per-TP aggregate (mean of seeds)."""
    if not LEADERBOARD.exists():
        return

    rows = []
    with open(LEADERBOARD, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    tp_groups = defaultdict(list)
    for r in rows:
        tp_groups[r.get("trade_penalty", "")].append(r)

    print(f"\n{'='*80}")
    print(f"  PER-TP SUMMARY (mean across seeds)")
    print(f"{'='*80}")
    print(f"{'TP':>6} {'#Seeds':>7} {'ValRet':>8} {'120dRet%':>9} "
          f"{'120dSort':>9} {'120dDD%':>8} {'90dRet%':>8}")
    print("-" * 60)

    summary_keys = ("val_return", "sim_120d_ret_pct", "sim_120d_sortino",
                    "sim_120d_max_dd_pct", "sim_90d_ret_pct")

    summaries = []
    for tp_str, group in sorted(tp_groups.items()):
        means = []
        for key in summary_keys:
            values = []
            for r in group:
                try:
                    values.append(float(r.get(key, "nan")))
                except (ValueError, TypeError):
                    pass
            means.append(np.mean(values) if values else float("nan"))
        summaries.append((tp_str, len(group), *means))

    # Sort by mean 120d sortino (index 4)
    summaries.sort(key=lambda x: x[4] if not np.isnan(x[4]) else float("-inf"), reverse=True)

    for tp_str, n, val_ret, r120, s120, dd120, r90 in summaries:
        print(f"{tp_str:>6} {n:>7} {fmt_val(val_ret)} {fmt_val(r120, 9)} "
              f"{fmt_val(s120, 9)} {fmt_val(dd120)} {fmt_val(r90)}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    os.chdir(REPO)

    # Verify data files exist
    for path in (TRAIN_DATA, VAL_DATA):
        if not Path(path).exists():
            print(f"ERROR: data file not found: {path}")
            sys.exit(1)

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    # Pre-load market sim dependencies once (avoid repeated data reads)
    import src.market_sim_early_exit as _mse
    _orig_early_exit = _mse.evaluate_drawdown_vs_profit_early_exit

    def _no_early_exit(*args, **kwargs):
        return _mse.EarlyExitDecision(
            should_stop=False, progress_fraction=0.0,
            total_return=0.0, max_drawdown=0.0,
        )
    _mse.evaluate_drawdown_vs_profit_early_exit = _no_early_exit

    from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
    from pufferlib_market.metrics import annualize_total_return
    from pufferlib_market.evaluate_tail import (
        TradingPolicy, _infer_num_actions, _infer_arch,
        _infer_hidden_size, _slice_tail,
    )

    sim_deps = (simulate_daily_policy, annualize_total_return,
                TradingPolicy, _infer_num_actions, _infer_arch,
                _infer_hidden_size, _slice_tail)

    val_data = read_mktd(Path(VAL_DATA))

    combos = [(tp, seed) for tp in TRADE_PENALTIES for seed in SEEDS]
    total = len(combos)

    print(f"Trade Penalty Fine-Grain Sweep")
    print(f"  TPs: {TRADE_PENALTIES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total configs: {total}")
    print(f"  Time budget per config: {TIME_BUDGET_S}s")
    print(f"  Estimated total: {total * TIME_BUDGET_S / 60:.0f} min (training only)")
    print(f"  Checkpoint root: {CKPT_ROOT}")
    print(f"  Leaderboard: {LEADERBOARD}")

    for i, (tp, seed) in enumerate(combos):
        name = ckpt_dir_name(tp, seed)
        print(f"\n[{i+1}/{total}] {name}")

        # Check if already done (checkpoint exists)
        already_trained = resolve_checkpoint(tp, seed) is not None

        if already_trained:
            print(f"  Checkpoint already exists, skipping training")
            train_result = {}
        else:
            train_result = run_training(tp, seed)
            if "error" in train_result:
                row = {
                    "name": name, "trade_penalty": tp, "seed": seed,
                    "error": train_result["error"],
                }
                append_leaderboard(row)
                continue

        # C-env evaluation
        print(f"  Evaluating (C env)...")
        eval_result = run_c_env_eval(tp, seed)

        # Market sim evaluation
        print(f"  Evaluating (market sim)...")
        sim_result = run_market_sim_eval(tp, seed, val_data, sim_deps)

        # Merge and log
        row = {"name": name, "trade_penalty": tp, "seed": seed}
        row.update(train_result)
        row.update(eval_result)
        row.update({k: v for k, v in sim_result.items() if k != "error" or "error" not in row})
        append_leaderboard(row)

    # Restore early exit
    _mse.evaluate_drawdown_vs_profit_early_exit = _orig_early_exit

    # Final reports
    print_leaderboard()
    print_tp_summary()


if __name__ == "__main__":
    main()
