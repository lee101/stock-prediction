#!/usr/bin/env python3
"""Fast seed sweep for stocks12 PPO: screen many seeds at 3M steps, continue promising ones to 35M.

Strategy based on production finding:
- s123 (production): best_return=0.051 at 1.44M steps → 0/111 neg
- Bad seeds: high best_return at 5-20M steps (overfitting)
- Key insight: LOW in-sample return + LOW val_neg at ~3M steps = generalization candidate

Usage:
    source .venv313/bin/activate
    python scripts/stocks12_fast_sweep.py \
        --seed-start 300 --seed-end 400 \
        --gpu-slots 8 \
        --screen-steps 3000000 --full-steps 35000000 \
        --checkpoint-root pufferlib_market/checkpoints/stocks12_sweep \
        --val-data pufferlib_market/data/stocks12_daily_val.bin \
        --data pufferlib_market/data/stocks12_daily_train.bin
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def val_eval_quick(ckpt_path: Path, val_data_path: str, n_windows: int = 20) -> dict:
    """Run quick val eval on a checkpoint. Returns dict with neg, med, trades_avg."""
    from pufferlib_market.evaluate_holdout import (
        _infer_arch, _infer_hidden_size, _infer_num_actions, _slice_window, TradingPolicy
    )
    from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy

    torch.set_num_threads(1)  # prevent PyTorch CPU thread pool deadlock in main process
    device = torch.device("cpu")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = payload.get("model", payload)
    hs = _infer_hidden_size(state_dict, arch="mlp")
    obs_size = list(state_dict.values())[0].shape[1]
    n_actions = _infer_num_actions(state_dict, fallback=25)
    policy = TradingPolicy(obs_size=obs_size, hidden=hs, num_actions=n_actions).to(device).eval()
    missing, _ = policy.load_state_dict(state_dict, strict=False)
    if hasattr(policy, "_use_encoder_norm"):
        if isinstance(payload, dict) and "use_encoder_norm" in payload:
            policy._use_encoder_norm = bool(payload["use_encoder_norm"])
        else:
            policy._use_encoder_norm = "encoder_norm.weight" not in missing

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device).view(1, -1)
        with torch.inference_mode():
            logits, _ = policy(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    val_data = read_mktd(val_data_path)
    T = val_data.num_timesteps
    eval_steps = 90
    all_starts = list(range(T - eval_steps))
    rng = np.random.default_rng(42)
    starts = sorted(rng.choice(all_starts, size=min(n_windows, len(all_starts)), replace=False).tolist())

    returns = []
    trades = []
    for start in starts:
        w = _slice_window(val_data, start=start, steps=eval_steps)
        r = simulate_daily_policy(
            w, policy_fn, max_steps=eval_steps, fee_rate=0.001, fill_buffer_bps=5.0,
            enable_drawdown_profit_early_exit=False,
        )
        returns.append(r.total_return)
        trades.append(r.num_trades)

    returns = np.array(returns)
    trades_avg = float(np.mean(trades))
    best_return = float(payload.get("best_return", 0.0)) if isinstance(payload, dict) else 0.0
    return {
        "med": float(np.median(returns) * 100),
        "p10": float(np.percentile(returns, 10) * 100),
        "neg": int((returns < 0).sum()),
        "n": len(returns),
        "trades_avg": trades_avg,
        "best_return": best_return,
        "global_step": int(payload.get("global_step", 0)) if isinstance(payload, dict) else 0,
    }


def build_train_cmd(
    seed: int,
    ckpt_dir: Path,
    data_path: str,
    val_data_path: str,
    total_steps: int,
    resume_from: str | None = None,
    extra_args: list[str] | None = None,
    trade_penalty: float = 0.05,
    ent_coef: float = 0.05,
    val_neg_threshold: int = 0,
    val_neg_patience: int = 5,
) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.train",
        "--data-path", data_path,
        "--total-timesteps", str(total_steps),
        "--hidden-size", "1024",
        "--num-envs", "128",
        "--anneal-lr",
        "--ent-coef", str(ent_coef),
        "--trade-penalty", str(trade_penalty),
        "--fee-rate", "0.001",
        "--max-steps", "252",
        "--seed", str(seed),
        "--val-data-path", val_data_path,
        "--val-eval-interval", "50",
        "--val-eval-windows", "50",
        "--checkpoint-dir", str(ckpt_dir),
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if val_neg_threshold > 0:
        cmd += ["--early-stop-val-neg-threshold", str(val_neg_threshold),
                "--early-stop-val-neg-patience", str(val_neg_patience)]
    if extra_args:
        cmd += extra_args
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Fast stocks12 seed sweep with screening")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--gpu-slots", type=int, default=6, help="Max concurrent training processes")
    parser.add_argument("--screen-steps", type=int, default=3_000_000, help="Steps for initial screening")
    parser.add_argument("--full-steps", type=int, default=35_000_000, help="Steps for full training")
    parser.add_argument("--checkpoint-root", required=True, help="Root dir for checkpoints")
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--val-data", required=True, help="Validation data path")
    parser.add_argument("--screen-val-windows", type=int, default=20)
    parser.add_argument("--max-neg-pct", type=float, default=0.45,
                        help="Max fraction of negative windows at screen to continue (default 0.45)")
    parser.add_argument("--max-best-return", type=float, default=0.3,
                        help="Max in-sample best_return at screen (higher = overfitting risk)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval seconds")
    parser.add_argument("--trade-penalty", type=float, default=0.05,
                        help="Trade penalty hyperparameter (default 0.05)")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy coefficient (default 0.05)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Tag for checkpoint subdirs, e.g. 'tp03' (default: auto from trade-penalty)")
    parser.add_argument("--extra-train-args", type=str, default="",
                        help="Extra space-separated args to pass to pufferlib_market.train, e.g. '--advantage-norm group_relative --group-relative-mix 1.0'")
    parser.add_argument("--val-neg-threshold", type=int, default=30,
                        help="Early-stop full training if val_neg > this for --val-neg-patience consecutive evals. "
                             "Default 30 (out of 50 val windows). 0 disables.")
    parser.add_argument("--val-neg-patience", type=int, default=5,
                        help="Consecutive val evals above threshold before early stopping (default 5).")
    parser.add_argument("--min-screen-med", type=float, default=1.0,
                        help="Min median val return %% at screen to qualify (default 1.0%%). "
                             "Filters do-nothing models with 0%% median.")
    parser.add_argument("--min-screen-trades", type=float, default=3.0,
                        help="Min avg trades per 90d window at screen to qualify (default 3.0). "
                             "Filters collapsed/degenerate models.")
    args = parser.parse_args()

    # Auto-generate run tag if not specified
    if args.run_tag is None:
        tp_str = f"tp{int(args.trade_penalty * 100):02d}"
        ent_str = f"ent{int(args.ent_coef * 100):02d}" if args.ent_coef != 0.05 else ""
        args.run_tag = tp_str + (f"_{ent_str}" if ent_str else "")

    ckpt_root = Path(args.checkpoint_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    screen_threshold = int(args.max_neg_pct * args.screen_val_windows)
    import shutil  # import here once for use below

    print(f"Sweep: seeds {args.seed_start}-{args.seed_end} ({len(seeds)} total) tag={args.run_tag}")
    print(f"HParams: trade_penalty={args.trade_penalty} ent_coef={args.ent_coef} extra={args.extra_train_args!r}")
    print(f"Screen: {args.screen_steps/1e6:.1f}M steps, val_neg<={screen_threshold}/{args.screen_val_windows} min_med={args.min_screen_med:.1f}% min_trades={args.min_screen_trades:.1f}")
    print(f"Full: {args.full_steps/1e6:.1f}M steps for qualified seeds (early-stop val_neg>{args.val_neg_threshold}/{50} for {args.val_neg_patience} evals)")
    print(f"GPU slots: {args.gpu_slots}")

    # Track state
    running: dict[int, subprocess.Popen] = {}   # seed -> process (screening phase)
    running_full: dict[int, subprocess.Popen] = {}  # seed -> process (full phase)
    screened: set[int] = set()   # seeds that finished screening
    qualified: set[int] = set()  # seeds that passed screening
    rejected: set[int] = set()   # seeds that failed screening
    done: set[int] = set()       # seeds that completed full training
    seed_queue = list(seeds)

    log_path = ckpt_root / "sweep_log.csv"
    with open(log_path, "w") as f:
        f.write("timestamp,seed,phase,result,neg,med,best_return,global_step\n")

    def log(seed, phase, result, neg=-1, med=0.0, best_return=0.0, global_step=0, trades_avg=0.0):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(log_path, "a") as f:
            f.write(f"{ts},{seed},{phase},{result},{neg},{med:.2f},{best_return:.4f},{global_step}\n")
        print(f"  [{ts}] seed={seed} {phase}={result} neg={neg} med={med:.1f}% trades={trades_avg:.1f} best_ret={best_return:.3f}", flush=True)

    # max_full_concurrent: full training takes priority but leaves 1 slot for screening
    max_full_concurrent = max(1, args.gpu_slots - 1)

    while seed_queue or running or running_full:
        # --- Step 1: pre-process queue (CPU-only: skip done seeds, eval already-screened) ---
        i = 0
        while i < len(seed_queue):
            seed = seed_queue[i]
            ckpt_dir = ckpt_root / f"{args.run_tag}_s{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            final = ckpt_dir / "final.pt"
            if final.exists():
                seed_queue.pop(i)
                done.add(seed)
                print(f"  seed={seed}: already has final.pt, skipping")
                continue
            screen_best_neg = ckpt_dir / "screen_best_neg.pt"
            using_best_neg = screen_best_neg.exists()
            screen_ckpt = screen_best_neg if using_best_neg else ckpt_dir / "screen_best.pt"
            if screen_ckpt.exists() and seed not in screened:
                try:
                    ev = val_eval_quick(screen_ckpt, args.val_data, args.screen_val_windows)
                    screened.add(seed)
                    pass_val = ev["neg"] <= screen_threshold
                    # max_best_return only applies to screen_best.pt (in-sample return as fraction).
                    # For screen_best_neg.pt, best_return stores val_med (pct) — skip this filter.
                    pass_ret = True if using_best_neg else (ev["best_return"] <= args.max_best_return)
                    pass_med = ev["med"] >= args.min_screen_med
                    pass_trades = ev["trades_avg"] >= args.min_screen_trades
                    if pass_val and pass_ret and pass_med and pass_trades:
                        qualified.add(seed)
                        log(seed, "screen", "QUALIFIED", ev["neg"], ev["med"], ev["best_return"], ev["global_step"], ev["trades_avg"])
                    else:
                        rejected.add(seed)
                        reasons = []
                        if not pass_val: reasons.append("HIGH_NEG")
                        if not pass_ret: reasons.append("HIGH_RET")
                        if not pass_med: reasons.append("LOW_MED")
                        if not pass_trades: reasons.append("LOW_TRADES")
                        log(seed, "screen", f"REJECTED_{'_'.join(reasons)}", ev["neg"], ev["med"], ev["best_return"], ev["global_step"], ev["trades_avg"])
                    seed_queue.pop(i)
                    continue
                except Exception as e:
                    print(f"  seed={seed}: error evaluating screen_ckpt: {e}")
            i += 1

        # --- Step 2: check completed screenings ---
        finished_screen = []
        for seed, proc in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                finished_screen.append(seed)
                ckpt_dir = ckpt_root / f"{args.run_tag}_s{seed}"
                screened.add(seed)

                # Rename best.pt → screen_best.pt (preserve for analysis)
                # Also copy best_neg.pt → screen_best_neg.pt so full training doesn't overwrite it
                best_pt = ckpt_dir / "best.pt"
                if best_pt.exists():
                    best_pt.rename(ckpt_dir / "screen_best.pt")
                else:
                    log(seed, "screen", "NO_CKPT")
                    rejected.add(seed)
                    continue
                best_neg_pt = ckpt_dir / "best_neg.pt"
                if best_neg_pt.exists():
                    shutil.copy2(best_neg_pt, ckpt_dir / "screen_best_neg.pt")

                # Use screen_best_neg.pt (lowest OOS-neg during screen) for qualification.
                # Falls back to screen_best.pt if not available.
                screen_best_neg = ckpt_dir / "screen_best_neg.pt"
                using_best_neg = screen_best_neg.exists()
                best_pt_to_eval = screen_best_neg if using_best_neg else ckpt_dir / "screen_best.pt"

                try:
                    ev = val_eval_quick(best_pt_to_eval, args.val_data, args.screen_val_windows)
                    pass_val = ev["neg"] <= screen_threshold
                    # max_best_return only applies to screen_best.pt (in-sample return as fraction).
                    # For screen_best_neg.pt, best_return stores val_med (pct) — skip this filter.
                    pass_ret = True if using_best_neg else (ev["best_return"] <= args.max_best_return)
                    pass_med = ev["med"] >= args.min_screen_med
                    pass_trades = ev["trades_avg"] >= args.min_screen_trades
                    if pass_val and pass_ret and pass_med and pass_trades:
                        qualified.add(seed)
                        log(seed, "screen", "QUALIFIED", ev["neg"], ev["med"], ev["best_return"], ev["global_step"], ev["trades_avg"])
                    else:
                        rejected.add(seed)
                        reasons = []
                        if not pass_val: reasons.append("HIGH_NEG")
                        if not pass_ret: reasons.append("HIGH_RET")
                        if not pass_med: reasons.append("LOW_MED")
                        if not pass_trades: reasons.append("LOW_TRADES")
                        log(seed, "screen", f"REJECTED_{'_'.join(reasons)}", ev["neg"], ev["med"], ev["best_return"], ev["global_step"], ev["trades_avg"])
                except Exception as e:
                    print(f"  seed={seed}: error evaluating: {e}")
                    rejected.add(seed)

        for seed in finished_screen:
            del running[seed]

        # --- Step 3: check completed full trainings ---
        for seed, proc in list(running_full.items()):
            ret = proc.poll()
            if ret is not None:
                del running_full[seed]
                done.add(seed)
                ckpt_dir = ckpt_root / f"{args.run_tag}_s{seed}"
                try:
                    best_pt = ckpt_dir / "best.pt"
                    if best_pt.exists():
                        ev = val_eval_quick(best_pt, args.val_data, args.screen_val_windows)
                        log(seed, "full", "DONE", ev["neg"], ev["med"], ev["best_return"], ev["global_step"], ev["trades_avg"])
                except Exception as e:
                    print(f"  seed={seed}: error evaluating final: {e}")
                    log(seed, "full", "DONE_NO_EVAL")

        # --- Step 4: launch full training for qualified seeds (priority over screening) ---
        # Caps at max_full_concurrent to leave ≥1 slot for screening
        for seed in sorted(qualified):
            if seed in done or seed in running_full:
                continue
            if len(running_full) >= max_full_concurrent:
                break
            total_active = len(running) + len(running_full)
            if total_active >= args.gpu_slots:
                break
            ckpt_dir = ckpt_root / f"{args.run_tag}_s{seed}"
            screen_best = ckpt_dir / "screen_best.pt"
            remaining_steps = args.full_steps - args.screen_steps
            if remaining_steps <= 0:
                done.add(seed)
                continue
            # Restore screen_best as best.pt for resume
            if screen_best.exists():
                shutil.copy(screen_best, ckpt_dir / "best.pt")
            extra = args.extra_train_args.split() if args.extra_train_args.strip() else None
            cmd = build_train_cmd(
                seed, ckpt_dir, args.data, args.val_data, remaining_steps,
                resume_from=str(ckpt_dir / "best.pt"),
                trade_penalty=args.trade_penalty, ent_coef=args.ent_coef,
                val_neg_threshold=getattr(args, "val_neg_threshold", 30),
                val_neg_patience=getattr(args, "val_neg_patience", 5),
                extra_args=extra,
            )
            logfile = open(ckpt_dir / "full.log", "w")
            proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent.parent))
            running_full[seed] = proc
            print(f"  Launched FULL training seed={seed} (PID {proc.pid})", flush=True)

        # --- Step 5: launch new screening jobs (fill remaining slots) ---
        # Reserve GPU slots for any qualified seeds waiting for full training
        full_waiting_count = sum(1 for s in qualified if s not in done and s not in running_full)
        full_slots_needed = max(0, min(max_full_concurrent, full_waiting_count) - len(running_full))
        screen_cap = args.gpu_slots - full_slots_needed
        total_active = len(running) + len(running_full)
        while seed_queue and total_active < screen_cap:
            seed = seed_queue[0]
            ckpt_dir = ckpt_root / f"{args.run_tag}_s{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # (Already-done/already-screened seeds were removed in step 1 — launch screen directly)
            seed_queue.pop(0)
            extra = args.extra_train_args.split() if args.extra_train_args.strip() else None
            screen_neg_thresh = int(0.9 * args.screen_val_windows)  # stop screen if >90% neg
            cmd = build_train_cmd(
                seed, ckpt_dir, args.data, args.val_data, args.screen_steps,
                trade_penalty=args.trade_penalty, ent_coef=args.ent_coef,
                val_neg_threshold=screen_neg_thresh,
                val_neg_patience=3,
                extra_args=extra,
            )
            logfile = open(ckpt_dir / "screen.log", "w")
            proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent.parent))
            running[seed] = proc
            total_active += 1
            print(f"  Launched screening seed={seed} (PID {proc.pid})", flush=True)

        total_active = len(running) + len(running_full)
        print(f"  Status: {len(seed_queue)} queued, {len(running)} screening, {len(running_full)} full-training, "
              f"{len(qualified)} qualified, {len(rejected)} rejected, {len(done)} done", flush=True)
        time.sleep(args.poll_interval)

    print(f"\nSweep complete!")
    print(f"Qualified ({len(qualified)}): {sorted(qualified)}")
    print(f"Rejected ({len(rejected)}): {sorted(rejected)[:20]}{'...' if len(rejected)>20 else ''}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
