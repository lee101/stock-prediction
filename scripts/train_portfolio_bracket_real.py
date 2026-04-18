#!/usr/bin/env python3
"""Train PortfolioBracketEnv PPO on REAL prices from a pufferlib_market .bin.

Loads OHLCV from any MKTD .bin, splits train/val along time, trains PPO,
then runs a deterministic eval rollout on the held-out tape.

Quick run:
    python scripts/train_portfolio_bracket_real.py \\
        --bin pufferlib_market/data/screened32_augmented_train.bin \\
        --val-bin pufferlib_market/data/screened32_full_val.bin \\
        --B 1024 --rollout-steps 64 --iters 200

This is the first end-to-end pass with real data. Goal: confirm PPO doesn't
NaN on a real OHLC tape and that final mean reward is positive on a
deterministic eval rollout. Apples-to-apples comparison vs the +32%/mo XGB
champion comes after this baseline is stable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import gpu_trading_env
from gpu_trading_env.ppo_trainer import (
    PortfolioBracketActor, PPOConfig,
    collect_rollout, compute_gae, ppo_update,
)


def deterministic_eval(env, policy, num_steps: int = 21,
                       random_starts: bool = True, seed: int = 0) -> dict:
    """Run policy.mu (no noise) for num_steps starting at random t_idx per env.

    Each env => one independent ``num_steps``-day episode return.
    With num_steps=21 and an OOS tape, the median across B envs is the
    apples-to-apples monthly return that compares to the XGB +32%/mo headline.
    """
    B = env.B
    T = env.T
    # Reset all envs first, then optionally randomize their start time.
    env.reset()
    if random_starts:
        g = torch.Generator(device=env.prices.device).manual_seed(seed)
        max_start = max(1, T - num_steps - 2)
        starts = torch.randint(1, max_start, (B,), generator=g, device=env.prices.device,
                               dtype=torch.int32)
        env.state["t_idx"].copy_(starts)
    init_eq = env.state["equity"].clone()
    obs = torch.nan_to_num(env.obs(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    with torch.no_grad():
        for _ in range(num_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                mu, _v = policy(obs)
            action = mu.float()
            r, d, _info = env.step(action)
            obs = torch.nan_to_num(env.obs(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            bankrupt = env.state["equity"] <= 1e-3
            if bankrupt.any():
                env.reset(bankrupt)
    final_eq = env.state["equity"]
    pct_return = (final_eq - init_eq) / init_eq.clamp_min(1e-6)
    pct_return = torch.nan_to_num(pct_return, nan=0.0, posinf=10.0, neginf=-1.0)
    # Cap absurd outliers at +500% / -100% — beyond that is fp32 garbage.
    pct_return = pct_return.clamp(-1.0, 5.0)
    return {
        "mean_return": pct_return.mean().item(),
        "median_return": pct_return.median().item(),
        "p10_return": pct_return.quantile(0.10).item(),
        "p25_return": pct_return.quantile(0.25).item(),
        "p75_return": pct_return.quantile(0.75).item(),
        "p90_return": pct_return.quantile(0.90).item(),
        "frac_negative": (pct_return < 0).float().mean().item(),
        "final_eq_mean": final_eq.mean().item(),
        "num_steps": num_steps,
        "num_envs": B,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True, help="train .bin path")
    p.add_argument("--val-bin", default=None, help="val .bin (optional, defaults to train tail)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--B", type=int, default=1024)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--episode-len", type=int, default=126)
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=0.278)
    p.add_argument("--fb-bps", type=float, default=5.0)
    p.add_argument("--out-dir", default="experiments/portfolio_bracket_ppo_v1")
    p.add_argument("--bps-scale", type=float, default=30.0,
                   help="max |limit-offset| in bps from prev_close")
    p.add_argument("--qty-scale", type=float, default=0.3,
                   help="max buy/sell fraction of equity per symbol per bar")
    p.add_argument("--log-std-init", type=float, default=-1.5,
                   help="initial log std of action noise (smaller = less exploration)")
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=25,
                   help="val eval + best-ckpt check every N iters (0 = off)")
    p.add_argument("--val-B", type=int, default=512)
    p.add_argument("--val-steps", type=int, default=21)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Need CUDA")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] bin={args.bin}")
    train_data = gpu_trading_env.load_bin_full(args.bin)
    print(f"[train] T={train_data['T']} S={train_data['S']} F={train_data['F']}")
    print(f"  symbols: {train_data['symbols'][:6]}... ({len(train_data['symbols'])} total)")

    # Use last val_frac of train for val if --val-bin not given.
    if args.val_bin:
        print(f"[load] val_bin={args.val_bin}")
        val_data = gpu_trading_env.load_bin_full(args.val_bin)
        train_prices = train_data["prices"]
        train_tradable = train_data["tradable"]
        train_features = train_data["features"]
        val_prices = val_data["prices"]
        val_tradable = val_data["tradable"]
        val_features = val_data["features"]
    else:
        T = train_data["T"]
        split = int(T * (1.0 - args.val_frac))
        train_prices = train_data["prices"][:split]
        train_tradable = train_data["tradable"][:split]
        train_features = train_data["features"][:split]
        val_prices = train_data["prices"][split:]
        val_tradable = train_data["tradable"][split:]
        val_features = train_data["features"][split:]
        print(f"[split] train_T={split}, val_T={T-split}")

    train_env = gpu_trading_env.make_portfolio_bracket(
        B=args.B,
        prices=train_prices, tradable_tape=train_tradable,
        features=train_features,
        params={
            "episode_len": args.episode_len,
            "fee_bps": args.fee_bps,
            "fill_buffer_bps": args.fb_bps,
            "max_leverage": args.max_leverage,
        },
    )
    obs_dim = train_env.obs().shape[-1]
    S = train_env.S
    print(f"[env] obs_dim={obs_dim} S={S} action=[B, {S}, 4]")

    policy = PortfolioBracketActor(
        obs_dim=obs_dim, num_symbols=S, hidden_dim=256, num_layers=3,
        bps_scale=args.bps_scale, qty_scale=args.qty_scale,
        log_std_init=args.log_std_init,
    ).to("cuda")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[policy] {n_params:,} params")

    cfg = PPOConfig(
        rollout_steps=args.rollout_steps, epochs=args.epochs, minibatches=4,
        lr=args.lr, ent_coef=args.ent_coef, reward_scale=args.reward_scale,
        target_kl=args.target_kl, bf16=True, log_every=10,
    )
    print(f"[ppo] iters={args.iters} rollout={cfg.rollout_steps} lr={cfg.lr} bf16={cfg.bf16}")

    # Persistent val env — reused across periodic eval calls.
    val_T = val_prices.size(0)
    eval_steps = args.val_steps
    val_env = gpu_trading_env.make_portfolio_bracket(
        B=args.val_B,
        prices=val_prices, tradable_tape=val_tradable,
        features=val_features,
        params={
            "episode_len": val_T,  # never auto-reset during eval
            "fee_bps": args.fee_bps,
            "fill_buffer_bps": args.fb_bps,
            "max_leverage": args.max_leverage,
        },
    )

    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    history = {
        "iter": [], "mean_reward": [], "pg_loss": [], "vf_loss": [],
        "entropy": [], "approx_kl": [],
        "val_iter": [], "val_median": [], "val_mean": [], "val_p10": [],
        "val_frac_negative": [],
    }
    best_val = {"median": -1e9, "iter": -1, "stats": None}

    torch.cuda.synchronize()
    t0 = time.time()
    for it in range(args.iters):
        buf, last_value = collect_rollout(train_env, policy, cfg)
        adv, ret = compute_gae(buf, last_value, cfg)
        stats = ppo_update(buf, adv, ret, policy, optim, cfg)
        mean_r = buf.rewards.mean().item()
        history["iter"].append(it)
        history["mean_reward"].append(mean_r)
        history["pg_loss"].append(stats["pg_loss"])
        history["vf_loss"].append(stats["vf_loss"])
        history["entropy"].append(stats["entropy"])
        history["approx_kl"].append(stats["approx_kl"])
        if (it + 1) % cfg.log_every == 0:
            print(
                f"[ppo it={it+1}/{args.iters}] mean_r={mean_r:+.5f} "
                f"pg={stats['pg_loss']:+.4f} vf={stats['vf_loss']:.4f} "
                f"ent={stats['entropy']:+.3f} kl={stats['approx_kl']:+.4f}",
                flush=True,
            )
        if args.val_every > 0 and (it + 1) % args.val_every == 0:
            val_stats = deterministic_eval(val_env, policy, num_steps=eval_steps,
                                           random_starts=True, seed=0)
            history["val_iter"].append(it)
            history["val_median"].append(val_stats["median_return"])
            history["val_mean"].append(val_stats["mean_return"])
            history["val_p10"].append(val_stats["p10_return"])
            history["val_frac_negative"].append(val_stats["frac_negative"])
            improved = val_stats["median_return"] > best_val["median"]
            marker = " *BEST*" if improved else ""
            print(
                f"  [val it={it+1}] median={val_stats['median_return']:+.4f} "
                f"mean={val_stats['mean_return']:+.4f} "
                f"p10={val_stats['p10_return']:+.4f} "
                f"frac_neg={val_stats['frac_negative']:.3f}{marker}",
                flush=True,
            )
            if improved:
                best_val = {"median": val_stats["median_return"],
                            "iter": it, "stats": val_stats}
                torch.save(
                    {"policy_state": policy.state_dict(), "cfg": cfg.__dict__,
                     "obs_dim": obs_dim, "num_symbols": S, "args": vars(args),
                     "val_stats": val_stats, "iter": it},
                    out / "policy_best.pt",
                )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    hist = history
    n_q = max(1, len(hist["mean_reward"]) // 4)
    early_r = float(np.mean(hist["mean_reward"][:n_q]))
    late_r = float(np.mean(hist["mean_reward"][-n_q:]))
    total_steps = args.iters * args.rollout_steps * args.B
    print(f"\n[train done] {elapsed:.1f}s  iter/s={args.iters/elapsed:.2f}  "
          f"M-step/s={total_steps/elapsed/1e6:.2f}")
    print(f"[train reward] early-q: {early_r:+.5f} -> late-q: {late_r:+.5f} "
          f"(delta {late_r-early_r:+.5f})")
    if best_val["iter"] >= 0:
        print(f"[best val] iter={best_val['iter']+1} "
              f"median={best_val['median']:+.4f} "
              f"(checkpoint: {out}/policy_best.pt)")

    # Final eval (for reference, in addition to the best-checkpoint one).
    print(f"\n[eval final] {eval_steps}-day eval on val tape, B={val_env.B}")
    eval_stats = deterministic_eval(val_env, policy, num_steps=eval_steps,
                                    random_starts=True, seed=0)
    print(f"[eval final result] {json.dumps(eval_stats, indent=2)}")

    # Persist artifacts.
    ckpt = {"policy_state": policy.state_dict(), "cfg": cfg.__dict__,
            "obs_dim": obs_dim, "num_symbols": S, "args": vars(args)}
    torch.save(ckpt, out / "policy.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    (out / "eval.json").write_text(json.dumps(eval_stats, indent=2))
    if best_val["stats"] is not None:
        (out / "eval_best.json").write_text(json.dumps(best_val["stats"], indent=2))
    print(f"[saved] {out}/policy.pt + history.json + eval.json "
          f"(+ policy_best.pt, eval_best.json if val_every>0)")


if __name__ == "__main__":
    main()
