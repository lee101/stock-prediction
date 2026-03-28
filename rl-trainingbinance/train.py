from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_local_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, CURRENT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


try:
    from data import (
        FeatureNormalizer,
        HourlyMarketData,
        apply_feature_normalizer,
        fit_feature_normalizer,
        load_hourly_market_data,
    )
except Exception:
    _data_mod = _load_local_module("rl_trainingbinance_train_data", "data.py")
    FeatureNormalizer = _data_mod.FeatureNormalizer
    HourlyMarketData = _data_mod.HourlyMarketData
    apply_feature_normalizer = _data_mod.apply_feature_normalizer
    fit_feature_normalizer = _data_mod.fit_feature_normalizer
    load_hourly_market_data = _data_mod.load_hourly_market_data

try:
    from env import BinanceHourlyEnv, EnvConfig
except Exception:
    _env_mod = _load_local_module("rl_trainingbinance_train_env", "env.py")
    BinanceHourlyEnv = _env_mod.BinanceHourlyEnv
    EnvConfig = _env_mod.EnvConfig

try:
    from model import PolicyConfig, RiskAwareActorCritic
except Exception:
    _model_mod = _load_local_module("rl_trainingbinance_train_model", "model.py")
    PolicyConfig = _model_mod.PolicyConfig
    RiskAwareActorCritic = _model_mod.RiskAwareActorCritic

try:
    from presets import DEFAULT_BINANCE_HOURLY_SYMBOLS, DEFAULT_SHORTABLE_SYMBOLS, parse_symbols
except Exception:
    _presets_mod = _load_local_module("rl_trainingbinance_train_presets", "presets.py")
    DEFAULT_BINANCE_HOURLY_SYMBOLS = _presets_mod.DEFAULT_BINANCE_HOURLY_SYMBOLS
    DEFAULT_SHORTABLE_SYMBOLS = _presets_mod.DEFAULT_SHORTABLE_SYMBOLS
    parse_symbols = _presets_mod.parse_symbols

try:
    from validate import (
        evaluate_validation_plan,
        flatten_window_summaries,
        parse_csv_ints,
        resolve_stride_hours,
        resolve_window_weights,
    )
except Exception:
    _validate_mod = _load_local_module("rl_trainingbinance_train_validate", "validate.py")
    evaluate_validation_plan = _validate_mod.evaluate_validation_plan
    flatten_window_summaries = _validate_mod.flatten_window_summaries
    parse_csv_ints = _validate_mod.parse_csv_ints
    resolve_stride_hours = _validate_mod.resolve_stride_hours
    resolve_window_weights = _validate_mod.resolve_window_weights


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_market(
    market: HourlyMarketData,
    *,
    lookback: int,
    train_split_ratio: float,
    purge_hours: int,
) -> tuple[HourlyMarketData, HourlyMarketData, dict[str, int]]:
    total = len(market)
    if total <= lookback + purge_hours + 24:
        raise ValueError("Not enough data to create train/validation split.")
    train_ratio = float(train_split_ratio)
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_split_ratio must be in [0.5, 1.0).")
    val_start = int(total * train_ratio)
    val_start = max(val_start, lookback + purge_hours + 1)
    train_end = val_start - purge_hours
    if train_end <= lookback + 2:
        raise ValueError("Training split is too short after applying purge.")
    validation_slice_start = max(0, val_start - lookback)
    train_market = market.slice(0, train_end)
    val_market = market.slice(validation_slice_start, total)
    return train_market, val_market, {
        "train_end": int(train_end),
        "validation_slice_start": int(validation_slice_start),
        "validation_slice_end": int(total),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: RiskAwareActorCritic,
    model_config: PolicyConfig,
    env_config: EnvConfig,
    symbols: list[str],
    shortable_symbols: list[str],
    optimizer: optim.Optimizer | None = None,
    update: int = 0,
    summary: dict[str, float] | None = None,
    validation_market_range: dict[str, int] | None = None,
    feature_normalizer: FeatureNormalizer | None = None,
    validation_window_config: dict[str, Any] | None = None,
    validation_window_summaries: dict[str, dict[str, float]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": model_config.to_dict(),
        "env_config": env_config.to_dict(),
        "symbols": list(symbols),
        "shortable_symbols": list(shortable_symbols),
        "update": int(update),
        "validation_summary": dict(summary or {}),
        "validation_market_range": dict(validation_market_range or {}),
        "feature_normalizer": feature_normalizer.to_dict() if feature_normalizer is not None else None,
        "validation_window_config": dict(validation_window_config or {}),
        "validation_window_summaries": dict(validation_window_summaries or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    _set_seed(int(args.seed))
    if torch.cuda.is_available() and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    elif hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if int(args.validate_every) < 1:
        raise ValueError("validate_every must be >= 1")
    if int(args.save_every) < 1:
        raise ValueError("save_every must be >= 1")
    if args.early_stop_patience is not None and int(args.early_stop_patience) < 1:
        raise ValueError("early_stop_patience must be >= 1 when provided")

    symbols = parse_symbols(args.symbols, default=DEFAULT_BINANCE_HOURLY_SYMBOLS)
    shortable_symbols = parse_symbols(args.shortable_symbols, default=DEFAULT_SHORTABLE_SYMBOLS)
    market = load_hourly_market_data(
        data_root=args.data_root,
        symbols=symbols,
        shortable_symbols=shortable_symbols,
        min_history_hours=int(args.min_history_hours),
    )
    train_market, val_market, split_info = _split_market(
        market,
        lookback=int(args.lookback),
        train_split_ratio=float(args.train_split_ratio),
        purge_hours=int(args.purge_hours),
    )
    feature_normalizer = fit_feature_normalizer(train_market)
    train_market = apply_feature_normalizer(train_market, feature_normalizer)
    val_market = apply_feature_normalizer(val_market, feature_normalizer)

    env_config = EnvConfig(
        lookback=int(args.lookback),
        episode_steps=int(args.episode_steps),
        initial_equity=float(args.initial_equity),
        max_gross_leverage=float(args.max_gross_leverage),
        max_position_weight=float(args.max_position_weight),
        spread_bps=float(args.spread_bps),
        slippage_bps=float(args.slippage_bps),
        short_borrow_apr=float(args.short_borrow_apr),
        margin_apr=float(args.margin_apr),
        downside_penalty=float(args.downside_penalty),
        drawdown_penalty=float(args.drawdown_penalty),
        turnover_penalty=float(args.turnover_penalty),
        concentration_penalty=float(args.concentration_penalty),
        leverage_penalty=float(args.leverage_penalty),
        smoothness_penalty=float(args.smoothness_penalty),
        volatility_penalty=float(args.volatility_penalty),
        reward_scale=float(args.reward_scale),
        random_reset=True,
    )

    model_config = PolicyConfig(
        lookback=int(args.lookback),
        num_assets=train_market.num_assets,
        feature_dim=train_market.feature_dim,
        portfolio_dim=train_market.num_assets + 4,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        mlp_hidden=int(args.mlp_hidden),
    )
    model = RiskAwareActorCritic(model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    validation_window_hours = parse_csv_ints(args.validation_window_hours)
    validation_window_weights = resolve_window_weights(validation_window_hours, args.validation_window_weights)
    validation_stride_hours = resolve_stride_hours(validation_window_hours, args.validation_stride_hours)
    validation_window_config = {
        "window_hours": list(validation_window_hours),
        "window_weights": list(validation_window_weights),
        "stride_hours": [int(item) if item is not None else None for item in validation_stride_hours],
        "purge_hours": int(args.validation_purge_hours),
    }

    envs = [
        BinanceHourlyEnv(
            train_market,
            env_config,
            slice_start=int(args.lookback),
            slice_end=len(train_market) - 1,
            seed=int(args.seed) + env_idx,
        )
        for env_idx in range(int(args.num_envs))
    ]
    obs = np.stack([env.reset() for env in envs], axis=0).astype(np.float32)

    rollout_steps = int(args.rollout_steps)
    num_envs = int(args.num_envs)
    obs_dim = int(envs[0].obs_dim)
    action_dim = int(train_market.num_assets)

    num_updates = int(args.updates)
    gamma = float(args.gamma)
    gae_lambda = float(args.gae_lambda)
    clip_eps = float(args.clip_eps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_summary: dict[str, float] | None = None
    best_window_summaries: dict[str, dict[str, float]] = {}
    last_validation_summary: dict[str, float] | None = None
    last_validation_window_summaries: dict[str, dict[str, float]] = {}
    validation_env_config = EnvConfig(**{**env_config.to_dict(), "random_reset": False})
    shortable_export = [symbol for symbol, allowed in zip(market.symbols, market.shortable_mask) if allowed > 0.5]
    early_stop_patience = None if args.early_stop_patience is None else int(args.early_stop_patience)
    early_stop_min_delta = float(args.early_stop_min_delta)
    no_improve_validations = 0
    completed_updates = 0

    for update in range(1, num_updates + 1):
        completed_updates = update
        buf_obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        buf_actions = np.zeros((rollout_steps, num_envs, action_dim), dtype=np.float32)
        buf_logprob = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        buf_rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        buf_dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        buf_values = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        model.eval()
        for step in range(rollout_steps):
            obs_tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                actions, log_prob, values = model.act(obs_tensor, deterministic=False)
            action_np = actions.cpu().numpy()
            buf_obs[step] = obs
            buf_actions[step] = action_np
            buf_logprob[step] = log_prob.cpu().numpy()
            buf_values[step] = values.cpu().numpy()

            next_obs: list[np.ndarray] = []
            for env_idx, env in enumerate(envs):
                obs_i, reward_i, done_i, _ = env.step(action_np[env_idx])
                buf_rewards[step, env_idx] = float(reward_i)
                buf_dones[step, env_idx] = float(done_i)
                if done_i:
                    obs_i = env.reset()
                next_obs.append(obs_i.astype(np.float32, copy=False))
            obs = np.stack(next_obs, axis=0)

        with torch.no_grad():
            next_value = model.forward(torch.from_numpy(obs).to(device=device, dtype=torch.float32))[1].cpu().numpy()

        advantages = np.zeros_like(buf_rewards)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for step in reversed(range(rollout_steps)):
            next_non_terminal = 1.0 - buf_dones[step]
            next_values = next_value if step == rollout_steps - 1 else buf_values[step + 1]
            delta = buf_rewards[step] + gamma * next_values * next_non_terminal - buf_values[step]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae
        returns = advantages + buf_values

        b_obs = torch.from_numpy(buf_obs.reshape(-1, obs_dim)).to(device=device, dtype=torch.float32)
        b_actions = torch.from_numpy(buf_actions.reshape(-1, action_dim)).to(device=device, dtype=torch.float32)
        b_logprob = torch.from_numpy(buf_logprob.reshape(-1)).to(device=device, dtype=torch.float32)
        b_adv = torch.from_numpy(advantages.reshape(-1)).to(device=device, dtype=torch.float32)
        b_returns = torch.from_numpy(returns.reshape(-1)).to(device=device, dtype=torch.float32)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        model.train()
        batch_size = b_obs.shape[0]
        mb_size = min(int(args.minibatch_size), batch_size)
        losses: dict[str, float] = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        num_minibatches = 0

        for _ in range(int(args.ppo_epochs)):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                idx = perm[start : start + mb_size]
                new_logprob, entropy, value = model.evaluate_actions(b_obs[idx], b_actions[idx])
                ratio = (new_logprob - b_logprob[idx]).exp()
                pg_loss1 = -b_adv[idx] * ratio
                pg_loss2 = -b_adv[idx] * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * torch.square(value - b_returns[idx]).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + float(args.vf_coef) * value_loss - float(args.ent_coef) * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()

                losses["policy"] += float(policy_loss.item())
                losses["value"] += float(value_loss.item())
                losses["entropy"] += float(entropy_loss.item())
                num_minibatches += 1

        for key in losses:
            losses[key] /= max(num_minibatches, 1)

        should_validate = (update % int(args.validate_every) == 0) or (update == num_updates)
        history_row = {
            "update": update,
            "loss_policy": losses["policy"],
            "loss_value": losses["value"],
            "entropy": losses["entropy"],
        }

        if should_validate:
            model.eval()
            evaluation = evaluate_validation_plan(
                market=val_market,
                model=model,
                env_config=validation_env_config,
                window_hours=validation_window_hours,
                purge_hours=int(args.validation_purge_hours),
                stride_hours=validation_stride_hours,
                window_weights=validation_window_weights,
                device=device,
                batch_size=args.validation_batch_size,
            )
            summary = dict(evaluation["aggregate_summary"])
            window_summaries = dict(evaluation["window_summaries"])
            last_validation_summary = dict(summary)
            last_validation_window_summaries = dict(window_summaries)
            history_row.update(summary)
            history_row.update(flatten_window_summaries(window_summaries))
            history.append(history_row)

            horizon_terms = " ".join(
                f"{label}_ret={window_summaries[label]['median_total_return']:+.4f}"
                for label in sorted(window_summaries.keys())
            )
            print(
                "update={update:03d} score={score:+.4f} {horizons} p90_dd={dd:.4f} loss_pi={pi:.4f} loss_v={v:.4f}".format(
                    update=update,
                    score=summary["score"],
                    horizons=horizon_terms,
                    dd=summary["p90_max_drawdown"],
                    pi=losses["policy"],
                    v=losses["value"],
                )
            )

            previous_best_score = best_score
            if summary["score"] > best_score:
                best_score = float(summary["score"])
                best_summary = dict(summary)
                best_window_summaries = dict(window_summaries)
                _save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    model_config=model_config,
                    env_config=env_config,
                    symbols=list(market.symbols),
                    shortable_symbols=shortable_export,
                    optimizer=optimizer,
                    update=update,
                    summary=summary,
                    validation_market_range=split_info,
                    feature_normalizer=feature_normalizer,
                    validation_window_config=validation_window_config,
                    validation_window_summaries=window_summaries,
                )
            materially_improved = float(summary["score"]) > float(previous_best_score) + early_stop_min_delta
            if materially_improved:
                no_improve_validations = 0
            elif early_stop_patience is not None:
                no_improve_validations += 1
                if no_improve_validations >= early_stop_patience:
                    print(
                        "early_stop update={update:03d} best_score={best:+.4f} current_score={current:+.4f}".format(
                            update=update,
                            best=best_score,
                            current=summary["score"],
                        )
                    )
                    break
        else:
            history_row["validated"] = False
            history.append(history_row)
            print(
                "update={update:03d} train_only loss_pi={pi:.4f} loss_v={v:.4f} entropy={ent:.4f}".format(
                    update=update,
                    pi=losses["policy"],
                    v=losses["value"],
                    ent=losses["entropy"],
                )
            )

        if update % int(args.save_every) == 0:
            _save_checkpoint(
                output_dir / f"checkpoint_{update:04d}.pt",
                model=model,
                model_config=model_config,
                env_config=env_config,
                symbols=list(market.symbols),
                shortable_symbols=shortable_export,
                optimizer=optimizer,
                update=update,
                summary=last_validation_summary,
                validation_market_range=split_info,
                feature_normalizer=feature_normalizer,
                validation_window_config=validation_window_config,
                validation_window_summaries=last_validation_window_summaries,
            )

    _save_checkpoint(
        output_dir / "final.pt",
        model=model,
        model_config=model_config,
        env_config=env_config,
        symbols=list(market.symbols),
        shortable_symbols=shortable_export,
        optimizer=optimizer,
        update=completed_updates,
        summary=best_summary,
        validation_market_range=split_info,
        feature_normalizer=feature_normalizer,
        validation_window_config=validation_window_config,
        validation_window_summaries=best_window_summaries,
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True))
    manifest = {
        "symbols": list(market.symbols),
        "shortable_symbols": shortable_export,
        "train_hours": len(train_market),
        "validation_hours": len(val_market),
        "validation_market_range": split_info,
        "validation_window_config": validation_window_config,
        "training_config": {
            "lookback": int(args.lookback),
            "episode_steps": int(args.episode_steps),
            "train_split_ratio": float(args.train_split_ratio),
            "purge_hours": int(args.purge_hours),
            "updates": int(args.updates),
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "ppo_epochs": int(args.ppo_epochs),
            "minibatch_size": int(args.minibatch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_eps": float(args.clip_eps),
            "vf_coef": float(args.vf_coef),
            "ent_coef": float(args.ent_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "validate_every": int(args.validate_every),
            "validation_purge_hours": int(args.validation_purge_hours),
            "validation_batch_size": int(args.validation_batch_size) if args.validation_batch_size is not None else None,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "completed_updates": int(completed_updates),
        },
        "feature_normalizer": feature_normalizer.to_dict(),
        "env_config": env_config.to_dict(),
        "model_config": model_config.to_dict(),
        "best_summary": best_summary or {},
        "best_window_summaries": best_window_summaries,
        "device": str(device),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def build_arg_parser(*, require_output_dir: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Risk-aware Binance hourly PPO training.")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--shortable-symbols", default=None)
    parser.add_argument("--min-history-hours", type=int, default=24 * 90)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--episode-steps", type=int, default=24 * 7)
    parser.add_argument("--train-split-ratio", type=float, default=0.85)
    parser.add_argument("--purge-hours", type=int, default=24)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--max-gross-leverage", type=float, default=5.0)
    parser.add_argument("--max-position-weight", type=float, default=1.25)
    parser.add_argument("--spread-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--margin-apr", type=float, default=0.05)
    parser.add_argument("--downside-penalty", type=float, default=2.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.5)
    parser.add_argument("--turnover-penalty", type=float, default=0.01)
    parser.add_argument("--concentration-penalty", type=float, default=0.02)
    parser.add_argument("--leverage-penalty", type=float, default=0.01)
    parser.add_argument("--smoothness-penalty", type=float, default=2.0)
    parser.add_argument("--volatility-penalty", type=float, default=0.10)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-window-hours", default=str(24 * 7))
    parser.add_argument("--validation-window-weights", default=None)
    parser.add_argument("--validation-purge-hours", type=int, default=24)
    parser.add_argument("--validation-stride-hours", default=None)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--output-dir", required=require_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest = train(args)
    print(json.dumps(manifest["best_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
