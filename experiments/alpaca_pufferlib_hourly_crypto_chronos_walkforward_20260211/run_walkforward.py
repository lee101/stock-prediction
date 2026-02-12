from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.distributions import Categorical


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RunConfig:
    name: str
    hidden_size: int = 256
    arch: str = "resmlp"
    lr: float = 3e-4
    ent_coef: float = 0.008
    anneal_lr: bool = True
    cash_penalty: float = 0.001
    drawdown_penalty: float = 0.0
    downside_penalty: float = 0.0
    smooth_downside_penalty: float = 0.0
    smooth_downside_temperature: float = 0.02
    trade_penalty: float = 0.0
    disable_shorts: bool = False


def _normalise_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _resolve_single_fee(symbols: Sequence[str], explicit_fee: float | None) -> float:
    if explicit_fee is not None:
        return float(explicit_fee)
    from src.fees import get_fee_for_symbol

    fee_values = {round(float(get_fee_for_symbol(sym)), 10) for sym in symbols}
    if not fee_values:
        raise ValueError("No symbols provided for fee resolution")
    if len(fee_values) > 1:
        raise ValueError(
            "pufferlib_market C env currently supports a single fee_rate; "
            f"resolved multiple fees for symbols: {sorted(fee_values)}"
        )
    return float(next(iter(fee_values)))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _annualize(total_return: float, periods: float, periods_per_year: float) -> float:
    if periods <= 0:
        return 0.0
    mult = 1.0 + float(total_return)
    if mult <= 0:
        return -1.0
    years = float(periods) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(mult ** (1.0 / years) - 1.0)


def _rank_key(metrics: Dict[str, float]) -> tuple[float, float]:
    return (float(metrics.get("annualized_return", -1e9)), float(metrics.get("sortino", -1e9)))


def _mask_short_logits(logits: torch.Tensor, num_actions: int) -> torch.Tensor:
    num_symbols = (int(num_actions) - 1) // 2
    if num_symbols <= 0:
        return logits
    masked = logits.clone()
    masked[:, 1 + num_symbols :] = torch.finfo(masked.dtype).min
    return masked


def _read_mktd_metadata(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(64)
        magic, version, num_symbols, num_timesteps, features_per_sym, price_features, _ = struct.unpack(
            "<4sIIIII40s",
            header,
        )
        if magic != b"MKTD":
            raise ValueError(f"Invalid MKTD magic in {path}")
        symbol_bytes = handle.read(int(num_symbols) * 16)

    symbols: list[str] = []
    for i in range(int(num_symbols)):
        raw = symbol_bytes[i * 16 : (i + 1) * 16]
        symbols.append(raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore"))
    return {
        "path": str(path),
        "version": int(version),
        "num_symbols": int(num_symbols),
        "num_timesteps": int(num_timesteps),
        "features_per_sym": int(features_per_sym),
        "price_features": int(price_features),
        "symbols": symbols,
    }


def _append_progress(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        if path.stat().st_size and not section.startswith("\n"):
            handle.write("\n")
        handle.write(section.rstrip() + "\n")


def _format_section(
    *,
    title: str,
    symbols: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"### {title} ({now})", f"- Symbols: {', '.join(symbols)}"]
    for row in rows:
        m = row.get("metrics", {})
        lines.append(
            "- {fold}/{run}: ret={ret:+.4f} ann={ann:+.2%} sortino={sortino:+.3f} mdd={mdd:+.3f} trades={trades:.1f}".format(
                fold=row.get("fold", "unknown"),
                run=row.get("run", "unknown"),
                ret=float(m.get("total_return", 0.0)),
                ann=float(m.get("annualized_return", 0.0)),
                sortino=float(m.get("sortino", 0.0)),
                mdd=float(m.get("max_drawdown", 0.0)),
                trades=float(m.get("num_trades", 0.0)),
            )
        )
    return "\n".join(lines)


def _export_chronos_binary(
    *,
    symbols: Sequence[str],
    forecast_cache_root: Path,
    data_root: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
    min_rows: int,
) -> Dict[str, Any]:
    from pufferlib_market.export_data import export_binary

    export_binary(
        symbols=list(symbols),
        forecast_cache_root=forecast_cache_root,
        data_root=data_root,
        output_path=output_path,
        min_rows=int(min_rows),
        start_date=start_date,
        end_date=end_date,
    )
    return _read_mktd_metadata(output_path)


def _align_fold_symbols(train_symbols: Sequence[str], eval_symbols: Sequence[str]) -> list[str]:
    eval_set = {sym.upper() for sym in eval_symbols}
    common = [sym for sym in train_symbols if sym.upper() in eval_set]
    return common


def _prepare_fold_data(
    *,
    fold_name: str,
    symbols: Sequence[str],
    forecast_cache_root: Path,
    data_root: Path,
    fold_cfg: Dict[str, Any],
    force_export: bool,
    min_symbols: int,
) -> Dict[str, Any]:
    fold_dir = EXP_DIR / "folds" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_bin = fold_dir / "train_mktd.bin"
    eval_bin = fold_dir / "eval_mktd.bin"
    episode_steps = int(fold_cfg["episode_steps"])

    requested = _normalise_symbols(symbols)
    if train_bin.exists() and eval_bin.exists() and not force_export:
        train_meta = _read_mktd_metadata(train_bin)
        eval_meta = _read_mktd_metadata(eval_bin)
        common = _align_fold_symbols(train_meta["symbols"], eval_meta["symbols"])
        if (
            len(common) >= int(min_symbols)
            and common == train_meta["symbols"] == eval_meta["symbols"]
            and common == requested
        ):
            return {
                "train_bin": train_bin,
                "eval_bin": eval_bin,
                "symbols": common,
                "train_meta": train_meta,
                "eval_meta": eval_meta,
                "cached": True,
            }

    train_meta = _export_chronos_binary(
        symbols=symbols,
        forecast_cache_root=forecast_cache_root,
        data_root=data_root,
        output_path=train_bin,
        start_date=str(fold_cfg["train_start"]),
        end_date=str(fold_cfg["train_end"]),
        min_rows=max(episode_steps + 1, 500),
    )
    eval_meta = _export_chronos_binary(
        symbols=train_meta["symbols"],
        forecast_cache_root=forecast_cache_root,
        data_root=data_root,
        output_path=eval_bin,
        start_date=str(fold_cfg["eval_start"]),
        end_date=str(fold_cfg["eval_end"]),
        min_rows=episode_steps + 1,
    )

    common = _align_fold_symbols(train_meta["symbols"], eval_meta["symbols"])
    if len(common) < int(min_symbols):
        raise RuntimeError(
            f"Fold {fold_name} has insufficient aligned symbols ({len(common)} < {min_symbols}): {common}"
        )

    if common != train_meta["symbols"] or common != eval_meta["symbols"]:
        # Re-export both splits with the strict common symbol set for shape safety.
        train_meta = _export_chronos_binary(
            symbols=common,
            forecast_cache_root=forecast_cache_root,
            data_root=data_root,
            output_path=train_bin,
            start_date=str(fold_cfg["train_start"]),
            end_date=str(fold_cfg["train_end"]),
            min_rows=max(episode_steps + 1, 500),
        )
        eval_meta = _export_chronos_binary(
            symbols=common,
            forecast_cache_root=forecast_cache_root,
            data_root=data_root,
            output_path=eval_bin,
            start_date=str(fold_cfg["eval_start"]),
            end_date=str(fold_cfg["eval_end"]),
            min_rows=episode_steps + 1,
        )

    return {
        "train_bin": train_bin,
        "eval_bin": eval_bin,
        "symbols": list(train_meta["symbols"]),
        "train_meta": train_meta,
        "eval_meta": eval_meta,
        "cached": False,
    }


def _train_one(
    run: RunConfig,
    *,
    train_bin: Path,
    env_cfg: Dict[str, Any],
    total_timesteps: int,
    num_envs: int,
    rollout_len: int,
    max_steps: int,
    device: str,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        "-m",
        "pufferlib_market.train",
        "--data-path",
        str(train_bin),
        "--max-steps",
        str(int(max_steps)),
        "--periods-per-year",
        str(float(env_cfg["periods_per_year"])),
        "--fee-rate",
        str(float(env_cfg["fee_rate"])),
        "--max-leverage",
        str(float(env_cfg["max_leverage"])),
        "--reward-scale",
        str(float(env_cfg["reward_scale"])),
        "--reward-clip",
        str(float(env_cfg["reward_clip"])),
        "--action-allocation-bins",
        str(int(env_cfg["action_allocation_bins"])),
        "--action-level-bins",
        str(int(env_cfg["action_level_bins"])),
        "--action-max-offset-bps",
        str(float(env_cfg["action_max_offset_bps"])),
        "--cash-penalty",
        str(float(run.cash_penalty)),
        "--drawdown-penalty",
        str(float(run.drawdown_penalty)),
        "--downside-penalty",
        str(float(run.downside_penalty)),
        "--smooth-downside-penalty",
        str(float(run.smooth_downside_penalty)),
        "--smooth-downside-temperature",
        str(float(run.smooth_downside_temperature)),
        "--trade-penalty",
        str(float(run.trade_penalty)),
        "--num-envs",
        str(int(num_envs)),
        "--rollout-len",
        str(int(rollout_len)),
        "--total-timesteps",
        str(int(total_timesteps)),
        "--hidden-size",
        str(int(run.hidden_size)),
        "--arch",
        run.arch,
        "--lr",
        str(float(run.lr)),
        "--ent-coef",
        str(float(run.ent_coef)),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if run.anneal_lr:
        cmd.append("--anneal-lr")
    if run.disable_shorts:
        cmd.append("--disable-shorts")
    if device == "cpu":
        cmd.append("--cpu")

    log_path = checkpoint_dir / "train.log"
    with log_path.open("w") as log_fp:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, stdout=log_fp, stderr=subprocess.STDOUT)

    for candidate in (checkpoint_dir / "best.pt", checkpoint_dir / "final.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def _policy_from_checkpoint(
    checkpoint: Path,
    *,
    obs_size: int,
    num_actions: int,
    hidden_size: int,
    arch: str,
    device: torch.device,
):
    from pufferlib_market.evaluate import ResidualTradingPolicy, TradingPolicy

    if arch == "resmlp":
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden_size).to(device)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden_size).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    data_path: Path,
    max_steps: int,
    fee_rate: float,
    max_leverage: float,
    periods_per_year: float,
    action_allocation_bins: int,
    action_level_bins: int,
    action_max_offset_bps: float,
    hidden_size: int,
    arch: str,
    eval_episodes: int,
    deterministic: bool,
    seed: int,
    device: str,
    disable_shorts: bool,
) -> Dict[str, float]:
    import pufferlib_market.binding as binding

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    with data_path.open("rb") as handle:
        header = handle.read(64)
    _, _, num_symbols, _, _, _ = struct.unpack("<4sIIIII", header[:24])
    obs_size = int(num_symbols) * 16 + 5 + int(num_symbols)
    per_symbol_actions = max(1, int(action_allocation_bins)) * max(1, int(action_level_bins))
    num_actions = 1 + 2 * int(num_symbols) * per_symbol_actions
    policy = _policy_from_checkpoint(
        checkpoint,
        obs_size=obs_size,
        num_actions=num_actions,
        hidden_size=hidden_size,
        arch=arch,
        device=dev,
    )

    binding.shared(data_path=str(data_path.resolve()))
    num_envs = int(min(max(8, eval_episodes // 4), 128))
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf,
        act_buf,
        rew_buf,
        term_buf,
        trunc_buf,
        num_envs,
        int(seed),
        max_steps=int(max_steps),
        fee_rate=float(fee_rate),
        max_leverage=float(max_leverage),
        periods_per_year=float(periods_per_year),
        action_allocation_bins=int(action_allocation_bins),
        action_level_bins=int(action_level_bins),
        action_max_offset_bps=float(action_max_offset_bps),
    )
    binding.vec_reset(vec_handle, int(seed))

    keys = ("total_return", "sortino", "max_drawdown", "num_trades", "win_rate", "avg_hold_hours")
    weighted = {key: 0.0 for key in keys}
    episodes_done = 0
    max_eval_steps = max_steps * max(eval_episodes, 1) * 4
    total_steps = 0

    while episodes_done < eval_episodes and total_steps < max_eval_steps:
        obs_tensor = torch.from_numpy(obs_buf.copy()).to(dev)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            if disable_shorts:
                logits = _mask_short_logits(logits, num_actions)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        act_buf[:] = action.detach().cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle)
        total_steps += num_envs

        info = binding.vec_log(vec_handle) or {}
        n = int(round(float(info.get("n", 0.0))))
        if n <= 0:
            continue
        episodes_done += n
        for key in keys:
            weighted[key] += float(info.get(key, 0.0)) * n

    binding.vec_close(vec_handle)
    if episodes_done <= 0:
        raise RuntimeError("Evaluation produced zero completed episodes")

    metrics = {key: weighted[key] / float(episodes_done) for key in keys}
    metrics["episodes"] = float(episodes_done)
    metrics["annualized_return"] = _annualize(
        metrics["total_return"],
        periods=float(max_steps),
        periods_per_year=float(periods_per_year),
    )
    return metrics


def _summarize_fold_best(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "mean_return": 0.0,
            "mean_annualized_return": 0.0,
            "mean_sortino": 0.0,
            "mean_max_drawdown": 0.0,
            "positive_folds": 0.0,
            "num_folds": 0.0,
        }
    returns = [float(row["metrics"]["total_return"]) for row in results]
    annual = [float(row["metrics"]["annualized_return"]) for row in results]
    sortino = [float(row["metrics"]["sortino"]) for row in results]
    mdd = [float(row["metrics"]["max_drawdown"]) for row in results]
    return {
        "mean_return": float(np.mean(returns)),
        "mean_annualized_return": float(np.mean(annual)),
        "mean_sortino": float(np.mean(sortino)),
        "mean_max_drawdown": float(np.mean(mdd)),
        "positive_folds": float(sum(1 for r in returns if r > 0.0)),
        "num_folds": float(len(results)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos-feature walk-forward PPO sweep (hourly crypto)")
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--record-progress", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0, help="Limit number of folds (0 = all)")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of run configs per fold (0 = all)")
    parser.add_argument("--fold-names", type=str, default="", help="Comma-separated fold names to run")
    parser.add_argument("--run-names", type=str, default="", help="Comma-separated run config names to run")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols override")
    parser.add_argument("--min-symbols", type=int, default=1, help="Minimum aligned symbols required per fold export")
    args = parser.parse_args()

    cfg = _load_json(EXP_DIR / "config.json")
    forecast_cache_root = Path(cfg["forecast_cache_root"])
    data_root = Path(cfg["data_root"])
    if args.symbols.strip():
        symbols = _normalise_symbols(args.symbols.split(","))
    else:
        symbols = _normalise_symbols(cfg["symbols"])
    env_cfg = dict(cfg["env"])
    env_cfg.setdefault("action_allocation_bins", 1)
    env_cfg.setdefault("action_level_bins", 1)
    env_cfg.setdefault("action_max_offset_bps", 0.0)
    env_cfg["fee_rate"] = _resolve_single_fee(symbols, explicit_fee=env_cfg.get("fee_rate"))
    eval_episodes = int(cfg.get("eval_episodes", 256))
    folds = list(cfg["folds"])
    if args.fold_names.strip():
        fold_filter = {name.strip() for name in args.fold_names.split(",") if name.strip()}
        folds = [fold for fold in folds if str(fold.get("name")) in fold_filter]
    if args.max_folds > 0:
        folds = folds[: int(args.max_folds)]
    if not folds:
        raise ValueError("No folds selected for execution")

    runs: list[RunConfig] = [
        RunConfig(name="r0_chronos_longshort_cash005", cash_penalty=0.005, ent_coef=0.01),
        RunConfig(
            name="r1_chronos_longshort_cash01_down5",
            cash_penalty=0.01,
            ent_coef=0.01,
            downside_penalty=5.0,
        ),
        RunConfig(
            name="r2_chronos_longonly_cash005_down5",
            cash_penalty=0.005,
            ent_coef=0.01,
            downside_penalty=5.0,
            trade_penalty=0.001,
            disable_shorts=True,
        ),
        RunConfig(
            name="r3_chronos_longonly_cash01_dd01_down5",
            cash_penalty=0.01,
            ent_coef=0.01,
            drawdown_penalty=0.01,
            downside_penalty=5.0,
            trade_penalty=0.001,
            disable_shorts=True,
        ),
        RunConfig(
            name="r4_chronos_longshort_cash01_down5_smooth2_t01",
            cash_penalty=0.01,
            ent_coef=0.01,
            downside_penalty=5.0,
            smooth_downside_penalty=2.0,
            smooth_downside_temperature=0.01,
        ),
        RunConfig(
            name="r5_chronos_longonly_cash005_down5_smooth2_t01",
            cash_penalty=0.005,
            ent_coef=0.01,
            downside_penalty=5.0,
            smooth_downside_penalty=2.0,
            smooth_downside_temperature=0.01,
            trade_penalty=0.001,
            disable_shorts=True,
        ),
    ]
    if args.max_runs > 0:
        runs = runs[: int(args.max_runs)]
    if args.run_names.strip():
        run_filter = {name.strip() for name in args.run_names.split(",") if name.strip()}
        runs = [run for run in runs if run.name in run_filter]
    if not runs:
        raise ValueError("No run configs selected for execution")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    fold_best: list[Dict[str, Any]] = []
    all_results: list[Dict[str, Any]] = []

    for fold_idx, fold_cfg in enumerate(folds, 1):
        fold_name = str(fold_cfg["name"])
        print(f"\n=== Fold [{fold_idx}/{len(folds)}] {fold_name} ===")
        prepared = _prepare_fold_data(
            fold_name=fold_name,
            symbols=symbols,
            forecast_cache_root=forecast_cache_root,
            data_root=data_root,
            fold_cfg=fold_cfg,
            force_export=bool(args.force_export),
            min_symbols=max(1, int(args.min_symbols)),
        )
        fold_symbols = list(prepared["symbols"])
        print(
            f"symbols={len(fold_symbols)} train_rows={prepared['train_meta']['num_timesteps']} "
            f"eval_rows={prepared['eval_meta']['num_timesteps']}"
        )

        fold_rows: list[Dict[str, Any]] = []
        max_steps = int(fold_cfg["episode_steps"])
        for run_idx, run in enumerate(runs, 1):
            print(f"  -> Run [{run_idx}/{len(runs)}] {run.name}")
            ckpt_dir = EXP_DIR / "folds" / fold_name / "checkpoints" / run.name
            best_ckpt = _train_one(
                run,
                train_bin=prepared["train_bin"],
                env_cfg=env_cfg,
                total_timesteps=int(args.total_timesteps),
                num_envs=int(args.num_envs),
                rollout_len=int(args.rollout_len),
                max_steps=max_steps,
                device=device,
                checkpoint_dir=ckpt_dir,
            )
            metrics = _evaluate_checkpoint(
                best_ckpt,
                data_path=prepared["eval_bin"],
                max_steps=max_steps,
                fee_rate=float(env_cfg["fee_rate"]),
                max_leverage=float(env_cfg["max_leverage"]),
                periods_per_year=float(env_cfg["periods_per_year"]),
                action_allocation_bins=int(env_cfg["action_allocation_bins"]),
                action_level_bins=int(env_cfg["action_level_bins"]),
                action_max_offset_bps=float(env_cfg["action_max_offset_bps"]),
                hidden_size=int(run.hidden_size),
                arch=run.arch,
                eval_episodes=eval_episodes,
                deterministic=True,
                seed=int(args.seed),
                device=device,
                disable_shorts=bool(run.disable_shorts),
            )
            row = {
                "fold": fold_name,
                "run": run.name,
                "checkpoint": str(best_ckpt),
                "config": run.__dict__,
                "metrics": metrics,
                "symbols": fold_symbols,
            }
            fold_rows.append(row)
            all_results.append(row)
            (ckpt_dir / "eval_report.json").write_text(json.dumps(row, indent=2, sort_keys=True))
            print(
                "     holdout ret={ret:+.4f} ann={ann:+.2%} sort={sort:+.3f} mdd={mdd:+.3f} trades={trades:.1f}".format(
                    ret=float(metrics["total_return"]),
                    ann=float(metrics["annualized_return"]),
                    sort=float(metrics["sortino"]),
                    mdd=float(metrics["max_drawdown"]),
                    trades=float(metrics["num_trades"]),
                )
            )

        ranked = sorted(fold_rows, key=lambda row: _rank_key(row["metrics"]), reverse=True)
        best = ranked[0]
        fold_best.append(best)
        print(
            f"  Best {fold_name}: {best['run']} "
            f"ret={float(best['metrics']['total_return']):+.4f} "
            f"ann={float(best['metrics']['annualized_return']):+.2%} "
            f"sort={float(best['metrics']['sortino']):+.3f}"
        )

        (EXP_DIR / "folds" / fold_name / "fold_results.json").write_text(json.dumps(ranked, indent=2, sort_keys=True))

    summary = _summarize_fold_best(fold_best)
    payload = {
        "config_path": str(EXP_DIR / "config.json"),
        "device": device,
        "total_timesteps": int(args.total_timesteps),
        "num_envs": int(args.num_envs),
        "rollout_len": int(args.rollout_len),
        "fold_best": fold_best,
        "summary": summary,
        "all_results": all_results,
    }
    out_path = EXP_DIR / "walkforward_results.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("\nWalk-forward summary:")
    print(
        "mean_ret={ret:+.4f} mean_ann={ann:+.2%} mean_sort={sort:+.3f} "
        "mean_mdd={mdd:+.3f} positive_folds={pf:.0f}/{nf:.0f}".format(
            ret=float(summary["mean_return"]),
            ann=float(summary["mean_annualized_return"]),
            sort=float(summary["mean_sortino"]),
            mdd=float(summary["mean_max_drawdown"]),
            pf=float(summary["positive_folds"]),
            nf=float(summary["num_folds"]),
        )
    )
    for idx, row in enumerate(fold_best, 1):
        metrics = row["metrics"]
        print(
            f"{idx}. {row['fold']} -> {row['run']}: "
            f"ret={float(metrics['total_return']):+.4f} ann={float(metrics['annualized_return']):+.2%} "
            f"sort={float(metrics['sortino']):+.3f} mdd={float(metrics['max_drawdown']):+.3f}"
        )

    if args.record_progress:
        rows = [
            {"fold": row["fold"], "run": row["run"], "metrics": row["metrics"]}
            for row in fold_best
        ]
        title_base = "Chronos WalkForward Hourly Crypto"
        if float(summary["mean_return"]) > 0.0 and float(summary["positive_folds"]) >= max(1.0, float(summary["num_folds"]) / 2.0):
            section = _format_section(title=f"{title_base} Success", symbols=symbols, rows=rows)
            _append_progress(REPO_ROOT / "alpacaprogress.md", section)
        else:
            section = _format_section(title=f"{title_base} Unsuccessful", symbols=symbols, rows=rows)
            _append_progress(REPO_ROOT / "unsuccessfulalpacaprogress.md", section)

    print(f"\nWrote walk-forward results to {out_path}")


__all__ = [
    "_align_fold_symbols",
    "_rank_key",
    "_summarize_fold_best",
]


if __name__ == "__main__":
    main()
