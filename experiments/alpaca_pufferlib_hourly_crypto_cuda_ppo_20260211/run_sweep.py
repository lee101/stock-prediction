from __future__ import annotations

import argparse
import json
import subprocess
import sys
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    ent_coef: float = 0.01
    anneal_lr: bool = True
    cash_penalty: float = 0.005
    drawdown_penalty: float = 0.0
    downside_penalty: float = 0.0
    trade_penalty: float = 0.0
    disable_shorts: bool = False


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


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


def _resolve_single_fee(symbols: Iterable[str], explicit_fee: float | None) -> float:
    if explicit_fee is not None:
        return float(explicit_fee)
    from src.fees import get_fee_for_symbol

    symbol_list = _normalise_symbols(symbols)
    fees = {round(float(get_fee_for_symbol(sym)), 10) for sym in symbol_list}
    if not fees:
        raise ValueError("No symbols available to resolve fee_rate")
    if len(fees) > 1:
        raise ValueError(
            "pufferlib_market C env expects one fee_rate but symbol fees differ: "
            f"{sorted(fees)} for symbols {symbol_list}"
        )
    return float(next(iter(fees)))


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


def _mask_short_logits(logits: torch.Tensor, num_actions: int) -> torch.Tensor:
    num_symbols = (int(num_actions) - 1) // 2
    if num_symbols <= 0:
        return logits
    masked = logits.clone()
    masked[:, 1 + num_symbols :] = torch.finfo(masked.dtype).min
    return masked


def _export_hourly_binary(
    *,
    symbols: Iterable[str],
    data_root: Path,
    output_path: Path,
    split_cfg: Dict[str, Any],
    force: bool,
) -> Dict[str, Any]:
    from pufferlib_market.export_data_hourly_price import export_binary

    if output_path.exists() and not force:
        return _read_mktd_metadata(output_path) | {"cached": True}

    return export_binary(
        symbols=list(symbols),
        data_root=data_root,
        output_path=output_path,
        start_date=split_cfg.get("start_date"),
        end_date=split_cfg.get("end_date"),
        min_hours=int(split_cfg.get("min_hours", 24 * 30)),
        min_coverage=float(split_cfg.get("min_coverage", 0.95)),
    )


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
        "symbols": symbols,
        "num_symbols": int(num_symbols),
        "num_timesteps": int(num_timesteps),
        "version": int(version),
        "features_per_sym": int(features_per_sym),
        "price_features": int(price_features),
    }


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

    metric_keys = (
        "total_return",
        "sortino",
        "max_drawdown",
        "num_trades",
        "win_rate",
        "avg_hold_hours",
    )
    weighted = {key: 0.0 for key in metric_keys}
    episodes_done = 0
    total_steps = 0
    max_eval_steps = max_steps * max(eval_episodes, 1) * 4

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

        log_info = binding.vec_log(vec_handle) or {}
        n_raw = float(log_info.get("n", 0.0))
        n = int(round(n_raw))
        if n <= 0:
            continue
        episodes_done += n
        for key in metric_keys:
            weighted[key] += float(log_info.get(key, 0.0)) * n

    binding.vec_close(vec_handle)

    if episodes_done <= 0:
        raise RuntimeError("Evaluation produced zero completed episodes")

    mean_metrics = {key: weighted[key] / float(episodes_done) for key in metric_keys}
    mean_metrics["episodes"] = float(episodes_done)
    mean_metrics["annualized_return"] = _annualize(
        mean_metrics["total_return"],
        periods=float(max_steps),
        periods_per_year=float(periods_per_year),
    )
    return mean_metrics


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
) -> Path:
    ckpt_dir = EXP_DIR / "checkpoints" / run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        str(ckpt_dir),
    ]
    if run.anneal_lr:
        cmd.append("--anneal-lr")
    if run.disable_shorts:
        cmd.append("--disable-shorts")
    if device == "cpu":
        cmd.append("--cpu")

    log_path = ckpt_dir / "train.log"
    with log_path.open("w") as log_fp:
        subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=True,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )

    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    final = ckpt_dir / "final.pt"
    if final.exists():
        return final
    raise FileNotFoundError(f"Missing checkpoint artifacts in {ckpt_dir}")


def _append_progress(path: Path, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        if path.stat().st_size and not section.startswith("\n"):
            handle.write("\n")
        handle.write(section.rstrip() + "\n")


def _format_section(
    *,
    title: str,
    symbols: list[str],
    data_range: str,
    rows: list[Dict[str, Any]],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"### {title} ({now})",
        f"- Symbols: {', '.join(symbols)}",
        f"- Holdout window: {data_range}",
    ]
    for row in rows:
        holdout = row.get("holdout", {})
        lines.append(
            "- {run}: ret={ret:+.4f} ann={ann:+.2%} sortino={sortino:+.3f} mdd={mdd:+.3f} trades={trades:.1f}".format(
                run=row.get("run", "unknown"),
                ret=float(holdout.get("total_return", 0.0)),
                ann=float(holdout.get("annualized_return", 0.0)),
                sortino=float(holdout.get("sortino", 0.0)),
                mdd=float(holdout.get("max_drawdown", 0.0)),
                trades=float(holdout.get("num_trades", 0.0)),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly Alpaca crypto PPO sweep on pufferlib_market C env")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--record-progress", action="store_true")
    args = parser.parse_args()

    cfg = _load_json(EXP_DIR / "config.json")
    symbols = _normalise_symbols(cfg["symbols"])
    data_root = Path(cfg["data_root"])
    env_cfg = dict(cfg["env"])
    env_cfg.setdefault("action_allocation_bins", 1)
    env_cfg.setdefault("action_level_bins", 1)
    env_cfg.setdefault("action_max_offset_bps", 0.0)
    env_cfg["fee_rate"] = _resolve_single_fee(symbols, explicit_fee=env_cfg.get("fee_rate"))
    train_cfg = dict(cfg["train"])
    eval_cfg = dict(cfg["eval"])

    train_bin = EXP_DIR / "train_hourly_mktd_v2.bin"
    eval_bin = EXP_DIR / "eval_holdout_hourly_mktd_v2.bin"

    train_report = _export_hourly_binary(
        symbols=symbols,
        data_root=data_root,
        output_path=train_bin,
        split_cfg=train_cfg,
        force=bool(args.force_export),
    )
    train_symbols = list(train_report.get("symbols", symbols))
    eval_report = _export_hourly_binary(
        symbols=train_symbols,
        data_root=data_root,
        output_path=eval_bin,
        split_cfg=eval_cfg,
        force=bool(args.force_export),
    )
    eval_symbols = list(eval_report.get("symbols", train_symbols))
    if eval_symbols != train_symbols:
        raise RuntimeError(
            "Train/Eval symbol mismatch after export. "
            f"train={train_symbols}, eval={eval_symbols}"
        )

    # Resolve runtime symbol list from exported binaries so train/eval stay identical.
    resolved_symbols = train_symbols
    max_steps = int(eval_cfg["episode_steps"])
    eval_episodes = int(eval_cfg.get("eval_episodes", 256))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    runs = [
        RunConfig(name="r0_longonly_cash001", ent_coef=0.01, cash_penalty=0.001, disable_shorts=True),
        RunConfig(name="r1_longonly_cash002", ent_coef=0.01, cash_penalty=0.002, disable_shorts=True),
        RunConfig(
            name="r2_longonly_cash001_down10",
            ent_coef=0.008,
            cash_penalty=0.001,
            downside_penalty=10.0,
            disable_shorts=True,
        ),
        RunConfig(
            name="r3_longonly_cash001_down25_trade002",
            ent_coef=0.008,
            cash_penalty=0.001,
            downside_penalty=25.0,
            trade_penalty=0.002,
            disable_shorts=True,
        ),
        RunConfig(name="r4_longshort_down10", ent_coef=0.008, downside_penalty=10.0, disable_shorts=False),
        RunConfig(
            name="r5_longonly_cash002_dd01_down10",
            ent_coef=0.008,
            cash_penalty=0.002,
            drawdown_penalty=0.01,
            downside_penalty=10.0,
            trade_penalty=0.001,
            disable_shorts=True,
        ),
    ]

    results: list[Dict[str, Any]] = []
    for idx, run in enumerate(runs, 1):
        print(f"\n=== [{idx}/{len(runs)}] {run.name} ===")
        best_ckpt = _train_one(
            run,
            train_bin=train_bin,
            env_cfg=env_cfg,
            total_timesteps=int(args.total_timesteps),
            num_envs=int(args.num_envs),
            rollout_len=int(args.rollout_len),
            max_steps=max_steps,
            device=device,
        )
        holdout = _evaluate_checkpoint(
            best_ckpt,
            data_path=eval_bin,
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
        entry = {
            "run": run.name,
            "checkpoint": str(best_ckpt),
            "config": run.__dict__,
            "holdout": holdout,
        }
        results.append(entry)
        (best_ckpt.parent / "eval_report.json").write_text(json.dumps(entry, indent=2, sort_keys=True))
        print(
            "holdout ret={ret:+.4f} ann={ann:+.2%} sortino={sortino:+.3f} mdd={mdd:+.3f} trades={trades:.1f} ep={ep:.0f}".format(
                ret=float(holdout.get("total_return", 0.0)),
                ann=float(holdout.get("annualized_return", 0.0)),
                sortino=float(holdout.get("sortino", 0.0)),
                mdd=float(holdout.get("max_drawdown", 0.0)),
                trades=float(holdout.get("num_trades", 0.0)),
                ep=float(holdout.get("episodes", 0.0)),
            )
        )

    out_path = EXP_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    ranked = sorted(
        results,
        key=lambda row: (
            float((row.get("holdout") or {}).get("annualized_return", -1e9)),
            float((row.get("holdout") or {}).get("sortino", -1e9)),
        ),
        reverse=True,
    )

    print("\nTop runs:")
    for i, row in enumerate(ranked[:5], 1):
        metrics = row.get("holdout", {})
        print(
            f"{i}. {row['run']}: "
            f"ret={float(metrics.get('total_return', 0.0)):+.4f} "
            f"ann={float(metrics.get('annualized_return', 0.0)):+.2%} "
            f"sortino={float(metrics.get('sortino', 0.0)):+.3f} "
            f"mdd={float(metrics.get('max_drawdown', 0.0)):+.3f}"
        )

    if args.record_progress:
        holdout_window = f"{eval_cfg.get('start_date')} -> {eval_cfg.get('end_date')}"
        successful = [row for row in results if float((row.get("holdout") or {}).get("total_return", 0.0)) > 0.0]
        unsuccessful = [row for row in results if row not in successful]

        if successful:
            section = _format_section(
                title="Hourly Crypto PPO Sweep Success",
                symbols=resolved_symbols,
                data_range=holdout_window,
                rows=sorted(
                    successful,
                    key=lambda row: float((row.get("holdout") or {}).get("annualized_return", -1e9)),
                    reverse=True,
                )[:5],
            )
            _append_progress(REPO_ROOT / "alpacaprogress.md", section)

        if unsuccessful:
            section = _format_section(
                title="Hourly Crypto PPO Sweep Unsuccessful Runs",
                symbols=resolved_symbols,
                data_range=holdout_window,
                rows=sorted(
                    unsuccessful,
                    key=lambda row: float((row.get("holdout") or {}).get("annualized_return", -1e9)),
                    reverse=True,
                ),
            )
            _append_progress(REPO_ROOT / "unsuccessfulalpacaprogress.md", section)

    print(f"\nWrote results: {out_path}")
    print("Train export:", json.dumps(train_report, indent=2, sort_keys=True))
    print("Eval export:", json.dumps(eval_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
