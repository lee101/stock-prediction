from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RunConfig:
    name: str
    hidden_size: int = 256
    arch: str = "mlp"  # mlp | resmlp
    lr: float = 3e-4
    ent_coef: float = 0.01
    anneal_lr: bool = True
    cash_penalty: float = 0.01
    drawdown_penalty: float = 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _export_if_missing(cfg: Dict[str, Any]) -> Dict[str, Path]:
    from pufferlib_market.export_data_daily import export_binary

    symbols = list(cfg["symbols"])
    data_root = Path(cfg["data_root"])

    train_bin = EXP_DIR / "train_mktd_v2.bin"
    eval_bin = EXP_DIR / "eval50d_mktd_v2.bin"

    if not train_bin.exists():
        export_binary(
            symbols=symbols,
            data_root=data_root,
            output_path=train_bin,
            start_date=cfg["train"]["start_date"],
            end_date=cfg["train"]["end_date"],
            min_days=200,
        )
    if not eval_bin.exists():
        export_binary(
            symbols=symbols,
            data_root=data_root,
            output_path=eval_bin,
            start_date=cfg["eval"]["start_date"],
            end_date=cfg["eval"]["end_date"],
            min_days=int(cfg["eval"]["episode_steps"]) + 1,
        )

    return {"train_bin": train_bin, "eval_bin": eval_bin}


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    data_path: Path,
    max_steps: int,
    fee_rate: float,
    max_leverage: float,
    periods_per_year: float,
    hidden_size: int,
    arch: str,
    device: str,
) -> Dict[str, float]:
    # Local import so sweep can still run export-only in minimal envs.
    import struct

    import pufferlib_market.binding as binding
    from pufferlib_market.evaluate import TradingPolicy, ResidualTradingPolicy

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    # Read binary header to get num_symbols.
    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, _, _, _ = struct.unpack("<4sIIIII", header[:24])
    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols

    if arch == "resmlp":
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden_size).to(dev)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden_size).to(dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    binding.shared(data_path=str(data_path.resolve()))

    obs_buf = np.zeros((1, obs_size), dtype=np.float32)
    act_buf = np.zeros((1,), dtype=np.int32)
    rew_buf = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf,
        act_buf,
        rew_buf,
        term_buf,
        trunc_buf,
        1,
        123,
        max_steps=max_steps,
        fee_rate=fee_rate,
        max_leverage=max_leverage,
        periods_per_year=periods_per_year,
    )
    binding.vec_reset(vec_handle, 123)

    log_info: Dict[str, float] = {}
    # Run until one episode completes (deterministic eval file uses max_offset=0).
    while not log_info:
        obs_tensor = torch.from_numpy(obs_buf.copy()).to(dev)
        with torch.no_grad():
            action = policy.get_action(obs_tensor, deterministic=True)
        act_buf[:] = action.detach().cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle)
        log_info = binding.vec_log(vec_handle) or {}

    binding.vec_close(vec_handle)

    return {
        "total_return": float(log_info.get("total_return", 0.0)),
        "sortino": float(log_info.get("sortino", 0.0)),
        "max_drawdown": float(log_info.get("max_drawdown", 0.0)),
        "num_trades": float(log_info.get("num_trades", 0.0)),
        "win_rate": float(log_info.get("win_rate", 0.0)),
        "avg_hold_hours": float(log_info.get("avg_hold_hours", 0.0)),
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
        str(max_steps),
        "--periods-per-year",
        str(env_cfg["periods_per_year"]),
        "--fee-rate",
        str(env_cfg["fee_rate"]),
        "--max-leverage",
        str(env_cfg["max_leverage"]),
        "--cash-penalty",
        str(run.cash_penalty),
        "--drawdown-penalty",
        str(run.drawdown_penalty),
        "--num-envs",
        str(num_envs),
        "--rollout-len",
        str(rollout_len),
        "--total-timesteps",
        str(total_timesteps),
        "--hidden-size",
        str(run.hidden_size),
        "--arch",
        run.arch,
        "--lr",
        str(run.lr),
        "--ent-coef",
        str(run.ent_coef),
        "--checkpoint-dir",
        str(ckpt_dir),
    ]
    if run.anneal_lr:
        cmd.append("--anneal-lr")
    if device == "cpu":
        cmd.append("--cpu")

    log_path = ckpt_dir / "train.log"
    with log_path.open("w") as log_fp:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, stdout=log_fp, stderr=subprocess.STDOUT)

    best = ckpt_dir / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Missing best checkpoint at {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small sweep for daily MKTD v2 experiments")
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = _load_json(EXP_DIR / "config.json")
    paths = _export_if_missing(cfg)

    env_cfg = dict(cfg["env"])
    max_steps = int(cfg["eval"]["episode_steps"])

    runs: List[RunConfig] = [
        RunConfig(name="daily_mix8_h256_lr3e4_ent001_anneal", hidden_size=256, lr=3e-4, ent_coef=0.01, anneal_lr=True),
        RunConfig(name="daily_mix8_h256_lr1e4_ent001_anneal", hidden_size=256, lr=1e-4, ent_coef=0.01, anneal_lr=True),
        RunConfig(name="daily_mix8_h256_lr5e4_ent001_anneal", hidden_size=256, lr=5e-4, ent_coef=0.01, anneal_lr=True),
        RunConfig(name="daily_mix8_h512_lr3e4_ent001_anneal", hidden_size=512, lr=3e-4, ent_coef=0.01, anneal_lr=True),
        RunConfig(name="daily_mix8_h1024_lr3e4_ent001_anneal", hidden_size=1024, lr=3e-4, ent_coef=0.01, anneal_lr=True),
        RunConfig(name="daily_mix8_h256_lr3e4_ent005_anneal", hidden_size=256, lr=3e-4, ent_coef=0.05, anneal_lr=True),
        RunConfig(name="daily_mix8_h256_lr3e4_ent0005_anneal", hidden_size=256, lr=3e-4, ent_coef=0.005, anneal_lr=True),
        RunConfig(name="daily_mix8_h256_lr3e4_ent001_noanneal", hidden_size=256, lr=3e-4, ent_coef=0.01, anneal_lr=False),
        RunConfig(name="daily_mix8_h256_lr3e4_ent001_dd01", hidden_size=256, lr=3e-4, ent_coef=0.01, anneal_lr=True, drawdown_penalty=0.1),
        RunConfig(name="daily_mix8_h256_lr3e4_ent001_cash0", hidden_size=256, lr=3e-4, ent_coef=0.01, anneal_lr=True, cash_penalty=0.0),
    ]

    results: List[Dict[str, Any]] = []
    for run in runs:
        best_ckpt = _train_one(
            run,
            train_bin=paths["train_bin"],
            env_cfg=env_cfg,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_len=args.rollout_len,
            max_steps=max_steps,
            device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        metrics = _evaluate_checkpoint(
            best_ckpt,
            data_path=paths["eval_bin"],
            max_steps=max_steps,
            fee_rate=float(env_cfg["fee_rate"]),
            max_leverage=float(env_cfg["max_leverage"]),
            periods_per_year=float(env_cfg["periods_per_year"]),
            hidden_size=run.hidden_size,
            arch=run.arch,
            device=args.device,
        )
        entry = {"run": run.name, "checkpoint": str(best_ckpt), "metrics": metrics, "config": run.__dict__}
        results.append(entry)
        (best_ckpt.parent / "eval_metrics.json").write_text(json.dumps(entry, indent=2))

    out_path = EXP_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Print top runs by total_return (then sortino).
    ranked = sorted(results, key=lambda r: (r["metrics"]["total_return"], r["metrics"]["sortino"]), reverse=True)
    top = ranked[:5]
    print("\nTop 5 (holdout):")
    for i, row in enumerate(top, 1):
        m = row["metrics"]
        print(f"{i}. {row['run']}: total_return={m['total_return']:+.4f} sortino={m['sortino']:+.2f} trades={m['num_trades']:.1f}")


if __name__ == "__main__":
    main()
