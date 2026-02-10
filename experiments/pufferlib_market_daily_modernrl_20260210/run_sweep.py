from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    arch: str = "resmlp"  # mlp | resmlp
    lr: float = 3e-4
    ent_coef: float = 0.01
    anneal_lr: bool = True
    cash_penalty: float = 0.01
    drawdown_penalty: float = 0.0
    downside_penalty: float = 0.0
    trade_penalty: float = 0.0


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
    import struct

    import pufferlib_market.binding as binding
    from pufferlib_market.evaluate import ResidualTradingPolicy, TradingPolicy

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

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
    # Deterministic eval file uses max_offset=0, so a single episode is representative.
    while not log_info:
        obs_tensor = torch.from_numpy(obs_buf.copy()).to(dev)
        with torch.no_grad():
            action = policy.get_action(obs_tensor, deterministic=True)
        act_buf[:] = action.detach().cpu().numpy().astype(np.int32)
        binding.vec_step(vec_handle)
        log_info = binding.vec_log(vec_handle) or {}

    binding.vec_close(vec_handle)

    total_return = float(log_info.get("total_return", 0.0))
    return {
        "total_return": total_return,
        "annualized_return": _annualize(total_return, periods=max_steps, periods_per_year=periods_per_year),
        "sortino": float(log_info.get("sortino", 0.0)),
        "max_drawdown": float(log_info.get("max_drawdown", 0.0)),
        "num_trades": float(log_info.get("num_trades", 0.0)),
        "win_rate": float(log_info.get("win_rate", 0.0)),
        "avg_hold_hours": float(log_info.get("avg_hold_hours", 0.0)),
    }


def _hourly_replay_report(
    checkpoint: Path,
    *,
    daily_data_path: Path,
    hourly_data_root: Path,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float,
    max_leverage: float,
    daily_periods_per_year: float,
    arch: str,
    hidden_size: int,
    device: str,
) -> Dict[str, Any]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "pufferlib_market.replay_eval",
        "--checkpoint",
        str(checkpoint),
        "--daily-data-path",
        str(daily_data_path),
        "--hourly-data-root",
        str(hourly_data_root),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--max-steps",
        str(int(max_steps)),
        "--fee-rate",
        str(float(fee_rate)),
        "--max-leverage",
        str(float(max_leverage)),
        "--daily-periods-per-year",
        str(float(daily_periods_per_year)),
        "--arch",
        arch,
        "--hidden-size",
        str(int(hidden_size)),
        "--deterministic",
        "--run-hourly-policy",
    ]
    if device == "cpu":
        cmd.append("--cpu")

    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


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
        "--downside-penalty",
        str(run.downside_penalty),
        "--trade-penalty",
        str(run.trade_penalty),
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
    p = argparse.ArgumentParser(description="Modern sweep for daily MKTD v2 experiments (with hourly replay eval)")
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--rollout-len", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = _load_json(EXP_DIR / "config.json")
    paths = _export_if_missing(cfg)

    env_cfg = dict(cfg["env"])
    eval_cfg = dict(cfg["eval"])
    max_steps = int(eval_cfg["episode_steps"])

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    runs: List[RunConfig] = [
        # Baseline (terminal correctness fix will change training dynamics vs older checkpoints).
        RunConfig(name="r0_base_resmlp_h256_lr3e4_ent001"),
        # Downside penalty sweep (Sortino proxy).
        RunConfig(name="r1_down50", downside_penalty=50.0),
        RunConfig(name="r2_down100", downside_penalty=100.0),
        # Trade penalty sweep (churn control).
        RunConfig(name="r3_trade002", trade_penalty=0.02),
        RunConfig(name="r4_trade005", trade_penalty=0.05),
        RunConfig(name="r5_down100_trade002", downside_penalty=100.0, trade_penalty=0.02),
        RunConfig(name="r6_down100_trade005", downside_penalty=100.0, trade_penalty=0.05),
        # Drawdown shaping + downside/trade combos.
        RunConfig(name="r7_dd01", drawdown_penalty=0.1),
        RunConfig(name="r8_dd01_down50", drawdown_penalty=0.1, downside_penalty=50.0),
        RunConfig(name="r9_dd01_down100", drawdown_penalty=0.1, downside_penalty=100.0),
        RunConfig(name="r10_dd01_down100_trade002", drawdown_penalty=0.1, downside_penalty=100.0, trade_penalty=0.02),
        RunConfig(name="r11_dd01_down100_trade005", drawdown_penalty=0.1, downside_penalty=100.0, trade_penalty=0.05),
    ]

    results: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs, 1):
        print(f"\n=== [{idx}/{len(runs)}] {run.name} ===")
        print(
            "train_cfg: "
            f"arch={run.arch} hidden={run.hidden_size} lr={run.lr:g} ent={run.ent_coef:g} "
            f"anneal={int(run.anneal_lr)} "
            f"cash_pen={run.cash_penalty:g} dd_pen={run.drawdown_penalty:g} "
            f"downside_pen={run.downside_penalty:g} trade_pen={run.trade_penalty:g}"
        )
        best_ckpt = _train_one(
            run,
            train_bin=paths["train_bin"],
            env_cfg=env_cfg,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            rollout_len=args.rollout_len,
            max_steps=max_steps,
            device=device,
        )

        daily_eval = _evaluate_checkpoint(
            best_ckpt,
            data_path=paths["eval_bin"],
            max_steps=max_steps,
            fee_rate=float(env_cfg["fee_rate"]),
            max_leverage=float(env_cfg["max_leverage"]),
            periods_per_year=float(env_cfg["periods_per_year"]),
            hidden_size=run.hidden_size,
            arch=run.arch,
            device=device,
        )

        replay = _hourly_replay_report(
            best_ckpt,
            daily_data_path=paths["eval_bin"],
            hourly_data_root=Path("trainingdatahourly"),
            start_date=str(eval_cfg["start_date"]),
            end_date=str(eval_cfg["end_date"]),
            max_steps=max_steps,
            fee_rate=float(env_cfg["fee_rate"]),
            max_leverage=float(env_cfg["max_leverage"]),
            daily_periods_per_year=float(env_cfg["periods_per_year"]),
            arch=run.arch,
            hidden_size=run.hidden_size,
            device=device,
        )

        entry = {
            "run": run.name,
            "checkpoint": str(best_ckpt),
            "config": run.__dict__,
            "daily_holdout": daily_eval,
            "replay_eval": replay,
        }
        results.append(entry)
        (best_ckpt.parent / "eval_report.json").write_text(json.dumps(entry, indent=2, sort_keys=True))

        hr = (replay.get("hourly_replay") or {}) if isinstance(replay, dict) else {}
        print(
            "holdout: "
            f"daily_ret={daily_eval.get('total_return', 0.0):+.4f} daily_sort={daily_eval.get('sortino', 0.0):+.2f} "
            f"| hourly_ret={hr.get('total_return', 0.0):+.4f} hourly_sort={hr.get('sortino', 0.0):+.2f} "
            f"max_orders_in_day={hr.get('max_orders_in_day', 0)}"
        )

    out_path = EXP_DIR / "sweep_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    def sort_key(row: Dict[str, Any]) -> tuple[float, float]:
        rep = row.get("replay_eval") or {}
        hr = rep.get("hourly_replay") or {}
        return (float(hr.get("annualized_return", -1e9)), float(hr.get("sortino", -1e9)))

    ranked = sorted(results, key=sort_key, reverse=True)
    top = ranked[:5]

    print("\nTop 5 (hourly replay holdout):")
    for i, row in enumerate(top, 1):
        d = row.get("daily_holdout") or {}
        rep = row.get("replay_eval") or {}
        hr = rep.get("hourly_replay") or {}
        hp = rep.get("hourly_policy") or {}
        print(
            f"{i}. {row['run']}: "
            f"daily_ret={d.get('total_return', 0.0):+.4f} daily_sort={d.get('sortino', 0.0):+.2f} "
            f"| hourly_ret={hr.get('total_return', 0.0):+.4f} hourly_sort={hr.get('sortino', 0.0):+.2f} "
            f"max_orders_in_day={hr.get('max_orders_in_day', 0)} "
            f"| hourly_policy_ret={hp.get('total_return', 0.0):+.4f} hourly_policy_sort={hp.get('sortino', 0.0):+.2f}"
        )


if __name__ == "__main__":
    main()
