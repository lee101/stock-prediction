"""Single-trainer PPO run on the C marketsim with stocks12 v5_rsi features.

Wraps `pufferlib_market.train` (BF16 baseline) or the fp4 PPO trainer once it
exists.  Writes JSON metrics + curve to `fp4/bench/results/<trainer>_<seed>_<date>.json`.

Run directly:

    python fp4/bench/bench_trading.py --trainer pufferlib_bf16 --steps 200000 --seed 0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CFG = REPO / "fp4" / "experiments" / "fp4_ppo_stocks12.yaml"


def _load_cfg(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _tiny_yaml(path.read_text())
    return yaml.safe_load(path.read_text())


def _tiny_yaml(text: str) -> dict[str, Any]:
    """Pure-python fallback when pyyaml is missing.  Handles the limited
    subset used by our config files (key/value, indented dicts, lists)."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if val == "":
            new: dict[str, Any] = {}
            parent[key] = new
            stack.append((indent, new))
        else:
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                parent[key] = [_coerce(v) for v in items]
            else:
                parent[key] = _coerce(val)
    return root


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v.strip("'\"")


def _trainer_available(name: str) -> tuple[bool, str]:
    if name == "pufferlib_bf16":
        try:
            import pufferlib  # noqa: F401
            from pufferlib_market import train  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    if name == "hf_trainer":
        try:
            import transformers  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    if name == "trl":
        try:
            import trl  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    if name == "fp4":
        try:
            from fp4 import linear  # type: ignore  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, f"fp4 lib not built yet: {type(exc).__name__}: {exc}"
    return False, f"unknown trainer {name!r}"


def _gpu_peak_mb_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _run_pufferlib_bf16(cfg: dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> dict[str, Any]:
    env = cfg["env"]
    ppo = cfg["ppo"]
    train_data = REPO / env["train_data"]
    val_data = REPO / env["val_data"]
    if not train_data.exists():
        return {"status": "skip", "reason": f"train data missing: {train_data}"}
    cmd = [
        sys.executable, "-m", "pufferlib_market.train",
        "--data-path", str(train_data),
        "--val-data-path", str(val_data) if val_data.exists() else str(train_data),
        "--total-timesteps", str(int(steps)),
        "--seed", str(int(seed)),
        "--hidden-size", str(int(ppo.get("hidden_size", 1024))),
        "--lr", str(float(ppo.get("lr", 3e-4))),
        "--ent-coef", str(float(ppo.get("ent_coef", 0.05))),
        "--num-envs", str(int(ppo.get("num_envs", 64))),
        "--rollout-len", str(int(ppo.get("rollout_len", 256))),
        "--ppo-epochs", str(int(ppo.get("ppo_epochs", 4))),
        "--minibatch-size", str(int(ppo.get("minibatch_size", 2048))),
        "--fee-rate", str(float(env.get("fee_rate", 0.001))),
        "--max-leverage", str(float(env.get("max_leverage_scalar_fallback", 1.5))),
        "--checkpoint-dir", str(ckpt_dir),
        "--save-every", "999999",
        "--lr-schedule", "cosine" if ppo.get("anneal_lr") else "none",
    ]
    if ppo.get("obs_norm"):
        cmd.append("--obs-norm")
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = str(REPO) + ":" + env_vars.get("PYTHONPATH", "")
    _gpu_peak_mb_reset()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env_vars, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "wall_sec": wall,
        "gpu_peak_mb": _gpu_peak_mb(),
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        "cmd": cmd,
    }


def _run_hf_trainer(cfg: dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> dict[str, Any]:
    return {"status": "skip", "reason": "hf_trainer adapter not implemented (RL-on-marketsim is custom; HF baseline is for the supervised forecaster)."}


def _run_trl(cfg: dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> dict[str, Any]:
    return {"status": "skip", "reason": "trl PPO adapter for marketsim not implemented in this iteration"}


def _run_fp4(cfg: dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> dict[str, Any]:
    try:
        from fp4 import trainer as fp4_trainer  # type: ignore
    except Exception as exc:
        return {"status": "skip", "reason": f"fp4 trainer not built yet ({exc})"}
    fn = getattr(fp4_trainer, "train_ppo", None)
    if fn is None:
        return {"status": "skip", "reason": "fp4.trainer.train_ppo missing"}
    _gpu_peak_mb_reset()
    t0 = time.perf_counter()
    try:
        out = fn(cfg=cfg, total_timesteps=int(steps), seed=int(seed), checkpoint_dir=str(ckpt_dir))
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - t0
    return {
        "status": "ok",
        "wall_sec": wall,
        "gpu_peak_mb": _gpu_peak_mb(),
        "trainer_output": out if isinstance(out, dict) else {"raw": str(out)},
    }


_RUNNERS = {
    "pufferlib_bf16": _run_pufferlib_bf16,
    "hf_trainer": _run_hf_trainer,
    "trl": _run_trl,
    "fp4": _run_fp4,
}


def _evaluate_ckpt(ckpt_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the latest checkpoint at multiple slippage levels using the C
    binary-fill marketsim via `pufferlib_market.evaluate_fast`."""
    ckpts = sorted(ckpt_dir.rglob("best.pt")) + sorted(ckpt_dir.rglob("final.pt"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.rglob("*.pt"))
    if not ckpts:
        return {"status": "skip", "reason": "no checkpoint produced"}
    ckpt = ckpts[-1]
    env = cfg["env"]
    val_data = REPO / env["val_data"]
    if not val_data.exists():
        return {"status": "skip", "reason": f"val data missing: {val_data}"}
    out: dict[str, Any] = {"checkpoint": str(ckpt), "by_slippage": {}}
    for bps in cfg["eval"]["slippage_bps"]:
        cmd = [
            sys.executable, "-m", "pufferlib_market.evaluate_fast",
            "--checkpoint", str(ckpt),
            "--data-path", str(val_data),
            "--fill-slippage-bps", str(bps),
            "--n-windows", str(int(cfg["eval"].get("n_windows", 20))),
            "--max-leverage", str(float(env.get("max_leverage_scalar_fallback", 1.5))),
            "--fee-rate", str(float(env.get("fee_rate", 0.001))),
            "--out", str(RESULTS_DIR / f"_eval_{ckpt.stem}_{bps}bps.json"),
        ]
        env_vars = os.environ.copy()
        env_vars["PYTHONPATH"] = str(REPO) + ":" + env_vars.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, cwd=str(REPO), env=env_vars, capture_output=True, text=True)
        rec: dict[str, Any] = {"returncode": proc.returncode}
        try:
            rec.update(json.loads((RESULTS_DIR / f"_eval_{ckpt.stem}_{bps}bps.json").read_text()))
        except Exception:
            rec["stdout_tail"] = proc.stdout[-500:]
            rec["stderr_tail"] = proc.stderr[-500:]
        out["by_slippage"][str(bps)] = rec
    return out


def run_one(trainer: str, cfg_path: Path, steps: int, seed: int, smoke: bool = False) -> dict[str, Any]:
    cfg = _load_cfg(cfg_path)
    avail, why = _trainer_available(trainer)
    date = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec: dict[str, Any] = {
        "trainer": trainer,
        "seed": seed,
        "steps": steps,
        "date": date,
        "config_path": str(cfg_path),
        "smoke": bool(smoke),
        "available": avail,
    }
    if not avail:
        rec["status"] = "skip"
        rec["reason"] = why
        out_path = RESULTS_DIR / f"{trainer}_s{seed}_{date}.json"
        out_path.write_text(json.dumps(rec, indent=2))
        rec["result_path"] = str(out_path)
        return rec
    ckpt_dir = RESULTS_DIR / f"_ckpt_{trainer}_s{seed}_{date}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runner = _RUNNERS[trainer]
    train_rec = runner(cfg, steps, seed, ckpt_dir)
    rec["train"] = train_rec
    if train_rec.get("status") == "ok":
        rec["eval"] = _evaluate_ckpt(ckpt_dir, cfg)
        wall = train_rec.get("wall_sec", 0.0) or 1e-9
        rec["steps_per_sec"] = float(steps) / wall
    rec["status"] = train_rec.get("status", "error")
    out_path = RESULTS_DIR / f"{trainer}_s{seed}_{date}.json"
    out_path.write_text(json.dumps(rec, indent=2, default=str))
    rec["result_path"] = str(out_path)
    if smoke:
        # leave checkpoint dir to inspect; cleanup is best-effort
        try:
            shutil.rmtree(ckpt_dir)
        except OSError:
            pass
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer", default="fp4", choices=list(_RUNNERS))
    p.add_argument("--config", default=str(DEFAULT_CFG))
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    rec = run_one(args.trainer, Path(args.config), args.steps, args.seed, smoke=args.smoke)
    print(json.dumps({k: v for k, v in rec.items() if k != "train"}, indent=2, default=str))
    return 0 if rec["status"] in ("ok", "skip") else 1


if __name__ == "__main__":
    raise SystemExit(main())
